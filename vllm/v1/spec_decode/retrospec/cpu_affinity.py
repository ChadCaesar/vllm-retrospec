# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Separate CPU verification, launch work, and host KV transfers."""

import os
from contextlib import suppress
from pathlib import Path
from threading import get_native_id

from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_cpu_list(value: str) -> set[int]:
    result = set()
    for part in value.strip().split(","):
        bounds = [int(number) for number in part.split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _plan_affinity(allowed_by_rank: list[set[int]], threads: int):
    """Reserve a whole NUMA node, with one verifier thread per physical core."""
    common = set.intersection(*allowed_by_rank)
    try:
        nodes = {
            int(path.name[4:]): _parse_cpu_list((path / "cpulist").read_text())
            for path in Path("/sys/devices/system/node").glob("node[0-9]*")
        }
        cores = {}
        for cpu in set.union(*allowed_by_rank):
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            cores[cpu] = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        for node, reserved in sorted(nodes.items(), reverse=True):
            physical = {}
            for cpu in sorted(common & reserved):
                physical.setdefault(cores[cpu], cpu)
            if len(physical) < threads:
                continue
            if all(
                len({cores[cpu] for cpu in allowed - reserved}) >= 2
                for allowed in allowed_by_rank
            ):
                return node, tuple(list(physical.values())[:threads]), reserved
    except (OSError, ValueError):
        pass
    return None


class RetroSpecCPUAffinity:
    def __init__(self):
        self.cpu_cpus: tuple[int, ...] | None = None
        self.original_masks: dict[int, set[int]] = {}
        self.bound_masks: dict[int, set[int]] = {}
        self.original_allowed: set[int] = set()
        self.gpu_allowed: set[int] = set()
        self.cpu_node: int | None = None
        self.rank = 0

    @classmethod
    def create(cls, group, threads: int, enabled: bool):
        affinity = cls()
        affinity.rank = group.rank_in_group
        if not enabled:
            return affinity

        def gather(value):
            if group.world_size == 1:
                return [value]
            import torch.distributed as dist

            values = [None] * group.world_size
            dist.all_gather_object(values, value, group=group.cpu_group)
            return values

        supported = (
            hasattr(os, "sched_getaffinity") and Path("/proc/self/task").is_dir()
        )
        # Honor user-supplied OpenMP placement, including an explicit opt-out.
        explicit = any(
            name in os.environ
            for name in ("GOMP_CPU_AFFINITY", "OMP_PLACES", "OMP_PROC_BIND")
        )
        try:
            allowed = set(os.sched_getaffinity(0)) if supported else set()
        except OSError:
            allowed, supported = set(), False
        profiles = gather((allowed, supported and not explicit))
        plan = None
        if group.rank_in_group == 0 and all(profile[1] for profile in profiles):
            plan = _plan_affinity([profile[0] for profile in profiles], threads)
        plan = group.broadcast_object(plan, src=0)
        if plan is None:
            if group.rank_in_group == 0:
                logger.info(
                    "RetroSpec CPU NUMA binding skipped: no compatible partition "
                    "or explicit OpenMP placement; retaining CPU affinity"
                )
            return affinity
        node, cpu_cpus, reserved = plan
        gpu_allowed = allowed - reserved
        prepared = True
        try:

            def core(cpu):
                topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                return (
                    (topology / "physical_package_id").read_text().strip(),
                    (topology / "core_id").read_text().strip(),
                )

            verifier_cores = {core(cpu) for cpu in cpu_cpus}
            for task in Path("/proc/self/task").iterdir():
                tid = int(task.name)
                try:
                    mask = set(os.sched_getaffinity(tid))
                except ProcessLookupError:
                    continue
                affinity.original_masks[tid] = mask
                # CUDA/NCCL can pin helpers to the verifier's NUMA node.
                # Retain their locality, using free physical cores on that
                # node rather than competing with the verifier or its SMT
                # siblings. Keep an exclusive binding if it has no alternative.
                launch_mask = mask & gpu_allowed
                if not launch_mask:
                    # A helper's mask can be wider than the launch thread's:
                    # inspect its own CPUs to include every SMT sibling.
                    launch_mask = {
                        cpu for cpu in mask if core(cpu) not in verifier_cores
                    }
                affinity.bound_masks[tid] = launch_mask or mask
        except OSError:
            prepared = False
        if not all(gather(prepared)):
            affinity.original_masks.clear()
            affinity.bound_masks.clear()
            return affinity
        affinity.original_allowed = allowed
        affinity.gpu_allowed = gpu_allowed
        applied = True
        try:
            for tid, mask in affinity.bound_masks.items():
                with suppress(ProcessLookupError):
                    os.sched_setaffinity(tid, mask)
        except OSError:
            applied = False
        if not all(gather(applied)):
            affinity.close()
            logger.warning(
                "Could not isolate RetroSpec CPU verifier; kept CPU affinity"
            )
            return affinity
        affinity.cpu_cpus = cpu_cpus
        affinity.cpu_node = node
        logger.info(
            "RetroSpec CPU verifier: NUMA node %s, CPUs %s; GPU launch CPUs %s",
            node,
            list(cpu_cpus),
            sorted(gpu_allowed),
        )
        return affinity

    def transfer_placement(self, device_index: int, workers: int):
        from .numa import RetroSpecHostPlacement, gpu_numa_node

        if self.cpu_cpus is None:
            return RetroSpecHostPlacement()
        node = gpu_numa_node(device_index)
        if node is None:
            return RetroSpecHostPlacement()
        try:
            if node == self.cpu_node:
                # The CPU target streams weights and full KV from this node.
                # Its memory controller is not free bandwidth for PCIe staging.
                # Keep that traffic on the launch side even for a remote GPU.
                for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
                    cpus = _parse_cpu_list((path / "cpulist").read_text())
                    if cpus & self.gpu_allowed:
                        node = int(path.name[4:])
                        break
            local = _parse_cpu_list(
                Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()
            )

            # Exclude every SMT sibling of a verifier core. Transfer workers
            # may use the remaining physical cores on the same NUMA node.
            def core(cpu):
                path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                return (
                    (path / "physical_package_id").read_text().strip(),
                    (path / "core_id").read_text().strip(),
                )

            reserved = {core(cpu) for cpu in self.cpu_cpus}
            physical = {}
            for cpu in sorted(local & self.original_allowed, reverse=True):
                identity = core(cpu)
                if identity not in reserved:
                    physical[identity] = cpu
            available = list(physical.values())
            offset = self.rank * workers % len(available) if available else 0
            available = available[offset:] + available[:offset]
            cpus = tuple(available[:workers])
        except (OSError, ValueError):
            return RetroSpecHostPlacement()
        if not cpus:
            return RetroSpecHostPlacement()
        logger.info("RetroSpec gather/H2D: NUMA node %s, CPUs %s", node, list(cpus))
        return RetroSpecHostPlacement(cpus, node)

    def reserve_transfer_cores(self, group, cpus: tuple[int, ...]):
        """Keep the launch thread off every rank's copy cores and SMT siblings."""
        if self.cpu_cpus is None:
            return

        def gather(value):
            if group.world_size == 1:
                return [value]
            import torch.distributed as dist

            values = [None] * group.world_size
            dist.all_gather_object(values, value, group=group.cpu_group)
            return values

        copies = set().union(*map(set, gather(cpus)))
        original = set()
        remaining = set()
        prepared = True
        try:
            original = set(os.sched_getaffinity(0))

            def core(cpu):
                path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                return (
                    (path / "physical_package_id").read_text().strip(),
                    (path / "core_id").read_text().strip(),
                )

            reserved = {core(cpu) for cpu in copies}
            remaining = {cpu for cpu in self.gpu_allowed if core(cpu) not in reserved}
            prepared = len({core(cpu) for cpu in remaining}) >= 2
        except (OSError, ValueError):
            prepared = False
        if not all(gather(prepared)):
            return
        applied = True
        try:
            os.sched_setaffinity(0, remaining)
        except OSError:
            applied = False
        if not all(gather(applied)):
            if applied:
                os.sched_setaffinity(0, original)
            return
        self.gpu_allowed = remaining
        self.bound_masks[get_native_id()] = remaining
        logger.info("RetroSpec GPU launch/control CPUs: %s", sorted(remaining))

    def close(self):
        if not self.original_masks:
            return
        # New threads inherit the restricted mask. Restore those too, without
        # overwriting affinity independently changed by the caller afterwards.
        for task in Path("/proc/self/task").iterdir():
            tid = int(task.name)
            bound = self.bound_masks.get(tid, self.gpu_allowed)
            original = self.original_masks.get(tid, self.original_allowed)
            try:
                if set(os.sched_getaffinity(tid)) == bound:
                    os.sched_setaffinity(tid, original)
            except OSError:
                pass
        self.original_masks.clear()
        self.bound_masks.clear()
        self.cpu_cpus = None
        self.cpu_node = None
