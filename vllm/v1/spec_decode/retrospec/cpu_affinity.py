# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep full CPU inference off the GPU workers' NUMA nodes when possible."""

import os
from contextlib import suppress
from pathlib import Path

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

    @classmethod
    def create(cls, group, threads: int, enabled: bool):
        affinity = cls()
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
            for task in Path("/proc/self/task").iterdir():
                tid = int(task.name)
                try:
                    mask = set(os.sched_getaffinity(tid))
                except ProcessLookupError:
                    continue
                affinity.original_masks[tid] = mask
                # CUDA/NCCL can pin helper threads to their GPU's NUMA node.
                # Preserve these existing bindings while moving the launch
                # thread and other freely scheduled worker threads.
                affinity.bound_masks[tid] = mask & gpu_allowed or mask
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
        logger.info(
            "RetroSpec CPU verifier: NUMA node %s, CPUs %s; GPU launch CPUs %s",
            node,
            list(cpu_cpus),
            sorted(gpu_allowed),
        )
        return affinity

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
