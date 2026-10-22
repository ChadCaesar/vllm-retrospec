# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode.retrospec import cpu_affinity


@pytest.fixture
def topology(tmp_path, monkeypatch):
    def path(value):
        return tmp_path / value.lstrip("/")

    for node in range(2):
        directory = path(f"/sys/devices/system/node/node{node}")
        directory.mkdir(parents=True)
        (directory / "cpulist").write_text(",".join(map(str, range(node, 16, 2))))
    for cpu in range(16):
        directory = path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        directory.mkdir(parents=True)
        (directory / "physical_package_id").write_text(str(cpu % 2))
        (directory / "core_id").write_text(str(cpu % 8 // 2))
    tasks = path("/proc/self/task")
    tasks.mkdir(parents=True)
    masks = {11: set(range(16)), 12: {0, 2}, 13: set(range(16))}
    for tid in masks:
        (tasks / str(tid)).mkdir()
    monkeypatch.setattr(cpu_affinity, "Path", path)
    monkeypatch.setattr(cpu_affinity, "get_native_id", lambda: 11)
    monkeypatch.setattr(
        cpu_affinity.os, "sched_getaffinity", lambda tid: masks[tid or 11].copy()
    )
    monkeypatch.setattr(
        cpu_affinity.os,
        "sched_setaffinity",
        lambda tid, mask: masks.update({tid or 11: set(mask)}),
    )
    for key in ("GOMP_CPU_AFFINITY", "OMP_PLACES", "OMP_PROC_BIND"):
        monkeypatch.delenv(key, raising=False)
    return path, masks


def group():
    return SimpleNamespace(
        world_size=1, rank_in_group=0, broadcast_object=lambda value, src: value
    )


def test_numa_plan_uses_physical_cores_within_every_rank_mask(topology):
    plan = cpu_affinity._plan_affinity([set(range(16)), set(range(1, 16))], 3)
    assert plan == (1, (1, 3, 5), set(range(1, 16, 2)))


@pytest.mark.parametrize(
    "masks,threads",
    [
        ([set(range(16))], 5),  # Eight SMT threads are only four physical cores.
        ([set(range(0, 16, 2)), set(range(1, 16, 2))], 2),
        ([{0, 8} | set(range(1, 16, 2))], 3),  # Leave GPU host cores available.
        ([set(range(0, 16, 2))], 2),  # Only one allowed NUMA node.
    ],
)
def test_numa_plan_retains_affinity_when_no_partition_fits(topology, masks, threads):
    assert cpu_affinity._plan_affinity(masks, threads) is None


def test_missing_cpu_topology_retains_affinity(topology):
    path, _ = topology
    path("/sys/devices/system/cpu/cpu3/topology/core_id").unlink()
    assert cpu_affinity._plan_affinity([set(range(16))], 3) is None


@pytest.mark.parametrize("key", ["GOMP_CPU_AFFINITY", "OMP_PLACES", "OMP_PROC_BIND"])
def test_explicit_openmp_placement_is_preserved(topology, monkeypatch, key):
    _, masks = topology
    before = {tid: mask.copy() for tid, mask in masks.items()}
    monkeypatch.setenv(key, "user setting")
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    assert affinity.cpu_cpus is None
    assert masks == before


def test_numa_binding_restores_existing_and_inherited_masks(topology):
    path, masks = topology
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    assert affinity.cpu_cpus == (1, 3, 5)
    assert masks == {11: set(range(0, 16, 2)), 12: {0, 2}, 13: set(range(0, 16, 2))}
    (path("/proc/self/task") / "14").mkdir()
    masks[14] = masks[11].copy()
    masks[12] = {4}  # A subsequent caller change must survive restoration.
    affinity.close()
    assert masks == {
        11: set(range(16)),
        12: {4},
        13: set(range(16)),
        14: set(range(16)),
    }
    affinity.close()


def test_affinity_application_failure_rolls_back_every_changed_thread(
    topology, monkeypatch
):
    _, masks = topology
    before = {tid: mask.copy() for tid, mask in masks.items()}
    original_set = cpu_affinity.os.sched_setaffinity
    failed = False

    def fail_once(tid, mask):
        nonlocal failed
        if tid == 12 and not failed:
            failed = True
            raise PermissionError("affinity denied")
        original_set(tid, mask)

    monkeypatch.setattr(cpu_affinity.os, "sched_setaffinity", fail_once)
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    assert failed
    assert affinity.cpu_cpus is None
    assert masks == before


def test_existing_exclusive_thread_binding_is_preserved(topology):
    _, masks = topology
    masks[12] = {3}
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    assert affinity.cpu_cpus == (1, 3, 5)
    assert masks[11] == set(range(0, 16, 2))
    assert masks[12] == {3}
    affinity.close()
    assert masks[11] == set(range(16))
    assert masks[12] == {3}


@pytest.mark.parametrize("failure_stage", ["prepare", "apply"])
def test_peer_affinity_failure_keeps_all_ranks_unbound(
    topology, monkeypatch, failure_stage
):
    import torch.distributed as dist

    _, masks = topology
    before = {tid: mask.copy() for tid, mask in masks.items()}
    checks = 0

    def gather(values, value, group):
        nonlocal checks
        peer = value
        if isinstance(value, bool):
            checks += 1
            if checks == (1 if failure_stage == "prepare" else 2):
                peer = False
        values[:] = [value, peer]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    tp = group()
    tp.world_size = 2
    tp.cpu_group = object()
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(tp, 3, True)
    assert affinity.cpu_cpus is None
    assert masks == before


def test_numa_binding_can_be_disabled(topology):
    _, masks = topology
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, False)
    assert affinity.cpu_cpus is None
    assert masks[11] == set(range(16))


def test_transfer_workers_use_local_cores_without_verifier_siblings(
    topology, monkeypatch
):
    from vllm.v1.spec_decode.retrospec import numa

    monkeypatch.setattr(numa, "gpu_numa_node", lambda index: index)
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    transfer = affinity.transfer_placement(1, 4)
    # Avoid the verifier's memory controller even for its locally attached GPU.
    assert transfer.node == 0
    assert transfer.cpus == (6, 4, 2, 0)
    other = affinity.transfer_placement(0, 2)
    assert other.node == 0
    assert other.cpus == (6, 4)
    affinity.close()


def test_launch_reservation_excludes_copy_siblings_and_restores_new_threads(topology):
    path, masks = topology
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    affinity.reserve_transfer_cores(group(), (4, 6))
    assert masks[11] == {0, 2, 8, 10}
    assert affinity.gpu_allowed == masks[11]
    (path("/proc/self/task") / "14").mkdir()
    masks[14] = masks[11].copy()
    affinity.close()
    assert masks[11] == masks[14] == set(range(16))


def test_launch_reservation_keeps_a_minimum_launch_partition(topology):
    _, masks = topology
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    before = masks[11].copy()
    affinity.reserve_transfer_cores(group(), (0, 2, 4))
    assert masks[11] == before
    affinity.close()


@pytest.mark.parametrize("failure_stage", ["prepare", "apply"])
def test_peer_launch_reservation_failure_keeps_original_masks(
    topology, monkeypatch, failure_stage
):
    import torch.distributed as dist

    _, masks = topology
    affinity = cpu_affinity.RetroSpecCPUAffinity.create(group(), 3, True)
    before = {tid: mask.copy() for tid, mask in masks.items()}
    checks = 0

    def gather(values, value, group):
        nonlocal checks
        peer = value
        if isinstance(value, bool):
            checks += 1
            if checks == (1 if failure_stage == "prepare" else 2):
                peer = False
        values[:] = [value, peer]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    tp = group()
    tp.world_size = 2
    tp.cpu_group = object()
    affinity.reserve_transfer_cores(tp, (4, 6))
    assert masks == before
    assert affinity.gpu_allowed == before[11]
    affinity.close()
