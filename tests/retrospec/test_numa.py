# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode.retrospec import numa


def test_host_job_restores_worker_policy_after_failure(monkeypatch):
    allowed = set(os.sched_getaffinity(0))
    calls = []

    class Policy:
        def get_mempolicy(self, mode, mask, *_):
            mode._obj.value = 2  # Preserve a pre-existing bind policy.
            mask[0] = 1
            return 0

        def set_mempolicy(self, mode, mask, *_):
            calls.append((mode, mask[0]))
            return 0

    monkeypatch.setattr(numa, "_numa_library", lambda: Policy())
    placement = numa.RetroSpecHostPlacement((min(allowed),), 1)

    def fail():
        assert os.sched_getaffinity(0) == {min(allowed)}
        raise ValueError("failed gather")

    with ThreadPoolExecutor(max_workers=1) as executor:
        before = executor.submit(os.sched_getaffinity, 0).result()
        with pytest.raises(ValueError, match="failed gather"):
            executor.submit(placement.run, fail).result()
        assert executor.submit(os.sched_getaffinity, 0).result() == before
    assert os.sched_getaffinity(0) == allowed
    assert calls == [(1, 2), (2, 1)]


def test_host_job_children_inherit_affinity_without_libnuma(monkeypatch):
    monkeypatch.setattr(numa, "_numa_library", lambda: None)
    allowed = set(os.sched_getaffinity(0))
    placement = numa.RetroSpecHostPlacement((min(allowed),), 1)

    def spawn():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(os.sched_getaffinity, 0).result()

    assert placement.run(spawn) == {min(allowed)}
    assert os.sched_getaffinity(0) == allowed


def test_shared_kv_policy_does_not_migrate_or_touch_pages(monkeypatch):
    seen = []

    class Policy:
        def mbind(self, pointer, size, mode, mask, maxnode, flags):
            seen.append((pointer.value, size.value, mode, mask[0], maxnode, flags))
            return 0

    monkeypatch.setattr(numa, "_numa_library", lambda: Policy())
    tensor = SimpleNamespace(
        data_ptr=lambda: 4096, numel=lambda: 16384, element_size=lambda: 2
    )
    assert numa.RetroSpecHostPlacement(node=1).bind_memory(tensor)
    assert seen == [(4096, 32768, 2, 2, 1024, 0)]


def test_dedicated_worker_keeps_placement_between_jobs(monkeypatch):
    monkeypatch.setattr(numa, "_numa_library", lambda: None)
    allowed = set(os.sched_getaffinity(0))
    placement = numa.RetroSpecHostPlacement((min(allowed),), 1)
    with ThreadPoolExecutor(
        max_workers=1, initializer=placement.initialize_worker
    ) as executor:
        for _ in range(3):
            assert executor.submit(os.sched_getaffinity, 0).result() == {min(allowed)}
    assert os.sched_getaffinity(0) == allowed


def test_prefetch_worker_applies_configuration_before_its_first_job(monkeypatch):
    from vllm.v1.spec_decode.retrospec.offload.cluster_store_support import (
        _ResidentPrefetchPriorityExecutor,
    )

    monkeypatch.setattr(numa, "_numa_library", lambda: None)
    allowed = os.sched_getaffinity(0)
    placement = numa.RetroSpecHostPlacement()
    executor = _ResidentPrefetchPriorityExecutor(placement)
    observed = []
    try:
        placement.cpus = (min(allowed),)
        for _ in range(3):
            executor.submit(
                0, lambda: observed.append(os.sched_getaffinity(0))
            ).result()
        assert observed == [{min(allowed)}] * 3
    finally:
        executor.shutdown()
    assert os.sched_getaffinity(0) == allowed


def test_gpu_node_uses_visible_device_uuid(monkeypatch, tmp_path):
    from vllm import platforms
    from vllm.utils import import_utils

    seen = []
    nvml = SimpleNamespace(
        nvmlInit=lambda: seen.append("init"),
        nvmlShutdown=lambda: seen.append("shutdown"),
        nvmlDeviceGetHandleByUUID=lambda uuid: seen.append(uuid) or uuid,
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(busId=b"00000000:A0:00.0"),
    )
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(get_device_uuid=lambda i: f"gpu{i}"),
    )
    monkeypatch.setattr(import_utils, "import_pynvml", lambda: nvml)
    path = tmp_path / "sys/bus/pci/devices/0000:a0:00.0/numa_node"
    path.parent.mkdir(parents=True)
    path.write_text("1")
    monkeypatch.setattr(numa, "Path", lambda value: tmp_path / value.lstrip("/"))
    assert numa.gpu_numa_node(0) == 1
    assert seen == ["init", "gpu0", "shutdown"]
    path.write_text("-1")
    assert numa.gpu_numa_node(0) is None
