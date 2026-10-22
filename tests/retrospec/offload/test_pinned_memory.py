# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.v1.spec_decode.retrospec.numa import RetroSpecHostPlacement
from vllm.v1.spec_decode.retrospec.pinned_memory import (
    RetroSpecPinnedMemoryManager,
)

pytestmark = pytest.mark.cpu_test


def test_disabled_pinned_manager_returns_pageable_storage_without_accounting():
    manager = RetroSpecPinnedMemoryManager(enabled=False, max_bytes=16)

    storage = manager.empty((8,), torch.float32, "disabled")

    assert not storage.is_pinned()
    assert manager.allocated_bytes == 0
    assert manager.available_bytes == 16
    manager.release(storage)
    manager.assert_empty()


def test_bulk_placement_does_not_move_later_control_allocations(monkeypatch):
    from vllm.v1.spec_decode.retrospec import numa

    monkeypatch.setattr(numa, "_numa_library", lambda: None)
    allowed = os.sched_getaffinity(0)
    observed = []
    allocate = torch.empty

    def empty(*args, **kwargs):
        observed.append(os.sched_getaffinity(0))
        kwargs.pop("pin_memory", None)
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    manager = RetroSpecPinnedMemoryManager(enabled=True, max_bytes=32)
    placement = RetroSpecHostPlacement((min(allowed),))
    bulk = manager.empty((4,), torch.float32, "h2d", placement=placement)
    control = manager.empty((4,), torch.float32, "control")
    assert observed == [{min(allowed)}, allowed]
    assert os.sched_getaffinity(0) == allowed
    manager.release(bulk)
    manager.release(control)
    manager.assert_empty()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pinned_manager_enforces_shared_budget_and_releases_storage():
    manager = RetroSpecPinnedMemoryManager(enabled=True, max_bytes=32)
    first = manager.empty((4,), torch.float32, "first")
    second = manager.empty((4,), torch.float32, "second")

    assert first.is_pinned()
    assert second.is_pinned()
    assert manager.allocated_bytes == 32
    with pytest.raises(RuntimeError, match="retrospec_max_pinned_memory"):
        manager.empty((1,), torch.float32, "overflow")

    manager.release(first)
    assert manager.allocated_bytes == 16
    replacement = manager.replace(second, (6,), torch.float32, "replacement")
    assert replacement.is_pinned()
    assert manager.allocated_bytes == 24

    manager.release(replacement)
    manager.assert_empty()
