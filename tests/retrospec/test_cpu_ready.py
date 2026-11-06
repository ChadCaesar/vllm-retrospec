# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.spec_decode.retrospec.cpu_combined import RetroSpecCPUCombinedCoordinator


def _coordinator():
    cpu = RetroSpecCPUCombinedCoordinator.__new__(RetroSpecCPUCombinedCoordinator)
    cpu.proposer = SimpleNamespace(performance_stats=Mock())
    cpu.group = SimpleNamespace(world_size=1)
    cpu.alert = False
    cpu.active = cpu._pause_epoch = cpu._restore_pending = None
    cpu._ready_preparation = cpu._local_preparation = None
    cpu._local_preparation_error = None
    cpu.mode = "suspending"
    task = object()
    future = Future()
    future.set_result(None)
    kv = SimpleNamespace(future=Mock(wraps=future), cancelled=Event())
    cpu.queue = deque([task])
    cpu.preparing = (task, kv, 0)
    return cpu, kv


def test_ready_snapshot_is_confirmed_once_but_cancel_remains_live(monkeypatch):
    cpu, kv = _coordinator()
    cpu.group = SimpleNamespace(world_size=2, cpu_group=object())
    reduce = Mock()
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.cpu_combined.dist.all_reduce", reduce
    )
    for _ in range(4):
        cpu.pump(None)
    reduce.assert_called_once()
    kv.future.done.assert_called_once()
    kv.future.result.assert_called_once()
    kv.cancelled.set()
    cpu.queue.clear()
    cpu.pump(None)
    assert cpu.preparing is None


def test_ready_cache_cannot_authorize_another_snapshot():
    cpu, first = _coordinator()
    cpu.pump(None)
    second = SimpleNamespace(future=Future(), cancelled=Event())
    cpu.preparing = (cpu.queue[0], second, 0)
    cpu.pump(None)
    assert cpu.preparing[1] is second
    assert not second.future.done()


def test_ready_cache_does_not_bypass_fifo_ownership():
    cpu, kv = _coordinator()
    cpu.pump(None)
    cpu.queue = deque([object()])
    cpu.pump(None, allow_start=False)
    assert cpu.preparing is None


def test_local_done_is_cached_while_another_rank_is_pending(monkeypatch):
    cpu, kv = _coordinator()
    cpu.group = SimpleNamespace(world_size=2, cpu_group=object())
    remote_pending = [True]

    def reduce(status, **kwargs):
        status[0] = remote_pending[0]

    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.cpu_combined.dist.all_reduce", reduce
    )
    cpu.pump(None)
    cpu.pump(None)
    assert cpu._ready_preparation is None
    kv.future.done.assert_called_once()
    kv.future.result.assert_called_once()
    remote_pending[0] = False
    cpu.pump(None)
    assert cpu._ready_preparation is kv


@pytest.mark.parametrize("remote", [False, True])
def test_failed_preparation_is_never_cached_as_ready(monkeypatch, remote):
    cpu, kv = _coordinator()
    if remote:
        cpu.group = SimpleNamespace(world_size=2, cpu_group=object())

        def reduce(status, **kwargs):
            status[1] = 1

        def gather(rows, value, **kwargs):
            rows[:] = [None, "remote gather failed"]

        monkeypatch.setattr(
            "vllm.v1.spec_decode.retrospec.cpu_combined.dist.all_reduce", reduce
        )
        monkeypatch.setattr(
            "vllm.v1.spec_decode.retrospec.cpu_combined.dist.all_gather_object", gather
        )
    else:
        failed = Future()
        failed.set_exception(RuntimeError("local gather failed"))
        kv.future = failed
    with pytest.raises(RuntimeError, match="gather failed"):
        cpu.pump(None)
    assert cpu._ready_preparation is None
