# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from threading import Event
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.spec_decode.retrospec.cpu_combined import RetroSpecCPUCombinedCoordinator
from vllm.v1.spec_decode.retrospec.cpu_risk import RiskTask
from vllm.v1.spec_decode.retrospec.cpu_state import RetroSpecCPUWindow
from vllm.v1.spec_decode.retrospec.cpu_worker import RetroSpecCPUWorker


def coordinator():
    cpu = RetroSpecCPUCombinedCoordinator.__new__(RetroSpecCPUCombinedCoordinator)
    cpu.proposer = SimpleNamespace(performance_stats=Mock())
    cpu.group = SimpleNamespace(world_size=1, broadcast_object=lambda value, src: value)
    cpu.worker = Mock()
    cpu.bridge = Mock()
    cpu.storage = object()
    cpu.risk_tail = SimpleNamespace(paths=("risk",), shape=(2, 4, 1, 65, 8))
    cpu.continuous_tail = SimpleNamespace(paths=("continuous",), shape=(2, 4, 1, 65, 8))
    cpu.session, cpu.epoch, cpu._continuous_epoch = 1, 5, 5
    cpu.mode = "continuous"
    cpu._suspend_nonce = 0
    cpu._suspend_pending = None
    cpu._tail_installed = False
    cpu._pause_epoch = cpu._restore_pending = None
    cpu._ready_preparation = cpu._local_preparation = None
    cpu.active = None
    cpu.alert = False
    cpu._base = cpu.common_end = cpu.authoritative_length = cpu.staged_length = 100
    cpu._seed = 3
    cpu.candidates = {100: 7}
    cpu.queue = deque([RiskTask(100, 7, monotonic())])
    future = Future()
    future.set_result(None)
    cpu.preparing = (
        cpu.queue[0],
        SimpleNamespace(future=future, cancelled=Event()),
        monotonic(),
    )
    cpu._deferred_messages = []
    return cpu


def test_ready_risk_is_sent_with_checkpoint_without_an_ack_round_trip():
    cpu = coordinator()
    cpu.pump(None)
    command = cpu.worker.send.call_args.args[0]
    assert command == (
        "suspend",
        6,
        "continuous",
        cpu.continuous_tail.shape,
        100,
        ("check_shared", 1, 7, "risk", cpu.risk_tail.shape, 100, 100, 3),
    )
    assert cpu.mode == "suspending" and cpu.active == (cpu.queue[0], 7)
    assert cpu.preparing is None and cpu._suspend_pending == 6
    assert not cpu._tail_installed


def test_continuous_publication_waits_for_saved_tail_and_preserves_pending_results():
    cpu = coordinator()
    cpu.pump(None)
    cpu.authoritative_length = 101
    events = []
    cpu.worker.wait_for_suspension.side_effect = lambda nonce: events.append(
        ("fence", nonce)
    )
    cpu.bridge.stage_cpu.side_effect = lambda storage, *args: events.append(
        ("stage", storage)
    )
    cpu.synchronize(None)
    assert events == [("fence", 6), ("stage", cpu.continuous_tail)]
    assert cpu._suspend_pending is None and cpu._tail_installed
    assert cpu.staged_length == 101


def test_checkpoint_dispatch_does_not_wait_without_a_new_publication():
    cpu = coordinator()
    cpu.pump(None)
    cpu.synchronize(None)
    cpu.worker.wait_for_suspension.assert_not_called()
    cpu.bridge.stage_cpu.assert_not_called()


def test_failed_checkpoint_cannot_publish_shared_risk_data():
    cpu = coordinator()
    cpu.pump(None)
    cpu.authoritative_length = 101
    cpu.worker.wait_for_suspension.side_effect = RuntimeError("checkpoint failed")
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        cpu.synchronize(None)
    cpu.bridge.stage_cpu.assert_not_called()
    assert cpu.staged_length == 100


@pytest.mark.parametrize("session,epoch,nonce", [(0, 5, 6), (1, 4, 6), (1, 5, 4)])
def test_old_checkpoint_ack_cannot_install_a_tail(session, epoch, nonce):
    cpu = coordinator()
    cpu.pump(None)
    cpu._poll_messages = lambda: [("suspended", session, epoch, nonce, 100, 0, 2)]
    cpu._consume_continuous = Mock(return_value=False)
    cpu.update_candidates = Mock()
    cpu.poll(RetroSpecCPUWindow(100, tokens=[7]))
    assert cpu._suspend_pending == 6 and not cpu._tail_installed
    assert cpu.active == (cpu.queue[0], 7)


def test_current_ack_installs_tail_without_dispatching_risk_again():
    cpu = coordinator()
    cpu.pump(None)
    cpu._poll_messages = lambda: [("suspended", 1, 5, 6, 100, 0, 2)]
    cpu._consume_continuous = Mock(return_value=False)
    cpu.update_candidates = Mock()
    cpu.poll(RetroSpecCPUWindow(100, tokens=[7]))
    assert cpu._suspend_pending is None and cpu._tail_installed
    assert cpu.mode == "risk"
    cpu.worker.send.assert_called_once()


def test_suspension_fence_retains_messages_in_wire_order():
    worker = RetroSpecCPUWorker.__new__(RetroSpecCPUWorker)
    worker.messages = deque()
    worker.connection = Mock()
    worker.suspended = None
    first = ("token", 1, 5, 101, 7)
    ack = ("suspended", 1, 5, 6)
    second = ("token", 1, 7, 101, 9)
    batches = iter([[first], [ack, second]])

    def poll():
        messages = next(batches)
        if ack in messages:
            worker.suspended = 6
        return messages

    worker.poll = poll
    worker.wait_for_suspension(6, timeout=1)
    assert list(worker.messages) == [first, ack, second]


def test_fused_dispatch_tracks_risk_progress_after_checkpoint_ack():
    worker = RetroSpecCPUWorker.__new__(RetroSpecCPUWorker)
    worker.process = Mock()
    worker.process.poll.return_value = None
    worker.connection = Mock()
    worker.connection.poll.side_effect = [True, False]
    worker.connection.recv.return_value = ("suspended", 1, 5, 6)
    worker.closed, worker.ready, worker.awaiting_progress = False, True, False
    worker.messages = deque()
    worker.last_progress = monotonic()
    check = ("check_shared", 1, 7, "risk", (2, 4), 100, 100, 3)
    worker.send(("suspend", 6, "continuous", (2, 4), 100, check))
    assert worker.last_command == ("check", 1, 7, 101)
    worker.poll()
    assert worker.suspended == 6 and worker.awaiting_progress
