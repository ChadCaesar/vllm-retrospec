# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
import weakref
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

from tests.retrospec.offload.cluster_store_helpers import (
    make_cluster_data,
    store_cluster_data,
)
from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.retrospec.cluster_store import RetroSpecClusterPageStore
from vllm.v1.spec_decode.retrospec.cpu_bridge import RetroSpecCPUKVBridge
from vllm.v1.spec_decode.retrospec.cpu_coordinator import RetroSpecCPUCoordinator
from vllm.v1.spec_decode.retrospec.cpu_model import attach_cache, make_cache
from vllm.v1.spec_decode.retrospec.cpu_state import RetroSpecCPUWindow
from vllm.v1.spec_decode.retrospec.cpu_worker import (
    RetroSpecCPUStorage,
    RetroSpecCPUWorker,
)
from vllm.v1.spec_decode.retrospec.decision import (
    RetroSpecDecisionPolicy,
    RetroSpecReason,
)
from vllm.v1.spec_decode.retrospec.numa import RetroSpecHostPlacement
from vllm.v1.spec_decode.retrospec.runtime.proposer_cpu import RetroSpecCPUProposalMixin
from vllm.v1.spec_decode.retrospec.state import RetroSpecBatchState


def test_cpu_storage_unlinks_paths_before_releasing_large_mappings():
    storage = RetroSpecCPUStorage((2, 1, 1, 8, 1))
    directory = Path(storage.directory)
    paths_present_at_release = []
    for tensor in (storage.kv, storage.journal):
        weakref.finalize(
            tensor, lambda: paths_present_at_release.append(directory.exists())
        )
    del tensor
    storage.close()
    assert paths_present_at_release == [False, False]
    assert not directory.exists()


@pytest.mark.parametrize("fail_at", [None, 1, 2])
def test_cpu_storage_memfd_and_file_fallback_share_data(monkeypatch, fail_at):
    original = os.memfd_create
    descriptors = []

    def allocate(*args):
        if len(descriptors) + 1 == fail_at:
            raise OSError("memfd unavailable")
        fd = original(*args)
        descriptors.append(fd)
        return fd

    monkeypatch.setattr(os, "memfd_create", allocate)
    owner = RetroSpecCPUStorage((2, 1, 1, 8, 1))
    peer = RetroSpecCPUStorage(owner.shape, owner.directory)
    try:
        assert owner.backing == peer.backing == ("memfd" if fail_at is None else "file")
        owner.kv.fill_(7)
        peer.journal.fill_(11)
        assert (peer.kv == 7).all()
        assert (owner.journal == 11).all()
    finally:
        peer.close()
        owner.close()
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_cpu_storage_uses_files_when_shared_ram_budget_is_insufficient(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    allocate = Mock(side_effect=AssertionError("RAM budget was ignored"))
    monkeypatch.setattr(os, "memfd_create", allocate)
    storage = RetroSpecCPUStorage((2, 1, 1, 8, 1))
    try:
        assert storage.backing == "file"
        assert not allocate.called
    finally:
        storage.close()


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(
    "rank,world_size,total_heads",
    [(0, 1, 2), (0, 2, 4), (1, 2, 4), (0, 2, 1), (1, 2, 1)],
)
def test_cpu_export_gathers_directly_into_shared_head_slots(
    monkeypatch, indexed, rank, world_size, total_heads
):
    group = SimpleNamespace(rank_in_group=rank, world_size=world_size)
    monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: group)
    heads = max(1, total_heads // world_size)
    store = RetroSpecClusterPageStore(page_size=2)
    index = SimpleNamespace(
        cluster_store=store, _indices={}, flush_staged_updates=Mock()
    )
    layers = {}
    end = 15 if indexed else 7
    table = torch.tensor([[3, -1, -1, -1, -1, -1, 1, 0]])
    if not indexed:
        table = torch.tensor([[3, 2, 1, 0]])
    storage = RetroSpecCPUStorage((2, 2, total_heads, 20, 1))
    try:
        storage.kv.fill_(-999)
        storage.journal.fill_(-777)
        expected = storage.kv.clone()
        expected_journal = storage.journal.clone()
        writer = total_heads >= world_size or rank == 0
        head_start = rank * heads if total_heads >= world_size else 0
        head_slice = slice(head_start, head_start + heads)
        for layer_index in range(2):
            name = f"layer{layer_index}"
            native = torch.arange(2 * 4 * 2 * heads).view(2, 4, 2, heads, 1)
            native = native.to(torch.bfloat16) + layer_index * 1000
            layers[name] = SimpleNamespace(
                impl=SimpleNamespace(num_kv_heads=heads), kv_cache=[native]
            )
            if indexed:
                descriptor = None
                segments = []
                for segment_index in range(2):
                    keys, values, assignments, counts = make_cluster_data()
                    keys = keys[:heads].to(torch.bfloat16) + segment_index * 200
                    values = values[:heads].to(torch.bfloat16) + segment_index * 200
                    blocks = store_cluster_data(
                        store,
                        name,
                        keys,
                        values,
                        assignments[:heads],
                        counts[:heads],
                        cluster_start=segment_index * 2,
                    )
                    current = blocks.full_verification_descriptor
                    descriptor = (
                        current if descriptor is None else descriptor.append(current)
                    )
                    segments.append(SimpleNamespace(cluster_blocks=blocks))
                    pages = blocks.page_metadata
                    old_keys, old_values, mask = store.gather_pages(
                        name,
                        pages.page_ids.unsqueeze(0),
                        pages.page_token_counts.unsqueeze(0),
                    )
                    if writer:
                        for kind, packed in enumerate((old_keys, old_values)):
                            for head in range(heads):
                                start = 2 + segment_index * 5
                                expected[
                                    kind,
                                    layer_index,
                                    head_start + head,
                                    start : start + 5,
                                ] = packed[0, head][mask[0, head]]
                index._indices[name] = {
                    "r": SimpleNamespace(
                        segments=segments,
                        indexed_end=12,
                        full_verification_descriptor=descriptor,
                    )
                }
            if writer:
                positions = (
                    list(range(2)) + list(range(12, end))
                    if indexed
                    else list(range(end))
                )
                for position in positions:
                    expected[:, layer_index, head_slice, position] = native[
                        :, table[0, position // 2], position % 2
                    ]
                expected_journal[:, layer_index, head_slice, end - 1] = native[
                    :, table[0, (end - 1) // 2], (end - 1) % 2
                ]
        # A retired historical native block must never be read. The old
        # padded gather is only the reference, never the export path.
        store.gather_pages = Mock(side_effect=AssertionError("padded gather"))
        store.gather_full_verification_into = Mock(
            wraps=store.gather_full_verification_into
        )
        bridge = RetroSpecCPUKVBridge.__new__(RetroSpecCPUKVBridge)
        bridge.export_placement = RetroSpecHostPlacement()
        bridge.proposer = SimpleNamespace(
            sparse_attention=SimpleNamespace(index=index), block_size=2
        )
        bridge.layers = layers
        metadata = SimpleNamespace(block_table_tensor=table)
        bridge.export_gpu(storage, metadata, "r", 0, end, True)
        torch.testing.assert_close(
            storage.kv.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
        assert (storage.journal == -777).all()
        assert store.gather_full_verification_into.call_count == (
            2 if indexed and writer else 0
        )
        bridge.export_gpu(storage, metadata, "r", end - 1, end, False)
        torch.testing.assert_close(
            storage.kv.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
        torch.testing.assert_close(
            storage.journal.view(torch.int16),
            expected_journal.view(torch.int16),
            rtol=0,
            atol=0,
        )
    finally:
        storage.close()
        store.close()


@pytest.mark.parametrize("quality_trigger", [None, 7, 19])
@pytest.mark.parametrize("cpu_during_verification", [False, True])
@pytest.mark.parametrize("execution_slice", [False, True])
def test_cpu_refills_only_length_triggered_verification(
    quality_trigger, cpu_during_verification, execution_slice
):
    class CPU:
        stop_ids = set()

        def begin(self, *args):
            pass

        def poll(self, window):
            if window.draft_end >= 8 and not window.confirmed:
                window.confirm(101, 1)
                window.confirm(102, 2)
            if (
                cpu_during_verification
                and window.draft_end == 18
                and window.confirmed == 2
                and proposer.ranges[-1] == (11, 18)
            ):
                window.confirm(103, 3)
                window.confirm(104, 4)
            return False

        def synchronize(self, *args, **kwargs):
            pass

        def freeze(self, window, metadata):
            self.committed = window.tokens[: window.confirmed]

    class Proposer(RetroSpecCPUProposalMixin):
        def _run_draft_step(self, batch, slot, metadata, active, sampling):
            margin = 0.0 if slot == quality_trigger else 1.0
            return torch.tensor([slot + 1]), torch.tensor([margin]), torch.ones(1)

        def _verify_draft_tokens(
            self,
            batch,
            ids,
            proposal_round,
            starts,
            metadata,
            sampling,
            confirmed_counts,
        ):
            start = int(starts[0])
            count = int(self.state.draft_counts[0])
            self.ranges.append((start + 1, start + count))
            quality = quality_trigger is not None and start + count > quality_trigger
            return SimpleNamespace(
                verified_counts=torch.tensor([count]),
                require_full=torch.tensor(
                    [quality or start + count - int(confirmed_counts[0]) >= 16]
                ),
                full_reasons=torch.tensor(
                    [
                        int(RetroSpecReason.EXPANDED_MARGIN)
                        if quality
                        else int(RetroSpecReason.PENDING_LIMIT)
                    ]
                ),
            )

    proposer = Proposer()
    proposer.cpu_coordinator = CPU()
    proposer.device = torch.device("cpu")
    proposer.speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=16,
        retrospec_max_draft_tokens=8,
        retrospec_draft_margin_threshold=0.5 if quality_trigger is not None else None,
    )
    proposer.policy = RetroSpecDecisionPolicy(proposer.speculative_config)
    proposer.num_speculative_tokens = 32
    proposer.max_model_len = 1000
    proposer._proposal_token_budgets = SimpleNamespace(
        gpu=torch.tensor([12 if execution_slice else 24])
    )
    proposer.proposal_start_positions = torch.tensor([100])
    proposer.proposal_input_ids = torch.tensor([99])
    proposer.positions = torch.zeros(1, dtype=torch.long)
    proposer.input_ids = torch.zeros(1, dtype=torch.long)
    proposer._draft_token_ids = torch.full((1, 32), -1, dtype=torch.int32)
    proposer.runner = SimpleNamespace(
        requests={"r": SimpleNamespace(sampling_params=None)}
    )
    proposer.state = RetroSpecBatchState(1, torch.device("cpu"))
    proposer.state.begin_batch(1)
    proposer.sparse_attention = SimpleNamespace(
        proposal_context=lambda *args: nullcontext(), selection_provenance_enabled=False
    )
    proposer.index_update_state = SimpleNamespace(
        next_update_positions=torch.tensor([999])
    )
    proposer.performance_stats = Mock()
    proposer.ranges = []
    result = proposer._propose_with_cpu(
        ["r"], [100], None, None, remaining_generation_budget=24
    )
    if quality_trigger == 7:
        assert proposer.cpu_coordinator.committed == [1, 2]
        assert proposer.ranges == [(3, 8)]
        assert result == [list(range(3, 9))]
    elif execution_slice:
        assert proposer.cpu_coordinator.committed == [1, 2]
        assert proposer.ranges == [(3, 10)]
        assert result == [list(range(3, 13))]
        assert proposer.cpu_coordinator.deferred_window == (
            "r",
            102,
            list(range(3, 13)),
            8,
        )
    elif cpu_during_verification:
        assert proposer.cpu_coordinator.committed == [1, 2, 3, 4]
        assert proposer.ranges == [(3, 10), (11, 18)] + (
            [(19, 20)] if quality_trigger == 19 else []
        )
        assert result == [list(range(5, 21))]
        if quality_trigger is None:
            proposer.performance_stats.add_counter.assert_any_call(
                "cpu_small_tail_full_handoffs"
            )
    else:
        assert proposer.cpu_coordinator.committed == [1, 2]
        assert proposer.ranges == [(3, 10), (11, 18)]
        assert result == [list(range(3, 19))]


def test_cpu_confirmation_slides_both_windows():
    window = RetroSpecCPUWindow(100, tokens=list(range(1, 9)))
    assert not window.confirm(101, 1)
    assert not window.confirm(102, 2)
    assert window.sparse_pending == window.full_pending == 6
    window.tokens.extend([9, 10])
    assert window.sparse_pending == window.full_pending == 8
    assert window.apply_sparse(0, 2, list(range(3, 11)))
    assert window.sparse_pending == 0
    assert window.full_pending == 8


def test_cpu_correction_invalidates_snapshot_without_rollback():
    window = RetroSpecCPUWindow(10, tokens=[1, 2, 3, 4])
    window.confirm(11, 1)
    snapshot = window.version
    assert window.confirm(12, 9)
    assert window.tokens == [1, 9]
    assert window.confirmed == 2
    assert not window.apply_sparse(snapshot, 1, [2, 3, 4])
    with pytest.raises(RuntimeError, match="committed output"):
        window.confirm(11, 8)


def test_cpu_progress_does_not_move_verification_snapshot():
    window = RetroSpecCPUWindow(10, tokens=[1, 2, 3, 4])
    window.confirm(11, 1)
    window.confirm(12, 2)
    assert window.apply_sparse(0, 0, [1, 2, 8])
    assert window.tokens == [1, 2, 8]
    assert window.confirmed == 2
    assert not window.apply_sparse(0, 0, [9, 2, 8])


def test_cpu_can_advance_beyond_gpu_draft():
    window = RetroSpecCPUWindow(10)
    assert window.confirm(11, 7)
    assert window.confirm(12, 8)
    assert window.tokens == [7, 8]
    assert window.full_pending == window.sparse_pending == 0
    with pytest.raises(RuntimeError, match="skipped a token"):
        window.confirm(14, 3)


def _make_cpu_coordinator():
    cpu = RetroSpecCPUCoordinator.__new__(RetroSpecCPUCoordinator)
    cpu.proposer = SimpleNamespace(
        max_model_len=64,
        runner=SimpleNamespace(requests={"r": SimpleNamespace(num_prompt_tokens=8)}),
        performance_stats=Mock(),
    )
    cpu.config = LlamaConfig(
        num_hidden_layers=2,
        num_key_value_heads=2,
        num_attention_heads=4,
        hidden_size=64,
    )
    cpu.group = SimpleNamespace(
        rank_in_group=0, broadcast_object=lambda value, src: value, barrier=Mock()
    )
    cpu.bridge = Mock()
    cpu.worker = Mock()
    cpu.storage = cpu.request_id = None
    cpu._pooled_storage = cpu._pending_release = None
    cpu.session = cpu.epoch = cpu.authoritative_length = 0
    cpu.committed_tokens = {}
    return cpu


def test_cpu_mapping_covers_later_proposal_rounds():
    cpu = _make_cpu_coordinator()
    params = SimpleNamespace(max_tokens=24, ignore_eos=True, stop_token_ids=[])
    try:
        cpu.begin("r", 8, 10, 4, None, params)
        assert cpu.storage.shape[-2] == 32
        cpu.authoritative_length = cpu._resume_start = 10
        cpu.begin("r", 14, 11, 4, None, params)
        assert cpu.storage.shape[-2] == 32
        cpu.bridge.export_gpu.assert_called_with(cpu.storage, None, "r", 10, 14, False)
        assert cpu.worker.send.call_args.args[0] == ("resume", 1, 2, 10, 14, 11, 18)
    finally:
        cpu.close()


def test_cpu_mapping_reuse_fences_old_session_and_preserves_capacity():
    cpu = _make_cpu_coordinator()
    params = SimpleNamespace(max_tokens=24, ignore_eos=True, stop_token_ids=[])
    try:
        cpu.begin("r", 8, 10, 4, None, params)
        storage = cpu.storage
        pointer = storage.kv.data_ptr()
        storage.kv.fill_(-17)
        paths = storage.paths
        cpu.release()
        assert cpu.storage is None
        assert cpu._pooled_storage is storage
        assert cpu._pending_release == (1, 2)
        cpu.worker.send.assert_called_with(("release", 1, 2))
        assert all(Path(path).exists() for path in paths)

        order = []

        def acknowledged(session, epoch):
            assert (session, epoch) == (1, 2)
            assert cpu.bridge.export_gpu.call_count == 1
            assert (storage.kv == -17).all()
            order.append("released")

        def exported(*args):
            assert order == ["released"]
            order.append("exported")

        cpu.worker.wait_for_release.side_effect = acknowledged
        cpu.bridge.export_gpu.side_effect = exported
        cpu.proposer.runner.requests["short"] = SimpleNamespace(num_prompt_tokens=4)
        params.max_tokens = 12
        cpu.begin("short", 4, 20, 4, None, params)
        assert order == ["released", "exported"]
        assert cpu.storage is storage
        assert storage.kv.data_ptr() == pointer
        assert storage.shape[-2] == 32
        assert (storage.kv == -17).all()  # No clearing or remapping on reuse.
        assert cpu._pending_release is None
        assert cpu._pooled_storage is None
        assert cpu.worker.send.call_args.args[0] == (
            "initialize",
            2,
            3,
            paths,
            storage.shape,
            4,
            20,
            8,
            (),
        )

        cpu.release()
        cpu.worker.wait_for_release.side_effect = None
        cpu.bridge.export_gpu.side_effect = None
        cpu.proposer.runner.requests["long"] = SimpleNamespace(num_prompt_tokens=60)
        params.max_tokens = 24
        cpu.begin("long", 60, 30, 4, None, params)
        cpu.worker.wait_for_release.assert_called_with(2, 4)
        assert cpu.storage.shape[-2] == cpu.proposer.max_model_len
        assert cpu.storage.paths != paths
        assert not any(Path(path).exists() for path in paths)
        paths = cpu.storage.paths
        cpu.release()
    finally:
        cpu.close()
    assert not any(Path(path).exists() for path in paths)


def test_cpu_pool_is_unlinked_before_waiting_for_process_teardown():
    cpu = _make_cpu_coordinator()
    storage = RetroSpecCPUStorage((2, 2, 2, 32, 16))
    cpu._pooled_storage = storage
    paths = storage.paths
    order = []

    def requested():
        assert all(Path(path).exists() for path in paths)
        order.append("close_requested")

    def joined():
        assert not any(Path(path).exists() for path in paths)
        order.append("joined")

    cpu.worker.request_close.side_effect = requested
    cpu.worker.close.side_effect = joined
    cpu.close()
    assert order == ["close_requested", "joined"]
    assert cpu._pooled_storage is None


@pytest.mark.parametrize("rank", [0, 1])
def test_cpu_reuse_failure_keeps_mapping_quarantined_on_every_rank(rank):
    cpu = _make_cpu_coordinator()
    storage = RetroSpecCPUStorage((2, 2, 2, 32, 16))
    cpu._pooled_storage = storage
    cpu._pending_release = (1, 2)
    storage.kv.fill_(-17)
    cpu.group.rank_in_group = rank
    if rank == 0:
        cpu.worker.wait_for_release.side_effect = RuntimeError("release timed out")
    else:
        cpu.worker = None
        cpu.group.broadcast_object = lambda value, src: (
            "release timed out",
            None,
            None,
            False,
        )
    try:
        with pytest.raises(RuntimeError, match="release timed out"):
            cpu._acquire_storage((2, 2, 2, 16, 16))
        assert cpu.storage is None
        assert cpu._pooled_storage is storage
        assert cpu._pending_release == (1, 2)
        assert (storage.kv == -17).all()
        cpu.bridge.export_gpu.assert_not_called()
    finally:
        cpu.close()


@pytest.mark.parametrize("rank", [0, 1])
def test_cpu_reuse_preserves_tp_mappings(rank):
    cpu = _make_cpu_coordinator()
    owner = RetroSpecCPUStorage((2, 2, 2, 32, 16))
    cpu.group.rank_in_group = rank
    cpu._pooled_storage = (
        owner if rank == 0 else RetroSpecCPUStorage(owner.shape, owner.directory)
    )
    storage = cpu._pooled_storage
    pointer = storage.kv.data_ptr()
    cpu._pending_release = (1, 2)
    cpu.group.broadcast_object = lambda value, src: (
        None,
        owner.directory,
        owner.shape,
        True,
    )
    try:
        cpu._acquire_storage((2, 2, 2, 16, 16))
        assert cpu.storage is storage
        assert cpu.storage.kv.data_ptr() == pointer
        assert cpu.storage.shape == owner.shape
    finally:
        cpu.close()
        if rank != 0:
            owner.close()


@pytest.mark.parametrize("shape", [(2, 2, 3, 16, 16), (2, 2, 2, 16, 8)])
def test_cpu_pool_rejects_incompatible_layout(shape):
    storage = RetroSpecCPUStorage((2, 2, 2, 32, 16))
    try:
        assert storage.can_reuse((2, 2, 2, 16, 16))
        assert not storage.can_reuse((2, 2, 2, 33, 16))
        assert not storage.can_reuse(shape)
    finally:
        storage.close()


def _make_protocol_worker(connection):
    worker = RetroSpecCPUWorker.__new__(RetroSpecCPUWorker)
    worker.connection = connection
    worker.process = Mock(pid=123)
    worker.process.poll.return_value = None
    worker.closed, worker.ready, worker.awaiting_progress = False, True, True
    worker.messages = deque()
    worker.released = None
    worker.started_at = worker.last_progress = time.monotonic()
    worker.last_command = ("release", 1, 4, 0)
    return worker


def test_cpu_release_wait_requires_matching_session_and_epoch():
    class Connection:
        messages = deque([("token", 1, 3, 11, 7), ("released", 1, 3)])
        waits = 0

        def poll(self, timeout=0):
            if timeout:
                self.waits += 1
                assert worker.released != (1, 4)
                self.messages.append(
                    ("released", 2, 4) if self.waits == 1 else ("released", 1, 4)
                )
            return bool(self.messages)

        def recv(self):
            return self.messages.popleft()

    connection = Connection()
    worker = _make_protocol_worker(connection)
    worker.wait_for_release(1, 4, timeout=1)
    assert connection.waits == 2
    assert worker.released == (1, 4)
    assert not worker.awaiting_progress
    assert not worker.messages


def test_cpu_release_wait_times_out_without_fence():
    connection = Mock()
    connection.poll.return_value = False
    worker = _make_protocol_worker(connection)
    with pytest.raises(RuntimeError, match="did not release KV.*session=1, epoch=4"):
        worker.wait_for_release(1, 4, timeout=0)


def test_cpu_freeze_invalidates_inflight_result_without_waiting():
    cpu = RetroSpecCPUCoordinator.__new__(RetroSpecCPUCoordinator)
    cpu.proposer = SimpleNamespace(num_speculative_tokens=32, performance_stats=Mock())
    cpu.group = SimpleNamespace(broadcast_object=lambda value, src: value)
    cpu.worker = Mock()
    cpu.worker.poll.return_value = [("token", 1, 4, 11, 7)]
    cpu.session, cpu.epoch, cpu.request_id = 1, 4, "r"
    cpu.committed_tokens = {}
    cpu.synchronize = Mock()
    window = RetroSpecCPUWindow(10, tokens=[7, 8])
    cpu.poll(window)
    cpu.freeze(window, None)
    cpu.worker.send.assert_called_once_with(("pause", 5))
    assert cpu.committed_tokens == {"r": [7]}
    cpu.worker.poll.return_value = [("token", 1, 4, 12, 9)]
    cpu.poll(window)
    assert window.confirmed == 1
    assert cpu.authoritative_length == 11


def test_cpu_worker_reports_lost_progress():
    worker = RetroSpecCPUWorker.__new__(RetroSpecCPUWorker)
    worker.connection = Mock()
    worker.connection.poll.return_value = False
    worker.process = Mock(pid=123)
    worker.process.poll.return_value = None
    worker.closed, worker.ready, worker.awaiting_progress = False, True, True
    worker.messages = deque()
    worker.last_progress = time.monotonic() - 301
    worker.last_command = ("resume", 1, 4, 100)
    with pytest.raises(RuntimeError, match="made no progress.*pid=123"):
        worker.poll()


def test_cpu_failure_is_propagated_to_every_tp_rank():
    cpu = RetroSpecCPUCoordinator.__new__(RetroSpecCPUCoordinator)
    cpu.group = SimpleNamespace(broadcast_object=lambda value, src: value)
    cpu.worker = Mock()
    cpu.worker.poll.side_effect = RuntimeError("CPU inference failed")
    with pytest.raises(RuntimeError, match="CPU inference failed"):
        cpu.poll(RetroSpecCPUWindow(10))


def test_deferred_selection_plans_survive_the_next_proposal_reset():
    proposer = RetroSpecCPUProposalMixin()
    rows = {
        field: torch.arange(32).reshape(8, 1, 4).clone()
        for field in proposer._CPU_PLAN_ROWS
    }
    table = SimpleNamespace(
        **rows,
        request_slot_ids=torch.tensor([3]),
        request_slot_generations=torch.tensor([7]),
    )
    index = SimpleNamespace(
        _selection_plan_tables={"layer": table},
        _selection_plan_written_layers=set(),
    )
    proposer.sparse_attention = SimpleNamespace(index=index)
    saved = proposer._cpu_save_plans(2, 5)
    for tensor in rows.values():
        tensor.zero_()
    table.request_slot_ids.zero_()
    table.request_slot_generations.zero_()
    proposer._cpu_restore_plans(saved)
    for tensor in rows.values():
        assert torch.equal(tensor[:3], torch.arange(8, 20).reshape(3, 1, 4))
    assert table.request_slot_ids.tolist() == [3]
    assert table.request_slot_generations.tolist() == [7]
    assert index._selection_plan_written_layers == {"layer"}


@pytest.mark.parametrize("config_class", [LlamaConfig, Qwen2Config])
def test_full_cpu_attention_accepts_independently_permuted_historical_heads(
    config_class,
):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(4)
        config = config_class(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        config._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(config).eval()
        cache = make_cache(config, 32)
        storage = torch.empty(2, 2, 2, 32, 16)
        with torch.inference_mode():
            model(
                torch.tensor([[1, 2, 3, 4, 5]]), past_key_values=cache, use_cache=True
            )
            for layer_index, layer in enumerate(cache.layers):
                for head in range(2):
                    permutation = torch.randperm(5)
                    storage[0, layer_index, head, :5] = layer.keys[0, head, permutation]
                    storage[1, layer_index, head, :5] = layer.values[
                        0, head, permutation
                    ]
            permuted = attach_cache(config, storage, 5)
            expected = model(
                torch.tensor([[6]]), past_key_values=cache, use_cache=True
            ).logits
            actual = model(
                torch.tensor([[6]]),
                past_key_values=permuted,
                cache_position=torch.tensor([5]),
                use_cache=True,
            ).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    finally:
        torch.set_num_threads(previous_threads)


def _wait_messages(worker, predicate, timeout=60):
    deadline = time.monotonic() + timeout
    messages = []
    while time.monotonic() < deadline:
        messages.extend(worker.poll())
        if predicate(messages):
            return messages
        time.sleep(0.01)
    raise AssertionError(f"CPU worker made no expected progress: {messages}")


@pytest.mark.parametrize("control", ["pause", "release"])
def test_cpu_partial_forward_can_be_cancelled_and_resumed(tmp_path, control):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    worker = storage = None
    try:
        torch.manual_seed(4)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        config._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(config).bfloat16().eval()
        model_path = tmp_path / "model"
        model.save_pretrained(model_path)
        shape = (2, 16, 4, 16, 64)
        storage = RetroSpecCPUStorage(shape)
        storage.kv.zero_()
        cache = make_cache(config, 16)
        with torch.inference_mode():
            output = model(
                torch.tensor([[1, 2, 3, 4, 5]]), past_key_values=cache, use_cache=True
            )
            seed = int(output.logits[0, -1].argmax())
            for i, layer in enumerate(cache.layers):
                storage.kv[0, i, :, :5].copy_(layer.keys[0, :, :5])
                storage.kv[1, i, :, :5].copy_(layer.values[0, :, :5])
            prefix = storage.kv[..., :5, :].clone()
            worker = RetroSpecCPUWorker(str(model_path), None, 1)
            worker.send(("initialize", 1, 1, storage.paths, shape, 5, seed, 6, ()))
            deadline = time.monotonic() + 60
            while not storage.kv[0, 0, :, 5].count_nonzero():
                worker.poll()
                assert time.monotonic() < deadline, "CPU did not enter its first layer"
                time.sleep(0.0001)
            worker.send((control, 1, 2) if control == "release" else ("pause", 2))
            acknowledged = "released" if control == "release" else "paused"
            messages = _wait_messages(
                worker, lambda values: any(value[0] == acknowledged for value in values)
            )
            assert any(value[0] == "interrupted" for value in messages)
            assert not any(value[0] == "token" for value in messages)
            assert torch.equal(storage.kv[..., :5, :], prefix)

            if control == "release":
                # The fence remains valid even when a poll already drained it.
                worker.wait_for_release(1, 2, timeout=1)
                storage.kv.fill_(-17)
                cache = make_cache(config, 16)
                output = model(
                    torch.tensor([[6, 7, 8]]), past_key_values=cache, use_cache=True
                )
                seed = int(output.logits[0, -1].argmax())
                for i, layer in enumerate(cache.layers):
                    storage.kv[0, i, :, :3].copy_(layer.keys[0, :, :3])
                    storage.kv[1, i, :, :3].copy_(layer.values[0, :, :3])
                worker.send(("initialize", 2, 3, storage.paths, shape, 3, seed, 4, ()))
            else:
                worker.send(("resume", 1, 3, 5, 5, seed, 6))
            messages = _wait_messages(
                worker, lambda values: any(value[0] == "token" for value in values)
            )
            output = model(
                torch.tensor([[seed]]), past_key_values=cache, use_cache=True
            )
            token = next(value[4] for value in messages if value[0] == "token")
            assert token == int(output.logits[0, -1].argmax())
            if control == "release":
                assert (storage.kv[..., 5:, :] == -17).all()
    finally:
        if worker is not None:
            worker.close()
        if storage is not None:
            storage.close()
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("bind_cpu", [False, True])
def test_cpu_process_reuses_gpu_kv_delta_without_reforward(tmp_path, bind_cpu):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    worker = storage = None
    try:
        torch.manual_seed(4)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        config._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(config).bfloat16().eval()
        model_path = tmp_path / "model"
        model.save_pretrained(model_path)
        shape = (2, 2, 2, 32, 16)
        storage = RetroSpecCPUStorage(shape)
        cache = make_cache(config, 32)
        with torch.inference_mode():
            output = model(
                torch.tensor([[1, 2, 3, 4, 5]]), past_key_values=cache, use_cache=True
            )
            for i, layer in enumerate(cache.layers):
                storage.kv[0, i, :, :5].copy_(layer.keys[0, :, :5])
                storage.kv[1, i, :, :5].copy_(layer.values[0, :, :5])
            seed = int(output.logits[0, -1].argmax())
            parent_affinity = set(os.sched_getaffinity(0))
            parent_environment = dict(os.environ)
            cpus = tuple(sorted(parent_affinity)[:2]) if bind_cpu else None
            worker = RetroSpecCPUWorker(str(model_path), None, 2, cpu_affinity=cpus)
            worker.send(("initialize", 1, 1, storage.paths, shape, 5, seed, 9, ()))
            messages = _wait_messages(
                worker, lambda values: sum(value[0] == "token" for value in values) == 4
            )
            actual = [value[4] for value in messages if value[0] == "token"]
            assert not Path(worker.socket_directory).exists()
            assert set(os.sched_getaffinity(0)) == parent_affinity
            assert dict(os.environ) == parent_environment
            if bind_cpu:
                for task in Path(f"/proc/{worker.process.pid}/task").iterdir():
                    assert set(os.sched_getaffinity(int(task.name))) <= set(cpus)
            expected = []
            for position in range(5, 9):
                output = model(
                    torch.tensor([[seed]]), past_key_values=cache, use_cache=True
                )
                seed = int(output.logits[0, -1].argmax())
                expected.append(seed)
            assert actual == expected
            worker.send(("pause", 2))
            _wait_messages(
                worker, lambda values: any(value[0] == "paused" for value in values)
            )

            cache.crop(5)
            output = model(
                torch.tensor([[9, 10]]), past_key_values=cache, use_cache=True
            )
            for i, layer in enumerate(cache.layers):
                storage.journal[0, i, :, 5:7].copy_(layer.keys[0, :, 5:7])
                storage.journal[1, i, :, 5:7].copy_(layer.values[0, :, 5:7])
            seed = int(output.logits[0, -1].argmax())
            worker.send(("resume", 1, 3, 5, 7, seed, 8))
            messages = _wait_messages(
                worker,
                lambda values: any(
                    value[0] == "token" and value[2] == 3 for value in values
                ),
            )
            output = model(
                torch.tensor([[seed]]), past_key_values=cache, use_cache=True
            )
            actual = next(
                value[4] for value in messages if value[0] == "token" and value[2] == 3
            )
            assert actual == int(output.logits[0, -1].argmax())
            for i, layer in enumerate(cache.layers):
                assert torch.equal(storage.kv[0, i, :, 5:7], layer.keys[0, :, 5:7])
    finally:
        if worker is not None:
            worker.close()
        if storage is not None:
            storage.close()
        torch.set_num_threads(previous_threads)
