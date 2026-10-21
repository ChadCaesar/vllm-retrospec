# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

from vllm.config import SpeculativeConfig
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
from vllm.v1.spec_decode.retrospec.runtime.proposer_cpu import RetroSpecCPUProposalMixin
from vllm.v1.spec_decode.retrospec.state import RetroSpecBatchState


@pytest.mark.parametrize("quality_trigger", [False, True])
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
            margin = 0.0 if slot == 7 and quality_trigger else 1.0
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
            return SimpleNamespace(
                verified_counts=torch.tensor([count]),
                require_full=torch.tensor(
                    [quality_trigger or start + count - int(confirmed_counts[0]) >= 16]
                ),
                full_reasons=torch.tensor(
                    [
                        int(RetroSpecReason.EXPANDED_MARGIN)
                        if quality_trigger
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
        retrospec_draft_margin_threshold=0.5 if quality_trigger else None,
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
    if quality_trigger:
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
        assert proposer.ranges == [(3, 10), (11, 18), (19, 20)]
        assert result == [list(range(5, 21))]
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


def test_cpu_mapping_covers_later_proposal_rounds():
    cpu = RetroSpecCPUCoordinator.__new__(RetroSpecCPUCoordinator)
    cpu.proposer = SimpleNamespace(
        max_model_len=64,
        runner=SimpleNamespace(requests={"r": SimpleNamespace(num_prompt_tokens=8)}),
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
    cpu.session = cpu.epoch = cpu.authoritative_length = 0
    cpu.committed_tokens = {}
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


def test_cpu_partial_forward_can_be_cancelled_and_resumed(tmp_path):
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
            worker.send(("pause", 2))
            messages = _wait_messages(
                worker, lambda values: any(value[0] == "paused" for value in values)
            )
            assert any(value[0] == "interrupted" for value in messages)
            assert not any(value[0] == "token" for value in messages)
            assert torch.equal(storage.kv[..., :5, :], prefix)

            worker.send(("resume", 1, 3, 5, 5, seed, 6))
            messages = _wait_messages(
                worker, lambda values: any(value[0] == "token" for value in values)
            )
            output = model(
                torch.tensor([[seed]]), past_key_values=cache, use_cache=True
            )
            token = next(value[4] for value in messages if value[0] == "token")
            assert token == int(output.logits[0, -1].argmax())
    finally:
        if worker is not None:
            worker.close()
        if storage is not None:
            storage.close()
        torch.set_num_threads(previous_threads)


def test_cpu_process_reuses_gpu_kv_delta_without_reforward(tmp_path):
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
            worker = RetroSpecCPUWorker(str(model_path), None, 2)
            worker.send(("initialize", 1, 1, storage.paths, shape, 5, seed, 9, ()))
            messages = _wait_messages(
                worker, lambda values: sum(value[0] == "token" for value in values) == 4
            )
            actual = [value[4] for value in messages if value[0] == "token"]
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
