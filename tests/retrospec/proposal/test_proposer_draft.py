# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.proposer import (
    attention_stats,
    initialize_single_pipeline_stage,
    make_common_metadata,
    make_runner,
    make_sampling_metadata,
    make_vllm_config,
    mock_proposal_execution,
    run_proposal,
)
from tests.retrospec.support.proposer import (
    disable_pin_memory_for_cpu_tests as disable_pin_memory_for_cpu_tests,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.retrospec import (
    RetroSpecAttentionMode,
    RetroSpecProposer,
)
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


def test_propose_rejects_random_sampling():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )

    with pytest.raises(NotImplementedError, match="greedy decoding only"):
        run_proposal(
            proposer,
            torch.tensor([1], dtype=torch.int32),
            make_sampling_metadata(all_greedy=False),
            make_common_metadata([1]),
        )


def test_propose_keeps_partial_prefill_rows_idle(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_max_draft_tokens=1,
            retrospec_stats_interval_seconds=3600.0,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    observed_masks: list[list[bool]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        observed_masks.append(active_mask.tolist())
        return (
            torch.tensor([10, 11], dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([8, 4]),
        proposal_active_mask=torch.tensor([False, True]),
    )

    assert observed_masks == [[False, True]]
    assert result == [[], [11]]
    assert proposer.state.stage.tolist() == [
        int(RetroSpecStage.IDLE),
        int(RetroSpecStage.FULL_VERIFY),
    ]
    assert proposer.state.active_mask.tolist() == [False, True]
    stats = proposer.performance_stats
    proposal_requests_index = stats._gpu_counter_indices["proposal_requests"]
    assert stats._gpu_counters[proposal_requests_index].item() == 1
    assert stats._gpu_histograms["draft_to_sparse_tokens"].tolist() == [0, 1, 0, 0, 0]
    assert stats._cpu_counters["first_proposal_requests"] == 1
    assert stats._cpu_times["first_proposal_batch_wall"][1] == 1
    assert proposer._last_proposed_counts == {"request-1": 1}


def test_propose_stops_requests_independently_on_draft_margin(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_draft_margin_threshold=1.0),
        torch.device("cpu"),
        make_runner(),
    )
    expected_masks = [
        [True, True, True],
        [True, False, True],
        [False, False, True],
    ]
    margins = [
        [2.0, 0.5, 2.0],
        [0.5, 0.0, 2.0],
        [0.0, 0.0, 0.5],
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        assert batch_size == 3
        assert active_mask.tolist() == expected_masks[draft_index]
        token_base = 10 * (draft_index + 1)
        token_ids = torch.tensor(
            [token_base + i for i in range(batch_size)], dtype=torch.int32
        )
        draft_margin = torch.tensor(margins[draft_index])
        hit_attn = torch.ones(batch_size)
        return token_ids, draft_margin, hit_attn

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1, 1]),
    )

    assert result == [[10, 20], [11], [12, 22, 32]]


def test_propose_stops_at_configured_max_draft_tokens(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=3),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1]),
    )

    assert result == [[1, 2, 3], [1, 2, 3]]


def test_propose_respects_per_request_generation_limit(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(max_model_len=6),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        token_base = 10 * (draft_index + 1)
        return (
            torch.tensor(
                [token_base + i for i in range(batch_size)], dtype=torch.int32
            ),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([4, 3, 5]),
    )

    assert result == [[10], [11, 21], []]


def test_propose_respects_per_request_output_budget(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    observed_masks: list[list[bool]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        observed_masks.append(active_mask.tolist())
        return (
            torch.tensor([10, 11, 12], dtype=torch.int32) + draft_index * 10,
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1, 1]),
        remaining_generation_tokens=[1, 3, 0],
        valid_sampled_tokens_count=torch.tensor([1, 1, 0], dtype=torch.int32),
    )

    assert observed_masks == [[False, True, False], [False, True, False]]
    assert result == [[], [11, 21], []]


def test_propose_rolls_back_rejected_tokens_before_drafting(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(max_model_len=6),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.tensor([draft_index + 1], dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([5]),
        num_rejected_tokens_gpu=torch.tensor([2], dtype=torch.int32),
    )

    assert result == [[1, 2]]


def test_run_draft_step_preserves_attention_seq_lens_dtype(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
    )

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert metadata.seq_lens.dtype == common_attn_metadata.seq_lens.dtype
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            return torch.zeros((input_ids.shape[0], 4))

        def compute_logits(self, hidden_states):
            return torch.tensor([[0.0, 1.0]])

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = lambda **kwargs: SimpleNamespace(
        sampled_token_ids=torch.tensor([[1]], dtype=torch.int32)
    )
    proposer.sparse_attention.begin_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(1)
    )
    proposer.input_ids[0] = 1
    proposer.positions[0] = 3
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposal.model.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    initialize_single_pipeline_stage(proposer)
    proposer._run_draft_step(
        batch_size=1,
        draft_index=0,
        common_attn_metadata=common_attn_metadata,
        active_mask=torch.tensor([True]),
        sampling_metadata=make_sampling_metadata(all_greedy=True),
    )

    proposer.sparse_attention.begin_step.assert_called_once()
    proposer.sparse_attention.end_step_statistics.assert_called_once_with()


def test_model_step_sanitizes_input_ids_for_inactive_rows(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 4], dtype=torch.int64),
    )

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [7, 0]
            return torch.zeros((input_ids.shape[0], 4))

        def compute_logits(self, hidden_states):
            return torch.zeros((hidden_states.shape[0], 2))

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = lambda **kwargs: SimpleNamespace(
        sampled_token_ids=torch.ones(2, 1, dtype=torch.int32)
    )
    proposer.sparse_attention.begin_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(2)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposal.model.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    initialize_single_pipeline_stage(proposer)
    proposer._run_model_step(
        batch_size=2,
        step_index=1,
        input_ids=torch.tensor([7, -1], dtype=torch.int32),
        positions=torch.tensor([3, 3], dtype=torch.int64),
        active_mask=torch.tensor([True, False]),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=RetroSpecAttentionMode.SPARSE_VERIFY,
        compute_margin=False,
    )


def test_model_step_uses_unpadded_dynamic_layout_without_padding_policy_rows(
    monkeypatch,
):
    dispatcher = Mock(
        dispatch_piecewise_cudagraph=Mock(
            return_value=(CUDAGraphMode.PIECEWISE, BatchDescriptor(4))
        )
    )
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(cudagraph_dispatcher=dispatcher),
    )
    proposer._cudagraph_registration_failure = None
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 4], dtype=torch.int64),
    )

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert metadata.num_actual_tokens == 2
            assert metadata.slot_mapping.shape == (2,)
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [7, 0]
            assert positions.tolist() == [3, 0]
            return torch.zeros((2, 4))

        def compute_logits(self, hidden_states):
            assert hidden_states.shape == (2, 4)
            return torch.zeros((2, 2))

    forward_context_kwargs: list[dict[str, Any]] = []

    def fake_forward_context(*args, **kwargs):
        forward_context_kwargs.append(kwargs)
        return nullcontext()

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = lambda **kwargs: SimpleNamespace(
        sampled_token_ids=torch.ones(2, 1, dtype=torch.int32)
    )
    proposer.sparse_attention.begin_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(2)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposal.model.set_forward_context",
        fake_forward_context,
    )

    initialize_single_pipeline_stage(proposer)
    proposer._run_model_step(
        batch_size=2,
        step_index=1,
        input_ids=torch.tensor([7, -1], dtype=torch.int32),
        positions=torch.tensor([3, 3], dtype=torch.int64),
        active_mask=torch.tensor([True, False]),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=RetroSpecAttentionMode.DRAFT,
        compute_margin=False,
    )

    context = forward_context_kwargs[0]
    assert context["num_tokens"] == 2
    assert context["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    assert context["batch_descriptor"] is None
    slot_mapping = context["slot_mapping"]["model.layers.0.self_attn.attn"]
    assert slot_mapping.tolist() == [3, -1]
    active_mask = proposer.sparse_attention.begin_step.call_args.args[2]
    assert active_mask.tolist() == [True, False]


def test_propose_stops_requests_independently_on_hit_attention(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_hit_attn_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    expected_masks = [
        [True, True],
        [True, False],
    ]
    hit_attn_values = [
        [0.8, 0.4],
        [0.3, 1.0],
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        assert active_mask.tolist() == expected_masks[draft_index]
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.tensor(hit_attn_values[draft_index]),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1]),
    )

    assert result == [[1, 2], [1]]
