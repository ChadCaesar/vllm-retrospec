# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.proposer import (
    attention_stats,
    initialize_single_pipeline_stage,
    initialize_verification,
    make_common_metadata,
    make_parallel_verification_output,
    make_runner,
    make_sampling_metadata,
    make_vllm_config,
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
    transition_trace,
)
from vllm.v1.spec_decode.retrospec.proposer import (
    RetroSpecParallelVerificationOutput,
    RetroSpecVerificationResult,
)
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


def test_verify_unchanged_sparse_tokens_keeps_complete_prefix(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    observed_rows: list[tuple[list[int], list[int]]] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        assert attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
        observed_rows.append((list(request_indices), list(token_indices)))
        return make_parallel_verification_output(
            list(request_indices), list(token_indices), [10, 20, 30]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [3]
    assert not verification.require_full.any()
    assert proposer._draft_token_ids[0, :3].tolist() == [10, 20, 30]
    assert observed_rows == [([0, 0, 0], [0, 1, 2])]
    assert proposer.state.stage.tolist() == [int(RetroSpecStage.DRAFT)]


def test_sparse_token_change_is_corrected_and_truncates_prefix(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    observed_modes: list[RetroSpecAttentionMode] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        observed_modes.append(attention_mode)
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            token_ids = [11, 99, 98]
        else:
            token_ids = [11]
        return make_parallel_verification_output(
            list(request_indices), list(token_indices), token_ids
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert not verification.require_full.any()
    assert proposer._draft_token_ids[0, :3].tolist() == [11, 20, 30]
    assert observed_modes == [
        RetroSpecAttentionMode.SPARSE_VERIFY,
        RetroSpecAttentionMode.EXPANDED_VERIFY,
    ]


def test_expanded_verification_preserves_sparse_boundary_across_shared_workspace(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, -1, -1, -1]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        shared_token_ids = proposer.pipeline_protocol._model_token_ids[:1]
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            shared_token_ids.fill_(10)
            margin = [0.1]
        else:
            shared_token_ids.fill_(99)
            margin = None
        return RetroSpecParallelVerificationOutput(
            request_indices=request_indices,
            token_indices=token_indices,
            token_ids=shared_token_ids,
            margin=None if margin is None else torch.tensor(margin),
            attention_mass=torch.ones(1),
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )

    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert verification.require_full.tolist() == [True]
    assert proposer._draft_token_ids[0, 0].item() == 99


def test_expanded_verification_passes_or_stops_requests_independently(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_sparse_margin_threshold=0.5,
            retrospec_expanded_margin_threshold=0.5,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1], [11, 21, 31, -1]], dtype=torch.int32),
        torch.tensor([3, 3], dtype=torch.int32),
    )
    observed_rows: list[tuple[RetroSpecAttentionMode, list[int], list[int]]] = []
    compaction_lengths: list[int] = []
    compact_mask_indices = proposer._compact_mask_indices

    def track_compaction(mask, output):
        compaction_lengths.append(mask.shape[0])
        return compact_mask_indices(mask, output)

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        request_indices = list(request_indices)
        token_indices = list(token_indices)
        observed_rows.append((attention_mode, request_indices, token_indices))
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return make_parallel_verification_output(
                request_indices,
                token_indices,
                [10, 20, 30, 11, 21, 31],
                margin=[0.1] * 6,
            )

        return make_parallel_verification_output(
            request_indices, token_indices, [10, 11], margin=[0.9, 0.1]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    monkeypatch.setattr(proposer, "_compact_mask_indices", track_compaction)
    verification = proposer._verify_draft_tokens(
        2,
        ["request-0", "request-1"],
        1,
        torch.zeros(2, dtype=torch.int32),
        make_common_metadata([1, 1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1, 1]
    assert verification.require_full.tolist() == [False, True]
    assert observed_rows == [
        (RetroSpecAttentionMode.SPARSE_VERIFY, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]),
        (RetroSpecAttentionMode.EXPANDED_VERIFY, [0, 1], [0, 0]),
    ]
    assert compaction_lengths == [8, 2]
    assert proposer.state.stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.FULL_VERIFY),
    ]


def test_verification_trace_records_request_boundaries(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_sparse_margin_threshold=0.5,
            retrospec_expanded_margin_threshold=0.5,
            retrospec_trace_transitions=True,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1], [11, 21, -1, -1]], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
    )

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return make_parallel_verification_output(
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [10, 20, 11, 21],
                margin=[0.9, 0.9, 0.1, 0.9],
            )
        return make_parallel_verification_output([1], [0], [11], margin=[0.1])

    logger_info = Mock()
    monkeypatch.setattr(
        proposer, "_run_parallel_verification", fake_run_parallel_verification
    )
    monkeypatch.setattr(transition_trace.logger, "info", logger_info)

    verification = proposer._verify_draft_tokens(
        2,
        ["request-0", "request-1"],
        3,
        torch.zeros(2, dtype=torch.int32),
        make_common_metadata([1, 1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [2, 1]
    payloads = [json.loads(call.args[1]) for call in logger_info.call_args_list]
    assert [payload["phase"] for payload in payloads] == [
        "sparse_boundary",
        "sparse_complete",
        "expanded_boundary",
    ]
    assert all(payload["proposal_round"] == 3 for payload in payloads)
    assert payloads[0]["records"][0]["request_id"] == "request-1"
    assert payloads[0]["records"][0]["reason_names"] == ["SPARSE_MARGIN"]
    assert payloads[1]["records"][0]["request_id"] == "request-0"
    assert payloads[1]["records"][0]["next_stage_name"] == "DRAFT"
    assert payloads[2]["records"][0]["request_id"] == "request-1"
    assert payloads[2]["records"][0]["reason_names"] == ["EXPANDED_MARGIN"]


def test_expanded_token_change_replaces_current_token_before_truncation(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return make_parallel_verification_output(
                [0, 0], [0, 1], [10, 20], margin=[0.1, 0.9]
            )
        return make_parallel_verification_output([0], [0], [99])

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert verification.require_full.tolist() == [True]
    assert proposer._draft_token_ids[0, 0].item() == 99


def test_verify_only_processes_current_logical_draft_interval(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, 40]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        pending_counts=torch.tensor([2], dtype=torch.int32),
    )
    observed_rows: list[tuple[list[int], list[int]]] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        assert attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
        observed_rows.append((list(request_indices), list(token_indices)))
        return make_parallel_verification_output(
            list(request_indices), list(token_indices), [30, 40]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.tensor([2], dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed_rows == [([0, 0], [2, 3])]
    assert verification.verified_counts.tolist() == [2]
    assert verification.require_full.tolist() == [True]


def test_sparse_bonus_seeds_next_round_without_committing_early(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=2),
        torch.device("cpu"),
        make_runner(),
    )
    draft_calls: list[tuple[int, int, int]] = []
    verified_rounds: list[tuple[int, int, list[int]]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        draft_calls.append(
            (
                draft_index,
                int(proposer.input_ids[0]),
                int(proposer.positions[0]),
            )
        )
        return (
            torch.tensor([draft_index + 1], dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    def fake_verify(
        batch_size,
        request_ids,
        proposal_round,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        verified_rounds.append(
            (
                int(round_start_counts[0]),
                int(proposer.state.draft_counts[0]),
                proposer._draft_token_ids[0].tolist(),
            )
        )
        if proposal_round == 1:
            return RetroSpecVerificationResult(
                verified_counts=torch.tensor([2], dtype=torch.int32),
                require_full=torch.tensor([False]),
                bonus_mask=torch.tensor([True]),
                bonus_token_ids=torch.tensor([99], dtype=torch.int32),
            )
        return RetroSpecVerificationResult(
            verified_counts=torch.tensor([2], dtype=torch.int32),
            require_full=torch.tensor([True]),
        )

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2]),
    )

    assert draft_calls == [(0, 7, 2), (1, 1, 3), (3, 99, 5)]
    assert verified_rounds == [(0, 2, [1, 2, -1, -1]), (2, 2, [1, 2, 99, 4])]
    assert result == [[1, 2, 99, 4]]


def test_propose_accumulates_multiple_draft_rounds(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_max_draft_tokens=2,
            retrospec_stats_interval_seconds=3600.0,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    round_starts: list[list[int]] = []

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

    def fake_verify(
        batch_size,
        request_ids,
        proposal_round,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        assert request_ids == ["request-0"]
        assert proposal_round == len(round_starts) + 1
        round_starts.append(round_start_counts.tolist())
        verified_counts = proposer.state.draft_counts.clone()
        require_full = (
            round_start_counts + verified_counts >= proposer.policy.pending_limit
        )
        return RetroSpecVerificationResult(verified_counts, require_full)

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2]),
    )

    assert round_starts == [[0], [2]]
    assert result == [[1, 2, 3, 4]]
    assert proposer.state.pending_counts.tolist() == [4]
    assert proposer.state.stage.tolist() == [int(RetroSpecStage.FULL_VERIFY)]

    stats = proposer.performance_stats
    gpu_counters = {
        name: int(stats._gpu_counters[index].item())
        for name, index in stats._gpu_counter_indices.items()
    }
    assert stats._cpu_counters["proposal_calls"] == 1
    assert gpu_counters == {
        "proposal_requests": 1,
        "draft_round_requests": 2,
        "draft_tokens": 4,
        "sparse_bonus_admitted": 0,
        "feedback_horizon_reductions": 0,
        "feedback_horizon_restores": 0,
        "verified_tokens": 4,
        "proposed_tokens": 4,
        "resident_cluster_hits": 0,
        "resident_cluster_misses": 0,
        "draft_compact_resident_pages": 0,
        "draft_compact_selected_clusters": 0,
        "resident_bound_direct_hits": 0,
        "resident_hash_fallback_lookups": 0,
        "resident_hash_fallback_hits": 0,
        "resident_hash_fallback_misses": 0,
        "resident_hash_probe_steps": 0,
        "resident_hash_max_probe": 0,
        "resident_binding_invalidations": 0,
        "verification_lookup_clusters": 0,
        "verification_resident_hits": 0,
        "verification_resident_misses": 0,
    }
    assert stats._cpu_times["proposal_wall"][1] == 1


def test_propose_handles_different_round_offsets_in_one_buffer(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=2),
        torch.device("cpu"),
        make_runner(),
    )
    round_starts: list[list[int]] = []
    draft_calls: list[tuple[int, list[bool], list[int]]] = []
    verified_by_round = [
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        draft_calls.append(
            (
                draft_index,
                active_mask.tolist(),
                proposer.positions[:batch_size].tolist(),
            )
        )
        return (
            torch.tensor(
                [draft_index * 10 + row for row in range(batch_size)],
                dtype=torch.int32,
            ),
            None,
            torch.ones(batch_size),
        )

    def fake_verify(
        batch_size,
        request_ids,
        proposal_round,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        assert request_ids == ["request-0", "request-1"]
        assert proposal_round == len(round_starts) + 1
        round_index = len(round_starts)
        round_starts.append(round_start_counts.tolist())
        verified_counts = verified_by_round[round_index]
        require_full = (
            round_start_counts + verified_counts >= proposer.policy.pending_limit
        )
        return RetroSpecVerificationResult(verified_counts, require_full)

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7, 8], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2, 4]),
    )

    assert round_starts == [[0, 0], [1, 2], [3, 4]]
    assert [(index, mask) for index, mask, _ in draft_calls] == [
        (0, [True, True]),
        (1, [True, True]),
        (1, [True, False]),
        (2, [True, True]),
        (3, [False, True]),
        (3, [True, False]),
    ]
    assert draft_calls[2][2] == [3, 6]
    assert draft_calls[-1][2] == [5, 8]
    assert result == [[0, 10, 20, 30], [1, 11, 21, 31]]
    assert proposer.state.pending_counts.tolist() == [4, 4]


@pytest.mark.parametrize(
    ("attention_mode", "expect_margin"),
    [
        (RetroSpecAttentionMode.SPARSE_VERIFY, True),
        (RetroSpecAttentionMode.EXPANDED_VERIFY, False),
    ],
)
def test_parallel_verification_flattens_tokens_and_preserves_sampling_rows(
    monkeypatch,
    attention_mode,
    expect_margin,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=6,
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 13], dtype=torch.int64),
    )
    proposer.proposal_start_positions[:2].copy_(torch.tensor([3, 5]))
    proposer.proposal_input_ids[:2].copy_(torch.tensor([7, 8]))
    proposer._draft_token_ids[:2, :2].copy_(torch.tensor([[10, 20], [11, 21]]))

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert draft_index == 0
            assert metadata.num_reqs == 3
            assert metadata.query_start_loc.tolist() == [0, 1, 2, 3]
            assert metadata.seq_lens.tolist() == [6, 5, 7]
            assert metadata.block_table_tensor.tolist() == [[2, 3], [0, 1], [2, 3]]
            assert metadata.slot_mapping.tolist() == [13, 4, 14]
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [8, 10, 11]
            assert positions.tolist() == [5, 4, 6]
            return torch.zeros((3, 4))

        def compute_logits(self, hidden_states):
            return torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0], [0.0, 1.0, 4.0]])

    sampling_calls: list[torch.Tensor] = []

    def sample(*, logits, sampling_metadata):
        sampling_calls.append(logits.clone())
        return SimpleNamespace(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = sample
    proposer.sparse_attention.begin_parallel_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(3)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposal.verification.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    initialize_single_pipeline_stage(proposer)
    result = proposer._run_parallel_verification(
        batch_size=2,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=attention_mode,
    )

    assert result.request_indices.tolist() == [1, 0, 1]
    assert result.token_indices.tolist() == [0, 1, 1]
    assert result.token_ids.tolist() == [1, 0, 2]
    assert (
        result.token_ids.data_ptr()
        == proposer.pipeline_protocol._model_token_ids.data_ptr()
    )
    assert (result.margin is not None) is expect_margin
    if result.margin is not None:
        assert result.margin.tolist() == [1.0, 2.0, 3.0]
    assert len(sampling_calls) == 0
    proposer.sparse_attention.begin_parallel_step.assert_called_once()
    begin_args = proposer.sparse_attention.begin_parallel_step.call_args.args
    assert begin_args[0] == attention_mode
    assert torch.equal(begin_args[1], torch.tensor([1, 0, 1], dtype=torch.int64))
    assert torch.equal(begin_args[2], torch.tensor([0, 1, 1], dtype=torch.int64))
    proposer.sparse_attention.end_step_statistics.assert_called_once_with()


def test_parallel_verification_uses_unpadded_dynamic_layout(monkeypatch):
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
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=6,
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 13], dtype=torch.int64),
    )
    proposer.proposal_start_positions[:2].copy_(torch.tensor([3, 5]))
    proposer.proposal_input_ids[:2].copy_(torch.tensor([7, 8]))
    proposer._draft_token_ids[:2, :2].copy_(torch.tensor([[10, 20], [11, 21]]))

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert metadata.num_actual_tokens == 3
            assert metadata.slot_mapping.tolist() == [13, 4, 14]
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [8, 10, 11]
            assert positions.tolist() == [5, 4, 6]
            return torch.zeros((3, 4))

        def compute_logits(self, hidden_states):
            assert hidden_states.shape == (3, 4)
            return torch.tensor([[0.0, 2.0], [3.0, 1.0], [0.0, 4.0]])

    forward_context_kwargs: list[dict[str, Any]] = []

    def fake_forward_context(*args, **kwargs):
        forward_context_kwargs.append(kwargs)
        return nullcontext()

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.sparse_attention.begin_parallel_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(3)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposal.verification.set_forward_context",
        fake_forward_context,
    )

    initialize_single_pipeline_stage(proposer)
    result = proposer._run_parallel_verification(
        batch_size=2,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=RetroSpecAttentionMode.EXPANDED_VERIFY,
    )

    assert result.request_indices.tolist() == [1, 0, 1]
    assert result.token_indices.tolist() == [0, 1, 1]
    assert result.token_ids.tolist() == [1, 0, 1]
    context = forward_context_kwargs[0]
    assert context["num_tokens"] == 3
    assert context["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    assert context["batch_descriptor"] is None
    slot_mapping = context["slot_mapping"]["model.layers.0.self_attn.attn"]
    assert slot_mapping.tolist() == [13, 4, 14]
    begin_args = proposer.sparse_attention.begin_parallel_step.call_args.args
    assert torch.equal(begin_args[1], torch.tensor([1, 0, 1], dtype=torch.int64))
    assert torch.equal(begin_args[2], torch.tensor([0, 1, 1], dtype=torch.int64))
