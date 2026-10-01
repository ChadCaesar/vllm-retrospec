# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.proposer import (
    disable_pin_memory_for_cpu_tests as disable_pin_memory_for_cpu_tests,
)
from tests.retrospec.support.proposer import (
    initialize_verification,
    make_common_metadata,
    make_parallel_verification_output,
    make_runner,
    make_sampling_metadata,
    make_vllm_config,
)
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.retrospec import (
    RetroSpecAttentionMode,
    RetroSpecProposer,
)
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


def test_sparse_bonus_pairs_append_only_with_capacity_and_budget():
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    proposer._sparse_bonus_enabled = True
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1], [11, 21, 31, -1]], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32),
        pending_counts=torch.tensor([0, 2], dtype=torch.int32),
    )
    starts = proposer.state.pending_counts.clone()
    active = proposer.state.active_mask.clone()
    requests, steps = proposer._build_verification_pairs(
        2, starts, proposer.state.draft_counts, active
    )
    combined_requests, combined_steps, bonus_requests = (
        proposer._append_sparse_bonus_pairs(
            2,
            requests,
            steps,
            starts,
            proposer.state.draft_counts,
            active,
            make_sampling_metadata(all_greedy=True),
        )
    )

    assert combined_requests.tolist() == [0, 0, 1, 0]
    assert combined_steps.tolist() == [0, 1, 2, 2]
    assert bonus_requests.tolist() == [0]

    proposer._sparse_bonus_enabled = False
    ordinary_requests, ordinary_steps, no_bonus = proposer._append_sparse_bonus_pairs(
        2,
        requests,
        steps,
        starts,
        proposer.state.draft_counts,
        active,
        make_sampling_metadata(all_greedy=True),
    )
    assert ordinary_requests.shape[0] == 3
    assert ordinary_steps.shape[0] == 3
    assert no_bonus.numel() == 0

    proposer._sparse_bonus_enabled = True
    proposer.max_parallel_tokens = 3
    _, _, no_capacity = proposer._append_sparse_bonus_pairs(
        2,
        requests,
        steps,
        starts,
        proposer.state.draft_counts,
        active,
        make_sampling_metadata(all_greedy=True),
    )
    assert no_capacity.numel() == 0
    proposer.max_parallel_tokens = (
        proposer.max_batch_size * proposer.num_speculative_tokens
    )

    _, _, processed_sampling = proposer._append_sparse_bonus_pairs(
        2,
        requests,
        steps,
        starts,
        proposer.state.draft_counts,
        active,
        replace(make_sampling_metadata(all_greedy=True), no_penalties=False),
    )
    assert processed_sampling.numel() == 0


def test_sparse_bonus_is_discarded_for_request_with_verification_boundary(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    proposer._sparse_bonus_enabled = True
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1], [11, 21, -1, -1]], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
    )
    observed: list[tuple[list[int], list[int], int | None]] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
        bonus_start_index=None,
    ):
        if attention_mode == RetroSpecAttentionMode.EXPANDED_VERIFY:
            return make_parallel_verification_output([1], [1], [99])
        observed.append(
            (request_indices.tolist(), token_indices.tolist(), bonus_start_index)
        )
        return make_parallel_verification_output(
            request_indices.tolist(),
            token_indices.tolist(),
            [10, 20, 11, 99, 30, 31],
        )

    monkeypatch.setattr(
        proposer, "_run_parallel_verification", fake_run_parallel_verification
    )
    verification = proposer._verify_draft_tokens(
        2,
        ["request-0", "request-1"],
        1,
        torch.zeros(2, dtype=torch.int32),
        make_common_metadata([1, 1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed == [([0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 2], 4)]
    assert verification.verified_counts.tolist() == [2, 2]
    assert verification.bonus_mask is not None
    assert verification.bonus_mask.tolist() == [True, False]
    assert verification.bonus_token_ids is not None
    assert verification.bonus_token_ids.tolist() == [30, 31]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_proposal_token_budget_subtracts_target_output_and_clamps(device):
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())

    budgets = proposer._prepare_proposal_token_budgets(
        [0, 2, 9],
        torch.tensor([1, 1, 2], dtype=torch.int32, device=device),
    )

    assert budgets.data_ptr() == proposer._proposal_token_budgets.gpu.data_ptr()
    assert budgets.tolist() == [0, 1, 4]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_verification_pair_compaction_stays_on_device(device):
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    round_starts = torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=device)
    draft_counts = torch.tensor([3, 2, 1, 4], dtype=torch.int32, device=device)
    active = torch.tensor([True, False, True, True], device=device)

    request_indices, token_indices = proposer._build_verification_pairs(
        4, round_starts, draft_counts, active
    )

    assert request_indices.device.type == device.type
    assert token_indices.device.type == device.type
    assert (
        request_indices.data_ptr() == proposer._verification_request_indices.data_ptr()
    )
    assert token_indices.data_ptr() == proposer._verification_token_indices.data_ptr()
    assert request_indices.tolist() == [0, 0, 0, 2, 3, 3, 3, 3]
    assert token_indices.tolist() == [0, 1, 2, 2, 0, 1, 2, 3]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_verification_compaction_reuses_fixed_output(device):
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    output = torch.empty(8, dtype=torch.int64, device=device)

    selected = proposer._compact_mask_indices(
        torch.tensor(
            [True, False, True, True, False, False, True, False],
            device=device,
        ),
        output,
    )

    assert selected.device.type == device.type
    assert selected.data_ptr() == output.data_ptr()
    assert selected.tolist() == [0, 2, 3, 6]

    empty = proposer._compact_mask_indices(
        torch.zeros(8, dtype=torch.bool, device=device),
        output,
    )
    assert empty.numel() == 0


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_dynamic_draft_control_skips_empty_columns_and_reuses_workspace(device):
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    proposer.state.begin_batch(
        4,
        torch.tensor([True, True, True, False], device=device),
    )
    proposer.state.set_pending_counts(
        torch.tensor([2, 0, 1, 0], dtype=torch.int32, device=device)
    )
    proposer.state.add_draft_counts(
        torch.tensor([4, 4, 4, 0], dtype=torch.int32, device=device)
    )
    proposer.positions[:4].copy_(
        torch.tensor([1, 1, 15, 1], dtype=torch.int64, device=device)
    )

    round_mask, round_starts = proposer._begin_draft_round(4)

    assert round_mask.data_ptr() == proposer._draft_round_mask.data_ptr()
    assert round_starts.data_ptr() == proposer._draft_round_start_counts.data_ptr()
    assert round_mask.tolist() == [True, True, False, False]
    assert round_starts.tolist() == [2, 0, 1, 0]
    assert proposer.state.draft_counts.tolist() == [0, 0, 0, 0]

    token_index, step_mask = proposer._prepare_next_draft_step(
        4, round_mask, round_starts
    )
    assert token_index == 0
    assert step_mask.data_ptr() == proposer._draft_stage_mask.data_ptr()
    assert step_mask.tolist() == [False, True, False, False]

    proposer.state.add_draft_counts(step_mask.to(torch.int32))
    proposer.state.set_stage(step_mask, RetroSpecStage.FULL_VERIFY)
    token_index, step_mask = proposer._prepare_next_draft_step(
        4, round_mask, round_starts
    )
    assert token_index == 2
    assert step_mask.tolist() == [True, False, False, False]

    proposer.state.set_stage(step_mask, RetroSpecStage.FULL_VERIFY)
    token_index, step_mask = proposer._prepare_next_draft_step(
        4, round_mask, round_starts
    )
    assert token_index == proposer.num_speculative_tokens
    assert not step_mask.any()


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_parallel_sampling_uses_one_raw_argmax_for_plain_greedy(device):
    sampler = Mock()
    proposer = RetroSpecProposer(
        make_vllm_config(), device, make_runner(sampler=sampler)
    )
    logits = torch.tensor(
        [[4.0, 4.0, 1.0], [3.0, 2.0, 1.0], [0.0, 1.0, 5.0]], device=device
    )
    metadata = make_sampling_metadata(all_greedy=True)
    reference = Sampler()(
        logits=logits.clone(), sampling_metadata=metadata
    ).sampled_token_ids.view(-1)

    sampled = proposer._sample_parallel_logits(
        batch_size=2,
        logits=logits,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64, device=device),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64, device=device),
        sampling_metadata=metadata,
        output=proposer._sparse_sampled_token_ids,
    )

    torch.testing.assert_close(sampled, reference)
    assert sampled.data_ptr() == proposer._sparse_sampled_token_ids.data_ptr()
    assert proposer._verification_step_logits is None
    sampler.assert_not_called()


def test_parallel_sampling_builds_pair_metadata_once_for_indexable_constraints(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm.v1.sample.ops.penalties.is_pin_memory_available", lambda: False
    )
    captured_metadata: list[SamplingMetadata] = []
    sampler = Sampler()
    sampler.pin_memory = False

    def sample(*, logits, sampling_metadata):
        captured_metadata.append(sampling_metadata)
        return sampler(logits=logits, sampling_metadata=sampling_metadata)

    proposer = RetroSpecProposer(
        make_vllm_config(), torch.device("cpu"), make_runner(sampler=sample)
    )
    allowed_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [True, False, False, False],
        ]
    )
    metadata = replace(
        make_sampling_metadata(all_greedy=True),
        no_penalties=False,
        prompt_token_ids=torch.tensor([[1, 2], [3, 0], [1, 3]]),
        frequency_penalties=torch.tensor([0.1, 0.2, 0.3]),
        presence_penalties=torch.tensor([0.4, 0.5, 0.6]),
        repetition_penalties=torch.tensor([1.1, 1.2, 1.3]),
        output_token_ids=[[1], [2], [3]],
        allowed_token_ids_mask=allowed_mask,
        bad_words_token_ids={0: [[1]], 2: [[2]]},
    )
    request_indices = torch.tensor([2, 0, 2], dtype=torch.int64)
    token_indices = torch.tensor([0, 1, 1], dtype=torch.int64)
    logits = torch.tensor(
        [[0.0, 1.0, 3.0, 2.0], [4.0, 1.0, 0.0, 2.0], [0.0, 5.0, 1.0, 2.0]]
    )
    reference_sampler = Sampler()
    reference_sampler.pin_memory = False
    reference_proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(sampler=reference_sampler),
    )
    reference_output = reference_proposer._sparse_sampled_token_ids[: logits.shape[0]]
    sampler_calls = reference_proposer._sample_parallel_logits_by_position(
        batch_size=3,
        logits=logits.clone(),
        request_indices=request_indices,
        token_indices=token_indices,
        sampling_metadata=metadata,
        sampled_token_ids=reference_output,
    )

    sampled = proposer._sample_parallel_logits(
        batch_size=3,
        logits=logits.clone(),
        request_indices=request_indices,
        token_indices=token_indices,
        sampling_metadata=metadata,
        output=proposer._sparse_sampled_token_ids,
    )

    assert sampler_calls == 2
    torch.testing.assert_close(sampled, reference_output)
    assert len(captured_metadata) == 1
    pair_metadata = captured_metadata[0]
    assert pair_metadata.max_num_logprobs is None
    assert pair_metadata.output_token_ids == [[3], [1], [3]]
    assert pair_metadata.bad_words_token_ids == {
        0: [[2]],
        1: [[1]],
        2: [[2]],
    }
    assert pair_metadata.prompt_token_ids.tolist() == [[1, 3], [1, 2], [1, 3]]
    assert pair_metadata.frequency_penalties.tolist() == pytest.approx([0.3, 0.1, 0.3])
    assert pair_metadata.presence_penalties.tolist() == pytest.approx([0.6, 0.4, 0.6])
    assert pair_metadata.repetition_penalties.tolist() == pytest.approx([1.3, 1.1, 1.3])
    assert torch.equal(pair_metadata.allowed_token_ids_mask, allowed_mask[[2, 0, 2]])


def test_parallel_sampling_retains_position_fallback_for_indexed_processors():
    sampling_calls: list[torch.Tensor] = []

    def sample(*, logits, sampling_metadata):
        sampling_calls.append(logits.clone())
        return SimpleNamespace(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    processors = LogitsProcessors()
    processors.non_argmax_invariant.append(Mock())
    metadata = replace(make_sampling_metadata(all_greedy=True), logitsprocs=processors)
    proposer = RetroSpecProposer(
        make_vllm_config(), torch.device("cpu"), make_runner(sampler=sample)
    )
    logits = torch.tensor([[0.0, 4.0, 1.0], [3.0, 2.0, 1.0], [0.0, 1.0, 5.0]])

    sampled = proposer._sample_parallel_logits(
        batch_size=2,
        logits=logits,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        sampling_metadata=metadata,
        output=proposer._sparse_sampled_token_ids,
    )

    assert sampled.tolist() == [1, 0, 2]
    assert len(sampling_calls) == 2
    assert all(call.shape == (2, 3) for call in sampling_calls)
    assert proposer._verification_step_logits is not None


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_first_verification_boundary_reduces_by_request(device):
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    request_indices = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64, device=device)
    boundary_mask = torch.tensor(
        [False, True, False, False, True, False], device=device
    )

    first = proposer._find_first_boundary_indices(4, request_indices, boundary_mask)

    assert first.device.type == device.type
    assert first.data_ptr() == proposer._verification_first_boundaries.data_ptr()
    assert first.tolist() == [1, 4, 6, 6]
