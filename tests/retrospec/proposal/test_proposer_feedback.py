# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    mock_proposal_execution,
    run_proposal,
)
from vllm.v1.spec_decode.retrospec import (
    RetroSpecAttentionMode,
    RetroSpecProposer,
)


def test_propose_stops_draft_at_index_update_boundary(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=4),
        torch.device("cpu"),
        make_runner(),
    )
    observed_indices: list[int] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        observed_indices.append(draft_index)
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1]),
        committed_positions=[1],
    )

    assert observed_indices == [0, 1, 2, 3]
    assert result == [[1, 2, 3, 4]]


def test_sparse_verification_requires_full_at_index_update_boundary(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=4),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, 40]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
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
            list(request_indices), list(token_indices), [10, 20, 30, 40]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    flush_prefetch = Mock()
    prime_full_verification = Mock(return_value=False)
    monkeypatch.setattr(
        proposer.sparse_attention,
        "flush_sparse_verification_prefetch",
        flush_prefetch,
    )
    monkeypatch.setattr(
        proposer.sparse_attention,
        "maybe_prime_full_verification",
        prime_full_verification,
    )

    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed_rows == [([0, 0, 0, 0], [0, 1, 2, 3])]
    flush_prefetch.assert_called_once_with()
    prime_full_verification.assert_called_once_with(4)
    assert verification.verified_counts.tolist() == [4]
    assert verification.require_full.tolist() == [True]


def test_sparse_full_trigger_skips_expanded_verification(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.max_model_len = 2
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
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
        return make_parallel_verification_output(
            list(request_indices),
            list(token_indices),
            [10, 20],
            margin=[0.1, 0.1],
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

    assert observed_modes == [RetroSpecAttentionMode.SPARSE_VERIFY]
    assert verification.verified_counts.tolist() == [1]
    assert verification.require_full.tolist() == [True]


def test_remove_requests_resets_proposer_index_update_boundary():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=4),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.index_update_state.begin_batch(["request"], [10])

    proposer.remove_requests({"request"})
    proposer.index_update_state.begin_batch(["request"], [100])

    assert proposer.index_update_state.next_update_positions.tolist() == [104]


def test_finished_request_records_terminal_proposal_outcome_and_flushes():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._seen_proposal_request_ids.update(("finished", "running"))
    proposer._last_proposed_counts.update({"finished": 4, "running": 3})
    proposer.performance_stats.flush = Mock()

    proposer.record_previous_proposal_outcomes(
        ("finished", "running"),
        (3, 4),
        {"finished"},
    )

    counters = proposer.performance_stats._cpu_counters
    assert counters["terminal_proposal_tokens"] == 4
    assert counters["terminal_committed_proposal_tokens"] == 2
    assert counters["terminal_wasted_proposal_tokens"] == 2
    assert proposer._last_proposed_counts == {"running": 3}
    assert proposer._last_committed_proposal_counts == {"running": 3}
    assert proposer._seen_proposal_request_ids == {"running"}
    proposer.performance_stats.flush.assert_called_once_with("request_finished")


def test_preempted_request_preserves_first_proposal_tracking():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._seen_proposal_request_ids.add("preempted")
    proposer._last_proposed_counts["preempted"] = 4
    proposer.performance_stats.flush = Mock()

    proposer.record_previous_proposal_outcomes(("preempted",), (2,), set())

    assert proposer._last_proposed_counts == {"preempted": 4}
    assert proposer._seen_proposal_request_ids == {"preempted"}
    proposer.performance_stats.flush.assert_not_called()


def test_finished_request_without_sample_count_clears_request_id_lifecycle():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._seen_proposal_request_ids.add("reused")
    proposer._last_proposed_counts["reused"] = 4
    proposer.performance_stats.flush = Mock()

    proposer.record_previous_proposal_outcomes((), (), {"reused"})

    assert "reused" not in proposer._last_proposed_counts
    assert "reused" not in proposer._seen_proposal_request_ids
    proposer.performance_stats.flush.assert_called_once_with("request_finished")


def test_sync_bookkeeping_records_committed_proposal_before_finish():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._last_proposed_counts.update({"first": 4, "second": 3})

    proposer.record_sampled_proposal_outcomes(("first", "second"), (3, 1))

    assert proposer._last_committed_proposal_counts == {"first": 2, "second": 0}


def test_verified_proposal_outcomes_use_consumed_lengths_not_next_proposal():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._last_proposed_counts.update({"first": 64, "second": 64})

    proposer.record_verified_proposal_outcomes((4, 8, 0), (3, 9, 1))

    counters = proposer.performance_stats._cpu_counters
    assert counters["proposal_verified_rounds"] == 2
    assert counters["proposal_verified_tokens"] == 12
    assert counters["proposal_accepted_tokens"] == 10
    assert counters["proposal_rejected_tokens"] == 2
    assert counters["proposal_fully_accepted"] == 1


def test_verified_proposal_outcomes_reject_mismatched_rows():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )

    with pytest.raises(ValueError, match="equal length"):
        proposer.record_verified_proposal_outcomes((4,), (3, 2))

    with pytest.raises(ValueError, match="equal length"):
        proposer.record_verified_proposal_outcomes((4,), (3,), ("first", "second"))


def test_full_verify_feedback_limits_only_low_acceptance_request():
    proposer = RetroSpecProposer(
        make_vllm_config(
            max_model_len=128,
            num_speculative_tokens=64,
            retrospec_max_draft_tokens=8,
            retrospec_stats_interval_seconds=0.0,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._feedback_horizon_enabled = True
    proposer.record_verified_proposal_outcomes((64, 64), (15, 33), ("low", "high"))

    budgets = proposer._prepare_proposal_token_budgets(
        [100, 100], torch.ones(2, dtype=torch.int32), ("low", "high")
    )

    assert budgets.tolist() == [16, 64]
    assert proposer._feedback_horizons == {"low": 16}
    proposer.remove_requests({"low"})
    assert "low" not in proposer._feedback_horizons


def test_full_verify_feedback_recovers_after_three_fully_accepted_rounds():
    proposer = RetroSpecProposer(
        make_vllm_config(
            max_model_len=128,
            num_speculative_tokens=64,
            retrospec_max_draft_tokens=8,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._feedback_horizon_enabled = True
    proposer.record_verified_proposal_outcomes((64,), (15,), ("request",))
    proposer.record_verified_proposal_outcomes((16,), (17,), ("request",))
    proposer.record_verified_proposal_outcomes((16,), (9,), ("request",))
    assert proposer._feedback_recovery["request"] == 0

    for _ in range(3):
        proposer.record_verified_proposal_outcomes((16,), (17,), ("request",))

    assert "request" not in proposer._feedback_horizons
    budgets = proposer._prepare_proposal_token_budgets(
        [100], torch.ones(1, dtype=torch.int32), ("request",)
    )
    assert budgets.tolist() == [64]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "tp_size,pp_size,host_enabled", [(1, 1, True), (2, 1, False), (1, 2, False)]
)
def test_full_verify_feedback_uses_device_state_for_multiple_ranks(
    tp_size, pp_size, host_enabled
):
    config = make_vllm_config(
        max_model_len=128,
        num_speculative_tokens=64,
        retrospec_max_draft_tokens=8,
    )
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = pp_size
    proposer = RetroSpecProposer(config, torch.device("cuda"), make_runner())

    assert proposer._feedback_horizon_enabled is host_enabled
    assert proposer._feedback_device_enabled is not host_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_device_feedback_tracks_reordered_requests_and_clears_reused_slot():
    config = make_vllm_config(
        max_model_len=128, num_speculative_tokens=64, retrospec_max_draft_tokens=8
    )
    config.parallel_config.tensor_parallel_size = 2
    device = torch.device("cuda", torch.cuda.current_device())
    proposer = RetroSpecProposer(config, device, make_runner())
    sampled = torch.ones(2, dtype=torch.int32, device=device)

    initial = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("low", "high")
    )
    assert initial.tolist() == [64, 64]
    proposer._update_device_feedback_horizon(
        ("low", "high"),
        (64, 64),
        torch.tensor([15, 33], dtype=torch.int32, device=device),
    )
    reordered = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("high", "low")
    )
    assert reordered.tolist() == [64, 16]

    for _ in range(3):
        proposer._update_device_feedback_horizon(
            ("high", "low"),
            (0, 16),
            torch.tensor([1, 17], dtype=torch.int32, device=device),
        )
    recovered = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("high", "low")
    )
    assert recovered.tolist() == [64, 64]

    proposer._update_device_feedback_horizon(
        ("high", "low"),
        (0, 64),
        torch.tensor([1, 15], dtype=torch.int32, device=device),
    )
    proposer.remove_requests({"low"})
    reused = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("high", "new")
    )
    assert reused.tolist() == [64, 64]
