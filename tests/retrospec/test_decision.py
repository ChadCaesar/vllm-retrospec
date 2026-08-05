# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.retrospec.decision import (
    RetroSpecDecision,
    RetroSpecDecisionPolicy,
    RetroSpecMetrics,
    RetroSpecReason,
)
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


def make_policy(**overrides: Any) -> RetroSpecDecisionPolicy:
    values: dict[str, Any] = {
        "method": "retrospec",
        "num_speculative_tokens": 8,
        "retrospec_min_draft_tokens": 2,
        "retrospec_max_draft_tokens": 4,
    }
    values.update(overrides)
    return RetroSpecDecisionPolicy(SpeculativeConfig(**values))


def evaluate(
    policy: RetroSpecDecisionPolicy,
    *,
    current_stage: RetroSpecStage = RetroSpecStage.DRAFT,
    request_stages: list[RetroSpecStage] | None = None,
    metrics: RetroSpecMetrics | None = None,
    draft_counts: list[int] | None = None,
    pending_counts: list[int] | None = None,
    **kwargs: torch.Tensor,
) -> RetroSpecDecision:
    if draft_counts is None:
        draft_counts = [2]
    if pending_counts is None:
        pending_counts = [0] * len(draft_counts)
    if request_stages is None:
        request_stages = [current_stage] * len(draft_counts)

    return policy.evaluate(
        current_stage=current_stage,
        request_stages=torch.tensor(request_stages, dtype=torch.int8),
        metrics=metrics or RetroSpecMetrics(),
        draft_counts=torch.tensor(draft_counts, dtype=torch.int32),
        pending_counts=torch.tensor(pending_counts, dtype=torch.int32),
        **kwargs,
    )


def assert_reason_set(value: torch.Tensor, reason: RetroSpecReason) -> None:
    assert int(value.item()) & int(reason)


def test_draft_without_threshold_or_limit_stays_in_draft():
    decision = evaluate(
        make_policy(),
        draft_counts=[1, 2],
    )

    assert decision.stop_draft.tolist() == [False, False]
    assert decision.require_expanded.tolist() == [False, False]
    assert decision.require_full.tolist() == [False, False]
    assert decision.reasons.tolist() == [0, 0]
    assert decision.next_stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.DRAFT),
    ]


def test_max_draft_tokens_transitions_to_sparse_verify():
    decision = evaluate(
        make_policy(),
        draft_counts=[3, 4, 5],
    )

    assert decision.stop_draft.tolist() == [False, True, True]
    assert decision.next_stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.SPARSE_VERIFY),
        int(RetroSpecStage.SPARSE_VERIFY),
    ]
    assert_reason_set(
        decision.reasons[1],
        RetroSpecReason.MAX_DRAFT_TOKENS,
    )


@pytest.mark.parametrize(
    ("config_field", "metric_field", "reason"),
    [
        (
            "retrospec_draft_margin_threshold",
            "draft_margin",
            RetroSpecReason.DRAFT_MARGIN,
        ),
        (
            "retrospec_hit_attn_threshold",
            "hit_attn",
            RetroSpecReason.HIT_ATTN,
        ),
    ],
)
def test_draft_thresholds_respect_minimum_length(
    config_field: str,
    metric_field: str,
    reason: RetroSpecReason,
):
    policy = make_policy(**{config_field: 0.5})
    metrics = RetroSpecMetrics(**{metric_field: torch.tensor([0.4, 0.4, 0.5])})

    decision = evaluate(
        policy,
        metrics=metrics,
        draft_counts=[1, 2, 2],
    )

    assert decision.stop_draft.tolist() == [False, True, False]
    assert_reason_set(decision.reasons[1], reason)
    assert not int(decision.reasons[0].item()) & int(reason)


@pytest.mark.parametrize(
    ("config_field", "metric_field", "reason"),
    [
        (
            "retrospec_sparse_margin_threshold",
            "sparse_margin",
            RetroSpecReason.SPARSE_MARGIN,
        ),
        (
            "retrospec_retrieval_attn_threshold",
            "retrieval_attn",
            RetroSpecReason.RETRIEVAL_ATTN,
        ),
    ],
)
def test_sparse_condition_transitions_to_expanded_verify(
    config_field: str,
    metric_field: str,
    reason: RetroSpecReason,
):
    policy = make_policy(**{config_field: 0.5})
    metrics = RetroSpecMetrics(**{metric_field: torch.tensor([0.4, 0.5])})

    decision = evaluate(
        policy,
        current_stage=RetroSpecStage.SPARSE_VERIFY,
        metrics=metrics,
        draft_counts=[2, 2],
    )

    assert decision.require_expanded.tolist() == [True, False]
    assert decision.next_stage.tolist() == [
        int(RetroSpecStage.EXPANDED_VERIFY),
        int(RetroSpecStage.DRAFT),
    ]
    assert_reason_set(decision.reasons[0], reason)


def test_sparse_token_change_transitions_to_expanded_verify():
    decision = evaluate(
        make_policy(),
        current_stage=RetroSpecStage.SPARSE_VERIFY,
        draft_counts=[2, 2],
        sparse_token_changed=torch.tensor([True, False]),
    )

    assert decision.require_expanded.tolist() == [True, False]
    assert_reason_set(
        decision.reasons[0],
        RetroSpecReason.SPARSE_TOKEN_CHANGED,
    )


@pytest.mark.parametrize(
    ("config_field", "metric_field", "reason"),
    [
        (
            "retrospec_expanded_margin_threshold",
            "expanded_margin",
            RetroSpecReason.EXPANDED_MARGIN,
        ),
        (
            "retrospec_expanded_attn_threshold",
            "expanded_attn",
            RetroSpecReason.EXPANDED_ATTN,
        ),
    ],
)
def test_expanded_condition_transitions_to_full_verify(
    config_field: str,
    metric_field: str,
    reason: RetroSpecReason,
):
    policy = make_policy(**{config_field: 0.5})
    metrics = RetroSpecMetrics(**{metric_field: torch.tensor([0.4, 0.5])})

    decision = evaluate(
        policy,
        current_stage=RetroSpecStage.EXPANDED_VERIFY,
        metrics=metrics,
        draft_counts=[2, 2],
    )

    assert decision.require_full.tolist() == [True, False]
    assert decision.next_stage.tolist() == [
        int(RetroSpecStage.FULL_VERIFY),
        int(RetroSpecStage.DRAFT),
    ]
    assert_reason_set(decision.reasons[0], reason)


def test_expanded_token_change_transitions_to_full_verify():
    decision = evaluate(
        make_policy(),
        current_stage=RetroSpecStage.EXPANDED_VERIFY,
        draft_counts=[2, 2],
        expanded_token_changed=torch.tensor([False, True]),
    )

    assert decision.require_full.tolist() == [False, True]
    assert_reason_set(
        decision.reasons[1],
        RetroSpecReason.EXPANDED_TOKEN_CHANGED,
    )


@pytest.mark.parametrize(
    "current_stage",
    [
        RetroSpecStage.SPARSE_VERIFY,
        RetroSpecStage.EXPANDED_VERIFY,
    ],
)
def test_pending_limit_transitions_to_full_from_verification_stage(
    current_stage: RetroSpecStage,
):
    decision = evaluate(
        make_policy(),
        current_stage=current_stage,
        draft_counts=[2, 2, 2],
        pending_counts=[7, 8, 9],
    )

    assert decision.require_full.tolist() == [False, True, True]
    assert_reason_set(decision.reasons[1], RetroSpecReason.PENDING_LIMIT)


@pytest.mark.parametrize(
    ("argument", "reason"),
    [
        ("generation_limit_reached", RetroSpecReason.GENERATION_LIMIT),
        ("index_update_required", RetroSpecReason.INDEX_UPDATE),
    ],
)
@pytest.mark.parametrize(
    "current_stage",
    [
        RetroSpecStage.SPARSE_VERIFY,
        RetroSpecStage.EXPANDED_VERIFY,
    ],
)
def test_external_condition_transitions_to_full_verify(
    current_stage: RetroSpecStage,
    argument: str,
    reason: RetroSpecReason,
):
    decision = evaluate(
        make_policy(),
        current_stage=current_stage,
        draft_counts=[2, 2],
        **{argument: torch.tensor([False, True])},
    )

    assert decision.require_full.tolist() == [False, True]
    assert_reason_set(decision.reasons[1], reason)


@pytest.mark.parametrize(
    ("argument", "value", "reason"),
    [
        ("pending_counts", [8], RetroSpecReason.PENDING_LIMIT),
        (
            "generation_limit_reached",
            torch.tensor([True]),
            RetroSpecReason.GENERATION_LIMIT,
        ),
        (
            "index_update_required",
            torch.tensor([True]),
            RetroSpecReason.INDEX_UPDATE,
        ),
    ],
)
def test_hard_trigger_stops_draft_before_full_verification(
    argument: str,
    value: Any,
    reason: RetroSpecReason,
):
    decision = evaluate(make_policy(), **{argument: value})

    assert decision.stop_draft.tolist() == [True]
    assert decision.require_full.tolist() == [False]
    assert decision.next_stage.tolist() == [int(RetroSpecStage.SPARSE_VERIFY)]
    assert_reason_set(decision.reasons[0], reason)


def test_full_verify_has_priority_over_expanded_verify():
    policy = make_policy(retrospec_sparse_margin_threshold=0.5)
    metrics = RetroSpecMetrics(sparse_margin=torch.tensor([0.4]))

    decision = evaluate(
        policy,
        current_stage=RetroSpecStage.SPARSE_VERIFY,
        metrics=metrics,
        pending_counts=[8],
    )

    assert decision.require_expanded.tolist() == [False]
    assert decision.require_full.tolist() == [True]
    assert decision.next_stage.tolist() == [int(RetroSpecStage.FULL_VERIFY)]
    assert_reason_set(decision.reasons[0], RetroSpecReason.SPARSE_MARGIN)
    assert_reason_set(decision.reasons[0], RetroSpecReason.PENDING_LIMIT)


def test_multiple_draft_reasons_are_preserved():
    policy = make_policy(
        retrospec_draft_margin_threshold=0.5,
        retrospec_hit_attn_threshold=0.5,
    )
    metrics = RetroSpecMetrics(
        draft_margin=torch.tensor([0.4]),
        hit_attn=torch.tensor([0.3]),
    )

    decision = evaluate(policy, metrics=metrics)

    assert decision.stop_draft.tolist() == [True]
    assert_reason_set(decision.reasons[0], RetroSpecReason.DRAFT_MARGIN)
    assert_reason_set(decision.reasons[0], RetroSpecReason.HIT_ATTN)


def test_only_requests_in_current_stage_are_evaluated():
    decision = evaluate(
        make_policy(),
        current_stage=RetroSpecStage.DRAFT,
        request_stages=[
            RetroSpecStage.DRAFT,
            RetroSpecStage.SPARSE_VERIFY,
        ],
        draft_counts=[4, 4],
    )

    assert decision.stop_draft.tolist() == [True, False]
    assert decision.next_stage.tolist() == [
        int(RetroSpecStage.SPARSE_VERIFY),
        int(RetroSpecStage.SPARSE_VERIFY),
    ]
    assert decision.reasons[1].item() == 0


def test_inactive_requests_transition_to_idle():
    decision = evaluate(
        make_policy(),
        draft_counts=[4, 4],
        active_mask=torch.tensor([True, False]),
    )

    assert decision.stop_draft.tolist() == [True, False]
    assert decision.next_stage.tolist() == [
        int(RetroSpecStage.SPARSE_VERIFY),
        int(RetroSpecStage.IDLE),
    ]
    assert decision.reasons[1].item() == 0


@pytest.mark.parametrize(
    "current_stage",
    [
        RetroSpecStage.IDLE,
        RetroSpecStage.FULL_VERIFY,
    ],
)
def test_terminal_stages_do_not_evaluate_metrics(
    current_stage: RetroSpecStage,
):
    policy = make_policy(retrospec_draft_margin_threshold=0.5)
    malformed_irrelevant_metrics = RetroSpecMetrics(draft_margin=torch.tensor([[0.1]]))

    decision = evaluate(
        policy,
        current_stage=current_stage,
        metrics=malformed_irrelevant_metrics,
    )

    assert decision.reasons.tolist() == [0]
    assert decision.next_stage.tolist() == [int(current_stage)]


@pytest.mark.parametrize(
    ("current_stage", "config_field", "metrics"),
    [
        (
            RetroSpecStage.DRAFT,
            "retrospec_sparse_margin_threshold",
            RetroSpecMetrics(sparse_margin=torch.tensor([[0.1]])),
        ),
        (
            RetroSpecStage.SPARSE_VERIFY,
            "retrospec_expanded_margin_threshold",
            RetroSpecMetrics(expanded_margin=torch.tensor([[0.1]])),
        ),
        (
            RetroSpecStage.EXPANDED_VERIFY,
            "retrospec_draft_margin_threshold",
            RetroSpecMetrics(draft_margin=torch.tensor([[0.1]])),
        ),
    ],
)
def test_irrelevant_stage_metrics_are_not_evaluated(
    current_stage: RetroSpecStage,
    config_field: str,
    metrics: RetroSpecMetrics,
):
    policy = make_policy(**{config_field: 0.5})

    decision = evaluate(
        policy,
        current_stage=current_stage,
        metrics=metrics,
    )

    assert decision.reasons.tolist() == [0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"draft_counts": torch.tensor([[1]], dtype=torch.int32)},
            "draft_counts must be one-dimensional",
        ),
        (
            {"draft_counts": torch.tensor([1.0])},
            "draft_counts must use an integer dtype",
        ),
        (
            {"request_stages": torch.tensor([1.0])},
            "request_stages must use an integer dtype",
        ),
        (
            {"active_mask": torch.tensor([1], dtype=torch.int32)},
            "torch.bool",
        ),
    ],
)
def test_invalid_common_inputs_raise(
    kwargs: dict[str, torch.Tensor],
    match: str,
):
    values: dict[str, Any] = {
        "current_stage": RetroSpecStage.DRAFT,
        "request_stages": torch.tensor(
            [int(RetroSpecStage.DRAFT)],
            dtype=torch.int8,
        ),
        "metrics": RetroSpecMetrics(),
        "draft_counts": torch.tensor([1], dtype=torch.int32),
        "pending_counts": torch.tensor([0], dtype=torch.int32),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        make_policy().evaluate(**values)


def test_current_stage_must_be_enum():
    with pytest.raises(ValueError, match="RetroSpecStage"):
        make_policy().evaluate(
            current_stage=1,  # type: ignore[arg-type]
            request_stages=torch.tensor([1], dtype=torch.int8),
            metrics=RetroSpecMetrics(),
            draft_counts=torch.tensor([1], dtype=torch.int32),
            pending_counts=torch.tensor([0], dtype=torch.int32),
        )


def test_relevant_metric_shape_must_match_batch():
    with pytest.raises(ValueError, match="draft_margin shape"):
        evaluate(
            make_policy(retrospec_draft_margin_threshold=0.5),
            metrics=RetroSpecMetrics(draft_margin=torch.tensor([0.1, 0.2])),
        )


@pytest.mark.parametrize(
    ("current_stage", "config_field", "metric_name"),
    [
        (
            RetroSpecStage.DRAFT,
            "retrospec_draft_margin_threshold",
            "draft_margin",
        ),
        (
            RetroSpecStage.DRAFT,
            "retrospec_hit_attn_threshold",
            "hit_attn",
        ),
        (
            RetroSpecStage.SPARSE_VERIFY,
            "retrospec_sparse_margin_threshold",
            "sparse_margin",
        ),
        (
            RetroSpecStage.SPARSE_VERIFY,
            "retrospec_retrieval_attn_threshold",
            "retrieval_attn",
        ),
        (
            RetroSpecStage.EXPANDED_VERIFY,
            "retrospec_expanded_margin_threshold",
            "expanded_margin",
        ),
        (
            RetroSpecStage.EXPANDED_VERIFY,
            "retrospec_expanded_attn_threshold",
            "expanded_attn",
        ),
    ],
)
def test_configured_stage_threshold_requires_its_metric(
    current_stage: RetroSpecStage,
    config_field: str,
    metric_name: str,
):
    with pytest.raises(ValueError, match=metric_name):
        evaluate(
            make_policy(**{config_field: 0.5}),
            current_stage=current_stage,
        )


@pytest.mark.parametrize(
    ("current_stage", "metric_field"),
    [
        (RetroSpecStage.DRAFT, "draft_margin"),
        (RetroSpecStage.DRAFT, "hit_attn"),
        (RetroSpecStage.SPARSE_VERIFY, "sparse_margin"),
        (RetroSpecStage.SPARSE_VERIFY, "retrieval_attn"),
        (RetroSpecStage.EXPANDED_VERIFY, "expanded_margin"),
        (RetroSpecStage.EXPANDED_VERIFY, "expanded_attn"),
    ],
)
def test_disabled_threshold_does_not_evaluate_its_metric(
    current_stage: RetroSpecStage,
    metric_field: str,
):
    malformed_metric = RetroSpecMetrics(**{metric_field: torch.tensor([[0.1]])})

    decision = evaluate(
        make_policy(),
        current_stage=current_stage,
        metrics=malformed_metric,
    )

    assert decision.reasons.tolist() == [0]


def test_pending_counts_are_validated_in_active_evaluation_stages():
    policy = make_policy()

    for current_stage in (
        RetroSpecStage.DRAFT,
        RetroSpecStage.SPARSE_VERIFY,
        RetroSpecStage.EXPANDED_VERIFY,
    ):
        with pytest.raises(ValueError, match="pending_counts"):
            policy.evaluate(
                current_stage=current_stage,
                request_stages=torch.tensor([int(current_stage)], dtype=torch.int8),
                metrics=RetroSpecMetrics(),
                draft_counts=torch.tensor([1], dtype=torch.int32),
                pending_counts=torch.tensor([[0]], dtype=torch.int32),
            )

    terminal_decision = policy.evaluate(
        current_stage=RetroSpecStage.FULL_VERIFY,
        request_stages=torch.tensor(
            [int(RetroSpecStage.FULL_VERIFY)], dtype=torch.int8
        ),
        metrics=RetroSpecMetrics(),
        draft_counts=torch.tensor([1], dtype=torch.int32),
        pending_counts=torch.tensor([[0]], dtype=torch.int32),
    )
    assert terminal_decision.next_stage.tolist() == [int(RetroSpecStage.FULL_VERIFY)]
