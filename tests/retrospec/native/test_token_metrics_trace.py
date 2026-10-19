# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.proposer import (
    disable_pin_memory_for_cpu_tests as disable_pin_memory_for_cpu_tests,
)
from tests.retrospec.support.proposer import (
    make_common_metadata,
    make_runner,
    make_sampling_metadata,
    make_vllm_config,
)
from vllm.v1.spec_decode.retrospec.proposer import RetroSpecProposer
from vllm.v1.spec_decode.retrospec.token_metrics_trace import (
    RetroSpecTokenMetricsTracer,
)


def traced_proposal(tracer, tokens=(10, 20, 30, 40), start=100):
    tracer.begin(["request"], torch.tensor([start]))
    for index, token_id in enumerate(tokens):
        tracer.record_draft(
            ["request"],
            index,
            torch.tensor([True]),
            torch.tensor([token_id]),
            torch.tensor([index + 0.1]),
            torch.tensor([0.5 + 0.01 * index]),
        )


def test_first_mismatch_uses_consumed_proposal_and_excludes_suffix():
    tracer = RetroSpecTokenMetricsTracer(True)
    events = []
    tracer._emit = events.append
    traced_proposal(tracer)
    tracer.submit(["request"], [[10, 20, 30, 40]])
    traced_proposal(tracer, tokens=(50, 60), start=200)
    tracer.submit(["request"], [[50, 60]])

    tracer.record_outcomes(["request"], [4], [2], [[10, 99]])
    record = events[0]["first_mismatch"]
    assert events[0]["proposal_id"] == 0
    assert record["token_index"] == 1
    assert record["position"] == 102
    assert record["token_id"] == 20
    assert record["target_token_id"] == 99
    assert record["draft_margin"] == pytest.approx(1.1)
    assert record["hit_attn"] == pytest.approx(0.51)
    assert [r["token_id"] for r in events[0]["accepted_tokens"]] == [10]
    assert events[0]["accepted_tokens"][0]["draft_margin"] == pytest.approx(0.1)

    tracer.record_outcomes(["request"], [2], [1], [[77]])
    assert events[1]["proposal_id"] == 1
    assert events[1]["first_mismatch"]["token_id"] == 50
    assert events[1]["accepted_tokens"] == []


@pytest.mark.parametrize("origin", ["sparse_correction", "expanded_correction"])
@pytest.mark.parametrize("changed", [False, True])
def test_corrections_do_not_reuse_metrics_of_a_different_token(origin, changed):
    tracer = RetroSpecTokenMetricsTracer(True)
    events = []
    tracer._emit = events.append
    traced_proposal(tracer, tokens=(10, 20))
    replacement = 99 if changed else 20
    tracer.record_replacements(
        ["request"],
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([20]),
        torch.tensor([replacement]),
        origin,
    )
    tracer.submit(["request"], [[10, replacement]])
    tracer.record_outcomes(["request"], [2], [2], [[10, 98]])
    record = events[0]["first_mismatch"]
    assert record["token_id"] == replacement
    if changed:
        assert record["origin"] == origin
        assert record["draft_margin"] is None
        assert record["hit_attn"] is None
    else:
        assert record["origin"] == "draft"
        assert record["draft_margin"] == pytest.approx(1.1)


def test_bonus_has_no_draft_metrics():
    tracer = RetroSpecTokenMetricsTracer(True)
    events = []
    tracer._emit = events.append
    traced_proposal(tracer, tokens=(10, 20))
    tracer.record_bonus(
        ["request"], torch.tensor([True]), torch.tensor([1]), torch.tensor([99])
    )
    tracer.submit(["request"], [[10, 99]])
    tracer.record_outcomes(["request"], [2], [2], [[10, 98]])
    record = events[0]["first_mismatch"]
    assert record["origin"] == "sparse_bonus"
    assert record["draft_margin"] is None
    assert record["hit_attn"] is None


def test_scheduled_prefix_all_accepted_has_no_first_mismatch():
    tracer = RetroSpecTokenMetricsTracer(True)
    events = []
    tracer._emit = events.append
    traced_proposal(tracer)
    tracer.submit(["request"], [[10, 20, 30, 40]])
    tracer.record_outcomes(["request"], [2], [3], [[10, 20, 99]])
    assert events[0]["submitted_count"] == 4
    assert events[0]["proposed_count"] == 2
    assert events[0]["first_mismatch"] is None
    assert [r["token_id"] for r in events[0]["accepted_tokens"]] == [10, 20]


def test_accepted_tokens_preserve_bonus_and_correction_provenance():
    tracer = RetroSpecTokenMetricsTracer(True)
    events = []
    tracer._emit = events.append
    traced_proposal(tracer)
    tracer.record_bonus(
        ["request"], torch.tensor([True]), torch.tensor([1]), torch.tensor([99])
    )
    tracer.record_replacements(
        ["request"],
        torch.tensor([0]),
        torch.tensor([2]),
        torch.tensor([30]),
        torch.tensor([88]),
        "sparse_correction",
    )
    tracer.submit(["request"], [[10, 99, 88, 40]])
    tracer.record_outcomes(["request"], [4], [4], [[10, 99, 88, 77]])
    accepted = events[0]["accepted_tokens"]
    assert [r["token_index"] for r in accepted] == [0, 1, 2]
    assert [r["token_id"] for r in accepted] == [10, 99, 88]
    assert [r["origin"] for r in accepted] == [
        "draft",
        "sparse_bonus",
        "sparse_correction",
    ]
    assert accepted[0]["draft_margin"] == pytest.approx(0.1)
    assert all(r["draft_margin"] is None for r in accepted[1:])
    assert all(r["hit_attn"] is None for r in accepted[1:])
    assert events[0]["first_mismatch"]["token_index"] == 3


def test_misaligned_target_prefix_fails_instead_of_mislabeling_metrics():
    tracer = RetroSpecTokenMetricsTracer(True)
    traced_proposal(tracer)
    tracer.submit(["request"], [[10, 20, 30, 40]])
    with pytest.raises(RuntimeError, match="accepted prefix"):
        tracer.record_outcomes(["request"], [4], [2], [[11, 99]])


def test_disabled_tracer_never_reads_tensors():
    tracer = RetroSpecTokenMetricsTracer(False)
    unreadable = Mock()
    tracer.begin(["request"], unreadable)
    tracer.record_draft(["request"], 0, unreadable, unreadable, None, unreadable)
    tracer.submit(["request"], [[10]])
    tracer.record_outcomes(["request"], [1], [1], None)
    assert unreadable.mock_calls == []


@pytest.mark.parametrize("enabled", [False, True])
def test_trace_observes_both_signals_without_enabling_thresholds(enabled):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_trace_token_metrics=enabled),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._run_model_step = Mock()
    proposer._run_draft_step(
        1,
        0,
        make_common_metadata([2]),
        torch.tensor([True]),
        make_sampling_metadata(all_greedy=True),
    )
    assert proposer._run_model_step.call_args.kwargs["compute_margin"] is enabled
    assert proposer.sparse_attention.reduce_draft_attention_mass is enabled
    assert proposer.policy.draft_margin_threshold is None
    assert proposer.policy.hit_attn_threshold is None
