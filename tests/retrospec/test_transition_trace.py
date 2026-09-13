# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.retrospec import transition_trace
from vllm.v1.spec_decode.retrospec.decision import RetroSpecReason
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage
from vllm.v1.spec_decode.retrospec.transition_trace import (
    RetroSpecTransitionTracer,
)


def extract_payload(logger_info: Mock) -> dict[str, object]:
    logger_info.assert_called_once()
    message, payload = logger_info.call_args.args
    assert message == "RetroSpec transition trace: %s"
    return json.loads(payload)


def test_disabled_trace_returns_before_tensor_validation(monkeypatch):
    logger_info = Mock()
    monkeypatch.setattr(transition_trace.logger, "info", logger_info)
    tracer = RetroSpecTransitionTracer(enabled=False)

    tracer.record_masked(
        request_ids=["request-0"],
        phase="draft_to_sparse",
        proposal_round=1,
        mask=torch.tensor([1], dtype=torch.int32),
        integer_fields={"position": torch.tensor([1, 2])},
    )

    logger_info.assert_not_called()


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
def test_record_masked_compacts_rows_and_decodes_enums(monkeypatch, device):
    logger_info = Mock()
    monkeypatch.setattr(transition_trace.logger, "info", logger_info)
    tracer = RetroSpecTransitionTracer(enabled=True)

    tracer.record_masked(
        request_ids=["request-0", "request-1", "request-2"],
        phase="draft_to_sparse",
        proposal_round=2,
        mask=torch.tensor([False, True, False], device=device),
        integer_fields={
            "position": torch.tensor([10, 20, 30], device=device),
            "request_stage": torch.tensor(
                [
                    int(RetroSpecStage.DRAFT),
                    int(RetroSpecStage.DRAFT),
                    int(RetroSpecStage.IDLE),
                ],
                dtype=torch.int8,
                device=device,
            ),
            "next_stage": torch.tensor(
                [
                    int(RetroSpecStage.DRAFT),
                    int(RetroSpecStage.SPARSE_VERIFY),
                    int(RetroSpecStage.IDLE),
                ],
                dtype=torch.int8,
                device=device,
            ),
            "reasons": torch.tensor(
                [
                    0,
                    int(RetroSpecReason.DRAFT_MARGIN | RetroSpecReason.HIT_ATTN),
                    0,
                ],
                dtype=torch.int32,
                device=device,
            ),
        },
        float_fields={
            "draft_margin": torch.tensor([0.9, 0.25, 0.8], device=device),
            "unused_metric": None,
        },
    )

    payload = extract_payload(logger_info)
    assert payload["phase"] == "draft_to_sparse"
    assert payload["proposal_round"] == 2
    assert payload["records"] == [
        {
            "draft_margin": pytest.approx(0.25),
            "next_stage": int(RetroSpecStage.SPARSE_VERIFY),
            "next_stage_name": "SPARSE_VERIFY",
            "position": 20,
            "reason_names": ["DRAFT_MARGIN", "HIT_ATTN"],
            "reasons": int(RetroSpecReason.DRAFT_MARGIN | RetroSpecReason.HIT_ATTN),
            "request_id": "request-1",
            "request_index": 1,
            "request_stage": int(RetroSpecStage.DRAFT),
            "request_stage_name": "DRAFT",
        }
    ]


def test_record_compact_preserves_original_request_indices(monkeypatch):
    logger_info = Mock()
    monkeypatch.setattr(transition_trace.logger, "info", logger_info)
    tracer = RetroSpecTransitionTracer(enabled=True)

    tracer.record_compact(
        request_ids=["request-0", "request-1", "request-2"],
        phase="expanded_boundary",
        proposal_round=3,
        request_indices=torch.tensor([2, 0]),
        integer_fields={
            "token_index": torch.tensor([4, 1]),
            "reasons": torch.tensor(
                [
                    int(RetroSpecReason.EXPANDED_TOKEN_CHANGED),
                    int(RetroSpecReason.GENERATION_LIMIT),
                ]
            ),
        },
        float_fields={"expanded_attention": torch.tensor([0.4, 0.8])},
    )

    payload = extract_payload(logger_info)
    records = payload["records"]
    assert isinstance(records, list)
    assert [record["request_id"] for record in records] == [
        "request-2",
        "request-0",
    ]
    assert [record["token_index"] for record in records] == [4, 1]
    assert records[0]["reason_names"] == ["EXPANDED_TOKEN_CHANGED"]
    assert records[1]["reason_names"] == ["GENERATION_LIMIT"]


@pytest.mark.parametrize(
    ("request_indices", "match"),
    [
        (torch.tensor([[0]]), "one-dimensional"),
        (torch.tensor([0.0]), "integer dtype"),
    ],
)
def test_record_compact_validates_request_indices(request_indices, match):
    tracer = RetroSpecTransitionTracer(enabled=True)

    with pytest.raises(ValueError, match=match):
        tracer.record_compact(
            request_ids=["request-0"],
            phase="sparse_boundary",
            proposal_round=1,
            request_indices=request_indices,
            integer_fields={},
        )


def test_record_compact_rejects_out_of_range_request_index(monkeypatch):
    monkeypatch.setattr(transition_trace.logger, "info", Mock())
    tracer = RetroSpecTransitionTracer(enabled=True)

    with pytest.raises(ValueError, match="outside the current batch"):
        tracer.record_compact(
            request_ids=["request-0"],
            phase="sparse_boundary",
            proposal_round=1,
            request_indices=torch.tensor([1]),
            integer_fields={"position": torch.tensor([10])},
        )
