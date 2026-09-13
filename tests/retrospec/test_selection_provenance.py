# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.retrospec import selection_provenance
from vllm.v1.spec_decode.retrospec.selection_provenance import (
    RetroSpecSelectionProvenanceTracer,
)


def extract_payload(logger_info: Mock) -> dict[str, object]:
    logger_info.assert_called_once()
    message, payload = logger_info.call_args.args
    assert message == "RetroSpec selection provenance: %s"
    return json.loads(payload)


def test_replay_mode_validation():
    with pytest.raises(ValueError, match="Unsupported RetroSpec replay mode"):
        RetroSpecSelectionProvenanceTracer("invalid")  # type: ignore[arg-type]


def test_disabled_trace_returns_before_tensor_digest(monkeypatch):
    logger_info = Mock()
    digest = Mock(side_effect=AssertionError("disabled trace inspected tensors"))
    monkeypatch.setattr(selection_provenance.logger, "info", logger_info)
    monkeypatch.setattr(RetroSpecSelectionProvenanceTracer, "_tensor_digest", digest)
    tracer = RetroSpecSelectionProvenanceTracer("off")

    tracer.record_index_segment(
        request_id="request",
        layer_name="layer",
        indexed_start=2,
        indexed_end=6,
        cluster_start=0,
        assignments=torch.tensor([[0, 0, 1, 1]]),
        cluster_sizes=torch.tensor([[2, 2]], dtype=torch.int32),
        cluster_keys=torch.ones(1, 2, 1),
        cluster_values=torch.ones(1, 2, 1),
        token_offsets_in_cluster=torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
    )

    logger_info.assert_not_called()
    digest.assert_not_called()


def test_index_segment_checksum_is_stable_and_order_sensitive(monkeypatch):
    logger_info = Mock()
    monkeypatch.setattr(selection_provenance.logger, "info", logger_info)
    tracer = RetroSpecSelectionProvenanceTracer("trace")
    kwargs = {
        "request_id": "request",
        "layer_name": "layer.0",
        "indexed_start": 16,
        "indexed_end": 20,
        "cluster_start": 3,
        "assignments": torch.tensor([[0, 0, 1, 1]]),
        "cluster_sizes": torch.tensor([[2, 2]], dtype=torch.int32),
        "cluster_keys": torch.tensor([[[1.0], [2.0]]], dtype=torch.bfloat16),
        "cluster_values": torch.tensor([[[3.0], [4.0]]], dtype=torch.bfloat16),
        "token_offsets_in_cluster": torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
    }

    tracer.record_index_segment(**kwargs)
    first = extract_payload(logger_info)
    logger_info.reset_mock()
    tracer.record_index_segment(**kwargs)
    second = extract_payload(logger_info)

    assert first == second
    assert first == {
        "checksum": first["checksum"],
        "cluster_end": 5,
        "cluster_start": 3,
        "indexed_end": 20,
        "indexed_start": 16,
        "layer_name": "layer.0",
        "mode": "trace",
        "num_clusters": 2,
        "phase": "index_segment",
        "request_id": "request",
    }

    logger_info.reset_mock()
    changed = dict(kwargs)
    changed["assignments"] = kwargs["assignments"].flip(1)
    tracer.record_index_segment(**changed)
    reordered = extract_payload(logger_info)
    assert reordered["checksum"] != first["checksum"]


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
def test_draft_selection_records_only_active_rows(monkeypatch, device):
    logger_info = Mock()
    monkeypatch.setattr(selection_provenance.logger, "info", logger_info)
    tracer = RetroSpecSelectionProvenanceTracer("trace")

    tracer.record_draft_selection(
        request_ids=("active", "idle"),
        layer_name="layer.0",
        proposal_round=4,
        draft_step=2,
        snapshot="used",
        physical_source="resident",
        index_revisions=(7, 11),
        active_mask=torch.tensor([True, False], device=device),
        query=torch.arange(8, dtype=torch.float32, device=device).view(2, 1, 4),
        request_slot_ids=torch.tensor([3, 5], device=device),
        request_slot_generations=torch.tensor([9, 13], device=device),
        sparse_exact_cluster_indices=torch.tensor(
            [[[4, -1]], [[6, 7]]], dtype=torch.int32, device=device
        ),
        sparse_estimation_cluster_indices=torch.tensor(
            [[[8]], [[9]]], dtype=torch.int32, device=device
        ),
        exact_cluster_handles=torch.tensor(
            [[[40, 999]], [[60, 70]]], dtype=torch.int64, device=device
        ),
        candidate_counts=torch.tensor([[3], [4]], dtype=torch.int32, device=device),
        selected_cluster_counts=torch.tensor(
            [[2], [3]], dtype=torch.int32, device=device
        ),
        hit_cluster_counts=torch.tensor([[1], [2]], dtype=torch.int32, device=device),
        miss_cluster_counts=torch.tensor([[1], [1]], dtype=torch.int32, device=device),
        hit_gate_ready=torch.tensor([[True], [False]], device=device),
    )

    payload = extract_payload(logger_info)
    assert payload["phase"] == "draft_selection"
    assert payload["mode"] == "trace"
    assert payload["proposal_round"] == 4
    assert payload["draft_step"] == 2
    assert payload["snapshot"] == "used"
    assert payload["physical_source"] == "resident"
    assert payload["records"] == [
        {
            "candidate_count": 3,
            "gate_ready_heads": 1,
            "hit_count": 1,
            "index_revision": 7,
            "miss_count": 1,
            "ordered_topk_checksum": payload["records"][0]["ordered_topk_checksum"],
            "query_checksum": payload["records"][0]["query_checksum"],
            "request_id": "active",
            "request_index": 0,
            "request_slot": 3,
            "request_slot_generation": 9,
            "selected_count": 2,
            "stable_handle_checksum": payload["records"][0]["stable_handle_checksum"],
        }
    ]


def test_draft_selection_validates_request_metadata():
    tracer = RetroSpecSelectionProvenanceTracer("trace")
    values = torch.ones(1, 1, 1)
    indices = torch.zeros(1, 1, 1, dtype=torch.int32)
    counts = torch.ones(1, 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="Index revisions"):
        tracer.record_draft_selection(
            request_ids=("request",),
            layer_name="layer",
            proposal_round=1,
            draft_step=0,
            snapshot="used",
            physical_source="resident",
            index_revisions=(),
            active_mask=torch.tensor([True]),
            query=values,
            request_slot_ids=torch.tensor([0]),
            request_slot_generations=torch.tensor([0]),
            sparse_exact_cluster_indices=indices,
            sparse_estimation_cluster_indices=indices[:, :, :0],
            exact_cluster_handles=indices.to(torch.int64),
            candidate_counts=counts,
            selected_cluster_counts=counts,
            hit_cluster_counts=counts,
            miss_cluster_counts=torch.zeros_like(counts),
            hit_gate_ready=torch.ones_like(counts, dtype=torch.bool),
        )
