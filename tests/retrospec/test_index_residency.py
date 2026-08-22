# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecGPUIndexResidencyManager,
    RetroSpecResidentIndex,
)

pytestmark = pytest.mark.cpu_test


def make_resident_index() -> RetroSpecResidentIndex:
    return RetroSpecResidentIndex(
        indexed_token_mask=torch.ones(1, 2, dtype=torch.bool),
        cluster_ids=torch.zeros(1, 1, 1, dtype=torch.int64),
        cluster_keys=torch.zeros(1, 1, 1, 1),
        cluster_values=torch.zeros(1, 1, 1, 1),
        cluster_token_counts=torch.ones(1, 1, 1, dtype=torch.int32),
        cluster_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        cluster_page_ids=torch.zeros(1, 1, 1, 1, dtype=torch.int64),
        cluster_page_token_counts=torch.ones(1, 1, 1, 1, dtype=torch.int32),
    )


def make_manager(max_resident_requests: int = 2):
    return RetroSpecGPUIndexResidencyManager(
        cpu_offload=True,
        pin_memory=False,
        max_resident_requests=max_resident_requests,
    )


def test_residency_manager_scopes_indices_to_active_request_set():
    manager = make_manager()
    packed = make_resident_index()

    manager.activate(["first", "second"])
    assert manager.active_request_ids == ("first", "second")
    assert manager.get("layer", ["first", "second"], 16) is None

    manager.put("layer", ["first", "second"], 16, packed)
    assert manager.get("layer", ["first", "second"], 16) is packed
    assert manager.num_resident_layers == 1

    manager.deactivate()
    assert manager.active_request_ids == ()
    assert manager.num_resident_layers == 0


def test_residency_manager_enforces_capacity_and_request_order():
    manager = make_manager(max_resident_requests=2)

    with pytest.raises(RuntimeError, match="exceeds max_num_seqs"):
        manager.activate(["first", "second", "third"])

    manager.activate(["first", "second"])
    try:
        with pytest.raises(RuntimeError, match="request order"):
            manager.get("layer", ["second", "first"], 16)
    finally:
        manager.deactivate()


def test_residency_manager_rejects_duplicate_or_unscoped_requests():
    manager = make_manager()

    with pytest.raises(ValueError, match="unique"):
        manager.activate(["request", "request"])

    with pytest.raises(RuntimeError, match="active proposal"):
        manager.get("layer", ["request"], 16)


def test_residency_manager_invalidates_only_affected_entries():
    manager = make_manager()
    manager.activate(["first", "second"])
    manager.put("first-layer", ["first", "second"], 16, make_resident_index())
    manager.put("second-layer", ["first", "second"], 16, make_resident_index())

    manager.invalidate_layer("first-layer")
    assert manager.num_resident_layers == 1

    manager.invalidate_requests(["second"])
    assert manager.num_resident_layers == 0
    manager.deactivate()


def test_cluster_summary_is_copied_to_cpu_authoritative_storage():
    manager = make_manager()
    keys = torch.tensor([[[1.0], [2.0]]])
    values = keys + 10
    counts = torch.tensor([[3, 4]], dtype=torch.int32)

    staged = manager.stage_cluster_summary(keys, values, counts)
    keys.fill_(-1)
    values.fill_(-1)
    counts.fill_(-1)
    summary = manager.finish_cluster_summary(staged)

    assert summary.cluster_keys.tolist() == [[[1.0], [2.0]]]
    assert summary.cluster_values.tolist() == [[[11.0], [12.0]]]
    assert summary.cluster_token_counts.tolist() == [[3, 4]]
    assert summary.cluster_keys.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cluster_summary_offloads_asynchronously_to_pinned_cpu_storage():
    manager = RetroSpecGPUIndexResidencyManager(
        cpu_offload=True,
        pin_memory=True,
        max_resident_requests=2,
    )
    keys = torch.arange(8, dtype=torch.float32, device="cuda").view(1, 2, 4)
    values = keys + 10
    counts = torch.tensor([[3, 4]], dtype=torch.int32, device="cuda")

    staged = manager.stage_cluster_summary(keys, values, counts)
    assert staged.ready_event is not None
    assert len(staged.source_tensors) == 3

    summary = manager.finish_cluster_summary(staged)

    assert summary.cluster_keys.is_pinned()
    assert summary.cluster_values.is_pinned()
    assert summary.cluster_token_counts.is_pinned()
    assert summary.cluster_keys.flatten().tolist() == pytest.approx(list(range(8)))
    assert summary.cluster_values.flatten().tolist() == pytest.approx(
        list(range(10, 18))
    )
    assert summary.cluster_token_counts.tolist() == [[3, 4]]
