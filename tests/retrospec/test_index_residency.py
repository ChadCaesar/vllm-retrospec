# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecGPUIndexResidencyManager,
    RetroSpecResidentIndex,
    RetroSpecResidentSegment,
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


def make_resident_segment(
    manager: RetroSpecGPUIndexResidencyManager,
    *,
    layer_name: str = "layer",
    request_id: str = "request",
    indexed_start: int = 2,
    cluster_start: int = 0,
) -> RetroSpecResidentSegment:
    keys = torch.tensor([[[float(cluster_start + 1)]]])
    values = keys + 10
    counts = torch.ones(1, 1, dtype=torch.int32)
    staged = manager.stage_cluster_summary(keys, values, counts)
    manager.finish_cluster_summary(staged)

    return manager.build_resident_segment(
        layer_name=layer_name,
        request_id=request_id,
        indexed_start=indexed_start,
        indexed_end=indexed_start + 2,
        cluster_start=cluster_start,
        staged_summary=staged,
        cluster_ids=torch.tensor([[cluster_start]], dtype=torch.int64),
        cluster_page_ids=torch.tensor([[[cluster_start]]], dtype=torch.int64),
        cluster_page_token_counts=torch.tensor([[[2]]], dtype=torch.int32),
    )


def test_residency_manager_scopes_indices_to_active_request_set():
    manager = make_manager()
    packed = make_resident_index()

    manager.activate(["first", "second"])
    assert manager.active_request_ids == ("first", "second")
    assert manager.get("layer", ["first", "second"], 16) is None

    manager.put("layer", ["first", "second"], 16, packed)
    assert manager.get("layer", ["first", "second"], 16) is packed
    assert manager.num_packed_layers == 1

    manager.deactivate()
    assert manager.active_request_ids == ()
    assert manager.num_packed_layers == 0


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


def test_resident_segments_survive_batch_deactivation():
    manager = make_manager()
    segment = make_resident_segment(manager)
    manager.publish_resident_segments([segment])

    assert manager.resident_request_ids == ("request",)
    assert manager.num_resident_layers == 1

    manager.activate(["request"])
    manager.put("layer", ["request"], 16, make_resident_index())
    assert manager.num_packed_layers == 1
    manager.deactivate()

    assert manager.num_packed_layers == 0
    assert manager.get_resident_segments("layer", "request") == (segment,)


def test_residency_manager_appends_and_invalidates_request_segments():
    manager = make_manager()
    first = make_resident_segment(manager)
    second = make_resident_segment(
        manager,
        indexed_start=4,
        cluster_start=1,
    )
    other_layer = make_resident_segment(manager, layer_name="other-layer")
    other_request = make_resident_segment(manager, request_id="other")
    manager.publish_resident_segments([first, second, other_layer, other_request])

    assert manager.resident_request_ids == ("other", "request")
    assert manager.num_resident_layers == 2
    assert manager.get_resident_segments("layer", "request") == (first, second)

    manager.invalidate_requests(["other"])
    assert manager.resident_request_ids == ("request",)
    assert manager.get_resident_segments("layer", "other") == ()

    manager.discard_request_layer("layer", "request")
    assert manager.get_resident_segments("layer", "request") == ()
    assert manager.get_resident_segments("other-layer", "request") == (other_layer,)


def test_residency_manager_enforces_persistent_request_capacity_atomically():
    manager = make_manager(max_resident_requests=1)
    first = make_resident_segment(manager, request_id="first")
    second = make_resident_segment(manager, request_id="second")
    manager.publish_resident_segments([first])

    with pytest.raises(RuntimeError, match="persistent GPU index residency"):
        manager.publish_resident_segments([second])

    assert manager.resident_request_ids == ("first",)
    assert manager.get_resident_segments("layer", "first") == (first,)
    assert manager.get_resident_segments("layer", "second") == ()


def test_residency_manager_rejects_noncontiguous_segment_publish():
    manager = make_manager()
    first = make_resident_segment(manager)
    noncontiguous = make_resident_segment(
        manager,
        indexed_start=6,
        cluster_start=1,
    )
    manager.publish_resident_segments([first])

    with pytest.raises(RuntimeError, match="indexed token prefix"):
        manager.publish_resident_segments([noncontiguous])

    assert manager.get_resident_segments("layer", "request") == (first,)


def test_cluster_summary_is_copied_to_cpu_authoritative_storage():
    manager = make_manager()
    keys = torch.tensor([[[1.0], [2.0]]])
    values = keys + 10
    counts = torch.tensor([[3, 4]], dtype=torch.int32)

    staged = manager.stage_cluster_summary(keys, values, counts)
    assert staged.resident_summary.cluster_keys is keys
    assert staged.resident_summary.cluster_values is values
    assert staged.resident_summary.cluster_token_counts is counts
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
    assert staged.resident_summary.cluster_keys is keys
    assert staged.resident_summary.cluster_values is values
    assert staged.resident_summary.cluster_token_counts is counts

    summary = manager.finish_cluster_summary(staged)

    assert summary.cluster_keys.is_pinned()
    assert summary.cluster_values.is_pinned()
    assert summary.cluster_token_counts.is_pinned()
    assert summary.cluster_keys.flatten().tolist() == pytest.approx(list(range(8)))
    assert summary.cluster_values.flatten().tolist() == pytest.approx(
        list(range(10, 18))
    )
    assert summary.cluster_token_counts.tolist() == [[3, 4]]
