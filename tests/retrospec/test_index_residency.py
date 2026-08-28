# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecGPUIndexResidencyManager,
    RetroSpecResidentSegment,
)

pytestmark = pytest.mark.cpu_test


def make_manager(max_resident_requests: int = 2):
    return RetroSpecGPUIndexResidencyManager(
        pin_memory=False,
        max_resident_requests=max_resident_requests,
        max_gpu_index_memory_bytes=1 << 20,
    )


def make_resident_segment(
    manager: RetroSpecGPUIndexResidencyManager,
    *,
    layer_name: str = "layer",
    request_id: str = "request",
    indexed_start: int = 2,
    cluster_start: int = 0,
    num_clusters: int = 1,
) -> RetroSpecResidentSegment:
    keys = torch.arange(
        cluster_start + 1,
        cluster_start + num_clusters + 1,
        dtype=torch.float32,
    ).view(1, num_clusters, 1)
    values = keys + 10
    counts = torch.full((1, num_clusters), 2, dtype=torch.int32)
    staged = manager.stage_cluster_summary(keys, values, counts)
    summary = manager.finish_cluster_summary(staged)
    cluster_ids = torch.arange(
        cluster_start, cluster_start + num_clusters, dtype=torch.int64
    ).view(1, num_clusters)

    return manager.build_resident_segment(
        layer_name=layer_name,
        request_id=request_id,
        indexed_start=indexed_start,
        indexed_end=indexed_start + 2 * num_clusters,
        cluster_start=cluster_start,
        resident_summary=staged.resident_summary,
        cluster_token_counts_cpu=summary.cluster_token_counts,
        cluster_ids=cluster_ids,
        cluster_page_ids=cluster_ids.unsqueeze(-1),
        cluster_page_token_counts=torch.full(
            (1, num_clusters, 1), 2, dtype=torch.int32
        ),
    )


def test_residency_manager_scopes_views_to_active_request_set():
    manager = make_manager()

    manager.activate(["first", "second"])
    assert manager.active_request_ids == ("first", "second")
    first = manager.get_active_view("layer", ["first", "second"], torch.device("cpu"))
    second = manager.get_active_view("layer", ["first", "second"], torch.device("cpu"))
    assert first is second
    assert first.request_slot_ids.tolist() == [-1, -1]

    manager.deactivate()
    assert manager.active_request_ids == ()


def test_residency_manager_enforces_capacity_and_request_order():
    manager = make_manager(max_resident_requests=2)

    with pytest.raises(RuntimeError, match="exceeds max_num_seqs"):
        manager.activate(["first", "second", "third"])

    manager.activate(["first", "second"])
    try:
        with pytest.raises(RuntimeError, match="request order"):
            manager.get_active_view("layer", ["second", "first"], torch.device("cpu"))
    finally:
        manager.deactivate()


def test_residency_manager_rejects_duplicate_or_unscoped_requests():
    manager = make_manager()

    with pytest.raises(ValueError, match="unique"):
        manager.activate(["request", "request"])

    with pytest.raises(RuntimeError, match="active proposal"):
        manager.get_active_view("layer", ["request"], torch.device("cpu"))


def test_resident_segments_survive_batch_deactivation():
    manager = make_manager()
    segment = make_resident_segment(manager)
    manager.publish_resident_segments([segment])

    assert manager.resident_request_ids == ("request",)
    assert manager.num_resident_layers == 1

    manager.activate(["request"])
    view = manager.get_active_view("layer", ["request"], torch.device("cpu"))
    assert view.request_slot_ids.tolist() == [0]
    manager.deactivate()

    assert manager.get_num_clusters("layer", "request") == 1
    assert manager.get_indexed_end("layer", "request") == 4


def test_active_view_omits_arena_when_layer_has_no_requested_resident_index():
    manager = make_manager()
    manager.publish_resident_segments(
        [make_resident_segment(manager, request_id="resident")]
    )

    manager.activate(["missing"])
    try:
        view = manager.get_active_view("layer", ["missing"], torch.device("cpu"))
        assert view.arena is None
        assert view.request_slot_ids.tolist() == [-1]
    finally:
        manager.deactivate()


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
    assert manager.get_num_clusters("layer", "request") == 2
    assert manager.get_indexed_end("layer", "request") == 6

    manager.invalidate_requests(["other"])
    assert manager.resident_request_ids == ("request",)
    assert manager.get_num_clusters("layer", "other") == 0

    manager.discard_request_layer("layer", "request")
    assert manager.get_num_clusters("layer", "request") == 0
    assert manager.get_num_clusters("other-layer", "request") == 1


def test_residency_manager_enforces_persistent_request_capacity_atomically():
    manager = make_manager(max_resident_requests=1)
    first = make_resident_segment(manager, request_id="first")
    second = make_resident_segment(manager, request_id="second")
    manager.publish_resident_segments([first])

    with pytest.raises(RuntimeError, match="persistent GPU index residency"):
        manager.publish_resident_segments([second])

    assert manager.resident_request_ids == ("first",)
    assert manager.get_num_clusters("layer", "first") == 1
    assert manager.get_num_clusters("layer", "second") == 0


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

    assert manager.get_num_clusters("layer", "request") == 1
    assert manager.get_indexed_end("layer", "request") == 4


def test_failed_publish_releases_new_request_slot():
    manager = make_manager(max_resident_requests=1)
    invalid = make_resident_segment(
        manager,
        request_id="invalid",
        cluster_start=1,
    )

    with pytest.raises(RuntimeError, match="start at cluster zero"):
        manager.publish_resident_segments([invalid])

    assert manager.resident_request_ids == ()
    manager.publish_resident_segments(
        [make_resident_segment(manager, request_id="replacement")]
    )
    assert manager.resident_request_ids == ("replacement",)


def test_resident_segment_rejects_inconsistent_cpu_page_descriptors():
    manager = make_manager()
    keys = torch.ones(1, 1, 1)
    staged = manager.stage_cluster_summary(
        keys,
        keys,
        torch.tensor([[2]], dtype=torch.int32),
    )
    summary = manager.finish_cluster_summary(staged)

    with pytest.raises(ValueError, match="do not match page descriptors"):
        manager.build_resident_segment(
            layer_name="layer",
            request_id="request",
            indexed_start=2,
            indexed_end=4,
            cluster_start=0,
            resident_summary=staged.resident_summary,
            cluster_token_counts_cpu=summary.cluster_token_counts,
            cluster_ids=torch.tensor([[0]]),
            cluster_page_ids=torch.tensor([[[0]]]),
            cluster_page_token_counts=torch.tensor([[[1]]], dtype=torch.int32),
        )


def test_resident_arena_uses_request_slots_and_ragged_page_offsets():
    manager = make_manager()
    keys = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    values = keys + 10
    counts = torch.tensor([[3, 1, 0], [1, 3, 1]], dtype=torch.int32)
    staged = manager.stage_cluster_summary(keys, values, counts)
    summary = manager.finish_cluster_summary(staged)
    segment = manager.build_resident_segment(
        layer_name="layer",
        request_id="request",
        indexed_start=2,
        indexed_end=8,
        cluster_start=0,
        resident_summary=staged.resident_summary,
        cluster_token_counts_cpu=summary.cluster_token_counts,
        cluster_ids=torch.tensor([[10, 11, -1], [12, 13, 14]]),
        cluster_page_ids=torch.tensor(
            [
                [[20, 21, -1], [22, -1, -1], [-1, -1, -1]],
                [[23, -1, -1], [24, 25, -1], [26, -1, -1]],
            ]
        ),
        cluster_page_token_counts=torch.tensor(
            [
                [[2, 1, 0], [1, 0, 0], [0, 0, 0]],
                [[1, 0, 0], [2, 1, 0], [1, 0, 0]],
            ],
            dtype=torch.int32,
        ),
    )
    manager.publish_resident_segments([segment])

    manager.activate(["request"])
    try:
        view = manager.get_active_view("layer", ["request"], torch.device("cpu"))
        assert view.arena is not None
        assert view.request_slot_ids.tolist() == [0]
        assert view.max_num_clusters == 3
        assert view.max_pages_per_cluster == 2
        assert view.max_num_pages == 4

        arena = view.arena
        assert arena.indexed_starts[0].item() == 2
        assert arena.indexed_ends[0].item() == 8
        assert arena.cluster_offsets[0].item() == 0
        assert arena.num_clusters[0].item() == 3
        assert arena.page_offsets[0].item() == 0
        assert arena.num_pages[0].tolist() == [3, 4]
        assert arena.cluster_page_starts[:, :3].tolist() == [
            [0, 2, 3],
            [0, 1, 3],
        ]
        assert arena.cluster_page_counts[:, :3].tolist() == [
            [2, 1, 0],
            [1, 2, 1],
        ]
        assert arena.page_ids[0, :3].tolist() == [20, 21, 22]
        assert arena.page_ids[1, :4].tolist() == [23, 24, 25, 26]

        assert arena.cluster_ids[:, :3].tolist() == [
            [10, 11, -1],
            [12, 13, 14],
        ]
    finally:
        manager.deactivate()


def test_request_slot_is_stable_until_removal_and_then_reused():
    manager = make_manager(max_resident_requests=2)
    manager.publish_resident_segments(
        [make_resident_segment(manager, request_id="first")]
    )

    manager.activate(["first"])
    try:
        first_view = manager.get_active_view("layer", ["first"], torch.device("cpu"))
        first_slot = first_view.request_slot_ids.item()
        assert first_view.arena is not None
        first_generation = first_view.arena.generations[first_slot].item()
    finally:
        manager.deactivate()

    manager.activate(["first"])
    try:
        repeated_view = manager.get_active_view("layer", ["first"], torch.device("cpu"))
        assert repeated_view.request_slot_ids.item() == first_slot
    finally:
        manager.deactivate()

    manager.invalidate_requests(["first"])
    manager.publish_resident_segments(
        [make_resident_segment(manager, request_id="replacement")]
    )
    manager.activate(["replacement"])
    try:
        replacement_view = manager.get_active_view(
            "layer", ["replacement"], torch.device("cpu")
        )
        assert replacement_view.request_slot_ids.item() == first_slot
        assert replacement_view.arena is not None
        assert replacement_view.arena.generations[first_slot].item() > first_generation
    finally:
        manager.deactivate()


def test_resident_arena_grows_from_published_cluster_count():
    manager = make_manager()
    first = make_resident_segment(manager, num_clusters=3)
    manager.publish_resident_segments([first])

    layer_state = manager._layer_arenas["layer"]
    assert layer_state.arena.cluster_keys.shape == (1, 64, 1)
    assert layer_state.arena.page_ids.shape == (1, 64)

    second = make_resident_segment(
        manager,
        indexed_start=8,
        cluster_start=3,
        num_clusters=62,
    )
    manager.publish_resident_segments([second])

    state = manager._resident_states["layer"]["request"]
    assert state.num_clusters == 65
    assert state.cluster_capacity == 128
    assert state.page_capacity == 128
    cluster_slice = slice(state.cluster_offset, state.cluster_offset + 65)
    assert layer_state.arena.cluster_ids[0, cluster_slice].tolist() == list(range(65))


def test_resident_arena_reuses_released_cluster_and_page_spans():
    manager = make_manager(max_resident_requests=2)
    manager.publish_resident_segments(
        [make_resident_segment(manager, request_id="first", num_clusters=3)]
    )
    first_state = manager._resident_states["layer"]["first"]

    manager.invalidate_requests(["first"])
    manager.publish_resident_segments(
        [make_resident_segment(manager, request_id="replacement", num_clusters=3)]
    )
    replacement_state = manager._resident_states["layer"]["replacement"]

    assert replacement_state.cluster_offset == first_state.cluster_offset
    assert replacement_state.page_offset == first_state.page_offset


def test_gpu_index_budget_failure_preserves_previous_resident_span():
    manager = RetroSpecGPUIndexResidencyManager(
        pin_memory=False,
        max_resident_requests=1,
        max_gpu_index_memory_bytes=3500,
    )
    manager.publish_resident_segments([make_resident_segment(manager, num_clusters=64)])
    original = manager._resident_states["layer"]["request"]

    with pytest.raises(RuntimeError, match="GPU index memory budget"):
        manager.publish_resident_segments(
            [
                make_resident_segment(
                    manager,
                    indexed_start=130,
                    cluster_start=64,
                )
            ]
        )

    current = manager._resident_states["layer"]["request"]
    assert current == original
    assert manager.get_num_clusters("layer", "request") == 64
    arena = manager._layer_arenas["layer"].arena
    assert arena.num_clusters[original.slot].item() == 64
    cluster_slice = slice(original.cluster_offset, original.cluster_offset + 64)
    assert arena.cluster_ids[0, cluster_slice].tolist() == list(range(64))


def test_multi_layer_publish_rolls_back_new_spans_atomically():
    manager = RetroSpecGPUIndexResidencyManager(
        pin_memory=False,
        max_resident_requests=1,
        max_gpu_index_memory_bytes=3500,
    )
    first = make_resident_segment(manager, layer_name="first")
    second = make_resident_segment(manager, layer_name="second")

    with pytest.raises(RuntimeError, match="GPU index memory budget"):
        manager.publish_resident_segments([first, second])

    assert manager.resident_request_ids == ()
    assert manager.get_num_clusters("first", "request") == 0
    manager.publish_resident_segments([first])
    state = manager._resident_states["first"]["request"]
    assert state.cluster_offset == 0
    assert state.page_offset == 0


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
        pin_memory=True,
        max_resident_requests=2,
        max_gpu_index_memory_bytes=1 << 20,
    )
    keys = torch.arange(8, dtype=torch.float32, device="cuda").view(1, 2, 4)
    values = keys + 10
    counts = torch.tensor([[3, 4]], dtype=torch.int32, device="cuda")

    staged = manager.stage_cluster_summary(keys, values, counts)
    assert staged.ready_event is not None
    assert staged.cluster_keys.is_pinned()
    assert staged.cluster_values.is_pinned()
    assert staged.cluster_token_counts.is_pinned()
    assert staged.resident_summary.cluster_keys is keys
    assert staged.resident_summary.cluster_values is values
    assert staged.resident_summary.cluster_token_counts is counts

    summary = manager.finish_cluster_summary(staged)

    assert not summary.cluster_keys.is_pinned()
    assert not summary.cluster_values.is_pinned()
    assert not summary.cluster_token_counts.is_pinned()
    assert summary.cluster_keys.flatten().tolist() == pytest.approx(list(range(8)))
    assert summary.cluster_values.flatten().tolist() == pytest.approx(
        list(range(10, 18))
    )
    assert summary.cluster_token_counts.tolist() == [[3, 4]]
    assert staged.staging_slot is not None
    assert not staged.staging_slot.in_use
    assert manager._pinned_memory.allocated_bytes > 0
    manager.close()
    assert manager._pinned_memory.allocated_bytes == 0
