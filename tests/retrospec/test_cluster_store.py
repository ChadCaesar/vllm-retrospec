# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.spec_decode.retrospec.cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterPageStore,
)


def make_cluster_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    keys = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[10.0], [11.0], [12.0], [13.0], [14.0]],
        ]
    )
    values = keys + 100.0
    assignments = torch.tensor(
        [
            [0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1],
        ],
        dtype=torch.int64,
    )
    cluster_token_counts = torch.tensor(
        [[3, 2], [2, 3]],
        dtype=torch.int32,
    )
    return keys, values, assignments, cluster_token_counts


def store_cluster_data(
    store: RetroSpecClusterPageStore,
    layer_name: str,
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
    assignments: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    request_id: str = "request",
    cluster_start: int = 0,
):
    return store.store_clusters(
        layer_name=layer_name,
        request_id=request_id,
        cluster_start=cluster_start,
        token_keys=token_keys,
        token_values=token_values,
        assignments=assignments,
        cluster_token_counts=cluster_token_counts,
    )


def get_block_metadata(store, table, device=None):
    return store.get_cluster_block_metadata(
        layer_name="layer",
        cluster_ids=table.cluster_ids,
        device=device,
    )


def get_runtime_blocks(store, table, device):
    cluster_ids = table.cluster_ids.to(device=device)
    metadata = store.get_cluster_block_metadata(
        layer_name="layer",
        cluster_ids=cluster_ids,
        device=device,
    )
    return cluster_ids, metadata


def test_cluster_store_packs_per_head_clusters_across_pages():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    table = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    metadata = get_block_metadata(store, table)

    assert metadata.page_ids.shape == (2, 2, 2)
    assert table.cluster_ids.shape == (2, 2)
    assert table.cluster_ids.tolist() == [[0, 1], [2, 3]]
    assert metadata.page_token_counts.tolist() == [
        [[2, 1], [2, 0]],
        [[2, 0], [2, 1]],
    ]
    assert store.num_allocated_pages("layer") == 6
    assert store.num_allocated_clusters("layer") == 4

    key_pages, value_pages = store.get_page_storage("layer")
    assert key_pages.shape[1:] == (2, 1)
    assert value_pages.shape == key_pages.shape

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert token_mask[0, 0].tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert token_mask[0, 1].tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert gathered_keys[0, 1, token_mask[0, 1], 0].tolist() == [
        11.0,
        13.0,
        10.0,
        12.0,
        14.0,
    ]
    assert gathered_values[token_mask.unsqueeze(-1)].tolist() == pytest.approx(
        (gathered_keys[token_mask.unsqueeze(-1)] + 100.0).tolist()
    )
    assert not gathered_keys[~token_mask.unsqueeze(-1)].any()
    assert not gathered_values[~token_mask.unsqueeze(-1)].any()


def test_cluster_store_tracks_request_head_local_cluster_identities():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="first",
        cluster_start=4,
    )
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="second",
        cluster_start=4,
    )
    other_layer = store_cluster_data(
        store,
        "other-layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="first",
        cluster_start=4,
    )

    first_identities = store.get_cluster_identities("layer", first.cluster_ids)
    second_identities = store.get_cluster_identities("layer", second.cluster_ids)
    other_identities = store.get_cluster_identities(
        "other-layer", other_layer.cluster_ids
    )

    assert first.cluster_ids.tolist() == [[0, 1], [2, 3]]
    assert second.cluster_ids.tolist() == [[4, 5], [6, 7]]
    assert other_layer.cluster_ids.tolist() == [[0, 1], [2, 3]]
    assert first_identities == {
        0: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 0), 4),
        1: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 0), 5),
        2: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 1), 4),
        3: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 1), 5),
    }
    assert second_identities == {
        4: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 0), 4),
        5: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 0), 5),
        6: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 1), 4),
        7: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 1), 5),
    }
    assert other_identities == first_identities

    selected = torch.tensor([[3, 0, 3, -1]], dtype=torch.int64)
    selected_identities = store.get_cluster_identities("layer", selected)
    assert list(selected_identities) == [3, 0]

    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("first", 0): 3,
        RetroSpecClusterGroup("first", 1): 3,
        RetroSpecClusterGroup("second", 0): 3,
        RetroSpecClusterGroup("second", 1): 3,
    }
    assert store._group_backing_page_counts["other-layer"] == {
        RetroSpecClusterGroup("first", 0): 3,
        RetroSpecClusterGroup("first", 1): 3,
    }


def test_cluster_store_distributes_soft_targets_by_backing_page_ownership():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="first",
    )
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="second",
    )

    capacity = store._resident_target_capacity(store._layer_pools["layer"])
    assert capacity == 6
    assert store._resident_group_targets("layer", capacity) == {
        RetroSpecClusterGroup("first", 0): 2,
        RetroSpecClusterGroup("first", 1): 2,
        RetroSpecClusterGroup("second", 0): 1,
        RetroSpecClusterGroup("second", 1): 1,
    }

    store.free("layer", first)
    capacity = store._resident_target_capacity(store._layer_pools["layer"])
    assert capacity == 3
    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("second", 0): 3,
        RetroSpecClusterGroup("second", 1): 3,
    }
    assert store._resident_group_targets("layer", capacity) == {
        RetroSpecClusterGroup("second", 0): 2,
        RetroSpecClusterGroup("second", 1): 1,
    }

    store.free("layer", second)
    assert store._group_backing_page_counts["layer"] == {}
    assert store._resident_group_targets("layer", 0) == {}


def test_cluster_store_accumulates_group_pages_across_request_segments():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="request",
        cluster_start=0,
    )
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="request",
        cluster_start=2,
    )

    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("request", 0): 6,
        RetroSpecClusterGroup("request", 1): 6,
    }

    store.free("layer", first)
    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("request", 0): 3,
        RetroSpecClusterGroup("request", 1): 3,
    }

    store.free("layer", second)
    assert store._group_backing_page_counts["layer"] == {}


def test_cluster_store_rejects_drift_in_group_page_accounting():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    group = RetroSpecClusterGroup("request", 0)
    store._group_backing_page_counts["layer"][group] += 1

    with pytest.raises(RuntimeError, match="does not match the layer page pool"):
        store._resident_group_targets("layer", capacity=3)


def test_cluster_identity_rejects_negative_indices():
    with pytest.raises(ValueError, match="kv_head_index"):
        RetroSpecClusterGroup(request_id="request", kv_head_index=-1)

    with pytest.raises(ValueError, match="local_cluster_id"):
        RetroSpecClusterIdentity(
            group=RetroSpecClusterGroup("request", 0),
            local_cluster_id=-1,
        )


def test_cluster_store_frees_and_reuses_pages():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    first_metadata = get_block_metadata(store, first)
    first_page_ids = set(first_metadata.page_ids[first_metadata.page_ids >= 0].tolist())
    first_cluster_ids = set(first.cluster_ids[first.cluster_ids >= 0].tolist())
    store.free("layer", first)

    assert store.num_allocated_pages("layer") == 0
    assert store.num_allocated_clusters("layer") == 0

    second = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    second_metadata = get_block_metadata(store, second)
    second_page_ids = set(
        second_metadata.page_ids[second_metadata.page_ids >= 0].tolist()
    )
    second_cluster_ids = set(second.cluster_ids[second.cluster_ids >= 0].tolist())

    assert second_page_ids == first_page_ids
    assert second_cluster_ids.isdisjoint(first_cluster_ids)
    assert store.num_allocated_pages("layer") == 6


def test_cluster_store_releases_pages_when_packing_fails():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    invalid_counts = cluster_token_counts.clone()
    invalid_counts[0, 0] = 2

    with pytest.raises(RuntimeError, match="assignment count"):
        store_cluster_data(
            store,
            "layer",
            keys,
            values,
            assignments,
            invalid_counts,
        )

    assert store.num_allocated_pages("layer") == 0


def test_cluster_store_releases_pages_when_resident_resize_fails():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    allocated_before = store.num_allocated_pages("layer")
    clusters_before = store.num_allocated_clusters("layer")
    group_page_counts_before = dict(store._group_backing_page_counts["layer"])
    store._resident_caches["layer"] = Mock(
        resize=Mock(side_effect=RuntimeError("resident resize failed"))
    )

    with pytest.raises(RuntimeError, match="resident resize failed"):
        store_cluster_data(
            store, "layer", keys, values, assignments, cluster_token_counts
        )

    assert store.num_allocated_pages("layer") == allocated_before
    assert store.num_allocated_clusters("layer") == clusters_before
    assert store._group_backing_page_counts["layer"] == group_page_counts_before
    del store._resident_caches["layer"]
    store.free("layer", first)


def test_cluster_store_handles_empty_clusters():
    store = RetroSpecClusterPageStore(page_size=2)
    keys = torch.empty(1, 0, 3)
    values = torch.empty_like(keys)
    assignments = torch.empty(1, 0, dtype=torch.int64)
    cluster_token_counts = torch.zeros(1, 2, dtype=torch.int32)

    table = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    metadata = get_block_metadata(store, table)
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert metadata.page_ids.shape == (1, 2, 0)
    assert table.cluster_ids.tolist() == [[-1, -1]]
    assert gathered_keys.shape == (1, 1, 0, 3)
    assert gathered_values.shape == gathered_keys.shape
    assert token_mask.shape == (1, 1, 0)
    assert store.num_allocated_pages("layer") == 0
    assert store.num_allocated_clusters("layer") == 0
    assert store.get_cluster_identities("layer", table.cluster_ids) == {}


def test_cluster_store_rejects_storage_access_before_allocation():
    store = RetroSpecClusterPageStore(page_size=2)

    with pytest.raises(RuntimeError, match="No RetroSpec page pool"):
        store.get_page_storage("missing")


def test_cpu_backing_store_preserves_page_layout_and_reuses_pages():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    first_metadata = get_block_metadata(store, first)
    first_page_ids = first_metadata.page_ids.clone()
    key_pages, value_pages = store.get_page_storage("layer")

    assert store.is_cpu_backed
    assert store.get_storage_device("layer") == torch.device("cpu")
    assert first.cluster_ids.device.type == "cpu"
    assert first_metadata.page_ids.device.type == "cpu"
    assert first_metadata.page_token_counts.device.type == "cpu"
    assert key_pages.device.type == "cpu"
    assert value_pages.device.type == "cpu"

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        first_metadata.page_ids.unsqueeze(0),
        first_metadata.page_token_counts.unsqueeze(0),
    )
    assert gathered_keys.device.type == "cpu"
    assert gathered_values.device.type == "cpu"
    assert token_mask.device.type == "cpu"
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    store.free("layer", first)
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    second_metadata = get_block_metadata(store, second)

    torch.testing.assert_close(second_metadata.page_ids, first_page_ids)
    assert not torch.equal(second.cluster_ids, first.cluster_ids)


def test_cpu_backing_store_growth_preserves_existing_logical_pages():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
    )
    first_keys = torch.arange(60, dtype=torch.float32).view(1, 60, 1)
    first_values = first_keys + 1000.0
    first_assignments = torch.zeros(1, 60, dtype=torch.int64)
    first_counts = torch.tensor([[60]], dtype=torch.int32)
    first = store_cluster_data(
        store,
        "layer",
        first_keys,
        first_values,
        first_assignments,
        first_counts,
    )
    first_metadata = get_block_metadata(store, first)

    second_keys = torch.arange(100, dtype=torch.float32).view(1, 100, 1)
    second_values = second_keys + 2000.0
    second_assignments = torch.zeros(1, 100, dtype=torch.int64)
    second_counts = torch.tensor([[100]], dtype=torch.int32)
    second = store_cluster_data(
        store,
        "layer",
        second_keys,
        second_values,
        second_assignments,
        second_counts,
    )

    key_pages, value_pages = store.get_page_storage("layer")
    assert key_pages.shape[0] == 128
    assert value_pages.shape == key_pages.shape
    assert store.num_allocated_pages("layer") == 80

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        first_metadata.page_ids.unsqueeze(0),
        first_metadata.page_token_counts.unsqueeze(0),
    )
    torch.testing.assert_close(
        gathered_keys[0, 0, token_mask[0, 0]],
        first_keys[0],
    )
    torch.testing.assert_close(
        gathered_values[0, 0, token_mask[0, 0]],
        first_values[0],
    )

    store.free("layer", first)
    store.free("layer", second)
    assert store.num_allocated_pages("layer") == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA with pinned host memory is required",
)
def test_cpu_backing_store_keeps_block_metadata_on_pinned_cpu():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        pin_memory=True,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    table = store_cluster_data(
        store,
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )
    metadata = get_block_metadata(store, table)
    key_pages, value_pages = store.get_page_storage("layer")

    assert table.cluster_ids.device.type == "cpu"
    assert table.cluster_ids.is_pinned()
    assert metadata.page_ids.device.type == "cpu"
    assert metadata.page_ids.is_pinned()
    assert metadata.page_token_counts.device.type == "cpu"
    assert metadata.page_token_counts.is_pinned()
    assert key_pages.device.type == "cpu"
    assert value_pages.device.type == "cpu"
    assert key_pages.is_pinned()
    assert value_pages.is_pinned()

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )
    assert gathered_keys.device.type == "cpu"
    assert gathered_values.device.type == "cpu"
    assert token_mask.device.type == "cpu"
    assert gathered_keys[0, 1, token_mask[0, 1], 0].tolist() == [
        11.0,
        13.0,
        10.0,
        12.0,
        14.0,
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the resident cluster cache",
)
def test_cpu_backing_store_admits_and_invalidates_resident_clusters():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )
    cluster_ids, metadata = get_runtime_blocks(store, table, torch.device("cuda"))

    access = store.admit_resident_clusters("layer", cluster_ids, metadata.page_ids)
    resident_cache = store._resident_caches["layer"]

    assert store.resident_capacity("layer") == 3
    assert store.num_resident_pages("layer") == 3
    assert store.num_resident_clusters("layer") == 2
    assert store.num_resident_groups("layer") == 2
    assert resident_cache._group_targets == {
        RetroSpecClusterGroup("request", 0): 2,
        RetroSpecClusterGroup("request", 1): 1,
    }
    assert (
        resident_cache._group_states[RetroSpecClusterGroup("request", 0)].num_pages == 2
    )
    assert (
        resident_cache._group_states[RetroSpecClusterGroup("request", 1)].num_pages == 1
    )
    assert access.hit_cluster_mask.tolist() == [[True, False], [True, False]]
    assert access.miss_cluster_mask.tolist() == [[False, True], [False, True]]
    assert len(resident_cache._pending_copy_batches) == 1

    cache_keys, cache_values = store.get_resident_page_storage("layer")
    backing_keys, backing_values = store.get_page_storage("layer")
    valid = access.cache_page_ids >= 0
    slots = access.cache_page_ids[valid].to(torch.int64)
    logical_ids = metadata.page_ids[valid].cpu().to(torch.int64)
    torch.testing.assert_close(
        cache_keys.index_select(0, slots).cpu(),
        backing_keys.index_select(0, logical_ids),
    )
    torch.testing.assert_close(
        cache_values.index_select(0, slots).cpu(),
        backing_values.index_select(0, logical_ids),
    )

    store.free("layer", table)
    assert resident_cache.num_pending_copy_batches == 0
    assert store.resident_capacity("layer") == 0
    assert store.num_resident_pages("layer") == 0
    assert store.num_resident_clusters("layer") == 0
    assert store.num_resident_groups("layer") == 0
    assert resident_cache._group_targets == {}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to resolve cluster pages",
)
def test_cpu_backing_store_stages_before_updating_resident_cache():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )
    cluster_ids, metadata = get_runtime_blocks(store, table, torch.device("cuda"))

    resolved = store.resolve_cluster_blocks("layer", cluster_ids, metadata.page_ids)
    torch.cuda.current_stream().synchronize()

    valid_pages = metadata.page_ids >= 0
    resident_pages = resolved.resident_page_ids >= 0
    staging_pages = resolved.staging_page_ids >= 0

    assert torch.equal(resident_pages | staging_pages, valid_pages)
    assert not torch.any(resident_pages & staging_pages)
    assert resident_pages.sum().item() == 0
    assert staging_pages.sum().item() == 6
    assert resolved.hit_cluster_mask.tolist() == [
        [False, False],
        [False, False],
    ]
    assert resolved.miss_cluster_mask.tolist() == [
        [True, True],
        [True, True],
    ]
    assert store.num_resident_pages("layer") == 0
    assert store.num_resident_clusters("layer") == 0

    backing_keys, backing_values = store.get_page_storage("layer")
    staging_slots = resolved.staging_page_ids[staging_pages].to(torch.int64)
    staging_logical_ids = metadata.page_ids[staging_pages].cpu().to(torch.int64)

    torch.testing.assert_close(
        resolved.staging_key_pages.index_select(0, staging_slots).cpu(),
        backing_keys.index_select(0, staging_logical_ids),
    )
    torch.testing.assert_close(
        resolved.staging_value_pages.index_select(0, staging_slots).cpu(),
        backing_values.index_select(0, staging_logical_ids),
    )

    access = store.admit_staged_clusters(
        layer_name="layer",
        cluster_ids=cluster_ids,
        logical_page_ids=metadata.page_ids,
        staging_page_ids=resolved.staging_page_ids,
        staging_key_pages=resolved.staging_key_pages,
        staging_value_pages=resolved.staging_value_pages,
    )

    assert access.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert access.miss_cluster_mask.tolist() == [
        [False, True],
        [False, True],
    ]
    assert store.num_resident_pages("layer") == 3
    assert store.num_resident_clusters("layer") == 2

    resident_keys, resident_values = store.get_resident_page_storage("layer")
    resident_pages = access.cache_page_ids >= 0
    resident_slots = access.cache_page_ids[resident_pages].to(torch.int64)
    resident_logical_ids = metadata.page_ids[resident_pages].cpu().to(torch.int64)
    torch.testing.assert_close(
        resident_keys.index_select(0, resident_slots).cpu(),
        backing_keys.index_select(0, resident_logical_ids),
    )
    torch.testing.assert_close(
        resident_values.index_select(0, resident_slots).cpu(),
        backing_values.index_select(0, resident_logical_ids),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to resolve cluster pages",
)
def test_cpu_backing_store_resident_only_resolution_does_not_admit_misses():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )
    cluster_ids, metadata = get_runtime_blocks(store, table, torch.device("cuda"))

    access = store.admit_resident_clusters("layer", cluster_ids, metadata.page_ids)
    store.get_resident_page_storage("layer")
    torch.cuda.current_stream().synchronize()

    resident_pages_before = store.num_resident_pages("layer")
    resident_clusters_before = store.num_resident_clusters("layer")
    resolved = store.resolve_cluster_blocks(
        "layer",
        cluster_ids,
        metadata.page_ids,
        mode="resident_only",
    )

    assert torch.equal(
        resolved.resident_page_ids,
        access.cache_page_ids,
    )
    assert torch.all(resolved.staging_page_ids == -1)
    assert resolved.staging_key_pages.shape[0] == 0
    assert resolved.staging_value_pages.shape[0] == 0
    assert resolved.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert resolved.miss_cluster_mask.tolist() == [
        [False, True],
        [False, True],
    ]
    assert store.num_resident_pages("layer") == resident_pages_before
    assert store.num_resident_clusters("layer") == resident_clusters_before


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to resolve cluster pages",
)
def test_gpu_reference_store_resolves_without_staging():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )
    cluster_ids, metadata = get_runtime_blocks(store, table, torch.device("cuda"))

    resolved = store.resolve_cluster_blocks("layer", cluster_ids, metadata.page_ids)
    stored_keys, stored_values = store.get_page_storage("layer")

    assert torch.equal(resolved.resident_page_ids, metadata.page_ids)
    assert torch.all(resolved.staging_page_ids == -1)
    assert resolved.resident_key_pages is stored_keys
    assert resolved.resident_value_pages is stored_values
    assert resolved.staging_key_pages.shape[0] == 0
    assert resolved.staging_value_pages.shape[0] == 0
    assert resolved.hit_cluster_mask.all()
    assert not resolved.miss_cluster_mask.any()
    assert resolved.resident_ready_event is None


def test_gpu_reference_full_verification_reuses_backing_pages():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = get_block_metadata(store, table)

    resolved = store.resolve_full_verification_blocks(
        "layer", table.cluster_ids, metadata.page_ids
    )
    stored_keys, stored_values = store.get_page_storage("layer")

    assert torch.equal(resolved.resident_page_ids, metadata.page_ids)
    assert torch.all(resolved.staging_page_ids == -1)
    assert resolved.resident_key_pages is stored_keys
    assert resolved.resident_value_pages is stored_values
    assert resolved.staging_key_pages.shape[0] == 0
    assert resolved.staging_value_pages.shape[0] == 0
    assert resolved.hit_cluster_mask.all()
    assert not resolved.miss_cluster_mask.any()
    assert resolved.resident_ready_event is None
    assert resolved.staging_ready_event is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for full-verification staging",
)
def test_cpu_backing_store_reuses_full_verification_buffer_across_layers():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first_table = store_cluster_data(
        store,
        "first-layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    second_table = store_cluster_data(
        store,
        "second-layer",
        (keys + 1000).to(device),
        (values + 1000).to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )

    first_cluster_ids = first_table.cluster_ids.to(device)
    first_metadata = store.get_cluster_block_metadata(
        "first-layer", first_cluster_ids, device=device
    )
    first_resolved = store.resolve_full_verification_blocks(
        "first-layer", first_cluster_ids, first_metadata.page_ids
    )
    assert first_resolved.staging_ready_event is not None
    torch.cuda.current_stream(device).wait_event(first_resolved.staging_ready_event)

    first_valid = first_metadata.page_ids >= 0
    first_slots = first_resolved.staging_page_ids[first_valid].to(torch.int64)
    first_logical_ids = first_metadata.page_ids[first_valid].cpu().to(torch.int64)
    first_backing_keys, first_backing_values = store.get_page_storage("first-layer")
    first_key_snapshot = first_resolved.staging_key_pages.index_select(0, first_slots)
    first_value_snapshot = first_resolved.staging_value_pages.index_select(
        0, first_slots
    )

    first_key_ptr = first_resolved.staging_key_pages.data_ptr()
    first_value_ptr = first_resolved.staging_value_pages.data_ptr()
    resident_pages_before = store.num_resident_pages("first-layer")

    second_cluster_ids = second_table.cluster_ids.to(device)
    second_metadata = store.get_cluster_block_metadata(
        "second-layer", second_cluster_ids, device=device
    )
    second_resolved = store.resolve_full_verification_blocks(
        "second-layer", second_cluster_ids, second_metadata.page_ids
    )
    assert second_resolved.staging_ready_event is not None
    torch.cuda.current_stream(device).wait_event(second_resolved.staging_ready_event)

    second_valid = second_metadata.page_ids >= 0
    second_slots = second_resolved.staging_page_ids[second_valid].to(torch.int64)
    second_logical_ids = second_metadata.page_ids[second_valid].cpu().to(torch.int64)
    second_backing_keys, second_backing_values = store.get_page_storage("second-layer")
    torch.testing.assert_close(
        first_key_snapshot.cpu(),
        first_backing_keys.index_select(0, first_logical_ids),
    )
    torch.testing.assert_close(
        first_value_snapshot.cpu(),
        first_backing_values.index_select(0, first_logical_ids),
    )
    torch.testing.assert_close(
        second_resolved.staging_key_pages.index_select(0, second_slots).cpu(),
        second_backing_keys.index_select(0, second_logical_ids),
    )
    torch.testing.assert_close(
        second_resolved.staging_value_pages.index_select(0, second_slots).cpu(),
        second_backing_values.index_select(0, second_logical_ids),
    )

    assert second_resolved.staging_key_pages.data_ptr() == first_key_ptr
    assert second_resolved.staging_value_pages.data_ptr() == first_value_ptr
    assert len(store._full_verification_buffers) == 1
    assert not second_resolved.hit_cluster_mask.any()
    assert second_resolved.miss_cluster_mask.all()
    assert store.num_resident_pages("first-layer") == resident_pages_before
    assert store.num_resident_pages("second-layer") == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for full-verification staging",
)
def test_full_verification_buffer_grows_for_a_larger_layer():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
    )

    small_keys = torch.arange(4, dtype=torch.float16, device=device).view(1, 4, 1)
    small_table = store_cluster_data(
        store,
        "small-layer",
        small_keys,
        small_keys + 100,
        torch.zeros(1, 4, dtype=torch.int64, device=device),
        torch.tensor([[4]], dtype=torch.int32, device=device),
    )
    small_cluster_ids = small_table.cluster_ids.to(device)
    small_metadata = store.get_cluster_block_metadata(
        "small-layer", small_cluster_ids, device=device
    )
    small_resolved = store.resolve_full_verification_blocks(
        "small-layer", small_cluster_ids, small_metadata.page_ids
    )
    small_key_ptr = small_resolved.staging_key_pages.data_ptr()

    large_keys = torch.arange(130, dtype=torch.float16, device=device).view(1, 130, 1)
    large_table = store_cluster_data(
        store,
        "large-layer",
        large_keys,
        large_keys + 100,
        torch.zeros(1, 130, dtype=torch.int64, device=device),
        torch.tensor([[130]], dtype=torch.int32, device=device),
    )
    large_cluster_ids = large_table.cluster_ids.to(device)
    large_metadata = store.get_cluster_block_metadata(
        "large-layer", large_cluster_ids, device=device
    )
    large_resolved = store.resolve_full_verification_blocks(
        "large-layer", large_cluster_ids, large_metadata.page_ids
    )
    assert large_resolved.staging_ready_event is not None
    torch.cuda.current_stream(device).wait_event(large_resolved.staging_ready_event)

    valid_pages = large_metadata.page_ids >= 0
    staging_slots = large_resolved.staging_page_ids[valid_pages].to(torch.int64)
    logical_page_ids = large_metadata.page_ids[valid_pages].cpu().to(torch.int64)
    backing_keys, backing_values = store.get_page_storage("large-layer")

    torch.testing.assert_close(
        large_resolved.staging_key_pages.index_select(0, staging_slots).cpu(),
        backing_keys.index_select(0, logical_page_ids),
    )
    torch.testing.assert_close(
        large_resolved.staging_value_pages.index_select(0, staging_slots).cpu(),
        backing_values.index_select(0, logical_page_ids),
    )
    transfer_buffer = store._full_verification_buffers[device]
    assert transfer_buffer.capacity == 128
    assert large_resolved.staging_key_pages.data_ptr() != small_key_ptr


def test_gpu_reference_store_rejects_resident_cache_operations():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = get_block_metadata(store, table)

    with pytest.raises(RuntimeError, match="only used by CPU-backed"):
        store.lookup_resident_clusters("layer", table.cluster_ids, metadata.page_ids)

    with pytest.raises(ValueError, match="Unsupported RetroSpec cluster resolve mode"):
        store.resolve_cluster_blocks(
            "layer",
            table.cluster_ids,
            metadata.page_ids,
            mode="invalid",  # type: ignore[arg-type]
        )


def test_cpu_backing_store_stages_and_commits_cpu_inputs():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    staged = store.stage_clusters(
        token_keys=keys,
        token_values=values,
        assignments=assignments,
        cluster_token_counts=cluster_token_counts,
    )

    assert staged.ready_event is None
    assert staged.metadata_device == torch.device("cpu")
    assert "layer" not in store._layer_pools

    table = store.store_staged_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        staged=staged,
    )
    metadata = get_block_metadata(store, table)
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert store.num_allocated_pages("layer") == 6
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    torch.testing.assert_close(
        gathered_values[token_mask.unsqueeze(-1)],
        gathered_keys[token_mask.unsqueeze(-1)] + 100.0,
    )


def test_cpu_backing_store_supports_two_phase_staging():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    staged_token_kv = store.stage_token_kv(keys, values)

    assert staged_token_kv.source_device == torch.device("cpu")
    assert staged_token_kv.ready_event is None
    assert staged_token_kv.token_keys is keys
    assert staged_token_kv.token_values is values

    staged = store.finish_stage_clusters(
        staged_token_kv,
        assignments,
        cluster_token_counts,
    )

    assert staged.ready_event is None
    assert staged.metadata_device == torch.device("cpu")
    assert staged.token_keys is keys
    assert staged.token_values is values
    assert staged.assignments is assignments
    assert staged.cluster_token_counts is cluster_token_counts

    table = store.store_staged_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        staged=staged,
    )

    assert store.num_allocated_pages("layer") == 6
    assert table.cluster_ids.tolist() == [[0, 1], [2, 3]]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_asynchronously_stages_cuda_inputs():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        pin_memory=True,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    keys = keys.to(device)
    values = values.to(device)
    assignments = assignments.to(device)
    cluster_token_counts = cluster_token_counts.to(device)

    staged_token_kv = store.stage_token_kv(keys, values)

    assert staged_token_kv.ready_event is not None
    assert staged_token_kv.source_device == device
    assert staged_token_kv.token_keys.device.type == "cpu"
    assert staged_token_kv.token_values.device.type == "cpu"
    assert staged_token_kv.token_keys.is_pinned()
    assert staged_token_kv.token_values.is_pinned()

    # Enqueue work after token-KV staging. finish_stage_clusters() must make
    # metadata D2H wait for this work without serializing the earlier KV copy.
    assignments = assignments.clone()
    cluster_token_counts = cluster_token_counts.clone()
    staged = store.finish_stage_clusters(
        staged_token_kv,
        assignments,
        cluster_token_counts,
    )

    assert staged.ready_event is not None
    assert staged.metadata_device == device
    assert all(
        tensor.device.type == "cpu" and tensor.is_pinned()
        for tensor in (
            staged.token_keys,
            staged.token_values,
            staged.assignments,
            staged.cluster_token_counts,
        )
    )
    assert "layer" not in store._layer_pools

    table = store.store_staged_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        staged=staged,
    )
    pool = store._layer_pools["layer"]
    metadata = store.get_cluster_block_metadata(
        "layer",
        table.cluster_ids,
        device=torch.device("cpu"),
    )
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert pool.storage_device == torch.device("cpu")
    assert pool.metadata_device == device
    assert pool.key_pages.is_pinned()
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    torch.testing.assert_close(
        gathered_values[token_mask.unsqueeze(-1)],
        gathered_keys[token_mask.unsqueeze(-1)] + 100.0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_reuses_pinned_staging_slot_after_build():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        pin_memory=True,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    keys = keys.to(device)
    values = values.to(device)
    assignments = assignments.to(device)
    cluster_token_counts = cluster_token_counts.to(device)

    first = store.stage_clusters(
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    first_slot = first.staging_slot
    assert first_slot is not None
    first_pointers = (
        first.token_keys.data_ptr(),
        first.token_values.data_ptr(),
        first.assignments.data_ptr(),
        first.cluster_token_counts.data_ptr(),
    )

    store.store_staged_clusters("first-layer", "request", 0, first)

    assert not first_slot.in_use

    second = store.stage_clusters(
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    second_pointers = (
        second.token_keys.data_ptr(),
        second.token_values.data_ptr(),
        second.assignments.data_ptr(),
        second.cluster_token_counts.data_ptr(),
    )

    assert second.staging_slot is first_slot
    assert second_pointers == first_pointers
    assert len(store._pinned_staging_slots[device]) == 1

    store.store_staged_clusters("second-layer", "request", 0, second)
    assert not first_slot.in_use


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_does_not_reuse_busy_pinned_staging_slot():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        pin_memory=True,
    )
    keys, values, _, _ = make_cluster_data()
    keys = keys.to(device)
    values = values.to(device)

    first = store.stage_token_kv(keys, values)
    second = store.stage_token_kv(keys, values)

    assert first.staging_slot is not None
    assert second.staging_slot is not None
    assert second.staging_slot is not first.staging_slot
    assert len(store._pinned_staging_slots[device]) == 2

    store.discard_staged_token_kv(first)
    store.discard_staged_token_kv(second)

    assert not first.staging_slot.in_use
    assert not second.staging_slot.in_use


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_grows_reused_pinned_staging_slot():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        pin_memory=True,
    )
    small_keys = torch.arange(4, dtype=torch.float32, device=device).view(1, 4, 1)
    small_values = small_keys + 10
    small = store.stage_token_kv(small_keys, small_values)
    slot = small.staging_slot
    assert slot is not None
    assert slot.token_key_storage is not None
    old_capacity = slot.token_key_storage.numel()
    store.discard_staged_token_kv(small)

    large_keys = torch.arange(12, dtype=torch.float32, device=device).view(1, 12, 1)
    large_values = large_keys + 20
    large = store.stage_token_kv(large_keys, large_values)
    large.wait()

    assert large.staging_slot is slot
    assert slot.token_key_storage is not None
    assert slot.token_key_storage.numel() == large_keys.numel()
    assert slot.token_key_storage.numel() > old_capacity
    torch.testing.assert_close(large.token_keys, large_keys.cpu())
    torch.testing.assert_close(large.token_values, large_values.cpu())

    store.discard_staged_token_kv(large)
    assert not slot.in_use


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_releases_pinned_slot_when_build_fails(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        pin_memory=True,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    staged = store.stage_clusters(
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    slot = staged.staging_slot
    assert slot is not None

    monkeypatch.setattr(
        store,
        "store_clusters",
        Mock(side_effect=RuntimeError("cluster build failed")),
    )

    with pytest.raises(RuntimeError, match="cluster build failed"):
        store.store_staged_clusters("layer", "request", 0, staged)

    assert not slot.in_use


def test_cluster_store_rejects_invalid_storage_metadata():
    with pytest.raises(ValueError, match="Unsupported"):
        RetroSpecClusterPageStore(
            page_size=2,
            storage_mode="invalid",  # type: ignore[arg-type]
        )

    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    metadata = get_block_metadata(store, table)

    invalid_page_ids = metadata.page_ids.unsqueeze(0).clone()
    invalid_page_ids[0, 0, 0, 0] = -2
    with pytest.raises(ValueError, match="at least -1"):
        store.gather_pages(
            "layer",
            invalid_page_ids,
            metadata.page_token_counts.unsqueeze(0),
        )

    negative_counts = cluster_token_counts.clone()
    negative_counts[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        store_cluster_data(
            store,
            "other",
            keys,
            values,
            assignments,
            negative_counts,
        )

    with pytest.raises(ValueError, match="cluster_start"):
        store_cluster_data(
            store,
            "other",
            keys,
            values,
            assignments,
            cluster_token_counts,
            cluster_start=-1,
        )


def test_cluster_store_rejects_cluster_id_page_descriptor_mismatch():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = get_block_metadata(store, table)

    mismatched_page_ids = metadata.page_ids.clone()
    mismatched_page_ids[0, 0] = metadata.page_ids[0, 1]

    with pytest.raises(RuntimeError, match="stable cluster ID"):
        store.resolve_cluster_blocks("layer", table.cluster_ids, mismatched_page_ids)


def test_cluster_store_rejects_released_cluster_id_after_page_reuse():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    first_metadata = get_block_metadata(store, first)
    stale_cluster_ids = first.cluster_ids.clone()
    reused_page_ids = first_metadata.page_ids.clone()
    store.free("layer", first)

    second = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    second_metadata = get_block_metadata(store, second)
    torch.testing.assert_close(second_metadata.page_ids, reused_page_ids)

    with pytest.raises(RuntimeError, match="not allocated"):
        store.resolve_cluster_blocks(
            "layer", stale_cluster_ids, second_metadata.page_ids
        )


def test_cluster_store_materializes_selected_cpu_block_metadata():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    selected_cluster_ids = torch.tensor(
        [[table.cluster_ids[0, 0], -1], [table.cluster_ids[1, 1], -1]],
        dtype=torch.int64,
    )

    metadata = store.get_cluster_block_metadata("layer", selected_cluster_ids)

    assert metadata.page_ids.device.type == "cpu"
    assert metadata.page_ids.shape == (2, 2, 2)
    assert metadata.page_token_counts.tolist() == [
        [[2, 1], [0, 0]],
        [[2, 1], [0, 0]],
    ]
    assert store.max_pages_per_cluster("layer", selected_cluster_ids) == 2


@pytest.mark.parametrize("page_size", [0, -1])
def test_cluster_store_rejects_non_positive_page_size(page_size):
    with pytest.raises(ValueError, match="positive"):
        RetroSpecClusterPageStore(page_size)
