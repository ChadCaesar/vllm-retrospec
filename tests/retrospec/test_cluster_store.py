# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.utils.platform_utils import is_pin_memory_available
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


def test_cluster_store_packs_per_head_clusters_across_pages():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    table = store.store_clusters(
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )

    assert table.page_ids.shape == (2, 2, 2)
    assert table.page_token_counts.tolist() == [
        [[2, 1], [2, 0]],
        [[2, 0], [2, 1]],
    ]
    assert store.num_allocated_pages("layer") == 6

    key_pages, value_pages = store.get_page_storage("layer")
    assert key_pages.shape[1:] == (2, 1)
    assert value_pages.shape == key_pages.shape

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        table.page_ids.unsqueeze(0),
        table.page_token_counts.unsqueeze(0),
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


def test_cluster_store_frees_and_reuses_pages():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store.store_clusters(
        "layer", keys, values, assignments, cluster_token_counts
    )
    first_page_ids = set(first.page_ids[first.page_ids >= 0].tolist())
    store.free("layer", first)

    assert store.num_allocated_pages("layer") == 0

    second = store.store_clusters(
        "layer", keys, values, assignments, cluster_token_counts
    )
    second_page_ids = set(second.page_ids[second.page_ids >= 0].tolist())

    assert second_page_ids == first_page_ids
    assert store.num_allocated_pages("layer") == 6


def test_cluster_store_releases_pages_when_packing_fails():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    invalid_counts = cluster_token_counts.clone()
    invalid_counts[0, 0] = 2

    with pytest.raises(RuntimeError, match="assignment count"):
        store.store_clusters(
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
    first = store.store_clusters(
        "layer", keys, values, assignments, cluster_token_counts
    )
    allocated_before = store.num_allocated_pages("layer")
    store._resident_caches["layer"] = Mock(
        resize=Mock(side_effect=RuntimeError("resident resize failed"))
    )

    with pytest.raises(RuntimeError, match="resident resize failed"):
        store.store_clusters("layer", keys, values, assignments, cluster_token_counts)

    assert store.num_allocated_pages("layer") == allocated_before
    del store._resident_caches["layer"]
    store.free("layer", first)


def test_cluster_store_handles_empty_clusters():
    store = RetroSpecClusterPageStore(page_size=2)
    keys = torch.empty(1, 0, 3)
    values = torch.empty_like(keys)
    assignments = torch.empty(1, 0, dtype=torch.int64)
    cluster_token_counts = torch.zeros(1, 2, dtype=torch.int32)

    table = store.store_clusters(
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        table.page_ids.unsqueeze(0),
        table.page_token_counts.unsqueeze(0),
    )

    assert table.page_ids.shape == (1, 2, 0)
    assert gathered_keys.shape == (1, 1, 0, 3)
    assert gathered_values.shape == gathered_keys.shape
    assert token_mask.shape == (1, 1, 0)
    assert store.num_allocated_pages("layer") == 0


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

    first = store.store_clusters(
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    first_page_ids = first.page_ids.clone()
    key_pages, value_pages = store.get_page_storage("layer")

    assert store.is_cpu_backed
    assert store.get_storage_device("layer") == torch.device("cpu")
    assert first.page_ids.device == keys.device
    assert first.page_token_counts.device == keys.device
    assert key_pages.device.type == "cpu"
    assert value_pages.device.type == "cpu"

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        first.page_ids.unsqueeze(0),
        first.page_token_counts.unsqueeze(0),
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
    second = store.store_clusters(
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )

    torch.testing.assert_close(second.page_ids, first_page_ids)


def test_cpu_backing_store_growth_preserves_existing_logical_pages():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
    )
    first_keys = torch.arange(60, dtype=torch.float32).view(1, 60, 1)
    first_values = first_keys + 1000.0
    first_assignments = torch.zeros(1, 60, dtype=torch.int64)
    first_counts = torch.tensor([[60]], dtype=torch.int32)
    first = store.store_clusters(
        "layer",
        first_keys,
        first_values,
        first_assignments,
        first_counts,
    )

    second_keys = torch.arange(100, dtype=torch.float32).view(1, 100, 1)
    second_values = second_keys + 2000.0
    second_assignments = torch.zeros(1, 100, dtype=torch.int64)
    second_counts = torch.tensor([[100]], dtype=torch.int32)
    second = store.store_clusters(
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
        first.page_ids.unsqueeze(0),
        first.page_token_counts.unsqueeze(0),
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
def test_cpu_backing_store_keeps_metadata_on_cuda_and_pages_pinned():
    store = RetroSpecClusterPageStore(
        page_size=2,
        storage_mode="cpu_offload",
        pin_memory=True,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    table = store.store_clusters(
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )
    key_pages, value_pages = store.get_page_storage("layer")

    assert table.page_ids.is_cuda
    assert table.page_token_counts.is_cuda
    assert key_pages.device.type == "cpu"
    assert value_pages.device.type == "cpu"
    assert key_pages.is_pinned()
    assert value_pages.is_pinned()

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        table.page_ids.unsqueeze(0),
        table.page_token_counts.unsqueeze(0),
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
    table = store.store_clusters(
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )

    access = store.admit_resident_clusters("layer", table.page_ids)

    assert store.resident_capacity("layer") == 3
    assert store.num_resident_pages("layer") == 3
    assert store.num_resident_clusters("layer") == 2
    assert access.hit_cluster_mask.tolist() == [[True, True], [False, False]]
    assert access.miss_cluster_mask.tolist() == [[False, False], [True, True]]

    cache_keys, cache_values = store.get_resident_page_storage("layer")
    backing_keys, backing_values = store.get_page_storage("layer")
    valid = access.cache_page_ids >= 0
    slots = access.cache_page_ids[valid].to(torch.int64)
    logical_ids = table.page_ids[valid].cpu().to(torch.int64)
    torch.testing.assert_close(
        cache_keys.index_select(0, slots).cpu(),
        backing_keys.index_select(0, logical_ids),
    )
    torch.testing.assert_close(
        cache_values.index_select(0, slots).cpu(),
        backing_values.index_select(0, logical_ids),
    )

    store.free("layer", table)
    assert store.resident_capacity("layer") == 0
    assert store.num_resident_pages("layer") == 0
    assert store.num_resident_clusters("layer") == 0


def test_gpu_reference_store_rejects_resident_cache_operations():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store.store_clusters(
        "layer", keys, values, assignments, cluster_token_counts
    )

    with pytest.raises(RuntimeError, match="only used by CPU-backed"):
        store.lookup_resident_pages("layer", table.page_ids)


def test_cluster_store_rejects_invalid_storage_metadata():
    with pytest.raises(ValueError, match="Unsupported"):
        RetroSpecClusterPageStore(
            page_size=2,
            storage_mode="invalid",  # type: ignore[arg-type]
        )

    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store.store_clusters(
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )

    invalid_page_ids = table.page_ids.unsqueeze(0).clone()
    invalid_page_ids[0, 0, 0, 0] = -2
    with pytest.raises(ValueError, match="at least -1"):
        store.gather_pages(
            "layer",
            invalid_page_ids,
            table.page_token_counts.unsqueeze(0),
        )

    negative_counts = cluster_token_counts.clone()
    negative_counts[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        store.store_clusters(
            "other",
            keys,
            values,
            assignments,
            negative_counts,
        )


@pytest.mark.parametrize("page_size", [0, -1])
def test_cluster_store_rejects_non_positive_page_size(page_size):
    with pytest.raises(ValueError, match="positive"):
        RetroSpecClusterPageStore(page_size)
