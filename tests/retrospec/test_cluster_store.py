# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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


@pytest.mark.parametrize("page_size", [0, -1])
def test_cluster_store_rejects_non_positive_page_size(page_size):
    with pytest.raises(ValueError, match="positive"):
        RetroSpecClusterPageStore(page_size)
