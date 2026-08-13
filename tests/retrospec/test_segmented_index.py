# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)


def make_index(
    segment_size_tokens: int = 4,
    blocks_per_cluster: int = 1,
) -> RetroSpecSegmentedTokenIndex:
    return RetroSpecSegmentedTokenIndex(
        block_size=2,
        num_speculative_tokens=1,
        retrieval_ratio=0.5,
        estimation_ratio=0.5,
        segment_size_tokens=segment_size_tokens,
        blocks_per_cluster=blocks_per_cluster,
        num_kmeans_iterations=2,
    )


def make_cache(
    num_blocks: int = 8,
    num_kv_heads: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.empty(num_blocks, 2, num_kv_heads, 1)
    values = torch.empty_like(keys)
    for block_id in range(num_blocks):
        keys[block_id].fill_(float(block_id))
        values[block_id].fill_(float(block_id * 10))
    return keys, values


def build_index(
    index: RetroSpecSegmentedTokenIndex,
    seq_len: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    index.build_or_update(
        layer_name="layer",
        request_ids=["request"],
        seq_lens=[seq_len],
        rows=[0],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )


def materialize_reference(
    index: RetroSpecSegmentedTokenIndex,
    selection,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return index.materialize_exact_reference(
        selection,
        keys,
        values,
        block_table,
    )


def test_segmented_index_builds_and_reuses_sparse_selection_plan():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    assert not index.needs_update("request", 10, ["layer"])
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 6
    assert record.num_clusters == 2
    assert len(record.segments) == 1

    segment = record.segments[0]
    assert (segment.indexed_start, segment.indexed_end) == (2, 6)
    assert segment.cluster_token_counts.tolist() == [[2, 2]]
    assert segment.cluster_pages.page_ids.shape == (1, 2, 1)
    assert segment.cluster_pages.page_token_counts.tolist() == [[[2], [2]]]
    assert index.cluster_store.num_allocated_pages("layer") == 2

    index.begin_proposal(["request"])
    try:
        sparse = index.select_segmented(
            request_ids=["request"],
            layer_name="layer",
            query=torch.ones(1, 1, 1),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([10], dtype=torch.int32),
            active_mask=torch.tensor([True]),
            scale=1.0,
        )
        expanded = index.materialize(
            sparse.plan,
            RetroSpecAttentionLevel.EXPANDED,
            keys,
            values,
            block_table,
        )
    finally:
        index.end_proposal()

    sparse_keys, sparse_values, _ = materialize_reference(
        index, sparse, keys, values, block_table
    )
    expanded_keys, _, _ = materialize_reference(
        index, expanded, keys, values, block_table
    )

    assert sparse.exact_token_counts.tolist() == [[8]]
    assert sparse_keys[0, 0, :, 0].tolist() == pytest.approx(
        [0.0, 0.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0]
    )
    assert sparse_values[0, 0, :, 0].tolist() == pytest.approx(
        [0.0, 0.0, 30.0, 30.0, 40.0, 40.0, 20.0, 20.0]
    )
    assert sparse.estimation_token_counts[0, 0, 0].item() == 2
    assert sparse.estimation_keys[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert sparse.estimation_values[0, 0, 0, 0].item() == pytest.approx(10.0)

    assert expanded.exact_token_counts.tolist() == [[10]]
    assert expanded_keys[0, 0, :, 0].tolist() == pytest.approx(
        [0.0, 0.0, 3.0, 3.0, 4.0, 4.0, 1.0, 1.0, 2.0, 2.0]
    )
    assert torch.count_nonzero(expanded.estimation_token_counts) == 0
    assert expanded.attention_mass.item() >= sparse.attention_mass.item()


def test_segmented_index_clusters_each_kv_head_independently():
    index = make_index()
    keys, values = make_cache(num_kv_heads=2)
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    # Indexed logical tokens are 2..5. Head zero separates the two blocks,
    # while head one has alternating features and therefore a different
    # token-to-cluster assignment.
    keys[1, :, 0, 0] = torch.tensor([1.0, 1.0])
    keys[2, :, 0, 0] = torch.tensor([-1.0, -1.0])
    keys[1, :, 1, 0] = torch.tensor([1.0, -1.0])
    keys[2, :, 1, 0] = torch.tensor([1.0, -1.0])

    build_index(index, 10, keys, values, block_table)
    segment = index._indices["layer"]["request"].segments[0]

    assert segment.cluster_token_counts.tolist() == [[2, 2], [2, 2]]
    clustered_keys, clustered_values, clustered_mask = index.cluster_store.gather_pages(
        "layer",
        segment.cluster_pages.page_ids.unsqueeze(0),
        segment.cluster_pages.page_token_counts.unsqueeze(0),
    )

    assert clustered_mask.all()
    assert clustered_keys[0, 0, :, 0].tolist() == [1.0, 1.0, -1.0, -1.0]
    assert clustered_keys[0, 1, :, 0].tolist() == [1.0, 1.0, -1.0, -1.0]
    assert clustered_values[0, 0, :, 0].tolist() == [10.0, 10.0, 20.0, 20.0]
    assert clustered_values[0, 1, :, 0].tolist() == [10.0, 20.0, 10.0, 20.0]


def test_segmented_index_excludes_empty_clusters_from_selection():
    index = make_index(segment_size_tokens=8, blocks_per_cluster=2)
    keys = torch.ones(8, 2, 1, 1)
    values = torch.ones_like(keys)
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 14, keys, values, block_table)

    packed = index._pack_indices("layer", ["request"], keys, block_table)

    assert packed.cluster_token_counts.tolist() == [[[8, 0]]]
    assert packed.cluster_mask.tolist() == [[[True, False]]]


def test_segmented_index_handles_mixed_long_and_short_requests():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["long", "short"],
        seq_lens=[10, 3],
        rows=[0, 1],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    index.begin_proposal(["long", "short"])
    try:
        selection = index.select_segmented(
            request_ids=["long", "short"],
            layer_name="layer",
            query=torch.ones(2, 1, 1),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([10, 3], dtype=torch.int32),
            active_mask=torch.tensor([True, False]),
            scale=1.0,
        )
    finally:
        index.end_proposal()

    _, _, exact_token_mask = materialize_reference(
        index, selection, keys, values, block_table
    )

    assert selection.exact_token_counts.tolist() == [[8], [3]]
    assert exact_token_mask[1, 0, :3].all()
    assert torch.count_nonzero(selection.estimation_token_counts[1]) == 0
    assert selection.attention_mass.tolist()[1] == pytest.approx(1.0)


def test_segmented_index_appends_complete_segments_and_handles_rollback():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)
    first_segment = index._indices["layer"]["request"].segments[0]
    assert index.needs_update("request", 14, ["layer"])

    build_index(index, 14, keys, values, block_table)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 10
    assert record.num_clusters == 4
    assert len(record.segments) == 2
    assert record.segments[0] is first_segment
    assert [
        (segment.indexed_start, segment.indexed_end) for segment in record.segments
    ] == [(2, 6), (6, 10)]
    assert [segment.cluster_start for segment in record.segments] == [0, 2]
    assert index.cluster_store.num_allocated_pages("layer") == 4

    assert index.needs_update("request", 6, ["layer"])
    build_index(index, 6, keys, values, block_table)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 2
    assert record.num_clusters == 0
    assert record.segments == []
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_rollback_invalidates_packed_page_ids_before_rebuild(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 14, keys, values, block_table)
    index._pack_indices("layer", ["request"], keys, block_table)

    def fail_clustering(**_kwargs):
        raise RuntimeError("clustering failed")

    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.segmented_index.segmented_kmeans_assignments",
        fail_clustering,
    )

    with pytest.raises(RuntimeError, match="clustering failed"):
        build_index(index, 10, keys, values, block_table)

    assert "layer" not in index._packed_index_cache
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_indexed_tokens_are_materialized_from_secondary_pages():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.begin_proposal(["request"])
    try:
        sparse = index.select_segmented(
            request_ids=["request"],
            layer_name="layer",
            query=torch.ones(1, 1, 1),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([10], dtype=torch.int32),
            active_mask=torch.tensor([True]),
            scale=1.0,
        )

        # The indexed source blocks may be recycled after the prefill copy.
        # Expanded materialization must still read their original KV from the
        # private cluster store, not from the primary cache.
        keys[1:3].fill_(99.0)
        values[1:3].fill_(999.0)
        expanded = index.materialize(
            sparse.plan,
            RetroSpecAttentionLevel.EXPANDED,
            keys,
            values,
            block_table,
        )
    finally:
        index.end_proposal()

    expanded_keys, expanded_values, _ = materialize_reference(
        index, expanded, keys, values, block_table
    )

    assert expanded_keys[0, 0, -4:, 0].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert expanded_values[0, 0, -4:, 0].tolist() == [
        10.0,
        10.0,
        20.0,
        20.0,
    ]


def test_segmented_index_removes_finished_request_state():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.remove_requests(["request"])

    assert "request" not in index._indices["layer"]
    assert index.needs_update("request", 10, ["layer"])
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_segmented_index_reuses_packed_index_until_update():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    first = index._pack_indices("layer", ["request"], keys, block_table)
    second = index._pack_indices("layer", ["request"], keys, block_table)
    assert second is first

    build_index(index, 14, keys, values, block_table)
    after_update = index._pack_indices("layer", ["request"], keys, block_table)

    assert after_update is not first


def test_removing_request_invalidates_packed_index():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    packed = index._pack_indices("layer", ["request"], keys, block_table)
    assert "layer" in index._packed_index_cache

    index.remove_requests(["request"])

    assert "request" not in index._indices["layer"]
    assert "layer" not in index._packed_index_cache

    rebuilt = index._pack_indices("layer", ["request"], keys, block_table)
    assert rebuilt is not packed
    assert not rebuilt.cluster_mask.any()


def test_packed_index_cache_tracks_request_order_and_block_table_width():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["long", "short"],
        seq_lens=[10, 3],
        rows=[0, 1],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    original = index._pack_indices("layer", ["long", "short"], keys, block_table)
    reordered = index._pack_indices("layer", ["short", "long"], keys, block_table)
    narrower = index._pack_indices(
        "layer",
        ["long", "short"],
        keys,
        block_table[:, :5],
    )

    assert reordered is not original
    assert not reordered.cluster_mask[0].any()
    assert reordered.cluster_mask[1].any()
    assert narrower is not reordered
    assert narrower.indexed_token_mask.shape == (2, 10)


def test_segmented_index_proposal_lifecycle_tracks_empty_batches():
    index = make_index()

    index.begin_proposal([])
    with pytest.raises(RuntimeError, match="already active"):
        index.begin_proposal([])
    index.end_proposal()

    with pytest.raises(RuntimeError, match="not active"):
        index.end_proposal()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_segmented_index_builds_and_selects_on_cuda():
    device = torch.device("cuda")
    index = make_index()
    keys, values = make_cache()
    keys = keys.to(device=device, dtype=torch.bfloat16)
    values = values.to(device=device, dtype=torch.bfloat16)
    block_table = torch.arange(
        7,
        dtype=torch.int32,
        device=device,
    ).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.begin_proposal(["request"])
    try:
        selection = index.select_segmented(
            request_ids=["request"],
            layer_name="layer",
            query=torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([10], dtype=torch.int32, device=device),
            active_mask=torch.tensor([True], device=device),
            scale=1.0,
        )
        torch.cuda.synchronize()
    finally:
        index.end_proposal()

    assert selection.exact_page_ids.device.type == "cuda"
    assert selection.exact_token_counts.tolist() == [[8]]
    assert selection.estimation_token_counts[0, 0, 0].item() == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"segment_size_tokens": 3}, "divisible by block_size"),
        ({"segment_size_tokens": 4, "blocks_per_cluster": 3}, "divisible"),
        ({"blocks_per_cluster": 0}, "positive"),
        ({"num_kmeans_iterations": 0}, "positive"),
    ],
)
def test_segmented_index_rejects_invalid_configuration(kwargs, message):
    values = {
        "block_size": 2,
        "num_speculative_tokens": 1,
        "retrieval_ratio": 0.5,
        "estimation_ratio": 0.5,
        "segment_size_tokens": 4,
        "blocks_per_cluster": 1,
        "num_kmeans_iterations": 2,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        RetroSpecSegmentedTokenIndex(**values)
