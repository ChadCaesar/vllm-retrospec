# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedBlockIndex,
)


def make_index(
    segment_size_tokens: int = 4,
    blocks_per_cluster: int = 1,
) -> RetroSpecSegmentedBlockIndex:
    return RetroSpecSegmentedBlockIndex(
        block_size=2,
        num_speculative_tokens=1,
        retrieval_ratio=0.5,
        estimation_ratio=0.5,
        segment_size_tokens=segment_size_tokens,
        blocks_per_cluster=blocks_per_cluster,
        num_kmeans_iterations=2,
    )


def make_cache(num_blocks: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.empty(num_blocks, 2, 1, 1)
    values = torch.empty_like(keys)
    for block_id in range(num_blocks):
        keys[block_id].fill_(float(block_id))
        values[block_id].fill_(float(block_id * 10))
    return keys, values


def build_index(
    index: RetroSpecSegmentedBlockIndex,
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


def test_segmented_index_builds_and_reuses_sparse_selection_plan():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    assert not index.needs_update("request", 10, ["layer"])
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 3
    assert record.num_clusters == 2
    assert len(record.segments) == 1

    segment = record.segments[0]
    assert segment.logical_block_ids.tolist() == [1, 2]
    assert segment.block_cluster_ids.tolist() == [0, 1]
    assert segment.cluster_token_counts.tolist() == [2, 2]

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

    assert sparse.exact_block_table[0, :4].tolist() == [0, 2, 3, 4]
    assert sparse.exact_seq_lens.tolist() == [8]
    assert sparse.estimation_token_counts[0, 0].item() == 2
    assert sparse.estimation_keys[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert sparse.estimation_values[0, 0, 0, 0].item() == pytest.approx(10.0)

    assert expanded.exact_block_table[0, :5].tolist() == [0, 1, 2, 3, 4]
    assert expanded.exact_seq_lens.tolist() == [10]
    assert torch.count_nonzero(expanded.estimation_token_counts) == 0
    assert expanded.attention_mass.item() >= sparse.attention_mass.item()


def test_segmented_index_excludes_empty_clusters_from_selection():
    index = make_index(segment_size_tokens=8, blocks_per_cluster=2)
    keys = torch.ones(8, 2, 1, 1)
    values = torch.ones_like(keys)
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 14, keys, values, block_table)

    packed = index._pack_indices("layer", ["request"], keys, block_table)

    assert packed.cluster_token_counts.tolist() == [[8, 0]]
    assert packed.cluster_mask.tolist() == [[True, False]]


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

    assert selection.exact_seq_lens.tolist() == [8, 3]
    assert selection.exact_block_table[1, :2].tolist() == [0, 1]
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
    assert record.indexed_end == 5
    assert record.num_clusters == 4
    assert len(record.segments) == 2
    assert record.segments[0] is first_segment

    logical_block_ids = torch.cat(
        [segment.logical_block_ids for segment in record.segments]
    )
    block_cluster_ids = torch.cat(
        [segment.block_cluster_ids for segment in record.segments]
    )
    assert logical_block_ids.tolist() == [1, 2, 3, 4]
    assert block_cluster_ids.tolist() == [0, 1, 2, 3]

    assert index.needs_update("request", 6, ["layer"])
    build_index(index, 6, keys, values, block_table)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 1
    assert record.num_clusters == 0
    assert record.segments == []


def test_segmented_index_removes_finished_request_state():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.remove_requests(["request"])

    assert "request" not in index._indices["layer"]
    assert index.needs_update("request", 10, ["layer"])


def test_segmented_index_reuses_packed_index_until_update():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.begin_proposal(["request"])
    try:
        first = index._pack_indices("layer", ["request"], keys, block_table)
    finally:
        index.end_proposal()

    index.begin_proposal(["request"])
    try:
        second = index._pack_indices("layer", ["request"], keys, block_table)
    finally:
        index.end_proposal()

    assert second is first

    build_index(index, 14, keys, values, block_table)

    index.begin_proposal(["request"])
    try:
        after_update = index._pack_indices("layer", ["request"], keys, block_table)
    finally:
        index.end_proposal()

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
    narrower = index._pack_indices("layer", ["long", "short"], keys, block_table[:, :5])

    assert reordered is not original
    assert not reordered.cluster_mask[0].any()
    assert reordered.cluster_mask[1].any()
    assert narrower is not reordered
    assert narrower.indexed_block_mask.shape == (2, 5)


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
    block_table = torch.arange(7, dtype=torch.int32, device=device).view(1, -1)
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

    assert selection.exact_block_table.device.type == "cuda"
    assert selection.exact_seq_lens.tolist() == [8]
    assert selection.estimation_token_counts[0, 0].item() == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"segment_size_tokens": 3}, "divisible by block_size"),
        ({"segment_size_tokens": 4, "blocks_per_cluster": 3}, "divisible"),
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
        RetroSpecSegmentedBlockIndex(**values)
