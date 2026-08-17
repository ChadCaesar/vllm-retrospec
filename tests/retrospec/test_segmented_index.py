# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterStorageMode,
)
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)


def make_index(
    segment_size_tokens: int = 4,
    blocks_per_cluster: int = 1,
    retrieval_ratio: float = 0.5,
    estimation_ratio: float = 0.5,
    cache_mode: RetroSpecClusterStorageMode = "gpu_reference",
    cache_ratio: float = 0.0,
    pin_memory: bool = False,
) -> RetroSpecSegmentedTokenIndex:
    return RetroSpecSegmentedTokenIndex(
        block_size=2,
        num_speculative_tokens=1,
        retrieval_ratio=retrieval_ratio,
        estimation_ratio=estimation_ratio,
        segment_size_tokens=segment_size_tokens,
        blocks_per_cluster=blocks_per_cluster,
        num_kmeans_iterations=2,
        cache_mode=cache_mode,
        cache_ratio=cache_ratio,
        pin_memory=pin_memory,
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


@pytest.mark.parametrize(
    ("cache_mode", "cache_ratio", "expected_cache_ratio"),
    [
        ("gpu_reference", 0.0, 0.0),
        ("cpu_offload", 0.0, 0.6),
        ("cpu_offload", 0.35, 0.35),
    ],
)
def test_segmented_index_configures_resident_cache_ratio(
    cache_mode: RetroSpecClusterStorageMode,
    cache_ratio: float,
    expected_cache_ratio: float,
):
    index = make_index(
        retrieval_ratio=0.2,
        cache_mode=cache_mode,
        cache_ratio=cache_ratio,
    )

    assert index.cluster_store.cache_ratio == pytest.approx(expected_cache_ratio)


def build_index(
    index: RetroSpecSegmentedTokenIndex,
    seq_len: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
    defer_cpu_store: bool = False,
) -> None:
    index.build_or_update(
        layer_name="layer",
        request_ids=["request"],
        seq_lens=[seq_len],
        rows=[0],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
        defer_cpu_store=defer_cpu_store,
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
    metadata = index.cluster_store.get_cluster_block_metadata(
        "layer", segment.cluster_blocks.cluster_ids
    )
    assert (segment.indexed_start, segment.indexed_end) == (2, 6)
    assert segment.cluster_token_counts.tolist() == [[2, 2]]
    assert segment.cluster_blocks.cluster_ids.tolist() == [[0, 1]]
    assert segment.cluster_blocks.cluster_ids.device.type == "cpu"
    assert not hasattr(segment.cluster_blocks, "page_ids")
    assert metadata.page_ids.shape == (1, 2, 1)
    assert metadata.page_token_counts.tolist() == [[[2], [2]]]
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

    sparse_keys, sparse_values, sparse_mask = materialize_reference(
        index, sparse, keys, values, block_table
    )
    expanded_keys, _, expanded_mask = materialize_reference(
        index, expanded, keys, values, block_table
    )

    sparse_keys = sparse_keys[0, 0, sparse_mask[0, 0], 0]
    sparse_values = sparse_values[0, 0, sparse_mask[0, 0], 0]
    expanded_keys = expanded_keys[0, 0, expanded_mask[0, 0], 0]

    assert sparse.exact_token_counts.tolist() == [[8]]
    assert sparse_keys.tolist() == pytest.approx(
        [0.0, 0.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0]
    )
    assert sparse_values.tolist() == pytest.approx(
        [0.0, 0.0, 30.0, 30.0, 40.0, 40.0, 20.0, 20.0]
    )
    assert sparse.estimation_token_counts[0, 0, 0].item() == 2
    assert sparse.estimation_keys[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert sparse.estimation_values[0, 0, 0, 0].item() == pytest.approx(10.0)

    assert expanded.exact_token_counts.tolist() == [[10]]
    assert expanded_keys.tolist() == pytest.approx(
        [0.0, 0.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0]
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
    metadata = index.cluster_store.get_cluster_block_metadata(
        "layer", segment.cluster_blocks.cluster_ids
    )

    assert segment.cluster_token_counts.tolist() == [[2, 2], [2, 2]]
    clustered_keys, clustered_values, clustered_mask = index.cluster_store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
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
    assert packed.cluster_ids[0, 0, 0].item() >= 0
    assert packed.cluster_ids[0, 0, 1].item() == -1


@pytest.mark.parametrize(
    ("retrieval_ratio", "estimation_ratio"),
    [(0.3, 0.4), (0.5, 0.5), (0.7, 0.0)],
)
def test_compact_cluster_zones_match_full_mask_selection(
    retrieval_ratio: float,
    estimation_ratio: float,
):
    index = make_index(
        retrieval_ratio=retrieval_ratio,
        estimation_ratio=estimation_ratio,
    )
    cluster_scores = torch.tensor(
        [
            [
                [0.12, 0.91, 0.33, 0.74, 0.28, 0.65, 0.47],
                [0.82, 0.13, 0.71, 0.24, 0.63, 0.35, 0.56],
                [0.42, 0.73, 0.14, 0.85, 0.26, 0.97, 0.38],
            ],
            [
                [0.19, 0.81, 0.32, 0.76, 0.25, 0.68, 0.43],
                [0.88, 0.17, 0.69, 0.21, 0.64, 0.36, 0.52],
                [0.41, 0.72, 0.15, 0.86, 0.27, 0.93, 0.39],
            ],
        ]
    )
    cluster_mask = torch.tensor(
        [
            [
                [True, True, True, True, True, True, True],
                [True, False, True, False, True, False, True],
                [False, False, False, False, False, False, False],
            ],
            [
                [False, True, False, False, False, False, False],
                [True, True, True, True, True, False, False],
                [False, True, True, False, True, True, False],
            ],
        ]
    )

    zones = index._select_cluster_zones(cluster_scores, cluster_mask)
    expected_zones = index._select_zone_masks(
        cluster_scores.flatten(0, 1),
        cluster_mask.flatten(0, 1),
    )

    packed_zones = (
        (zones.sparse_retrieval_indices, zones.sparse_retrieval_mask),
        (zones.sparse_estimation_indices, zones.sparse_estimation_mask),
        (zones.expanded_retrieval_indices, zones.expanded_retrieval_mask),
        (zones.expanded_estimation_indices, zones.expanded_estimation_mask),
    )
    for (indices, mask), expected in zip(packed_zones, expected_zones):
        expected = expected.view_as(cluster_mask)
        for batch_id in range(cluster_scores.shape[0]):
            for head_id in range(cluster_scores.shape[1]):
                actual_indices = indices[batch_id, head_id][mask[batch_id, head_id]]
                expected_indices = torch.nonzero(
                    expected[batch_id, head_id], as_tuple=False
                ).flatten()
                assert (
                    actual_indices.sort().values.tolist() == expected_indices.tolist()
                )

    sparse_mass = index._sum_selected_scores(
        cluster_scores,
        zones.sparse_retrieval_indices,
        zones.sparse_retrieval_mask,
    )
    expected_sparse_mass = (
        cluster_scores * expected_zones[0].view_as(cluster_mask)
    ).sum(dim=2)
    torch.testing.assert_close(sparse_mass, expected_sparse_mass)


def test_bounded_mask_packing_uses_fixed_width_and_preserves_valid_indices():
    mask = torch.tensor(
        [
            [[False, True, False, True, False, True]],
            [[True, False, False, False, False, False]],
        ]
    )

    indices, packed_mask = RetroSpecSegmentedTokenIndex._pack_bounded_mask_indices(
        mask, output_width=4
    )

    assert indices.shape == (2, 1, 4)
    assert packed_mask.tolist() == [
        [[True, True, True, False]],
        [[True, False, False, False]],
    ]
    assert indices[0, 0, packed_mask[0, 0]].tolist() == [1, 3, 5]
    assert indices[1, 0, packed_mask[1, 0]].tolist() == [0]


def test_primary_exact_capacity_covers_every_up_to_date_layout():
    index = make_index(segment_size_tokens=8)
    max_num_tokens = 128
    capacity = min(
        max_num_tokens,
        index.segment_size_tokens + (index.num_recent_blocks + 1) * index.block_size,
    )

    for seq_len in range(1, max_num_tokens + 1):
        block_table = torch.empty(1, max_num_tokens // index.block_size)
        _, valid_mask, forced_exact_mask = index._build_token_layout(
            block_table,
            torch.tensor([seq_len], dtype=torch.int32),
        )
        indexed_mask = torch.zeros_like(valid_mask)
        indexed_mask[:, index.block_size : index._desired_indexed_end(seq_len)] = True
        forced_exact_mask |= valid_mask & ~indexed_mask

        assert forced_exact_mask.sum().item() <= capacity


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

    first_identities = index.cluster_store.get_cluster_identities(
        "layer", record.segments[0].cluster_blocks.cluster_ids
    )
    second_identities = index.cluster_store.get_cluster_identities(
        "layer", record.segments[1].cluster_blocks.cluster_ids
    )
    assert [identity.local_cluster_id for identity in first_identities.values()] == [
        0,
        1,
    ]
    assert [identity.local_cluster_id for identity in second_identities.values()] == [
        2,
        3,
    ]
    assert all(
        identity.group.request_id == "request" and identity.group.kv_head_index == 0
        for identity in (*first_identities.values(), *second_identities.values())
    )

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

    expanded_keys, expanded_values, expanded_mask = materialize_reference(
        index, expanded, keys, values, block_table
    )

    expanded_keys = expanded_keys[0, 0, expanded_mask[0, 0], 0]
    expanded_values = expanded_values[0, 0, expanded_mask[0, 0], 0]

    assert sorted(expanded_keys[-4:].tolist()) == [1.0, 1.0, 2.0, 2.0]
    assert sorted(expanded_values[-4:].tolist()) == [10.0, 10.0, 20.0, 20.0]


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
    materialize_metadata = Mock(wraps=index.cluster_store.get_cluster_block_metadata)
    index.cluster_store.get_cluster_block_metadata = materialize_metadata

    first = index._pack_indices("layer", ["request"], keys, block_table)
    second = index._pack_indices("layer", ["request"], keys, block_table)
    assert second is first
    assert materialize_metadata.call_count == 1

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


def test_cpu_offload_can_defer_flush_and_discard_index_updates():
    index = make_index(cache_mode="cpu_offload")
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )

    assert index.has_staged_updates
    assert index.cluster_store.num_allocated_pages("layer") == 0
    assert index.needs_update("request", 10, ["layer"])
    with pytest.raises(RuntimeError, match="staged index updates"):
        index.begin_proposal(["request"])

    index.discard_staged_updates()

    assert not index.has_staged_updates
    assert index.cluster_store.num_allocated_pages("layer") == 0
    assert index.needs_update("request", 10, ["layer"])

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )
    index.flush_staged_updates()

    assert not index.has_staged_updates
    assert index.cluster_store.num_allocated_pages("layer") == 2
    assert not index.needs_update("request", 10, ["layer"])


def test_cpu_offload_direct_build_preserves_synchronous_api():
    index = make_index(cache_mode="cpu_offload")
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)

    assert not index.has_staged_updates
    assert index.cluster_store.num_allocated_pages("layer") == 2
    index.begin_proposal(["request"])
    index.end_proposal()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_offload_draft_estimates_misses_and_uses_resident_hits():
    device = torch.device("cuda")
    index = make_index(
        cache_mode="cpu_offload",
        cache_ratio=0.5,
    )
    keys, values = make_cache()
    keys = keys.to(device=device, dtype=torch.bfloat16)
    values = values.to(device=device, dtype=torch.bfloat16)
    block_table = torch.arange(
        7,
        dtype=torch.int32,
        device=device,
    ).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    selection_kwargs = {
        "request_ids": ["request"],
        "layer_name": "layer",
        "query": torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
        "key_cache": keys,
        "value_cache": values,
        "block_table": block_table,
        "seq_lens": torch.tensor([10], dtype=torch.int32, device=device),
        "active_mask": torch.tensor([True], device=device),
        "scale": 1.0,
    }

    index.begin_proposal(["request"])
    try:
        cold = index.select_segmented(**selection_kwargs)
    finally:
        index.end_proposal()

    assert cold.resolved_pages is not None
    assert index.cluster_store.num_resident_pages("layer") == 0
    assert cold.exact_token_counts.tolist() == [[6]]
    assert cold.estimation_token_counts.tolist() == [[[2, 2]]]
    assert cold.estimation_keys[0, 0, :, 0].tolist() == pytest.approx([1.0, 2.0])
    assert cold.estimation_values[0, 0, :, 0].tolist() == pytest.approx([10.0, 20.0])
    assert cold.hit_attn.item() == pytest.approx(0.0)
    assert not cold.exact_page_token_counts.any()
    assert cold.plan.sparse_exact_page_token_counts.sum().item() == 2

    index.cluster_store.admit_resident_clusters(
        "layer",
        cold.plan.sparse_exact_cluster_ids,
        cold.plan.sparse_exact_page_ids,
    )
    index.cluster_store.get_resident_page_storage("layer")
    torch.cuda.current_stream().synchronize()

    index.begin_proposal(["request"])
    try:
        warm = index.select_segmented(**selection_kwargs)
        verification = index.materialize(
            warm.plan,
            RetroSpecAttentionLevel.SPARSE,
            keys,
            values,
            block_table,
        )
    finally:
        index.end_proposal()

    assert warm.resolved_pages is not None
    assert index.cluster_store.num_resident_pages("layer") == 1
    assert warm.exact_token_counts.tolist() == [[8]]
    assert warm.estimation_token_counts.tolist() == [[[2, 0]]]
    assert warm.hit_attn.item() == pytest.approx(warm.plan.sparse_attn.item())

    assert verification.resolved_pages is None
    assert verification.exact_token_counts.tolist() == [[8]]
    assert verification.estimation_token_counts.tolist() == [[[2]]]
    assert verification.attention_mass.item() == pytest.approx(
        warm.plan.sparse_attn.item()
    )


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

    assert selection.exact_cluster_ids.device.type == "cuda"
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
