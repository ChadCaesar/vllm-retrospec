# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.retrospec import segmented_index as segmented_index_module
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)


def make_index(
    prefill_segment_size_tokens: int = 4,
    generation_update_interval: int = 2,
    blocks_per_cluster: int = 1,
    retrieval_ratio: float = 0.5,
    estimation_ratio: float = 0.5,
    cache_ratio: float = 0.0,
    pin_memory: bool = False,
    max_pending_cluster_builds: int = 2,
    max_resident_requests: int = 1,
) -> RetroSpecSegmentedTokenIndex:
    return RetroSpecSegmentedTokenIndex(
        block_size=2,
        num_speculative_tokens=1,
        retrieval_ratio=retrieval_ratio,
        estimation_ratio=estimation_ratio,
        prefill_segment_size_tokens=prefill_segment_size_tokens,
        generation_update_interval=generation_update_interval,
        blocks_per_cluster=blocks_per_cluster,
        num_kmeans_iterations=2,
        max_model_len=64,
        max_pending_cluster_builds=max_pending_cluster_builds,
        cache_ratio=cache_ratio,
        pin_memory=pin_memory,
        max_resident_requests=max_resident_requests,
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


@contextmanager
def active_residency(
    index: RetroSpecSegmentedTokenIndex,
    request_ids: list[str],
):
    index.begin_full_verification_residency(request_ids)
    try:
        yield
    finally:
        index.end_full_verification_residency()


@pytest.mark.parametrize(
    ("cache_ratio", "expected_cache_ratio"),
    [(0.0, 0.6), (0.35, 0.35)],
)
def test_segmented_index_configures_resident_cache_ratio(
    cache_ratio: float,
    expected_cache_ratio: float,
):
    index = make_index(
        retrieval_ratio=0.2,
        cache_ratio=cache_ratio,
    )

    assert index.cluster_store.cache_ratio == pytest.approx(expected_cache_ratio)


def test_sparse_verification_prefetch_masks_inactive_draft_rows():
    index = make_index(cache_ratio=0.5, pin_memory=True)
    plan = Mock(
        layer_name="layer",
        sparse_exact_cluster_ids=torch.tensor([[[0, 1]], [[2, 3]]], dtype=torch.int64),
        sparse_exact_page_ids=torch.tensor(
            [[[[0], [1]]], [[[2], [3]]]], dtype=torch.int64
        ),
    )
    index.cluster_store.prefetch_resident_clusters = Mock()

    index.prefetch_sparse_verification(
        plan,
        active_mask=torch.tensor([True, False]),
    )

    call_kwargs = index.cluster_store.prefetch_resident_clusters.call_args.kwargs
    assert call_kwargs["layer_name"] == "layer"
    assert call_kwargs["cluster_ids"].tolist() == [[[0, 1]], [[-1, -1]]]


def test_sparse_verification_prefetch_skips_empty_page_table():
    index = make_index(cache_ratio=0.5, pin_memory=True)
    plan = Mock(
        sparse_exact_cluster_ids=torch.full((1, 1, 1), -1, dtype=torch.int64),
        sparse_exact_page_ids=torch.empty((1, 1, 1, 0), dtype=torch.int64),
    )
    index.cluster_store.prefetch_resident_clusters = Mock()

    index.prefetch_sparse_verification(plan, active_mask=torch.tensor([True]))

    index.cluster_store.prefetch_resident_clusters.assert_not_called()


def test_sparse_verification_prefetch_requires_pinned_cpu_backing():
    index = make_index(pin_memory=False)
    plan = Mock(
        sparse_exact_cluster_ids=torch.tensor([[[0]]]),
        sparse_exact_page_ids=torch.tensor([[[[0]]]]),
    )
    index.cluster_store.prefetch_resident_clusters = Mock()

    index.prefetch_sparse_verification(
        plan,
        active_mask=torch.tensor([True]),
    )

    index.cluster_store.prefetch_resident_clusters.assert_not_called()


def build_index(
    index: RetroSpecSegmentedTokenIndex,
    seq_len: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
    defer_cpu_store: bool = False,
    is_prefill: bool = True,
    prefill_complete: bool = False,
) -> None:
    index.build_or_update(
        layer_name="layer",
        request_ids=["request"],
        seq_lens=[seq_len],
        is_prefill=[is_prefill],
        rows=[0],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
        defer_cpu_store=defer_cpu_store,
        prefill_complete=[prefill_complete],
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

    assert not index.needs_update("request", 10, ["layer"], True)
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


def test_generation_appends_smaller_segments_after_prefill():
    index = make_index(
        prefill_segment_size_tokens=8,
        generation_update_interval=4,
    )
    keys, values = make_cache(num_blocks=12)
    block_table = torch.arange(12, dtype=torch.int32).view(1, -1)

    build_index(index, 14, keys, values, block_table, is_prefill=True)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 10
    assert record.num_clusters == 4
    assert [
        (segment.indexed_start, segment.indexed_end) for segment in record.segments
    ] == [(2, 10)]

    assert not index.needs_update("request", 16, ["layer"], False)
    assert index.needs_update("request", 18, ["layer"], False)

    build_index(index, 18, keys, values, block_table, is_prefill=False)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 14
    assert record.num_clusters == 6
    assert [
        (segment.indexed_start, segment.indexed_end, segment.cluster_start)
        for segment in record.segments
    ] == [(2, 10, 0), (10, 14, 4)]
    assert not index.needs_update("request", 18, ["layer"], False)


def test_completed_prefill_clusters_aligned_tail():
    index = make_index(
        prefill_segment_size_tokens=8,
        generation_update_interval=4,
    )
    keys, values = make_cache(num_blocks=12)
    block_table = torch.arange(12, dtype=torch.int32).view(1, -1)

    assert index._desired_indexed_end(16, None, True) == 10
    assert index._desired_indexed_end(16, None, True, prefill_complete=True) == 12

    build_index(
        index,
        16,
        keys,
        values,
        block_table,
        is_prefill=True,
        prefill_complete=True,
    )

    record = index._indices["layer"]["request"]
    assert record.indexed_end == 12
    assert record.num_clusters == 5
    cluster_token_counts = record.segments[0].cluster_token_counts
    assert cluster_token_counts.shape == (1, 5)
    assert cluster_token_counts[0, :4].sum().item() == 8
    assert cluster_token_counts[0, 4].item() == 2
    assert [
        (segment.indexed_start, segment.indexed_end) for segment in record.segments
    ] == [(2, 12)]


def test_completed_chunked_prefill_appends_only_adaptive_tail():
    index = make_index(
        prefill_segment_size_tokens=8,
        generation_update_interval=4,
    )
    keys, values = make_cache(num_blocks=12)
    block_table = torch.arange(12, dtype=torch.int32).view(1, -1)

    build_index(index, 14, keys, values, block_table, is_prefill=True)
    build_index(
        index,
        16,
        keys,
        values,
        block_table,
        is_prefill=True,
        prefill_complete=True,
    )

    record = index._indices["layer"]["request"]
    assert record.indexed_end == 12
    assert [
        (segment.indexed_start, segment.indexed_end, segment.cluster_start)
        for segment in record.segments
    ] == [(2, 10, 0), (10, 12, 4)]


def test_prefill_complete_rejects_generation_phase():
    index = make_index()

    with pytest.raises(ValueError, match="prefill_complete requires is_prefill"):
        index.needs_update("request", 10, ["layer"], False, prefill_complete=True)


def test_prefill_and_generation_sizes_do_not_need_to_divide_each_other():
    index = make_index(
        prefill_segment_size_tokens=12,
        generation_update_interval=8,
    )
    keys, values = make_cache(num_blocks=16)
    block_table = torch.arange(16, dtype=torch.int32).view(1, -1)

    build_index(index, 18, keys, values, block_table, is_prefill=True)
    assert index._indices["layer"]["request"].indexed_end == 14
    assert not index.needs_update("request", 24, ["layer"], False)
    assert index.needs_update("request", 26, ["layer"], False)

    build_index(index, 26, keys, values, block_table, is_prefill=False)
    record = index._indices["layer"]["request"]
    assert [
        (segment.indexed_start, segment.indexed_end) for segment in record.segments
    ] == [
        (2, 14),
        (14, 22),
    ]


def test_generation_rollback_rebuilds_on_generation_boundaries():
    index = make_index(
        prefill_segment_size_tokens=8,
        generation_update_interval=4,
    )
    keys, values = make_cache(num_blocks=12)
    block_table = torch.arange(12, dtype=torch.int32).view(1, -1)

    build_index(index, 14, keys, values, block_table, is_prefill=True)
    build_index(index, 18, keys, values, block_table, is_prefill=False)
    assert index._indices["layer"]["request"].indexed_end == 14

    build_index(index, 13, keys, values, block_table, is_prefill=False)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 6
    assert record.num_clusters == 2
    assert [
        (segment.indexed_start, segment.indexed_end) for segment in record.segments
    ] == [(2, 6)]


def test_cpu_offload_keeps_incremental_generation_segments_on_cpu():
    index = make_index(
        prefill_segment_size_tokens=8,
        generation_update_interval=4,
    )
    keys, values = make_cache(num_blocks=12)
    block_table = torch.arange(12, dtype=torch.int32).view(1, -1)

    build_index(index, 14, keys, values, block_table, is_prefill=True)
    build_index(index, 18, keys, values, block_table, is_prefill=False)

    record = index._indices["layer"]["request"]
    assert len(record.segments) == 2
    for segment in record.segments:
        assert segment.cluster_keys.device.type == "cpu"
        assert segment.cluster_values.device.type == "cpu"
        assert segment.cluster_token_counts.device.type == "cpu"


def test_fully_stored_indexed_end_uses_slowest_layer():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    assert (
        index.get_fully_stored_indexed_end("request", ["layer", "missing-layer"])
        == index.block_size
    )

    index.build_or_update(
        layer_name="other-layer",
        request_ids=["request"],
        seq_lens=[14],
        is_prefill=[True],
        rows=[0],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    assert index._indices["layer"]["request"].indexed_end == 6
    assert index._indices["other-layer"]["request"].indexed_end == 10
    assert index.get_fully_stored_indexed_end("request", ["layer", "other-layer"]) == 6


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
    index = make_index(
        prefill_segment_size_tokens=8,
        generation_update_interval=4,
        blocks_per_cluster=2,
    )
    keys = torch.ones(8, 2, 1, 1)
    values = torch.ones_like(keys)
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 14, keys, values, block_table)

    with active_residency(index, ["request"]):
        view = index._get_resident_view("layer", ["request"], keys)
        assert view.arena is not None
        slot = view.request_slot_ids.item()
        assert view.arena.cluster_token_counts[slot, 0, :2].tolist() == [8, 0]
        assert view.arena.cluster_ids[slot, 0, 0].item() >= 0
        assert view.arena.cluster_ids[slot, 0, 1].item() == -1


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

    ranking_scores = cluster_scores.masked_fill(~cluster_mask, float("-inf"))
    candidate_counts = cluster_mask.sum(dim=2, dtype=torch.int32)
    zones = index._select_cluster_zones(
        cluster_scores, ranking_scores, candidate_counts
    )
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
    index = make_index(prefill_segment_size_tokens=8)
    max_num_tokens = 128
    capacity = min(
        max_num_tokens,
        max(
            index.prefill_segment_size_tokens,
            index.generation_update_interval,
        )
        + (index.num_recent_blocks + 1) * index.block_size,
    )

    for seq_len in range(1, max_num_tokens + 1):
        block_table = torch.empty(1, max_num_tokens // index.block_size)
        _, valid_mask, forced_exact_mask = index._build_token_layout(
            block_table,
            torch.tensor([seq_len], dtype=torch.int32),
        )
        indexed_mask = torch.zeros_like(valid_mask)
        desired_end = index._desired_indexed_end(seq_len, None, True)
        indexed_mask[:, index.block_size : desired_end] = True
        forced_exact_mask |= valid_mask & ~indexed_mask

        assert forced_exact_mask.sum().item() <= capacity


def test_segmented_index_handles_mixed_long_and_short_requests():
    index = make_index(max_resident_requests=2)
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["long", "short"],
        seq_lens=[10, 3],
        is_prefill=[True, True],
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


def test_full_verification_plan_covers_clustered_and_primary_tokens():
    index = make_index(max_resident_requests=2)
    keys, values = make_cache(num_kv_heads=2)
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["long", "short"],
        seq_lens=[10, 3],
        is_prefill=[True, True],
        rows=[0, 1],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    index.begin_full_verification_residency(["long", "short"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["long", "short"],
            layer_name="layer",
            seq_lens=[10, 3],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    assert plan.layer_name == "layer"
    assert plan.exact_token_counts.tolist() == [[10, 10], [3, 3]]
    assert plan.primary_exact_token_mask.sum(dim=2).tolist() == [[6, 6], [3, 3]]
    assert plan.exact_page_token_counts.sum(dim=(2, 3)).tolist() == [
        [4, 4],
        [0, 0],
    ]
    assert plan.primary_exact_token_indices[0, 0].tolist() == [0, 1, 6, 7, 8, 9]
    assert plan.primary_exact_token_mask[1, 0].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    short_primary_indices = plan.primary_exact_token_indices[1, 0][
        plan.primary_exact_token_mask[1, 0]
    ]
    assert short_primary_indices.tolist() == [0, 1, 2]
    assert (plan.exact_page_ids[0] >= 0).sum(dim=(1, 2)).tolist() == [2, 2]
    assert not (plan.exact_page_ids[1] >= 0).any()
    assert torch.equal(plan.exact_page_ids_cpu, plan.exact_page_ids.cpu())


def test_full_verification_plan_handles_request_without_cluster_pages():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 3, keys, values, block_table)

    index.begin_full_verification_residency(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[3],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    assert plan.exact_token_counts.tolist() == [[3]]
    assert plan.primary_exact_token_indices.tolist() == [[[0, 1, 2]]]
    assert plan.primary_exact_token_mask.all()
    assert plan.exact_page_ids.shape == (1, 1, 1, 0)
    assert plan.exact_page_ids.numel() == 0
    assert plan.exact_page_ids_cpu.shape == plan.exact_page_ids.shape
    assert plan.exact_page_token_counts.numel() == 0


def test_cpu_offload_sparse_selection_handles_request_without_cluster_pages():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 3, keys, values, block_table)

    index.begin_proposal(["request"])
    try:
        selection = index.select_segmented(
            request_ids=["request"],
            layer_name="layer",
            query=torch.ones(1, 1, 1),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([3], dtype=torch.int32),
            active_mask=torch.tensor([True]),
            scale=1.0,
        )
    finally:
        index.end_proposal()

    assert selection.exact_token_counts.tolist() == [[3]]
    primary_indices = selection.plan.primary_exact_token_indices[
        selection.plan.primary_exact_token_mask
    ]
    assert primary_indices.tolist() == [0, 1, 2]
    assert selection.exact_page_ids.shape == (1, 1, 1, 0)
    assert selection.resolved_pages is None


def test_full_verification_plan_rejects_staged_index_updates():
    index = make_index()
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

    try:
        with pytest.raises(RuntimeError, match="index updates are staged"):
            index.build_full_verification_plan(
                request_ids=["request"],
                layer_name="layer",
                seq_lens=[10],
                key_cache=keys,
                block_table=block_table,
            )
    finally:
        index.discard_staged_updates()


def test_full_verification_plan_allows_another_layer_to_be_staged():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)
    index.build_or_update(
        layer_name="other-layer",
        request_ids=["request"],
        seq_lens=[10],
        is_prefill=[True],
        rows=[0],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
        defer_cpu_store=True,
    )

    # This unit exercises the layer-local plan guard directly. Production
    # full verification rejects any unflushed transaction at context entry.
    index._gpu_index_residency.activate(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[10],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index._gpu_index_residency.deactivate()
        index.discard_staged_updates()

    assert plan.exact_token_counts.tolist() == [[10]]


def test_prepare_full_verification_rolls_back_uncommitted_clusters():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    assert index.cluster_store.num_allocated_pages("layer") == 2

    index.prepare_full_verification(
        request_ids=["request"],
        context_lens=[5],
        layer_names=["layer"],
    )
    index.begin_full_verification_residency(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[5],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    record = index._indices["layer"]["request"]
    assert record.segments == []
    assert index.cluster_store.num_allocated_pages("layer") == 0
    assert plan.exact_token_counts.tolist() == [[5]]
    assert plan.primary_exact_token_mask.sum().item() == 5


def test_full_verification_plan_accepts_an_empty_context():
    index = make_index()
    keys, _ = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    index.begin_full_verification_residency(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[0],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    assert plan.exact_token_counts.tolist() == [[0]]
    assert plan.primary_exact_token_indices.shape == (1, 1, 0)
    assert plan.primary_exact_token_mask.shape == (1, 1, 0)
    assert plan.exact_page_ids.numel() == 0


def test_full_verification_plan_rejects_unapplied_rollback():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    with pytest.raises(RuntimeError, match="rolled-back cluster state"):
        index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[5],
            key_cache=keys,
            block_table=block_table,
        )


def test_segmented_index_appends_complete_segments_and_handles_rollback():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)
    first_segment = index._indices["layer"]["request"].segments[0]
    assert index.needs_update("request", 14, ["layer"], True)

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

    assert index.needs_update("request", 6, ["layer"], True)
    build_index(index, 6, keys, values, block_table)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 2
    assert record.num_clusters == 0
    assert record.segments == []
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_rollback_invalidates_active_view_before_rebuild(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 14, keys, values, block_table)

    def fail_clustering(**_kwargs):
        raise RuntimeError("clustering failed")

    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.segmented_index.segmented_kmeans",
        fail_clustering,
    )

    with active_residency(index, ["request"]):
        original = index._get_resident_view("layer", ["request"], keys)
        with pytest.raises(RuntimeError, match="clustering failed"):
            build_index(index, 10, keys, values, block_table)
        rebuilt = index._get_resident_view("layer", ["request"], keys)

    assert rebuilt is not original
    assert rebuilt.request_slot_ids.tolist() == [-1]
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
    assert index.needs_update("request", 10, ["layer"], True)
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_segmented_index_reuses_active_view_until_update():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)
    with active_residency(index, ["request"]):
        first = index._get_resident_view("layer", ["request"], keys)
        second = index._get_resident_view("layer", ["request"], keys)
        assert second is first

        build_index(index, 14, keys, values, block_table)
        after_update = index._get_resident_view("layer", ["request"], keys)

        assert after_update is not first


def test_removing_request_invalidates_active_view():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    with active_residency(index, ["request"]):
        view = index._get_resident_view("layer", ["request"], keys)

        index.remove_requests(["request"])

        assert "request" not in index._indices["layer"]
        rebuilt = index._get_resident_view("layer", ["request"], keys)
        assert rebuilt is not view
        assert rebuilt.request_slot_ids.tolist() == [-1]


def test_active_view_tracks_request_order_without_block_table_width():
    index = make_index(max_resident_requests=2)
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["long", "short"],
        seq_lens=[10, 3],
        is_prefill=[True, True],
        rows=[0, 1],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    with active_residency(index, ["long", "short"]):
        original = index._get_resident_view("layer", ["long", "short"], keys)
        with pytest.raises(RuntimeError, match="request order"):
            index._get_resident_view("layer", ["short", "long"], keys)
        repeated = index._get_resident_view("layer", ["long", "short"], keys)

        assert repeated is original
        assert original.request_slot_ids.shape == (2,)


def test_segmented_index_proposal_lifecycle_tracks_empty_batches():
    index = make_index()

    index.begin_proposal([])
    with pytest.raises(RuntimeError, match="already active"):
        index.begin_proposal([])
    index.end_proposal()

    with pytest.raises(RuntimeError, match="not active"):
        index.end_proposal()


def test_cpu_offload_keeps_request_indices_after_batch_deactivation():
    index = make_index(max_resident_requests=2)
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["first", "second"],
        seq_lens=[10, 10],
        is_prefill=[True, True],
        rows=[0, 1],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    for request_id in ("first", "second"):
        segment = index._indices["layer"][request_id].segments[0]
        assert segment.cluster_keys.device.type == "cpu"
        assert segment.cluster_values.device.type == "cpu"
        assert segment.cluster_token_counts.device.type == "cpu"

    index.begin_proposal(["first", "second"])
    try:
        view = index._gpu_index_residency.get_active_view(
            "layer", ["first", "second"], keys.device
        )
        assert view.arena is not None
        for row, request_id in enumerate(("first", "second")):
            segment = index._indices["layer"][request_id].segments[0]
            slot = int(view.request_slot_ids[row].item())
            num_clusters = segment.cluster_token_counts.shape[1]
            assert torch.equal(
                view.arena.cluster_keys[slot, :, :num_clusters],
                segment.cluster_keys,
            )
            assert torch.equal(
                view.arena.cluster_values[slot, :, :num_clusters],
                segment.cluster_values,
            )
            assert torch.equal(
                view.arena.cluster_token_counts[slot, :, :num_clusters],
                segment.cluster_token_counts,
            )
    finally:
        index.end_proposal()

    assert index._gpu_index_residency.resident_request_ids == (
        "first",
        "second",
    )
    assert index._gpu_index_residency.num_resident_layers == 1

    index.begin_proposal(["first", "second"])
    try:
        view = index._get_resident_view("layer", ["first", "second"], keys)
        assert view.arena is not None
        assert (view.request_slot_ids >= 0).all()
    finally:
        index.end_proposal()

    assert index._gpu_index_residency.num_resident_layers == 1

    index.remove_requests(["first"])
    assert index._gpu_index_residency.resident_request_ids == ("second",)
    assert index._gpu_index_residency.get_num_clusters("layer", "first") == 0


def test_cpu_offload_appends_resident_segments_across_index_updates():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    first_num_clusters = index._gpu_index_residency.get_num_clusters("layer", "request")
    first_indexed_end = index._gpu_index_residency.get_indexed_end("layer", "request")

    index.begin_proposal(["request"])
    try:
        first_view = index._get_resident_view("layer", ["request"], keys)
        assert first_view.arena is not None
        first_slot = first_view.request_slot_ids.item()
        assert first_view.arena.num_clusters[first_slot].item() == 2
    finally:
        index.end_proposal()

    build_index(index, 14, keys, values, block_table)

    updated_num_clusters = index._gpu_index_residency.get_num_clusters(
        "layer", "request"
    )
    updated_indexed_end = index._gpu_index_residency.get_indexed_end("layer", "request")
    assert first_num_clusters == 2
    assert updated_num_clusters == 4
    assert first_indexed_end == 6
    assert updated_indexed_end == 10

    index.begin_proposal(["request"])
    try:
        updated_view = index._get_resident_view("layer", ["request"], keys)
        assert updated_view.arena is not None
        updated_slot = updated_view.request_slot_ids.item()
        assert updated_view.arena.num_clusters[updated_slot].item() == 4
    finally:
        index.end_proposal()


def test_cpu_offload_rejects_more_requests_than_reserved_capacity():
    index = make_index(max_resident_requests=2)

    with pytest.raises(RuntimeError, match="exceeds max_num_seqs"):
        index.begin_proposal(["first", "second", "third"])


def test_cpu_offload_can_defer_flush_and_discard_index_updates():
    index = make_index()
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
    assert index.needs_update("request", 10, ["layer"], True)
    with pytest.raises(RuntimeError, match="staged index updates"):
        index.begin_proposal(["request"])

    # The background worker may finish page construction before the index is
    # published. Its pages remain private to the staged transaction.
    index._staged_segments[0].build_future.result()
    assert index.cluster_store.num_allocated_pages("layer") == 2
    assert index.needs_update("request", 10, ["layer"], True)

    index.discard_staged_updates()

    assert not index.has_staged_updates
    assert index._cluster_build_executor is None
    assert not index._pending_cluster_builds
    assert index.cluster_store.num_allocated_pages("layer") == 0
    assert index.needs_update("request", 10, ["layer"], True)

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
    assert index._cluster_build_executor is None
    assert not index._pending_cluster_builds
    assert index.cluster_store.num_allocated_pages("layer") == 2
    assert not index.needs_update("request", 10, ["layer"], True)


def test_cpu_offload_direct_build_preserves_synchronous_api():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)

    assert not index.has_staged_updates
    assert index.cluster_store.num_allocated_pages("layer") == 2
    index.begin_proposal(["request"])
    index.end_proposal()


def test_cpu_offload_recreates_builder_for_later_index_transaction():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)
    first_segment = index._indices["layer"]["request"].segments[0]

    assert index._cluster_build_executor is None

    build_index(index, 14, keys, values, block_table)

    record = index._indices["layer"]["request"]
    assert index._cluster_build_executor is None
    assert len(record.segments) == 2
    assert record.segments[0] is first_segment
    assert record.indexed_end == 10
    assert index.cluster_store.num_allocated_pages("layer") == 4


def test_cpu_offload_builds_cluster_pages_on_background_worker(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_started = threading.Event()
    allow_build = threading.Event()
    worker_names: list[str] = []
    original_store = index.cluster_store.store_staged_clusters

    def store_staged_clusters(*args, **kwargs):
        worker_names.append(threading.current_thread().name)
        build_started.set()
        if not allow_build.wait(timeout=5):
            raise RuntimeError("background build was not released")
        return original_store(*args, **kwargs)

    monkeypatch.setattr(
        index.cluster_store,
        "store_staged_clusters",
        store_staged_clusters,
    )

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )

    try:
        assert build_started.wait(timeout=5)
        assert index.has_staged_updates
        assert index.needs_update("request", 10, ["layer"], True)
        assert worker_names[0].startswith("retrospec-cluster-page")

        allow_build.set()
        index.flush_staged_updates()
    finally:
        allow_build.set()
        if index.has_staged_updates:
            index.discard_staged_updates()

    assert not index.needs_update("request", 10, ["layer"], True)
    assert index.cluster_store.num_allocated_pages("layer") == 2


def test_cpu_offload_backpressure_waits_for_oldest_pending_build():
    index = make_index(max_pending_cluster_builds=2)
    first_build: Future = Future()
    second_build: Future = Future()
    index._pending_cluster_builds.extend((first_build, second_build))
    wait_started = threading.Event()
    slot_available = threading.Event()
    errors: list[BaseException] = []

    def wait_for_slot():
        wait_started.set()
        try:
            index._wait_for_cluster_build_slot()
        except BaseException as exc:
            errors.append(exc)
        finally:
            slot_available.set()

    waiter = threading.Thread(target=wait_for_slot)
    waiter.start()

    try:
        assert wait_started.wait(timeout=5)
        assert not slot_available.wait(timeout=0.1)
        first_build.set_result(Mock())
        assert slot_available.wait(timeout=5)
    finally:
        if not first_build.done():
            first_build.set_result(Mock())
        waiter.join(timeout=5)

    assert not waiter.is_alive()
    assert not errors
    assert list(index._pending_cluster_builds) == [second_build]


def test_cpu_offload_backpressure_reaps_completed_builds_and_propagates_errors():
    index = make_index(max_pending_cluster_builds=2)
    completed_build: Future = Future()
    failed_build: Future = Future()
    completed_build.set_result(Mock())
    failed_build.set_exception(RuntimeError("background build failed"))
    index._pending_cluster_builds.extend((completed_build, failed_build))

    with pytest.raises(RuntimeError, match="background build failed"):
        index._wait_for_cluster_build_slot()

    assert not index._pending_cluster_builds


def test_cpu_offload_flush_rolls_back_all_layers_after_build_failure(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    original_store = index.cluster_store.store_staged_clusters

    def store_staged_clusters(*args, **kwargs):
        if kwargs["layer_name"] == "failed-layer":
            raise RuntimeError("background build failed")
        return original_store(*args, **kwargs)

    monkeypatch.setattr(
        index.cluster_store,
        "store_staged_clusters",
        store_staged_clusters,
    )

    for layer_name in ("completed-layer", "failed-layer"):
        index.build_or_update(
            layer_name=layer_name,
            request_ids=["request"],
            seq_lens=[10],
            is_prefill=[True],
            rows=[0],
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            defer_cpu_store=True,
        )

    with pytest.raises(RuntimeError, match="background build failed"):
        index.flush_staged_updates()

    assert not index.has_staged_updates
    assert index._cluster_build_executor is None
    assert index.needs_update("request", 10, ["completed-layer"], True)
    assert index.needs_update("request", 10, ["failed-layer"], True)
    assert index.cluster_store.num_allocated_pages("completed-layer") == 0
    assert index.cluster_store.num_allocated_pages("failed-layer") == 0


def test_cpu_offload_stages_token_kv_before_clustering(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    call_order: list[str] = []

    original_stage_token_kv = index.cluster_store.stage_token_kv
    original_finish_stage_clusters = index.cluster_store.finish_stage_clusters
    original_wait_for_slot = index._wait_for_cluster_build_slot

    def wait_for_cluster_build_slot():
        call_order.append("wait_for_cluster_build_slot")
        return original_wait_for_slot()

    def stage_token_kv(*args, **kwargs):
        call_order.append("stage_token_kv")
        return original_stage_token_kv(*args, **kwargs)

    def finish_stage_clusters(*args, **kwargs):
        call_order.append("finish_stage_clusters")
        return original_finish_stage_clusters(*args, **kwargs)

    original_clustering = segmented_index_module.segmented_kmeans

    def segmented_kmeans(*args, **kwargs):
        call_order.append("segmented_kmeans")
        return original_clustering(*args, **kwargs)

    monkeypatch.setattr(index.cluster_store, "stage_token_kv", stage_token_kv)
    monkeypatch.setattr(
        index,
        "_wait_for_cluster_build_slot",
        wait_for_cluster_build_slot,
    )
    monkeypatch.setattr(
        index.cluster_store,
        "finish_stage_clusters",
        finish_stage_clusters,
    )
    monkeypatch.setattr(
        segmented_index_module,
        "segmented_kmeans",
        segmented_kmeans,
    )

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )

    try:
        assert call_order == [
            "wait_for_cluster_build_slot",
            "stage_token_kv",
            "segmented_kmeans",
            "finish_stage_clusters",
        ]
    finally:
        index.discard_staged_updates()


def test_cpu_offload_waits_for_staged_kv_when_clustering_fails(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    staged_token_kv = Mock()
    discard_staged_token_kv = Mock(
        wraps=index.cluster_store.discard_staged_token_kv,
    )

    monkeypatch.setattr(
        index.cluster_store,
        "stage_token_kv",
        Mock(return_value=staged_token_kv),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "discard_staged_token_kv",
        discard_staged_token_kv,
    )

    monkeypatch.setattr(
        segmented_index_module,
        "segmented_kmeans",
        Mock(side_effect=RuntimeError("clustering failed")),
    )

    with pytest.raises(RuntimeError, match="clustering failed"):
        build_index(index, 10, keys, values, block_table)

    staged_token_kv.wait.assert_called_once_with()
    discard_staged_token_kv.assert_called_once_with(staged_token_kv)
    assert not index.has_staged_updates


def test_cpu_offload_discards_staged_kv_when_metadata_staging_fails(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    staged_token_kv = Mock()
    discard_staged_token_kv = Mock(
        wraps=index.cluster_store.discard_staged_token_kv,
    )

    monkeypatch.setattr(
        index.cluster_store,
        "stage_token_kv",
        Mock(return_value=staged_token_kv),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "finish_stage_clusters",
        Mock(side_effect=RuntimeError("metadata staging failed")),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "discard_staged_token_kv",
        discard_staged_token_kv,
    )

    with pytest.raises(RuntimeError, match="metadata staging failed"):
        build_index(index, 10, keys, values, block_table)

    staged_token_kv.wait.assert_called_once_with()
    discard_staged_token_kv.assert_called_once_with(staged_token_kv)
    assert not index.has_staged_updates


def test_cpu_offload_discards_staged_clusters_when_submission_fails(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    discard_staged_clusters = Mock(
        wraps=index.cluster_store.discard_staged_clusters,
    )

    monkeypatch.setattr(
        index,
        "_submit_cluster_build",
        Mock(side_effect=RuntimeError("submission failed")),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "discard_staged_clusters",
        discard_staged_clusters,
    )

    with pytest.raises(RuntimeError, match="submission failed"):
        build_index(index, 10, keys, values, block_table)

    discard_staged_clusters.assert_called_once()
    assert not index.has_staged_updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_offload_draft_estimates_misses_and_uses_resident_hits():
    device = torch.device("cuda")
    index = make_index(
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

    index.begin_proposal(["request"])
    try:
        resident_view = index._gpu_index_residency.get_active_view(
            "layer", ["request"], device
        )
        assert resident_view.arena is not None
        resident_key_ptr = resident_view.arena.cluster_keys.data_ptr()
        assert resident_view.arena.cluster_keys.device.type == "cuda"
        assert resident_view.arena.cluster_values.device.type == "cuda"
        assert resident_view.arena.cluster_token_counts.device.type == "cuda"
        assert resident_view.arena.cluster_ids.device.type == "cuda"
        assert resident_view.arena.page_ids.device.type == "cuda"
    finally:
        index.end_proposal()
    assert index._indices["layer"]["request"].segments[0].cluster_keys.device.type == (
        "cpu"
    )

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

    index.begin_proposal(["request"])
    try:
        persistent_view = index._gpu_index_residency.get_active_view(
            "layer", ["request"], device
        )
        assert persistent_view.arena is not None
        assert persistent_view.arena.cluster_keys.data_ptr() == resident_key_ptr
    finally:
        index.end_proposal()
    assert cold.resolved_pages is not None
    assert index.cluster_store.num_resident_pages("layer") == 0
    assert cold.exact_token_counts.tolist() == [[6]]
    assert cold.estimation_token_counts.tolist() == [[[2, 2]]]
    assert cold.estimation_keys[0, 0, :, 0].tolist() == pytest.approx([1.0, 2.0])
    assert cold.estimation_values[0, 0, :, 0].tolist() == pytest.approx([10.0, 20.0])
    assert cold.hit_attn.item() == pytest.approx(1.0)
    assert not cold.resolved_pages.hit_gate_ready_mask.any()
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
    assert warm.resolved_pages.hit_gate_ready_mask.all()
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
    assert selection.exact_token_counts.tolist() == [[6]]
    assert selection.estimation_token_counts[0, 0, 0].item() == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prefill_segment_size_tokens": 3}, "divisible by block_size"),
        ({"generation_update_interval": 3}, "divisible by block_size"),
        ({"prefill_segment_size_tokens": 4, "blocks_per_cluster": 3}, "divisible"),
        ({"generation_update_interval": 2, "blocks_per_cluster": 2}, "divisible"),
        ({"blocks_per_cluster": 0}, "positive"),
        ({"num_kmeans_iterations": 0}, "positive"),
        ({"max_pending_cluster_builds": 0}, "positive"),
    ],
)
def test_segmented_index_rejects_invalid_configuration(kwargs, message):
    values = {
        "block_size": 2,
        "num_speculative_tokens": 1,
        "retrieval_ratio": 0.5,
        "estimation_ratio": 0.5,
        "prefill_segment_size_tokens": 4,
        "generation_update_interval": 2,
        "blocks_per_cluster": 1,
        "num_kmeans_iterations": 2,
        "max_model_len": 64,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        RetroSpecSegmentedTokenIndex(**values)
