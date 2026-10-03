# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch

from tests.retrospec.offload.segmented_index_helpers import (
    build_index,
    make_cache,
    make_empty_resident_view,
    make_index,
    materialize_reference,
)
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)


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

    selection_scores = cluster_scores.masked_fill(~cluster_mask, float("-inf"))
    candidate_counts = cluster_mask.sum(dim=2, dtype=torch.int32)
    zones = index._select_cluster_zones(
        selection_scores,
        candidate_counts,
        make_empty_resident_view(2, 7, cluster_scores.device),
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
        zones.sparse_retrieval_scores,
        zones.sparse_retrieval_mask,
    )
    expected_sparse_mass = (
        cluster_scores * expected_zones[0].view_as(cluster_mask)
    ).sum(dim=2)
    torch.testing.assert_close(sparse_mass, expected_sparse_mass)

    expanded_mass = index._sum_selected_scores(
        zones.expanded_retrieval_scores,
        zones.expanded_retrieval_mask,
    )
    expected_expanded_mass = (
        cluster_scores * expected_zones[2].view_as(cluster_mask)
    ).sum(dim=2)
    torch.testing.assert_close(expanded_mass, expected_expanded_mass)


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


def test_bounded_mask_packing_writes_preallocated_outputs():
    mask = torch.tensor([[[False, True, False, True]]])
    indices = torch.full((1, 1, 3), -1, dtype=torch.int64)
    packed_mask = torch.ones_like(indices, dtype=torch.bool)
    topk_order = torch.empty_like(indices)

    result_indices, result_mask = (
        RetroSpecSegmentedTokenIndex._pack_bounded_mask_indices(
            mask,
            output_width=3,
            output_indices=indices,
            output_mask=packed_mask,
            topk_order=topk_order,
        )
    )

    assert result_indices.data_ptr() == indices.data_ptr()
    assert result_mask.data_ptr() == packed_mask.data_ptr()
    assert packed_mask.tolist() == [[[True, True, False]]]
    assert indices[packed_mask].tolist() == [1, 3]


def test_selection_plan_table_grows_only_between_proposals():
    index = make_index(num_speculative_tokens=2, max_resident_requests=2)
    first_view = make_empty_resident_view(1, 2, torch.device("cpu"))
    larger_view = make_empty_resident_view(2, 4, torch.device("cpu"))

    index.begin_proposal(["request"])
    try:
        index._get_selection_plan_step(
            "layer", 0, first_view, 1, 1, 1, torch.float32, torch.device("cpu")
        )
        index._selection_plan_written_layers.add("layer")
        with pytest.raises(RuntimeError, match="shape changed"):
            index._get_selection_plan_step(
                "layer",
                1,
                larger_view,
                2,
                1,
                1,
                torch.float32,
                torch.device("cpu"),
            )
    finally:
        index.end_proposal()

    index.begin_proposal(["first", "second"])
    try:
        index._get_selection_plan_step(
            "layer",
            0,
            larger_view,
            2,
            1,
            1,
            torch.float32,
            torch.device("cpu"),
        )
        assert index._selection_plan_tables["layer"].batch_capacity == 2
    finally:
        index.end_proposal()


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
        with patch.object(
            index,
            "_resolve_ranked_draft_clusters",
            wraps=index._resolve_ranked_draft_clusters,
        ) as resolve:
            cold = index.select_segmented(**selection_kwargs)
        assert resolve.call_args.kwargs["emit_misses"] is False
        cold_sparse_indices = cold.plan.ranked_cluster_indices[
            ..., : cold.plan.sparse_retrieval_width
        ].clone()
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
    assert cold.resolved_clusters is not None
    assert cold.prefetch_miss_cluster_ids is None
    assert cold.prefetch_miss_positions is None
    assert cold.prefetch_miss_count is None
    assert cold.prefetch_num_groups == 0
    assert cold.prefetch_num_ranks == 0
    assert index.cluster_store.num_resident_pages("layer") == 0
    assert cold.exact_token_counts.tolist() == [[6]]
    assert cold.plan.candidate_counts.gt(0).all()
    assert cold.plan.ranked_cluster_indices.ge(0).all()
    assert cold.resolved_clusters.resident_bucket_ids.tolist() == [[[-1]]]
    assert cold.hit_attn.item() == pytest.approx(1.0)
    assert not cold.resolved_clusters.hit_gate_ready.any()
    assert not cold.resolved_clusters.clustered_token_counts.any()
    scratch = index._draft_selection_scratch
    assert scratch is not None

    index.begin_proposal(["request"])
    try:
        resident_view = index._gpu_index_residency.get_active_view(
            "layer", ["request"], device
        )
        logical_cluster_ids, logical_page_ids, logical_page_token_counts = (
            index._build_resident_exact_cluster_selection(
                resident_view,
                cold_sparse_indices.clamp_min(0),
                cold_sparse_indices >= 0,
            )
        )
    finally:
        index.end_proposal()
    assert logical_page_token_counts.sum().item() == 2

    index.cluster_store.admit_resident_clusters(
        "layer",
        logical_cluster_ids,
        logical_page_ids,
    )
    index.cluster_store.get_resident_page_storage("layer")
    torch.cuda.current_stream().synchronize()

    index.begin_proposal(["request"])
    try:
        warm = index.select_segmented(**selection_kwargs)
        indexed = index.get_indexed_selection(
            "layer",
            RetroSpecAttentionLevel.SPARSE,
            torch.tensor([0], dtype=torch.int64, device=device),
            torch.tensor([0], dtype=torch.int64, device=device),
        )
        verification = index.materialize_indexed_reference(indexed)
    finally:
        index.end_proposal()

    assert warm.resolved_clusters is not None
    assert warm.prefetch_miss_cluster_ids is None
    assert warm.prefetch_miss_positions is None
    assert warm.prefetch_miss_count is None
    assert index.cluster_store.num_resident_pages("layer") == 1
    assert warm.exact_token_counts.tolist() == [[8]]
    assert warm.plan.candidate_counts.gt(0).all()
    assert warm.plan.ranked_cluster_indices.ge(0).all()
    assert warm.resolved_clusters.resident_bucket_ids.ge(0).all()
    assert warm.resolved_clusters.hit_gate_ready.all()
    assert warm.hit_attn.item() == pytest.approx(warm.plan.sparse_attn.item())

    assert verification.resolved_pages is None
    assert verification.exact_token_counts.tolist() == [[8]]
    assert verification.estimation_token_counts.tolist() == [[[2]]]
    assert verification.attention_mass.item() == pytest.approx(
        warm.plan.sparse_attn.item()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ready_selected_replay_admits_current_topk_before_draft_attention():
    device = torch.device("cuda")
    index = make_index(
        cache_ratio=0.5,
        pin_memory=True,
        replay_mode="ready_selected",
    )
    keys, values = make_cache()
    keys = keys.to(device=device, dtype=torch.bfloat16)
    values = values.to(device=device, dtype=torch.bfloat16)
    block_table = torch.arange(7, dtype=torch.int32, device=device).view(1, -1)
    build_index(index, 10, keys, values, block_table)
    record_selection = Mock()
    index.selection_provenance.record_draft_selection = record_selection

    index.begin_proposal(["request"])
    try:
        with patch.object(
            index,
            "_resolve_ranked_draft_clusters",
            wraps=index._resolve_ranked_draft_clusters,
        ) as resolve:
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
                proposal_round=2,
            )
    finally:
        index.end_proposal()

    assert [call.kwargs["emit_misses"] for call in resolve.call_args_list] == [
        True,
        False,
    ]
    assert index.cluster_store.num_resident_pages("layer") == 1
    assert selection.resolved_clusters is not None
    assert selection.resolved_clusters.hit_gate_ready.all()
    assert selection.resolved_clusters.miss_cluster_counts.sum().item() == 0
    assert selection.prefetch_miss_cluster_ids is None
    assert selection.prefetch_miss_positions is None
    assert selection.prefetch_miss_count is None
    assert selection.prefetch_num_groups == 0
    assert selection.prefetch_num_ranks == 0
    assert [call.kwargs["snapshot"] for call in record_selection.call_args_list] == [
        "before_ready",
        "used",
    ]
    assert all(
        call.kwargs["proposal_round"] == 2 for call in record_selection.call_args_list
    )
    index.close()


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
        with (
            patch.object(
                index,
                "_select_cluster_zones",
                side_effect=AssertionError("CUDA path constructed packed zones"),
            ),
            patch.object(
                index,
                "_make_reference_plan",
                side_effect=AssertionError("CUDA path used the reference plan"),
            ),
        ):
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

    assert selection.resolved_clusters.cluster_handles.device.type == "cuda"
    assert selection.resolved_clusters.resident_bucket_ids.device.type == "cuda"
    assert selection.exact_token_counts.tolist() == [[6]]
    assert selection.plan.ranked_cluster_indices[0, 0, 0].item() >= 0
