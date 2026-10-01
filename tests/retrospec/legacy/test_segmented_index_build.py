# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.segmented_index import (
    build_index,
    make_cache,
    make_empty_resident_view,
    make_index,
    materialize_reference,
)
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecResidentBatchView,
)
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecTokenSelectionPlan,
)


def test_index_shares_one_pinned_memory_budget_across_components():
    index = make_index()

    assert index.cluster_store._pinned_memory is index._pinned_memory
    assert index._gpu_index_residency._pinned_memory is index._pinned_memory

    index.close()
    assert index._pinned_memory.allocated_bytes == 0


def test_segmented_index_cpu_timer_uses_performance_stats():
    index = make_index()
    timer = nullcontext()
    index.performance_stats = Mock()
    index.performance_stats.cpu_timer.return_value = timer

    assert index._cpu_timer("draft_bucket_resolve_wall") is timer
    index.performance_stats.cpu_timer.assert_called_once_with(
        "draft_bucket_resolve_wall"
    )
    index.close()


def test_segmented_index_cpu_timer_is_disabled_without_stats():
    index = make_index()

    with index._cpu_timer("draft_bucket_resolve_wall"):
        pass

    index.close()


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


def test_sparse_verification_prefetch_forwards_compact_gpu_misses():
    index = make_index(cache_ratio=0.5, pin_memory=True)
    cluster_ids = torch.tensor([0, 1, -1, -1], dtype=torch.int64)
    positions = torch.tensor([0, 1, -1, -1], dtype=torch.int64)
    count = torch.tensor([2], dtype=torch.int32)
    selection = Mock(
        plan=Mock(layer_name="layer"),
        prefetch_miss_cluster_ids=cluster_ids,
        prefetch_miss_positions=positions,
        prefetch_miss_count=count,
        prefetch_num_groups=2,
        prefetch_num_ranks=2,
    )
    index.cluster_store.prefetch_resident_cluster_wave = Mock(return_value=True)

    index.prefetch_sparse_verification(
        selection,
        active_mask=torch.tensor([True, False]),
    )

    records = index.cluster_store.prefetch_resident_cluster_wave.call_args.args[0]
    assert len(records) == 1
    assert records[0].layer_name == "layer"
    assert records[0].miss_cluster_ids is cluster_ids
    assert records[0].miss_positions is positions
    assert records[0].miss_count is count
    assert records[0].num_groups == 2
    assert records[0].num_ranks == 2
    assert records[0].source == "draft"


def test_sparse_verification_prefetch_flushes_cluster_store_commands():
    index = make_index(cache_ratio=0.5, pin_memory=True)
    index.cluster_store.flush_resident_prefetch_commands = Mock()

    index.flush_sparse_verification_prefetch()

    index.cluster_store.flush_resident_prefetch_commands.assert_called_once_with()


def test_sparse_verification_prefetch_skips_empty_access_record():
    index = make_index(cache_ratio=0.5, pin_memory=True)
    selection = Mock(
        plan=Mock(layer_name="layer"),
        prefetch_miss_cluster_ids=torch.empty(0, dtype=torch.int64),
        prefetch_miss_positions=torch.empty(0, dtype=torch.int64),
        prefetch_miss_count=torch.zeros(1, dtype=torch.int32),
        prefetch_num_groups=1,
        prefetch_num_ranks=1,
    )
    index.cluster_store.prefetch_resident_cluster_wave = Mock()

    index.prefetch_sparse_verification(selection, active_mask=torch.tensor([True]))

    index.cluster_store.prefetch_resident_cluster_wave.assert_not_called()


def test_sparse_verification_prefetch_requires_pinned_cpu_backing():
    index = make_index(pin_memory=False)
    selection = Mock(
        prefetch_miss_cluster_ids=torch.tensor([0]),
        prefetch_miss_positions=torch.tensor([0]),
        prefetch_miss_count=torch.tensor([1], dtype=torch.int32),
        prefetch_num_groups=1,
        prefetch_num_ranks=1,
    )
    index.cluster_store.prefetch_resident_cluster_wave = Mock()

    index.prefetch_sparse_verification(
        selection,
        active_mask=torch.tensor([True]),
    )

    index.cluster_store.prefetch_resident_cluster_wave.assert_not_called()


def test_prefill_hint_requires_pinned_cpu_backing():
    index = make_index(pin_memory=False)
    index._gpu_index_residency.activate = Mock()

    submitted = index.prefetch_final_prefill_queries(
        ["request"],
        {"layer": (torch.ones(1, 1, 1), 1.0)},
    )

    assert submitted == 0
    index._gpu_index_residency.activate.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_draft_materialization_skips_resident_lookup_without_arena():
    device = torch.device("cuda")
    index = make_index(pin_memory=True)
    plan = RetroSpecTokenSelectionPlan(
        layer_name="layer",
        request_slot_ids=torch.full((1,), -1, dtype=torch.int64, device=device),
        request_slot_generations=torch.zeros(1, dtype=torch.int64, device=device),
        primary_exact_token_indices=torch.tensor(
            [[[0, 1]]], dtype=torch.int64, device=device
        ),
        primary_exact_token_mask=torch.ones(1, 1, 2, dtype=torch.bool, device=device),
        sparse_exact_cluster_indices=torch.full(
            (1, 1, 1), -1, dtype=torch.int32, device=device
        ),
        sparse_estimation_cluster_indices=torch.empty(
            1, 1, 0, dtype=torch.int32, device=device
        ),
        expanded_exact_cluster_indices=torch.full(
            (1, 1, 1), -1, dtype=torch.int32, device=device
        ),
        expanded_estimation_cluster_indices=torch.empty(
            1, 1, 0, dtype=torch.int32, device=device
        ),
        sparse_attn=torch.ones(1, device=device),
        expanded_attn=torch.ones(1, device=device),
    )
    index.cluster_store.resolve_ranked_draft_clusters = Mock()
    index._selection_plan_tables["layer"] = SimpleNamespace(
        head_size=1,
        dtype=torch.float32,
        valid_rows=torch.zeros(1, 1, dtype=torch.bool, device=device),
    )

    index.begin_proposal(["request"])
    try:
        selection = index._materialize_draft_selection(
            plan=plan,
            output_workspace=None,
            view=make_empty_resident_view(1, 1, device),
            active_mask=torch.ones(1, dtype=torch.bool, device=device),
        )
    finally:
        index.end_proposal()

    assert selection.resolved_pages is None
    assert selection.exact_token_counts.tolist() == [[2]]
    index.cluster_store.resolve_ranked_draft_clusters.assert_not_called()


def test_prefill_warmup_selection_obeys_page_budget():
    index = make_index(retrieval_ratio=0.5, prefill_warmup_multiplier=4)
    ranked_indices = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int64)
    candidate_counts = torch.tensor([[4]], dtype=torch.int32)
    view = RetroSpecResidentBatchView(
        arena=SimpleNamespace(
            cluster_page_counts=torch.tensor([[1, 2, 1, 1]], dtype=torch.int32),
            cluster_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.int64),
            cluster_offsets=torch.tensor([0], dtype=torch.int64),
        ),
        request_slot_ids=torch.tensor([0], dtype=torch.int64),
        max_num_clusters=4,
        max_pages_per_cluster=2,
        max_num_pages=5,
    )

    selected, mask = index._select_prefill_warmup(
        ranked_indices=ranked_indices,
        candidate_counts=candidate_counts,
        view=view,
        active_mask=torch.tensor([True]),
        warmup_page_budgets=torch.tensor([[3]], dtype=torch.int64),
    )

    assert selected.tolist() == [[[0, 1, 2, 3]]]
    assert mask.tolist() == [[[True, True, False, False]]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_final_prefill_query_prefetches_ranked_clusters():
    device = torch.device("cuda")
    index = make_index(
        retrieval_ratio=0.5,
        cache_ratio=1.0,
        pin_memory=True,
    )
    keys, values = make_cache()
    keys = keys.to(device=device, dtype=torch.bfloat16)
    values = values.to(device=device, dtype=torch.bfloat16)
    block_table = torch.arange(7, dtype=torch.int32, device=device).view(1, -1)
    build_index(index, 10, keys, values, block_table, prefill_complete=True)

    query = torch.ones(1, 1, 1, dtype=torch.bfloat16, device=device)
    score_resident_view = Mock(wraps=index._score_resident_view)
    index._score_resident_view = score_resident_view
    prefetch_wave = Mock(wraps=index.cluster_store.prefetch_resident_cluster_wave)
    index.cluster_store.prefetch_resident_cluster_wave = prefetch_wave

    submitted = index.prefetch_final_prefill_queries(
        ["request"],
        {"layer": (query, 1.0)},
    )
    assert submitted == 1
    assert index.cluster_store.num_resident_pages("layer") == 0
    index.cluster_store.synchronize_resident_prefetches(("layer",))

    assert index.cluster_store.num_resident_pages("layer") > 0
    score_resident_view.assert_called_once()
    assert score_resident_view.call_args.kwargs["prefill_hint"] is True
    records = prefetch_wave.call_args.args[0]
    assert len(records) == 1
    assert records[0].source == "prefill_hint"


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
    torch.testing.assert_close(
        segment.cluster_blocks.page_metadata.page_ids,
        metadata.page_ids,
    )
    torch.testing.assert_close(
        segment.cluster_blocks.page_metadata.page_token_counts,
        metadata.page_token_counts,
    )
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
        sparse_estimation_counts = sparse.estimation_token_counts.clone()
        sparse_estimation_keys = sparse.estimation_keys.clone()
        sparse_estimation_values = sparse.estimation_values.clone()
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
    assert sparse_estimation_counts[0, 0, 0].item() == 2
    assert sparse_estimation_keys[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert sparse_estimation_values[0, 0, 0, 0].item() == pytest.approx(10.0)

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
    assert record.segments[0].cluster_token_counts.shape == (1, 4)
    assert record.segments[0].cluster_token_counts.sum().item() == 8
    assert record.segments[1].cluster_token_counts.tolist() == [[2]]
    assert [
        (segment.indexed_start, segment.indexed_end) for segment in record.segments
    ] == [(2, 10), (10, 12)]


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
