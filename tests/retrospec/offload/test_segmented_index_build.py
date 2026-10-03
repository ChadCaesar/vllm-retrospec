# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.offload.segmented_index_helpers import (
    active_residency,
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
from vllm.v1.spec_decode.retrospec.offload import (
    segmented_build as segmented_build_module,
)
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
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
        cluster_offset = view.arena.cluster_offsets[slot].item()
        cluster_slice = slice(cluster_offset, cluster_offset + 2)
        assert view.arena.cluster_token_counts[0, cluster_slice].tolist() == [8, 0]
        assert view.arena.cluster_ids[0, cluster_offset].item() >= 0
        assert view.arena.cluster_ids[0, cluster_offset + 1].item() == -1


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
        "vllm.v1.spec_decode.retrospec.offload.segmented_build.segmented_kmeans",
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


def test_provenance_rejects_index_revision_changes_inside_proposal():
    index = make_index(replay_mode="trace")
    index._indices["layer"] = {"request": index._empty_index()}

    index.begin_proposal(["request"])
    try:
        revisions = index._get_proposal_index_revisions("layer", ["request"])
        assert revisions == (index._indices["layer"]["request"].revision,)

        index._indices["layer"]["request"] = index._empty_index()
        with pytest.raises(RuntimeError, match="changed inside a proposal"):
            index._get_proposal_index_revisions("layer", ["request"])
    finally:
        index.end_proposal()

    assert index._proposal_index_revisions == {}
    index.close()


def test_provenance_records_published_index_segment_ranges():
    index = make_index(replay_mode="trace")
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    record_segment = Mock()
    index.selection_provenance.record_index_segment = record_segment

    build_index(index, 10, keys, values, block_table)

    record_segment.assert_called_once()
    call = record_segment.call_args.kwargs
    assert call["request_id"] == "request"
    assert call["layer_name"] == "layer"
    assert call["indexed_start"] == 2
    assert call["indexed_end"] == 6
    assert call["cluster_start"] == 0
    assert call["assignments"].shape == (1, 4)
    assert call["cluster_sizes"].shape == (1, 2)
    index.close()


def test_freeze_resident_replay_matches_proposal_lifetime(monkeypatch):
    index = make_index(replay_mode="freeze_resident")
    begin_replay = Mock()
    end_replay = Mock()
    monkeypatch.setattr(index.cluster_store, "begin_resident_replay", begin_replay)
    monkeypatch.setattr(index.cluster_store, "end_resident_replay", end_replay)

    index.begin_proposal(["request"])
    begin_replay.assert_called_once_with(())
    assert index._proposal_resident_frozen
    index.end_proposal()

    end_replay.assert_called_once_with()
    assert not index._proposal_resident_frozen
    index.close()


def test_freeze_resident_replay_unwinds_failed_activation(monkeypatch):
    index = make_index(replay_mode="freeze_resident")
    begin_replay = Mock()
    end_replay = Mock()
    monkeypatch.setattr(index.cluster_store, "begin_resident_replay", begin_replay)
    monkeypatch.setattr(index.cluster_store, "end_resident_replay", end_replay)
    monkeypatch.setattr(
        index._gpu_index_residency,
        "activate",
        Mock(side_effect=RuntimeError("activation failed")),
    )

    with pytest.raises(RuntimeError, match="activation failed"):
        index.begin_proposal(["request"])

    begin_replay.assert_called_once_with(())
    end_replay.assert_called_once_with()
    assert not index._proposal_resident_frozen
    assert index._proposal_index_revisions == {}
    assert not index._proposal_active
    index.close()


def test_ready_selected_replay_requires_pinned_staging():
    with pytest.raises(ValueError, match="requires pinned CPU staging memory"):
        make_index(replay_mode="ready_selected")


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
            cluster_offset = int(view.arena.cluster_offsets[slot].item())
            cluster_slice = slice(cluster_offset, cluster_offset + num_clusters)
            assert torch.equal(
                view.arena.cluster_keys[:, cluster_slice], segment.cluster_keys
            )
            assert torch.equal(
                view.arena.cluster_values[:, cluster_slice], segment.cluster_values
            )
            assert torch.equal(
                view.arena.cluster_token_counts[:, cluster_slice],
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
        index._staged_segments[0].build_future.result()
        metadata_lookup = Mock(
            side_effect=AssertionError("publish rebuilt persistent block metadata")
        )
        monkeypatch.setattr(
            index.cluster_store,
            "get_cluster_block_metadata",
            metadata_lookup,
        )
        index.flush_staged_updates()
        metadata_lookup.assert_not_called()
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

    original_clustering = segmented_build_module.segmented_kmeans

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
        segmented_build_module,
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
        segmented_build_module,
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
        ({"cpu_page_build_workers": 0}, "positive"),
        ({"full_verify_gather_workers": 0}, "positive"),
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
