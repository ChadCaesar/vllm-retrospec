# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton

from .resident_kernel_helpers import (
    _find_resident_buckets,
    _record_resident_lookup_statistics,
)


@triton.jit
def _resolve_compact_draft_pages_kernel(
    ranked_values,
    arena_resident_table_buckets,
    arena_cluster_page_starts,
    arena_cluster_page_counts,
    arena_page_ids,
    arena_page_token_counts,
    arena_cluster_offsets,
    arena_page_offsets,
    request_slot_ids,
    active_mask,
    table_handles,
    table_versions,
    table_page_counts,
    table_page_slots,
    table_hit_gate_ready,
    table_last_access_epochs,
    access_epoch,
    fallback_token_counts,
    sparse_cluster_indices,
    cluster_handles,
    output_page_slots,
    output_page_token_counts,
    output_page_counts,
    output_clustered_token_counts,
    output_hit_attention_by_head,
    output_selected_counts,
    output_hit_counts,
    output_miss_counts,
    output_gate_ready,
    output_miss_handles,
    output_miss_positions,
    output_miss_count,
    statistics_buffer,
    ranked_stride_0,
    ranked_stride_1,
    ranked_stride_2,
    fallback_count_row_stride,
    table_page_stride,
    ARENA_CLUSTER_CAPACITY: tl.constexpr,
    ARENA_PAGE_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SPARSE_WIDTH: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    TABLE_CAPACITY: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_OUTPUT_PAGES: tl.constexpr,
    BLOCK_SPARSE: tl.constexpr,
    EMIT_MISSES: tl.constexpr,
    UPDATE_STATISTICS: tl.constexpr,
    RESIDENT_HIT_COUNTER_INDEX: tl.constexpr,
    RESIDENT_MISS_COUNTER_INDEX: tl.constexpr,
    RESIDENT_PAGE_COUNTER_INDEX: tl.constexpr,
    SELECTED_CLUSTER_COUNTER_INDEX: tl.constexpr,
    BOUND_DIRECT_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_LOOKUP_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_MISS_COUNTER_INDEX: tl.constexpr,
    HASH_PROBE_STEP_COUNTER_INDEX: tl.constexpr,
    HASH_MAX_PROBE_COUNTER_INDEX: tl.constexpr,
    BINDING_INVALIDATION_COUNTER_INDEX: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // NUM_KV_HEADS
    kv_head_idx = row % NUM_KV_HEADS

    request_slot = tl.load(request_slot_ids + batch_idx)
    request_active = tl.load(active_mask + batch_idx).to(tl.int1)
    request_valid = request_active & (request_slot >= 0)
    safe_slot = tl.maximum(request_slot, 0).to(tl.int64)
    request_cluster_offset = tl.load(
        arena_cluster_offsets + safe_slot, mask=request_valid, other=0
    ).to(tl.int64)
    request_page_offset = tl.load(
        arena_page_offsets + safe_slot, mask=request_valid, other=0
    ).to(tl.int64)

    output_page_offsets = tl.arange(0, BLOCK_OUTPUT_PAGES)
    tl.store(
        output_page_slots + row * PAGE_CAPACITY + output_page_offsets,
        -1,
        mask=output_page_offsets < PAGE_CAPACITY,
    )
    tl.store(
        output_page_token_counts + row * PAGE_CAPACITY + output_page_offsets,
        0,
        mask=output_page_offsets < PAGE_CAPACITY,
    )

    ranks = tl.arange(0, BLOCK_SPARSE)
    valid_ranks = ranks < SPARSE_WIDTH
    cluster_offsets = row * SPARSE_WIDTH + ranks
    local_cluster_indices = tl.load(
        sparse_cluster_indices + cluster_offsets, mask=valid_ranks, other=-1
    ).to(tl.int64)
    selected_cluster_handles = tl.load(
        cluster_handles + cluster_offsets, mask=valid_ranks, other=-1
    ).to(tl.int64)
    selected = (
        request_valid
        & valid_ranks
        & (local_cluster_indices >= 0)
        & (selected_cluster_handles >= 0)
    )
    ranked_offsets = (
        batch_idx * ranked_stride_0
        + kv_head_idx * ranked_stride_1
        + ranks * ranked_stride_2
    )
    arena_cluster_storage_offsets = (
        kv_head_idx * ARENA_CLUSTER_CAPACITY
        + request_cluster_offset
        + tl.maximum(local_cluster_indices, 0)
    )

    logical_page_starts = tl.load(
        arena_cluster_page_starts + arena_cluster_storage_offsets,
        mask=selected,
        other=0,
    ).to(tl.int64)
    logical_page_counts = tl.load(
        arena_cluster_page_counts + arena_cluster_storage_offsets,
        mask=selected,
        other=0,
    ).to(tl.int32)

    bound_buckets = tl.load(
        arena_resident_table_buckets + arena_cluster_storage_offsets,
        mask=selected,
        other=-1,
    ).to(tl.int64)
    (
        matched_buckets,
        direct_matches,
        fallback_lookups,
        probe_counts,
    ) = _find_resident_buckets(
        selected_cluster_handles,
        selected,
        bound_buckets,
        table_handles,
        table_versions,
        TABLE_CAPACITY=TABLE_CAPACITY,
        BLOCK_WIDTH=BLOCK_SPARSE,
        TRACK_STATISTICS=UPDATE_STATISTICS,
    )

    found = matched_buckets >= 0
    safe_buckets = tl.maximum(matched_buckets, 0)
    versions_before = tl.atomic_add(
        table_versions + safe_buckets, 0, mask=found, sem="acquire"
    )
    stored_handles = tl.load(table_handles + safe_buckets, mask=found, other=-1)
    resident_page_counts = tl.load(
        table_page_counts + safe_buckets, mask=found, other=0
    )
    cluster_gate_ready = tl.load(
        table_hit_gate_ready + safe_buckets, mask=found, other=0
    )

    page_offsets = tl.arange(0, BLOCK_PAGES)
    valid_page_offsets = page_offsets < MAX_PAGES
    arena_page_indices = (
        request_page_offset + logical_page_starts[:, None] + page_offsets[None, :]
    )
    arena_page_storage_offsets = kv_head_idx * ARENA_PAGE_CAPACITY + arena_page_indices
    valid_logical_pages = (
        selected[:, None]
        & valid_page_offsets[None, :]
        & (page_offsets[None, :] < logical_page_counts[:, None])
    )
    logical_page_ids = tl.load(
        arena_page_ids + arena_page_storage_offsets,
        mask=valid_logical_pages,
        other=-1,
    ).to(tl.int64)
    logical_token_counts = tl.load(
        arena_page_token_counts + arena_page_storage_offsets,
        mask=valid_logical_pages,
        other=0,
    ).to(tl.int32)
    valid_logical_pages &= (logical_page_ids >= 0) & (logical_token_counts > 0)
    actual_page_counts = tl.sum(valid_logical_pages.to(tl.int32), axis=1)

    resident_slots = tl.load(
        table_page_slots
        + safe_buckets[:, None] * table_page_stride
        + page_offsets[None, :],
        mask=(
            found[:, None]
            & valid_page_offsets[None, :]
            & (page_offsets[None, :] < resident_page_counts[:, None])
            & (page_offsets[None, :] < actual_page_counts[:, None])
        ),
        other=-1,
    )
    versions_after = tl.atomic_add(
        table_versions + safe_buckets, 0, mask=found, sem="acquire"
    )
    resolved_page_counts = tl.sum(
        (valid_logical_pages & (resident_slots >= 0)).to(tl.int32), axis=1
    )
    stable_hits = (
        selected
        & found
        & (stored_handles == selected_cluster_handles)
        & (versions_before == versions_after)
        & ((versions_before & 1) == 0)
        & (actual_page_counts > 0)
        & (resident_page_counts >= actual_page_counts)
        & (resolved_page_counts == actual_page_counts)
    )
    misses = selected & ~stable_hits

    tl.store(
        arena_resident_table_buckets + arena_cluster_storage_offsets,
        tl.where(stable_hits, safe_buckets, -1),
        mask=selected,
    )

    tl.atomic_max(
        table_last_access_epochs + safe_buckets,
        access_epoch,
        mask=stable_hits,
        sem="relaxed",
    )

    selected_page_counts = tl.where(stable_hits, actual_page_counts, 0)
    compact_ends = tl.cumsum(selected_page_counts, axis=0)
    compact_starts = compact_ends - selected_page_counts
    compact_offsets = compact_starts[:, None] + page_offsets[None, :]
    valid_output_pages = (
        stable_hits[:, None] & valid_logical_pages & (compact_offsets < PAGE_CAPACITY)
    )
    tl.store(
        output_page_slots + row * PAGE_CAPACITY + compact_offsets,
        resident_slots,
        mask=valid_output_pages,
    )
    tl.store(
        output_page_token_counts + row * PAGE_CAPACITY + compact_offsets,
        logical_token_counts,
        mask=valid_output_pages,
    )

    fallback_offsets = row * fallback_count_row_stride + ranks
    fallback_counts = tl.load(
        fallback_token_counts + fallback_offsets, mask=valid_ranks, other=0
    )
    tl.store(
        fallback_token_counts + fallback_offsets,
        tl.where(misses, fallback_counts, 0),
        mask=valid_ranks,
    )

    num_selected = tl.sum(selected.to(tl.int32), axis=0)
    num_hits = tl.sum(stable_hits.to(tl.int32), axis=0)
    num_misses = tl.sum(misses.to(tl.int32), axis=0)

    if EMIT_MISSES:
        miss_prefix = tl.cumsum(misses.to(tl.int32), axis=0)
        miss_base = tl.atomic_add(output_miss_count, num_misses)
        miss_slots = miss_base + miss_prefix - 1
        tl.store(
            output_miss_handles + miss_slots, selected_cluster_handles, mask=misses
        )
        tl.store(output_miss_positions + miss_slots, cluster_offsets, mask=misses)

    ranked_scores = tl.load(ranked_values + ranked_offsets, mask=selected, other=0.0)
    hit_attention = tl.sum(tl.where(stable_hits, ranked_scores, 0.0), axis=0)
    clustered_tokens_by_rank = tl.sum(
        tl.where(stable_hits[:, None] & valid_logical_pages, logical_token_counts, 0),
        axis=1,
    )
    num_resident_pages = tl.sum(selected_page_counts, axis=0)
    num_clustered_tokens = tl.sum(clustered_tokens_by_rank, axis=0)
    tl.store(output_page_counts + row, num_resident_pages)
    tl.store(output_clustered_token_counts + row, num_clustered_tokens)
    tl.store(output_hit_attention_by_head + row, hit_attention)
    tl.store(output_selected_counts + row, num_selected)
    tl.store(output_hit_counts + row, num_hits)
    tl.store(output_miss_counts + row, num_misses)
    tl.store(
        output_gate_ready + row,
        tl.sum((stable_hits & cluster_gate_ready).to(tl.int32), axis=0) > 0,
    )

    if UPDATE_STATISTICS:
        tl.atomic_add(
            statistics_buffer + RESIDENT_HIT_COUNTER_INDEX,
            num_hits.to(tl.int64),
            sem="relaxed",
        )
        tl.atomic_add(
            statistics_buffer + RESIDENT_MISS_COUNTER_INDEX,
            num_misses.to(tl.int64),
            sem="relaxed",
        )
        tl.atomic_add(
            statistics_buffer + RESIDENT_PAGE_COUNTER_INDEX,
            num_resident_pages.to(tl.int64),
            sem="relaxed",
        )
        tl.atomic_add(
            statistics_buffer + SELECTED_CLUSTER_COUNTER_INDEX,
            num_selected.to(tl.int64),
            sem="relaxed",
        )
        _record_resident_lookup_statistics(
            statistics_buffer,
            selected,
            bound_buckets,
            direct_matches,
            fallback_lookups,
            probe_counts,
            stable_hits,
            BOUND_DIRECT_HIT_COUNTER_INDEX=BOUND_DIRECT_HIT_COUNTER_INDEX,
            HASH_FALLBACK_LOOKUP_COUNTER_INDEX=(HASH_FALLBACK_LOOKUP_COUNTER_INDEX),
            HASH_FALLBACK_HIT_COUNTER_INDEX=HASH_FALLBACK_HIT_COUNTER_INDEX,
            HASH_FALLBACK_MISS_COUNTER_INDEX=HASH_FALLBACK_MISS_COUNTER_INDEX,
            HASH_PROBE_STEP_COUNTER_INDEX=HASH_PROBE_STEP_COUNTER_INDEX,
            HASH_MAX_PROBE_COUNTER_INDEX=HASH_MAX_PROBE_COUNTER_INDEX,
            BINDING_INVALIDATION_COUNTER_INDEX=(BINDING_INVALIDATION_COUNTER_INDEX),
        )


@triton.jit
def _resolve_ranked_draft_buckets_kernel(
    ranked_values,
    ranked_indices,
    candidate_counts,
    arena_cluster_ids,
    arena_resident_table_buckets,
    arena_cluster_token_counts,
    arena_cluster_page_counts,
    arena_cluster_offsets,
    arena_generations,
    request_slot_ids,
    active_mask,
    table_handles,
    table_versions,
    table_page_counts,
    table_page_slots,
    table_hit_gate_ready,
    table_last_access_epochs,
    access_epoch,
    output_valid_rows,
    output_request_slot_ids,
    output_request_slot_generations,
    output_cluster_handles,
    output_resident_buckets,
    output_clustered_token_counts,
    output_hit_attention_by_head,
    output_selected_counts,
    output_hit_counts,
    output_miss_counts,
    output_gate_ready,
    output_miss_handles,
    output_miss_positions,
    output_miss_count,
    statistics_buffer,
    ranked_value_stride_0,
    ranked_value_stride_1,
    ranked_value_stride_2,
    ranked_index_stride_0,
    ranked_index_stride_1,
    ranked_index_stride_2,
    table_page_stride,
    max_pages_per_cluster,
    ARENA_CLUSTER_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SPARSE_WIDTH: tl.constexpr,
    TABLE_CAPACITY: tl.constexpr,
    BLOCK_SPARSE: tl.constexpr,
    CAPTURE_REQUEST_DESCRIPTORS: tl.constexpr,
    EMIT_MISSES: tl.constexpr,
    UPDATE_STATISTICS: tl.constexpr,
    RETRIEVAL_RATIO: tl.constexpr,
    RESIDENT_HIT_COUNTER_INDEX: tl.constexpr,
    RESIDENT_MISS_COUNTER_INDEX: tl.constexpr,
    RESIDENT_PAGE_COUNTER_INDEX: tl.constexpr,
    SELECTED_CLUSTER_COUNTER_INDEX: tl.constexpr,
    BOUND_DIRECT_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_LOOKUP_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_MISS_COUNTER_INDEX: tl.constexpr,
    HASH_PROBE_STEP_COUNTER_INDEX: tl.constexpr,
    HASH_MAX_PROBE_COUNTER_INDEX: tl.constexpr,
    BINDING_INVALIDATION_COUNTER_INDEX: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // NUM_KV_HEADS
    kv_head_idx = row % NUM_KV_HEADS
    group_offset = batch_idx * NUM_KV_HEADS + kv_head_idx

    request_slot = tl.load(request_slot_ids + batch_idx)
    request_active = tl.load(active_mask + batch_idx).to(tl.int1)
    request_valid = request_active & (request_slot >= 0)
    safe_slot = tl.maximum(request_slot, 0).to(tl.int64)
    request_cluster_offset = tl.load(
        arena_cluster_offsets + safe_slot, mask=request_valid, other=0
    ).to(tl.int64)

    publish_row = kv_head_idx == 0
    tl.store(output_valid_rows + batch_idx, request_active, mask=publish_row)
    if CAPTURE_REQUEST_DESCRIPTORS:
        descriptor_valid = publish_row & (request_slot >= 0)
        request_generation = tl.load(
            arena_generations + safe_slot, mask=descriptor_valid, other=-1
        )
        tl.store(output_request_slot_ids + batch_idx, request_slot, mask=publish_row)
        tl.store(
            output_request_slot_generations + batch_idx,
            tl.where(descriptor_valid, request_generation, -1),
            mask=publish_row,
        )

    candidate_count = tl.load(candidate_counts + group_offset).to(tl.int32)
    retrieval_count = tl.ceil(candidate_count.to(tl.float32) * RETRIEVAL_RATIO).to(
        tl.int32
    )
    retrieval_count = tl.minimum(retrieval_count, candidate_count)
    ranks = tl.arange(0, BLOCK_SPARSE)
    valid_ranks = ranks < SPARSE_WIDTH
    row_offsets = row * SPARSE_WIDTH + ranks
    ranked_index_offsets = (
        batch_idx * ranked_index_stride_0
        + kv_head_idx * ranked_index_stride_1
        + ranks * ranked_index_stride_2
    )
    local_cluster_indices = tl.load(
        ranked_indices + ranked_index_offsets,
        mask=request_valid & valid_ranks & (ranks < retrieval_count),
        other=-1,
    ).to(tl.int64)
    selected = (
        request_valid
        & valid_ranks
        & (ranks < retrieval_count)
        & (local_cluster_indices >= 0)
    )
    cluster_storage_offsets = (
        kv_head_idx * ARENA_CLUSTER_CAPACITY
        + request_cluster_offset
        + tl.maximum(local_cluster_indices, 0)
    )
    cluster_token_counts = tl.load(
        arena_cluster_token_counts + cluster_storage_offsets,
        mask=selected,
        other=0,
    ).to(tl.int32)
    selected &= cluster_token_counts > 0
    cluster_handles = tl.load(
        arena_cluster_ids + cluster_storage_offsets, mask=selected, other=-1
    ).to(tl.int64)
    selected &= cluster_handles >= 0

    logical_page_counts = tl.load(
        arena_cluster_page_counts + cluster_storage_offsets,
        mask=selected,
        other=0,
    ).to(tl.int32)
    bound_buckets = tl.load(
        arena_resident_table_buckets + cluster_storage_offsets,
        mask=selected,
        other=-1,
    ).to(tl.int64)
    (
        matched_buckets,
        direct_matches,
        fallback_lookups,
        probe_counts,
    ) = _find_resident_buckets(
        cluster_handles,
        selected,
        bound_buckets,
        table_handles,
        table_versions,
        TABLE_CAPACITY=TABLE_CAPACITY,
        BLOCK_WIDTH=BLOCK_SPARSE,
        TRACK_STATISTICS=UPDATE_STATISTICS,
    )

    found = matched_buckets >= 0
    safe_buckets = tl.maximum(matched_buckets, 0)
    versions_before = tl.atomic_add(
        table_versions + safe_buckets, 0, mask=found, sem="acquire"
    )
    stored_handles = tl.load(table_handles + safe_buckets, mask=found, other=-1)
    resident_page_counts = tl.load(
        table_page_counts + safe_buckets, mask=found, other=0
    ).to(tl.int32)
    cluster_gate_ready = tl.load(
        table_hit_gate_ready + safe_buckets, mask=found, other=0
    )

    page_range_valid = (
        selected
        & found
        & (logical_page_counts > 0)
        & (logical_page_counts <= max_pages_per_cluster)
    )
    first_page_slots = tl.load(
        table_page_slots + safe_buckets * table_page_stride,
        mask=page_range_valid,
        other=-1,
    )
    last_page_offsets = tl.maximum(logical_page_counts - 1, 0)
    last_page_slots = tl.load(
        table_page_slots + safe_buckets * table_page_stride + last_page_offsets,
        mask=page_range_valid,
        other=-1,
    )
    versions_after = tl.atomic_add(
        table_versions + safe_buckets, 0, mask=found, sem="acquire"
    )
    stable_hits = (
        selected
        & found
        & (stored_handles == cluster_handles)
        & (versions_before == versions_after)
        & ((versions_before & 1) == 0)
        & page_range_valid
        & (resident_page_counts == logical_page_counts)
        & (first_page_slots >= 0)
        & (last_page_slots >= 0)
    )
    misses = selected & ~stable_hits

    tl.store(
        output_cluster_handles + row_offsets,
        tl.where(selected, cluster_handles, -1),
        mask=valid_ranks,
    )
    tl.store(
        output_resident_buckets + row_offsets,
        tl.where(stable_hits, safe_buckets, -1),
        mask=valid_ranks,
    )
    tl.store(
        arena_resident_table_buckets + cluster_storage_offsets,
        tl.where(stable_hits, safe_buckets, -1),
        mask=selected,
    )
    tl.atomic_max(
        table_last_access_epochs + safe_buckets,
        access_epoch,
        mask=stable_hits,
        sem="relaxed",
    )

    num_selected = tl.sum(selected.to(tl.int32), axis=0)
    num_hits = tl.sum(stable_hits.to(tl.int32), axis=0)
    num_misses = tl.sum(misses.to(tl.int32), axis=0)
    selected_page_counts = tl.where(stable_hits, logical_page_counts, 0)
    num_resident_pages = tl.sum(selected_page_counts, axis=0)
    num_clustered_tokens = tl.sum(
        tl.where(stable_hits, cluster_token_counts, 0), axis=0
    )

    if EMIT_MISSES:
        miss_prefix = tl.cumsum(misses.to(tl.int32), axis=0)
        miss_base = tl.atomic_add(output_miss_count, num_misses)
        miss_slots = miss_base + miss_prefix - 1
        tl.store(output_miss_handles + miss_slots, cluster_handles, mask=misses)
        tl.store(output_miss_positions + miss_slots, row_offsets, mask=misses)

    ranked_value_offsets = (
        batch_idx * ranked_value_stride_0
        + kv_head_idx * ranked_value_stride_1
        + ranks * ranked_value_stride_2
    )
    ranked_scores = tl.load(
        ranked_values + ranked_value_offsets, mask=selected, other=0.0
    )
    hit_attention = tl.sum(tl.where(stable_hits, ranked_scores, 0.0), axis=0)
    tl.store(output_clustered_token_counts + row, num_clustered_tokens)
    tl.store(output_hit_attention_by_head + row, hit_attention)
    tl.store(output_selected_counts + row, num_selected)
    tl.store(output_hit_counts + row, num_hits)
    tl.store(output_miss_counts + row, num_misses)
    tl.store(
        output_gate_ready + row,
        tl.sum((stable_hits & cluster_gate_ready).to(tl.int32), axis=0) > 0,
    )

    if UPDATE_STATISTICS:
        tl.atomic_add(
            statistics_buffer + RESIDENT_HIT_COUNTER_INDEX,
            num_hits.to(tl.int64),
            sem="relaxed",
        )
        tl.atomic_add(
            statistics_buffer + RESIDENT_MISS_COUNTER_INDEX,
            num_misses.to(tl.int64),
            sem="relaxed",
        )
        tl.atomic_add(
            statistics_buffer + RESIDENT_PAGE_COUNTER_INDEX,
            num_resident_pages.to(tl.int64),
            sem="relaxed",
        )
        tl.atomic_add(
            statistics_buffer + SELECTED_CLUSTER_COUNTER_INDEX,
            num_selected.to(tl.int64),
            sem="relaxed",
        )
        _record_resident_lookup_statistics(
            statistics_buffer,
            selected,
            bound_buckets,
            direct_matches,
            fallback_lookups,
            probe_counts,
            stable_hits,
            BOUND_DIRECT_HIT_COUNTER_INDEX=BOUND_DIRECT_HIT_COUNTER_INDEX,
            HASH_FALLBACK_LOOKUP_COUNTER_INDEX=(HASH_FALLBACK_LOOKUP_COUNTER_INDEX),
            HASH_FALLBACK_HIT_COUNTER_INDEX=HASH_FALLBACK_HIT_COUNTER_INDEX,
            HASH_FALLBACK_MISS_COUNTER_INDEX=HASH_FALLBACK_MISS_COUNTER_INDEX,
            HASH_PROBE_STEP_COUNTER_INDEX=HASH_PROBE_STEP_COUNTER_INDEX,
            HASH_MAX_PROBE_COUNTER_INDEX=HASH_MAX_PROBE_COUNTER_INDEX,
            BINDING_INVALIDATION_COUNTER_INDEX=(BINDING_INVALIDATION_COUNTER_INDEX),
        )


@triton.jit
def _finalize_ranked_compact_draft_attention_kernel(
    ranked_values,
    candidate_counts,
    request_slot_ids,
    active_mask,
    hit_attention_by_head,
    gate_ready,
    output_attention,
    sparse_attention,
    expanded_attention,
    NUM_KV_HEADS: tl.constexpr,
    RANKING_WIDTH: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
    RANKED_STRIDE_0: tl.constexpr,
    RANKED_STRIDE_1: tl.constexpr,
    RANKED_STRIDE_2: tl.constexpr,
    RETRIEVAL_RATIO: tl.constexpr,
    ESTIMATION_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_offsets = tl.arange(0, BLOCK_HEADS)
    rank_offsets = tl.arange(0, BLOCK_RANK)
    valid_heads = head_offsets < NUM_KV_HEADS
    row_offsets = batch_idx * NUM_KV_HEADS + head_offsets

    candidate_count = tl.load(
        candidate_counts + row_offsets, mask=valid_heads, other=0
    ).to(tl.int32)
    retrieval_count = tl.ceil(candidate_count.to(tl.float32) * RETRIEVAL_RATIO).to(
        tl.int32
    )
    retrieval_count = tl.minimum(retrieval_count, candidate_count)
    estimation_count = tl.ceil(candidate_count.to(tl.float32) * ESTIMATION_RATIO).to(
        tl.int32
    )
    estimation_count = tl.minimum(estimation_count, candidate_count - retrieval_count)
    expanded_count = tl.minimum(retrieval_count * 2, retrieval_count + estimation_count)

    score_offsets = (
        batch_idx * RANKED_STRIDE_0
        + head_offsets[:, None] * RANKED_STRIDE_1
        + rank_offsets[None, :] * RANKED_STRIDE_2
    )
    score_mask = valid_heads[:, None] & (rank_offsets[None, :] < RANKING_WIDTH)
    scores = tl.load(ranked_values + score_offsets, mask=score_mask, other=0.0)
    sparse_by_head = tl.sum(
        tl.where(rank_offsets[None, :] < retrieval_count[:, None], scores, 0.0),
        axis=1,
    )
    expanded_by_head = tl.sum(
        tl.where(rank_offsets[None, :] < expanded_count[:, None], scores, 0.0),
        axis=1,
    )
    sparse_by_head = tl.where(candidate_count > 0, sparse_by_head, 1.0)
    expanded_by_head = tl.where(candidate_count > 0, expanded_by_head, 1.0)

    hit_attention = tl.load(
        hit_attention_by_head + row_offsets, mask=valid_heads, other=0.0
    )
    head_gate_ready = tl.load(gate_ready + row_offsets, mask=valid_heads, other=0)
    draft_by_head = tl.where(
        (candidate_count > 0) & head_gate_ready, hit_attention, 1.0
    )

    sparse_value = (
        tl.sum(tl.where(valid_heads, sparse_by_head, 0.0), axis=0) / NUM_KV_HEADS
    )
    expanded_value = (
        tl.sum(tl.where(valid_heads, expanded_by_head, 0.0), axis=0) / NUM_KV_HEADS
    )
    draft_value = (
        tl.sum(tl.where(valid_heads, draft_by_head, 0.0), axis=0) / NUM_KV_HEADS
    )

    request_slot = tl.load(request_slot_ids + batch_idx)
    request_active = tl.load(active_mask + batch_idx).to(tl.int1)
    request_valid = request_active & (request_slot >= 0)
    tl.store(output_attention + batch_idx, tl.where(request_valid, draft_value, 1.0))
    tl.store(sparse_attention + batch_idx, tl.where(request_valid, sparse_value, 1.0))
    tl.store(
        expanded_attention + batch_idx,
        tl.where(request_valid, expanded_value, 1.0),
    )
