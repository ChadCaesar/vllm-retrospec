# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _resident_handle_hash(cluster_handle):
    """Match the uint32 avalanche hash used by the CPU table publisher."""
    value = cluster_handle.to(tl.int64)
    value = (value ^ (value >> 32)) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


@triton.jit
def _find_resident_buckets(
    cluster_handles,
    selected,
    bound_buckets,
    table_handles,
    table_versions,
    TABLE_CAPACITY: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    TRACK_STATISTICS: tl.constexpr,
):
    bound_valid = selected & (bound_buckets >= 0) & (bound_buckets < TABLE_CAPACITY)
    safe_bound_buckets = tl.maximum(bound_buckets, 0)
    bound_versions_before = tl.atomic_add(
        table_versions + safe_bound_buckets, 0, mask=bound_valid, sem="acquire"
    )
    bound_handles = tl.load(
        table_handles + safe_bound_buckets, mask=bound_valid, other=-1
    )
    bound_versions_after = tl.atomic_add(
        table_versions + safe_bound_buckets, 0, mask=bound_valid, sem="acquire"
    )
    bound_stable = (bound_versions_before == bound_versions_after) & (
        (bound_versions_before & 1) == 0
    )
    direct_matches = bound_valid & bound_stable & (bound_handles == cluster_handles)

    matched_buckets = tl.where(
        direct_matches,
        bound_buckets,
        tl.full((BLOCK_WIDTH,), -1, tl.int64),
    )
    first_buckets = _resident_handle_hash(cluster_handles) & (TABLE_CAPACITY - 1)
    fallback_lookups = selected & ~direct_matches
    searching = fallback_lookups
    searching_count = tl.sum(searching.to(tl.int32), axis=0)
    probe_counts = tl.zeros((BLOCK_WIDTH,), tl.int32)
    probe = 0
    while tl.condition((probe < 64) & (searching_count > 0), disable_licm=True):
        if TRACK_STATISTICS:
            probe_counts += searching.to(tl.int32)

        buckets = (first_buckets + probe) & (TABLE_CAPACITY - 1)
        versions_before = tl.atomic_add(
            table_versions + buckets, 0, mask=searching, sem="acquire"
        )
        stored_handles = tl.load(table_handles + buckets, mask=searching, other=-1)
        versions_after = tl.atomic_add(
            table_versions + buckets, 0, mask=searching, sem="acquire"
        )
        stable = (versions_before == versions_after) & ((versions_before & 1) == 0)
        matched = searching & stable & (stored_handles == cluster_handles)
        empty = searching & stable & (stored_handles == -1)
        matched_buckets = tl.where(matched, buckets, matched_buckets)
        searching &= ~matched & ~empty
        searching_count = tl.sum(searching.to(tl.int32), axis=0)
        probe += 1

    return matched_buckets, direct_matches, fallback_lookups, probe_counts


@triton.jit
def _record_resident_lookup_statistics(
    statistics_buffer,
    selected,
    bound_buckets,
    direct_matches,
    fallback_lookups,
    probe_counts,
    stable_hits,
    BOUND_DIRECT_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_LOOKUP_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_MISS_COUNTER_INDEX: tl.constexpr,
    HASH_PROBE_STEP_COUNTER_INDEX: tl.constexpr,
    HASH_MAX_PROBE_COUNTER_INDEX: tl.constexpr,
    BINDING_INVALIDATION_COUNTER_INDEX: tl.constexpr,
):
    direct_hits = direct_matches & stable_hits
    fallback_hits = fallback_lookups & stable_hits
    fallback_misses = fallback_lookups & ~stable_hits
    binding_invalidations = selected & (bound_buckets >= 0) & ~direct_hits

    tl.atomic_add(
        statistics_buffer + BOUND_DIRECT_HIT_COUNTER_INDEX,
        tl.sum(direct_hits.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_FALLBACK_LOOKUP_COUNTER_INDEX,
        tl.sum(fallback_lookups.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_FALLBACK_HIT_COUNTER_INDEX,
        tl.sum(fallback_hits.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_FALLBACK_MISS_COUNTER_INDEX,
        tl.sum(fallback_misses.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_PROBE_STEP_COUNTER_INDEX,
        tl.sum(probe_counts, axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_max(
        statistics_buffer + HASH_MAX_PROBE_COUNTER_INDEX,
        tl.max(probe_counts, axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + BINDING_INVALIDATION_COUNTER_INDEX,
        tl.sum(binding_invalidations.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
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


@triton.jit
def _reset_verification_miss_hash_kernel(
    miss_table_handles,
    output_miss_count,
    output_unique_miss_count,
    output_invalid_descriptor_count,
    TABLE_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < TABLE_CAPACITY
    tl.store(miss_table_handles + offsets, -1, mask=valid)

    reset_counter = offsets == 0
    tl.store(output_miss_count + offsets, 0, mask=reset_counter)
    tl.store(output_unique_miss_count + offsets, 0, mask=reset_counter)
    tl.store(output_invalid_descriptor_count + offsets, 0, mask=reset_counter)


@triton.jit
def _resolve_compact_verification_pages_vector_kernel(
    selected_cluster_indices,
    plan_valid_rows,
    request_slot_ids,
    request_slot_generations,
    arena_cluster_ids,
    arena_cluster_page_starts,
    arena_cluster_page_counts,
    arena_page_ids,
    arena_page_token_counts,
    arena_cluster_offsets,
    arena_page_offsets,
    arena_generations,
    table_handles,
    table_versions,
    table_page_counts,
    table_page_slots,
    table_last_access_epochs,
    access_epoch,
    output_resident_page_ids,
    output_staging_page_ids,
    output_page_token_counts,
    output_page_counts,
    output_selected_counts,
    output_hit_counts,
    output_miss_counts,
    miss_table_handles,
    miss_table_unique_indices,
    output_miss_hash_buckets,
    output_miss_page_offsets,
    output_miss_count,
    output_unique_handles,
    output_unique_logical_page_ids,
    output_unique_page_counts,
    output_unique_miss_count,
    output_invalid_descriptor_count,
    table_page_stride,
    output_unique_logical_page_stride,
    selected_stride_0,
    selected_stride_1,
    selected_stride_2,
    NUM_KV_HEADS: tl.constexpr,
    NUM_CLUSTERS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    ARENA_CLUSTER_CAPACITY: tl.constexpr,
    ARENA_PAGE_CAPACITY: tl.constexpr,
    TABLE_CAPACITY: tl.constexpr,
    MISS_TABLE_CAPACITY: tl.constexpr,
    BLOCK_CLUSTERS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_OUTPUT_PAGES: tl.constexpr,
):
    row = tl.program_id(0)
    query_idx = row // NUM_KV_HEADS
    kv_head_idx = row % NUM_KV_HEADS
    plan_valid = tl.load(plan_valid_rows + query_idx).to(tl.int1)
    request_slot = tl.load(request_slot_ids + query_idx).to(tl.int64)
    expected_generation = tl.load(request_slot_generations + query_idx).to(tl.int64)
    valid_slot = request_slot >= 0
    safe_slot = tl.maximum(request_slot, 0)
    actual_generation = tl.load(
        arena_generations + safe_slot, mask=valid_slot, other=-1
    ).to(tl.int64)
    descriptor_valid = (
        plan_valid & valid_slot & (actual_generation == expected_generation)
    )

    output_offsets = tl.arange(0, BLOCK_OUTPUT_PAGES)
    valid_output = output_offsets < PAGE_CAPACITY
    output_base = row * PAGE_CAPACITY
    tl.store(
        output_resident_page_ids + output_base + output_offsets,
        -1,
        mask=valid_output,
    )
    tl.store(
        output_staging_page_ids + output_base + output_offsets,
        -1,
        mask=valid_output,
    )
    tl.store(
        output_page_token_counts + output_base + output_offsets,
        0,
        mask=valid_output,
    )

    ranks = tl.arange(0, BLOCK_CLUSTERS)
    valid_ranks = ranks < NUM_CLUSTERS
    source_offsets = (
        query_idx * selected_stride_0
        + kv_head_idx * selected_stride_1
        + ranks * selected_stride_2
    )
    local_cluster_indices = tl.load(
        selected_cluster_indices + source_offsets, mask=valid_ranks, other=-1
    ).to(tl.int64)
    selected = valid_ranks & (local_cluster_indices >= 0)
    has_selected = tl.sum(selected.to(tl.int32), axis=0) > 0
    selected &= descriptor_valid

    request_cluster_offset = tl.load(
        arena_cluster_offsets + safe_slot, mask=descriptor_valid, other=0
    ).to(tl.int64)
    request_page_offset = tl.load(
        arena_page_offsets + safe_slot, mask=descriptor_valid, other=0
    ).to(tl.int64)
    absolute_cluster_indices = request_cluster_offset + tl.maximum(
        local_cluster_indices, 0
    )
    arena_cluster_indices = (
        kv_head_idx * ARENA_CLUSTER_CAPACITY + absolute_cluster_indices
    )
    handles = tl.load(
        arena_cluster_ids + arena_cluster_indices, mask=selected, other=-1
    ).to(tl.int64)
    logical_page_starts = tl.load(
        arena_cluster_page_starts + arena_cluster_indices, mask=selected, other=0
    ).to(tl.int64)
    logical_page_counts = tl.load(
        arena_cluster_page_counts + arena_cluster_indices, mask=selected, other=0
    ).to(tl.int32)
    selected &= (handles >= 0) & (logical_page_counts > 0)

    first_buckets = _resident_handle_hash(handles) & (TABLE_CAPACITY - 1)
    matched_buckets = tl.full((BLOCK_CLUSTERS,), -1, tl.int64)
    searching = selected
    for probe in tl.range(0, 64, num_stages=1, loop_unroll_factor=1):
        buckets = (first_buckets + probe) & (TABLE_CAPACITY - 1)
        versions_before = tl.atomic_add(
            table_versions + buckets, 0, mask=searching, sem="acquire"
        )
        stored_handles = tl.load(table_handles + buckets, mask=searching, other=-1)
        versions_after = tl.atomic_add(
            table_versions + buckets, 0, mask=searching, sem="acquire"
        )
        stable = (versions_before == versions_after) & ((versions_before & 1) == 0)
        matched = searching & stable & (stored_handles == handles)
        matched_buckets = tl.where(matched, buckets, matched_buckets)
        searching &= ~matched & ~(stable & (stored_handles == -1))

    found = matched_buckets >= 0
    safe_buckets = tl.maximum(matched_buckets, 0)
    versions_before = tl.atomic_add(
        table_versions + safe_buckets, 0, mask=found, sem="acquire"
    )
    stored_handles = tl.load(table_handles + safe_buckets, mask=found, other=-1)
    resident_page_counts = tl.load(
        table_page_counts + safe_buckets, mask=found, other=0
    )
    page_offsets = tl.arange(0, BLOCK_PAGES)
    valid_page_offsets = page_offsets < MAX_PAGES
    resident_slots = tl.load(
        table_page_slots
        + safe_buckets[:, None] * table_page_stride
        + page_offsets[None, :],
        mask=found[:, None]
        & valid_page_offsets[None, :]
        & (page_offsets[None, :] < resident_page_counts[:, None])
        & (page_offsets[None, :] < logical_page_counts[:, None]),
        other=-1,
    )
    versions_after = tl.atomic_add(
        table_versions + safe_buckets, 0, mask=found, sem="acquire"
    )
    resolved_page_counts = tl.sum(
        (
            valid_page_offsets[None, :]
            & (page_offsets[None, :] < logical_page_counts[:, None])
            & (resident_slots >= 0)
        ).to(tl.int32),
        axis=1,
    )
    stable_hits = (
        selected
        & found
        & (stored_handles == handles)
        & (versions_before == versions_after)
        & ((versions_before & 1) == 0)
        & (resident_page_counts >= logical_page_counts)
        & (resolved_page_counts == logical_page_counts)
    )

    arena_page_indices = (
        request_page_offset + logical_page_starts[:, None] + page_offsets[None, :]
    )
    arena_page_sources = kv_head_idx * ARENA_PAGE_CAPACITY + arena_page_indices
    valid_logical_pages = (
        selected[:, None]
        & valid_page_offsets[None, :]
        & (page_offsets[None, :] < logical_page_counts[:, None])
    )
    logical_page_ids = tl.load(
        arena_page_ids + arena_page_sources, mask=valid_logical_pages, other=-1
    )
    logical_page_token_counts = tl.load(
        arena_page_token_counts + arena_page_sources,
        mask=valid_logical_pages,
        other=0,
    )
    valid_logical_pages &= (logical_page_ids >= 0) & (logical_page_token_counts > 0)
    valid_counts = tl.sum(valid_logical_pages.to(tl.int32), axis=1)
    selected &= valid_counts == logical_page_counts
    stable_hits &= selected
    misses = selected & ~stable_hits

    tl.atomic_max(
        table_last_access_epochs + safe_buckets,
        access_epoch,
        mask=stable_hits,
        sem="relaxed",
    )

    selected_page_counts = tl.where(selected, logical_page_counts, 0)
    compact_ends = tl.cumsum(selected_page_counts, axis=0)
    compact_starts = compact_ends - selected_page_counts
    compact_offsets = compact_starts[:, None] + page_offsets[None, :]
    valid_compact_pages = valid_logical_pages & (compact_offsets < PAGE_CAPACITY)
    tl.store(
        output_page_token_counts + output_base + compact_offsets,
        logical_page_token_counts,
        mask=valid_compact_pages,
    )
    tl.store(
        output_resident_page_ids + output_base + compact_offsets,
        resident_slots,
        mask=valid_compact_pages & stable_hits[:, None],
    )

    miss_buckets = tl.full((BLOCK_CLUSTERS,), -1, tl.int64)
    claimed_unique = tl.full((BLOCK_CLUSTERS,), False, tl.int1)
    searching = misses
    searching_count = tl.sum(searching.to(tl.int32), axis=0)
    first_miss_buckets = handles & (MISS_TABLE_CAPACITY - 1)
    empty_handles = tl.full((BLOCK_CLUSTERS,), -1, tl.int64)
    inactive_handles = tl.full((BLOCK_CLUSTERS,), -2, tl.int64)
    probe = 0
    while tl.condition(
        (probe < MISS_TABLE_CAPACITY) & (searching_count > 0), disable_licm=True
    ):
        buckets = (first_miss_buckets + probe) & (MISS_TABLE_CAPACITY - 1)
        safe_buckets = tl.where(searching, buckets, 0)
        previous_handles = tl.atomic_cas(
            miss_table_handles + safe_buckets,
            tl.where(searching, empty_handles, inactive_handles),
            tl.where(searching, handles, inactive_handles),
            sem="acq_rel",
        )
        inserted = searching & (previous_handles == -1)
        matched = searching & (inserted | (previous_handles == handles))
        miss_buckets = tl.where(matched, buckets, miss_buckets)
        claimed_unique |= inserted
        searching &= ~matched
        searching_count = tl.sum(searching.to(tl.int32), axis=0)
        probe += 1

    unique_prefix = tl.cumsum(claimed_unique.to(tl.int32), axis=0)
    num_unique = tl.sum(claimed_unique.to(tl.int32), axis=0)
    unique_base = tl.atomic_add(output_unique_miss_count, num_unique)
    unique_indices = unique_base + unique_prefix - 1
    tl.store(
        miss_table_unique_indices + miss_buckets,
        unique_indices,
        mask=claimed_unique,
    )
    tl.store(output_unique_handles + unique_indices, handles, mask=claimed_unique)
    tl.store(
        output_unique_page_counts + unique_indices,
        logical_page_counts,
        mask=claimed_unique,
    )
    tl.store(
        output_unique_logical_page_ids
        + unique_indices[:, None] * output_unique_logical_page_stride
        + page_offsets[None, :],
        tl.where(valid_logical_pages, logical_page_ids, -1),
        mask=claimed_unique[:, None] & valid_page_offsets[None, :],
    )

    miss_prefix = tl.cumsum(misses.to(tl.int32), axis=0)
    num_misses = tl.sum(misses.to(tl.int32), axis=0)
    miss_base = tl.atomic_add(output_miss_count, num_misses)
    miss_slots = miss_base + miss_prefix - 1
    tl.store(output_miss_hash_buckets + miss_slots, miss_buckets, mask=misses)
    tl.store(
        output_miss_page_offsets + miss_slots,
        output_base + compact_starts,
        mask=misses,
    )

    invalid_descriptor = ~plan_valid | (has_selected & ~descriptor_valid)
    hash_failures = misses & (miss_buckets < 0)
    invalid_count = invalid_descriptor.to(tl.int32) + tl.sum(
        hash_failures.to(tl.int32), axis=0
    )
    tl.atomic_add(
        output_invalid_descriptor_count,
        invalid_count,
    )
    tl.store(output_page_counts + row, tl.sum(selected_page_counts, axis=0))
    tl.store(output_selected_counts + row, tl.sum(selected.to(tl.int32), axis=0))
    tl.store(output_hit_counts + row, tl.sum(stable_hits.to(tl.int32), axis=0))
    tl.store(output_miss_counts + row, tl.sum(misses.to(tl.int32), axis=0))


@triton.jit
def _map_compact_verification_miss_indices_kernel(
    miss_hash_buckets,
    miss_table_unique_indices,
    miss_count,
    unique_miss_count,
    output_miss_unique_indices,
    output_invalid_descriptor_count,
    miss_capacity,
    table_capacity,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_misses = tl.load(miss_count)
    num_unique_misses = tl.load(unique_miss_count)
    active = (offsets < miss_capacity) & (offsets < num_misses)

    buckets = tl.load(miss_hash_buckets + offsets, mask=active, other=-1)
    valid_bucket = active & (buckets >= 0) & (buckets < table_capacity)
    safe_buckets = tl.maximum(buckets, 0)
    unique_indices = tl.load(
        miss_table_unique_indices + safe_buckets, mask=valid_bucket, other=-1
    )
    valid_mapping = (
        valid_bucket & (unique_indices >= 0) & (unique_indices < num_unique_misses)
    )
    tl.store(
        output_miss_unique_indices + offsets,
        tl.where(valid_mapping, unique_indices, -1),
        mask=offsets < miss_capacity,
    )
    tl.atomic_add(
        output_invalid_descriptor_count,
        tl.sum((active & ~valid_mapping).to(tl.int32), axis=0),
    )


@triton.jit
def _scatter_compact_staging_page_ids_kernel(
    miss_unique_indices,
    miss_output_page_offsets,
    unique_staging_starts,
    unique_page_counts,
    output_page_ids,
    num_misses,
    num_unique_misses,
    MAX_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    miss_index = tl.program_id(0)
    valid_miss = miss_index < num_misses
    unique_index = tl.load(miss_unique_indices + miss_index, mask=valid_miss, other=-1)
    valid_unique = valid_miss & (unique_index >= 0) & (unique_index < num_unique_misses)
    safe_unique_index = tl.maximum(unique_index, 0)
    output_start = tl.load(
        miss_output_page_offsets + miss_index, mask=valid_unique, other=0
    )
    staging_start = tl.load(
        unique_staging_starts + safe_unique_index, mask=valid_unique, other=0
    )
    page_count = tl.load(
        unique_page_counts + safe_unique_index, mask=valid_unique, other=0
    )
    page_offsets = tl.arange(0, BLOCK_PAGES)
    valid_page = valid_unique & (page_offsets < page_count) & (page_offsets < MAX_PAGES)
    tl.store(
        output_page_ids + output_start + page_offsets,
        staging_start + page_offsets,
        mask=valid_page,
    )


@triton.jit
def _lookup_resident_handles_kernel(
    cluster_handles,
    logical_page_ids,
    plan_row_indices,
    active_mask,
    table_handles,
    table_versions,
    table_page_counts,
    table_page_slots,
    table_hit_gate_ready,
    table_last_access_epochs,
    access_epoch,
    output_page_slots,
    output_hit_mask,
    output_miss_mask,
    output_hit_gate_ready,
    output_access_kinds,
    num_output_clusters,
    source_clusters_per_row,
    handle_stride,
    logical_page_stride_0,
    logical_page_stride_1,
    output_page_stride_0,
    output_page_stride_1,
    table_page_stride_0,
    table_page_stride_1,
    CLUSTERS_PER_REQUEST: tl.constexpr,
    TABLE_CAPACITY: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_PLAN_ROWS: tl.constexpr,
):
    output_cluster_index = tl.program_id(0)
    valid_cluster = output_cluster_index < num_output_clusters

    request_index = output_cluster_index // CLUSTERS_PER_REQUEST
    cluster_in_row = output_cluster_index % CLUSTERS_PER_REQUEST
    source_cluster_index = output_cluster_index
    if USE_PLAN_ROWS:
        source_row = tl.load(plan_row_indices + request_index)
        source_cluster_index = source_row * source_clusters_per_row + cluster_in_row

    handle = tl.load(
        cluster_handles + source_cluster_index * handle_stride,
        mask=valid_cluster,
        other=-1,
    ).to(tl.int64)
    if USE_ACTIVE_MASK:
        active = tl.load(active_mask + request_index, mask=valid_cluster, other=0)
    else:
        active = True
    valid_cluster &= (handle >= 0) & active

    first_bucket = _resident_handle_hash(handle) & (TABLE_CAPACITY - 1)
    matched_bucket = -1
    searching = valid_cluster

    for probe in tl.static_range(64):
        bucket = (first_bucket + probe) & (TABLE_CAPACITY - 1)
        version_before = tl.atomic_add(
            table_versions + bucket,
            0,
            mask=searching,
            sem="acquire",
        )
        stored_handle = tl.load(
            table_handles + bucket,
            mask=searching,
            other=-1,
        )
        version_after = tl.atomic_add(
            table_versions + bucket,
            0,
            mask=searching,
            sem="acquire",
        )

        stable = (version_before == version_after) & ((version_before & 1) == 0)
        matched = searching & stable & (stored_handle == handle)
        matched_bucket = tl.where(matched, bucket, matched_bucket)
        empty = stable & (stored_handle == -1)
        searching &= ~matched & ~empty

    found = matched_bucket >= 0
    safe_bucket = tl.maximum(matched_bucket, 0)
    version_before = tl.atomic_add(
        table_versions + safe_bucket,
        0,
        mask=found,
        sem="acquire",
    )
    stored_handle = tl.load(
        table_handles + safe_bucket,
        mask=found,
        other=-1,
    )
    page_count = tl.load(
        table_page_counts + safe_bucket,
        mask=found,
        other=0,
    )
    gate_ready = tl.load(
        table_hit_gate_ready + safe_bucket,
        mask=found,
        other=0,
    )

    page_offsets = tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < MAX_PAGES
    logical_pages = tl.load(
        logical_page_ids
        + source_cluster_index * logical_page_stride_0
        + page_offsets * logical_page_stride_1,
        mask=valid_cluster & page_mask,
        other=-1,
    )
    resident_slots = tl.load(
        table_page_slots
        + safe_bucket * table_page_stride_0
        + page_offsets * table_page_stride_1,
        mask=(found & page_mask & (page_offsets < page_count) & (logical_pages >= 0)),
        other=-1,
    )

    version_after = tl.atomic_add(
        table_versions + safe_bucket,
        0,
        mask=found,
        sem="acquire",
    )
    stable_hit = (
        found
        & (stored_handle == handle)
        & (version_before == version_after)
        & ((version_before & 1) == 0)
        & (page_count > 0)
    )
    miss = valid_cluster & ~stable_hit

    tl.atomic_max(
        table_last_access_epochs + safe_bucket,
        access_epoch,
        mask=stable_hit,
        sem="relaxed",
    )

    tl.store(
        output_page_slots
        + output_cluster_index * output_page_stride_0
        + page_offsets * output_page_stride_1,
        tl.where(stable_hit, resident_slots, -1),
        mask=page_mask,
    )

    tl.store(output_hit_mask + output_cluster_index, stable_hit)
    tl.store(output_miss_mask + output_cluster_index, miss)
    tl.store(output_hit_gate_ready + output_cluster_index, stable_hit & gate_ready)
    access_kind = tl.where(stable_hit, 1, tl.where(miss, 2, 0))
    tl.store(output_access_kinds + output_cluster_index, access_kind)


@triton.jit
def _compact_resident_misses_kernel(
    cluster_handles,
    miss_mask,
    plan_row_indices,
    output_handles,
    output_positions,
    output_count,
    num_output_handles,
    source_handles_per_row,
    output_handles_per_row,
    USE_PLAN_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_output_handles

    source_offsets = offsets
    if USE_PLAN_ROWS:
        query_indices = offsets // output_handles_per_row
        offsets_in_row = offsets % output_handles_per_row
        source_rows = tl.load(plan_row_indices + query_indices, mask=valid, other=0)
        source_offsets = source_rows * source_handles_per_row + offsets_in_row

    handles = tl.load(cluster_handles + source_offsets, mask=valid, other=-1)
    misses = tl.load(miss_mask + offsets, mask=valid, other=0)
    selected = valid & misses & (handles >= 0)

    selected_i32 = selected.to(tl.int32)
    local_offsets = tl.cumsum(selected_i32, axis=0) - 1
    block_count = tl.sum(selected_i32, axis=0)
    output_start = tl.atomic_add(output_count, block_count)

    destinations = output_start + local_offsets
    tl.store(output_handles + destinations, handles, mask=selected)
    tl.store(output_positions + destinations, offsets, mask=selected)


@triton.jit
def _scatter_staging_page_ids_kernel(
    miss_positions,
    staging_starts,
    page_counts,
    output_page_ids,
    num_misses,
    output_page_stride,
    MAX_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    miss_index = tl.program_id(0)
    valid_miss = miss_index < num_misses

    position = tl.load(miss_positions + miss_index, mask=valid_miss, other=0)
    staging_start = tl.load(staging_starts + miss_index, mask=valid_miss, other=0)
    page_count = tl.load(page_counts + miss_index, mask=valid_miss, other=0)

    page_offsets = tl.arange(0, BLOCK_PAGES)
    valid_page = valid_miss & (page_offsets < page_count)
    output_offsets = position * output_page_stride + page_offsets
    tl.store(
        output_page_ids + output_offsets,
        staging_start + page_offsets,
        mask=valid_page & (page_offsets < MAX_PAGES),
    )


@triton.jit
def _update_resident_handles_kernel(
    bucket_ids,
    cluster_handles,
    page_counts,
    page_slots,
    hit_gate_ready,
    table_handles,
    table_versions,
    table_page_counts,
    table_page_slots,
    table_hit_gate_ready,
    num_updates,
    input_page_stride_0,
    input_page_stride_1,
    table_page_stride_0,
    table_page_stride_1,
    MAX_PAGES: tl.constexpr,
):
    update_index = tl.program_id(0)
    valid = update_index < num_updates

    bucket = tl.load(bucket_ids + update_index, mask=valid, other=0)
    handle = tl.load(cluster_handles + update_index, mask=valid, other=-2)
    page_count = tl.load(page_counts + update_index, mask=valid, other=0)
    gate_ready = tl.load(hit_gate_ready + update_index, mask=valid, other=0)

    version_ptr = table_versions + bucket
    tl.atomic_add(version_ptr, 1, mask=valid, sem="acq_rel")

    for page_index in tl.static_range(MAX_PAGES):
        slot = tl.load(
            page_slots
            + update_index * input_page_stride_0
            + page_index * input_page_stride_1,
            mask=valid & (page_index < page_count),
            other=-1,
        )
        tl.store(
            table_page_slots
            + bucket * table_page_stride_0
            + page_index * table_page_stride_1,
            slot,
            mask=valid,
        )

    tl.store(table_page_counts + bucket, page_count, mask=valid)
    tl.store(table_hit_gate_ready + bucket, gate_ready, mask=valid)
    tl.store(table_handles + bucket, handle, mask=valid)
    tl.atomic_add(version_ptr, 1, mask=valid, sem="release")


@triton.jit
def _publish_resident_table_bindings_kernel(
    binding_commands,
    arena_cluster_ids,
    arena_resident_table_buckets,
    arena_cluster_offsets,
    arena_num_clusters,
    arena_generations,
    num_bindings,
    command_stride_0,
    command_stride_1,
    cluster_stride_0,
    cluster_stride_1,
):
    binding_index = tl.program_id(0)
    valid = binding_index < num_bindings
    command_offset = binding_index * command_stride_0
    request_slot = tl.load(binding_commands + command_offset, mask=valid, other=0)
    expected_generation = tl.load(
        binding_commands + command_offset + command_stride_1,
        mask=valid,
        other=-1,
    )
    kv_head_index = tl.load(
        binding_commands + command_offset + 2 * command_stride_1,
        mask=valid,
        other=0,
    )
    local_cluster_index = tl.load(
        binding_commands + command_offset + 3 * command_stride_1,
        mask=valid,
        other=0,
    )
    expected_handle = tl.load(
        binding_commands + command_offset + 4 * command_stride_1,
        mask=valid,
        other=-1,
    )
    table_bucket = tl.load(
        binding_commands + command_offset + 5 * command_stride_1,
        mask=valid,
        other=-1,
    )

    current_generation = tl.load(arena_generations + request_slot, mask=valid, other=-1)
    num_clusters = tl.load(arena_num_clusters + request_slot, mask=valid, other=0)
    cluster_offset = tl.load(arena_cluster_offsets + request_slot, mask=valid, other=0)
    valid &= current_generation == expected_generation
    valid &= local_cluster_index >= 0
    valid &= local_cluster_index < num_clusters

    storage_index = cluster_offset + local_cluster_index
    flat_offset = kv_head_index * cluster_stride_0 + storage_index * cluster_stride_1
    current_handle = tl.load(arena_cluster_ids + flat_offset, mask=valid, other=-1)
    valid &= current_handle == expected_handle

    tl.store(
        arena_resident_table_buckets + flat_offset,
        table_bucket,
        mask=valid,
    )
