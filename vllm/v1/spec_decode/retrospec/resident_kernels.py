# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _resolve_compact_draft_pages_kernel(
    cluster_handles,
    logical_page_ids,
    logical_page_token_counts,
    retrieval_scores,
    active_mask,
    has_clusters,
    table_handles,
    table_versions,
    table_page_counts,
    table_page_slots,
    table_hit_gate_ready,
    table_last_access_epochs,
    access_epoch,
    fallback_token_counts,
    output_page_slots,
    output_page_token_counts,
    output_page_counts,
    output_clustered_token_counts,
    output_hit_attention,
    output_selected_counts,
    output_hit_counts,
    output_miss_counts,
    output_gate_ready,
    output_miss_handles,
    output_miss_positions,
    output_miss_count,
    retrieval_score_row_stride,
    fallback_count_row_stride,
    table_page_stride,
    TABLE_CAPACITY: tl.constexpr,
    NUM_CLUSTERS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_OUTPUT_PAGES: tl.constexpr,
    EMIT_MISSES: tl.constexpr,
):
    row = tl.program_id(0)
    active = tl.load(active_mask + row)
    row_has_clusters = tl.load(has_clusters + row)

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

    compact_page_count = 0
    clustered_token_count = 0
    selected_count = 0
    hit_count = 0
    miss_count = 0
    hit_attention = 0.0
    gate_ready = False

    for rank in tl.range(0, NUM_CLUSTERS):
        cluster_offset = row * NUM_CLUSTERS + rank
        handle = tl.load(cluster_handles + cluster_offset).to(tl.int64)
        selected = active & (handle >= 0)
        selected_count += selected.to(tl.int32)

        first_bucket = handle & (TABLE_CAPACITY - 1)
        matched_bucket = -1
        searching = selected
        for probe in tl.static_range(64):
            bucket = (first_bucket + probe) & (TABLE_CAPACITY - 1)
            version_before = tl.atomic_add(
                table_versions + bucket, 0, mask=searching, sem="acquire"
            )
            stored_handle = tl.load(table_handles + bucket, mask=searching, other=-1)
            version_after = tl.atomic_add(
                table_versions + bucket, 0, mask=searching, sem="acquire"
            )
            stable = (version_before == version_after) & ((version_before & 1) == 0)
            matched = searching & stable & (stored_handle == handle)
            matched_bucket = tl.where(matched, bucket, matched_bucket)
            empty = stable & (stored_handle == -1)
            searching &= ~matched & ~empty

        found = matched_bucket >= 0
        safe_bucket = tl.maximum(matched_bucket, 0)
        version_before = tl.atomic_add(
            table_versions + safe_bucket, 0, mask=found, sem="acquire"
        )
        stored_handle = tl.load(table_handles + safe_bucket, mask=found, other=-1)
        resident_page_count = tl.load(
            table_page_counts + safe_bucket, mask=found, other=0
        )
        cluster_gate_ready = tl.load(
            table_hit_gate_ready + safe_bucket, mask=found, other=0
        )

        page_offsets = tl.arange(0, BLOCK_PAGES)
        valid_page_offset = page_offsets < MAX_PAGES
        logical_counts = tl.load(
            logical_page_token_counts + cluster_offset * MAX_PAGES + page_offsets,
            mask=valid_page_offset,
            other=0,
        )
        logical_ids = tl.load(
            logical_page_ids + cluster_offset * MAX_PAGES + page_offsets,
            mask=valid_page_offset,
            other=-1,
        )
        valid_logical_pages = (
            valid_page_offset & (logical_counts > 0) & (logical_ids >= 0)
        )
        logical_page_count = tl.sum(valid_logical_pages.to(tl.int32), axis=0)
        resident_slots = tl.load(
            table_page_slots + safe_bucket * table_page_stride + page_offsets,
            mask=found
            & valid_page_offset
            & (page_offsets < resident_page_count)
            & (page_offsets < logical_page_count),
            other=-1,
        )
        version_after = tl.atomic_add(
            table_versions + safe_bucket, 0, mask=found, sem="acquire"
        )
        resolved_page_count = tl.sum(
            (valid_logical_pages & (resident_slots >= 0)).to(tl.int32), axis=0
        )
        stable_hit = (
            selected
            & found
            & (stored_handle == handle)
            & (version_before == version_after)
            & ((version_before & 1) == 0)
            & (logical_page_count > 0)
            & (resident_page_count >= logical_page_count)
            & (resolved_page_count == logical_page_count)
        )
        miss = selected & ~stable_hit

        tl.atomic_max(
            table_last_access_epochs + safe_bucket,
            access_epoch,
            mask=stable_hit,
            sem="relaxed",
        )

        output_offsets = compact_page_count + page_offsets
        valid_output_page = (
            stable_hit & valid_logical_pages & (output_offsets < PAGE_CAPACITY)
        )
        tl.store(
            output_page_slots + row * PAGE_CAPACITY + output_offsets,
            resident_slots,
            mask=valid_output_page,
        )
        tl.store(
            output_page_token_counts + row * PAGE_CAPACITY + output_offsets,
            logical_counts,
            mask=valid_output_page,
        )

        fallback_offset = row * fallback_count_row_stride + rank
        fallback_count = tl.load(fallback_token_counts + fallback_offset)
        tl.store(
            fallback_token_counts + fallback_offset,
            tl.where(miss, fallback_count, 0),
        )

        if EMIT_MISSES:
            miss_slot = tl.atomic_add(output_miss_count, 1, mask=miss)
            tl.store(output_miss_handles + miss_slot, handle, mask=miss)
            tl.store(
                output_miss_positions + miss_slot,
                cluster_offset,
                mask=miss,
            )

        score = tl.load(retrieval_scores + row * retrieval_score_row_stride + rank)
        hit_attention += tl.where(stable_hit, score, 0.0)
        compact_page_count += tl.where(stable_hit, logical_page_count, 0)
        clustered_token_count += tl.where(
            stable_hit,
            tl.sum(tl.where(valid_logical_pages, logical_counts, 0), axis=0),
            0,
        )
        hit_count += stable_hit.to(tl.int32)
        miss_count += miss.to(tl.int32)
        gate_ready |= stable_hit & cluster_gate_ready

    tl.store(output_page_counts + row, compact_page_count)
    tl.store(output_clustered_token_counts + row, clustered_token_count)
    tl.store(output_hit_attention + row, hit_attention)
    tl.store(output_selected_counts + row, selected_count)
    tl.store(output_hit_counts + row, hit_count)
    tl.store(output_miss_counts + row, miss_count)
    tl.store(output_gate_ready + row, gate_ready & row_has_clusters)


@triton.jit
def _finalize_compact_draft_attention_kernel(
    hit_attention_by_head,
    has_clusters,
    gate_ready,
    active_mask,
    output_attention,
    NUM_HEADS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    request_index = tl.program_id(0)
    head_offsets = tl.arange(0, BLOCK_HEADS)
    head_mask = head_offsets < NUM_HEADS
    row_offsets = request_index * NUM_HEADS + head_offsets

    head_attention = tl.load(
        hit_attention_by_head + row_offsets, mask=head_mask, other=0.0
    )
    head_has_clusters = tl.load(has_clusters + row_offsets, mask=head_mask, other=0)
    head_gate_ready = tl.load(gate_ready + row_offsets, mask=head_mask, other=0)
    head_attention = tl.where(head_has_clusters & head_gate_ready, head_attention, 1.0)
    request_attention = (
        tl.sum(tl.where(head_mask, head_attention, 0.0), axis=0) / NUM_HEADS
    )
    active = tl.load(active_mask + request_index)
    tl.store(output_attention + request_index, tl.where(active, request_attention, 1.0))


@triton.jit
def _resolve_compact_verification_pages_vector_kernel(
    selected_cluster_indices,
    plan_row_indices,
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
    output_miss_handles,
    output_miss_logical_page_ids,
    output_miss_page_counts,
    output_miss_page_offsets,
    output_miss_count,
    output_invalid_descriptor_count,
    table_page_stride,
    output_miss_logical_page_stride,
    selected_stride_0,
    selected_stride_1,
    selected_stride_2,
    PLAN_BATCH_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_CLUSTERS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    ARENA_CLUSTER_CAPACITY: tl.constexpr,
    ARENA_PAGE_CAPACITY: tl.constexpr,
    TABLE_CAPACITY: tl.constexpr,
    BLOCK_CLUSTERS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_OUTPUT_PAGES: tl.constexpr,
):
    row = tl.program_id(0)
    query_idx = row // NUM_KV_HEADS
    kv_head_idx = row % NUM_KV_HEADS
    plan_row = tl.load(plan_row_indices + query_idx).to(tl.int64)
    request_idx = plan_row % PLAN_BATCH_CAPACITY
    request_slot = tl.load(request_slot_ids + request_idx).to(tl.int64)
    expected_generation = tl.load(request_slot_generations + request_idx).to(tl.int64)
    valid_slot = request_slot >= 0
    safe_slot = tl.maximum(request_slot, 0)
    actual_generation = tl.load(
        arena_generations + safe_slot, mask=valid_slot, other=-1
    ).to(tl.int64)
    descriptor_valid = valid_slot & (actual_generation == expected_generation)

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
        plan_row * selected_stride_0
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

    first_buckets = handles & (TABLE_CAPACITY - 1)
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

    miss_prefix = tl.cumsum(misses.to(tl.int32), axis=0)
    num_misses = tl.sum(misses.to(tl.int32), axis=0)
    miss_base = tl.atomic_add(output_miss_count, num_misses)
    miss_slots = miss_base + miss_prefix - 1
    tl.store(output_miss_handles + miss_slots, handles, mask=misses)
    tl.store(output_miss_page_counts + miss_slots, logical_page_counts, mask=misses)
    tl.store(
        output_miss_page_offsets + miss_slots,
        output_base + compact_starts,
        mask=misses,
    )
    tl.store(
        output_miss_logical_page_ids
        + miss_slots[:, None] * output_miss_logical_page_stride
        + page_offsets[None, :],
        logical_page_ids,
        mask=misses[:, None] & valid_page_offsets[None, :],
    )

    invalid_descriptor = has_selected & ~descriptor_valid
    tl.atomic_add(
        output_invalid_descriptor_count,
        invalid_descriptor.to(tl.int32),
    )
    tl.store(output_page_counts + row, tl.sum(selected_page_counts, axis=0))
    tl.store(output_selected_counts + row, tl.sum(selected.to(tl.int32), axis=0))
    tl.store(output_hit_counts + row, tl.sum(stable_hits.to(tl.int32), axis=0))
    tl.store(output_miss_counts + row, tl.sum(misses.to(tl.int32), axis=0))


@triton.jit
def _scatter_compact_staging_page_ids_kernel(
    miss_output_page_offsets,
    staging_starts,
    page_counts,
    output_page_ids,
    num_misses,
    MAX_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    miss_index = tl.program_id(0)
    valid_miss = miss_index < num_misses
    output_start = tl.load(
        miss_output_page_offsets + miss_index, mask=valid_miss, other=0
    )
    staging_start = tl.load(staging_starts + miss_index, mask=valid_miss, other=0)
    page_count = tl.load(page_counts + miss_index, mask=valid_miss, other=0)
    page_offsets = tl.arange(0, BLOCK_PAGES)
    valid_page = valid_miss & (page_offsets < page_count) & (page_offsets < MAX_PAGES)
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

    first_bucket = handle & (TABLE_CAPACITY - 1)
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


def lookup_resident_handles(
    cluster_handles: torch.Tensor,
    logical_page_ids: torch.Tensor,
    active_mask: torch.Tensor | None,
    table_handles: torch.Tensor,
    table_versions: torch.Tensor,
    table_page_counts: torch.Tensor,
    table_page_slots: torch.Tensor,
    table_hit_gate_ready: torch.Tensor,
    table_last_access_epochs: torch.Tensor,
    access_epoch: int,
    output_page_slots: torch.Tensor,
    output_hit_mask: torch.Tensor,
    output_miss_mask: torch.Tensor,
    output_hit_gate_ready: torch.Tensor,
    output_access_kinds: torch.Tensor,
    plan_row_indices: torch.Tensor | None = None,
) -> None:
    if cluster_handles.device.type != "cuda":
        raise ValueError("Resident handle lookup requires CUDA tensors")
    if cluster_handles.ndim != 3:
        raise ValueError("Cluster handles must have shape [batch, heads, clusters]")
    if logical_page_ids.shape[:-1] != cluster_handles.shape:
        raise ValueError("Logical pages do not match cluster handles")
    if plan_row_indices is not None:
        if plan_row_indices.ndim != 1:
            raise ValueError("Plan row indices must be one-dimensional")
        if plan_row_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Plan row indices must be integral")
        if plan_row_indices.device != cluster_handles.device:
            raise ValueError("Plan row indices must use the lookup device")

    output_batch = (
        cluster_handles.shape[0]
        if plan_row_indices is None
        else plan_row_indices.shape[0]
    )
    output_cluster_shape = (output_batch, *cluster_handles.shape[1:])
    output_page_shape = (*output_cluster_shape, logical_page_ids.shape[-1])
    if output_page_slots.shape != output_page_shape:
        raise ValueError("Resident page output has the wrong indexed shape")
    for output in (
        output_hit_mask,
        output_miss_mask,
        output_hit_gate_ready,
        output_access_kinds,
    ):
        if output.shape != output_cluster_shape:
            raise ValueError("Resident lookup output has the wrong indexed shape")
    if active_mask is not None and active_mask.shape != (output_batch,):
        raise ValueError("active_mask does not match the batch size")

    if table_handles.numel() == 0:
        output_page_slots.fill_(-1)
        output_hit_mask.zero_()
        source_handles = cluster_handles
        if plan_row_indices is not None:
            source_handles = cluster_handles.index_select(0, plan_row_indices)
        valid = source_handles >= 0
        if active_mask is not None:
            valid &= active_mask[:, None, None]
        output_miss_mask.copy_(valid)
        output_hit_gate_ready.zero_()
        output_access_kinds.copy_(valid.to(torch.uint8) * 2)
        return

    table_capacity = table_handles.numel()
    if table_capacity & (table_capacity - 1):
        raise ValueError("Resident handle-table capacity must be a power of two")
    if table_last_access_epochs.shape != table_handles.shape:
        raise ValueError("Resident access epochs must match the handle table")
    if table_last_access_epochs.dtype != torch.int64:
        raise ValueError("Resident access epochs must use int64")
    if table_last_access_epochs.device != table_handles.device:
        raise ValueError("Resident access epochs must use the handle-table device")
    if access_epoch <= 0:
        raise ValueError("Resident access epoch must be positive")

    flat_handles = cluster_handles.reshape(-1)
    flat_pages = logical_page_ids.reshape(
        flat_handles.numel(), logical_page_ids.shape[-1]
    )
    flat_output_pages = output_page_slots.reshape(-1, logical_page_ids.shape[-1])
    clusters_per_request = cluster_handles.shape[1] * cluster_handles.shape[2]
    mask_source = cluster_handles if active_mask is None else active_mask
    plan_row_source = cluster_handles if plan_row_indices is None else plan_row_indices
    num_output_clusters = output_hit_mask.numel()

    _lookup_resident_handles_kernel[(num_output_clusters,)](
        flat_handles,
        flat_pages,
        plan_row_source,
        mask_source,
        table_handles,
        table_versions,
        table_page_counts,
        table_page_slots,
        table_hit_gate_ready,
        table_last_access_epochs,
        access_epoch,
        flat_output_pages,
        output_hit_mask.reshape(-1),
        output_miss_mask.reshape(-1),
        output_hit_gate_ready.reshape(-1),
        output_access_kinds.reshape(-1),
        num_output_clusters,
        clusters_per_request,
        flat_handles.stride(0),
        flat_pages.stride(0),
        flat_pages.stride(1),
        flat_output_pages.stride(0),
        flat_output_pages.stride(1),
        table_page_slots.stride(0),
        table_page_slots.stride(1),
        CLUSTERS_PER_REQUEST=clusters_per_request,
        TABLE_CAPACITY=table_capacity,
        MAX_PAGES=logical_page_ids.shape[-1],
        BLOCK_PAGES=triton.next_power_of_2(logical_page_ids.shape[-1]),
        USE_ACTIVE_MASK=active_mask is not None,
        USE_PLAN_ROWS=plan_row_indices is not None,
    )


def compact_resident_misses(
    cluster_handles: torch.Tensor,
    miss_mask: torch.Tensor,
    output_handles: torch.Tensor,
    output_positions: torch.Tensor,
    output_count: torch.Tensor,
    plan_row_indices: torch.Tensor | None = None,
) -> None:
    if cluster_handles.device.type != "cuda":
        raise ValueError("Resident miss compaction requires CUDA tensors")
    if cluster_handles.ndim != 3 or miss_mask.ndim != 3:
        raise ValueError("Cluster handles and miss mask must be three-dimensional")
    if plan_row_indices is None:
        if cluster_handles.shape != miss_mask.shape:
            raise ValueError("Cluster handles and miss mask must have equal shapes")
    else:
        if plan_row_indices.shape != (miss_mask.shape[0],):
            raise ValueError("Plan rows must contain one entry per miss-mask row")
        if plan_row_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Plan row indices must be integral")
        if plan_row_indices.device != cluster_handles.device:
            raise ValueError("Plan row indices must use the compaction device")
        if cluster_handles.shape[1:] != miss_mask.shape[1:]:
            raise ValueError("Indexed cluster and miss-mask rows must match")
    if output_handles.numel() < miss_mask.numel():
        raise ValueError("Compact handle output does not have enough capacity")
    if output_positions.numel() < miss_mask.numel():
        raise ValueError("Compact position output does not have enough capacity")
    if output_count.shape != (1,):
        raise ValueError("Compact miss count must contain one element")

    output_count.zero_()
    if miss_mask.numel() == 0:
        return

    block_size = 256
    source_handles_per_row = cluster_handles.shape[1] * cluster_handles.shape[2]
    output_handles_per_row = miss_mask.shape[1] * miss_mask.shape[2]
    plan_row_source = cluster_handles if plan_row_indices is None else plan_row_indices
    _compact_resident_misses_kernel[(triton.cdiv(miss_mask.numel(), block_size),)](
        cluster_handles.reshape(-1),
        miss_mask.reshape(-1),
        plan_row_source,
        output_handles,
        output_positions,
        output_count,
        miss_mask.numel(),
        source_handles_per_row,
        output_handles_per_row,
        USE_PLAN_ROWS=plan_row_indices is not None,
        BLOCK_SIZE=block_size,
    )


def resolve_compact_draft_pages(
    cluster_handles: torch.Tensor,
    logical_page_ids: torch.Tensor,
    logical_page_token_counts: torch.Tensor,
    retrieval_scores: torch.Tensor,
    active_mask: torch.Tensor,
    has_clusters: torch.Tensor,
    table_handles: torch.Tensor,
    table_versions: torch.Tensor,
    table_page_counts: torch.Tensor,
    table_page_slots: torch.Tensor,
    table_hit_gate_ready: torch.Tensor,
    table_last_access_epochs: torch.Tensor,
    access_epoch: int,
    fallback_token_counts: torch.Tensor,
    output_page_slots: torch.Tensor,
    output_page_token_counts: torch.Tensor,
    output_page_counts: torch.Tensor,
    output_clustered_token_counts: torch.Tensor,
    output_attention: torch.Tensor,
    output_hit_attention_by_head: torch.Tensor,
    output_selected_counts: torch.Tensor,
    output_hit_counts: torch.Tensor,
    output_miss_counts: torch.Tensor,
    output_gate_ready: torch.Tensor,
    output_miss_handles: torch.Tensor,
    output_miss_positions: torch.Tensor,
    output_miss_count: torch.Tensor,
    emit_misses: bool = True,
) -> None:
    """Resolve and compact one DRAFT selection without CPU synchronization."""
    if cluster_handles.device.type != "cuda":
        raise ValueError("Compact draft resolution requires CUDA tensors")
    if cluster_handles.ndim != 3:
        raise ValueError("Cluster handles must have shape [batch, heads, clusters]")
    if logical_page_ids.shape != logical_page_token_counts.shape:
        raise ValueError("Logical page IDs and token counts must match")
    if logical_page_ids.shape[:-1] != cluster_handles.shape:
        raise ValueError("Logical pages do not match cluster handles")
    if retrieval_scores.shape != cluster_handles.shape:
        raise ValueError("Retrieval scores do not match cluster handles")
    if fallback_token_counts.shape != cluster_handles.shape:
        raise ValueError("Fallback counts do not match cluster handles")

    batch_size, num_heads, num_clusters = cluster_handles.shape
    row_shape = (batch_size, num_heads)
    page_capacity = num_clusters * logical_page_ids.shape[-1]
    if active_mask.shape != (batch_size,):
        raise ValueError("active_mask does not match the draft batch")
    if has_clusters.shape != row_shape:
        raise ValueError("has_clusters does not match the draft rows")
    if output_page_slots.shape != (*row_shape, page_capacity):
        raise ValueError("Compact page-slot output has the wrong shape")
    if output_page_token_counts.shape != output_page_slots.shape:
        raise ValueError("Compact page token counts have the wrong shape")
    for output in (
        output_page_counts,
        output_clustered_token_counts,
        output_hit_attention_by_head,
        output_selected_counts,
        output_hit_counts,
        output_miss_counts,
        output_gate_ready,
    ):
        if output.shape != row_shape:
            raise ValueError("Compact draft row output has the wrong shape")
    if output_attention.shape != (batch_size,):
        raise ValueError("Compact draft attention output has the wrong shape")
    if output_miss_count.shape != (1,):
        raise ValueError("Compact miss count must contain one element")
    if output_miss_handles.numel() < cluster_handles.numel():
        raise ValueError("Compact miss-handle output does not have enough capacity")
    if output_miss_positions.numel() < cluster_handles.numel():
        raise ValueError("Compact miss-position output does not have enough capacity")
    if table_handles.numel() == 0 or table_handles.numel() & (
        table_handles.numel() - 1
    ):
        raise ValueError(
            "Resident handle-table capacity must be a nonzero power of two"
        )
    if table_page_slots.shape[1] < logical_page_ids.shape[-1]:
        raise ValueError("Resident handle table has too few page slots")
    if access_epoch <= 0:
        raise ValueError("Resident access epoch must be positive")

    output_miss_count.zero_()
    if num_clusters == 0:
        output_page_slots.fill_(-1)
        output_page_token_counts.zero_()
        output_page_counts.zero_()
        output_clustered_token_counts.zero_()
        output_hit_attention_by_head.zero_()
        output_selected_counts.zero_()
        output_hit_counts.zero_()
        output_miss_counts.zero_()
        output_gate_ready.zero_()
        output_attention.fill_(1.0)
        return

    num_rows = batch_size * num_heads
    _resolve_compact_draft_pages_kernel[(num_rows,)](
        cluster_handles,
        logical_page_ids,
        logical_page_token_counts,
        retrieval_scores,
        active_mask,
        has_clusters,
        table_handles,
        table_versions,
        table_page_counts,
        table_page_slots,
        table_hit_gate_ready,
        table_last_access_epochs,
        access_epoch,
        fallback_token_counts,
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
        retrieval_scores.stride(-2),
        fallback_token_counts.stride(-2),
        table_page_slots.stride(0),
        TABLE_CAPACITY=table_handles.numel(),
        NUM_CLUSTERS=num_clusters,
        MAX_PAGES=logical_page_ids.shape[-1],
        PAGE_CAPACITY=page_capacity,
        BLOCK_PAGES=triton.next_power_of_2(logical_page_ids.shape[-1]),
        BLOCK_OUTPUT_PAGES=triton.next_power_of_2(page_capacity),
        EMIT_MISSES=emit_misses,
    )
    _finalize_compact_draft_attention_kernel[(batch_size,)](
        output_hit_attention_by_head,
        has_clusters,
        output_gate_ready,
        active_mask,
        output_attention,
        NUM_HEADS=num_heads,
        BLOCK_HEADS=triton.next_power_of_2(num_heads),
    )


def resolve_compact_verification_pages(
    selected_cluster_indices: torch.Tensor,
    plan_row_indices: torch.Tensor,
    request_slot_ids: torch.Tensor,
    request_slot_generations: torch.Tensor,
    arena_cluster_ids: torch.Tensor,
    arena_cluster_page_starts: torch.Tensor,
    arena_cluster_page_counts: torch.Tensor,
    arena_page_ids: torch.Tensor,
    arena_page_token_counts: torch.Tensor,
    arena_cluster_offsets: torch.Tensor,
    arena_page_offsets: torch.Tensor,
    arena_generations: torch.Tensor,
    table_handles: torch.Tensor,
    table_versions: torch.Tensor,
    table_page_counts: torch.Tensor,
    table_page_slots: torch.Tensor,
    table_last_access_epochs: torch.Tensor,
    access_epoch: int,
    output_resident_page_ids: torch.Tensor,
    output_staging_page_ids: torch.Tensor,
    output_page_token_counts: torch.Tensor,
    output_page_counts: torch.Tensor,
    output_selected_counts: torch.Tensor,
    output_hit_counts: torch.Tensor,
    output_miss_counts: torch.Tensor,
    output_miss_handles: torch.Tensor,
    output_miss_logical_page_ids: torch.Tensor,
    output_miss_page_counts: torch.Tensor,
    output_miss_page_offsets: torch.Tensor,
    output_miss_count: torch.Tensor,
    output_invalid_descriptor_count: torch.Tensor,
) -> None:
    if selected_cluster_indices.device.type != "cuda":
        raise ValueError("Compact verification resolution requires CUDA")
    if selected_cluster_indices.ndim != 3:
        raise ValueError("Selected clusters must have shape [rows, heads, clusters]")
    if plan_row_indices.ndim != 1:
        raise ValueError("Plan rows must be one-dimensional")
    if request_slot_ids.shape != request_slot_generations.shape:
        raise ValueError("Request slot descriptors must have equal shapes")
    if request_slot_ids.ndim != 1:
        raise ValueError("Request slot descriptors must be one-dimensional")

    num_queries = plan_row_indices.shape[0]
    _, num_kv_heads, num_clusters = selected_cluster_indices.shape
    max_pages = output_miss_logical_page_ids.shape[1]
    page_capacity = num_clusters * max_pages
    expected_page_shape = (num_queries, num_kv_heads, page_capacity)
    if output_resident_page_ids.shape != expected_page_shape:
        raise ValueError("Compact resident-page output has the wrong shape")
    if output_staging_page_ids.shape != expected_page_shape:
        raise ValueError("Compact staging-page output has the wrong shape")
    if output_page_token_counts.shape != expected_page_shape:
        raise ValueError("Compact page-count metadata has the wrong shape")
    row_shape = (num_queries, num_kv_heads)
    for output in (
        output_page_counts,
        output_selected_counts,
        output_hit_counts,
        output_miss_counts,
    ):
        if output.shape != row_shape:
            raise ValueError("Compact verification row output has the wrong shape")
    miss_capacity = num_queries * num_kv_heads * num_clusters
    if output_miss_handles.numel() < miss_capacity:
        raise ValueError("Verification miss output is too small")
    if output_miss_logical_page_ids.shape[0] < miss_capacity:
        raise ValueError("Verification miss-page output is too small")
    if any(
        output.numel() < miss_capacity
        for output in (output_miss_page_counts, output_miss_page_offsets)
    ):
        raise ValueError("Verification miss metadata output is too small")
    if output_miss_count.shape != (1,):
        raise ValueError("Verification miss count must contain one element")
    if output_invalid_descriptor_count.shape != (1,):
        raise ValueError("Invalid descriptor count must contain one element")

    tensors = (
        selected_cluster_indices,
        plan_row_indices,
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
        output_resident_page_ids,
        output_staging_page_ids,
        output_page_token_counts,
        output_page_counts,
        output_selected_counts,
        output_hit_counts,
        output_miss_counts,
        output_miss_handles,
        output_miss_logical_page_ids,
        output_miss_page_counts,
        output_miss_page_offsets,
        output_miss_count,
        output_invalid_descriptor_count,
    )
    if any(tensor.device != selected_cluster_indices.device for tensor in tensors):
        raise ValueError("Compact verification tensors must use one CUDA device")

    output_miss_count.zero_()
    output_invalid_descriptor_count.zero_()
    if num_queries == 0:
        return
    if num_clusters == 0 or max_pages == 0:
        output_resident_page_ids.fill_(-1)
        output_staging_page_ids.fill_(-1)
        output_page_token_counts.zero_()
        output_page_counts.zero_()
        output_selected_counts.zero_()
        output_hit_counts.zero_()
        output_miss_counts.zero_()
        return

    _resolve_compact_verification_pages_vector_kernel[(num_queries * num_kv_heads,)](
        selected_cluster_indices,
        plan_row_indices,
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
        output_miss_handles,
        output_miss_logical_page_ids,
        output_miss_page_counts,
        output_miss_page_offsets,
        output_miss_count,
        output_invalid_descriptor_count,
        table_page_slots.stride(0),
        output_miss_logical_page_ids.stride(0),
        selected_cluster_indices.stride(0),
        selected_cluster_indices.stride(1),
        selected_cluster_indices.stride(2),
        PLAN_BATCH_CAPACITY=request_slot_ids.shape[0],
        NUM_KV_HEADS=num_kv_heads,
        NUM_CLUSTERS=num_clusters,
        MAX_PAGES=max_pages,
        PAGE_CAPACITY=page_capacity,
        ARENA_CLUSTER_CAPACITY=arena_cluster_ids.shape[1],
        ARENA_PAGE_CAPACITY=arena_page_ids.shape[1],
        TABLE_CAPACITY=table_handles.numel(),
        BLOCK_CLUSTERS=triton.next_power_of_2(num_clusters),
        BLOCK_PAGES=triton.next_power_of_2(max_pages),
        BLOCK_OUTPUT_PAGES=triton.next_power_of_2(page_capacity),
    )


def scatter_compact_staging_page_ids(
    miss_output_page_offsets: torch.Tensor,
    staging_starts: torch.Tensor,
    page_counts: torch.Tensor,
    num_misses: int,
    max_pages: int,
    output_page_ids: torch.Tensor,
) -> None:
    if output_page_ids.device.type != "cuda":
        raise ValueError("Compact staging-page scatter requires CUDA")
    if num_misses < 0:
        raise ValueError("num_misses must be non-negative")
    if max_pages <= 0 and num_misses:
        raise ValueError("max_pages must be positive for non-empty misses")
    if any(
        tensor.device != output_page_ids.device
        for tensor in (miss_output_page_offsets, staging_starts, page_counts)
    ):
        raise ValueError("Compact staging tensors must use one CUDA device")
    if any(
        tensor.numel() < num_misses
        for tensor in (miss_output_page_offsets, staging_starts, page_counts)
    ):
        raise ValueError("Compact staging input does not have enough capacity")
    if num_misses == 0:
        return

    _scatter_compact_staging_page_ids_kernel[(num_misses,)](
        miss_output_page_offsets,
        staging_starts,
        page_counts,
        output_page_ids.reshape(-1),
        num_misses,
        MAX_PAGES=max_pages,
        BLOCK_PAGES=triton.next_power_of_2(max_pages),
    )


def scatter_staging_page_ids(
    miss_positions: torch.Tensor,
    staging_starts: torch.Tensor,
    page_counts: torch.Tensor,
    num_misses: int,
    output_page_ids: torch.Tensor,
) -> None:
    if output_page_ids.device.type != "cuda":
        raise ValueError("Staging-page scatter requires CUDA tensors")
    if output_page_ids.ndim < 2:
        raise ValueError("Staging-page output must include a page dimension")
    if num_misses < 0:
        raise ValueError("num_misses must be non-negative")
    if any(
        tensor.device != output_page_ids.device
        for tensor in (miss_positions, staging_starts, page_counts)
    ):
        raise ValueError("Staging-page scatter tensors must use one CUDA device")
    if any(
        tensor.numel() < num_misses
        for tensor in (miss_positions, staging_starts, page_counts)
    ):
        raise ValueError("Staging-page scatter input does not have enough capacity")

    output_page_ids.fill_(-1)
    if num_misses == 0:
        return

    max_pages = output_page_ids.shape[-1]
    _scatter_staging_page_ids_kernel[(num_misses,)](
        miss_positions,
        staging_starts,
        page_counts,
        output_page_ids.reshape(-1),
        num_misses,
        max_pages,
        MAX_PAGES=max_pages,
        BLOCK_PAGES=triton.next_power_of_2(max_pages),
    )


def update_resident_handles(
    bucket_ids: torch.Tensor,
    cluster_handles: torch.Tensor,
    page_counts: torch.Tensor,
    page_slots: torch.Tensor,
    hit_gate_ready: torch.Tensor,
    table_handles: torch.Tensor,
    table_versions: torch.Tensor,
    table_page_counts: torch.Tensor,
    table_page_slots: torch.Tensor,
    table_hit_gate_ready: torch.Tensor,
) -> None:
    if cluster_handles.numel() == 0:
        return

    _update_resident_handles_kernel[(cluster_handles.numel(),)](
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
        cluster_handles.numel(),
        page_slots.stride(0),
        page_slots.stride(1),
        table_page_slots.stride(0),
        table_page_slots.stride(1),
        MAX_PAGES=table_page_slots.shape[1],
    )
