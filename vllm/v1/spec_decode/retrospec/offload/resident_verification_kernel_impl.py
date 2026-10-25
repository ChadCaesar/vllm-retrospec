# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton

from .resident_kernel_helpers import _resident_handle_hash


@triton.jit(do_not_specialize=["TABLE_CAPACITY"])
def _reset_verification_miss_hash_kernel(
    miss_table_handles,
    output_miss_count,
    output_unique_miss_count,
    output_invalid_descriptor_count,
    TABLE_CAPACITY,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < TABLE_CAPACITY
    tl.store(miss_table_handles + offsets, -1, mask=valid)

    reset_counter = offsets == 0
    tl.store(output_miss_count + offsets, 0, mask=reset_counter)
    tl.store(output_unique_miss_count + offsets, 0, mask=reset_counter)
    tl.store(output_invalid_descriptor_count + offsets, 0, mask=reset_counter)


@triton.jit(do_not_specialize=["num_pages"])
def _reset_compact_verification_pages_kernel(
    output_resident_page_ids,
    output_staging_page_ids,
    output_page_token_counts,
    num_pages,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_pages
    tl.store(output_resident_page_ids + offsets, -1, mask=valid)
    tl.store(output_staging_page_ids + offsets, -1, mask=valid)
    tl.store(output_page_token_counts + offsets, 0, mask=valid)


@triton.jit(
    do_not_specialize=[
        "access_epoch",
        "table_page_stride",
        "output_unique_logical_page_stride",
        "selected_stride_0",
        "selected_stride_1",
        "selected_stride_2",
        "NUM_KV_HEADS",
        "NUM_CLUSTERS",
        "MAX_PAGES",
        "PAGE_CAPACITY",
        "ARENA_CLUSTER_CAPACITY",
        "ARENA_PAGE_CAPACITY",
        "TABLE_CAPACITY",
        "MISS_TABLE_CAPACITY",
    ]
)
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
    NUM_KV_HEADS,
    NUM_CLUSTERS,
    MAX_PAGES,
    PAGE_CAPACITY,
    ARENA_CLUSTER_CAPACITY,
    ARENA_PAGE_CAPACITY,
    TABLE_CAPACITY,
    MISS_TABLE_CAPACITY,
    BLOCK_CLUSTERS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
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

    output_base = row * PAGE_CAPACITY

    # Keep rank/page intermediates bounded by one cluster tile instead of
    # expanding the complete query row (hundreds of clusters at long contexts).
    page_count = 0
    selected_count = 0
    hit_count = 0
    miss_count = 0
    hash_failure_count = 0
    has_selected = False
    rank_start = 0
    while tl.condition(rank_start < NUM_CLUSTERS, disable_licm=True):
        ranks = rank_start + tl.arange(0, BLOCK_CLUSTERS)
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
        has_selected |= tl.sum(selected.to(tl.int32), axis=0) > 0
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
        probe = 0
        while tl.condition(
            (probe < 64) & (tl.sum(searching.to(tl.int32), axis=0) > 0),
            disable_licm=True,
        ):
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
            probe += 1

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
        compact_starts = page_count + compact_ends - selected_page_counts
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
        first_miss_buckets = handles & (MISS_TABLE_CAPACITY - 1)
        empty_handles = tl.full((BLOCK_CLUSTERS,), -1, tl.int64)
        inactive_handles = tl.full((BLOCK_CLUSTERS,), -2, tl.int64)
        probe = 0
        while tl.condition(
            (probe < MISS_TABLE_CAPACITY)
            & (tl.sum(searching.to(tl.int32), axis=0) > 0),
            disable_licm=True,
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

        hash_failures = misses & (miss_buckets < 0)
        hash_failure_count += tl.sum(hash_failures.to(tl.int32), axis=0)
        page_count += tl.sum(selected_page_counts, axis=0)
        selected_count += tl.sum(selected.to(tl.int32), axis=0)
        hit_count += tl.sum(stable_hits.to(tl.int32), axis=0)
        miss_count += num_misses
        rank_start += BLOCK_CLUSTERS

    invalid_descriptor = ~plan_valid | (has_selected & ~descriptor_valid)
    tl.atomic_add(
        output_invalid_descriptor_count,
        invalid_descriptor.to(tl.int32) + hash_failure_count,
    )
    tl.store(output_page_counts + row, page_count)
    tl.store(output_selected_counts + row, selected_count)
    tl.store(output_hit_counts + row, hit_count)
    tl.store(output_miss_counts + row, miss_count)


@triton.jit(do_not_specialize=["miss_capacity", "table_capacity"])
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


@triton.jit(do_not_specialize=["num_misses", "num_unique_misses", "MAX_PAGES"])
def _scatter_compact_staging_page_ids_kernel(
    miss_unique_indices,
    miss_output_page_offsets,
    unique_staging_starts,
    unique_page_counts,
    output_page_ids,
    num_misses,
    num_unique_misses,
    MAX_PAGES,
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
