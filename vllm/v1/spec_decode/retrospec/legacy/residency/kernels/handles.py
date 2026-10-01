# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.retrospec.legacy.residency.kernels.common import (
    _resident_handle_hash,
)

_DRAFT_RESOLVE_STATISTIC_COUNT = 11


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
