# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton

from .resident_verification_kernel_impl import (
    _map_compact_verification_miss_indices_kernel,
    _reset_verification_miss_hash_kernel,
    _resolve_compact_verification_pages_vector_kernel,
    _scatter_compact_staging_page_ids_kernel,
)


def resolve_compact_verification_pages(
    selected_cluster_indices: torch.Tensor,
    plan_valid_rows: torch.Tensor,
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
    output_miss_hash_buckets: torch.Tensor,
    output_miss_unique_indices: torch.Tensor,
    output_miss_page_offsets: torch.Tensor,
    output_miss_count: torch.Tensor,
    output_unique_handles: torch.Tensor,
    output_unique_logical_page_ids: torch.Tensor,
    output_unique_page_counts: torch.Tensor,
    output_unique_miss_count: torch.Tensor,
    miss_table_handles: torch.Tensor,
    miss_table_unique_indices: torch.Tensor,
    output_invalid_descriptor_count: torch.Tensor,
) -> None:
    if selected_cluster_indices.device.type != "cuda":
        raise ValueError("Compact verification resolution requires CUDA")
    if selected_cluster_indices.ndim != 3:
        raise ValueError("Selected clusters must have shape [rows, heads, clusters]")
    if plan_valid_rows.ndim != 1 or plan_valid_rows.dtype != torch.bool:
        raise ValueError("Plan validity must be one-dimensional and boolean")
    if request_slot_ids.shape != request_slot_generations.shape:
        raise ValueError("Request slot descriptors must have equal shapes")
    if request_slot_ids.ndim != 1:
        raise ValueError("Request slot descriptors must be one-dimensional")

    num_queries = selected_cluster_indices.shape[0]
    if plan_valid_rows.shape != (num_queries,):
        raise ValueError("Plan validity must contain one entry per query")
    if request_slot_ids.shape != (num_queries,):
        raise ValueError("Request slot descriptors must contain one entry per query")
    _, num_kv_heads, num_clusters = selected_cluster_indices.shape
    max_pages = output_unique_logical_page_ids.shape[1]
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
    if output_miss_hash_buckets.numel() < miss_capacity:
        raise ValueError("Verification miss output is too small")
    if any(
        output.numel() < miss_capacity
        for output in (output_miss_unique_indices, output_miss_page_offsets)
    ):
        raise ValueError("Verification miss metadata output is too small")
    if output_unique_handles.numel() < miss_capacity:
        raise ValueError("Unique verification output is too small")
    if output_unique_logical_page_ids.shape[0] < miss_capacity:
        raise ValueError("Unique verification page output is too small")
    if output_unique_page_counts.numel() < miss_capacity:
        raise ValueError("Unique verification page counts are too small")
    if output_miss_count.shape != (1,):
        raise ValueError("Verification miss count must contain one element")
    if output_unique_miss_count.shape != (1,):
        raise ValueError("Unique verification miss count must contain one element")
    if output_invalid_descriptor_count.shape != (1,):
        raise ValueError("Invalid descriptor count must contain one element")
    miss_table_capacity = miss_table_handles.numel()
    if miss_table_capacity < max(2, 2 * miss_capacity):
        raise ValueError("Verification miss hash table is too small")
    if miss_table_capacity & (miss_table_capacity - 1):
        raise ValueError("Verification miss hash capacity must be a power of two")
    if miss_table_unique_indices.shape != miss_table_handles.shape:
        raise ValueError("Verification miss hash arrays must have equal shapes")

    tensors = (
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
        output_resident_page_ids,
        output_staging_page_ids,
        output_page_token_counts,
        output_page_counts,
        output_selected_counts,
        output_hit_counts,
        output_miss_counts,
        output_miss_hash_buckets,
        output_miss_unique_indices,
        output_miss_page_offsets,
        output_miss_count,
        output_unique_handles,
        output_unique_logical_page_ids,
        output_unique_page_counts,
        output_unique_miss_count,
        miss_table_handles,
        miss_table_unique_indices,
        output_invalid_descriptor_count,
    )
    if any(tensor.device != selected_cluster_indices.device for tensor in tensors):
        raise ValueError("Compact verification tensors must use one CUDA device")

    reset_block_size = 256
    _reset_verification_miss_hash_kernel[
        (triton.cdiv(miss_table_capacity, reset_block_size),)
    ](
        miss_table_handles,
        output_miss_count,
        output_unique_miss_count,
        output_invalid_descriptor_count,
        TABLE_CAPACITY=miss_table_capacity,
        BLOCK_SIZE=reset_block_size,
    )
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
        table_page_slots.stride(0),
        output_unique_logical_page_ids.stride(0),
        selected_cluster_indices.stride(0),
        selected_cluster_indices.stride(1),
        selected_cluster_indices.stride(2),
        NUM_KV_HEADS=num_kv_heads,
        NUM_CLUSTERS=num_clusters,
        MAX_PAGES=max_pages,
        PAGE_CAPACITY=page_capacity,
        ARENA_CLUSTER_CAPACITY=arena_cluster_ids.shape[1],
        ARENA_PAGE_CAPACITY=arena_page_ids.shape[1],
        TABLE_CAPACITY=table_handles.numel(),
        MISS_TABLE_CAPACITY=miss_table_capacity,
        BLOCK_CLUSTERS=triton.next_power_of_2(num_clusters),
        BLOCK_PAGES=triton.next_power_of_2(max_pages),
        BLOCK_OUTPUT_PAGES=triton.next_power_of_2(page_capacity),
    )

    mapping_block_size = 256
    _map_compact_verification_miss_indices_kernel[
        (triton.cdiv(miss_capacity, mapping_block_size),)
    ](
        output_miss_hash_buckets,
        miss_table_unique_indices,
        output_miss_count,
        output_unique_miss_count,
        output_miss_unique_indices,
        output_invalid_descriptor_count,
        miss_capacity,
        miss_table_capacity,
        BLOCK_SIZE=mapping_block_size,
    )


def scatter_compact_staging_page_ids(
    miss_unique_indices: torch.Tensor,
    miss_output_page_offsets: torch.Tensor,
    unique_staging_starts: torch.Tensor,
    unique_page_counts: torch.Tensor,
    num_misses: int,
    num_unique_misses: int,
    max_pages: int,
    output_page_ids: torch.Tensor,
) -> None:
    if output_page_ids.device.type != "cuda":
        raise ValueError("Compact staging-page scatter requires CUDA")
    if num_misses < 0 or num_unique_misses < 0:
        raise ValueError("Verification miss counts must be non-negative")
    if num_unique_misses > num_misses:
        raise ValueError("Unique verification misses exceed total misses")
    if max_pages <= 0 and num_misses:
        raise ValueError("max_pages must be positive for non-empty misses")
    if any(
        tensor.device != output_page_ids.device
        for tensor in (
            miss_unique_indices,
            miss_output_page_offsets,
            unique_staging_starts,
            unique_page_counts,
        )
    ):
        raise ValueError("Compact staging tensors must use one CUDA device")
    if any(
        tensor.numel() < num_misses
        for tensor in (miss_unique_indices, miss_output_page_offsets)
    ):
        raise ValueError("Compact staging input does not have enough capacity")
    if any(
        tensor.numel() < num_unique_misses
        for tensor in (unique_staging_starts, unique_page_counts)
    ):
        raise ValueError("Unique staging input does not have enough capacity")
    if num_misses == 0:
        return

    _scatter_compact_staging_page_ids_kernel[(num_misses,)](
        miss_unique_indices,
        miss_output_page_offsets,
        unique_staging_starts,
        unique_page_counts,
        output_page_ids.reshape(-1),
        num_misses,
        num_unique_misses,
        MAX_PAGES=max_pages,
        BLOCK_PAGES=triton.next_power_of_2(max_pages),
    )
