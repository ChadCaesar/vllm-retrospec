# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.legacy.residency.kernels.draft import (
    _finalize_ranked_compact_draft_attention_kernel,
    _resolve_compact_draft_pages_kernel,
    _resolve_ranked_draft_buckets_kernel,
)

_DRAFT_RESOLVE_STATISTIC_COUNT = 11


def resolve_compact_draft_pages(
    *,
    ranked_values: torch.Tensor,
    candidate_counts: torch.Tensor,
    arena_resident_table_buckets: torch.Tensor,
    arena_cluster_page_starts: torch.Tensor,
    arena_cluster_page_counts: torch.Tensor,
    arena_page_ids: torch.Tensor,
    arena_page_token_counts: torch.Tensor,
    arena_cluster_offsets: torch.Tensor,
    arena_page_offsets: torch.Tensor,
    request_slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    table_handles: torch.Tensor,
    table_versions: torch.Tensor,
    table_page_counts: torch.Tensor,
    table_page_slots: torch.Tensor,
    table_hit_gate_ready: torch.Tensor,
    table_last_access_epochs: torch.Tensor,
    access_epoch: int,
    retrieval_ratio: float,
    estimation_ratio: float,
    expanded_retrieval_width: int,
    max_pages_per_cluster: int,
    fallback_token_counts: torch.Tensor,
    sparse_cluster_indices: torch.Tensor,
    cluster_handles: torch.Tensor,
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
    sparse_attention: torch.Tensor,
    expanded_attention: torch.Tensor,
    emit_misses: bool = True,
    statistics_buffer: torch.Tensor | None = None,
    statistics_indices: tuple[int, ...] | None = None,
) -> None:
    """Resolve a prepacked DRAFT plan into resident page descriptors."""
    if ranked_values.device.type != "cuda":
        raise ValueError("Compact draft resolution requires CUDA")
    if ranked_values.ndim != 3:
        raise ValueError("Ranked values must have shape [batch, heads, ranks]")
    if candidate_counts.dtype != torch.int32:
        raise ValueError("Candidate counts must use int32")
    if max_pages_per_cluster <= 0:
        raise ValueError("Maximum pages per cluster must be positive")

    batch_size, num_kv_heads, ranking_width = ranked_values.shape
    sparse_width = sparse_cluster_indices.shape[2]
    row_shape = (batch_size, num_kv_heads)
    page_capacity = sparse_width * max_pages_per_cluster

    if candidate_counts.shape != row_shape:
        raise ValueError("Candidate counts do not match ranked rows")
    if request_slot_ids.shape != (batch_size,):
        raise ValueError("Request slots do not match ranked rows")
    if active_mask.shape != (batch_size,):
        raise ValueError("Active mask does not match ranked rows")
    if arena_resident_table_buckets.shape != arena_cluster_page_starts.shape:
        raise ValueError("Resident bucket bindings do not match cluster descriptors")
    if arena_resident_table_buckets.dtype != torch.int32:
        raise ValueError("Resident bucket bindings must use int32")
    if sparse_cluster_indices.shape[:2] != row_shape:
        raise ValueError("Sparse journal has the wrong row shape")
    if cluster_handles.shape != sparse_cluster_indices.shape:
        raise ValueError("Draft cluster handles have the wrong shape")
    if cluster_handles.dtype != torch.int64:
        raise ValueError("Draft cluster handles must use int64")
    if fallback_token_counts.shape != sparse_cluster_indices.shape:
        raise ValueError("Fallback summary counts have the wrong shape")
    if output_page_slots.shape != (*row_shape, page_capacity):
        raise ValueError("Compact page output has the wrong shape")
    if output_page_token_counts.shape != output_page_slots.shape:
        raise ValueError("Compact page token counts have the wrong shape")
    if expanded_retrieval_width < 0:
        raise ValueError("Expanded retrieval width must be non-negative")
    if ranking_width < max(sparse_width, expanded_retrieval_width):
        raise ValueError("Ranked workspace is too narrow")
    if table_page_slots.shape[1] < max_pages_per_cluster:
        raise ValueError("Resident table has too few page slots")
    if output_miss_handles.numel() < cluster_handles.numel():
        raise ValueError("Miss-handle output has insufficient capacity")
    if output_miss_positions.numel() < cluster_handles.numel():
        raise ValueError("Miss-position output has insufficient capacity")
    if output_miss_count.shape != (1,):
        raise ValueError("Miss count must contain one element")
    if table_handles.numel() == 0 or table_handles.numel() & (
        table_handles.numel() - 1
    ):
        raise ValueError("Resident handle-table capacity must be a power of two")
    if access_epoch <= 0:
        raise ValueError("Resident access epoch must be positive")

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
            raise ValueError("Compact row output has the wrong shape")
    for output in (output_attention, sparse_attention, expanded_attention):
        if output.shape != (batch_size,):
            raise ValueError("Attention output has the wrong shape")

    update_statistics = statistics_buffer is not None
    if update_statistics != (statistics_indices is not None):
        raise ValueError(
            "Statistics buffer and counter indices must be provided together"
        )

    if statistics_buffer is None:
        statistics_buffer = output_miss_count
        statistics_indices = (0,) * _DRAFT_RESOLVE_STATISTIC_COUNT
    else:
        if statistics_buffer.ndim != 1:
            raise ValueError("Statistics buffer must be one-dimensional")
        if statistics_buffer.dtype != torch.int64:
            raise ValueError("Statistics buffer must use int64")
        if statistics_buffer.device != ranked_values.device:
            raise ValueError("Statistics buffer must use the resolve device")
        if (
            statistics_indices is None
            or len(statistics_indices) != _DRAFT_RESOLVE_STATISTIC_COUNT
        ):
            raise ValueError(
                f"DRAFT resolve requires {_DRAFT_RESOLVE_STATISTIC_COUNT} "
                "counter indices"
            )
        if any(
            index < 0 or index >= statistics_buffer.numel()
            for index in statistics_indices
        ):
            raise ValueError("Statistics counter index is outside the buffer")

    assert statistics_indices is not None
    (
        resident_hit_counter_index,
        resident_miss_counter_index,
        resident_page_counter_index,
        selected_cluster_counter_index,
        bound_direct_hit_counter_index,
        hash_fallback_lookup_counter_index,
        hash_fallback_hit_counter_index,
        hash_fallback_miss_counter_index,
        hash_probe_step_counter_index,
        hash_max_probe_counter_index,
        binding_invalidation_counter_index,
    ) = statistics_indices

    tensors = (
        ranked_values,
        candidate_counts,
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
        fallback_token_counts,
        sparse_cluster_indices,
        cluster_handles,
        output_page_slots,
        output_page_token_counts,
        output_page_counts,
        output_clustered_token_counts,
        output_attention,
        output_hit_attention_by_head,
        output_selected_counts,
        output_hit_counts,
        output_miss_counts,
        output_gate_ready,
        output_miss_handles,
        output_miss_positions,
        output_miss_count,
        statistics_buffer,
        sparse_attention,
        expanded_attention,
    )
    if any(tensor.device != ranked_values.device for tensor in tensors):
        raise ValueError("Compact draft tensors must use one device")

    if emit_misses:
        output_miss_count.zero_()
    if sparse_width == 0:
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
        sparse_attention.fill_(1.0)
        expanded_attention.fill_(1.0)
        return

    _resolve_compact_draft_pages_kernel[(batch_size * num_kv_heads,)](
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
        ranked_values.stride(0),
        ranked_values.stride(1),
        ranked_values.stride(2),
        fallback_token_counts.stride(1),
        table_page_slots.stride(0),
        ARENA_CLUSTER_CAPACITY=arena_cluster_page_starts.shape[1],
        ARENA_PAGE_CAPACITY=arena_page_ids.shape[1],
        NUM_KV_HEADS=num_kv_heads,
        SPARSE_WIDTH=sparse_width,
        MAX_PAGES=max_pages_per_cluster,
        PAGE_CAPACITY=page_capacity,
        TABLE_CAPACITY=table_handles.numel(),
        BLOCK_PAGES=triton.next_power_of_2(max_pages_per_cluster),
        BLOCK_OUTPUT_PAGES=triton.next_power_of_2(page_capacity),
        BLOCK_SPARSE=triton.next_power_of_2(sparse_width),
        EMIT_MISSES=emit_misses,
        UPDATE_STATISTICS=update_statistics,
        RESIDENT_HIT_COUNTER_INDEX=resident_hit_counter_index,
        RESIDENT_MISS_COUNTER_INDEX=resident_miss_counter_index,
        RESIDENT_PAGE_COUNTER_INDEX=resident_page_counter_index,
        SELECTED_CLUSTER_COUNTER_INDEX=selected_cluster_counter_index,
        BOUND_DIRECT_HIT_COUNTER_INDEX=bound_direct_hit_counter_index,
        HASH_FALLBACK_LOOKUP_COUNTER_INDEX=hash_fallback_lookup_counter_index,
        HASH_FALLBACK_HIT_COUNTER_INDEX=hash_fallback_hit_counter_index,
        HASH_FALLBACK_MISS_COUNTER_INDEX=hash_fallback_miss_counter_index,
        HASH_PROBE_STEP_COUNTER_INDEX=hash_probe_step_counter_index,
        HASH_MAX_PROBE_COUNTER_INDEX=hash_max_probe_counter_index,
        BINDING_INVALIDATION_COUNTER_INDEX=binding_invalidation_counter_index,
    )
    _finalize_ranked_compact_draft_attention_kernel[(batch_size,)](
        ranked_values,
        candidate_counts,
        request_slot_ids,
        active_mask,
        output_hit_attention_by_head,
        output_gate_ready,
        output_attention,
        sparse_attention,
        expanded_attention,
        NUM_KV_HEADS=num_kv_heads,
        RANKING_WIDTH=ranking_width,
        BLOCK_HEADS=triton.next_power_of_2(num_kv_heads),
        BLOCK_RANK=triton.next_power_of_2(max(expanded_retrieval_width, 1)),
        RANKED_STRIDE_0=ranked_values.stride(0),
        RANKED_STRIDE_1=ranked_values.stride(1),
        RANKED_STRIDE_2=ranked_values.stride(2),
        RETRIEVAL_RATIO=retrieval_ratio,
        ESTIMATION_RATIO=estimation_ratio,
    )


def resolve_ranked_draft_buckets(
    *,
    ranked_values: torch.Tensor,
    ranked_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    arena_cluster_ids: torch.Tensor,
    arena_resident_table_buckets: torch.Tensor,
    arena_cluster_token_counts: torch.Tensor,
    arena_cluster_page_counts: torch.Tensor,
    arena_cluster_offsets: torch.Tensor,
    arena_generations: torch.Tensor,
    request_slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    table_handles: torch.Tensor,
    table_versions: torch.Tensor,
    table_page_counts: torch.Tensor,
    table_page_slots: torch.Tensor,
    table_hit_gate_ready: torch.Tensor,
    table_last_access_epochs: torch.Tensor,
    access_epoch: int,
    retrieval_ratio: float,
    estimation_ratio: float,
    expanded_retrieval_width: int,
    max_pages_per_cluster: int,
    output_valid_rows: torch.Tensor,
    output_request_slot_ids: torch.Tensor,
    output_request_slot_generations: torch.Tensor,
    output_cluster_handles: torch.Tensor,
    output_resident_buckets: torch.Tensor,
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
    sparse_attention: torch.Tensor,
    expanded_attention: torch.Tensor,
    capture_request_descriptors: bool,
    emit_misses: bool = True,
    statistics_buffer: torch.Tensor | None = None,
    statistics_indices: tuple[int, ...] | None = None,
) -> None:
    """Resolve ranked DRAFT clusters into stable resident-table buckets."""
    if ranked_values.device.type != "cuda" or ranked_values.ndim != 3:
        raise ValueError("Ranked values must be CUDA [batch, heads, ranks]")
    if ranked_indices.shape != ranked_values.shape:
        raise ValueError("Ranked indices must match ranked values")
    if ranked_indices.dtype != torch.int64:
        raise ValueError("Ranked indices must use int64")
    if candidate_counts.dtype != torch.int32:
        raise ValueError("Candidate counts must use int32")
    if max_pages_per_cluster <= 0:
        raise ValueError("Maximum pages per cluster must be positive")

    batch_size, num_kv_heads, ranking_width = ranked_values.shape
    row_shape = (batch_size, num_kv_heads)
    sparse_width = output_cluster_handles.shape[2]
    if candidate_counts.shape != row_shape:
        raise ValueError("Candidate counts do not match ranked rows")
    if request_slot_ids.shape != (batch_size,):
        raise ValueError("Request slots do not match ranked rows")
    if active_mask.shape != (batch_size,) or active_mask.dtype != torch.bool:
        raise ValueError("Active mask must be a one-dimensional bool tensor")
    if output_cluster_handles.shape[:2] != row_shape:
        raise ValueError("Cluster-handle output has the wrong shape")
    if output_cluster_handles.dtype != torch.int64:
        raise ValueError("Cluster handles must use int64")
    if output_resident_buckets.shape != output_cluster_handles.shape:
        raise ValueError("Resident-bucket output has the wrong shape")
    if output_resident_buckets.dtype != torch.int32:
        raise ValueError("Resident buckets must use int32")
    if table_page_slots.shape[1] < max_pages_per_cluster:
        raise ValueError("Resident table has too few page slots")
    if expanded_retrieval_width < sparse_width:
        raise ValueError("Expanded retrieval width is smaller than sparse retrieval")
    if ranking_width < expanded_retrieval_width:
        raise ValueError("Ranked workspace is too narrow")
    if output_miss_handles.numel() < output_cluster_handles.numel():
        raise ValueError("Miss-handle output has insufficient capacity")
    if output_miss_positions.numel() < output_cluster_handles.numel():
        raise ValueError("Miss-position output has insufficient capacity")
    if output_miss_count.shape != (1,):
        raise ValueError("Miss count must contain one element")
    if table_handles.numel() == 0 or table_handles.numel() & (
        table_handles.numel() - 1
    ):
        raise ValueError("Resident handle-table capacity must be a power of two")
    if access_epoch <= 0:
        raise ValueError("Resident access epoch must be positive")
    plan_row_outputs = (
        output_valid_rows,
        output_request_slot_ids,
        output_request_slot_generations,
    )
    if any(output.shape != (batch_size,) for output in plan_row_outputs):
        raise ValueError("Plan row outputs have the wrong shape")
    if output_valid_rows.dtype != torch.bool:
        raise ValueError("Plan valid rows must use bool")
    if any(
        output.dtype not in (torch.int32, torch.int64)
        for output in (
            output_request_slot_ids,
            output_request_slot_generations,
        )
    ):
        raise ValueError("Request descriptors must use integral tensors")

    row_outputs = (
        output_clustered_token_counts,
        output_hit_attention_by_head,
        output_selected_counts,
        output_hit_counts,
        output_miss_counts,
        output_gate_ready,
    )
    if any(output.shape != row_shape for output in row_outputs):
        raise ValueError("Ranked resolver row output has the wrong shape")
    for output in (output_attention, sparse_attention, expanded_attention):
        if output.shape != (batch_size,):
            raise ValueError("Attention output has the wrong shape")

    update_statistics = statistics_buffer is not None
    if update_statistics != (statistics_indices is not None):
        raise ValueError(
            "Statistics buffer and counter indices must be provided together"
        )
    if statistics_buffer is None:
        statistics_buffer = output_miss_count
        statistics_indices = (0,) * _DRAFT_RESOLVE_STATISTIC_COUNT
    else:
        if statistics_buffer.ndim != 1 or statistics_buffer.dtype != torch.int64:
            raise ValueError("Statistics buffer must be one-dimensional int64")
        if statistics_buffer.device != ranked_values.device:
            raise ValueError("Statistics buffer must use the resolve device")
        if (
            statistics_indices is None
            or len(statistics_indices) != _DRAFT_RESOLVE_STATISTIC_COUNT
        ):
            raise ValueError(
                f"DRAFT resolve requires {_DRAFT_RESOLVE_STATISTIC_COUNT} "
                "counter indices"
            )
        if any(
            index < 0 or index >= statistics_buffer.numel()
            for index in statistics_indices
        ):
            raise ValueError("Statistics counter index is outside the buffer")

    tensors = (
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
        *plan_row_outputs,
        output_cluster_handles,
        output_resident_buckets,
        *row_outputs,
        output_attention,
        output_miss_handles,
        output_miss_positions,
        output_miss_count,
        statistics_buffer,
        sparse_attention,
        expanded_attention,
    )
    if any(tensor.device != ranked_values.device for tensor in tensors):
        raise ValueError("Ranked DRAFT tensors must use one device")

    if emit_misses:
        output_miss_count.zero_()
    assert statistics_indices is not None
    _resolve_ranked_draft_buckets_kernel[(batch_size * num_kv_heads,)](
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
        ranked_values.stride(0),
        ranked_values.stride(1),
        ranked_values.stride(2),
        ranked_indices.stride(0),
        ranked_indices.stride(1),
        ranked_indices.stride(2),
        table_page_slots.stride(0),
        max_pages_per_cluster,
        ARENA_CLUSTER_CAPACITY=arena_cluster_ids.shape[1],
        NUM_KV_HEADS=num_kv_heads,
        SPARSE_WIDTH=sparse_width,
        TABLE_CAPACITY=table_handles.numel(),
        BLOCK_SPARSE=triton.next_power_of_2(max(sparse_width, 1)),
        CAPTURE_REQUEST_DESCRIPTORS=capture_request_descriptors,
        EMIT_MISSES=emit_misses,
        UPDATE_STATISTICS=update_statistics,
        RETRIEVAL_RATIO=retrieval_ratio,
        RESIDENT_HIT_COUNTER_INDEX=statistics_indices[0],
        RESIDENT_MISS_COUNTER_INDEX=statistics_indices[1],
        RESIDENT_PAGE_COUNTER_INDEX=statistics_indices[2],
        SELECTED_CLUSTER_COUNTER_INDEX=statistics_indices[3],
        BOUND_DIRECT_HIT_COUNTER_INDEX=statistics_indices[4],
        HASH_FALLBACK_LOOKUP_COUNTER_INDEX=statistics_indices[5],
        HASH_FALLBACK_HIT_COUNTER_INDEX=statistics_indices[6],
        HASH_FALLBACK_MISS_COUNTER_INDEX=statistics_indices[7],
        HASH_PROBE_STEP_COUNTER_INDEX=statistics_indices[8],
        HASH_MAX_PROBE_COUNTER_INDEX=statistics_indices[9],
        BINDING_INVALIDATION_COUNTER_INDEX=statistics_indices[10],
    )
    _finalize_ranked_compact_draft_attention_kernel[(batch_size,)](
        ranked_values,
        candidate_counts,
        request_slot_ids,
        active_mask,
        output_hit_attention_by_head,
        output_gate_ready,
        output_attention,
        sparse_attention,
        expanded_attention,
        NUM_KV_HEADS=num_kv_heads,
        RANKING_WIDTH=ranking_width,
        BLOCK_HEADS=triton.next_power_of_2(num_kv_heads),
        BLOCK_RANK=triton.next_power_of_2(max(expanded_retrieval_width, 1)),
        RANKED_STRIDE_0=ranked_values.stride(0),
        RANKED_STRIDE_1=ranked_values.stride(1),
        RANKED_STRIDE_2=ranked_values.stride(2),
        RETRIEVAL_RATIO=retrieval_ratio,
        ESTIMATION_RATIO=estimation_ratio,
    )
