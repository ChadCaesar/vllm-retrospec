# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton

from .resident_kernel_impl import (
    _compact_resident_misses_kernel,
    _finalize_ranked_compact_draft_attention_kernel,
    _lookup_resident_handles_kernel,
    _map_compact_verification_miss_indices_kernel,
    _publish_resident_table_bindings_kernel,
    _reset_verification_miss_hash_kernel,
    _resolve_compact_draft_pages_kernel,
    _resolve_compact_verification_pages_vector_kernel,
    _resolve_ranked_draft_buckets_kernel,
    _scatter_compact_staging_page_ids_kernel,
    _scatter_staging_page_ids_kernel,
    _update_resident_handles_kernel,
)
from .resident_kernel_impl import (
    _find_resident_buckets as _find_resident_buckets,
)
from .resident_kernel_impl import (
    _record_resident_lookup_statistics as _record_resident_lookup_statistics,
)
from .resident_kernel_impl import (
    _resident_handle_hash as _resident_handle_hash,
)

_DRAFT_RESOLVE_STATISTIC_COUNT = 11


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


def publish_resident_table_bindings(
    binding_commands: torch.Tensor,
    arena_cluster_ids: torch.Tensor,
    arena_resident_table_buckets: torch.Tensor,
    arena_cluster_offsets: torch.Tensor,
    arena_num_clusters: torch.Tensor,
    arena_generations: torch.Tensor,
) -> None:
    if binding_commands.ndim != 2 or binding_commands.shape[1] != 6:
        raise ValueError("Resident binding commands must have shape [count, 6]")
    if binding_commands.dtype != torch.int64:
        raise ValueError("Resident binding commands must use int64")
    num_bindings = binding_commands.shape[0]
    if arena_cluster_ids.shape != arena_resident_table_buckets.shape:
        raise ValueError("Resident bucket bindings must match cluster IDs")
    if arena_cluster_ids.device.type != "cuda":
        raise ValueError("Resident binding publication requires CUDA tensors")
    if binding_commands.device != arena_cluster_ids.device:
        raise ValueError("Resident binding commands must use the arena device")
    if num_bindings == 0:
        return

    _publish_resident_table_bindings_kernel[(num_bindings,)](
        binding_commands,
        arena_cluster_ids,
        arena_resident_table_buckets,
        arena_cluster_offsets,
        arena_num_clusters,
        arena_generations,
        num_bindings,
        binding_commands.stride(0),
        binding_commands.stride(1),
        arena_cluster_ids.stride(0),
        arena_cluster_ids.stride(1),
    )


# Preserve the original qualified path for serialized classes.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = __name__.replace(".offload.", ".")
del _legacy_type
