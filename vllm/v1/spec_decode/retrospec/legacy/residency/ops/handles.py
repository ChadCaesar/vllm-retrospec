# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.legacy.residency.kernels.handles import (
    _compact_resident_misses_kernel,
    _lookup_resident_handles_kernel,
    _publish_resident_table_bindings_kernel,
    _scatter_staging_page_ids_kernel,
    _update_resident_handles_kernel,
)


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
