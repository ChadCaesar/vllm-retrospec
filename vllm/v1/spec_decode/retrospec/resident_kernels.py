# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _lookup_resident_handles_kernel(
    cluster_handles,
    logical_page_ids,
    active_mask,
    table_handles,
    table_versions,
    table_page_counts,
    table_page_slots,
    table_hit_gate_ready,
    output_page_slots,
    output_hit_mask,
    output_miss_mask,
    output_hit_gate_ready,
    output_access_kinds,
    num_clusters,
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
):
    cluster_index = tl.program_id(0)
    valid_cluster = cluster_index < num_clusters

    handle = tl.load(
        cluster_handles + cluster_index * handle_stride,
        mask=valid_cluster,
        other=-1,
    ).to(tl.int64)
    request_index = cluster_index // CLUSTERS_PER_REQUEST
    active = tl.load(active_mask + request_index, mask=valid_cluster, other=0)
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
        + cluster_index * logical_page_stride_0
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

    tl.store(
        output_page_slots
        + cluster_index * output_page_stride_0
        + page_offsets * output_page_stride_1,
        tl.where(stable_hit, resident_slots, -1),
        mask=page_mask,
    )

    tl.store(output_hit_mask + cluster_index, stable_hit)
    tl.store(output_miss_mask + cluster_index, miss)
    tl.store(output_hit_gate_ready + cluster_index, stable_hit & gate_ready)
    access_kind = tl.where(stable_hit, 1, tl.where(miss, 2, 0))
    tl.store(output_access_kinds + cluster_index, access_kind)


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
    active_mask: torch.Tensor,
    table_handles: torch.Tensor,
    table_versions: torch.Tensor,
    table_page_counts: torch.Tensor,
    table_page_slots: torch.Tensor,
    table_hit_gate_ready: torch.Tensor,
    output_page_slots: torch.Tensor,
    output_hit_mask: torch.Tensor,
    output_miss_mask: torch.Tensor,
    output_hit_gate_ready: torch.Tensor,
    output_access_kinds: torch.Tensor,
) -> None:
    if cluster_handles.device.type != "cuda":
        raise ValueError("Resident handle lookup requires CUDA tensors")
    if cluster_handles.ndim != 3:
        raise ValueError("Cluster handles must have shape [batch, heads, clusters]")
    if logical_page_ids.shape[:-1] != cluster_handles.shape:
        raise ValueError("Logical pages do not match cluster handles")
    if active_mask.shape != (cluster_handles.shape[0],):
        raise ValueError("active_mask does not match the batch size")

    if table_handles.numel() == 0:
        output_page_slots.fill_(-1)
        output_hit_mask.zero_()
        valid = (cluster_handles >= 0) & active_mask[:, None, None]
        output_miss_mask.copy_(valid)
        output_hit_gate_ready.zero_()
        output_access_kinds.copy_(valid.to(torch.uint8) * 2)
        return

    table_capacity = table_handles.numel()
    if table_capacity & (table_capacity - 1):
        raise ValueError("Resident handle-table capacity must be a power of two")

    flat_handles = cluster_handles.reshape(-1)
    flat_pages = logical_page_ids.reshape(
        flat_handles.numel(), logical_page_ids.shape[-1]
    )
    flat_output_pages = output_page_slots.reshape_as(flat_pages)
    clusters_per_request = cluster_handles.shape[1] * cluster_handles.shape[2]

    _lookup_resident_handles_kernel[(flat_handles.numel(),)](
        flat_handles,
        flat_pages,
        active_mask,
        table_handles,
        table_versions,
        table_page_counts,
        table_page_slots,
        table_hit_gate_ready,
        flat_output_pages,
        output_hit_mask.reshape(-1),
        output_miss_mask.reshape(-1),
        output_hit_gate_ready.reshape(-1),
        output_access_kinds.reshape(-1),
        flat_handles.numel(),
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
