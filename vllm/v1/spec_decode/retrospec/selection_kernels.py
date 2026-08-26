# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_resident_estimation_kernel(
    cluster_keys,
    cluster_values,
    cluster_token_counts,
    request_slot_ids,
    selected_indices,
    selected_mask,
    output_keys,
    output_values,
    output_token_counts,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    value_stride_3,
    count_stride_0,
    count_stride_1,
    count_stride_2,
    request_slot_stride_0,
    selected_stride_0,
    selected_stride_1,
    selected_stride_2,
    selected_mask_stride_0,
    selected_mask_stride_1,
    selected_mask_stride_2,
    output_key_stride_0,
    output_key_stride_1,
    output_key_stride_2,
    output_key_stride_3,
    output_value_stride_0,
    output_value_stride_1,
    output_value_stride_2,
    output_value_stride_3,
    output_count_stride_0,
    output_count_stride_1,
    output_count_stride_2,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    selected_idx = tl.program_id(2)

    selected_offset = (
        batch_idx * selected_stride_0
        + kv_head_idx * selected_stride_1
        + selected_idx * selected_stride_2
    )
    request_slot = tl.load(request_slot_ids + batch_idx * request_slot_stride_0)
    cluster_idx = tl.load(selected_indices + selected_offset)
    selected_mask_offset = (
        batch_idx * selected_mask_stride_0
        + kv_head_idx * selected_mask_stride_1
        + selected_idx * selected_mask_stride_2
    )
    valid = tl.load(selected_mask + selected_mask_offset).to(tl.int1) & (
        request_slot >= 0
    )

    safe_slot = tl.maximum(request_slot, 0).to(tl.int64)
    safe_cluster = tl.maximum(cluster_idx, 0).to(tl.int64)
    count_offset = (
        safe_slot * count_stride_0
        + kv_head_idx * count_stride_1
        + safe_cluster * count_stride_2
    )
    token_count = tl.load(cluster_token_counts + count_offset, mask=valid, other=0)
    valid &= token_count > 0

    head_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < HEAD_SIZE
    source_key_offsets = (
        safe_slot * key_stride_0
        + kv_head_idx * key_stride_1
        + safe_cluster * key_stride_2
        + head_offsets * key_stride_3
    )
    source_value_offsets = (
        safe_slot * value_stride_0
        + kv_head_idx * value_stride_1
        + safe_cluster * value_stride_2
        + head_offsets * value_stride_3
    )
    output_key_offsets = (
        batch_idx * output_key_stride_0
        + kv_head_idx * output_key_stride_1
        + selected_idx * output_key_stride_2
        + head_offsets * output_key_stride_3
    )
    output_value_offsets = (
        batch_idx * output_value_stride_0
        + kv_head_idx * output_value_stride_1
        + selected_idx * output_value_stride_2
        + head_offsets * output_value_stride_3
    )
    copy_mask = valid & head_mask

    keys = tl.load(cluster_keys + source_key_offsets, mask=copy_mask, other=0.0)
    values = tl.load(cluster_values + source_value_offsets, mask=copy_mask, other=0.0)
    tl.store(output_keys + output_key_offsets, keys, mask=head_mask)
    tl.store(output_values + output_value_offsets, values, mask=head_mask)

    output_count_offset = (
        batch_idx * output_count_stride_0
        + kv_head_idx * output_count_stride_1
        + selected_idx * output_count_stride_2
    )
    tl.store(output_token_counts + output_count_offset, tl.where(valid, token_count, 0))


@triton.jit
def _gather_resident_exact_pages_kernel(
    cluster_ids,
    cluster_page_offsets,
    page_ids,
    page_token_counts,
    request_slot_ids,
    selected_indices,
    selected_mask,
    output_cluster_ids,
    output_page_ids,
    output_page_token_counts,
    cluster_stride_0,
    cluster_stride_1,
    cluster_stride_2,
    offset_stride_0,
    offset_stride_1,
    offset_stride_2,
    page_stride_0,
    page_stride_1,
    page_stride_2,
    page_count_stride_0,
    page_count_stride_1,
    page_count_stride_2,
    request_slot_stride_0,
    selected_stride_0,
    selected_stride_1,
    selected_stride_2,
    selected_mask_stride_0,
    selected_mask_stride_1,
    selected_mask_stride_2,
    output_cluster_stride_0,
    output_cluster_stride_1,
    output_cluster_stride_2,
    output_page_stride_0,
    output_page_stride_1,
    output_page_stride_2,
    output_page_stride_3,
    output_page_count_stride_0,
    output_page_count_stride_1,
    output_page_count_stride_2,
    output_page_count_stride_3,
    MAX_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    selected_idx = tl.program_id(2)

    selected_offset = (
        batch_idx * selected_stride_0
        + kv_head_idx * selected_stride_1
        + selected_idx * selected_stride_2
    )
    request_slot = tl.load(request_slot_ids + batch_idx * request_slot_stride_0)
    cluster_idx = tl.load(selected_indices + selected_offset)
    selected_mask_offset = (
        batch_idx * selected_mask_stride_0
        + kv_head_idx * selected_mask_stride_1
        + selected_idx * selected_mask_stride_2
    )
    valid_cluster = tl.load(selected_mask + selected_mask_offset).to(tl.int1) & (
        request_slot >= 0
    )
    safe_slot = tl.maximum(request_slot, 0).to(tl.int64)
    safe_cluster = tl.maximum(cluster_idx, 0).to(tl.int64)

    cluster_offset = (
        safe_slot * cluster_stride_0
        + kv_head_idx * cluster_stride_1
        + safe_cluster * cluster_stride_2
    )
    cluster_id = tl.load(cluster_ids + cluster_offset, mask=valid_cluster, other=-1)
    valid_cluster &= cluster_id >= 0
    output_cluster_offset = (
        batch_idx * output_cluster_stride_0
        + kv_head_idx * output_cluster_stride_1
        + selected_idx * output_cluster_stride_2
    )
    tl.store(
        output_cluster_ids + output_cluster_offset,
        tl.where(valid_cluster, cluster_id, -1),
    )

    page_start_offset = (
        safe_slot * offset_stride_0
        + kv_head_idx * offset_stride_1
        + safe_cluster * offset_stride_2
    )
    page_start = tl.load(
        cluster_page_offsets + page_start_offset, mask=valid_cluster, other=0
    )
    page_end = tl.load(
        cluster_page_offsets + page_start_offset + offset_stride_2,
        mask=valid_cluster,
        other=0,
    )

    for page_offset in tl.static_range(0, MAX_PAGES):
        source_page_idx = page_start + page_offset
        valid_page = valid_cluster & (source_page_idx < page_end)
        source_page_offset = (
            safe_slot * page_stride_0
            + kv_head_idx * page_stride_1
            + source_page_idx * page_stride_2
        )
        source_page_count_offset = (
            safe_slot * page_count_stride_0
            + kv_head_idx * page_count_stride_1
            + source_page_idx * page_count_stride_2
        )
        page_id = tl.load(page_ids + source_page_offset, mask=valid_page, other=-1)
        page_count = tl.load(
            page_token_counts + source_page_count_offset, mask=valid_page, other=0
        )
        output_page_offset = (
            batch_idx * output_page_stride_0
            + kv_head_idx * output_page_stride_1
            + selected_idx * output_page_stride_2
            + page_offset * output_page_stride_3
        )
        output_page_count_offset = (
            batch_idx * output_page_count_stride_0
            + kv_head_idx * output_page_count_stride_1
            + selected_idx * output_page_count_stride_2
            + page_offset * output_page_count_stride_3
        )
        tl.store(output_page_ids + output_page_offset, page_id)
        tl.store(output_page_token_counts + output_page_count_offset, page_count)


def gather_resident_estimation(
    cluster_keys: torch.Tensor,
    cluster_values: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    request_slot_ids: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_mask: torch.Tensor,
    output_keys: torch.Tensor,
    output_values: torch.Tensor,
    output_token_counts: torch.Tensor,
) -> None:
    """Gather resident cluster summaries into caller-owned CUDA storage."""
    batch_size, num_kv_heads, max_selected = selected_indices.shape
    head_size = cluster_keys.shape[3]
    expected_output_shape = (batch_size, num_kv_heads, max_selected, head_size)
    if output_keys.shape != expected_output_shape or output_values.shape != (
        expected_output_shape
    ):
        raise ValueError("Estimation output shapes do not match selected clusters")
    if output_token_counts.shape != expected_output_shape[:-1]:
        raise ValueError("Estimation count output shape does not match")
    if max_selected == 0:
        return

    block_d = triton.next_power_of_2(head_size)
    _gather_resident_estimation_kernel[(batch_size, num_kv_heads, max_selected)](
        cluster_keys,
        cluster_values,
        cluster_token_counts,
        request_slot_ids,
        selected_indices,
        selected_mask,
        output_keys,
        output_values,
        output_token_counts,
        *cluster_keys.stride(),
        *cluster_values.stride(),
        *cluster_token_counts.stride(),
        request_slot_ids.stride(0),
        *selected_indices.stride(),
        *selected_mask.stride(),
        *output_keys.stride(),
        *output_values.stride(),
        *output_token_counts.stride(),
        HEAD_SIZE=head_size,
        BLOCK_D=block_d,
    )


def gather_resident_exact_pages(
    cluster_ids: torch.Tensor,
    cluster_page_offsets: torch.Tensor,
    page_ids: torch.Tensor,
    page_token_counts: torch.Tensor,
    request_slot_ids: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_mask: torch.Tensor,
    output_cluster_ids: torch.Tensor,
    output_page_ids: torch.Tensor,
    output_page_token_counts: torch.Tensor,
) -> None:
    """Gather resident page descriptors into caller-owned CUDA storage."""
    batch_size, num_kv_heads, max_selected = selected_indices.shape
    max_pages = output_page_ids.shape[3]
    if output_cluster_ids.shape != selected_indices.shape:
        raise ValueError("Cluster-ID output shape does not match selected clusters")
    if output_page_ids.shape != (
        batch_size,
        num_kv_heads,
        max_selected,
        max_pages,
    ):
        raise ValueError("Page-ID output shape does not match selected clusters")
    if output_page_token_counts.shape != output_page_ids.shape:
        raise ValueError("Page-count output shape does not match page IDs")
    if max_selected == 0:
        return
    if max_pages == 0:
        output_cluster_ids.fill_(-1)
        return

    _gather_resident_exact_pages_kernel[(batch_size, num_kv_heads, max_selected)](
        cluster_ids,
        cluster_page_offsets,
        page_ids,
        page_token_counts,
        request_slot_ids,
        selected_indices,
        selected_mask,
        output_cluster_ids,
        output_page_ids,
        output_page_token_counts,
        *cluster_ids.stride(),
        *cluster_page_offsets.stride(),
        *page_ids.stride(),
        *page_token_counts.stride(),
        request_slot_ids.stride(0),
        *selected_indices.stride(),
        *selected_mask.stride(),
        *output_cluster_ids.stride(),
        *output_page_ids.stride(),
        *output_page_token_counts.stride(),
        MAX_PAGES=max_pages,
    )
