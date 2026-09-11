# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _emit_ranked_cluster_plan_kernel(
    cluster_ids,
    cluster_page_starts,
    cluster_page_counts,
    page_ids,
    page_token_counts,
    cluster_offsets,
    page_offsets,
    request_slot_ids,
    active_mask,
    ranked_indices,
    candidate_counts,
    sparse_cluster_indices,
    expanded_cluster_indices,
    draft_cluster_ids,
    draft_page_ids,
    draft_page_token_counts,
    CLUSTER_CAPACITY: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    RANKING_WIDTH: tl.constexpr,
    SPARSE_WIDTH: tl.constexpr,
    EXPANDED_WIDTH: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    RANKED_STRIDE_0: tl.constexpr,
    RANKED_STRIDE_1: tl.constexpr,
    RANKED_STRIDE_2: tl.constexpr,
    RETRIEVAL_RATIO: tl.constexpr,
    ESTIMATION_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    rank_idx = tl.program_id(2)

    group_offset = batch_idx * NUM_KV_HEADS + kv_head_idx
    candidate_count = tl.load(candidate_counts + group_offset).to(tl.int32)
    retrieval_count = tl.ceil(candidate_count.to(tl.float32) * RETRIEVAL_RATIO).to(
        tl.int32
    )
    retrieval_count = tl.minimum(retrieval_count, candidate_count)
    estimation_count = tl.ceil(candidate_count.to(tl.float32) * ESTIMATION_RATIO).to(
        tl.int32
    )
    estimation_count = tl.minimum(estimation_count, candidate_count - retrieval_count)
    total_compute_count = retrieval_count + estimation_count
    expanded_retrieval_count = tl.minimum(retrieval_count * 2, total_compute_count)

    request_slot = tl.load(request_slot_ids + batch_idx)
    request_active = tl.load(active_mask + batch_idx).to(tl.int1)
    request_valid = request_active & (request_slot >= 0)
    safe_slot = tl.maximum(request_slot, 0).to(tl.int64)
    request_cluster_offset = tl.load(
        cluster_offsets + safe_slot, mask=request_valid, other=0
    ).to(tl.int64)
    request_page_offset = tl.load(
        page_offsets + safe_slot, mask=request_valid, other=0
    ).to(tl.int64)

    ranked_offset = (
        batch_idx * RANKED_STRIDE_0
        + kv_head_idx * RANKED_STRIDE_1
        + rank_idx * RANKED_STRIDE_2
    )
    local_cluster_idx = tl.load(
        ranked_indices + ranked_offset,
        mask=rank_idx < RANKING_WIDTH,
        other=0,
    ).to(tl.int64)
    absolute_cluster_idx = request_cluster_offset + tl.maximum(local_cluster_idx, 0)
    arena_cluster_offset = kv_head_idx * CLUSTER_CAPACITY + absolute_cluster_idx
    cluster_id = tl.load(
        cluster_ids + arena_cluster_offset, mask=request_valid, other=-1
    ).to(tl.int64)
    valid_cluster = request_valid & (cluster_id >= 0)
    page_start = tl.load(
        cluster_page_starts + arena_cluster_offset,
        mask=valid_cluster,
        other=0,
    ).to(tl.int64)
    num_pages = tl.load(
        cluster_page_counts + arena_cluster_offset,
        mask=valid_cluster,
        other=0,
    ).to(tl.int32)

    sparse_valid = valid_cluster & (rank_idx < retrieval_count)
    expanded_valid = valid_cluster & (rank_idx < expanded_retrieval_count)

    if SPARSE_WIDTH > 0:
        sparse_output_valid = rank_idx < SPARSE_WIDTH
        sparse_cluster_offset = group_offset * SPARSE_WIDTH + rank_idx
        tl.store(
            sparse_cluster_indices + sparse_cluster_offset,
            tl.where(sparse_valid, local_cluster_idx, -1),
            mask=sparse_output_valid,
        )
        tl.store(
            draft_cluster_ids + sparse_cluster_offset,
            tl.where(sparse_valid, cluster_id, -1),
            mask=sparse_output_valid,
        )

    if EXPANDED_WIDTH > 0:
        expanded_output_valid = rank_idx < EXPANDED_WIDTH
        expanded_cluster_offset = group_offset * EXPANDED_WIDTH + rank_idx
        tl.store(
            expanded_cluster_indices + expanded_cluster_offset,
            tl.where(expanded_valid, local_cluster_idx, -1),
            mask=expanded_output_valid,
        )

    for page_idx in tl.static_range(0, MAX_PAGES):
        source_page_idx = request_page_offset + page_start + page_idx
        source_page_offset = kv_head_idx * PAGE_CAPACITY + source_page_idx

        sparse_page_valid = sparse_valid & (page_idx < num_pages)
        page_id = tl.load(
            page_ids + source_page_offset,
            mask=sparse_page_valid,
            other=-1,
        ).to(tl.int64)
        page_token_count = tl.load(
            page_token_counts + source_page_offset,
            mask=sparse_page_valid,
            other=0,
        ).to(tl.int32)

        if SPARSE_WIDTH > 0:
            sparse_output_valid = rank_idx < SPARSE_WIDTH
            sparse_page_offset = (
                group_offset * SPARSE_WIDTH + rank_idx
            ) * MAX_PAGES + page_idx
            tl.store(
                draft_page_ids + sparse_page_offset,
                tl.where(sparse_page_valid, page_id, -1),
                mask=sparse_output_valid,
            )
            tl.store(
                draft_page_token_counts + sparse_page_offset,
                tl.where(sparse_page_valid, page_token_count, 0),
                mask=sparse_output_valid,
            )


@triton.jit
def _capture_request_descriptors_kernel(
    request_slot_ids,
    arena_generations,
    output_slot_ids,
    output_generations,
    num_requests,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_requests
    slots = tl.load(request_slot_ids + offsets, mask=valid, other=-1)
    valid_slot = valid & (slots >= 0)
    generations = tl.load(
        arena_generations + tl.maximum(slots, 0), mask=valid_slot, other=-1
    )
    tl.store(output_slot_ids + offsets, slots, mask=valid)
    tl.store(
        output_generations + offsets,
        tl.where(valid_slot, generations, -1),
        mask=valid,
    )


@triton.jit
def _emit_ranked_estimation_plan_kernel(
    cluster_keys,
    cluster_values,
    cluster_token_counts,
    cluster_offsets,
    request_slot_ids,
    active_mask,
    ranked_indices,
    candidate_counts,
    sparse_estimation_cluster_indices,
    expanded_estimation_cluster_indices,
    draft_estimation_keys,
    draft_estimation_values,
    draft_estimation_token_counts,
    CLUSTER_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    RANKING_WIDTH: tl.constexpr,
    SPARSE_WIDTH: tl.constexpr,
    ESTIMATION_WIDTH: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RANKED_STRIDE_0: tl.constexpr,
    RANKED_STRIDE_1: tl.constexpr,
    RANKED_STRIDE_2: tl.constexpr,
    RETRIEVAL_RATIO: tl.constexpr,
    ESTIMATION_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    output_idx = tl.program_id(2)

    group_offset = batch_idx * NUM_KV_HEADS + kv_head_idx
    candidate_count = tl.load(candidate_counts + group_offset).to(tl.int32)
    retrieval_count = tl.ceil(candidate_count.to(tl.float32) * RETRIEVAL_RATIO).to(
        tl.int32
    )
    retrieval_count = tl.minimum(retrieval_count, candidate_count)
    estimation_count = tl.ceil(candidate_count.to(tl.float32) * ESTIMATION_RATIO).to(
        tl.int32
    )
    estimation_count = tl.minimum(estimation_count, candidate_count - retrieval_count)
    total_compute_count = retrieval_count + estimation_count
    expanded_retrieval_count = tl.minimum(retrieval_count * 2, total_compute_count)

    request_slot = tl.load(request_slot_ids + batch_idx)
    request_active = tl.load(active_mask + batch_idx).to(tl.int1)
    request_valid = request_active & (request_slot >= 0)
    safe_slot = tl.maximum(request_slot, 0).to(tl.int64)
    request_cluster_offset = tl.load(
        cluster_offsets + safe_slot, mask=request_valid, other=0
    ).to(tl.int64)

    head_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < HEAD_SIZE
    draft_width = ESTIMATION_WIDTH + SPARSE_WIDTH

    if SPARSE_WIDTH > 0:
        retrieval_rank = output_idx
        retrieval_valid = request_valid & (output_idx < retrieval_count)
        retrieval_local_idx = tl.load(
            ranked_indices
            + batch_idx * RANKED_STRIDE_0
            + kv_head_idx * RANKED_STRIDE_1
            + retrieval_rank * RANKED_STRIDE_2,
            mask=retrieval_valid,
            other=0,
        ).to(tl.int64)
        retrieval_cluster_idx = request_cluster_offset + tl.maximum(
            retrieval_local_idx, 0
        )
        retrieval_count_offset = kv_head_idx * CLUSTER_CAPACITY + retrieval_cluster_idx
        retrieval_token_count = tl.load(
            cluster_token_counts + retrieval_count_offset,
            mask=retrieval_valid,
            other=0,
        ).to(tl.int32)
        retrieval_valid &= retrieval_token_count > 0
        retrieval_source_offsets = retrieval_count_offset * HEAD_SIZE + head_offsets
        retrieval_output_idx = ESTIMATION_WIDTH + output_idx
        retrieval_output_offsets = (
            group_offset * draft_width + retrieval_output_idx
        ) * HEAD_SIZE + head_offsets
        retrieval_copy_mask = retrieval_valid & head_mask
        retrieval_keys = tl.load(
            cluster_keys + retrieval_source_offsets,
            mask=retrieval_copy_mask,
            other=0.0,
        )
        retrieval_values = tl.load(
            cluster_values + retrieval_source_offsets,
            mask=retrieval_copy_mask,
            other=0.0,
        )
        tl.store(
            draft_estimation_keys + retrieval_output_offsets,
            retrieval_keys,
            mask=(output_idx < SPARSE_WIDTH) & head_mask,
        )
        tl.store(
            draft_estimation_values + retrieval_output_offsets,
            retrieval_values,
            mask=(output_idx < SPARSE_WIDTH) & head_mask,
        )
        tl.store(
            draft_estimation_token_counts
            + group_offset * draft_width
            + retrieval_output_idx,
            tl.where(retrieval_valid, retrieval_token_count, 0),
            mask=output_idx < SPARSE_WIDTH,
        )

    if ESTIMATION_WIDTH > 0:
        sparse_rank = retrieval_count + output_idx
        sparse_valid = request_valid & (output_idx < estimation_count)
        sparse_local_idx = tl.load(
            ranked_indices
            + batch_idx * RANKED_STRIDE_0
            + kv_head_idx * RANKED_STRIDE_1
            + sparse_rank * RANKED_STRIDE_2,
            mask=sparse_valid,
            other=0,
        ).to(tl.int64)
        sparse_cluster_idx = request_cluster_offset + tl.maximum(sparse_local_idx, 0)
        sparse_count_offset = kv_head_idx * CLUSTER_CAPACITY + sparse_cluster_idx
        sparse_token_count = tl.load(
            cluster_token_counts + sparse_count_offset,
            mask=sparse_valid,
            other=0,
        ).to(tl.int32)
        sparse_valid &= sparse_token_count > 0
        sparse_plan_offset = group_offset * ESTIMATION_WIDTH + output_idx
        tl.store(
            sparse_estimation_cluster_indices + sparse_plan_offset,
            tl.where(sparse_valid, sparse_local_idx, -1),
            mask=output_idx < ESTIMATION_WIDTH,
        )
        sparse_source_offsets = sparse_count_offset * HEAD_SIZE + head_offsets
        sparse_output_offsets = (
            group_offset * draft_width + output_idx
        ) * HEAD_SIZE + head_offsets
        sparse_copy_mask = sparse_valid & head_mask
        sparse_keys = tl.load(
            cluster_keys + sparse_source_offsets,
            mask=sparse_copy_mask,
            other=0.0,
        )
        sparse_values = tl.load(
            cluster_values + sparse_source_offsets,
            mask=sparse_copy_mask,
            other=0.0,
        )
        tl.store(
            draft_estimation_keys + sparse_output_offsets,
            sparse_keys,
            mask=(output_idx < ESTIMATION_WIDTH) & head_mask,
        )
        tl.store(
            draft_estimation_values + sparse_output_offsets,
            sparse_values,
            mask=(output_idx < ESTIMATION_WIDTH) & head_mask,
        )
        tl.store(
            draft_estimation_token_counts + group_offset * draft_width + output_idx,
            tl.where(sparse_valid, sparse_token_count, 0),
            mask=output_idx < ESTIMATION_WIDTH,
        )

        expanded_rank = expanded_retrieval_count + output_idx
        expanded_count = total_compute_count - expanded_retrieval_count
        expanded_valid = request_valid & (output_idx < expanded_count)
        expanded_local_idx = tl.load(
            ranked_indices
            + batch_idx * RANKED_STRIDE_0
            + kv_head_idx * RANKED_STRIDE_1
            + expanded_rank * RANKED_STRIDE_2,
            mask=expanded_valid,
            other=0,
        ).to(tl.int64)
        expanded_cluster_idx = request_cluster_offset + tl.maximum(
            expanded_local_idx, 0
        )
        expanded_count_offset = kv_head_idx * CLUSTER_CAPACITY + expanded_cluster_idx
        expanded_token_count = tl.load(
            cluster_token_counts + expanded_count_offset,
            mask=expanded_valid,
            other=0,
        ).to(tl.int32)
        expanded_valid &= expanded_token_count > 0
        expanded_plan_offset = group_offset * ESTIMATION_WIDTH + output_idx
        tl.store(
            expanded_estimation_cluster_indices + expanded_plan_offset,
            tl.where(expanded_valid, expanded_local_idx, -1),
            mask=output_idx < ESTIMATION_WIDTH,
        )


@triton.jit
def _ranked_attention_mass_kernel(
    ranked_values,
    candidate_counts,
    request_slot_ids,
    active_mask,
    sparse_attn,
    expanded_attn,
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

    candidate_count = tl.load(
        candidate_counts + batch_idx * NUM_KV_HEADS + head_offsets,
        mask=valid_heads,
        other=0,
    ).to(tl.int32)
    retrieval_count = tl.ceil(candidate_count.to(tl.float32) * RETRIEVAL_RATIO).to(
        tl.int32
    )
    retrieval_count = tl.minimum(retrieval_count, candidate_count)
    estimation_count = tl.ceil(candidate_count.to(tl.float32) * ESTIMATION_RATIO).to(
        tl.int32
    )
    estimation_count = tl.minimum(estimation_count, candidate_count - retrieval_count)
    total_compute_count = retrieval_count + estimation_count
    expanded_retrieval_count = tl.minimum(retrieval_count * 2, total_compute_count)

    score_offsets = (
        batch_idx * RANKED_STRIDE_0
        + head_offsets[:, None] * RANKED_STRIDE_1
        + rank_offsets[None, :] * RANKED_STRIDE_2
    )
    score_mask = valid_heads[:, None] & (rank_offsets[None, :] < RANKING_WIDTH)
    scores = tl.load(ranked_values + score_offsets, mask=score_mask, other=0.0)

    sparse_scores = tl.where(
        rank_offsets[None, :] < retrieval_count[:, None], scores, 0.0
    )
    expanded_scores = tl.where(
        rank_offsets[None, :] < expanded_retrieval_count[:, None], scores, 0.0
    )
    sparse_by_head = tl.sum(sparse_scores, axis=1)
    expanded_by_head = tl.sum(expanded_scores, axis=1)
    sparse_by_head = tl.where(candidate_count > 0, sparse_by_head, 1.0)
    expanded_by_head = tl.where(candidate_count > 0, expanded_by_head, 1.0)
    sparse_by_head = tl.where(valid_heads, sparse_by_head, 0.0)
    expanded_by_head = tl.where(valid_heads, expanded_by_head, 0.0)

    sparse_value = tl.sum(sparse_by_head, axis=0) / NUM_KV_HEADS
    expanded_value = tl.sum(expanded_by_head, axis=0) / NUM_KV_HEADS
    request_active = tl.load(active_mask + batch_idx).to(tl.int1)
    request_slot = tl.load(request_slot_ids + batch_idx)
    request_valid = request_active & (request_slot >= 0)
    tl.store(sparse_attn + batch_idx, tl.where(request_valid, sparse_value, 1.0))
    tl.store(expanded_attn + batch_idx, tl.where(request_valid, expanded_value, 1.0))


def capture_request_descriptors(
    request_slot_ids: torch.Tensor,
    arena_generations: torch.Tensor,
    output_slot_ids: torch.Tensor,
    output_generations: torch.Tensor,
) -> None:
    if request_slot_ids.device.type != "cuda":
        raise ValueError("Request descriptor capture requires CUDA")
    if request_slot_ids.ndim != 1:
        raise ValueError("Request slots must be one-dimensional")
    if output_slot_ids.shape != request_slot_ids.shape:
        raise ValueError("Captured request slots have the wrong shape")
    if output_generations.shape != request_slot_ids.shape:
        raise ValueError("Captured request generations have the wrong shape")
    tensors = (
        request_slot_ids,
        arena_generations,
        output_slot_ids,
        output_generations,
    )
    if any(tensor.device != request_slot_ids.device for tensor in tensors):
        raise ValueError("Request descriptor tensors must use one CUDA device")
    if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in tensors):
        raise ValueError("Request descriptors must use integral tensors")

    block_size = 256
    _capture_request_descriptors_kernel[
        (triton.cdiv(request_slot_ids.numel(), block_size),)
    ](
        request_slot_ids,
        arena_generations,
        output_slot_ids,
        output_generations,
        request_slot_ids.numel(),
        BLOCK_SIZE=block_size,
    )


def emit_ranked_selection_plan(
    *,
    ranked_values: torch.Tensor,
    ranked_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    cluster_keys: torch.Tensor,
    cluster_values: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_page_starts: torch.Tensor,
    cluster_page_counts: torch.Tensor,
    page_ids: torch.Tensor,
    page_token_counts: torch.Tensor,
    cluster_offsets: torch.Tensor,
    page_offsets: torch.Tensor,
    request_slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    retrieval_ratio: float,
    estimation_ratio: float,
    sparse_exact_cluster_indices: torch.Tensor,
    sparse_estimation_cluster_indices: torch.Tensor,
    expanded_exact_cluster_indices: torch.Tensor,
    expanded_estimation_cluster_indices: torch.Tensor,
    draft_exact_cluster_ids: torch.Tensor,
    draft_exact_page_ids: torch.Tensor,
    draft_exact_page_token_counts: torch.Tensor,
    draft_estimation_keys: torch.Tensor,
    draft_estimation_values: torch.Tensor,
    draft_estimation_token_counts: torch.Tensor,
    sparse_attn: torch.Tensor,
    expanded_attn: torch.Tensor,
) -> None:
    """Emit exact ranked cluster zones directly into persistent plan storage."""
    if ranked_values.device.type != "cuda":
        raise ValueError("Rank-driven plan emission requires CUDA")
    if ranked_values.shape != ranked_indices.shape:
        raise ValueError("Ranked values and indices must have equal shapes")
    if candidate_counts.shape != ranked_values.shape[:2]:
        raise ValueError("Candidate counts do not match ranked clusters")

    batch_size, num_kv_heads, ranking_width = ranked_indices.shape
    if active_mask.shape != (batch_size,):
        raise ValueError("Active mask does not match ranked clusters")
    if request_slot_ids.shape != (batch_size,):
        raise ValueError("Request slots do not match ranked clusters")
    if ranked_indices.dtype != torch.int64:
        raise ValueError("Ranked cluster indices must use int64")
    if candidate_counts.dtype != torch.int32:
        raise ValueError("Candidate counts must use int32")

    sparse_width = sparse_exact_cluster_indices.shape[2]
    expanded_width = expanded_exact_cluster_indices.shape[2]
    estimation_width = sparse_estimation_cluster_indices.shape[2]
    draft_width = draft_estimation_token_counts.shape[2]
    max_pages = draft_exact_page_ids.shape[3]
    if draft_width != estimation_width + sparse_width:
        raise ValueError("Draft estimation workspace has an invalid width")
    expected_estimation_shape = (batch_size, num_kv_heads, estimation_width)
    if sparse_estimation_cluster_indices.shape != expected_estimation_shape:
        raise ValueError("Sparse estimation journal has the wrong shape")
    if expanded_estimation_cluster_indices.shape != expected_estimation_shape:
        raise ValueError("Expanded estimation journal has the wrong shape")
    if draft_exact_cluster_ids.shape != sparse_exact_cluster_indices.shape:
        raise ValueError("DRAFT cluster workspace has the wrong shape")
    if draft_exact_page_token_counts.shape != draft_exact_page_ids.shape:
        raise ValueError("DRAFT page workspace has inconsistent shapes")
    if draft_exact_page_ids.shape[:3] != sparse_exact_cluster_indices.shape:
        raise ValueError("DRAFT page workspace does not match sparse clusters")
    if ranking_width < max(expanded_width, sparse_width + estimation_width):
        raise ValueError("Ranked workspace is too narrow for the selection plan")

    tensors = (
        ranked_values,
        ranked_indices,
        candidate_counts,
        cluster_keys,
        cluster_values,
        cluster_token_counts,
        cluster_ids,
        cluster_page_starts,
        cluster_page_counts,
        page_ids,
        page_token_counts,
        cluster_offsets,
        page_offsets,
        request_slot_ids,
        active_mask,
        sparse_exact_cluster_indices,
        sparse_estimation_cluster_indices,
        expanded_exact_cluster_indices,
        expanded_estimation_cluster_indices,
        draft_exact_cluster_ids,
        draft_exact_page_ids,
        draft_exact_page_token_counts,
        draft_estimation_keys,
        draft_estimation_values,
        draft_estimation_token_counts,
        sparse_attn,
        expanded_attn,
    )
    if any(tensor.device != ranked_values.device for tensor in tensors):
        raise ValueError("Ranked plan tensors must use one CUDA device")
    if any(not tensor.is_contiguous() for tensor in tensors[2:]):
        raise ValueError("Ranked plan tensors must be contiguous")

    exact_width = max(sparse_width, expanded_width)
    if exact_width > 0:
        _emit_ranked_cluster_plan_kernel[(batch_size, num_kv_heads, exact_width)](
            cluster_ids,
            cluster_page_starts,
            cluster_page_counts,
            page_ids,
            page_token_counts,
            cluster_offsets,
            page_offsets,
            request_slot_ids,
            active_mask,
            ranked_indices,
            candidate_counts,
            sparse_exact_cluster_indices,
            expanded_exact_cluster_indices,
            draft_exact_cluster_ids,
            draft_exact_page_ids,
            draft_exact_page_token_counts,
            CLUSTER_CAPACITY=cluster_ids.shape[1],
            PAGE_CAPACITY=page_ids.shape[1],
            NUM_KV_HEADS=num_kv_heads,
            RANKING_WIDTH=ranking_width,
            SPARSE_WIDTH=sparse_width,
            EXPANDED_WIDTH=expanded_width,
            MAX_PAGES=max_pages,
            RANKED_STRIDE_0=ranked_indices.stride(0),
            RANKED_STRIDE_1=ranked_indices.stride(1),
            RANKED_STRIDE_2=ranked_indices.stride(2),
            RETRIEVAL_RATIO=retrieval_ratio,
            ESTIMATION_RATIO=estimation_ratio,
        )

    summary_width = max(sparse_width, estimation_width)
    if summary_width > 0:
        _emit_ranked_estimation_plan_kernel[(batch_size, num_kv_heads, summary_width)](
            cluster_keys,
            cluster_values,
            cluster_token_counts,
            cluster_offsets,
            request_slot_ids,
            active_mask,
            ranked_indices,
            candidate_counts,
            sparse_estimation_cluster_indices,
            expanded_estimation_cluster_indices,
            draft_estimation_keys,
            draft_estimation_values,
            draft_estimation_token_counts,
            CLUSTER_CAPACITY=cluster_keys.shape[1],
            NUM_KV_HEADS=num_kv_heads,
            RANKING_WIDTH=ranking_width,
            SPARSE_WIDTH=sparse_width,
            ESTIMATION_WIDTH=estimation_width,
            HEAD_SIZE=cluster_keys.shape[2],
            BLOCK_D=triton.next_power_of_2(cluster_keys.shape[2]),
            RANKED_STRIDE_0=ranked_indices.stride(0),
            RANKED_STRIDE_1=ranked_indices.stride(1),
            RANKED_STRIDE_2=ranked_indices.stride(2),
            RETRIEVAL_RATIO=retrieval_ratio,
            ESTIMATION_RATIO=estimation_ratio,
        )

    _ranked_attention_mass_kernel[(batch_size,)](
        ranked_values,
        candidate_counts,
        request_slot_ids,
        active_mask,
        sparse_attn,
        expanded_attn,
        NUM_KV_HEADS=num_kv_heads,
        RANKING_WIDTH=ranking_width,
        BLOCK_HEADS=triton.next_power_of_2(num_kv_heads),
        BLOCK_RANK=triton.next_power_of_2(max(expanded_width, 1)),
        RANKED_STRIDE_0=ranked_values.stride(0),
        RANKED_STRIDE_1=ranked_values.stride(1),
        RANKED_STRIDE_2=ranked_values.stride(2),
        RETRIEVAL_RATIO=retrieval_ratio,
        ESTIMATION_RATIO=estimation_ratio,
    )


@triton.jit
def _add_indexed_values_kernel(
    destination,
    source,
    row_indices,
    num_rows,
    destination_stride,
    source_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_rows
    source_rows = tl.load(row_indices + offsets, mask=valid, other=0)
    values = tl.load(source + source_rows * source_stride, mask=valid, other=0.0)
    current = tl.load(destination + offsets * destination_stride, mask=valid, other=0.0)
    tl.store(destination + offsets * destination_stride, current + values, mask=valid)


def add_indexed_values(
    destination: torch.Tensor,
    source: torch.Tensor,
    row_indices: torch.Tensor,
) -> None:
    if destination.device.type != "cuda":
        raise ValueError("Indexed accumulation requires CUDA tensors")
    if destination.ndim != 1 or source.ndim != 1 or row_indices.ndim != 1:
        raise ValueError("Indexed accumulation expects one-dimensional tensors")
    if destination.shape != row_indices.shape:
        raise ValueError("Destination and indexed rows must have equal shapes")
    if row_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Indexed rows must be integral")
    if destination.dtype != source.dtype:
        raise ValueError("Indexed accumulation dtypes must match")
    if any(tensor.device != destination.device for tensor in (source, row_indices)):
        raise ValueError("Indexed accumulation tensors must use one device")
    if destination.numel() == 0:
        return

    block_size = 256
    _add_indexed_values_kernel[(triton.cdiv(destination.numel(), block_size),)](
        destination,
        source,
        row_indices,
        destination.numel(),
        destination.stride(0),
        source.stride(0),
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _gather_resident_estimation_kernel(
    cluster_keys,
    cluster_values,
    cluster_token_counts,
    cluster_offsets,
    request_slot_ids,
    selected_indices,
    selected_mask,
    output_keys,
    output_values,
    output_token_counts,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    count_stride_0,
    count_stride_1,
    cluster_offset_stride_0,
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
    request_cluster_offset = tl.load(
        cluster_offsets + safe_slot * cluster_offset_stride_0,
        mask=valid,
        other=0,
    ).to(tl.int64)
    safe_cluster = request_cluster_offset + tl.maximum(cluster_idx, 0).to(tl.int64)
    count_offset = kv_head_idx * count_stride_0 + safe_cluster * count_stride_1
    token_count = tl.load(cluster_token_counts + count_offset, mask=valid, other=0)
    valid &= token_count > 0

    head_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < HEAD_SIZE
    source_key_offsets = (
        kv_head_idx * key_stride_0
        + safe_cluster * key_stride_1
        + head_offsets * key_stride_2
    )
    source_value_offsets = (
        kv_head_idx * value_stride_0
        + safe_cluster * value_stride_1
        + head_offsets * value_stride_2
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
    cluster_page_starts,
    cluster_page_counts,
    page_ids,
    page_token_counts,
    cluster_offsets,
    page_offsets,
    request_slot_ids,
    selected_indices,
    selected_mask,
    output_cluster_ids,
    output_page_ids,
    output_page_token_counts,
    cluster_stride_0,
    cluster_stride_1,
    page_start_stride_0,
    page_start_stride_1,
    cluster_page_count_stride_0,
    cluster_page_count_stride_1,
    page_stride_0,
    page_stride_1,
    page_count_stride_0,
    page_count_stride_1,
    cluster_offset_stride_0,
    page_offset_stride_0,
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
    request_cluster_offset = tl.load(
        cluster_offsets + safe_slot * cluster_offset_stride_0,
        mask=valid_cluster,
        other=0,
    ).to(tl.int64)
    request_page_offset = tl.load(
        page_offsets + safe_slot * page_offset_stride_0,
        mask=valid_cluster,
        other=0,
    ).to(tl.int64)
    absolute_cluster = request_cluster_offset + safe_cluster

    cluster_offset = (
        kv_head_idx * cluster_stride_0 + absolute_cluster * cluster_stride_1
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

    page_start = tl.load(
        cluster_page_starts
        + kv_head_idx * page_start_stride_0
        + absolute_cluster * page_start_stride_1,
        mask=valid_cluster,
        other=0,
    )
    cluster_num_pages = tl.load(
        cluster_page_counts
        + kv_head_idx * cluster_page_count_stride_0
        + absolute_cluster * cluster_page_count_stride_1,
        mask=valid_cluster,
        other=0,
    )

    for page_offset in tl.static_range(0, MAX_PAGES):
        source_page_idx = request_page_offset + page_start + page_offset
        valid_page = valid_cluster & (page_offset < cluster_num_pages)
        source_page_offset = (
            kv_head_idx * page_stride_0 + source_page_idx * page_stride_1
        )
        source_page_count_offset = (
            kv_head_idx * page_count_stride_0 + source_page_idx * page_count_stride_1
        )
        page_id = tl.load(page_ids + source_page_offset, mask=valid_page, other=-1)
        page_token_count = tl.load(
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
        tl.store(output_page_token_counts + output_page_count_offset, page_token_count)


def gather_resident_estimation(
    cluster_keys: torch.Tensor,
    cluster_values: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    cluster_offsets: torch.Tensor,
    request_slot_ids: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_mask: torch.Tensor,
    output_keys: torch.Tensor,
    output_values: torch.Tensor,
    output_token_counts: torch.Tensor,
) -> None:
    """Gather resident cluster summaries into caller-owned CUDA storage."""
    batch_size, num_kv_heads, max_selected = selected_indices.shape
    head_size = cluster_keys.shape[2]
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
        cluster_offsets,
        request_slot_ids,
        selected_indices,
        selected_mask,
        output_keys,
        output_values,
        output_token_counts,
        *cluster_keys.stride(),
        *cluster_values.stride(),
        *cluster_token_counts.stride(),
        cluster_offsets.stride(0),
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
    cluster_page_starts: torch.Tensor,
    cluster_page_counts: torch.Tensor,
    page_ids: torch.Tensor,
    page_token_counts: torch.Tensor,
    cluster_offsets: torch.Tensor,
    page_offsets: torch.Tensor,
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
        cluster_page_starts,
        cluster_page_counts,
        page_ids,
        page_token_counts,
        cluster_offsets,
        page_offsets,
        request_slot_ids,
        selected_indices,
        selected_mask,
        output_cluster_ids,
        output_page_ids,
        output_page_token_counts,
        *cluster_ids.stride(),
        *cluster_page_starts.stride(),
        *cluster_page_counts.stride(),
        *page_ids.stride(),
        *page_token_counts.stride(),
        cluster_offsets.stride(0),
        page_offsets.stride(0),
        request_slot_ids.stride(0),
        *selected_indices.stride(),
        *selected_mask.stride(),
        *output_cluster_ids.stride(),
        *output_page_ids.stride(),
        *output_page_token_counts.stride(),
        MAX_PAGES=max_pages,
    )
