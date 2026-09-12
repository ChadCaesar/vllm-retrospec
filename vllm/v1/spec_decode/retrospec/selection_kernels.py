# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _emit_primary_exact_token_plan_kernel(
    seq_lens,
    indexed_starts,
    indexed_ends,
    indexed_requests,
    output_indices,
    output_mask,
    seq_len_stride,
    indexed_start_stride,
    indexed_end_stride,
    indexed_request_stride,
    output_index_stride_0,
    output_index_stride_1,
    output_index_stride_2,
    output_mask_stride_0,
    output_mask_stride_1,
    output_mask_stride_2,
    num_outputs,
    NUM_KV_HEADS: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    MAX_NUM_TOKENS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    NUM_RECENT_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_valid = offsets < num_outputs

    rank = offsets % OUTPUT_WIDTH
    row = offsets // OUTPUT_WIDTH
    batch_idx = row // NUM_KV_HEADS
    kv_head_idx = row % NUM_KV_HEADS
    rank64 = rank.to(tl.int64)

    seq_len = tl.load(
        seq_lens + batch_idx * seq_len_stride, mask=output_valid, other=1
    ).to(tl.int64)
    seq_len = tl.minimum(tl.maximum(seq_len, 1), MAX_NUM_TOKENS)

    valid_blocks = (seq_len + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE
    recent_start_block = tl.maximum(valid_blocks - NUM_RECENT_BLOCKS, 0)
    recent_start = tl.minimum(recent_start_block * CACHE_BLOCK_SIZE, seq_len)
    sink_end = tl.minimum(CACHE_BLOCK_SIZE, seq_len)

    indexed = tl.load(
        indexed_requests + batch_idx * indexed_request_stride,
        mask=output_valid,
        other=0,
    ).to(tl.int1)
    indexed_start = tl.load(
        indexed_starts + batch_idx * indexed_start_stride,
        mask=output_valid,
        other=0,
    ).to(tl.int64)
    indexed_end = tl.load(
        indexed_ends + batch_idx * indexed_end_stride,
        mask=output_valid,
        other=0,
    ).to(tl.int64)
    indexed_start = tl.minimum(tl.maximum(indexed_start, 0), seq_len)
    indexed_end = tl.minimum(tl.maximum(indexed_end, indexed_start), seq_len)

    prefix_end = tl.where(indexed, tl.maximum(indexed_start, sink_end), seq_len)
    suffix_start = tl.where(indexed, tl.minimum(indexed_end, recent_start), seq_len)
    intervals_overlap = suffix_start <= prefix_end
    exact_count = tl.where(
        intervals_overlap, seq_len, prefix_end + seq_len - suffix_start
    )
    logical_token = tl.where(
        intervals_overlap | (rank64 < prefix_end),
        rank64,
        suffix_start + rank64 - prefix_end,
    )

    inside_logical_width = rank64 < MAX_NUM_TOKENS
    selected = output_valid & inside_logical_width & (rank64 < exact_count)
    output_index_offset = (
        batch_idx * output_index_stride_0
        + kv_head_idx * output_index_stride_1
        + rank * output_index_stride_2
    )
    output_mask_offset = (
        batch_idx * output_mask_stride_0
        + kv_head_idx * output_mask_stride_1
        + rank * output_mask_stride_2
    )
    invalid_index = tl.where(inside_logical_width, MAX_NUM_TOKENS - 1, 0)
    tl.store(
        output_indices + output_index_offset,
        tl.where(selected, logical_token, invalid_index),
        mask=output_valid,
    )
    tl.store(output_mask + output_mask_offset, selected, mask=output_valid)


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
def _emit_ranked_draft_plan_kernel(
    cluster_keys,
    cluster_values,
    cluster_ids,
    cluster_token_counts,
    cluster_offsets,
    request_slot_ids,
    active_mask,
    ranked_indices,
    candidate_counts,
    sparse_exact_cluster_indices,
    expanded_exact_cluster_indices,
    sparse_estimation_cluster_indices,
    expanded_estimation_cluster_indices,
    draft_exact_cluster_handles,
    draft_estimation_keys,
    draft_estimation_values,
    draft_estimation_token_counts,
    CLUSTER_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SPARSE_WIDTH: tl.constexpr,
    EXPANDED_WIDTH: tl.constexpr,
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

    prefix_valid = request_valid & (output_idx < expanded_retrieval_count)
    prefix_local_idx = tl.load(
        ranked_indices
        + batch_idx * RANKED_STRIDE_0
        + kv_head_idx * RANKED_STRIDE_1
        + output_idx * RANKED_STRIDE_2,
        mask=prefix_valid,
        other=-1,
    ).to(tl.int64)
    prefix_valid &= prefix_local_idx >= 0
    prefix_cluster_idx = request_cluster_offset + tl.maximum(prefix_local_idx, 0)
    prefix_storage_offset = kv_head_idx * CLUSTER_CAPACITY + prefix_cluster_idx
    prefix_token_count = tl.load(
        cluster_token_counts + prefix_storage_offset,
        mask=prefix_valid,
        other=0,
    ).to(tl.int32)
    prefix_cluster_handle = tl.load(
        cluster_ids + prefix_storage_offset, mask=prefix_valid, other=-1
    ).to(tl.int64)
    prefix_valid &= (prefix_token_count > 0) & (prefix_cluster_handle >= 0)

    if SPARSE_WIDTH > 0:
        retrieval_valid = prefix_valid & (output_idx < retrieval_count)
        retrieval_plan_offset = group_offset * SPARSE_WIDTH + output_idx
        tl.store(
            sparse_exact_cluster_indices + retrieval_plan_offset,
            tl.where(retrieval_valid, prefix_local_idx, -1),
            mask=output_idx < SPARSE_WIDTH,
        )
        tl.store(
            draft_exact_cluster_handles + retrieval_plan_offset,
            tl.where(retrieval_valid, prefix_cluster_handle, -1),
            mask=output_idx < SPARSE_WIDTH,
        )
        retrieval_source_offsets = prefix_storage_offset * HEAD_SIZE + head_offsets
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
            tl.where(retrieval_valid, prefix_token_count, 0),
            mask=output_idx < SPARSE_WIDTH,
        )

    if EXPANDED_WIDTH > 0:
        expanded_exact_plan_offset = group_offset * EXPANDED_WIDTH + output_idx
        tl.store(
            expanded_exact_cluster_indices + expanded_exact_plan_offset,
            tl.where(prefix_valid, prefix_local_idx, -1),
            mask=output_idx < EXPANDED_WIDTH,
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


def emit_primary_exact_token_plan(
    *,
    seq_lens: torch.Tensor,
    indexed_starts: torch.Tensor,
    indexed_ends: torch.Tensor,
    indexed_requests: torch.Tensor,
    num_kv_heads: int,
    max_num_tokens: int,
    block_size: int,
    num_recent_blocks: int,
    output_indices: torch.Tensor,
    output_mask: torch.Tensor,
) -> None:
    """Emit ordered native exact-token descriptors from interval bounds."""
    if seq_lens.device.type != "cuda":
        raise ValueError("Primary exact-token plan emission requires CUDA")
    if seq_lens.ndim != 1:
        raise ValueError("Sequence lengths must be one-dimensional")
    if any(
        tensor.shape != seq_lens.shape
        for tensor in (indexed_starts, indexed_ends, indexed_requests)
    ):
        raise ValueError("Indexed bounds must match sequence lengths")
    if output_indices.ndim != 3 or output_mask.ndim != 3:
        raise ValueError(
            "Primary exact-token outputs must have shape [batch, heads, width]"
        )

    batch_size = seq_lens.shape[0]
    output_width = output_indices.shape[2]
    expected_shape = (batch_size, num_kv_heads, output_width)
    if output_indices.shape != expected_shape:
        raise ValueError("Primary exact-token indices have the wrong shape")
    if output_mask.shape != expected_shape:
        raise ValueError("Primary exact-token mask has the wrong shape")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if max_num_tokens <= 0:
        raise ValueError("max_num_tokens must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_recent_blocks < 0:
        raise ValueError("num_recent_blocks must be non-negative")
    if output_indices.dtype != torch.int64:
        raise ValueError("Primary exact-token indices must use int64")
    if output_mask.dtype != torch.bool:
        raise ValueError("Primary exact-token mask must use bool")
    if indexed_requests.dtype != torch.bool:
        raise ValueError("Indexed request mask must use bool")

    integer_tensors = (seq_lens, indexed_starts, indexed_ends)
    if any(
        tensor.dtype not in (torch.int32, torch.int64) for tensor in integer_tensors
    ):
        raise ValueError(
            "Sequence lengths and indexed bounds must use integral tensors"
        )
    tensors = (
        seq_lens,
        indexed_starts,
        indexed_ends,
        indexed_requests,
        output_indices,
        output_mask,
    )
    if any(tensor.device != seq_lens.device for tensor in tensors):
        raise ValueError("Primary exact-token plan tensors must use one CUDA device")
    if output_width == 0 or batch_size == 0:
        return

    num_outputs = batch_size * num_kv_heads * output_width
    launch_block_size = 256
    _emit_primary_exact_token_plan_kernel[
        (triton.cdiv(num_outputs, launch_block_size),)
    ](
        seq_lens,
        indexed_starts,
        indexed_ends,
        indexed_requests,
        output_indices,
        output_mask,
        seq_lens.stride(0),
        indexed_starts.stride(0),
        indexed_ends.stride(0),
        indexed_requests.stride(0),
        output_indices.stride(0),
        output_indices.stride(1),
        output_indices.stride(2),
        output_mask.stride(0),
        output_mask.stride(1),
        output_mask.stride(2),
        num_outputs,
        NUM_KV_HEADS=num_kv_heads,
        OUTPUT_WIDTH=output_width,
        MAX_NUM_TOKENS=max_num_tokens,
        CACHE_BLOCK_SIZE=block_size,
        NUM_RECENT_BLOCKS=num_recent_blocks,
        BLOCK_SIZE=launch_block_size,
    )


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


def emit_ranked_draft_plan(
    *,
    ranked_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    cluster_keys: torch.Tensor,
    cluster_values: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    cluster_offsets: torch.Tensor,
    request_slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    retrieval_ratio: float,
    estimation_ratio: float,
    sparse_exact_width: int,
    sparse_exact_cluster_indices: torch.Tensor,
    expanded_exact_cluster_indices: torch.Tensor,
    sparse_estimation_cluster_indices: torch.Tensor,
    expanded_estimation_cluster_indices: torch.Tensor,
    draft_exact_cluster_handles: torch.Tensor,
    draft_estimation_keys: torch.Tensor,
    draft_estimation_values: torch.Tensor,
    draft_estimation_token_counts: torch.Tensor,
) -> None:
    """Emit exact descriptors, estimation journals, and DRAFT summaries."""
    if ranked_indices.device.type != "cuda":
        raise ValueError("Ranked estimation emission requires CUDA")
    if ranked_indices.ndim != 3:
        raise ValueError("Ranked indices must have shape [batch, heads, ranks]")
    if ranked_indices.dtype != torch.int64:
        raise ValueError("Ranked indices must use int64")

    batch_size, num_kv_heads, ranking_width = ranked_indices.shape
    expanded_exact_width = expanded_exact_cluster_indices.shape[2]
    estimation_width = sparse_estimation_cluster_indices.shape[2]
    draft_width = draft_estimation_token_counts.shape[2]
    head_size = cluster_keys.shape[2]

    if candidate_counts.shape != (batch_size, num_kv_heads):
        raise ValueError("Candidate counts do not match ranked indices")
    if candidate_counts.dtype != torch.int32:
        raise ValueError("Candidate counts must use int32")
    if request_slot_ids.shape != (batch_size,):
        raise ValueError("Request slots do not match ranked indices")
    if active_mask.shape != (batch_size,):
        raise ValueError("Active mask does not match ranked indices")
    if sparse_exact_cluster_indices.shape != (
        batch_size,
        num_kv_heads,
        sparse_exact_width,
    ):
        raise ValueError("Sparse exact journal has the wrong shape")
    if draft_exact_cluster_handles.shape != sparse_exact_cluster_indices.shape:
        raise ValueError("Draft exact handles have the wrong shape")
    if draft_width != estimation_width + sparse_exact_width:
        raise ValueError("Draft estimation workspace has an invalid width")
    if expanded_exact_cluster_indices.shape[:2] != (
        batch_size,
        num_kv_heads,
    ):
        raise ValueError("Expanded exact journal has the wrong row shape")
    if expanded_estimation_cluster_indices.shape != (
        batch_size,
        num_kv_heads,
        estimation_width,
    ):
        raise ValueError("Expanded estimation journal has the wrong shape")
    if draft_estimation_keys.shape != (
        batch_size,
        num_kv_heads,
        draft_width,
        head_size,
    ):
        raise ValueError("Draft estimation keys have the wrong shape")
    if draft_estimation_values.shape != draft_estimation_keys.shape:
        raise ValueError("Draft estimation values have the wrong shape")
    if draft_estimation_token_counts.shape != (
        batch_size,
        num_kv_heads,
        draft_width,
    ):
        raise ValueError("Draft estimation counts have the wrong shape")
    if ranking_width < max(expanded_exact_width, sparse_exact_width + estimation_width):
        raise ValueError("Ranked workspace is too narrow")

    tensors = (
        ranked_indices,
        candidate_counts,
        cluster_keys,
        cluster_values,
        cluster_ids,
        cluster_token_counts,
        cluster_offsets,
        request_slot_ids,
        active_mask,
        sparse_exact_cluster_indices,
        expanded_exact_cluster_indices,
        sparse_estimation_cluster_indices,
        expanded_estimation_cluster_indices,
        draft_exact_cluster_handles,
        draft_estimation_keys,
        draft_estimation_values,
        draft_estimation_token_counts,
    )
    if any(tensor.device != ranked_indices.device for tensor in tensors):
        raise ValueError("Ranked draft-plan tensors must use one device")

    output_width = max(sparse_exact_width, expanded_exact_width, estimation_width)
    if output_width == 0:
        return

    _emit_ranked_draft_plan_kernel[(batch_size, num_kv_heads, output_width)](
        cluster_keys,
        cluster_values,
        cluster_ids,
        cluster_token_counts,
        cluster_offsets,
        request_slot_ids,
        active_mask,
        ranked_indices,
        candidate_counts,
        sparse_exact_cluster_indices,
        expanded_exact_cluster_indices,
        sparse_estimation_cluster_indices,
        expanded_estimation_cluster_indices,
        draft_exact_cluster_handles,
        draft_estimation_keys,
        draft_estimation_values,
        draft_estimation_token_counts,
        CLUSTER_CAPACITY=cluster_keys.shape[1],
        NUM_KV_HEADS=num_kv_heads,
        SPARSE_WIDTH=sparse_exact_width,
        EXPANDED_WIDTH=expanded_exact_width,
        ESTIMATION_WIDTH=estimation_width,
        HEAD_SIZE=head_size,
        BLOCK_D=triton.next_power_of_2(head_size),
        RANKED_STRIDE_0=ranked_indices.stride(0),
        RANKED_STRIDE_1=ranked_indices.stride(1),
        RANKED_STRIDE_2=ranked_indices.stride(2),
        RETRIEVAL_RATIO=retrieval_ratio,
        ESTIMATION_RATIO=estimation_ratio,
    )


@triton.jit
def _pack_indexed_verification_plan_kernel(
    request_indices,
    token_indices,
    valid_rows,
    request_slot_ids,
    request_slot_generations,
    exact_cluster_indices,
    estimation_cluster_indices,
    attention_mass,
    output_plan_row_indices,
    output_plan_valid_rows,
    output_request_slot_ids,
    output_request_slot_generations,
    output_exact_cluster_indices,
    output_estimation_cluster_indices,
    output_estimation_cluster_mask,
    output_attention_mass,
    valid_row_stride_0,
    valid_row_stride_1,
    exact_stride_0,
    exact_stride_1,
    exact_stride_2,
    estimation_stride_0,
    estimation_stride_1,
    estimation_stride_2,
    output_exact_stride_0,
    output_exact_stride_1,
    output_exact_stride_2,
    output_estimation_stride_0,
    output_estimation_stride_1,
    output_estimation_stride_2,
    NUM_STEPS: tl.constexpr,
    BATCH_CAPACITY: tl.constexpr,
    EXACT_WIDTH: tl.constexpr,
    ESTIMATION_WIDTH: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    ranks = tl.arange(0, BLOCK_RANK)

    request_idx = tl.load(request_indices + pair_idx).to(tl.int64)
    token_idx = tl.load(token_indices + pair_idx).to(tl.int64)
    indices_in_bounds = (
        (request_idx >= 0)
        & (request_idx < BATCH_CAPACITY)
        & (token_idx >= 0)
        & (token_idx < NUM_STEPS)
    )
    safe_request_idx = tl.where(indices_in_bounds, request_idx, 0)
    safe_token_idx = tl.where(indices_in_bounds, token_idx, 0)
    plan_row = safe_token_idx * BATCH_CAPACITY + safe_request_idx
    row_valid = tl.load(
        valid_rows
        + safe_token_idx * valid_row_stride_0
        + safe_request_idx * valid_row_stride_1,
        mask=indices_in_bounds,
        other=0,
    ).to(tl.int1)
    plan_valid = indices_in_bounds & row_valid

    request_slot = tl.load(
        request_slot_ids + safe_request_idx, mask=plan_valid, other=-1
    ).to(tl.int64)
    request_generation = tl.load(
        request_slot_generations + safe_request_idx, mask=plan_valid, other=-1
    ).to(tl.int64)
    packed_attention = tl.load(
        attention_mass + plan_row, mask=plan_valid, other=1.0
    ).to(tl.float32)
    scalar_writer = kv_head_idx == 0
    tl.store(output_plan_row_indices + pair_idx, plan_row, mask=scalar_writer)
    tl.store(output_plan_valid_rows + pair_idx, plan_valid, mask=scalar_writer)
    tl.store(output_request_slot_ids + pair_idx, request_slot, mask=scalar_writer)
    tl.store(
        output_request_slot_generations + pair_idx,
        request_generation,
        mask=scalar_writer,
    )
    tl.store(output_attention_mass + pair_idx, packed_attention, mask=scalar_writer)

    exact_rank_valid = ranks < EXACT_WIDTH
    exact_source_offsets = (
        plan_row * exact_stride_0
        + kv_head_idx * exact_stride_1
        + ranks * exact_stride_2
    )
    exact_indices = tl.load(
        exact_cluster_indices + exact_source_offsets,
        mask=plan_valid & exact_rank_valid,
        other=-1,
    )
    exact_output_offsets = (
        pair_idx * output_exact_stride_0
        + kv_head_idx * output_exact_stride_1
        + ranks * output_exact_stride_2
    )
    tl.store(
        output_exact_cluster_indices + exact_output_offsets,
        exact_indices,
        mask=exact_rank_valid,
    )

    estimation_rank_valid = ranks < ESTIMATION_WIDTH
    estimation_source_offsets = (
        plan_row * estimation_stride_0
        + kv_head_idx * estimation_stride_1
        + ranks * estimation_stride_2
    )
    estimation_indices = tl.load(
        estimation_cluster_indices + estimation_source_offsets,
        mask=plan_valid & estimation_rank_valid,
        other=-1,
    )
    estimation_output_offsets = (
        pair_idx * output_estimation_stride_0
        + kv_head_idx * output_estimation_stride_1
        + ranks * output_estimation_stride_2
    )
    tl.store(
        output_estimation_cluster_indices + estimation_output_offsets,
        estimation_indices,
        mask=estimation_rank_valid,
    )
    tl.store(
        output_estimation_cluster_mask + estimation_output_offsets,
        plan_valid & (estimation_indices >= 0),
        mask=estimation_rank_valid,
    )


def pack_indexed_verification_plan(
    *,
    request_indices: torch.Tensor,
    token_indices: torch.Tensor,
    valid_rows: torch.Tensor,
    request_slot_ids: torch.Tensor,
    request_slot_generations: torch.Tensor,
    exact_cluster_indices: torch.Tensor,
    estimation_cluster_indices: torch.Tensor,
    attention_mass: torch.Tensor,
    output_plan_row_indices: torch.Tensor,
    output_plan_valid_rows: torch.Tensor,
    output_request_slot_ids: torch.Tensor,
    output_request_slot_generations: torch.Tensor,
    output_exact_cluster_indices: torch.Tensor,
    output_estimation_cluster_indices: torch.Tensor,
    output_estimation_cluster_mask: torch.Tensor,
    output_attention_mass: torch.Tensor,
) -> None:
    """Pack persistent verification-plan rows into query-row descriptors."""
    if request_indices.ndim != 1 or token_indices.ndim != 1:
        raise ValueError("Indexed verification inputs must be one-dimensional")
    if request_indices.shape != token_indices.shape:
        raise ValueError("Indexed verification inputs must have equal shapes")
    if valid_rows.ndim != 2:
        raise ValueError("Plan validity must have shape [steps, batch]")
    if request_slot_ids.shape != request_slot_generations.shape:
        raise ValueError("Request slot descriptors must have equal shapes")
    if request_slot_ids.shape != (valid_rows.shape[1],):
        raise ValueError("Request slot descriptors do not match plan capacity")
    if exact_cluster_indices.ndim != 3 or estimation_cluster_indices.ndim != 3:
        raise ValueError("Cluster plans must have shape [rows, heads, width]")

    num_pairs = request_indices.numel()
    num_plan_rows = valid_rows.numel()
    num_kv_heads = exact_cluster_indices.shape[1]
    if exact_cluster_indices.shape[0] != num_plan_rows:
        raise ValueError("Exact cluster plan has the wrong row capacity")
    if estimation_cluster_indices.shape[:2] != (num_plan_rows, num_kv_heads):
        raise ValueError("Estimation cluster plan has the wrong row shape")
    if attention_mass.shape != (num_plan_rows,):
        raise ValueError("Attention plan has the wrong row capacity")
    if output_plan_row_indices.shape != (num_pairs,):
        raise ValueError("Packed plan rows have the wrong shape")
    if output_plan_valid_rows.shape != (num_pairs,):
        raise ValueError("Packed plan validity has the wrong shape")
    if output_request_slot_ids.shape != (num_pairs,):
        raise ValueError("Packed request slots have the wrong shape")
    if output_request_slot_generations.shape != (num_pairs,):
        raise ValueError("Packed request generations have the wrong shape")
    if output_exact_cluster_indices.shape != (
        num_pairs,
        num_kv_heads,
        exact_cluster_indices.shape[2],
    ):
        raise ValueError("Packed exact clusters have the wrong shape")
    expected_estimation_shape = (
        num_pairs,
        num_kv_heads,
        estimation_cluster_indices.shape[2],
    )
    if output_estimation_cluster_indices.shape != expected_estimation_shape:
        raise ValueError("Packed estimation clusters have the wrong shape")
    if output_estimation_cluster_mask.shape != expected_estimation_shape:
        raise ValueError("Packed estimation mask has the wrong shape")
    if output_attention_mass.shape != (num_pairs,):
        raise ValueError("Packed attention mass has the wrong shape")

    integer_inputs = (
        request_indices,
        token_indices,
        request_slot_ids,
        request_slot_generations,
        exact_cluster_indices,
        estimation_cluster_indices,
    )
    if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in integer_inputs):
        raise ValueError("Indexed verification descriptors must be integral")
    if valid_rows.dtype != torch.bool or output_plan_valid_rows.dtype != torch.bool:
        raise ValueError("Plan validity tensors must be boolean")
    if output_estimation_cluster_mask.dtype != torch.bool:
        raise ValueError("Packed estimation mask must be boolean")
    outputs = (
        output_plan_row_indices,
        output_plan_valid_rows,
        output_request_slot_ids,
        output_request_slot_generations,
        output_exact_cluster_indices,
        output_estimation_cluster_indices,
        output_estimation_cluster_mask,
        output_attention_mass,
    )
    tensors = (*integer_inputs, valid_rows, attention_mass, *outputs)
    if any(tensor.device != request_indices.device for tensor in tensors):
        raise ValueError("Indexed verification tensors must use one device")
    if num_pairs == 0:
        return

    if request_indices.device.type != "cuda":
        request_indices_i64 = request_indices.to(torch.int64)
        token_indices_i64 = token_indices.to(torch.int64)
        in_bounds = (
            (request_indices_i64 >= 0)
            & (request_indices_i64 < valid_rows.shape[1])
            & (token_indices_i64 >= 0)
            & (token_indices_i64 < valid_rows.shape[0])
        )
        safe_requests = request_indices_i64.masked_fill(~in_bounds, 0)
        safe_tokens = token_indices_i64.masked_fill(~in_bounds, 0)
        plan_rows = safe_tokens * valid_rows.shape[1] + safe_requests
        plan_valid = in_bounds & valid_rows.flatten().index_select(0, plan_rows)
        output_plan_row_indices.copy_(plan_rows)
        output_plan_valid_rows.copy_(plan_valid)
        output_request_slot_ids.copy_(
            request_slot_ids.index_select(0, safe_requests).masked_fill(~plan_valid, -1)
        )
        output_request_slot_generations.copy_(
            request_slot_generations.index_select(0, safe_requests).masked_fill(
                ~plan_valid, -1
            )
        )
        exact = exact_cluster_indices.index_select(0, plan_rows)
        estimation = estimation_cluster_indices.index_select(0, plan_rows)
        output_exact_cluster_indices.copy_(
            exact.masked_fill(~plan_valid[:, None, None], -1)
        )
        output_estimation_cluster_indices.copy_(
            estimation.masked_fill(~plan_valid[:, None, None], -1)
        )
        output_estimation_cluster_mask.copy_(
            plan_valid[:, None, None] & (estimation >= 0)
        )
        output_attention_mass.copy_(
            attention_mass.index_select(0, plan_rows).masked_fill(~plan_valid, 1.0)
        )
        return

    block_rank = triton.next_power_of_2(
        max(exact_cluster_indices.shape[2], estimation_cluster_indices.shape[2], 1)
    )
    _pack_indexed_verification_plan_kernel[(num_pairs, num_kv_heads)](
        request_indices,
        token_indices,
        valid_rows,
        request_slot_ids,
        request_slot_generations,
        exact_cluster_indices,
        estimation_cluster_indices,
        attention_mass,
        output_plan_row_indices,
        output_plan_valid_rows,
        output_request_slot_ids,
        output_request_slot_generations,
        output_exact_cluster_indices,
        output_estimation_cluster_indices,
        output_estimation_cluster_mask,
        output_attention_mass,
        valid_rows.stride(0),
        valid_rows.stride(1),
        exact_cluster_indices.stride(0),
        exact_cluster_indices.stride(1),
        exact_cluster_indices.stride(2),
        estimation_cluster_indices.stride(0),
        estimation_cluster_indices.stride(1),
        estimation_cluster_indices.stride(2),
        output_exact_cluster_indices.stride(0),
        output_exact_cluster_indices.stride(1),
        output_exact_cluster_indices.stride(2),
        output_estimation_cluster_indices.stride(0),
        output_estimation_cluster_indices.stride(1),
        output_estimation_cluster_indices.stride(2),
        NUM_STEPS=valid_rows.shape[0],
        BATCH_CAPACITY=valid_rows.shape[1],
        EXACT_WIDTH=exact_cluster_indices.shape[2],
        ESTIMATION_WIDTH=estimation_cluster_indices.shape[2],
        BLOCK_RANK=block_rank,
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
