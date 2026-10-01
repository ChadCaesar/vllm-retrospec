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


@triton.jit
def _pack_ranked_verification_plan_kernel(
    request_indices,
    token_indices,
    valid_rows,
    request_slot_ids,
    request_slot_generations,
    ranked_cluster_indices,
    candidate_counts,
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
    ranked_stride_0,
    ranked_stride_1,
    ranked_stride_2,
    candidate_stride_0,
    candidate_stride_1,
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
    EXPANDED: tl.constexpr,
    RETRIEVAL_RATIO: tl.constexpr,
    ESTIMATION_RATIO: tl.constexpr,
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

    candidate_count = tl.load(
        candidate_counts
        + plan_row * candidate_stride_0
        + kv_head_idx * candidate_stride_1,
        mask=plan_valid,
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
    exact_count = retrieval_count
    if EXPANDED:
        exact_count = tl.minimum(retrieval_count * 2, total_compute_count)
    selected_estimation_count = total_compute_count - exact_count

    ranked_row_offset = plan_row * ranked_stride_0 + kv_head_idx * ranked_stride_1
    exact_rank_valid = plan_valid & (ranks < EXACT_WIDTH) & (ranks < exact_count)
    exact_indices = tl.load(
        ranked_cluster_indices + ranked_row_offset + ranks * ranked_stride_2,
        mask=exact_rank_valid,
        other=-1,
    )
    exact_output_offsets = (
        pair_idx * output_exact_stride_0
        + kv_head_idx * output_exact_stride_1
        + ranks * output_exact_stride_2
    )
    tl.store(
        output_exact_cluster_indices + exact_output_offsets,
        tl.where(exact_rank_valid, exact_indices, -1),
        mask=ranks < EXACT_WIDTH,
    )

    estimation_rank_valid = (
        plan_valid & (ranks < ESTIMATION_WIDTH) & (ranks < selected_estimation_count)
    )
    estimation_ranks = exact_count + ranks
    estimation_indices = tl.load(
        ranked_cluster_indices + ranked_row_offset + estimation_ranks * ranked_stride_2,
        mask=estimation_rank_valid,
        other=-1,
    )
    estimation_output_offsets = (
        pair_idx * output_estimation_stride_0
        + kv_head_idx * output_estimation_stride_1
        + ranks * output_estimation_stride_2
    )
    tl.store(
        output_estimation_cluster_indices + estimation_output_offsets,
        tl.where(estimation_rank_valid, estimation_indices, -1),
        mask=ranks < ESTIMATION_WIDTH,
    )
    tl.store(
        output_estimation_cluster_mask + estimation_output_offsets,
        estimation_rank_valid & (estimation_indices >= 0),
        mask=ranks < ESTIMATION_WIDTH,
    )


def pack_ranked_verification_plan(
    *,
    request_indices: torch.Tensor,
    token_indices: torch.Tensor,
    valid_rows: torch.Tensor,
    request_slot_ids: torch.Tensor,
    request_slot_generations: torch.Tensor,
    ranked_cluster_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    attention_mass: torch.Tensor,
    output_plan_row_indices: torch.Tensor,
    output_plan_valid_rows: torch.Tensor,
    output_request_slot_ids: torch.Tensor,
    output_request_slot_generations: torch.Tensor,
    output_exact_cluster_indices: torch.Tensor,
    output_estimation_cluster_indices: torch.Tensor,
    output_estimation_cluster_mask: torch.Tensor,
    output_attention_mass: torch.Tensor,
    retrieval_ratio: float,
    estimation_ratio: float,
    expanded: bool,
) -> None:
    """Expand ranked journal rows into packed verification descriptors."""
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
    if ranked_cluster_indices.ndim != 3:
        raise ValueError("Ranked journal must have shape [rows, heads, ranks]")
    if candidate_counts.shape != ranked_cluster_indices.shape[:2]:
        raise ValueError("Candidate counts do not match ranked journal rows")
    if not 0.0 < retrieval_ratio <= 1.0:
        raise ValueError("Retrieval ratio must be in (0, 1]")
    if not 0.0 <= estimation_ratio <= 1.0:
        raise ValueError("Estimation ratio must be in [0, 1]")

    num_pairs = request_indices.numel()
    num_plan_rows = valid_rows.numel()
    num_kv_heads = ranked_cluster_indices.shape[1]
    ranking_width = ranked_cluster_indices.shape[2]
    if ranked_cluster_indices.shape[0] != num_plan_rows:
        raise ValueError("Ranked journal has the wrong row capacity")
    if ranking_width <= 0:
        raise ValueError("Ranked journal width must be positive")
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
    if output_exact_cluster_indices.ndim != 3 or output_exact_cluster_indices.shape[
        :2
    ] != (num_pairs, num_kv_heads):
        raise ValueError("Packed exact clusters have the wrong shape")
    if output_exact_cluster_indices.shape[2] > ranking_width:
        raise ValueError("Packed exact clusters exceed the ranked journal")
    if output_estimation_cluster_indices.ndim != 3:
        raise ValueError("Packed estimation clusters have the wrong shape")
    expected_estimation_shape = (
        num_pairs,
        num_kv_heads,
        output_estimation_cluster_indices.shape[2],
    )
    if output_estimation_cluster_indices.shape != expected_estimation_shape:
        raise ValueError("Packed estimation clusters have the wrong shape")
    if output_estimation_cluster_mask.shape != expected_estimation_shape:
        raise ValueError("Packed estimation mask has the wrong shape")
    if output_estimation_cluster_indices.shape[2] > ranking_width:
        raise ValueError("Packed estimation clusters exceed the ranked journal")
    if output_attention_mass.shape != (num_pairs,):
        raise ValueError("Packed attention mass has the wrong shape")

    integer_inputs = (
        request_indices,
        token_indices,
        request_slot_ids,
        request_slot_generations,
        ranked_cluster_indices,
        candidate_counts,
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
        ranked = ranked_cluster_indices.index_select(0, plan_rows)
        candidates = candidate_counts.index_select(0, plan_rows)
        retrieval_counts = torch.ceil(candidates * retrieval_ratio).to(torch.int64)
        retrieval_counts = torch.minimum(retrieval_counts, candidates.to(torch.int64))
        estimation_counts = torch.ceil(candidates * estimation_ratio).to(torch.int64)
        estimation_counts = torch.minimum(
            estimation_counts, candidates.to(torch.int64) - retrieval_counts
        )
        total_counts = retrieval_counts + estimation_counts
        exact_counts = retrieval_counts
        if expanded:
            exact_counts = torch.minimum(retrieval_counts * 2, total_counts)
        selected_estimation_counts = total_counts - exact_counts

        exact_ranks = torch.arange(
            output_exact_cluster_indices.shape[2], device=request_indices.device
        )
        exact_mask = plan_valid[:, None, None] & (
            exact_ranks < exact_counts.unsqueeze(-1)
        )
        safe_exact_ranks = exact_ranks.expand_as(output_exact_cluster_indices)
        output_exact_cluster_indices.copy_(
            ranked.gather(2, safe_exact_ranks).masked_fill(~exact_mask, -1)
        )

        estimation_ranks = torch.arange(
            output_estimation_cluster_indices.shape[2],
            device=request_indices.device,
        )
        estimation_mask = plan_valid[:, None, None] & (
            estimation_ranks < selected_estimation_counts.unsqueeze(-1)
        )
        ranked_estimation_ranks = exact_counts.unsqueeze(-1) + estimation_ranks
        safe_estimation_ranks = ranked_estimation_ranks.clamp_max(ranked.shape[2] - 1)
        estimation = ranked.gather(2, safe_estimation_ranks)
        output_estimation_cluster_indices.copy_(
            estimation.masked_fill(~estimation_mask, -1)
        )
        output_estimation_cluster_mask.copy_(estimation_mask & (estimation >= 0))
        output_attention_mass.copy_(
            attention_mass.index_select(0, plan_rows).masked_fill(~plan_valid, 1.0)
        )
        return

    block_rank = triton.next_power_of_2(
        max(
            output_exact_cluster_indices.shape[2],
            output_estimation_cluster_indices.shape[2],
            1,
        )
    )
    _pack_ranked_verification_plan_kernel[(num_pairs, num_kv_heads)](
        request_indices,
        token_indices,
        valid_rows,
        request_slot_ids,
        request_slot_generations,
        ranked_cluster_indices,
        candidate_counts,
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
        ranked_cluster_indices.stride(0),
        ranked_cluster_indices.stride(1),
        ranked_cluster_indices.stride(2),
        candidate_counts.stride(0),
        candidate_counts.stride(1),
        output_exact_cluster_indices.stride(0),
        output_exact_cluster_indices.stride(1),
        output_exact_cluster_indices.stride(2),
        output_estimation_cluster_indices.stride(0),
        output_estimation_cluster_indices.stride(1),
        output_estimation_cluster_indices.stride(2),
        NUM_STEPS=valid_rows.shape[0],
        BATCH_CAPACITY=valid_rows.shape[1],
        EXACT_WIDTH=output_exact_cluster_indices.shape[2],
        ESTIMATION_WIDTH=output_estimation_cluster_indices.shape[2],
        BLOCK_RANK=block_rank,
        EXPANDED=expanded,
        RETRIEVAL_RATIO=retrieval_ratio,
        ESTIMATION_RATIO=estimation_ratio,
    )


def pack_ranked_verification_exact_plan(
    *,
    request_indices: torch.Tensor,
    token_indices: torch.Tensor,
    valid_rows: torch.Tensor,
    request_slot_ids: torch.Tensor,
    request_slot_generations: torch.Tensor,
    ranked_cluster_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    attention_mass: torch.Tensor,
    output_plan_row_indices: torch.Tensor,
    output_plan_valid_rows: torch.Tensor,
    output_request_slot_ids: torch.Tensor,
    output_request_slot_generations: torch.Tensor,
    output_exact_cluster_indices: torch.Tensor,
    output_attention_mass: torch.Tensor,
    empty_estimation_cluster_indices: torch.Tensor,
    empty_estimation_cluster_mask: torch.Tensor,
    retrieval_ratio: float,
    estimation_ratio: float,
    expanded: bool,
) -> None:
    """Pack only the exact portion needed by cross-layer page resolution."""
    expected_empty_shape = (
        request_indices.numel(),
        ranked_cluster_indices.shape[1],
        0,
    )
    if empty_estimation_cluster_indices.shape != expected_empty_shape:
        raise ValueError("Empty estimation indices have the wrong shape")
    if empty_estimation_cluster_mask.shape != expected_empty_shape:
        raise ValueError("Empty estimation mask has the wrong shape")
    pack_ranked_verification_plan(
        request_indices=request_indices,
        token_indices=token_indices,
        valid_rows=valid_rows,
        request_slot_ids=request_slot_ids,
        request_slot_generations=request_slot_generations,
        ranked_cluster_indices=ranked_cluster_indices,
        candidate_counts=candidate_counts,
        attention_mass=attention_mass,
        output_plan_row_indices=output_plan_row_indices,
        output_plan_valid_rows=output_plan_valid_rows,
        output_request_slot_ids=output_request_slot_ids,
        output_request_slot_generations=output_request_slot_generations,
        output_exact_cluster_indices=output_exact_cluster_indices,
        output_estimation_cluster_indices=empty_estimation_cluster_indices,
        output_estimation_cluster_mask=empty_estimation_cluster_mask,
        output_attention_mass=output_attention_mass,
        retrieval_ratio=retrieval_ratio,
        estimation_ratio=estimation_ratio,
        expanded=expanded,
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
