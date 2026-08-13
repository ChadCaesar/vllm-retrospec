# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _merge_weighted_estimation_kernel(
    output,
    query,
    estimation_keys,
    estimation_values,
    estimation_token_counts,
    exact_output,
    exact_lse,
    scale,
    num_estimation_vectors,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    query_stride_0,
    query_stride_1,
    query_stride_2,
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
    exact_output_stride_0,
    exact_output_stride_1,
    exact_output_stride_2,
    exact_lse_stride_0,
    exact_lse_stride_1,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    kv_head_idx = query_head_idx // QUERIES_PER_KV

    head_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < HEAD_SIZE

    query_offsets = (
        batch_idx * query_stride_0
        + query_head_idx * query_stride_1
        + head_offsets * query_stride_2
    )
    query_vector = tl.load(
        query + query_offsets,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)

    exact_output_offsets = (
        batch_idx * exact_output_stride_0
        + query_head_idx * exact_output_stride_1
        + head_offsets * exact_output_stride_2
    )
    exact_vector = tl.load(
        exact_output + exact_output_offsets,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)

    # vLLM attention LSE tensors use the logical layout
    # [num_query_heads, batch_size].
    exact_lse_value = tl.load(
        exact_lse + query_head_idx * exact_lse_stride_0 + batch_idx * exact_lse_stride_1
    ).to(tl.float32)

    # Treat both +inf and -inf as an empty exact attention state.
    has_exact = (exact_lse_value > float("-inf")) & (exact_lse_value < float("inf"))

    running_max = tl.where(
        has_exact,
        exact_lse_value,
        float("-inf"),
    )
    running_sum = tl.where(has_exact, 1.0, 0.0)
    accumulator = tl.where(
        has_exact & head_mask,
        exact_vector,
        0.0,
    )

    for vector_start in tl.range(
        0,
        num_estimation_vectors,
        BLOCK_M,
    ):
        vector_offsets = vector_start + tl.arange(0, BLOCK_M)
        vector_mask = vector_offsets < num_estimation_vectors

        count_offsets = (
            batch_idx * count_stride_0
            + kv_head_idx * count_stride_1
            + vector_offsets * count_stride_2
        )
        token_counts = tl.load(
            estimation_token_counts + count_offsets,
            mask=vector_mask,
            other=0,
        )

        valid_vectors = vector_mask & (token_counts > 0)

        key_offsets = (
            batch_idx * key_stride_0
            + kv_head_idx * key_stride_1
            + vector_offsets[:, None] * key_stride_2
            + head_offsets[None, :] * key_stride_3
        )
        key_vectors = tl.load(
            estimation_keys + key_offsets,
            mask=valid_vectors[:, None] & head_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        logits = tl.sum(
            key_vectors * query_vector[None, :],
            axis=1,
        )
        logits *= scale

        # A centroid represents token_counts equivalent KV tokens:
        #
        #   token_counts * exp(q · centroid)
        #     = exp(q · centroid + log(token_counts))
        logits += tl.log(token_counts.to(tl.float32))
        logits = tl.where(
            valid_vectors,
            logits,
            float("-inf"),
        )

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, block_max)

        old_scale = tl.where(
            running_sum > 0,
            tl.exp(running_max - new_max),
            0.0,
        )
        probabilities = tl.where(
            valid_vectors,
            tl.exp(logits - new_max),
            0.0,
        )

        value_offsets = (
            batch_idx * value_stride_0
            + kv_head_idx * value_stride_1
            + vector_offsets[:, None] * value_stride_2
            + head_offsets[None, :] * value_stride_3
        )
        value_vectors = tl.load(
            estimation_values + value_offsets,
            mask=valid_vectors[:, None] & head_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        accumulator *= old_scale
        accumulator += tl.sum(
            probabilities[:, None] * value_vectors,
            axis=0,
        )

        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max

    has_attention = running_sum > 0
    safe_running_sum = tl.maximum(running_sum, 1.0)
    merged_output = tl.where(
        has_attention,
        accumulator / safe_running_sum,
        0.0,
    )

    output_offsets = (
        batch_idx * output_stride_0
        + query_head_idx * output_stride_1
        + head_offsets * output_stride_2
    )
    tl.store(
        output + output_offsets,
        merged_output,
        mask=head_mask,
    )


def merge_weighted_estimation(
    output: torch.Tensor,
    query: torch.Tensor,
    estimation_keys: torch.Tensor,
    estimation_values: torch.Tensor,
    estimation_token_counts: torch.Tensor,
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
    scale: float,
) -> None:
    """Merge weighted centroid attention into an exact attention state.

    Shapes:
        output/query/exact_output:
            [batch, num_query_heads, head_size]
        estimation_keys/estimation_values:
            [batch, num_kv_heads, num_estimation_vectors, head_size]
        estimation_token_counts:
            [batch, num_kv_heads, num_estimation_vectors]
        exact_lse:
            [num_query_heads, batch]
    """
    if query.device.type != "cuda":
        raise ValueError("Weighted estimation attention requires CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Weighted estimation attention requires fp16 or bf16")
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, num_query_heads, head_size]")
    if output.shape != query.shape:
        raise ValueError("output shape must match query")
    if exact_output.shape != query.shape:
        raise ValueError("exact output shape must match query")
    if estimation_keys.shape != estimation_values.shape:
        raise ValueError("Estimation key and value shapes must match")
    if estimation_keys.ndim != 4:
        raise ValueError(
            "Estimation KV must have shape [batch, num_kv_heads, vectors, head_size]"
        )
    if estimation_token_counts.shape != estimation_keys.shape[:3]:
        raise ValueError("Estimation token counts do not match estimation KV")

    batch_size, num_query_heads, head_size = query.shape
    key_batch_size, num_kv_heads, num_vectors, key_head_size = estimation_keys.shape

    if key_batch_size != batch_size:
        raise ValueError("Estimation KV batch size does not match query")
    if key_head_size != head_size:
        raise ValueError("Estimation KV head size does not match query")
    if num_kv_heads <= 0:
        raise ValueError("Estimation KV must contain at least one KV head")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "The number of query heads must be divisible by the number of KV heads"
        )
    if exact_lse.shape != (num_query_heads, batch_size):
        raise ValueError("Exact LSE must have shape [num_query_heads, batch]")

    tensors = (
        output,
        estimation_keys,
        estimation_values,
        estimation_token_counts,
        exact_output,
        exact_lse,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("All weighted attention tensors must be on one device")
    if output.dtype != query.dtype or exact_output.dtype != query.dtype:
        raise ValueError("Query, exact output and destination must have the same dtype")
    if estimation_keys.dtype != estimation_values.dtype:
        raise ValueError("Estimation keys and values must have the same dtype")
    if estimation_keys.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("Estimation KV must use a floating-point dtype")
    if estimation_token_counts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Estimation token counts must be integral")
    if exact_lse.dtype != torch.float32:
        raise ValueError("Exact LSE must use float32")

    if batch_size == 0:
        return

    if num_vectors == 0:
        output.copy_(exact_output)
        return

    queries_per_kv = num_query_heads // num_kv_heads
    block_d = triton.next_power_of_2(head_size)

    # Keep the tile small enough that both K and V tiles fit comfortably
    # alongside the fp32 online-softmax accumulator.
    block_m = 16

    _merge_weighted_estimation_kernel[(batch_size, num_query_heads)](
        output,
        query,
        estimation_keys,
        estimation_values,
        estimation_token_counts,
        exact_output,
        exact_lse,
        scale,
        num_vectors,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        estimation_keys.stride(0),
        estimation_keys.stride(1),
        estimation_keys.stride(2),
        estimation_keys.stride(3),
        estimation_values.stride(0),
        estimation_values.stride(1),
        estimation_values.stride(2),
        estimation_values.stride(3),
        estimation_token_counts.stride(0),
        estimation_token_counts.stride(1),
        estimation_token_counts.stride(2),
        exact_output.stride(0),
        exact_output.stride(1),
        exact_output.stride(2),
        exact_lse.stride(0),
        exact_lse.stride(1),
        QUERIES_PER_KV=queries_per_kv,
        HEAD_SIZE=head_size,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
