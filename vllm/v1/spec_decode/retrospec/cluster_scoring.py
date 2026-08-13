# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _cluster_softmax_lse_kernel(
    logits,
    cluster_mask,
    cluster_token_counts,
    softmax_lse,
    scale,
    num_clusters,
    logits_stride_0,
    logits_stride_1,
    logits_stride_2,
    logits_stride_3,
    mask_stride_0,
    mask_stride_1,
    mask_stride_2,
    count_stride_0,
    count_stride_1,
    count_stride_2,
    lse_stride_0,
    lse_stride_1,
    lse_stride_2,
    BLOCK_C: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    query_group_idx = tl.program_id(2)

    running_max = float("-inf")
    running_sum = 0.0

    for cluster_start in tl.range(0, num_clusters, BLOCK_C):
        cluster_offsets = cluster_start + tl.arange(0, BLOCK_C)
        cluster_in_bounds = cluster_offsets < num_clusters

        mask_offsets = (
            batch_idx * mask_stride_0
            + kv_head_idx * mask_stride_1
            + cluster_offsets * mask_stride_2
        )
        count_offsets = (
            batch_idx * count_stride_0
            + kv_head_idx * count_stride_1
            + cluster_offsets * count_stride_2
        )

        candidate_mask = tl.load(
            cluster_mask + mask_offsets,
            mask=cluster_in_bounds,
            other=0,
        ).to(tl.int1)
        token_counts = tl.load(
            cluster_token_counts + count_offsets,
            mask=cluster_in_bounds,
            other=0,
        )

        valid_clusters = cluster_in_bounds & candidate_mask & (token_counts > 0)

        logit_offsets = (
            batch_idx * logits_stride_0
            + kv_head_idx * logits_stride_1
            + query_group_idx * logits_stride_2
            + cluster_offsets * logits_stride_3
        )
        block_logits = tl.load(
            logits + logit_offsets,
            mask=valid_clusters,
            other=0.0,
        ).to(tl.float32)

        safe_token_counts = tl.maximum(
            token_counts.to(tl.float32),
            1.0,
        )
        block_logits = block_logits * scale + tl.log(safe_token_counts)
        block_logits = tl.where(
            valid_clusters,
            block_logits,
            float("-inf"),
        )

        block_max = tl.max(block_logits, axis=0)
        next_max = tl.maximum(running_max, block_max)

        previous_scale = tl.where(
            running_sum > 0,
            tl.exp(running_max - next_max),
            0.0,
        )
        block_probabilities = tl.where(
            valid_clusters,
            tl.exp(block_logits - next_max),
            0.0,
        )

        running_sum = running_sum * previous_scale + tl.sum(block_probabilities, axis=0)
        running_max = next_max

    has_clusters = running_sum > 0
    row_lse = tl.where(
        has_clusters,
        running_max + tl.log(running_sum),
        0.0,
    )

    lse_offset = (
        batch_idx * lse_stride_0
        + kv_head_idx * lse_stride_1
        + query_group_idx * lse_stride_2
    )
    tl.store(softmax_lse + lse_offset, row_lse)


@triton.jit
def _reduce_grouped_cluster_scores_kernel(
    logits,
    cluster_mask,
    cluster_token_counts,
    softmax_lse,
    output,
    scale,
    num_clusters,
    logits_stride_0,
    logits_stride_1,
    logits_stride_2,
    logits_stride_3,
    mask_stride_0,
    mask_stride_1,
    mask_stride_2,
    count_stride_0,
    count_stride_1,
    count_stride_2,
    lse_stride_0,
    lse_stride_1,
    lse_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    QUERIES_PER_KV: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    cluster_block_idx = tl.program_id(2)

    cluster_offsets = cluster_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    cluster_in_bounds = cluster_offsets < num_clusters

    mask_offsets = (
        batch_idx * mask_stride_0
        + kv_head_idx * mask_stride_1
        + cluster_offsets * mask_stride_2
    )
    count_offsets = (
        batch_idx * count_stride_0
        + kv_head_idx * count_stride_1
        + cluster_offsets * count_stride_2
    )

    candidate_mask = tl.load(
        cluster_mask + mask_offsets,
        mask=cluster_in_bounds,
        other=0,
    ).to(tl.int1)
    token_counts = tl.load(
        cluster_token_counts + count_offsets,
        mask=cluster_in_bounds,
        other=0,
    )

    valid_clusters = cluster_in_bounds & candidate_mask & (token_counts > 0)

    safe_token_counts = tl.maximum(
        token_counts.to(tl.float32),
        1.0,
    )
    log_token_counts = tl.log(safe_token_counts)

    score_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for query_group_idx in tl.static_range(0, QUERIES_PER_KV):
        logit_offsets = (
            batch_idx * logits_stride_0
            + kv_head_idx * logits_stride_1
            + query_group_idx * logits_stride_2
            + cluster_offsets * logits_stride_3
        )
        query_logits = tl.load(
            logits + logit_offsets,
            mask=valid_clusters,
            other=0.0,
        ).to(tl.float32)

        lse_offset = (
            batch_idx * lse_stride_0
            + kv_head_idx * lse_stride_1
            + query_group_idx * lse_stride_2
        )
        row_lse = tl.load(softmax_lse + lse_offset)

        weighted_logits = query_logits * scale + log_token_counts
        probabilities = tl.where(
            valid_clusters,
            tl.exp(weighted_logits - row_lse),
            0.0,
        )
        score_sum += probabilities

    cluster_scores = score_sum / QUERIES_PER_KV
    cluster_scores = tl.where(
        valid_clusters,
        cluster_scores,
        0.0,
    )

    output_offsets = (
        batch_idx * output_stride_0
        + kv_head_idx * output_stride_1
        + cluster_offsets * output_stride_2
    )
    tl.store(
        output + output_offsets,
        cluster_scores,
        mask=cluster_in_bounds,
    )


def reduce_grouped_cluster_scores(
    logits: torch.Tensor,
    cluster_mask: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Normalize grouped cluster logits and average GQA probabilities.

    Args:
        logits:
            Raw query-centroid dot products with shape
            [batch, num_kv_heads, queries_per_kv, num_clusters].
        cluster_mask:
            Valid cluster mask with shape
            [batch, num_kv_heads, num_clusters].
        cluster_token_counts:
            Number of source tokens represented by every centroid, with the
            same shape as cluster_mask.
        scale:
            Query-key attention scale.

    Returns:
        Float32 cluster probabilities with shape
        [batch, num_kv_heads, num_clusters].
    """
    if logits.device.type != "cuda":
        raise ValueError("Fused cluster scoring requires CUDA tensors")
    if logits.dtype != torch.float32:
        raise ValueError("Fused cluster logits must use float32")
    if logits.ndim != 4:
        raise ValueError(
            "logits must have shape [batch, num_kv_heads, queries_per_kv, num_clusters]"
        )
    if cluster_mask.dtype != torch.bool:
        raise ValueError("cluster_mask must use boolean dtype")
    if cluster_token_counts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("cluster_token_counts must use an integral dtype")

    batch_size, num_kv_heads, queries_per_kv, num_clusters = logits.shape
    expected_metadata_shape = (
        batch_size,
        num_kv_heads,
        num_clusters,
    )

    if cluster_mask.shape != expected_metadata_shape:
        raise ValueError("cluster_mask shape does not match cluster logits")
    if cluster_token_counts.shape != expected_metadata_shape:
        raise ValueError("cluster_token_counts shape does not match cluster logits")
    if queries_per_kv <= 0:
        raise ValueError("queries_per_kv must be positive")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if (
        cluster_mask.device != logits.device
        or cluster_token_counts.device != logits.device
    ):
        raise ValueError("Cluster scoring tensors must be on one CUDA device")

    output = torch.empty(
        expected_metadata_shape,
        dtype=torch.float32,
        device=logits.device,
    )

    if batch_size == 0:
        return output

    softmax_lse = torch.empty(
        (
            batch_size,
            num_kv_heads,
            queries_per_kv,
        ),
        dtype=torch.float32,
        device=logits.device,
    )

    block_c = 256

    _cluster_softmax_lse_kernel[
        (
            batch_size,
            num_kv_heads,
            queries_per_kv,
        )
    ](
        logits,
        cluster_mask,
        cluster_token_counts,
        softmax_lse,
        scale,
        num_clusters,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        logits.stride(3),
        cluster_mask.stride(0),
        cluster_mask.stride(1),
        cluster_mask.stride(2),
        cluster_token_counts.stride(0),
        cluster_token_counts.stride(1),
        cluster_token_counts.stride(2),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(2),
        BLOCK_C=block_c,
        num_warps=4,
    )

    _reduce_grouped_cluster_scores_kernel[
        (
            batch_size,
            num_kv_heads,
            triton.cdiv(num_clusters, block_c),
        )
    ](
        logits,
        cluster_mask,
        cluster_token_counts,
        softmax_lse,
        output,
        scale,
        num_clusters,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        logits.stride(3),
        cluster_mask.stride(0),
        cluster_mask.stride(1),
        cluster_mask.stride(2),
        cluster_token_counts.stride(0),
        cluster_token_counts.stride(1),
        cluster_token_counts.stride(2),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        QUERIES_PER_KV=queries_per_kv,
        BLOCK_C=block_c,
        num_warps=4,
    )

    return output
