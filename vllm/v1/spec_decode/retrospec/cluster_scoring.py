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
    ranking_output,
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
    ranking_stride_0,
    ranking_stride_1,
    ranking_stride_2,
    QUERIES_PER_KV: tl.constexpr,
    WRITE_RANKING: tl.constexpr,
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

    probability_scores = tl.where(
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
        probability_scores,
        mask=cluster_in_bounds,
    )

    if WRITE_RANKING:
        ranking_scores = tl.where(
            valid_clusters,
            cluster_scores,
            float("-inf"),
        )
        ranking_offsets = (
            batch_idx * ranking_stride_0
            + kv_head_idx * ranking_stride_1
            + cluster_offsets * ranking_stride_2
        )
        tl.store(
            ranking_output + ranking_offsets,
            ranking_scores,
            mask=cluster_in_bounds,
        )


RESIDENT_CLUSTER_SCORE_TILE_SIZE = 32


@triton.jit
def _resident_cluster_tile_stats_kernel(
    query,
    cluster_keys,
    cluster_ids,
    cluster_token_counts,
    cluster_offsets,
    num_clusters,
    request_slot_ids,
    tile_max,
    tile_sum,
    tile_candidate_counts,
    scale,
    max_num_clusters,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    id_stride_0,
    id_stride_1,
    count_stride_0,
    count_stride_1,
    cluster_offset_stride_0,
    tile_max_stride_0,
    tile_max_stride_1,
    tile_max_stride_2,
    tile_max_stride_3,
    tile_sum_stride_0,
    tile_sum_stride_1,
    tile_sum_stride_2,
    tile_sum_stride_3,
    tile_count_stride_0,
    tile_count_stride_1,
    tile_count_stride_2,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)

    request_slot = tl.load(request_slot_ids + batch_idx)
    valid_request = request_slot >= 0
    safe_slot = tl.maximum(request_slot, 0)
    request_num_clusters = tl.load(
        num_clusters + safe_slot, mask=valid_request, other=0
    )
    request_cluster_offset = tl.load(
        cluster_offsets + safe_slot * cluster_offset_stride_0,
        mask=valid_request,
        other=0,
    ).to(tl.int64)

    local_clusters = tile_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    absolute_clusters = request_cluster_offset + local_clusters
    in_bounds = local_clusters < max_num_clusters
    in_request = valid_request & in_bounds & (local_clusters < request_num_clusters)
    cluster_id = tl.load(
        cluster_ids + kv_head_idx * id_stride_0 + absolute_clusters * id_stride_1,
        mask=in_request,
        other=-1,
    )
    token_counts = tl.load(
        cluster_token_counts
        + kv_head_idx * count_stride_0
        + absolute_clusters * count_stride_1,
        mask=in_request,
        other=0,
    )
    valid_clusters = in_request & (cluster_id >= 0) & (token_counts > 0)

    query_offsets = tl.arange(0, BLOCK_Q)
    head_offsets = tl.arange(0, BLOCK_D)
    valid_query = query_offsets < QUERIES_PER_KV
    valid_head = head_offsets < HEAD_SIZE
    query_head_indices = kv_head_idx * QUERIES_PER_KV + query_offsets
    query_vectors = tl.load(
        query
        + batch_idx * query_stride_0
        + query_head_indices[:, None] * query_stride_1
        + head_offsets[None, :] * query_stride_2,
        mask=valid_query[:, None] & valid_head[None, :],
        other=0.0,
    )
    cluster_vectors = tl.load(
        cluster_keys
        + kv_head_idx * key_stride_0
        + head_offsets[:, None] * key_stride_2
        + absolute_clusters[None, :] * key_stride_1,
        mask=valid_head[:, None] & valid_clusters[None, :],
        other=0.0,
    )
    log_token_counts = tl.log(tl.maximum(token_counts.to(tl.float32), 1.0))
    valid_count = tl.sum(valid_clusters.to(tl.int32), axis=0)
    has_candidates = valid_count > 0
    logits = tl.dot(
        query_vectors,
        cluster_vectors,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    weighted_logits = logits * scale + log_token_counts[None, :]
    valid_logits = valid_query[:, None] & valid_clusters[None, :]
    weighted_logits = tl.where(valid_logits, weighted_logits, float("-inf"))
    block_max = tl.max(weighted_logits, axis=1)
    safe_block_max = tl.where(has_candidates, block_max, 0.0)
    block_sum = tl.sum(
        tl.where(
            valid_logits,
            tl.exp(weighted_logits - safe_block_max[:, None]),
            0.0,
        ),
        axis=1,
    )

    max_offsets = (
        batch_idx * tile_max_stride_0
        + kv_head_idx * tile_max_stride_1
        + query_offsets * tile_max_stride_2
        + tile_idx * tile_max_stride_3
    )
    sum_offsets = (
        batch_idx * tile_sum_stride_0
        + kv_head_idx * tile_sum_stride_1
        + query_offsets * tile_sum_stride_2
        + tile_idx * tile_sum_stride_3
    )
    tl.store(tile_max + max_offsets, safe_block_max, mask=valid_query)
    tl.store(tile_sum + sum_offsets, block_sum, mask=valid_query)

    count_offset = (
        batch_idx * tile_count_stride_0
        + kv_head_idx * tile_count_stride_1
        + tile_idx * tile_count_stride_2
    )
    tl.store(tile_candidate_counts + count_offset, valid_count)


@triton.jit
def _resident_cluster_lse_kernel(
    tile_max,
    tile_sum,
    tile_candidate_counts,
    softmax_lse,
    candidate_counts,
    num_tiles,
    tile_max_stride_0,
    tile_max_stride_1,
    tile_max_stride_2,
    tile_max_stride_3,
    tile_sum_stride_0,
    tile_sum_stride_1,
    tile_sum_stride_2,
    tile_sum_stride_3,
    tile_count_stride_0,
    tile_count_stride_1,
    tile_count_stride_2,
    lse_stride_0,
    lse_stride_1,
    lse_stride_2,
    candidate_stride_0,
    candidate_stride_1,
    BLOCK_T: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    query_group_idx = tl.program_id(2)
    running_max = float("-inf")
    running_sum = 0.0
    candidate_count = 0
    for tile_idx in tl.static_range(0, BLOCK_T):
        valid_tile = tile_idx < num_tiles
        max_offset = (
            batch_idx * tile_max_stride_0
            + kv_head_idx * tile_max_stride_1
            + query_group_idx * tile_max_stride_2
            + tile_idx * tile_max_stride_3
        )
        sum_offset = (
            batch_idx * tile_sum_stride_0
            + kv_head_idx * tile_sum_stride_1
            + query_group_idx * tile_sum_stride_2
            + tile_idx * tile_sum_stride_3
        )
        partial_max = tl.load(tile_max + max_offset, mask=valid_tile, other=0.0).to(
            tl.float32
        )
        partial_sum = tl.load(tile_sum + sum_offset, mask=valid_tile, other=0.0).to(
            tl.float32
        )
        has_partial = valid_tile & (partial_sum > 0)
        next_max = tl.where(
            has_partial,
            tl.maximum(running_max, partial_max),
            running_max,
        )
        previous_scale = tl.where(
            running_sum > 0,
            tl.exp(running_max - next_max),
            0.0,
        )
        partial_scale = tl.where(
            has_partial,
            tl.exp(partial_max - next_max),
            0.0,
        )
        running_sum = running_sum * previous_scale + partial_sum * partial_scale
        running_max = next_max

        count_offset = (
            batch_idx * tile_count_stride_0
            + kv_head_idx * tile_count_stride_1
            + tile_idx * tile_count_stride_2
        )
        candidate_count += tl.load(
            tile_candidate_counts + count_offset,
            mask=valid_tile,
            other=0,
        ).to(tl.int32)

    row_lse = tl.where(
        running_sum > 0,
        running_max + tl.log(running_sum),
        0.0,
    )
    lse_offset = (
        batch_idx * lse_stride_0
        + kv_head_idx * lse_stride_1
        + query_group_idx * lse_stride_2
    )
    tl.store(softmax_lse + lse_offset, row_lse)

    output_count_offset = (
        batch_idx * candidate_stride_0 + kv_head_idx * candidate_stride_1
    )
    tl.store(
        candidate_counts + output_count_offset,
        candidate_count,
        mask=query_group_idx == 0,
    )


@triton.jit
def _resident_cluster_scores_kernel(
    query,
    cluster_keys,
    cluster_ids,
    cluster_token_counts,
    cluster_offsets,
    num_clusters,
    request_slot_ids,
    softmax_lse,
    output,
    scale,
    max_num_clusters,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    id_stride_0,
    id_stride_1,
    count_stride_0,
    count_stride_1,
    cluster_offset_stride_0,
    lse_stride_0,
    lse_stride_1,
    lse_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)

    request_slot = tl.load(request_slot_ids + batch_idx)
    valid_request = request_slot >= 0
    safe_slot = tl.maximum(request_slot, 0)
    request_num_clusters = tl.load(
        num_clusters + safe_slot, mask=valid_request, other=0
    )
    request_cluster_offset = tl.load(
        cluster_offsets + safe_slot * cluster_offset_stride_0,
        mask=valid_request,
        other=0,
    ).to(tl.int64)

    local_clusters = tile_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    absolute_clusters = request_cluster_offset + local_clusters
    in_bounds = local_clusters < max_num_clusters
    in_request = valid_request & in_bounds & (local_clusters < request_num_clusters)
    cluster_id = tl.load(
        cluster_ids + kv_head_idx * id_stride_0 + absolute_clusters * id_stride_1,
        mask=in_request,
        other=-1,
    )
    token_counts = tl.load(
        cluster_token_counts
        + kv_head_idx * count_stride_0
        + absolute_clusters * count_stride_1,
        mask=in_request,
        other=0,
    )
    valid_clusters = in_request & (cluster_id >= 0) & (token_counts > 0)

    query_offsets = tl.arange(0, BLOCK_Q)
    head_offsets = tl.arange(0, BLOCK_D)
    valid_query = query_offsets < QUERIES_PER_KV
    valid_head = head_offsets < HEAD_SIZE
    query_head_indices = kv_head_idx * QUERIES_PER_KV + query_offsets
    query_vectors = tl.load(
        query
        + batch_idx * query_stride_0
        + query_head_indices[:, None] * query_stride_1
        + head_offsets[None, :] * query_stride_2,
        mask=valid_query[:, None] & valid_head[None, :],
        other=0.0,
    )
    cluster_vectors = tl.load(
        cluster_keys
        + kv_head_idx * key_stride_0
        + head_offsets[:, None] * key_stride_2
        + absolute_clusters[None, :] * key_stride_1,
        mask=valid_head[:, None] & valid_clusters[None, :],
        other=0.0,
    )
    log_token_counts = tl.log(tl.maximum(token_counts.to(tl.float32), 1.0))
    logits = tl.dot(
        query_vectors,
        cluster_vectors,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    lse_offsets = (
        batch_idx * lse_stride_0
        + kv_head_idx * lse_stride_1
        + query_offsets * lse_stride_2
    )
    row_lse = tl.load(softmax_lse + lse_offsets, mask=valid_query, other=0.0)
    probabilities = tl.where(
        valid_query[:, None] & valid_clusters[None, :],
        tl.exp(logits * scale + log_token_counts[None, :] - row_lse[:, None]),
        0.0,
    )
    scores = tl.sum(probabilities, axis=0) / QUERIES_PER_KV
    output_offsets = (
        batch_idx * output_stride_0
        + kv_head_idx * output_stride_1
        + local_clusters * output_stride_2
    )
    tl.store(
        output + output_offsets,
        tl.where(valid_clusters, scores, float("-inf")),
        mask=in_bounds,
    )


def _prepare_float_output(
    output: torch.Tensor | None,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Allocate or validate a reusable float32 output tensor."""
    if output is None:
        return torch.empty(
            shape,
            dtype=torch.float32,
            device=device,
        )

    if output.shape != shape:
        raise ValueError(f"{name} has an unexpected shape")
    if output.dtype != torch.float32:
        raise ValueError(f"{name} must use float32")
    if output.device != device:
        raise ValueError(f"{name} must be on the logits device")

    return output


def score_resident_clusters(
    query: torch.Tensor,
    cluster_keys: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    cluster_offsets: torch.Tensor,
    num_clusters: torch.Tensor,
    request_slot_ids: torch.Tensor,
    scale: float,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    tile_max: torch.Tensor,
    tile_sum: torch.Tensor,
    tile_candidate_counts: torch.Tensor,
    candidate_counts: torch.Tensor,
) -> torch.Tensor:
    """Score resident summaries without materializing grouped logits."""
    if query.device.type != "cuda":
        raise ValueError("Resident cluster scoring requires CUDA tensors")
    if query.ndim != 3 or cluster_keys.ndim != 3:
        raise ValueError("Resident query or cluster-key shape is invalid")

    batch_size, num_query_heads, head_size = query.shape
    num_kv_heads, _, key_head_size = cluster_keys.shape
    if output.ndim != 3:
        raise ValueError("Resident cluster-score workspace is invalid")
    max_num_clusters = output.shape[2]
    if head_size != key_head_size:
        raise ValueError("Resident query and cluster head sizes do not match")
    if query.dtype != cluster_keys.dtype:
        raise ValueError("Resident query and cluster keys must use one dtype")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Resident cluster scoring requires a floating-point dtype")
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError("Resident query heads are incompatible with KV heads")
    if max_num_clusters <= 0:
        raise ValueError("Resident scoring workspace must contain a cluster slot")

    cluster_shape = cluster_keys.shape[:2]
    if cluster_ids.shape != cluster_shape:
        raise ValueError("Resident cluster IDs do not match cluster keys")
    if cluster_token_counts.shape != cluster_shape:
        raise ValueError("Resident cluster counts do not match cluster keys")
    if num_clusters.ndim != 1:
        raise ValueError("Resident request cluster counts have an invalid shape")
    if cluster_offsets.shape != num_clusters.shape:
        raise ValueError("Resident request cluster offsets have an invalid shape")
    if request_slot_ids.shape != (batch_size,):
        raise ValueError("Resident request slots do not match the query batch")

    queries_per_kv = num_query_heads // num_kv_heads
    num_tiles = triton.cdiv(max_num_clusters, RESIDENT_CLUSTER_SCORE_TILE_SIZE)
    score_shape = (batch_size, num_kv_heads, max_num_clusters)
    lse_shape = (batch_size, num_kv_heads, queries_per_kv)
    tile_shape = (batch_size, num_kv_heads, queries_per_kv, num_tiles)
    tile_count_shape = (batch_size, num_kv_heads, num_tiles)
    if output.shape != score_shape or output.dtype != torch.float32:
        raise ValueError("Resident cluster-score workspace is invalid")
    if softmax_lse.shape != lse_shape or softmax_lse.dtype != torch.float32:
        raise ValueError("Resident softmax workspace is invalid")
    if tile_max.shape != tile_shape or tile_max.dtype != torch.float32:
        raise ValueError("Resident tile-max workspace is invalid")
    if tile_sum.shape != tile_shape or tile_sum.dtype != torch.float32:
        raise ValueError("Resident tile-sum workspace is invalid")
    if (
        tile_candidate_counts.shape != tile_count_shape
        or tile_candidate_counts.dtype != torch.int32
    ):
        raise ValueError("Resident tile-count workspace is invalid")
    if (
        candidate_counts.shape != score_shape[:2]
        or candidate_counts.dtype != torch.int32
    ):
        raise ValueError("Resident candidate-count workspace is invalid")

    tensors = (
        cluster_keys,
        cluster_ids,
        cluster_token_counts,
        cluster_offsets,
        num_clusters,
        request_slot_ids,
        output,
        softmax_lse,
        tile_max,
        tile_sum,
        tile_candidate_counts,
        candidate_counts,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("Resident cluster scoring tensors must use one CUDA device")
    if cluster_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Resident cluster IDs must use an integral dtype")
    if cluster_token_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("Resident cluster counts must use an integral dtype")
    if num_clusters.dtype not in (torch.int32, torch.int64):
        raise ValueError("Resident request cluster counts must be integral")
    if cluster_offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("Resident request cluster offsets must be integral")
    if request_slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Resident request slots must be integral")
    if batch_size == 0:
        return output

    block_d = max(16, triton.next_power_of_2(head_size))
    block_q = max(16, triton.next_power_of_2(queries_per_kv))
    block_c = RESIDENT_CLUSTER_SCORE_TILE_SIZE
    _resident_cluster_tile_stats_kernel[(batch_size, num_kv_heads, num_tiles)](
        query,
        cluster_keys,
        cluster_ids,
        cluster_token_counts,
        cluster_offsets,
        num_clusters,
        request_slot_ids,
        tile_max,
        tile_sum,
        tile_candidate_counts,
        scale,
        max_num_clusters,
        *query.stride(),
        *cluster_keys.stride(),
        *cluster_ids.stride(),
        *cluster_token_counts.stride(),
        cluster_offsets.stride(0),
        *tile_max.stride(),
        *tile_sum.stride(),
        *tile_candidate_counts.stride(),
        QUERIES_PER_KV=queries_per_kv,
        HEAD_SIZE=head_size,
        BLOCK_Q=block_q,
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=4,
    )
    _resident_cluster_lse_kernel[(batch_size, num_kv_heads, queries_per_kv)](
        tile_max,
        tile_sum,
        tile_candidate_counts,
        softmax_lse,
        candidate_counts,
        num_tiles,
        *tile_max.stride(),
        *tile_sum.stride(),
        *tile_candidate_counts.stride(),
        *softmax_lse.stride(),
        *candidate_counts.stride(),
        BLOCK_T=triton.next_power_of_2(num_tiles),
        num_warps=4,
    )
    _resident_cluster_scores_kernel[(batch_size, num_kv_heads, num_tiles)](
        query,
        cluster_keys,
        cluster_ids,
        cluster_token_counts,
        cluster_offsets,
        num_clusters,
        request_slot_ids,
        softmax_lse,
        output,
        scale,
        max_num_clusters,
        *query.stride(),
        *cluster_keys.stride(),
        *cluster_ids.stride(),
        *cluster_token_counts.stride(),
        cluster_offsets.stride(0),
        *softmax_lse.stride(),
        *output.stride(),
        QUERIES_PER_KV=queries_per_kv,
        HEAD_SIZE=head_size,
        BLOCK_Q=block_q,
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output


def reduce_grouped_cluster_scores(
    logits: torch.Tensor,
    cluster_mask: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    scale: float,
    output: torch.Tensor | None = None,
    softmax_lse: torch.Tensor | None = None,
    ranking_output: torch.Tensor | None = None,
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
        output:
            Optional reusable float32 probability-score output.
        softmax_lse:
            Optional reusable float32 grouped softmax workspace.
        ranking_output:
            Optional reusable float32 ranking output. Invalid clusters are
            written as negative infinity.

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
    metadata_shape = (
        batch_size,
        num_kv_heads,
        num_clusters,
    )
    lse_shape = (
        batch_size,
        num_kv_heads,
        queries_per_kv,
    )

    if cluster_mask.shape != metadata_shape:
        raise ValueError("cluster_mask shape does not match cluster logits")
    if cluster_token_counts.shape != metadata_shape:
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

    output = _prepare_float_output(
        output,
        metadata_shape,
        logits.device,
        "output",
    )
    softmax_lse = _prepare_float_output(
        softmax_lse,
        lse_shape,
        logits.device,
        "softmax_lse",
    )

    if ranking_output is not None:
        ranking_output = _prepare_float_output(
            ranking_output,
            metadata_shape,
            logits.device,
            "ranking_output",
        )

    reusable_outputs = [
        ("output", output),
        ("softmax_lse", softmax_lse),
    ]
    if ranking_output is not None:
        reusable_outputs.append(("ranking_output", ranking_output))

    for name, tensor in reusable_outputs:
        if torch._C._overlaps(logits, tensor):
            raise ValueError(f"logits and {name} must not overlap")

    for index, (name, tensor) in enumerate(reusable_outputs):
        for other_name, other_tensor in reusable_outputs[index + 1 :]:
            if torch._C._overlaps(tensor, other_tensor):
                raise ValueError(f"{name} and {other_name} must not overlap")

    if batch_size == 0:
        return output

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

    write_ranking = ranking_output is not None
    ranking_buffer = output if ranking_output is None else ranking_output

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
        ranking_buffer,
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
        ranking_buffer.stride(0),
        ranking_buffer.stride(1),
        ranking_buffer.stride(2),
        QUERIES_PER_KV=queries_per_kv,
        WRITE_RANKING=write_ranking,
        BLOCK_C=block_c,
        num_warps=4,
    )

    return output
