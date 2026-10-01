# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _cluster_rank_dot_kernel(
    query,
    keys,
    logits,
    request_rows,
    query_s0: tl.constexpr,
    query_s1: tl.constexpr,
    query_s2: tl.constexpr,
    keys_s0: tl.constexpr,
    keys_s1: tl.constexpr,
    keys_s2: tl.constexpr,
    logits_s0: tl.constexpr,
    logits_s1: tl.constexpr,
    logits_s2: tl.constexpr,
    num_heads: tl.constexpr,
    num_groups: tl.constexpr,
    num_clusters: tl.constexpr,
    head_size: tl.constexpr,
    block_c: tl.constexpr,
    block_d: tl.constexpr,
    has_request_map: tl.constexpr,
):
    cluster_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    row = batch_head // num_heads
    head = batch_head % num_heads
    request = tl.load(request_rows + row).to(tl.int32) if has_request_map else row
    groups = tl.arange(0, 16)
    clusters = cluster_tile * block_c + tl.arange(0, block_c)
    dimensions = tl.arange(0, block_d)
    query_tile = tl.load(
        query
        + row * query_s0
        + head * query_s1
        + groups[:, None] * query_s2
        + dimensions[None, :],
        mask=(groups[:, None] < num_groups) & (dimensions[None, :] < head_size),
        other=0,
    )
    key_tile = tl.load(
        keys
        + request * keys_s0
        + head * keys_s1
        + clusters[:, None] * keys_s2
        + dimensions[None, :],
        mask=(clusters[:, None] < num_clusters) & (dimensions[None, :] < head_size),
        other=0,
    )
    products = tl.dot(query_tile, tl.trans(key_tile))
    tl.store(
        logits
        + row * logits_s0
        + head * logits_s1
        + groups[:, None] * logits_s2
        + clusters[None, :],
        products,
        mask=(groups[:, None] < num_groups) & (clusters[None, :] < num_clusters),
    )


@triton.jit
def _cluster_rank_int8_kernel(
    query,
    keys,
    key_scales,
    logits,
    request_rows,
    query_s0: tl.constexpr,
    query_s1: tl.constexpr,
    query_s2: tl.constexpr,
    keys_s0: tl.constexpr,
    keys_s1: tl.constexpr,
    keys_s2: tl.constexpr,
    scale_s0: tl.constexpr,
    scale_s1: tl.constexpr,
    logits_s0: tl.constexpr,
    logits_s1: tl.constexpr,
    logits_s2: tl.constexpr,
    num_heads: tl.constexpr,
    num_groups: tl.constexpr,
    num_clusters: tl.constexpr,
    head_size: tl.constexpr,
    block_c: tl.constexpr,
    block_d: tl.constexpr,
):
    cluster_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    row = batch_head // num_heads
    head = batch_head % num_heads
    request = tl.load(request_rows + row).to(tl.int32)
    groups = tl.arange(0, 16)
    clusters = cluster_tile * block_c + tl.arange(0, block_c)
    dimensions = tl.arange(0, block_d)
    q = tl.load(
        query
        + row * query_s0
        + head * query_s1
        + groups[:, None] * query_s2
        + dimensions[None, :],
        mask=(groups[:, None] < num_groups) & (dimensions[None, :] < head_size),
        other=0,
    ).to(tl.float32)
    q_scale = tl.maximum(tl.max(tl.abs(q), 1) / 127.0, 1.0e-8)
    scaled_q = q / q_scale[:, None]
    rounded_q = tl.where(
        scaled_q >= 0, tl.floor(scaled_q + 0.5), tl.ceil(scaled_q - 0.5)
    )
    q_int8 = tl.minimum(tl.maximum(rounded_q, -127.0), 127.0).to(tl.int8)
    k_int8 = tl.load(
        keys
        + request * keys_s0
        + head * keys_s1
        + clusters[:, None] * keys_s2
        + dimensions[None, :],
        mask=(clusters[:, None] < num_clusters) & (dimensions[None, :] < head_size),
        other=0,
    )
    k_scale = tl.load(
        key_scales + request * scale_s0 + head * scale_s1 + clusters,
        mask=clusters < num_clusters,
        other=0,
    )
    products = tl.dot(q_int8, tl.trans(k_int8), out_dtype=tl.int32)
    approximate = products.to(tl.float32) * q_scale[:, None] * k_scale[None, :]
    tl.store(
        logits
        + row * logits_s0
        + head * logits_s1
        + groups[:, None] * logits_s2
        + clusters[None, :],
        approximate,
        mask=(groups[:, None] < num_groups) & (clusters[None, :] < num_clusters),
    )


@triton.jit
def _cluster_scores_kernel(
    logits,
    counts,
    active_mask,
    scores,
    candidate_counts,
    request_rows,
    logits_s0: tl.constexpr,
    logits_s1: tl.constexpr,
    logits_s2: tl.constexpr,
    logits_s3: tl.constexpr,
    counts_s0: tl.constexpr,
    counts_s1: tl.constexpr,
    scores_s0: tl.constexpr,
    scores_s1: tl.constexpr,
    candidate_s0: tl.constexpr,
    num_groups: tl.constexpr,
    num_clusters: tl.constexpr,
    scale: tl.constexpr,
    block_g: tl.constexpr,
    block_c: tl.constexpr,
    has_request_map: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    request = tl.load(request_rows + row).to(tl.int32) if has_request_map else row
    groups = tl.arange(0, block_g)
    clusters = tl.arange(0, block_c)
    in_range = clusters < num_clusters
    active = tl.load(active_mask + row)
    count = tl.load(
        counts + request * counts_s0 + head * counts_s1 + clusters,
        mask=in_range,
        other=0,
    )
    valid = in_range & (count > 0) & active
    raw = tl.load(
        logits
        + row * logits_s0
        + head * logits_s1
        + groups[:, None] * logits_s2
        + clusters[None, :] * logits_s3,
        mask=(groups[:, None] < num_groups) & in_range[None, :],
        other=0,
    )
    weighted = raw * scale + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    weighted = tl.where(
        valid[None, :] & (groups[:, None] < num_groups), weighted, -1.0e30
    )
    maximum = tl.max(weighted, 1)
    exponent = tl.where(
        valid[None, :] & (groups[:, None] < num_groups),
        tl.exp(weighted - maximum[:, None]),
        0.0,
    )
    normalizer = tl.maximum(tl.sum(exponent, 1), 1.0e-20)
    probability = exponent / normalizer[:, None]
    score = tl.sum(probability, 0) / num_groups
    score = tl.where(valid, score, float("-inf"))
    tl.store(
        scores + row * scores_s0 + head * scores_s1 + clusters,
        score,
        mask=in_range,
    )
    tl.store(
        candidate_counts + row * candidate_s0 + head, tl.sum(valid.to(tl.int32), 0)
    )


@triton.jit
def _cluster_mass_kernel(
    ranked_scores,
    candidate_counts,
    active_mask,
    sparse_mass,
    verification_mass,
    expanded_mass,
    scores_s0: tl.constexpr,
    scores_s1: tl.constexpr,
    scores_s2: tl.constexpr,
    counts_s0: tl.constexpr,
    num_heads: tl.constexpr,
    ranking_width: tl.constexpr,
    retrieval_ratio: tl.constexpr,
    estimation_ratio: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.arange(0, block_h)
    ranks = tl.arange(0, block_k)
    valid_head = heads < num_heads
    counts = tl.load(
        candidate_counts + row * counts_s0 + heads,
        mask=valid_head,
        other=0,
    )
    retrieval = tl.minimum(
        tl.ceil(counts.to(tl.float32) * retrieval_ratio).to(tl.int32), counts
    )
    estimation = tl.minimum(
        tl.ceil(counts.to(tl.float32) * estimation_ratio).to(tl.int32),
        counts - retrieval,
    )
    verification_end = tl.minimum(retrieval * 2, retrieval + estimation)
    expanded_end = tl.minimum(retrieval * 3, retrieval + estimation)
    values = tl.load(
        ranked_scores
        + row * scores_s0
        + heads[:, None] * scores_s1
        + ranks[None, :] * scores_s2,
        mask=valid_head[:, None] & (ranks[None, :] < ranking_width),
        other=0,
    )
    values = tl.maximum(values, 0.0)
    sparse = (
        tl.sum(tl.sum(tl.where(ranks[None, :] < retrieval[:, None], values, 0.0), 1), 0)
        / num_heads
    )
    verification = (
        tl.sum(
            tl.sum(
                tl.where(ranks[None, :] < verification_end[:, None], values, 0.0),
                1,
            ),
            0,
        )
        / num_heads
    )
    expanded = (
        tl.sum(
            tl.sum(tl.where(ranks[None, :] < expanded_end[:, None], values, 0.0), 1), 0
        )
        / num_heads
    )
    active = tl.load(active_mask + row)
    tl.store(sparse_mass + row, tl.where(active, sparse, 1.0))
    tl.store(verification_mass + row, tl.where(active, verification, 1.0))
    tl.store(expanded_mass + row, tl.where(active, expanded, 1.0))
