# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU-only clustered index and sparse attention for RetroSpec.

The native vLLM KV blocks remain the sole exact-token storage. The reverse
cluster layout contains logical token indices, not copies of K/V vectors.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

from .clustering import segmented_kmeans
from .index import RetroSpecIndexBase
from .performance import RetroSpecPerformanceStats


@dataclass(frozen=True)
class _NativeLayerRecord:
    indexed_end: int
    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    token_indices: torch.Tensor
    cluster_offsets: torch.Tensor
    key_storage: torch.Tensor | None = None
    value_storage: torch.Tensor | None = None
    count_storage: torch.Tensor | None = None
    token_storage: torch.Tensor | None = None
    offset_storage: torch.Tensor | None = None


@dataclass(frozen=True)
class _NativeBatchLayer:
    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    token_indices: torch.Tensor
    cluster_offsets: torch.Tensor
    indexed_ends: torch.Tensor
    request_slots: torch.Tensor | None = None
    quantized_keys: torch.Tensor | None = None
    key_scales: torch.Tensor | None = None


@dataclass(frozen=True)
class _NativeRankedPlan:
    ranked: torch.Tensor
    candidate_counts: torch.Tensor
    sparse_mass: torch.Tensor
    verification_mass: torch.Tensor
    expanded_mass: torch.Tensor


@dataclass(frozen=True)
class _NativePlanWorkspace:
    ranked: torch.Tensor
    candidate_counts: torch.Tensor
    sparse_mass: torch.Tensor
    verification_mass: torch.Tensor
    expanded_mass: torch.Tensor
    scores: torch.Tensor
    topk_values: torch.Tensor


@dataclass(frozen=True)
class _NativeAttentionWorkspace:
    output: torch.Tensor
    maximum: torch.Tensor
    denominator: torch.Tensor


@triton.jit
def _scatter_cluster_tokens_kernel(
    assignments,
    token_offsets,
    cluster_offsets,
    token_indices,
    assignment_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    cluster_stride: tl.constexpr,
    output_stride: tl.constexpr,
    num_tokens: tl.constexpr,
    first_token: tl.constexpr,
    block_size: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.program_id(1) * block_size + tl.arange(0, block_size)
    valid = tokens < num_tokens
    clusters = tl.load(
        assignments + head * assignment_stride + tokens, mask=valid, other=0
    )
    offsets = tl.load(
        token_offsets + head * offset_stride + tokens, mask=valid, other=0
    )
    starts = tl.load(
        cluster_offsets + head * cluster_stride + clusters, mask=valid, other=0
    )
    tl.store(
        token_indices + head * output_stride + starts + offsets,
        first_token + tokens,
        mask=valid,
    )


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


@triton.jit
def _native_sparse_attention_kernel(
    query,
    key_cache,
    value_cache,
    block_table,
    token_indices,
    cluster_offsets,
    cluster_keys,
    cluster_values,
    cluster_counts,
    ranked_clusters,
    candidate_counts,
    indexed_ends,
    seq_lens,
    request_indices,
    layout_slots,
    output,
    partial_output,
    partial_maximum,
    partial_denominator,
    query_s0: tl.constexpr,
    query_s1: tl.constexpr,
    query_s2: tl.constexpr,
    key_s0: tl.constexpr,
    key_s1: tl.constexpr,
    key_s2: tl.constexpr,
    value_s0: tl.constexpr,
    value_s1: tl.constexpr,
    value_s2: tl.constexpr,
    block_s0: tl.constexpr,
    block_s1: tl.constexpr,
    token_s0: tl.constexpr,
    token_s1: tl.constexpr,
    offset_s0: tl.constexpr,
    offset_s1: tl.constexpr,
    summary_s0: tl.constexpr,
    summary_s1: tl.constexpr,
    count_s0: tl.constexpr,
    count_s1: tl.constexpr,
    ranked_s0: tl.constexpr,
    ranked_s1: tl.constexpr,
    count_row_s0: tl.constexpr,
    output_s0: tl.constexpr,
    output_s1: tl.constexpr,
    output_s2: tl.constexpr,
    partial_s0: tl.constexpr,
    partial_s1: tl.constexpr,
    partial_s2: tl.constexpr,
    partial_m0: tl.constexpr,
    partial_m1: tl.constexpr,
    scale: tl.constexpr,
    retrieval_ratio: tl.constexpr,
    estimation_ratio: tl.constexpr,
    expanded: tl.constexpr,
    sparse_verify: tl.constexpr,
    num_kv_heads: tl.constexpr,
    queries_per_kv: tl.constexpr,
    num_clusters: tl.constexpr,
    ranking_width: tl.constexpr,
    page_size: tl.constexpr,
    head_size: tl.constexpr,
    block_d: tl.constexpr,
    block_t: tl.constexpr,
    block_e: tl.constexpr,
    num_partitions: tl.constexpr,
):
    row = tl.program_id(0)
    query_head = tl.program_id(1)
    partition = tl.program_id(2)
    kv_head = query_head // queries_per_kv
    request = tl.load(request_indices + row).to(tl.int32)
    request = tl.load(layout_slots + request).to(tl.int32)
    seq_len = tl.load(seq_lens + row).to(tl.int32)
    indexed_end = tl.load(indexed_ends + request).to(tl.int32)
    candidate_count = tl.load(candidate_counts + row * count_row_s0 + kv_head).to(
        tl.int32
    )
    retrieval_count = tl.minimum(
        tl.ceil(candidate_count.to(tl.float32) * retrieval_ratio).to(tl.int32),
        candidate_count,
    )
    estimation_count = tl.minimum(
        tl.ceil(candidate_count.to(tl.float32) * estimation_ratio).to(tl.int32),
        candidate_count - retrieval_count,
    )
    estimation_end = tl.minimum(retrieval_count + estimation_count, candidate_count)
    if expanded:
        retrieval_count = tl.minimum(retrieval_count * 3, estimation_end)
    elif sparse_verify:
        retrieval_count = tl.minimum(retrieval_count * 2, estimation_end)

    dimensions = tl.arange(0, block_d)
    dimension_mask = dimensions < head_size
    q = tl.load(
        query + row * query_s0 + query_head * query_s1 + dimensions * query_s2,
        mask=dimension_mask,
        other=0,
    ).to(tl.float32)
    running_max = float("-inf")
    running_sum = 0.0
    running_output = tl.zeros((block_d,), dtype=tl.float32)
    token_offsets = tl.arange(0, block_t)

    # The permanent sink and the unindexed/recent suffix are native KV.
    if partition == 0:
        for zone in tl.static_range(2):
            zone_start = 0 if zone == 0 else indexed_end
            zone_end = tl.minimum(page_size, seq_len) if zone == 0 else seq_len
            for start in range(zone_start, zone_end, block_t):
                logical = start + token_offsets
                valid = logical < zone_end
                physical = tl.load(
                    block_table + row * block_s0 + (logical // page_size) * block_s1,
                    mask=valid,
                    other=0,
                ).to(tl.int32)
                key_ptrs = (
                    key_cache
                    + physical[:, None] * key_s0
                    + (logical % page_size)[:, None] * key_s1
                    + kv_head * key_s2
                    + dimensions[None, :]
                )
                value_ptrs = (
                    value_cache
                    + physical[:, None] * value_s0
                    + (logical % page_size)[:, None] * value_s1
                    + kv_head * value_s2
                    + dimensions[None, :]
                )
                token_mask = valid[:, None] & dimension_mask[None, :]
                keys = tl.load(key_ptrs, mask=token_mask, other=0).to(tl.float32)
                values = tl.load(value_ptrs, mask=token_mask, other=0).to(tl.float32)
                logits = tl.sum(keys * q[None, :], 1) * scale
                logits = tl.where(valid, logits, float("-inf"))
                block_max = tl.max(logits, 0)
                next_max = tl.maximum(running_max, block_max)
                previous_scale = tl.where(
                    running_sum > 0, tl.exp(running_max - next_max), 0.0
                )
                weights = tl.where(valid, tl.exp(logits - next_max), 0.0)
                running_output = running_output * previous_scale + tl.sum(
                    values * weights[:, None], 0
                )
                running_sum = running_sum * previous_scale + tl.sum(weights, 0)
                running_max = next_max

    # Top-k cluster tokens are looked up through a compact reverse index. No
    # gathered KV page or CPU-resident cache is involved.
    for rank in range(partition, retrieval_count, num_partitions):
        cluster = tl.load(
            ranked_clusters + row * ranked_s0 + kv_head * ranked_s1 + rank
        ).to(tl.int32)
        start = tl.load(
            cluster_offsets + request * offset_s0 + kv_head * offset_s1 + cluster
        ).to(tl.int32)
        end = tl.load(
            cluster_offsets + request * offset_s0 + kv_head * offset_s1 + cluster + 1
        ).to(tl.int32)
        for position in range(start, end, block_t):
            packed = position + token_offsets
            valid = packed < end
            logical = tl.load(
                token_indices + request * token_s0 + kv_head * token_s1 + packed,
                mask=valid,
                other=0,
            ).to(tl.int32)
            physical = tl.load(
                block_table + row * block_s0 + (logical // page_size) * block_s1,
                mask=valid,
                other=0,
            ).to(tl.int32)
            key_ptrs = (
                key_cache
                + physical[:, None] * key_s0
                + (logical % page_size)[:, None] * key_s1
                + kv_head * key_s2
                + dimensions[None, :]
            )
            value_ptrs = (
                value_cache
                + physical[:, None] * value_s0
                + (logical % page_size)[:, None] * value_s1
                + kv_head * value_s2
                + dimensions[None, :]
            )
            token_mask = valid[:, None] & dimension_mask[None, :]
            keys = tl.load(key_ptrs, mask=token_mask, other=0).to(tl.float32)
            values = tl.load(value_ptrs, mask=token_mask, other=0).to(tl.float32)
            logits = tl.sum(keys * q[None, :], 1) * scale
            logits = tl.where(valid, logits, float("-inf"))
            block_max = tl.max(logits, 0)
            next_max = tl.maximum(running_max, block_max)
            previous_scale = tl.where(
                running_sum > 0, tl.exp(running_max - next_max), 0.0
            )
            weights = tl.where(valid, tl.exp(logits - next_max), 0.0)
            running_output = running_output * previous_scale + tl.sum(
                values * weights[:, None], 0
            )
            running_sum = running_sum * previous_scale + tl.sum(weights, 0)
            running_max = next_max

    estimate_offsets = tl.arange(0, block_e)
    for rank_start in range(
        retrieval_count + partition * block_e,
        estimation_end,
        num_partitions * block_e,
    ):
        ranks = rank_start + estimate_offsets
        valid = ranks < estimation_end
        cluster = tl.load(
            ranked_clusters + row * ranked_s0 + kv_head * ranked_s1 + ranks,
            mask=valid,
            other=0,
        ).to(tl.int32)
        summary_offsets = (
            request * summary_s0 + kv_head * summary_s1 + cluster * head_size
        )
        summary_mask = valid[:, None] & dimension_mask[None, :]
        keys = tl.load(
            cluster_keys + summary_offsets[:, None] + dimensions[None, :],
            mask=summary_mask,
            other=0,
        ).to(tl.float32)
        values = tl.load(
            cluster_values + summary_offsets[:, None] + dimensions[None, :],
            mask=summary_mask,
            other=0,
        ).to(tl.float32)
        counts = tl.load(
            cluster_counts + request * count_s0 + kv_head * count_s1 + cluster,
            mask=valid,
            other=1,
        )
        logits = tl.sum(keys * q[None, :], 1) * scale
        logits += tl.log(tl.maximum(counts, 1).to(tl.float32))
        logits = tl.where(valid, logits, float("-inf"))
        next_max = tl.maximum(running_max, tl.max(logits, 0))
        previous_scale = tl.where(running_sum > 0, tl.exp(running_max - next_max), 0.0)
        weights = tl.where(valid, tl.exp(logits - next_max), 0.0)
        running_output = running_output * previous_scale + tl.sum(
            values * weights[:, None], 0
        )
        running_sum = running_sum * previous_scale + tl.sum(weights, 0)
        running_max = next_max

    if num_partitions == 1:
        result = running_output / tl.maximum(running_sum, 1.0e-20)
        tl.store(
            output + row * output_s0 + query_head * output_s1 + dimensions * output_s2,
            result,
            mask=dimension_mask,
        )
    else:
        partial_index = (
            row * partial_s0 + query_head * partial_s1 + partition * partial_s2
        )
        tl.store(
            partial_output + partial_index + dimensions,
            running_output,
            mask=dimension_mask,
        )
        scalar_index = row * partial_m0 + query_head * partial_m1 + partition
        tl.store(partial_maximum + scalar_index, running_max)
        tl.store(partial_denominator + scalar_index, running_sum)


@triton.jit
def _native_grouped_sparse_attention_kernel(
    query,
    key_cache,
    value_cache,
    block_table,
    token_indices,
    cluster_offsets,
    cluster_keys,
    cluster_values,
    cluster_counts,
    ranked_clusters,
    candidate_counts,
    indexed_ends,
    seq_lens,
    request_indices,
    layout_slots,
    output,
    partial_output,
    partial_maximum,
    partial_denominator,
    query_s0: tl.constexpr,
    query_s1: tl.constexpr,
    query_s2: tl.constexpr,
    key_s0: tl.constexpr,
    key_s1: tl.constexpr,
    key_s2: tl.constexpr,
    value_s0: tl.constexpr,
    value_s1: tl.constexpr,
    value_s2: tl.constexpr,
    block_s0: tl.constexpr,
    block_s1: tl.constexpr,
    token_s0: tl.constexpr,
    token_s1: tl.constexpr,
    offset_s0: tl.constexpr,
    offset_s1: tl.constexpr,
    summary_s0: tl.constexpr,
    summary_s1: tl.constexpr,
    count_s0: tl.constexpr,
    count_s1: tl.constexpr,
    ranked_s0: tl.constexpr,
    ranked_s1: tl.constexpr,
    count_row_s0: tl.constexpr,
    output_s0: tl.constexpr,
    output_s1: tl.constexpr,
    output_s2: tl.constexpr,
    partial_s0: tl.constexpr,
    partial_s1: tl.constexpr,
    partial_s2: tl.constexpr,
    partial_m0: tl.constexpr,
    partial_m1: tl.constexpr,
    scale: tl.constexpr,
    retrieval_ratio: tl.constexpr,
    estimation_ratio: tl.constexpr,
    expanded: tl.constexpr,
    sparse_verify: tl.constexpr,
    num_kv_heads: tl.constexpr,
    queries_per_kv: tl.constexpr,
    num_clusters: tl.constexpr,
    ranking_width: tl.constexpr,
    page_size: tl.constexpr,
    head_size: tl.constexpr,
    block_d: tl.constexpr,
    block_t: tl.constexpr,
    block_e: tl.constexpr,
    num_partitions: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    partition = tl.program_id(2)
    groups = tl.arange(0, 16)
    query_heads = kv_head * queries_per_kv + groups
    valid_group = groups < queries_per_kv
    dimensions = tl.arange(0, block_d)
    dimension_mask = dimensions < head_size
    token_offsets = tl.arange(0, block_t)
    estimate_offsets = tl.arange(0, block_e)
    request = tl.load(request_indices + row).to(tl.int32)
    request = tl.load(layout_slots + request).to(tl.int32)
    seq_len = tl.load(seq_lens + row).to(tl.int32)
    indexed_end = tl.load(indexed_ends + request).to(tl.int32)
    candidate_count = tl.load(candidate_counts + row * count_row_s0 + kv_head).to(
        tl.int32
    )
    retrieval_count = tl.minimum(
        tl.ceil(candidate_count.to(tl.float32) * retrieval_ratio).to(tl.int32),
        candidate_count,
    )
    estimation_count = tl.minimum(
        tl.ceil(candidate_count.to(tl.float32) * estimation_ratio).to(tl.int32),
        candidate_count - retrieval_count,
    )
    estimation_end = retrieval_count + estimation_count
    if expanded:
        retrieval_count = tl.minimum(retrieval_count * 3, estimation_end)
    elif sparse_verify:
        retrieval_count = tl.minimum(retrieval_count * 2, estimation_end)

    q = tl.load(
        query
        + row * query_s0
        + query_heads[:, None] * query_s1
        + dimensions[None, :] * query_s2,
        mask=valid_group[:, None] & dimension_mask[None, :],
        other=0,
    )
    running_max = tl.full((16,), float("-inf"), tl.float32)
    running_sum = tl.zeros((16,), tl.float32)
    running_output = tl.zeros((16, block_d), tl.float32)

    if partition == 0:
        for zone in tl.static_range(2):
            zone_start = 0 if zone == 0 else indexed_end
            zone_end = tl.minimum(page_size, seq_len) if zone == 0 else seq_len
            for start in range(zone_start, zone_end, block_t):
                logical = start + token_offsets
                valid = logical < zone_end
                physical = tl.load(
                    block_table + row * block_s0 + (logical // page_size) * block_s1,
                    mask=valid,
                    other=0,
                ).to(tl.int32)
                key_ptrs = (
                    key_cache
                    + physical[:, None] * key_s0
                    + (logical % page_size)[:, None] * key_s1
                    + kv_head * key_s2
                    + dimensions[None, :]
                )
                value_ptrs = (
                    value_cache
                    + physical[:, None] * value_s0
                    + (logical % page_size)[:, None] * value_s1
                    + kv_head * value_s2
                    + dimensions[None, :]
                )
                token_mask = valid[:, None] & dimension_mask[None, :]
                keys = tl.load(key_ptrs, mask=token_mask, other=0)
                values = tl.load(value_ptrs, mask=token_mask, other=0)
                logits = tl.dot(q, tl.trans(keys)) * scale
                valid_logits = valid_group[:, None] & valid[None, :]
                logits = tl.where(valid_logits, logits, float("-inf"))
                block_max = tl.max(logits, 1)
                next_max = tl.maximum(running_max, block_max)
                previous_scale = tl.where(
                    running_sum > 0, tl.exp(running_max - next_max), 0.0
                )
                weights = tl.where(
                    valid_logits, tl.exp(logits - next_max[:, None]), 0.0
                )
                running_output = running_output * previous_scale[:, None] + tl.dot(
                    weights.to(tl.float16), values.to(tl.float16)
                )
                running_sum = running_sum * previous_scale + tl.sum(weights, 1)
                running_max = next_max

    for rank in range(partition, retrieval_count, num_partitions):
        cluster = tl.load(
            ranked_clusters + row * ranked_s0 + kv_head * ranked_s1 + rank
        ).to(tl.int32)
        start = tl.load(
            cluster_offsets + request * offset_s0 + kv_head * offset_s1 + cluster
        ).to(tl.int32)
        end = tl.load(
            cluster_offsets + request * offset_s0 + kv_head * offset_s1 + cluster + 1
        ).to(tl.int32)
        for position in range(start, end, block_t):
            packed = position + token_offsets
            valid = packed < end
            logical = tl.load(
                token_indices + request * token_s0 + kv_head * token_s1 + packed,
                mask=valid,
                other=0,
            ).to(tl.int32)
            physical = tl.load(
                block_table + row * block_s0 + (logical // page_size) * block_s1,
                mask=valid,
                other=0,
            ).to(tl.int32)
            key_ptrs = (
                key_cache
                + physical[:, None] * key_s0
                + (logical % page_size)[:, None] * key_s1
                + kv_head * key_s2
                + dimensions[None, :]
            )
            value_ptrs = (
                value_cache
                + physical[:, None] * value_s0
                + (logical % page_size)[:, None] * value_s1
                + kv_head * value_s2
                + dimensions[None, :]
            )
            token_mask = valid[:, None] & dimension_mask[None, :]
            keys = tl.load(key_ptrs, mask=token_mask, other=0)
            values = tl.load(value_ptrs, mask=token_mask, other=0)
            logits = tl.dot(q, tl.trans(keys)) * scale
            valid_logits = valid_group[:, None] & valid[None, :]
            logits = tl.where(valid_logits, logits, float("-inf"))
            block_max = tl.max(logits, 1)
            next_max = tl.maximum(running_max, block_max)
            previous_scale = tl.where(
                running_sum > 0, tl.exp(running_max - next_max), 0.0
            )
            weights = tl.where(valid_logits, tl.exp(logits - next_max[:, None]), 0.0)
            running_output = running_output * previous_scale[:, None] + tl.dot(
                weights.to(tl.float16), values.to(tl.float16)
            )
            running_sum = running_sum * previous_scale + tl.sum(weights, 1)
            running_max = next_max

    for rank_start in range(
        retrieval_count + partition * block_e,
        estimation_end,
        num_partitions * block_e,
    ):
        ranks = rank_start + estimate_offsets
        valid = ranks < estimation_end
        cluster = tl.load(
            ranked_clusters + row * ranked_s0 + kv_head * ranked_s1 + ranks,
            mask=valid,
            other=0,
        ).to(tl.int32)
        summary_offsets = (
            request * summary_s0 + kv_head * summary_s1 + cluster * head_size
        )
        summary_mask = valid[:, None] & dimension_mask[None, :]
        keys = tl.load(
            cluster_keys + summary_offsets[:, None] + dimensions[None, :],
            mask=summary_mask,
            other=0,
        )
        values = tl.load(
            cluster_values + summary_offsets[:, None] + dimensions[None, :],
            mask=summary_mask,
            other=0,
        )
        counts = tl.load(
            cluster_counts + request * count_s0 + kv_head * count_s1 + cluster,
            mask=valid,
            other=1,
        )
        logits = tl.dot(q, tl.trans(keys)) * scale
        logits += tl.log(tl.maximum(counts, 1).to(tl.float32))[None, :]
        valid_logits = valid_group[:, None] & valid[None, :]
        logits = tl.where(valid_logits, logits, float("-inf"))
        next_max = tl.maximum(running_max, tl.max(logits, 1))
        previous_scale = tl.where(running_sum > 0, tl.exp(running_max - next_max), 0.0)
        weights = tl.where(valid_logits, tl.exp(logits - next_max[:, None]), 0.0)
        running_output = running_output * previous_scale[:, None] + tl.dot(
            weights.to(tl.float16), values.to(tl.float16)
        )
        running_sum = running_sum * previous_scale + tl.sum(weights, 1)
        running_max = next_max

    scalar_index = row * partial_m0 + query_heads * partial_m1 + partition
    if num_partitions == 1:
        result = running_output / tl.maximum(running_sum[:, None], 1.0e-20)
        tl.store(
            output
            + row * output_s0
            + query_heads[:, None] * output_s1
            + dimensions[None, :] * output_s2,
            result,
            mask=valid_group[:, None] & dimension_mask[None, :],
        )
    else:
        tl.store(
            partial_output
            + row * partial_s0
            + query_heads[:, None] * partial_s1
            + partition * partial_s2
            + dimensions[None, :],
            running_output,
            mask=valid_group[:, None] & dimension_mask[None, :],
        )
        tl.store(partial_maximum + scalar_index, running_max, mask=valid_group)
        tl.store(partial_denominator + scalar_index, running_sum, mask=valid_group)


@triton.jit
def _merge_native_attention_partitions(
    partial_output,
    partial_maximum,
    partial_denominator,
    output,
    partial_s0: tl.constexpr,
    partial_s1: tl.constexpr,
    partial_s2: tl.constexpr,
    partial_m0: tl.constexpr,
    partial_m1: tl.constexpr,
    output_s0: tl.constexpr,
    output_s1: tl.constexpr,
    output_s2: tl.constexpr,
    head_size: tl.constexpr,
    num_partitions: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    parts = tl.arange(0, num_partitions)
    dimensions = tl.arange(0, block_d)
    scalar_offsets = row * partial_m0 + head * partial_m1 + parts
    maxima = tl.load(partial_maximum + scalar_offsets)
    denominators = tl.load(partial_denominator + scalar_offsets)
    outputs = tl.load(
        partial_output
        + row * partial_s0
        + head * partial_s1
        + parts[:, None] * partial_s2
        + dimensions[None, :],
        mask=dimensions[None, :] < head_size,
        other=0,
    )
    maximum = tl.max(maxima, 0)
    weights = tl.exp(maxima - maximum)
    denominator = tl.sum(denominators * weights, 0)
    numerator = tl.sum(outputs * weights[:, None], 0)
    tl.store(
        output + row * output_s0 + head * output_s1 + dimensions * output_s2,
        numerator / tl.maximum(denominator, 1.0e-20),
        mask=dimensions < head_size,
    )


class RetroSpecGPUNativeIndex(RetroSpecIndexBase):
    """Per-request GPU summaries and reverse logical-token layouts."""

    def __init__(
        self,
        *,
        block_size: int,
        num_speculative_tokens: int,
        retrieval_ratio: float,
        estimation_ratio: float,
        prefill_segment_size_tokens: int,
        generation_update_interval: int,
        blocks_per_cluster: int,
        num_kmeans_iterations: int,
        draft_rank_dtype: str = "native",
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        super().__init__(
            block_size, num_speculative_tokens, retrieval_ratio, estimation_ratio
        )
        self.num_speculative_tokens = num_speculative_tokens
        self.prefill_segment_size_tokens = prefill_segment_size_tokens
        self.generation_update_interval = generation_update_interval
        self.tokens_per_cluster = blocks_per_cluster * block_size
        self.num_kmeans_iterations = num_kmeans_iterations
        if draft_rank_dtype not in ("int8", "native"):
            raise ValueError("draft_rank_dtype must be 'int8' or 'native'")
        self.draft_rank_dtype = draft_rank_dtype
        self.performance_stats = performance_stats
        self._records: dict[str, dict[str, _NativeLayerRecord]] = {}
        self._staged: dict[tuple[str, str], _NativeLayerRecord | None] = {}
        self._request_ids: tuple[str, ...] = ()
        self._batch_layers: dict[str, _NativeBatchLayer] = {}
        self._active_cluster_counts: dict[str, int] = {}
        self._workspaces: dict[str, _NativeBatchLayer] = {}
        self._workspace_generations: dict[str, int] = {}
        self._arena_slots: dict[str, dict[str, int]] = {}
        self._arena_free_slots: dict[str, list[int]] = {}
        self._arena_versions: dict[str, dict[str, _NativeLayerRecord]] = {}
        self._plans: dict[str, dict[int, _NativeRankedPlan]] = {}
        self._plan_workspaces: dict[str, _NativePlanWorkspace] = {}
        self._active_plan_workspaces: dict[str, _NativePlanWorkspace] = {}
        self._logit_workspaces: dict[str, torch.Tensor] = {}
        self._shared_attention_workspace: _NativeAttentionWorkspace | None = None
        self._request_slots: torch.Tensor | None = None

    @property
    def has_staged_updates(self) -> bool:
        return bool(self._staged)

    @staticmethod
    def _append_record(
        record: _NativeLayerRecord, part: _NativeLayerRecord
    ) -> _NativeLayerRecord:
        old_clusters = record.counts.shape[1]
        old_tokens = record.token_indices.shape[1]
        total_clusters = old_clusters + part.counts.shape[1]
        total_tokens = old_tokens + part.token_indices.shape[1]
        old_key_storage = (
            record.key_storage if record.key_storage is not None else record.keys
        )
        old_value_storage = (
            record.value_storage if record.value_storage is not None else record.values
        )
        old_count_storage = (
            record.count_storage if record.count_storage is not None else record.counts
        )
        old_token_storage = (
            record.token_storage
            if record.token_storage is not None
            else record.token_indices
        )
        old_offset_storage = (
            record.offset_storage
            if record.offset_storage is not None
            else record.cluster_offsets
        )
        cluster_capacity = triton.next_power_of_2(
            max(total_clusters, old_key_storage.shape[1])
        )
        token_capacity = triton.next_power_of_2(
            max(total_tokens, old_token_storage.shape[1])
        )
        if old_key_storage.shape[1] < total_clusters:
            key_storage = record.keys.new_empty(
                record.keys.shape[0], cluster_capacity, record.keys.shape[2]
            )
            value_storage = record.values.new_empty(
                record.values.shape[0], cluster_capacity, record.values.shape[2]
            )
            count_storage = record.counts.new_empty(
                record.counts.shape[0], cluster_capacity
            )
            offset_storage = record.cluster_offsets.new_empty(
                record.cluster_offsets.shape[0], cluster_capacity + 1
            )
            key_storage[:, :old_clusters].copy_(record.keys)
            value_storage[:, :old_clusters].copy_(record.values)
            count_storage[:, :old_clusters].copy_(record.counts)
            offset_storage[:, : old_clusters + 1].copy_(record.cluster_offsets)
        else:
            key_storage = old_key_storage
            value_storage = old_value_storage
            count_storage = old_count_storage
            offset_storage = old_offset_storage
        if old_token_storage.shape[1] < total_tokens:
            token_storage = record.token_indices.new_empty(
                record.token_indices.shape[0], token_capacity
            )
            token_storage[:, :old_tokens].copy_(record.token_indices)
        else:
            token_storage = old_token_storage
        key_storage[:, old_clusters:total_clusters].copy_(part.keys)
        value_storage[:, old_clusters:total_clusters].copy_(part.values)
        count_storage[:, old_clusters:total_clusters].copy_(part.counts)
        token_storage[:, old_tokens:total_tokens].copy_(part.token_indices)
        offset_storage[:, old_clusters + 1 : total_clusters + 1].copy_(
            part.cluster_offsets[:, 1:] + old_tokens
        )
        return _NativeLayerRecord(
            part.indexed_end,
            key_storage[:, :total_clusters],
            value_storage[:, :total_clusters],
            count_storage[:, :total_clusters],
            token_storage[:, :total_tokens],
            offset_storage[:, : total_clusters + 1],
            key_storage,
            value_storage,
            count_storage,
            token_storage,
            offset_storage,
        )

    def has_staged_request_layer(self, layer_name: str, request_id: str) -> bool:
        return (layer_name, request_id) in self._staged

    def _desired_end(
        self,
        seq_len: int,
        indexed_end: int,
        is_prefill: bool,
        prefill_complete: bool,
    ) -> int:
        stable_end = (
            max(seq_len // self.block_size - self.num_recent_blocks, 1)
            * self.block_size
        )
        start = self.block_size if stable_end < indexed_end else indexed_end
        quantum = (
            self.tokens_per_cluster
            if is_prefill and prefill_complete
            else self.prefill_segment_size_tokens
            if is_prefill
            else self.generation_update_interval
        )
        return start + max(stable_end - start, 0) // quantum * quantum

    def needs_update(
        self,
        request_id: str,
        seq_len: int,
        layer_names: Sequence[str],
        is_prefill: bool,
        prefill_complete: bool = False,
    ) -> bool:
        for layer_name in layer_names:
            record = self._records.get(layer_name, {}).get(request_id)
            current_end = self.block_size if record is None else record.indexed_end
            if (
                self._desired_end(seq_len, current_end, is_prefill, prefill_complete)
                != current_end
            ):
                return True
        return False

    def build_or_update(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        is_prefill: Sequence[bool],
        rows: Sequence[int],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        prefill_complete: Sequence[bool] | None = None,
    ) -> None:
        if prefill_complete is None:
            prefill_complete = (False,) * len(request_ids)
        if key_cache.device.type != "cuda":
            raise ValueError("GPU-native RetroSpec requires CUDA KV cache")
        for row in rows:
            request_id = request_ids[row]
            key = (layer_name, request_id)
            if key in self._staged:
                raise RuntimeError(
                    "A GPU-native request/layer update is already staged"
                )
            record = self._records.get(layer_name, {}).get(request_id)
            current_end = self.block_size if record is None else record.indexed_end
            desired_end = self._desired_end(
                int(seq_lens[row]),
                current_end,
                bool(is_prefill[row]),
                bool(prefill_complete[row]),
            )
            if desired_end < current_end:
                record = None
                current_end = self.block_size
            if desired_end <= current_end:
                if request_id in self._records.get(layer_name, {}):
                    self._staged[key] = None
                continue

            phase_start = current_end
            parts: list[_NativeLayerRecord] = []
            regular = (
                self.prefill_segment_size_tokens
                if is_prefill[row]
                else self.generation_update_interval
            )
            while phase_start < desired_end:
                phase_size = min(regular, desired_end - phase_start)
                if phase_size % self.tokens_per_cluster:
                    raise RuntimeError(
                        "GPU-native clustering phase is not cluster-aligned"
                    )
                with self._cuda_timer("gpu_native_index_gather"):
                    logical_blocks = torch.arange(
                        phase_start // self.block_size,
                        (phase_start + phase_size) // self.block_size,
                        dtype=torch.int64,
                        device=block_table.device,
                    )
                    physical_blocks = (
                        block_table[row].index_select(0, logical_blocks).long()
                    )
                    num_heads, head_size = key_cache.shape[2:]
                    token_keys = (
                        key_cache.index_select(0, physical_blocks)
                        .reshape(phase_size, num_heads, head_size)
                        .transpose(0, 1)
                        .contiguous()
                    )
                    token_values = (
                        value_cache.index_select(0, physical_blocks)
                        .reshape(phase_size, num_heads, head_size)
                        .transpose(0, 1)
                        .contiguous()
                    )
                with self._cuda_timer("gpu_native_index_kmeans"):
                    clustered = segmented_kmeans(
                        token_keys,
                        token_values,
                        phase_size,
                        self.tokens_per_cluster,
                        self.num_kmeans_iterations,
                    )
                with self._cuda_timer("gpu_native_index_reverse"):
                    counts = clustered.cluster_sizes.to(torch.int32)
                    offsets = F.pad(counts.cumsum(1, dtype=torch.int32), (1, 0))
                    token_indices = torch.empty_like(clustered.assignments)
                    _scatter_cluster_tokens_kernel[
                        (num_heads, triton.cdiv(phase_size, 256))
                    ](
                        clustered.assignments,
                        clustered.token_offsets_in_cluster,
                        offsets,
                        token_indices,
                        clustered.assignments.stride(0),
                        clustered.token_offsets_in_cluster.stride(0),
                        offsets.stride(0),
                        token_indices.stride(0),
                        phase_size,
                        phase_start,
                        256,
                    )
                parts.append(
                    _NativeLayerRecord(
                        phase_start + phase_size,
                        clustered.cluster_keys,
                        clustered.cluster_values,
                        counts,
                        token_indices,
                        offsets,
                    )
                )
                phase_start += phase_size

            with self._cuda_timer("gpu_native_index_append"):
                for part in parts:
                    if record is None:
                        record = part
                    else:
                        record = self._append_record(record, part)
            assert record is not None
            self._staged[key] = record
        return None

    def flush_staged_updates(self) -> None:
        for (layer_name, request_id), record in self._staged.items():
            if record is None:
                self._records.get(layer_name, {}).pop(request_id, None)
            else:
                self._records.setdefault(layer_name, {})[request_id] = record
        self._staged.clear()

    def discard_staged_updates(self) -> None:
        self._staged.clear()

    def get_fully_stored_indexed_end(
        self, request_id: str, layer_names: Sequence[str]
    ) -> int:
        return (
            min(
                (
                    self._records.get(name, {}).get(request_id).indexed_end
                    if request_id in self._records.get(name, {})
                    else self.block_size
                )
                for name in layer_names
            )
            if layer_names
            else self.block_size
        )

    def remove_requests(self, request_ids: Sequence[str]) -> None:
        removed = set(request_ids)
        for layer_name, records in self._records.items():
            for request_id in removed:
                records.pop(request_id, None)
                slot = self._arena_slots.get(layer_name, {}).pop(request_id, None)
                if slot is not None:
                    self._arena_free_slots.setdefault(layer_name, []).append(slot)
                    self._arena_versions.get(layer_name, {}).pop(request_id, None)
                    workspace = self._workspaces.get(layer_name)
                    if workspace is not None:
                        workspace.counts[slot].zero_()
                        workspace.indexed_ends[slot] = self.block_size
        for key in tuple(self._staged):
            if key[1] in removed:
                del self._staged[key]

    def has_cluster_pages(self, layer_name: str, request_ids: Sequence[str]) -> bool:
        return any(
            (record := self._records.get(layer_name, {}).get(request_id)) is not None
            and record.counts.shape[1] > 0
            for request_id in request_ids
        )

    def begin_proposal(self, request_ids: Sequence[str]) -> None:
        if self._request_ids or self._staged:
            raise RuntimeError("GPU-native proposal cannot overlap another transaction")
        self._request_ids = tuple(request_ids)
        self._batch_layers.clear()
        self._active_cluster_counts.clear()
        self._plans.clear()
        self._active_plan_workspaces.clear()

    def end_proposal(self) -> None:
        self._request_ids = ()
        self._batch_layers.clear()
        self._active_cluster_counts.clear()
        self._plans.clear()
        self._active_plan_workspaces.clear()

    def _batch_layer(
        self, layer_name: str, device: torch.device, dtype: torch.dtype
    ) -> _NativeBatchLayer:
        cached = self._batch_layers.get(layer_name)
        if cached is not None:
            return cached
        layer_records = self._records.get(layer_name, {})
        records = [layer_records.get(request_id) for request_id in self._request_ids]
        present = next((record for record in records if record is not None), None)
        if present is None:
            raise RuntimeError("GPU-native layer has no cluster index")
        heads, _, head_size = present.keys.shape
        max_clusters = max(record.counts.shape[1] for record in layer_records.values())
        max_tokens = max(
            record.token_indices.shape[1] for record in layer_records.values()
        )
        active_clusters = max(
            record.counts.shape[1] if record is not None else 0 for record in records
        )
        active_tokens = max(
            record.token_indices.shape[1] if record is not None else 0
            for record in records
        )
        batch = len(records)
        slots = self._arena_slots.setdefault(layer_name, {})
        free_slots = self._arena_free_slots.setdefault(layer_name, [])
        for request_id in layer_records:
            if request_id not in slots:
                slots[request_id] = (
                    free_slots.pop()
                    if free_slots
                    else max(slots.values(), default=0) + 1
                )
        required_slots = max(slots.values(), default=0) + 1
        workspace = self._workspaces.get(layer_name)
        if (
            workspace is None
            or workspace.keys.device.type != device.type
            or (
                device.index is not None and workspace.keys.device.index != device.index
            )
            or workspace.keys.dtype != dtype
            or workspace.keys.shape[0] < required_slots
            or workspace.keys.shape[1] != heads
            or workspace.keys.shape[2] < max_clusters
            or workspace.keys.shape[3] != head_size
            or workspace.token_indices.shape[2] < max_tokens
            or (
                self.draft_rank_dtype == "int8"
                and dtype in (torch.float16, torch.bfloat16)
                and workspace.quantized_keys is None
            )
        ):
            batch_capacity = triton.next_power_of_2(
                max(
                    required_slots,
                    workspace.keys.shape[0] if workspace is not None else 2,
                )
            )
            cluster_capacity = triton.next_power_of_2(
                max(
                    max_clusters,
                    workspace.keys.shape[2] if workspace is not None else 1,
                )
            )
            token_capacity = triton.next_power_of_2(
                max(
                    max_tokens,
                    workspace.token_indices.shape[2] if workspace is not None else 1,
                )
            )
            keys = torch.empty(
                batch_capacity,
                heads,
                cluster_capacity,
                head_size,
                dtype=dtype,
                device=device,
            )
            workspace = _NativeBatchLayer(
                keys=keys,
                values=torch.empty_like(keys),
                counts=torch.empty(
                    batch_capacity,
                    heads,
                    cluster_capacity,
                    dtype=torch.int32,
                    device=device,
                ),
                token_indices=torch.empty(
                    batch_capacity,
                    heads,
                    token_capacity,
                    dtype=torch.int32,
                    device=device,
                ),
                cluster_offsets=torch.empty(
                    batch_capacity,
                    heads,
                    cluster_capacity + 1,
                    dtype=torch.int32,
                    device=device,
                ),
                indexed_ends=torch.empty(
                    batch_capacity,
                    dtype=torch.int32,
                    device=device,
                ),
                quantized_keys=(
                    torch.empty(
                        batch_capacity,
                        heads,
                        cluster_capacity,
                        head_size,
                        dtype=torch.int8,
                        device=device,
                    )
                    if self.draft_rank_dtype == "int8"
                    and dtype in (torch.float16, torch.bfloat16)
                    else None
                ),
                key_scales=(
                    torch.empty(
                        batch_capacity,
                        heads,
                        cluster_capacity,
                        dtype=torch.float32,
                        device=device,
                    )
                    if self.draft_rank_dtype == "int8"
                    and dtype in (torch.float16, torch.bfloat16)
                    else None
                ),
            )
            self._workspaces[layer_name] = workspace
            self._workspace_generations[layer_name] = (
                self._workspace_generations.get(layer_name, 0) + 1
            )
            self._arena_versions[layer_name] = {}
            workspace.counts[0].zero_()
            workspace.indexed_ends[0] = self.block_size
        versions = self._arena_versions.setdefault(layer_name, {})
        for request_id, record in layer_records.items():
            if versions.get(request_id) is record:
                continue
            slot = slots[request_id]
            clusters = record.counts.shape[1]
            num_tokens = record.token_indices.shape[1]
            workspace.counts[slot].zero_()
            workspace.keys[slot, :, :clusters].copy_(record.keys)
            workspace.values[slot, :, :clusters].copy_(record.values)
            if workspace.quantized_keys is not None:
                assert workspace.key_scales is not None
                with self._cuda_timer("gpu_native_index_quantize"):
                    centered = record.keys.float()
                    scales = (centered.abs().amax(dim=-1) / 127.0).clamp_min(1.0e-8)
                    workspace.quantized_keys[slot, :, :clusters].copy_(
                        (centered / scales.unsqueeze(-1))
                        .round()
                        .clamp(-127, 127)
                        .to(torch.int8)
                    )
                    workspace.key_scales[slot, :, :clusters].copy_(scales)
            workspace.counts[slot, :, :clusters].copy_(record.counts)
            workspace.token_indices[slot, :, :num_tokens].copy_(record.token_indices)
            workspace.cluster_offsets[slot, :, : clusters + 1].copy_(
                record.cluster_offsets
            )
            workspace.indexed_ends[slot] = record.indexed_end
            resident_record = _NativeLayerRecord(
                record.indexed_end,
                workspace.keys[slot, :, :clusters],
                workspace.values[slot, :, :clusters],
                workspace.counts[slot, :, :clusters],
                workspace.token_indices[slot, :, :num_tokens],
                workspace.cluster_offsets[slot, :, : clusters + 1],
            )
            layer_records[request_id] = resident_record
            versions[request_id] = resident_record
        request_slots = torch.tensor(
            [slots.get(request_id, 0) for request_id in self._request_ids],
            dtype=torch.int32,
            device=device,
        )
        layout = _NativeBatchLayer(
            workspace.keys[:, :, :active_clusters],
            workspace.values[:, :, :active_clusters],
            workspace.counts[:, :, :active_clusters],
            workspace.token_indices[:, :, :active_tokens],
            workspace.cluster_offsets[:, :, : active_clusters + 1],
            workspace.indexed_ends,
            request_slots,
            workspace.quantized_keys[:, :, :active_clusters]
            if workspace.quantized_keys is not None
            else None,
            workspace.key_scales[:, :, :active_clusters]
            if workspace.key_scales is not None
            else None,
        )
        self._batch_layers[layer_name] = layout
        self._active_cluster_counts[layer_name] = active_clusters
        self._prepare_plan_workspace(layer_name, layout, batch, device)
        return layout

    def _prepare_plan_workspace(
        self,
        layer_name: str,
        layout: _NativeBatchLayer,
        batch: int,
        device: torch.device,
    ) -> None:
        heads, clusters = layout.counts.shape[1:]
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        width = sparse_width + estimation_width
        workspace = self._plan_workspaces.get(layer_name)
        if (
            workspace is None
            or workspace.ranked.device.type != device.type
            or (
                device.index is not None
                and workspace.ranked.device.index != device.index
            )
            or workspace.ranked.shape[1] < batch
            or workspace.ranked.shape[2] != heads
            or workspace.ranked.shape[3] < width
            or workspace.scores.shape[2] < clusters
        ):
            batch_capacity = layout.keys.shape[0]
            plan_shape = (self.num_speculative_tokens, batch_capacity, heads, width)
            workspace = _NativePlanWorkspace(
                ranked=torch.empty(plan_shape, dtype=torch.int64, device=device),
                candidate_counts=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    heads,
                    dtype=torch.int32,
                    device=device,
                ),
                sparse_mass=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    dtype=torch.float32,
                    device=device,
                ),
                verification_mass=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    dtype=torch.float32,
                    device=device,
                ),
                expanded_mass=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    dtype=torch.float32,
                    device=device,
                ),
                scores=torch.empty(
                    batch_capacity,
                    heads,
                    clusters,
                    dtype=torch.float32,
                    device=device,
                ),
                topk_values=torch.empty(
                    batch_capacity,
                    heads,
                    width,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            self._plan_workspaces[layer_name] = workspace
        workspace.candidate_counts[:, :batch].zero_()
        workspace.sparse_mass[:, :batch].fill_(1.0)
        workspace.verification_mass[:, :batch].fill_(1.0)
        workspace.expanded_mass[:, :batch].fill_(1.0)
        self._active_plan_workspaces[layer_name] = workspace

    def _compute_rank_draft(
        self,
        query: torch.Tensor,
        scale: float,
        active_mask: torch.Tensor,
        layout: _NativeBatchLayer,
        output: _NativeRankedPlan | None = None,
        score_buffer: torch.Tensor | None = None,
        topk_values: torch.Tensor | None = None,
        logits_buffer: torch.Tensor | None = None,
        request_rows: torch.Tensor | None = None,
        exact: bool = False,
    ) -> _NativeRankedPlan:
        batch, query_heads, head_size = query.shape
        if request_rows is not None:
            if request_rows.shape != (batch,):
                raise ValueError("Mapped rank request rows must match the query batch")
            if request_rows.device != query.device or request_rows.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("Mapped rank request rows must be GPU integers")
        mapped_rows = request_rows
        if layout.request_slots is not None:
            mapped_rows = (
                layout.request_slots.index_select(0, request_rows.long())
                if request_rows is not None
                else layout.request_slots[:batch]
            )
        heads, clusters = layout.counts.shape[1:]
        group_size = query_heads // heads
        if (
            clusters > 8192
            or triton.next_power_of_2(group_size) * triton.next_power_of_2(clusters)
            > 65536
        ):
            mapped_layout = layout
            if mapped_rows is not None:
                mapped_layout = _NativeBatchLayer(
                    keys=layout.keys.index_select(0, mapped_rows.long()),
                    values=layout.values,
                    counts=layout.counts.index_select(0, mapped_rows.long()),
                    token_indices=layout.token_indices,
                    cluster_offsets=layout.cluster_offsets,
                    indexed_ends=layout.indexed_ends,
                )
            result = self._compute_rank_draft_torch(
                query, scale, active_mask, mapped_layout
            )
            if output is None:
                return result
            output.ranked.copy_(result.ranked)
            output.candidate_counts.copy_(result.candidate_counts)
            output.sparse_mass.copy_(result.sparse_mass)
            output.verification_mass.copy_(result.verification_mass)
            output.expanded_mass.copy_(result.expanded_mass)
            return output
        grouped_query = query.reshape(batch, heads, group_size, head_size)
        if (
            not exact
            and self.draft_rank_dtype == "int8"
            and layout.quantized_keys is not None
            and layout.key_scales is not None
            and query.dtype in (torch.float16, torch.bfloat16)
            and group_size <= 16
            and 32 <= head_size <= 256
        ):
            logits = logits_buffer
            if logits is None:
                logits = torch.empty(
                    batch,
                    heads,
                    group_size,
                    clusters,
                    dtype=torch.float32,
                    device=query.device,
                )
            if mapped_rows is None:
                mapped_rows = torch.arange(
                    batch, dtype=torch.int32, device=query.device
                )
            with self._cuda_timer("gpu_native_rank_dot"):
                _cluster_rank_int8_kernel[(triton.cdiv(clusters, 128), batch * heads)](
                    grouped_query,
                    layout.quantized_keys,
                    layout.key_scales,
                    logits,
                    mapped_rows,
                    *grouped_query.stride()[:3],
                    *layout.quantized_keys.stride()[:3],
                    *layout.key_scales.stride()[:2],
                    *logits.stride()[:3],
                    heads,
                    group_size,
                    clusters,
                    head_size,
                    128,
                    max(32, triton.next_power_of_2(head_size)),
                    num_warps=4,
                )
        elif (
            query.dtype in (torch.float16, torch.bfloat16)
            and layout.keys.dtype == query.dtype
            and group_size <= 16
            and head_size <= 256
        ):
            logits = logits_buffer
            if logits is None:
                logits = torch.empty(
                    batch,
                    heads,
                    group_size,
                    clusters,
                    dtype=torch.float32,
                    device=query.device,
                )
            with self._cuda_timer("gpu_native_rank_dot"):
                _cluster_rank_dot_kernel[(triton.cdiv(clusters, 128), batch * heads)](
                    grouped_query,
                    layout.keys,
                    logits,
                    mapped_rows if mapped_rows is not None else query,
                    *grouped_query.stride()[:3],
                    *layout.keys.stride()[:3],
                    *logits.stride()[:3],
                    heads,
                    group_size,
                    clusters,
                    head_size,
                    128,
                    triton.next_power_of_2(head_size),
                    mapped_rows is not None,
                    num_warps=4,
                )
        else:
            keys = (
                layout.keys.index_select(0, mapped_rows.long())
                if mapped_rows is not None
                else layout.keys[:batch]
            )
            logits = torch.einsum(
                "bhgd,bhcd->bhgc", grouped_query.float(), keys.float()
            )
        scores = score_buffer
        if scores is None:
            scores = torch.empty(
                batch, heads, clusters, dtype=torch.float32, device=query.device
            )
        candidates = (
            output.candidate_counts
            if output is not None
            else torch.empty(batch, heads, dtype=torch.int32, device=query.device)
        )
        _cluster_scores_kernel[(batch, heads)](
            logits,
            layout.counts,
            active_mask,
            scores,
            candidates,
            mapped_rows if mapped_rows is not None else query,
            *logits.stride(),
            layout.counts.stride(0),
            layout.counts.stride(1),
            scores.stride(0),
            scores.stride(1),
            candidates.stride(0),
            group_size,
            clusters,
            scale,
            triton.next_power_of_2(group_size),
            triton.next_power_of_2(clusters),
            mapped_rows is not None,
            num_warps=8 if clusters <= 2048 else 16,
        )
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        if output is None:
            with self._cuda_timer("gpu_native_rank_topk"):
                ranked_scores, ranked = torch.topk(
                    scores, k=sparse_width + estimation_width, dim=2, sorted=True
                )
            sparse_mass = torch.empty(batch, dtype=torch.float32, device=query.device)
            verification_mass = torch.empty_like(sparse_mass)
            expanded_mass = torch.empty_like(sparse_mass)
        else:
            assert topk_values is not None
            with self._cuda_timer("gpu_native_rank_topk"):
                ranked_scores, ranked = torch.topk(
                    scores,
                    k=sparse_width + estimation_width,
                    dim=2,
                    sorted=True,
                    out=(topk_values, output.ranked),
                )
            sparse_mass = output.sparse_mass
            verification_mass = output.verification_mass
            expanded_mass = output.expanded_mass
        _cluster_mass_kernel[(batch,)](
            ranked_scores,
            candidates,
            active_mask,
            sparse_mass,
            verification_mass,
            expanded_mass,
            *ranked_scores.stride(),
            candidates.stride(0),
            heads,
            ranked_scores.shape[2],
            self.retrieval_ratio,
            self.estimation_ratio,
            triton.next_power_of_2(heads),
            triton.next_power_of_2(ranked_scores.shape[2]),
            num_warps=4,
        )
        return output or _NativeRankedPlan(
            ranked, candidates, sparse_mass, verification_mass, expanded_mass
        )

    def _compute_rank_draft_torch(
        self,
        query: torch.Tensor,
        scale: float,
        active_mask: torch.Tensor,
        layout: _NativeBatchLayer,
    ) -> _NativeRankedPlan:
        batch, query_heads, head_size = query.shape
        heads, clusters = layout.counts.shape[1:]
        group_size = query_heads // heads
        logits = (
            torch.einsum(
                "bhgd,bhcd->bhgc",
                query.reshape(batch, heads, group_size, head_size).float(),
                (
                    layout.keys.index_select(0, layout.request_slots[:batch].long())
                    if layout.request_slots is not None
                    else layout.keys[:batch]
                ).float(),
            )
            * scale
        )
        counts = (
            layout.counts.index_select(0, layout.request_slots[:batch].long())
            if layout.request_slots is not None
            else layout.counts[:batch]
        )
        valid = (counts > 0) & active_mask[:, None, None]
        logits += counts.clamp_min(1).float().log()[:, :, None, :]
        probabilities = torch.softmax(
            logits.masked_fill(~valid[:, :, None, :], -1e30), dim=-1
        )
        scores = probabilities.mean(2).masked_fill(~valid, float("-inf"))
        candidates = valid.sum(2, dtype=torch.int32)
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        ranked = torch.topk(
            scores, k=sparse_width + estimation_width, dim=2, sorted=True
        ).indices.to(torch.int32)
        retrieval = torch.ceil(candidates.float() * self.retrieval_ratio).int()
        estimation = torch.ceil(candidates.float() * self.estimation_ratio).int()
        estimation = torch.minimum(estimation, candidates - retrieval)
        rank_ids = torch.arange(ranked.shape[2], device=query.device)
        ranked_scores = scores.gather(2, ranked.long()).clamp_min(0)
        sparse_mass = (
            (ranked_scores * (rank_ids < retrieval[:, :, None])).sum(2).mean(1)
        )
        verification_end = torch.minimum(retrieval * 2, retrieval + estimation)
        verification_mass = (
            (ranked_scores * (rank_ids < verification_end[:, :, None])).sum(2).mean(1)
        )
        expanded_end = torch.minimum(retrieval * 3, retrieval + estimation)
        expanded_mass = (
            (ranked_scores * (rank_ids < expanded_end[:, :, None])).sum(2).mean(1)
        )
        sparse_mass = torch.where(active_mask, sparse_mass, 1.0)
        verification_mass = torch.where(active_mask, verification_mass, 1.0)
        expanded_mass = torch.where(active_mask, expanded_mass, 1.0)
        return _NativeRankedPlan(
            ranked, candidates, sparse_mass, verification_mass, expanded_mass
        )

    def _rank_draft(
        self,
        layer_name: str,
        query: torch.Tensor,
        scale: float,
        active_mask: torch.Tensor,
        step: int,
        layout: _NativeBatchLayer,
    ) -> _NativeRankedPlan:
        batch = query.shape[0]
        workspace = self._active_plan_workspaces[layer_name]
        heads, clusters = layout.counts.shape[1:]
        group_size = query.shape[1] // heads
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        width = sparse_width + estimation_width
        logits = None
        if (
            clusters <= 8192
            and triton.next_power_of_2(group_size) * triton.next_power_of_2(clusters)
            <= 65536
            and query.dtype in (torch.float16, torch.bfloat16)
            and layout.keys.dtype == query.dtype
            and group_size <= 16
            and query.shape[2] <= 256
        ):
            logits = self._logit_workspaces.get(layer_name)
            if (
                logits is None
                or logits.shape[0] < batch
                or logits.shape[1] != heads
                or logits.shape[2] != group_size
                or logits.shape[3] < clusters
                or logits.device != query.device
            ):
                logits = torch.empty(
                    batch,
                    heads,
                    group_size,
                    clusters,
                    dtype=torch.float32,
                    device=query.device,
                )
                self._logit_workspaces[layer_name] = logits
        slot = _NativeRankedPlan(
            workspace.ranked[step, :batch, :, :width],
            workspace.candidate_counts[step, :batch],
            workspace.sparse_mass[step, :batch],
            workspace.verification_mass[step, :batch],
            workspace.expanded_mass[step, :batch],
        )
        plan = self._compute_rank_draft(
            query,
            scale,
            active_mask,
            layout,
            output=slot,
            score_buffer=workspace.scores[:batch, :, :clusters],
            topk_values=workspace.topk_values[:batch, :, :width],
            logits_buffer=logits[:batch, :, :, :clusters]
            if logits is not None
            else None,
        )
        self._plans.setdefault(layer_name, {})[step] = plan
        return plan

    def _rank_parallel_bonus(
        self,
        layer_name: str,
        query: torch.Tensor,
        scale: float,
        request_rows: torch.Tensor,
        token_steps: torch.Tensor,
        active_mask: torch.Tensor,
        layout: _NativeBatchLayer,
    ) -> _NativeRankedPlan:
        if (
            request_rows.shape != (query.shape[0],)
            or token_steps.shape != request_rows.shape
        ):
            raise ValueError("Bonus plan indices must match the query batch")
        if token_steps.device != query.device or token_steps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Bonus plan steps must be GPU integers")
        plan = self._compute_rank_draft(
            query, scale, active_mask, layout, request_rows=request_rows, exact=True
        )
        workspace = self._active_plan_workspaces[layer_name]
        rows = request_rows.long()
        steps = token_steps.long()
        workspace.ranked[steps, rows] = plan.ranked
        workspace.candidate_counts[steps, rows] = plan.candidate_counts
        workspace.sparse_mass[steps, rows] = plan.sparse_mass
        workspace.verification_mass[steps, rows] = plan.verification_mass
        workspace.expanded_mass[steps, rows] = plan.expanded_mass
        return plan

    def _attention_workspace(
        self, query: torch.Tensor, partitions: int
    ) -> _NativeAttentionWorkspace:
        workspace = self._shared_attention_workspace
        rows, heads, head_size = query.shape
        if (
            workspace is None
            or workspace.output.shape[0] < rows
            or workspace.output.shape[1] != heads
            or workspace.output.shape[2] < partitions
            or workspace.output.shape[3] != head_size
            or workspace.output.device != query.device
        ):
            row_capacity = rows
            if workspace is not None:
                previous_rows = workspace.output.shape[0]
                row_capacity = max(
                    rows, previous_rows * 2 if rows > previous_rows else previous_rows
                )
            workspace = _NativeAttentionWorkspace(
                output=torch.empty(
                    row_capacity,
                    heads,
                    partitions,
                    head_size,
                    device=query.device,
                    dtype=torch.float32,
                ),
                maximum=torch.empty(
                    row_capacity,
                    heads,
                    partitions,
                    device=query.device,
                    dtype=torch.float32,
                ),
                denominator=torch.empty(
                    row_capacity,
                    heads,
                    partitions,
                    device=query.device,
                    dtype=torch.float32,
                ),
            )
            self._shared_attention_workspace = workspace
        return workspace

    def forward(
        self,
        *,
        layer_name: str,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        active_mask: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        step: int,
        expanded: bool = False,
        sparse_verify: bool = False,
        request_indices: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        bonus_start_index: int | None = None,
    ) -> torch.Tensor:
        if not self._request_ids:
            raise RuntimeError("GPU-native attention requires an active proposal")
        with self._cuda_timer("gpu_native_batch_layout"):
            layout = self._batch_layer(layer_name, query.device, query.dtype)
        draft_attention = request_indices is None
        if draft_attention:
            attention_timer_name = "gpu_native_draft_attention"
            request_slots = self._request_slots
            if (
                request_slots is None
                or request_slots.shape[0] < query.shape[0]
                or request_slots.device != query.device
            ):
                request_slots = torch.arange(
                    query.shape[0], dtype=torch.int32, device=query.device
                )
                self._request_slots = request_slots
            request_indices = request_slots[: query.shape[0]]
            with self._cuda_timer("gpu_native_rank"):
                plan = self._rank_draft(
                    layer_name, query, scale, active_mask, step, layout
                )
            mass = plan.sparse_mass
        else:
            attention_timer_name = (
                "gpu_native_expanded_verify_attention"
                if expanded
                else "gpu_native_sparse_verify_attention"
                if sparse_verify
                else "gpu_native_verify_attention"
            )
            if token_indices is None:
                raise ValueError("Verification requires draft token indices")
            if bonus_start_index is not None and (
                not sparse_verify or not 0 < bonus_start_index < query.shape[0]
            ):
                raise ValueError("Bonus rows require a sparse verification suffix")
            # Draft rows reuse their ranking; only bonus rows lack a plan.
            if sparse_verify and bonus_start_index is not None:
                with self._cuda_timer("gpu_native_sparse_rank"):
                    self._rank_parallel_bonus(
                        layer_name,
                        query[bonus_start_index:],
                        scale,
                        request_indices[bonus_start_index:],
                        token_indices[bonus_start_index:],
                        active_mask[bonus_start_index:],
                        layout,
                    )
            plans = self._plans.get(layer_name, {})
            if not plans:
                raise RuntimeError("Verification has no GPU-native draft plan")
            workspace = self._active_plan_workspaces[layer_name]
            request_rows = request_indices.long()
            draft_steps = token_indices.long()
            mass_table = (
                workspace.expanded_mass
                if expanded
                else workspace.verification_mass
                if sparse_verify
                else workspace.sparse_mass
            )
            selected_mass = mass_table[draft_steps, request_rows]
            plan = _NativeRankedPlan(
                workspace.ranked[draft_steps, request_rows],
                workspace.candidate_counts[draft_steps, request_rows],
                selected_mass,
                selected_mass,
                selected_mass,
            )
            mass = plan.sparse_mass

        heads = layout.counts.shape[1]
        if query.shape[1] % heads:
            raise ValueError("GPU-native query and KV heads are incompatible")
        cluster_count = self._active_cluster_counts[layer_name]
        num_partitions = (
            8 if cluster_count >= 4096 else 4 if cluster_count >= 1536 else 1
        )
        attention_workspace = (
            self._attention_workspace(query, num_partitions)
            if num_partitions > 1
            else None
        )
        partial_output = attention_workspace.output if attention_workspace else output
        partial_maximum = attention_workspace.maximum if attention_workspace else output
        partial_denominator = (
            attention_workspace.denominator if attention_workspace else output
        )
        queries_per_kv = query.shape[1] // heads
        grouped_attention = (
            sparse_verify
            and 4 <= queries_per_kv <= 16
            and query.dtype in (torch.float16, torch.bfloat16)
            and key_cache.dtype == query.dtype
            and value_cache.dtype == query.dtype
            and layout.keys.dtype == query.dtype
            and layout.values.dtype == query.dtype
            and 32 <= query.shape[2] <= 256
        )
        attention_kernel = (
            _native_grouped_sparse_attention_kernel
            if grouped_attention
            else _native_sparse_attention_kernel
        )
        attention_heads = heads if grouped_attention else query.shape[1]
        with self._cuda_timer(attention_timer_name):
            attention_kernel[(query.shape[0], attention_heads, num_partitions)](
                query,
                key_cache,
                value_cache,
                block_table,
                layout.token_indices,
                layout.cluster_offsets,
                layout.keys,
                layout.values,
                layout.counts,
                plan.ranked,
                plan.candidate_counts,
                layout.indexed_ends,
                seq_lens,
                request_indices,
                layout.request_slots
                if layout.request_slots is not None
                else request_indices,
                output,
                partial_output,
                partial_maximum,
                partial_denominator,
                *query.stride(),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                block_table.stride(0),
                block_table.stride(1),
                layout.token_indices.stride(0),
                layout.token_indices.stride(1),
                layout.cluster_offsets.stride(0),
                layout.cluster_offsets.stride(1),
                layout.keys.stride(0),
                layout.keys.stride(1),
                layout.counts.stride(0),
                layout.counts.stride(1),
                plan.ranked.stride(0),
                plan.ranked.stride(1),
                plan.candidate_counts.stride(0),
                *output.stride(),
                *partial_output.stride()[:3],
                partial_maximum.stride(0),
                partial_maximum.stride(1),
                scale,
                self.retrieval_ratio,
                self.estimation_ratio,
                expanded,
                sparse_verify,
                heads,
                queries_per_kv,
                layout.counts.shape[2],
                plan.ranked.shape[2],
                self.block_size,
                query.shape[2],
                triton.next_power_of_2(query.shape[2]),
                32,
                32,
                num_partitions,
            )
            if attention_workspace is not None:
                _merge_native_attention_partitions[(query.shape[0], query.shape[1])](
                    attention_workspace.output,
                    attention_workspace.maximum,
                    attention_workspace.denominator,
                    output,
                    *attention_workspace.output.stride()[:3],
                    attention_workspace.maximum.stride(0),
                    attention_workspace.maximum.stride(1),
                    *output.stride(),
                    query.shape[2],
                    num_partitions,
                    triton.next_power_of_2(query.shape[2]),
                )
        return mass

    def _cuda_timer(self, name: str):
        if self.performance_stats is None:
            return nullcontext()
        return self.performance_stats.cuda_timer(name)

    def configure_sparse_prefetch_wave(self, num_layers: int) -> None:
        del num_layers

    def flush_sparse_verification_prefetch(self) -> None:
        pass

    def begin_indexed_verification_transaction(self, *args: object) -> None:
        pass

    def end_indexed_verification_transaction(self) -> None:
        pass

    def prime_full_verification_pipeline(self, *args: object, **kwargs: object) -> bool:
        return False

    def prefetch_final_prefill_queries(self, *args: object, **kwargs: object) -> None:
        pass

    def close(self) -> None:
        self._records.clear()
        self._staged.clear()
        self._batch_layers.clear()
        self._workspaces.clear()
        self._workspace_generations.clear()
        self._arena_slots.clear()
        self._arena_free_slots.clear()
        self._arena_versions.clear()
        self._plans.clear()
        self._plan_workspaces.clear()
        self._active_plan_workspaces.clear()
        self._logit_workspaces.clear()
        self._request_slots = None
