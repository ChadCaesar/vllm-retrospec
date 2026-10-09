# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


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

    # Top-k cluster tokens are looked up through a compact reverse index.
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
