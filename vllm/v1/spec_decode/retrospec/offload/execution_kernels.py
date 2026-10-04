# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _multi_source_exact_partition_kernel(
    query,
    request_indices,
    plan_row_indices,
    key_cache,
    value_cache,
    block_table,
    token_indices,
    token_mask,
    page_token_counts,
    compact_page_counts,
    resident_page_ids,
    resident_key_pages,
    resident_value_pages,
    staging_page_ids,
    staging_key_pages,
    staging_value_pages,
    partial_output,
    partial_max,
    partial_sum,
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
    block_table_stride_0,
    block_table_stride_1,
    resident_key_stride_0,
    resident_key_stride_1,
    resident_key_stride_2,
    resident_value_stride_0,
    resident_value_stride_1,
    resident_value_stride_2,
    staging_key_stride_0,
    staging_key_stride_1,
    staging_key_stride_2,
    staging_value_stride_0,
    staging_value_stride_1,
    staging_value_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    stats_stride_0,
    stats_stride_1,
    stats_stride_2,
    scale,
    source_partition_offset,
    NUM_KV_HEADS: tl.constexpr,
    QUERIES_PER_KV_HEAD: tl.constexpr,
    MAX_PRIMARY_TOKENS: tl.constexpr,
    MAX_PAGE_SLOTS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    IDENTITY_REQUESTS: tl.constexpr,
    USE_PLAN_ROWS: tl.constexpr,
    HAS_RESIDENT: tl.constexpr,
    HAS_STAGING: tl.constexpr,
    COMPACT_PAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)

    primary_request_idx = query_idx
    if not IDENTITY_REQUESTS:
        primary_request_idx = tl.load(request_indices + query_idx)
    metadata_row_idx = primary_request_idx
    resolved_page_row_idx = primary_request_idx
    if USE_PLAN_ROWS:
        metadata_row_idx = tl.load(plan_row_indices + query_idx)
        resolved_page_row_idx = query_idx
    page_metadata_row_idx = metadata_row_idx
    if COMPACT_PAGES:
        page_metadata_row_idx = resolved_page_row_idx
    kv_head_idx = query_head_idx // QUERIES_PER_KV_HEAD

    dimension_offsets = tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < HEAD_SIZE
    query_offsets = (
        query_idx * query_stride_0
        + query_head_idx * query_stride_1
        + dimension_offsets * query_stride_2
    )
    query_vector = tl.load(query + query_offsets, mask=dimension_mask, other=0.0)
    query_vector = query_vector.to(tl.float32)

    running_max = float("-inf")
    running_sum = 0.0
    running_output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partition_start = (source_partition_offset + partition_idx) * PARTITION_SIZE

    for chunk_start in tl.range(0, PARTITION_SIZE, BLOCK_TOKENS):
        token_offsets = partition_start + chunk_start + tl.arange(0, BLOCK_TOKENS)
        source_valid = token_offsets < MAX_PRIMARY_TOKENS + MAX_PAGE_SLOTS * PAGE_SIZE

        primary_valid = source_valid & (token_offsets < MAX_PRIMARY_TOKENS)
        primary_metadata_offsets = (
            metadata_row_idx * NUM_KV_HEADS + kv_head_idx
        ) * MAX_PRIMARY_TOKENS + token_offsets
        primary_valid &= tl.load(
            token_mask + primary_metadata_offsets, mask=primary_valid, other=0
        ).to(tl.int1)
        logical_token_indices = tl.load(
            token_indices + primary_metadata_offsets, mask=primary_valid, other=0
        )
        logical_block_indices = logical_token_indices // PAGE_SIZE
        block_offsets = logical_token_indices % PAGE_SIZE
        physical_block_indices = tl.load(
            block_table
            + primary_request_idx * block_table_stride_0
            + logical_block_indices * block_table_stride_1,
            mask=primary_valid,
            other=0,
        ).to(tl.int64)

        primary_key_offsets = (
            physical_block_indices[:, None] * key_stride_0
            + block_offsets[:, None] * key_stride_1
            + kv_head_idx * key_stride_2
            + dimension_offsets[None, :] * key_stride_3
        )
        primary_value_offsets = (
            physical_block_indices[:, None] * value_stride_0
            + block_offsets[:, None] * value_stride_1
            + kv_head_idx * value_stride_2
            + dimension_offsets[None, :] * value_stride_3
        )
        vector_mask = primary_valid[:, None] & dimension_mask[None, :]
        key_vectors = tl.load(
            key_cache + primary_key_offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        value_vectors = tl.load(
            value_cache + primary_value_offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)

        page_token_offsets = token_offsets - MAX_PRIMARY_TOKENS
        page_slot_indices = page_token_offsets // PAGE_SIZE
        offsets_in_page = page_token_offsets % PAGE_SIZE
        page_valid = source_valid & (token_offsets >= MAX_PRIMARY_TOKENS)
        if COMPACT_PAGES:
            compact_count = tl.load(
                compact_page_counts + resolved_page_row_idx * NUM_KV_HEADS + kv_head_idx
            )
            page_valid &= page_slot_indices < compact_count
        page_count_offsets = (
            page_metadata_row_idx * NUM_KV_HEADS + kv_head_idx
        ) * MAX_PAGE_SLOTS + page_slot_indices
        resolved_page_offsets = (
            resolved_page_row_idx * NUM_KV_HEADS + kv_head_idx
        ) * MAX_PAGE_SLOTS + page_slot_indices
        page_counts = tl.load(
            page_token_counts + page_count_offsets, mask=page_valid, other=0
        )
        page_valid &= offsets_in_page < page_counts

        resident_valid = page_valid
        resident_ids = tl.zeros((BLOCK_TOKENS,), dtype=tl.int64)
        if HAS_RESIDENT:
            resident_ids = tl.load(
                resident_page_ids + resolved_page_offsets,
                mask=page_valid,
                other=-1,
            ).to(tl.int64)
            resident_valid &= resident_ids >= 0
            resident_key_offsets = (
                resident_ids[:, None] * resident_key_stride_0
                + offsets_in_page[:, None] * resident_key_stride_1
                + dimension_offsets[None, :] * resident_key_stride_2
            )
            resident_value_offsets = (
                resident_ids[:, None] * resident_value_stride_0
                + offsets_in_page[:, None] * resident_value_stride_1
                + dimension_offsets[None, :] * resident_value_stride_2
            )
            resident_mask = resident_valid[:, None] & dimension_mask[None, :]
            key_vectors += tl.load(
                resident_key_pages + resident_key_offsets,
                mask=resident_mask,
                other=0.0,
            ).to(tl.float32)
            value_vectors += tl.load(
                resident_value_pages + resident_value_offsets,
                mask=resident_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            resident_valid = page_valid & False

        staging_valid = page_valid & ~resident_valid
        if HAS_STAGING:
            staging_ids = tl.load(
                staging_page_ids + resolved_page_offsets,
                mask=staging_valid,
                other=-1,
            ).to(tl.int64)
            staging_valid &= staging_ids >= 0
            staging_key_offsets = (
                staging_ids[:, None] * staging_key_stride_0
                + offsets_in_page[:, None] * staging_key_stride_1
                + dimension_offsets[None, :] * staging_key_stride_2
            )
            staging_value_offsets = (
                staging_ids[:, None] * staging_value_stride_0
                + offsets_in_page[:, None] * staging_value_stride_1
                + dimension_offsets[None, :] * staging_value_stride_2
            )
            staging_mask = staging_valid[:, None] & dimension_mask[None, :]
            key_vectors += tl.load(
                staging_key_pages + staging_key_offsets,
                mask=staging_mask,
                other=0.0,
            ).to(tl.float32)
            value_vectors += tl.load(
                staging_value_pages + staging_value_offsets,
                mask=staging_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            staging_valid = page_valid & False

        valid_tokens = primary_valid | resident_valid | staging_valid
        scores = tl.sum(key_vectors * query_vector[None, :], axis=1) * scale
        scores = tl.where(valid_tokens, scores, float("-inf"))
        chunk_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, chunk_max)
        old_scale = tl.where(running_sum > 0.0, tl.exp(running_max - new_max), 0.0)
        probabilities = tl.where(valid_tokens, tl.exp(scores - new_max), 0.0)
        chunk_sum = tl.sum(probabilities, axis=0)
        running_output = running_output * old_scale + tl.sum(
            probabilities[:, None] * value_vectors, axis=0
        )
        running_sum = running_sum * old_scale + chunk_sum
        running_max = new_max

    normalized_output = tl.where(running_sum > 0.0, running_output / running_sum, 0.0)
    output_offsets = (
        query_idx * output_stride_0
        + query_head_idx * output_stride_1
        + partition_idx * output_stride_2
        + dimension_offsets * output_stride_3
    )
    stats_offset = (
        query_idx * stats_stride_0
        + query_head_idx * stats_stride_1
        + partition_idx * stats_stride_2
    )
    tl.store(partial_output + output_offsets, normalized_output, mask=dimension_mask)
    tl.store(partial_max + stats_offset, running_max)
    tl.store(partial_sum + stats_offset, running_sum)


@triton.jit
def _ranked_draft_attention_partition_kernel(
    query,
    key_cache,
    value_cache,
    block_table,
    primary_token_indices,
    primary_token_mask,
    request_slot_ids,
    ranked_cluster_indices,
    candidate_counts,
    resident_bucket_ids,
    cluster_keys,
    cluster_values,
    cluster_token_counts,
    cluster_page_starts,
    cluster_page_counts,
    page_token_counts,
    cluster_offsets,
    page_offsets,
    resident_table_page_slots,
    resident_key_pages,
    resident_value_pages,
    partial_output,
    partial_max,
    partial_sum,
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
    block_table_stride_0,
    block_table_stride_1,
    resident_key_stride_0,
    resident_key_stride_1,
    resident_key_stride_2,
    resident_value_stride_0,
    resident_value_stride_1,
    resident_value_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    stats_stride_0,
    stats_stride_1,
    stats_stride_2,
    scale,
    source_partition_offset,
    NUM_KV_HEADS: tl.constexpr,
    QUERIES_PER_KV_HEAD: tl.constexpr,
    MAX_PRIMARY_TOKENS: tl.constexpr,
    RETRIEVAL_WIDTH: tl.constexpr,
    ESTIMATION_WIDTH: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CLUSTER_CAPACITY: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    RESIDENT_PAGE_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    RANKING_WIDTH: tl.constexpr,
    RETRIEVAL_RATIO: tl.constexpr,
    ESTIMATION_RATIO: tl.constexpr,
):
    query_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    kv_head_idx = query_head_idx // QUERIES_PER_KV_HEAD

    dimension_offsets = tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < HEAD_SIZE
    query_offsets = (
        query_idx * query_stride_0
        + query_head_idx * query_stride_1
        + dimension_offsets * query_stride_2
    )
    query_vector = tl.load(query + query_offsets, mask=dimension_mask, other=0.0)
    query_vector = query_vector.to(tl.float32)

    request_slot = tl.load(request_slot_ids + query_idx).to(tl.int64)
    request_valid = request_slot >= 0
    safe_request_slot = tl.maximum(request_slot, 0)
    request_cluster_offset = tl.load(
        cluster_offsets + safe_request_slot, mask=request_valid, other=0
    ).to(tl.int64)
    request_page_offset = tl.load(
        page_offsets + safe_request_slot, mask=request_valid, other=0
    ).to(tl.int64)
    candidate_count = tl.load(
        candidate_counts + query_idx * NUM_KV_HEADS + kv_head_idx,
        mask=request_valid,
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
    ranked_row_offset = (query_idx * NUM_KV_HEADS + kv_head_idx) * RANKING_WIDTH

    resident_region_start = MAX_PRIMARY_TOKENS
    resident_cluster_stride = MAX_PAGES * PAGE_SIZE
    resident_region_end = (
        resident_region_start + RETRIEVAL_WIDTH * resident_cluster_stride
    )
    estimation_region_end = resident_region_end + ESTIMATION_WIDTH
    source_capacity = estimation_region_end + RETRIEVAL_WIDTH

    running_max = float("-inf")
    running_sum = 0.0
    running_output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partition_start = (source_partition_offset + partition_idx) * PARTITION_SIZE

    for chunk_start in tl.range(0, PARTITION_SIZE, BLOCK_TOKENS):
        token_offsets = partition_start + chunk_start + tl.arange(0, BLOCK_TOKENS)
        source_valid = token_offsets < source_capacity

        primary_valid = source_valid & (token_offsets < MAX_PRIMARY_TOKENS)
        primary_metadata_offsets = (
            query_idx * NUM_KV_HEADS + kv_head_idx
        ) * MAX_PRIMARY_TOKENS + token_offsets
        primary_valid &= tl.load(
            primary_token_mask + primary_metadata_offsets,
            mask=primary_valid,
            other=0,
        ).to(tl.int1)
        logical_token_indices = tl.load(
            primary_token_indices + primary_metadata_offsets,
            mask=primary_valid,
            other=0,
        ).to(tl.int64)
        logical_block_indices = logical_token_indices // PAGE_SIZE
        primary_block_offsets = logical_token_indices % PAGE_SIZE
        physical_block_indices = tl.load(
            block_table
            + query_idx * block_table_stride_0
            + logical_block_indices * block_table_stride_1,
            mask=primary_valid,
            other=0,
        ).to(tl.int64)
        primary_key_offsets = (
            physical_block_indices[:, None] * key_stride_0
            + primary_block_offsets[:, None] * key_stride_1
            + kv_head_idx * key_stride_2
            + dimension_offsets[None, :] * key_stride_3
        )
        primary_value_offsets = (
            physical_block_indices[:, None] * value_stride_0
            + primary_block_offsets[:, None] * value_stride_1
            + kv_head_idx * value_stride_2
            + dimension_offsets[None, :] * value_stride_3
        )
        primary_vector_mask = primary_valid[:, None] & dimension_mask[None, :]
        key_vectors = tl.load(
            key_cache + primary_key_offsets,
            mask=primary_vector_mask,
            other=0.0,
        ).to(tl.float32)
        value_vectors = tl.load(
            value_cache + primary_value_offsets,
            mask=primary_vector_mask,
            other=0.0,
        ).to(tl.float32)

        resident_linear_offsets = token_offsets - resident_region_start
        resident_ranks = resident_linear_offsets // resident_cluster_stride
        resident_inner_offsets = resident_linear_offsets % resident_cluster_stride
        resident_page_offsets = resident_inner_offsets // PAGE_SIZE
        resident_token_offsets = resident_inner_offsets % PAGE_SIZE
        resident_valid = (
            source_valid
            & (token_offsets >= resident_region_start)
            & (token_offsets < resident_region_end)
        )
        safe_resident_ranks = tl.maximum(resident_ranks, 0)
        exact_row_offset = (query_idx * NUM_KV_HEADS + kv_head_idx) * RETRIEVAL_WIDTH
        resident_valid &= resident_ranks < retrieval_count
        local_cluster_indices = tl.load(
            ranked_cluster_indices + ranked_row_offset + safe_resident_ranks,
            mask=resident_valid,
            other=-1,
        ).to(tl.int64)
        resident_buckets = tl.load(
            resident_bucket_ids + exact_row_offset + safe_resident_ranks,
            mask=resident_valid,
            other=-1,
        ).to(tl.int64)
        resident_valid &= (local_cluster_indices >= 0) & (resident_buckets >= 0)
        cluster_storage_offsets = (
            kv_head_idx * CLUSTER_CAPACITY
            + request_cluster_offset
            + tl.maximum(local_cluster_indices, 0)
        )
        logical_page_starts = tl.load(
            cluster_page_starts + cluster_storage_offsets,
            mask=resident_valid,
            other=0,
        ).to(tl.int64)
        logical_page_counts = tl.load(
            cluster_page_counts + cluster_storage_offsets,
            mask=resident_valid,
            other=0,
        ).to(tl.int32)
        resident_valid &= resident_page_offsets < logical_page_counts
        page_storage_offsets = (
            kv_head_idx * PAGE_CAPACITY
            + request_page_offset
            + logical_page_starts
            + resident_page_offsets
        )
        valid_page_token_counts = tl.load(
            page_token_counts + page_storage_offsets,
            mask=resident_valid,
            other=0,
        ).to(tl.int32)
        resident_valid &= resident_token_offsets < valid_page_token_counts
        resident_page_slots = tl.load(
            resident_table_page_slots
            + tl.maximum(resident_buckets, 0) * RESIDENT_PAGE_STRIDE
            + resident_page_offsets,
            mask=resident_valid,
            other=-1,
        ).to(tl.int64)
        resident_valid &= resident_page_slots >= 0
        resident_key_offsets = (
            resident_page_slots[:, None] * resident_key_stride_0
            + resident_token_offsets[:, None] * resident_key_stride_1
            + dimension_offsets[None, :] * resident_key_stride_2
        )
        resident_value_offsets = (
            resident_page_slots[:, None] * resident_value_stride_0
            + resident_token_offsets[:, None] * resident_value_stride_1
            + dimension_offsets[None, :] * resident_value_stride_2
        )
        resident_vector_mask = resident_valid[:, None] & dimension_mask[None, :]
        key_vectors += tl.load(
            resident_key_pages + resident_key_offsets,
            mask=resident_vector_mask,
            other=0.0,
        ).to(tl.float32)
        value_vectors += tl.load(
            resident_value_pages + resident_value_offsets,
            mask=resident_vector_mask,
            other=0.0,
        ).to(tl.float32)

        estimation_ranks = token_offsets - resident_region_end
        fallback_ranks = token_offsets - estimation_region_end
        estimation_valid = (
            source_valid
            & (token_offsets >= resident_region_end)
            & (token_offsets < estimation_region_end)
        )
        fallback_valid = source_valid & (token_offsets >= estimation_region_end)
        safe_estimation_ranks = tl.maximum(estimation_ranks, 0)
        safe_fallback_ranks = tl.maximum(fallback_ranks, 0)
        estimation_valid &= estimation_ranks < estimation_count
        fallback_valid &= fallback_ranks < retrieval_count
        ranked_estimation_ranks = retrieval_count + safe_estimation_ranks
        estimation_local_indices = tl.load(
            ranked_cluster_indices + ranked_row_offset + ranked_estimation_ranks,
            mask=estimation_valid,
            other=-1,
        ).to(tl.int64)
        fallback_local_indices = tl.load(
            ranked_cluster_indices + ranked_row_offset + safe_fallback_ranks,
            mask=fallback_valid,
            other=-1,
        ).to(tl.int64)
        fallback_buckets = tl.load(
            resident_bucket_ids + exact_row_offset + safe_fallback_ranks,
            mask=fallback_valid,
            other=-1,
        ).to(tl.int64)
        summary_local_indices = tl.where(
            estimation_valid, estimation_local_indices, fallback_local_indices
        )
        summary_valid = (estimation_valid & (estimation_local_indices >= 0)) | (
            fallback_valid & (fallback_local_indices >= 0) & (fallback_buckets < 0)
        )
        summary_storage_offsets = (
            kv_head_idx * CLUSTER_CAPACITY
            + request_cluster_offset
            + tl.maximum(summary_local_indices, 0)
        )
        summary_token_counts = tl.load(
            cluster_token_counts + summary_storage_offsets,
            mask=summary_valid,
            other=0,
        ).to(tl.int32)
        summary_valid &= summary_token_counts > 0
        summary_offsets = (
            summary_storage_offsets[:, None] * HEAD_SIZE + dimension_offsets[None, :]
        )
        summary_vector_mask = summary_valid[:, None] & dimension_mask[None, :]
        key_vectors += tl.load(
            cluster_keys + summary_offsets,
            mask=summary_vector_mask,
            other=0.0,
        ).to(tl.float32)
        value_vectors += tl.load(
            cluster_values + summary_offsets,
            mask=summary_vector_mask,
            other=0.0,
        ).to(tl.float32)

        valid_tokens = primary_valid | resident_valid | summary_valid
        scores = tl.sum(key_vectors * query_vector[None, :], axis=1) * scale
        scores += tl.where(
            summary_valid,
            tl.log(tl.maximum(summary_token_counts.to(tl.float32), 1.0)),
            0.0,
        )
        scores = tl.where(valid_tokens, scores, float("-inf"))
        chunk_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, chunk_max)
        safe_new_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        old_scale = tl.where(running_sum > 0.0, tl.exp(running_max - safe_new_max), 0.0)
        probabilities = tl.where(valid_tokens, tl.exp(scores - safe_new_max), 0.0)
        running_output = running_output * old_scale + tl.sum(
            probabilities[:, None] * value_vectors, axis=0
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = tl.where(running_sum > 0.0, new_max, float("-inf"))

    normalized_output = tl.where(running_sum > 0.0, running_output / running_sum, 0.0)
    output_offsets = (
        query_idx * output_stride_0
        + query_head_idx * output_stride_1
        + partition_idx * output_stride_2
        + dimension_offsets * output_stride_3
    )
    stats_offset = (
        query_idx * stats_stride_0
        + query_head_idx * stats_stride_1
        + partition_idx * stats_stride_2
    )
    tl.store(partial_output + output_offsets, normalized_output, mask=dimension_mask)
    tl.store(partial_max + stats_offset, running_max)
    tl.store(partial_sum + stats_offset, running_sum)


@triton.jit
def _accumulate_exact_partition_wave_kernel(
    partial_output,
    partial_max,
    partial_sum,
    accumulated_output,
    accumulated_max,
    accumulated_sum,
    partial_output_stride_0,
    partial_output_stride_1,
    partial_output_stride_2,
    partial_output_stride_3,
    partial_stats_stride_0,
    partial_stats_stride_1,
    partial_stats_stride_2,
    accumulated_output_stride_0,
    accumulated_output_stride_1,
    accumulated_output_stride_3,
    accumulated_stats_stride_0,
    accumulated_stats_stride_1,
    accumulated_stats_stride_2,
    num_partitions,
    RESET: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    dimension_offsets = tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < HEAD_SIZE

    accumulated_stats_offset = (
        query_idx * accumulated_stats_stride_0
        + query_head_idx * accumulated_stats_stride_1
    )
    previous_max = float("-inf")
    previous_sum = 0.0
    previous_output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if not RESET:
        previous_max = tl.load(accumulated_max + accumulated_stats_offset)
        previous_sum = tl.load(accumulated_sum + accumulated_stats_offset)
        accumulated_output_offsets = (
            query_idx * accumulated_output_stride_0
            + query_head_idx * accumulated_output_stride_1
            + dimension_offsets * accumulated_output_stride_3
        )
        previous_output = tl.load(
            accumulated_output + accumulated_output_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

    global_max = previous_max
    for partition_idx in tl.range(0, num_partitions):
        stats_offset = (
            query_idx * partial_stats_stride_0
            + query_head_idx * partial_stats_stride_1
            + partition_idx * partial_stats_stride_2
        )
        global_max = tl.maximum(global_max, tl.load(partial_max + stats_offset))

    safe_global_max = tl.where(global_max == float("-inf"), 0.0, global_max)
    global_sum = tl.where(
        previous_sum > 0.0,
        previous_sum * tl.exp(previous_max - safe_global_max),
        0.0,
    )
    global_output = global_sum * previous_output
    for partition_idx in tl.range(0, num_partitions):
        stats_offset = (
            query_idx * partial_stats_stride_0
            + query_head_idx * partial_stats_stride_1
            + partition_idx * partial_stats_stride_2
        )
        partition_max = tl.load(partial_max + stats_offset)
        partition_sum = tl.load(partial_sum + stats_offset)
        partition_weight = tl.where(
            partition_sum > 0.0,
            partition_sum * tl.exp(partition_max - safe_global_max),
            0.0,
        )
        partial_offsets = (
            query_idx * partial_output_stride_0
            + query_head_idx * partial_output_stride_1
            + partition_idx * partial_output_stride_2
            + dimension_offsets * partial_output_stride_3
        )
        partition_output = tl.load(
            partial_output + partial_offsets, mask=dimension_mask, other=0.0
        ).to(tl.float32)
        global_output += partition_weight * partition_output
        global_sum += partition_weight

    normalized_output = tl.where(global_sum > 0.0, global_output / global_sum, 0.0)
    accumulated_output_offsets = (
        query_idx * accumulated_output_stride_0
        + query_head_idx * accumulated_output_stride_1
        + dimension_offsets * accumulated_output_stride_3
    )
    tl.store(
        accumulated_output + accumulated_output_offsets,
        normalized_output,
        mask=dimension_mask,
    )
    tl.store(
        accumulated_max + accumulated_stats_offset,
        tl.where(global_sum > 0.0, global_max, float("-inf")),
    )
    tl.store(accumulated_sum + accumulated_stats_offset, global_sum)


@triton.jit
def _reduce_exact_partitions_kernel(
    partial_output,
    partial_max,
    partial_sum,
    output,
    output_lse,
    partial_output_stride_0,
    partial_output_stride_1,
    partial_output_stride_2,
    partial_output_stride_3,
    stats_stride_0,
    stats_stride_1,
    stats_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    lse_stride_0,
    lse_stride_1,
    num_partitions,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    dimension_offsets = tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < HEAD_SIZE

    global_max = float("-inf")
    for partition_idx in tl.range(0, num_partitions):
        stats_offset = (
            query_idx * stats_stride_0
            + query_head_idx * stats_stride_1
            + partition_idx * stats_stride_2
        )
        partition_max = tl.load(partial_max + stats_offset)
        global_max = tl.maximum(global_max, partition_max)

    safe_global_max = tl.where(global_max == float("-inf"), 0.0, global_max)
    global_sum = 0.0
    global_output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for partition_idx in tl.range(0, num_partitions):
        stats_offset = (
            query_idx * stats_stride_0
            + query_head_idx * stats_stride_1
            + partition_idx * stats_stride_2
        )
        partition_max = tl.load(partial_max + stats_offset)
        partition_sum = tl.load(partial_sum + stats_offset)
        partition_scale = tl.where(
            partition_sum > 0.0,
            partition_sum * tl.exp(partition_max - safe_global_max),
            0.0,
        )
        partial_offsets = (
            query_idx * partial_output_stride_0
            + query_head_idx * partial_output_stride_1
            + partition_idx * partial_output_stride_2
            + dimension_offsets * partial_output_stride_3
        )
        partition_output = tl.load(
            partial_output + partial_offsets, mask=dimension_mask, other=0.0
        ).to(tl.float32)
        global_output += partition_scale * partition_output
        global_sum += partition_scale

    normalized_output = tl.where(global_sum > 0.0, global_output / global_sum, 0.0)
    output_offsets = (
        query_idx * output_stride_0
        + query_head_idx * output_stride_1
        + dimension_offsets * output_stride_2
    )
    tl.store(output + output_offsets, normalized_output, mask=dimension_mask)
    lse_offset = query_head_idx * lse_stride_0 + query_idx * lse_stride_1
    lse = tl.where(
        global_sum > 0.0, safe_global_max + tl.log(global_sum), float("-inf")
    )
    tl.store(output_lse + lse_offset, lse)


@triton.jit
def _reduce_proposal_partitions_kernel(
    query,
    plan_row_indices,
    estimation_keys,
    estimation_values,
    estimation_token_counts,
    partial_output,
    partial_max,
    partial_sum,
    output,
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
    partial_output_stride_0,
    partial_output_stride_1,
    partial_output_stride_2,
    partial_output_stride_3,
    stats_stride_0,
    stats_stride_1,
    stats_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    scale,
    num_partitions,
    num_estimation_vectors,
    USE_PLAN_ROWS: tl.constexpr,
    QUERIES_PER_KV_HEAD: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // QUERIES_PER_KV_HEAD

    estimation_row_idx = query_idx
    if USE_PLAN_ROWS:
        estimation_row_idx = tl.load(plan_row_indices + query_idx)

    dimension_offsets = tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < HEAD_SIZE
    query_offsets = (
        query_idx * query_stride_0
        + query_head_idx * query_stride_1
        + dimension_offsets * query_stride_2
    )
    query_vector = tl.load(query + query_offsets, mask=dimension_mask, other=0.0).to(
        tl.float32
    )

    global_max = float("-inf")
    for partition_idx in tl.range(0, num_partitions):
        stats_offset = (
            query_idx * stats_stride_0
            + query_head_idx * stats_stride_1
            + partition_idx * stats_stride_2
        )
        partition_max = tl.load(partial_max + stats_offset)
        global_max = tl.maximum(global_max, partition_max)

    safe_global_max = tl.where(global_max == float("-inf"), 0.0, global_max)
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for partition_idx in tl.range(0, num_partitions):
        stats_offset = (
            query_idx * stats_stride_0
            + query_head_idx * stats_stride_1
            + partition_idx * stats_stride_2
        )
        partition_max = tl.load(partial_max + stats_offset)
        partition_sum = tl.load(partial_sum + stats_offset)
        partition_weight = tl.where(
            partition_sum > 0.0,
            partition_sum * tl.exp(partition_max - safe_global_max),
            0.0,
        )
        partial_offsets = (
            query_idx * partial_output_stride_0
            + query_head_idx * partial_output_stride_1
            + partition_idx * partial_output_stride_2
            + dimension_offsets * partial_output_stride_3
        )
        partition_output = tl.load(
            partial_output + partial_offsets, mask=dimension_mask, other=0.0
        ).to(tl.float32)
        accumulator += partition_weight * partition_output
        running_sum += partition_weight

    running_max = tl.where(running_sum > 0.0, global_max, float("-inf"))
    for vector_start in tl.range(0, num_estimation_vectors, BLOCK_M):
        vector_offsets = vector_start + tl.arange(0, BLOCK_M)
        vector_mask = vector_offsets < num_estimation_vectors
        count_offsets = (
            estimation_row_idx * count_stride_0
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
            estimation_row_idx * key_stride_0
            + kv_head_idx * key_stride_1
            + vector_offsets[:, None] * key_stride_2
            + dimension_offsets[None, :] * key_stride_3
        )
        key_vectors = tl.load(
            estimation_keys + key_offsets,
            mask=valid_vectors[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(key_vectors * query_vector[None, :], axis=1) * scale
        logits += tl.log(tl.maximum(token_counts.to(tl.float32), 1.0))
        logits = tl.where(valid_vectors, logits, float("-inf"))

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, block_max)
        safe_new_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        old_scale = tl.where(running_sum > 0.0, tl.exp(running_max - safe_new_max), 0.0)
        probabilities = tl.where(valid_vectors, tl.exp(logits - safe_new_max), 0.0)

        value_offsets = (
            estimation_row_idx * value_stride_0
            + kv_head_idx * value_stride_1
            + vector_offsets[:, None] * value_stride_2
            + dimension_offsets[None, :] * value_stride_3
        )
        value_vectors = tl.load(
            estimation_values + value_offsets,
            mask=valid_vectors[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator *= old_scale
        accumulator += tl.sum(probabilities[:, None] * value_vectors, axis=0)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = tl.where(running_sum > 0.0, new_max, float("-inf"))

    normalized_output = tl.where(
        running_sum > 0.0,
        accumulator / tl.maximum(running_sum, 1.0),
        0.0,
    )
    output_offsets = (
        query_idx * output_stride_0
        + query_head_idx * output_stride_1
        + dimension_offsets * output_stride_2
    )
    tl.store(output + output_offsets, normalized_output, mask=dimension_mask)


@triton.jit
def _parallel_cluster_prefix_kernel(
    query,
    query_start_loc,
    cluster_keys,
    cluster_values,
    cluster_token_offsets,
    cluster_token_counts,
    partial_output,
    partial_max,
    partial_sum,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    cluster_key_stride_0,
    cluster_key_stride_1,
    cluster_value_stride_0,
    cluster_value_stride_1,
    metadata_stride_0,
    metadata_stride_1,
    partial_output_stride_0,
    partial_output_stride_1,
    partial_output_stride_2,
    partial_output_stride_3,
    stats_stride_0,
    stats_stride_1,
    stats_stride_2,
    scale,
    NUM_KV_HEADS: tl.constexpr,
    QUERIES_PER_KV_HEAD: tl.constexpr,
    MAX_CLUSTER_TOKENS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    request_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    query_tile_split_idx = tl.program_id(2)
    query_tile_idx = query_tile_split_idx // NUM_SPLITS
    split_idx = query_tile_split_idx % NUM_SPLITS
    kv_head_idx = query_head_idx // QUERIES_PER_KV_HEAD

    request_query_start = tl.load(query_start_loc + request_idx)
    request_query_end = tl.load(query_start_loc + request_idx + 1)
    query_indices = (
        request_query_start + query_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    )
    query_valid = query_indices < request_query_end
    dimension_offsets = tl.arange(0, BLOCK_D)
    dimension_valid = dimension_offsets < HEAD_SIZE
    query_offsets = (
        query_indices[:, None] * query_stride_0
        + query_head_idx * query_stride_1
        + dimension_offsets[None, :] * query_stride_2
    )
    query_vectors = tl.load(
        query + query_offsets,
        mask=query_valid[:, None] & dimension_valid[None, :],
        other=0.0,
    )

    metadata_offset = request_idx * metadata_stride_0 + kv_head_idx * metadata_stride_1
    token_base = tl.load(cluster_token_offsets + metadata_offset).to(tl.int64)
    token_count = tl.load(cluster_token_counts + metadata_offset)
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_output = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for token_start in tl.range(
        split_idx * BLOCK_N,
        MAX_CLUSTER_TOKENS,
        NUM_SPLITS * BLOCK_N,
    ):
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_valid = token_offsets < token_count
        storage_indices = token_base + token_offsets
        key_offsets = (
            storage_indices[:, None] * cluster_key_stride_0
            + dimension_offsets[None, :] * cluster_key_stride_1
        )
        value_offsets = (
            storage_indices[:, None] * cluster_value_stride_0
            + dimension_offsets[None, :] * cluster_value_stride_1
        )
        vector_valid = token_valid[:, None] & dimension_valid[None, :]
        key_vectors = tl.load(cluster_keys + key_offsets, mask=vector_valid, other=0.0)
        value_vectors = tl.load(
            cluster_values + value_offsets, mask=vector_valid, other=0.0
        )
        score_valid = query_valid[:, None] & token_valid[None, :]
        scores = tl.dot(query_vectors, tl.trans(key_vectors)) * scale
        scores = tl.where(score_valid, scores, float("-inf"))
        block_max = tl.max(scores, axis=1)
        updated_max = tl.maximum(running_max, block_max)
        safe_updated_max = tl.where(updated_max == float("-inf"), 0.0, updated_max)
        previous_scale = tl.where(
            running_sum > 0.0, tl.exp(running_max - safe_updated_max), 0.0
        )
        probabilities = tl.where(
            score_valid, tl.exp(scores - safe_updated_max[:, None]), 0.0
        )
        running_output = running_output * previous_scale[:, None] + tl.dot(
            probabilities.to(value_vectors.dtype), value_vectors
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
        running_max = updated_max

    normalized_output = tl.where(
        running_sum[:, None] > 0.0,
        running_output / running_sum[:, None],
        0.0,
    )
    partial_output_offsets = (
        query_indices[:, None] * partial_output_stride_0
        + query_head_idx * partial_output_stride_1
        + split_idx * partial_output_stride_2
        + dimension_offsets[None, :] * partial_output_stride_3
    )
    tl.store(
        partial_output + partial_output_offsets,
        normalized_output,
        mask=query_valid[:, None] & dimension_valid[None, :],
    )
    stats_offsets = (
        query_indices * stats_stride_0
        + query_head_idx * stats_stride_1
        + split_idx * stats_stride_2
    )
    tl.store(partial_max + stats_offsets, running_max, mask=query_valid)
    tl.store(partial_sum + stats_offsets, running_sum, mask=query_valid)


@triton.jit
def _parallel_native_suffix_kernel(
    query,
    local_keys,
    local_values,
    query_start_loc,
    key_cache,
    value_cache,
    block_table,
    token_indices,
    token_mask,
    partial_output,
    partial_max,
    partial_sum,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    local_key_stride_0,
    local_key_stride_1,
    local_key_stride_2,
    local_value_stride_0,
    local_value_stride_1,
    local_value_stride_2,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    value_stride_3,
    block_table_stride_0,
    block_table_stride_1,
    partial_output_stride_0,
    partial_output_stride_1,
    partial_output_stride_2,
    partial_output_stride_3,
    stats_stride_0,
    stats_stride_1,
    stats_stride_2,
    scale,
    NUM_KV_HEADS: tl.constexpr,
    QUERIES_PER_KV_HEAD: tl.constexpr,
    MAX_PRIMARY_TOKENS: tl.constexpr,
    MAX_QUERY_LEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    request_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    query_tile_split_idx = tl.program_id(2)
    query_tile_idx = query_tile_split_idx // NUM_SPLITS
    split_idx = query_tile_split_idx % NUM_SPLITS
    kv_head_idx = query_head_idx // QUERIES_PER_KV_HEAD

    request_query_start = tl.load(query_start_loc + request_idx)
    request_query_end = tl.load(query_start_loc + request_idx + 1)
    query_indices = (
        request_query_start + query_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    )
    query_valid = query_indices < request_query_end
    query_positions = query_indices - request_query_start
    dimension_offsets = tl.arange(0, BLOCK_D)
    dimension_valid = dimension_offsets < HEAD_SIZE
    query_offsets = (
        query_indices[:, None] * query_stride_0
        + query_head_idx * query_stride_1
        + dimension_offsets[None, :] * query_stride_2
    )
    query_vectors = tl.load(
        query + query_offsets,
        mask=query_valid[:, None] & dimension_valid[None, :],
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_output = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    max_native_tokens = MAX_PRIMARY_TOKENS + MAX_QUERY_LEN

    for token_start in tl.range(
        split_idx * BLOCK_N,
        max_native_tokens,
        NUM_SPLITS * BLOCK_N,
    ):
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        if MAX_PRIMARY_TOKENS > 0:
            is_primary = token_offsets < MAX_PRIMARY_TOKENS
            primary_offsets = tl.minimum(token_offsets, MAX_PRIMARY_TOKENS - 1)
            metadata_offsets = (
                request_idx * NUM_KV_HEADS + kv_head_idx
            ) * MAX_PRIMARY_TOKENS + primary_offsets
            primary_valid = is_primary & tl.load(
                token_mask + metadata_offsets, mask=is_primary, other=0
            ).to(tl.int1)
            logical_token_indices = tl.load(
                token_indices + metadata_offsets, mask=primary_valid, other=0
            )
            logical_block_indices = logical_token_indices // PAGE_SIZE
            block_offsets = logical_token_indices % PAGE_SIZE
            physical_block_indices = tl.load(
                block_table
                + request_idx * block_table_stride_0
                + logical_block_indices * block_table_stride_1,
                mask=primary_valid,
                other=0,
            ).to(tl.int64)
            primary_key_offsets = (
                physical_block_indices[:, None] * key_stride_0
                + block_offsets[:, None] * key_stride_1
                + kv_head_idx * key_stride_2
                + dimension_offsets[None, :] * key_stride_3
            )
            primary_value_offsets = (
                physical_block_indices[:, None] * value_stride_0
                + block_offsets[:, None] * value_stride_1
                + kv_head_idx * value_stride_2
                + dimension_offsets[None, :] * value_stride_3
            )
            primary_vector_valid = primary_valid[:, None] & dimension_valid[None, :]
            primary_keys = tl.load(
                key_cache + primary_key_offsets, mask=primary_vector_valid, other=0.0
            )
            primary_values = tl.load(
                value_cache + primary_value_offsets,
                mask=primary_vector_valid,
                other=0.0,
            )
        else:
            is_primary = token_offsets < 0
            primary_valid = is_primary
            primary_keys = tl.zeros((BLOCK_N, BLOCK_D), query.dtype.element_ty)
            primary_values = tl.zeros((BLOCK_N, BLOCK_D), query.dtype.element_ty)

        local_offsets = token_offsets - MAX_PRIMARY_TOKENS
        local_valid = (~is_primary) & (
            request_query_start + local_offsets < request_query_end
        )
        local_storage_indices = request_query_start + tl.maximum(local_offsets, 0)
        local_key_offsets = (
            local_storage_indices[:, None] * local_key_stride_0
            + kv_head_idx * local_key_stride_1
            + dimension_offsets[None, :] * local_key_stride_2
        )
        local_value_offsets = (
            local_storage_indices[:, None] * local_value_stride_0
            + kv_head_idx * local_value_stride_1
            + dimension_offsets[None, :] * local_value_stride_2
        )
        local_vector_valid = local_valid[:, None] & dimension_valid[None, :]
        suffix_keys = tl.load(
            local_keys + local_key_offsets, mask=local_vector_valid, other=0.0
        )
        suffix_values = tl.load(
            local_values + local_value_offsets, mask=local_vector_valid, other=0.0
        )
        key_vectors = tl.where(is_primary[:, None], primary_keys, suffix_keys)
        value_vectors = tl.where(is_primary[:, None], primary_values, suffix_values)
        local_causal = local_valid[None, :] & (
            local_offsets[None, :] <= query_positions[:, None]
        )
        score_valid = query_valid[:, None] & (primary_valid[None, :] | local_causal)
        scores = tl.dot(query_vectors, tl.trans(key_vectors)) * scale
        scores = tl.where(score_valid, scores, float("-inf"))
        block_max = tl.max(scores, axis=1)
        updated_max = tl.maximum(running_max, block_max)
        safe_updated_max = tl.where(updated_max == float("-inf"), 0.0, updated_max)
        previous_scale = tl.where(
            running_sum > 0.0, tl.exp(running_max - safe_updated_max), 0.0
        )
        probabilities = tl.where(
            score_valid, tl.exp(scores - safe_updated_max[:, None]), 0.0
        )
        running_output = running_output * previous_scale[:, None] + tl.dot(
            probabilities.to(value_vectors.dtype), value_vectors
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
        running_max = updated_max

    normalized_output = tl.where(
        running_sum[:, None] > 0.0,
        running_output / running_sum[:, None],
        0.0,
    )
    partial_output_offsets = (
        query_indices[:, None] * partial_output_stride_0
        + query_head_idx * partial_output_stride_1
        + split_idx * partial_output_stride_2
        + dimension_offsets[None, :] * partial_output_stride_3
    )
    tl.store(
        partial_output + partial_output_offsets,
        normalized_output,
        mask=query_valid[:, None] & dimension_valid[None, :],
    )
    stats_offsets = (
        query_indices * stats_stride_0
        + query_head_idx * stats_stride_1
        + split_idx * stats_stride_2
    )
    tl.store(partial_max + stats_offsets, running_max, mask=query_valid)
    tl.store(partial_sum + stats_offsets, running_sum, mask=query_valid)
