# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

from .workspace import EXACT_ATTENTION_PARTITION_SIZE

_EXACT_ATTENTION_BLOCK_TOKENS = 64
_PARALLEL_FULL_BLOCK_QUERIES = 16
_PARALLEL_FULL_BLOCK_TOKENS = 64
_PARALLEL_FULL_NATIVE_BLOCK_TOKENS = 32
_PARALLEL_FULL_NUM_SPLITS = 8


@dataclass(frozen=True)
class RetroSpecExactPrimaryKVSource:
    """Primary exact KV and the logical-token metadata used to address it."""

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    token_indices: torch.Tensor
    token_mask: torch.Tensor
    ready_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class RetroSpecExactPageKVSource:
    """One resolved GPU page source for clustered exact KV."""

    key_pages: torch.Tensor
    value_pages: torch.Tensor
    page_ids: torch.Tensor
    ready_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class RetroSpecExactKVSource:
    """GPU-visible descriptors for native, resident and staging exact KV."""

    primary: RetroSpecExactPrimaryKVSource
    page_token_counts: torch.Tensor
    resident_pages: RetroSpecExactPageKVSource | None = None
    staging_pages: RetroSpecExactPageKVSource | None = None


@dataclass(frozen=True)
class RetroSpecCompactKVSource:
    """Token-contiguous clustered KV staged for full verification."""

    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    token_offsets: torch.Tensor
    token_counts: torch.Tensor
    max_tokens_per_head: int
    ready_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class RetroSpecFullVerificationKVSource:
    """Specialized cluster-prefix and native-suffix full-verify sources."""

    primary: RetroSpecExactPrimaryKVSource
    clustered: RetroSpecCompactKVSource | None = None


@triton.jit
def _multi_source_exact_partition_kernel(
    query,
    request_indices,
    key_cache,
    value_cache,
    block_table,
    token_indices,
    token_mask,
    page_token_counts,
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
    HAS_RESIDENT: tl.constexpr,
    HAS_STAGING: tl.constexpr,
):
    query_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)

    request_idx = query_idx
    if not IDENTITY_REQUESTS:
        request_idx = tl.load(request_indices + query_idx)
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
    partition_start = partition_idx * PARTITION_SIZE

    for chunk_start in tl.range(0, PARTITION_SIZE, BLOCK_TOKENS):
        token_offsets = partition_start + chunk_start + tl.arange(0, BLOCK_TOKENS)
        source_valid = token_offsets < MAX_PRIMARY_TOKENS + MAX_PAGE_SLOTS * PAGE_SIZE

        primary_valid = source_valid & (token_offsets < MAX_PRIMARY_TOKENS)
        primary_metadata_offsets = (
            request_idx * NUM_KV_HEADS + kv_head_idx
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
        page_metadata_offsets = (
            request_idx * NUM_KV_HEADS + kv_head_idx
        ) * MAX_PAGE_SLOTS + page_slot_indices
        page_counts = tl.load(
            page_token_counts + page_metadata_offsets, mask=page_valid, other=0
        )
        page_valid &= offsets_in_page < page_counts

        resident_valid = page_valid
        resident_ids = tl.zeros((BLOCK_TOKENS,), dtype=tl.int64)
        if HAS_RESIDENT:
            resident_ids = tl.load(
                resident_page_ids + page_metadata_offsets,
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
                staging_page_ids + page_metadata_offsets,
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


class RetroSpecExactAttentionWorkspace:
    """Reusable workspace for partitioned attention over three KV sources."""

    def __init__(
        self,
        page_size: int,
        max_num_queries: int,
        partition_capacity: int,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if max_num_queries <= 0:
            raise ValueError("max_num_queries must be positive")
        if partition_capacity <= 0:
            raise ValueError("partition_capacity must be positive")
        if partition_capacity & (partition_capacity - 1):
            raise ValueError("partition_capacity must be a power of two")

        self.page_size = page_size
        self.max_num_queries = max_num_queries
        self._partition_capacity = partition_capacity
        self._configuration: tuple[torch.dtype, torch.device, int, int] | None = None
        self._partial_output: torch.Tensor | None = None
        self._partial_max: torch.Tensor | None = None
        self._partial_sum: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._output_lse: torch.Tensor | None = None
        self._secondary_output: torch.Tensor | None = None
        self._secondary_output_lse: torch.Tensor | None = None

    def _ensure_output_workspace(self, query: torch.Tensor) -> None:
        _, num_query_heads, head_size = query.shape
        configuration = (query.dtype, query.device, num_query_heads, head_size)
        if self._configuration is not None and self._configuration != configuration:
            raise ValueError("Exact-attention workspace configuration changed")
        if self._configuration is not None:
            return
        output = torch.empty(
            self.max_num_queries,
            num_query_heads,
            head_size,
            dtype=query.dtype,
            device=query.device,
        )
        output_lse = torch.empty(
            num_query_heads * self.max_num_queries,
            dtype=torch.float32,
            device=query.device,
        )
        self._output = output
        self._output_lse = output_lse
        self._secondary_output = torch.empty_like(output)
        self._secondary_output_lse = torch.empty_like(output_lse)
        self._configuration = configuration

    def _ensure_workspace(
        self,
        query: torch.Tensor,
        required_partitions: int,
    ) -> None:
        if required_partitions > self._partition_capacity:
            raise RuntimeError(
                "Exact-attention source exceeds the planned workspace capacity: "
                f"required {required_partitions} partitions, but the capacity "
                f"planner reserved {self._partition_capacity}"
            )

        self._ensure_output_workspace(query)
        if self._partial_output is not None:
            return

        _, num_query_heads, head_size = query.shape
        partial_shape = (
            self.max_num_queries,
            num_query_heads,
            self._partition_capacity,
            head_size,
        )
        stats_shape = partial_shape[:-1]
        self._partial_output = torch.empty(
            partial_shape, dtype=query.dtype, device=query.device
        )
        self._partial_max = torch.empty(
            stats_shape, dtype=torch.float32, device=query.device
        )
        self._partial_sum = torch.empty_like(self._partial_max)

    def _validate_source(
        self,
        source: RetroSpecExactKVSource,
        query: torch.Tensor,
        request_indices: torch.Tensor | None,
    ) -> tuple[int, int, int]:
        if query.ndim != 3:
            raise ValueError("Query must have shape [queries, query_heads, head_size]")
        if query.device.type != "cuda":
            raise ValueError("Triton exact attention requires CUDA tensors")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Triton exact attention requires FP16 or BF16 queries")
        if query.shape[0] > self.max_num_queries:
            raise ValueError("Query count exceeds exact-attention workspace capacity")

        primary = source.primary
        if primary.key_cache.shape != primary.value_cache.shape:
            raise ValueError("Primary key and value cache shapes must match")
        if primary.key_cache.ndim != 4:
            raise ValueError(
                "Primary KV cache must have shape "
                "[blocks, block_size, kv_heads, head_size]"
            )
        if primary.key_cache.shape[1] != self.page_size:
            raise ValueError("Primary KV block size does not match page size")
        if primary.key_cache.dtype != query.dtype:
            raise ValueError("Primary KV dtype does not match query")
        if primary.token_indices.shape != primary.token_mask.shape:
            raise ValueError("Primary token indices and mask shapes must match")
        if primary.token_indices.ndim != 3:
            raise ValueError(
                "Primary token metadata must have shape [batch, kv_heads, tokens]"
            )
        if primary.token_mask.dtype != torch.bool:
            raise ValueError("Primary token mask must be boolean")
        if source.page_token_counts.ndim != 4:
            raise ValueError(
                "Page metadata must have shape [batch, kv_heads, clusters, pages]"
            )
        if primary.block_table.ndim != 2:
            raise ValueError("Block table must have shape [batch, blocks]")

        batch_size, num_kv_heads, max_primary_tokens = primary.token_indices.shape
        if source.page_token_counts.shape[:2] != (batch_size, num_kv_heads):
            raise ValueError("Primary and page metadata batch shapes must match")
        if primary.block_table.shape[0] != batch_size:
            raise ValueError("Block table batch size does not match exact metadata")
        if primary.key_cache.shape[2:] != (num_kv_heads, query.shape[2]):
            raise ValueError("Primary KV shape does not match query metadata")
        if query.shape[1] % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        integral_tensors = (
            primary.block_table,
            primary.token_indices,
            source.page_token_counts,
        )
        if any(
            tensor.dtype not in (torch.int32, torch.int64)
            for tensor in integral_tensors
        ):
            raise ValueError("Exact-attention metadata must be integral")

        tensors = (
            primary.key_cache,
            primary.value_cache,
            primary.block_table,
            primary.token_indices,
            primary.token_mask,
            source.page_token_counts,
        )
        if any(tensor.device != query.device for tensor in tensors):
            raise ValueError("All exact-attention tensors must be on the query device")

        if request_indices is None:
            if query.shape[0] != batch_size:
                raise ValueError(
                    "Identity request mapping requires one query per request"
                )
        else:
            if request_indices.shape != (query.shape[0],):
                raise ValueError("request_indices must contain one entry per query")
            if request_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("request_indices must be integral")
            if request_indices.device != query.device:
                raise ValueError("request_indices must be on the query device")

        max_page_slots = (
            source.page_token_counts.shape[2] * source.page_token_counts.shape[3]
        )
        if (
            max_page_slots
            and source.resident_pages is None
            and source.staging_pages is None
        ):
            raise RuntimeError(
                "Exact page metadata requires a resident or staging source"
            )

        expected_page_shape = source.page_token_counts.shape
        for page_source in (source.resident_pages, source.staging_pages):
            if page_source is None:
                continue
            if page_source.page_ids.shape != expected_page_shape:
                raise ValueError("Exact page IDs and token-count shapes must match")
            if page_source.page_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Exact page IDs must be integral")
            if page_source.key_pages.shape != page_source.value_pages.shape:
                raise ValueError("Exact key and value page shapes must match")
            if page_source.key_pages.ndim != 3:
                raise ValueError(
                    "Exact pages must have shape [pages, page_size, head_size]"
                )
            if page_source.key_pages.shape[1:] != (self.page_size, query.shape[2]):
                raise ValueError("Exact page shape does not match query")
            if (
                page_source.key_pages.dtype != query.dtype
                or page_source.value_pages.dtype != query.dtype
            ):
                raise ValueError("Exact page dtype does not match query")
            if any(
                tensor.device != query.device
                for tensor in (
                    page_source.page_ids,
                    page_source.key_pages,
                    page_source.value_pages,
                )
            ):
                raise ValueError("Exact pages must be on the query device")

        return num_kv_heads, max_primary_tokens, max_page_slots

    def run_parallel_full_verification(
        self,
        source: RetroSpecFullVerificationKVSource,
        query: torch.Tensor,
        local_keys: torch.Tensor,
        local_values: torch.Tensor,
        scale: float,
        query_start_loc: torch.Tensor,
        max_query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run specialized clustered-prefix and causal native-suffix attention."""
        if query.ndim != 3:
            raise ValueError("Query must have shape [queries, query_heads, head_size]")
        if query.device.type != "cuda":
            raise ValueError("Parallel full verification requires CUDA tensors")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Parallel full verification requires FP16 or BF16")
        if query.shape[0] > self.max_num_queries:
            raise ValueError("Query count exceeds exact-attention workspace capacity")
        if local_keys.shape != local_values.shape or local_keys.ndim != 3:
            raise ValueError("Local KV must have shape [queries, kv_heads, head_size]")
        if local_keys.shape[0] != query.shape[0]:
            raise ValueError("Local KV must contain one entry per query")
        if local_keys.dtype != query.dtype or local_keys.device != query.device:
            raise ValueError("Local KV layout must match the query")
        if query_start_loc.ndim != 1 or query_start_loc.device != query.device:
            raise ValueError("query_start_loc must be a one-dimensional CUDA tensor")
        if query_start_loc.dtype not in (torch.int32, torch.int64):
            raise ValueError("query_start_loc must be integral")
        if max_query_len <= 0 and query.shape[0] > 0:
            raise ValueError("max_query_len must be positive for non-empty queries")

        primary = source.primary
        if primary.key_cache.shape != primary.value_cache.shape:
            raise ValueError("Primary key and value cache shapes must match")
        if primary.key_cache.ndim != 4 or primary.key_cache.shape[1] != self.page_size:
            raise ValueError("Primary KV cache has an invalid page layout")
        if primary.key_cache.dtype != query.dtype:
            raise ValueError("Primary KV dtype does not match query")
        if primary.token_indices.shape != primary.token_mask.shape:
            raise ValueError("Primary token indices and mask shapes must match")
        if primary.token_indices.ndim != 3 or primary.token_mask.dtype != torch.bool:
            raise ValueError("Primary token metadata has an invalid layout")
        batch_size, num_kv_heads, max_primary_tokens = primary.token_indices.shape
        if query_start_loc.shape != (batch_size + 1,):
            raise ValueError("query_start_loc must contain one boundary per request")
        if primary.block_table.shape[0] != batch_size:
            raise ValueError("Block table batch size does not match primary metadata")
        if primary.key_cache.shape[2:] != (num_kv_heads, query.shape[2]):
            raise ValueError("Primary KV shape does not match query metadata")
        if local_keys.shape[1:] != (num_kv_heads, query.shape[2]):
            raise ValueError("Local KV shape does not match primary metadata")
        if query.shape[1] % num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads")
        primary_tensors = (
            primary.key_cache,
            primary.value_cache,
            primary.block_table,
            primary.token_indices,
            primary.token_mask,
        )
        if any(tensor.device != query.device for tensor in primary_tensors):
            raise ValueError("All primary KV tensors must use the query device")

        clustered = source.clustered
        if clustered is not None:
            if clustered.key_tokens.shape != clustered.value_tokens.shape:
                raise ValueError("Clustered key and value token shapes must match")
            if clustered.key_tokens.ndim != 2:
                raise ValueError("Clustered KV must have shape [tokens, head_size]")
            if clustered.key_tokens.shape[1] != query.shape[2]:
                raise ValueError("Clustered KV head size does not match query")
            if clustered.token_offsets.shape != (batch_size, num_kv_heads):
                raise ValueError("Clustered token offsets have an invalid shape")
            if clustered.token_counts.shape != (batch_size, num_kv_heads):
                raise ValueError("Clustered token counts have an invalid shape")
            cluster_tensors = (
                clustered.key_tokens,
                clustered.value_tokens,
                clustered.token_offsets,
                clustered.token_counts,
            )
            if any(tensor.device != query.device for tensor in cluster_tensors):
                raise ValueError("All clustered KV tensors must use the query device")
            if clustered.key_tokens.dtype != query.dtype:
                raise ValueError("Clustered KV dtype does not match query")
            if clustered.token_offsets.dtype not in (torch.int32, torch.int64):
                raise ValueError("Clustered token offsets must be integral")
            if clustered.token_counts.dtype not in (torch.int32, torch.int64):
                raise ValueError("Clustered token counts must be integral")

        num_queries, num_query_heads, head_size = query.shape
        num_splits = min(_PARALLEL_FULL_NUM_SPLITS, self._partition_capacity)
        self._ensure_workspace(query, num_splits)
        assert self._partial_output is not None
        assert self._partial_max is not None
        assert self._partial_sum is not None
        assert self._output is not None
        assert self._output_lse is not None
        assert self._secondary_output is not None
        assert self._secondary_output_lse is not None

        cluster_output = self._output[:num_queries]
        cluster_lse = self._output_lse[: num_query_heads * num_queries].view(
            num_query_heads, num_queries
        )
        native_output = self._secondary_output[:num_queries]
        native_lse = self._secondary_output_lse[: num_query_heads * num_queries].view(
            num_query_heads, num_queries
        )
        if num_queries == 0:
            return cluster_output, cluster_lse, native_output, native_lse

        current_stream = torch.cuda.current_stream(query.device)
        if primary.ready_event is not None:
            current_stream.wait_event(primary.ready_event)
        if clustered is not None and clustered.ready_event is not None:
            current_stream.wait_event(clustered.ready_event)

        block_d = triton.next_power_of_2(head_size)
        query_tiles = triton.cdiv(max_query_len, _PARALLEL_FULL_BLOCK_QUERIES)
        launch_grid = (batch_size, num_query_heads, query_tiles * num_splits)

        if clustered is None or clustered.max_tokens_per_head == 0:
            cluster_output.zero_()
            cluster_lse.fill_(float("-inf"))
        else:
            _parallel_cluster_prefix_kernel[launch_grid](
                query,
                query_start_loc,
                clustered.key_tokens,
                clustered.value_tokens,
                clustered.token_offsets,
                clustered.token_counts,
                self._partial_output,
                self._partial_max,
                self._partial_sum,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                clustered.key_tokens.stride(0),
                clustered.key_tokens.stride(1),
                clustered.value_tokens.stride(0),
                clustered.value_tokens.stride(1),
                clustered.token_offsets.stride(0),
                clustered.token_offsets.stride(1),
                self._partial_output.stride(0),
                self._partial_output.stride(1),
                self._partial_output.stride(2),
                self._partial_output.stride(3),
                self._partial_max.stride(0),
                self._partial_max.stride(1),
                self._partial_max.stride(2),
                scale,
                NUM_KV_HEADS=num_kv_heads,
                QUERIES_PER_KV_HEAD=num_query_heads // num_kv_heads,
                MAX_CLUSTER_TOKENS=clustered.max_tokens_per_head,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
                BLOCK_M=_PARALLEL_FULL_BLOCK_QUERIES,
                BLOCK_N=_PARALLEL_FULL_BLOCK_TOKENS,
                NUM_SPLITS=num_splits,
                num_warps=4,
                num_stages=2,
            )
            _reduce_exact_partitions_kernel[(num_queries, num_query_heads)](
                self._partial_output,
                self._partial_max,
                self._partial_sum,
                cluster_output,
                cluster_lse,
                self._partial_output.stride(0),
                self._partial_output.stride(1),
                self._partial_output.stride(2),
                self._partial_output.stride(3),
                self._partial_max.stride(0),
                self._partial_max.stride(1),
                self._partial_max.stride(2),
                cluster_output.stride(0),
                cluster_output.stride(1),
                cluster_output.stride(2),
                cluster_lse.stride(0),
                cluster_lse.stride(1),
                num_splits,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
            )

        _parallel_native_suffix_kernel[launch_grid](
            query,
            local_keys,
            local_values,
            query_start_loc,
            primary.key_cache,
            primary.value_cache,
            primary.block_table,
            primary.token_indices,
            primary.token_mask,
            self._partial_output,
            self._partial_max,
            self._partial_sum,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            local_keys.stride(0),
            local_keys.stride(1),
            local_keys.stride(2),
            local_values.stride(0),
            local_values.stride(1),
            local_values.stride(2),
            primary.key_cache.stride(0),
            primary.key_cache.stride(1),
            primary.key_cache.stride(2),
            primary.key_cache.stride(3),
            primary.value_cache.stride(0),
            primary.value_cache.stride(1),
            primary.value_cache.stride(2),
            primary.value_cache.stride(3),
            primary.block_table.stride(0),
            primary.block_table.stride(1),
            self._partial_output.stride(0),
            self._partial_output.stride(1),
            self._partial_output.stride(2),
            self._partial_output.stride(3),
            self._partial_max.stride(0),
            self._partial_max.stride(1),
            self._partial_max.stride(2),
            scale,
            NUM_KV_HEADS=num_kv_heads,
            QUERIES_PER_KV_HEAD=num_query_heads // num_kv_heads,
            MAX_PRIMARY_TOKENS=max_primary_tokens,
            MAX_QUERY_LEN=max_query_len,
            PAGE_SIZE=self.page_size,
            HEAD_SIZE=head_size,
            BLOCK_D=block_d,
            BLOCK_M=_PARALLEL_FULL_BLOCK_QUERIES,
            BLOCK_N=_PARALLEL_FULL_NATIVE_BLOCK_TOKENS,
            NUM_SPLITS=num_splits,
            num_warps=4,
            num_stages=2,
        )
        _reduce_exact_partitions_kernel[(num_queries, num_query_heads)](
            self._partial_output,
            self._partial_max,
            self._partial_sum,
            native_output,
            native_lse,
            self._partial_output.stride(0),
            self._partial_output.stride(1),
            self._partial_output.stride(2),
            self._partial_output.stride(3),
            self._partial_max.stride(0),
            self._partial_max.stride(1),
            self._partial_max.stride(2),
            native_output.stride(0),
            native_output.stride(1),
            native_output.stride(2),
            native_lse.stride(0),
            native_lse.stride(1),
            num_splits,
            HEAD_SIZE=head_size,
            BLOCK_D=block_d,
        )
        return cluster_output, cluster_lse, native_output, native_lse

    def run(
        self,
        source: RetroSpecExactKVSource,
        query: torch.Tensor,
        scale: float,
        request_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run exact attention without materializing a contiguous KV copy."""
        num_kv_heads, max_primary_tokens, max_page_slots = self._validate_source(
            source, query, request_indices
        )
        num_queries, num_query_heads, head_size = query.shape
        num_source_tokens = max_primary_tokens + max_page_slots * self.page_size
        num_partitions = triton.cdiv(
            max(num_source_tokens, 1), EXACT_ATTENTION_PARTITION_SIZE
        )
        self._ensure_workspace(query, num_partitions)

        assert self._partial_output is not None
        assert self._partial_max is not None
        assert self._partial_sum is not None
        assert self._output is not None
        assert self._output_lse is not None

        output = self._output[:num_queries]
        num_output_lse_elements = num_query_heads * num_queries
        output_lse = self._output_lse[:num_output_lse_elements].view(
            num_query_heads, num_queries
        )
        if num_queries == 0:
            return output, output_lse

        primary = source.primary
        current_stream = torch.cuda.current_stream(query.device)
        for exact_source in (
            primary,
            source.resident_pages,
            source.staging_pages,
        ):
            if exact_source is not None and exact_source.ready_event is not None:
                current_stream.wait_event(exact_source.ready_event)

        resident = source.resident_pages
        staging = source.staging_pages
        dummy_page_ids = source.page_token_counts
        dummy_key_pages = primary.key_cache[:, :, 0, :]
        dummy_value_pages = primary.value_cache[:, :, 0, :]
        resident_page_ids = dummy_page_ids if resident is None else resident.page_ids
        resident_key_pages = dummy_key_pages if resident is None else resident.key_pages
        resident_value_pages = (
            dummy_value_pages if resident is None else resident.value_pages
        )
        staging_page_ids = dummy_page_ids if staging is None else staging.page_ids
        staging_key_pages = dummy_key_pages if staging is None else staging.key_pages
        staging_value_pages = (
            dummy_value_pages if staging is None else staging.value_pages
        )
        request_mapping = (
            primary.token_indices if request_indices is None else request_indices
        )
        block_d = triton.next_power_of_2(head_size)

        _multi_source_exact_partition_kernel[
            (
                num_queries,
                num_query_heads,
                num_partitions,
            )
        ](
            query,
            request_mapping,
            primary.key_cache,
            primary.value_cache,
            primary.block_table,
            primary.token_indices,
            primary.token_mask,
            source.page_token_counts,
            resident_page_ids,
            resident_key_pages,
            resident_value_pages,
            staging_page_ids,
            staging_key_pages,
            staging_value_pages,
            self._partial_output,
            self._partial_max,
            self._partial_sum,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            primary.key_cache.stride(0),
            primary.key_cache.stride(1),
            primary.key_cache.stride(2),
            primary.key_cache.stride(3),
            primary.value_cache.stride(0),
            primary.value_cache.stride(1),
            primary.value_cache.stride(2),
            primary.value_cache.stride(3),
            primary.block_table.stride(0),
            primary.block_table.stride(1),
            resident_key_pages.stride(0),
            resident_key_pages.stride(1),
            resident_key_pages.stride(2),
            resident_value_pages.stride(0),
            resident_value_pages.stride(1),
            resident_value_pages.stride(2),
            staging_key_pages.stride(0),
            staging_key_pages.stride(1),
            staging_key_pages.stride(2),
            staging_value_pages.stride(0),
            staging_value_pages.stride(1),
            staging_value_pages.stride(2),
            self._partial_output.stride(0),
            self._partial_output.stride(1),
            self._partial_output.stride(2),
            self._partial_output.stride(3),
            self._partial_max.stride(0),
            self._partial_max.stride(1),
            self._partial_max.stride(2),
            scale,
            NUM_KV_HEADS=num_kv_heads,
            QUERIES_PER_KV_HEAD=num_query_heads // num_kv_heads,
            MAX_PRIMARY_TOKENS=max_primary_tokens,
            MAX_PAGE_SLOTS=max_page_slots,
            PAGE_SIZE=self.page_size,
            HEAD_SIZE=head_size,
            BLOCK_D=block_d,
            PARTITION_SIZE=EXACT_ATTENTION_PARTITION_SIZE,
            BLOCK_TOKENS=_EXACT_ATTENTION_BLOCK_TOKENS,
            IDENTITY_REQUESTS=request_indices is None,
            HAS_RESIDENT=resident is not None,
            HAS_STAGING=staging is not None,
        )

        _reduce_exact_partitions_kernel[(num_queries, num_query_heads)](
            self._partial_output,
            self._partial_max,
            self._partial_sum,
            output,
            output_lse,
            self._partial_output.stride(0),
            self._partial_output.stride(1),
            self._partial_output.stride(2),
            self._partial_output.stride(3),
            self._partial_max.stride(0),
            self._partial_max.stride(1),
            self._partial_max.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output_lse.stride(0),
            output_lse.stride(1),
            num_partitions,
            HEAD_SIZE=head_size,
            BLOCK_D=block_d,
        )
        return output, output_lse
