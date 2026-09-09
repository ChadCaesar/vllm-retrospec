# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

EXACT_ATTENTION_PARTITION_SIZE = 1024


def exact_attention_primary_token_capacity(
    max_model_len: int,
    prefill_segment_size: int,
    generation_update_interval: int,
    num_speculative_tokens: int,
    block_size: int,
) -> int:
    """Return the fixed primary-token width used by selection plans."""
    dimensions = {
        "max_model_len": max_model_len,
        "prefill_segment_size": prefill_segment_size,
        "generation_update_interval": generation_update_interval,
        "num_speculative_tokens": num_speculative_tokens,
        "block_size": block_size,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    num_recent_blocks = (num_speculative_tokens + block_size - 1) // block_size + 1
    max_unindexed_tokens = max(
        prefill_segment_size,
        generation_update_interval,
    )
    primary_token_capacity = max_unindexed_tokens + (num_recent_blocks + 1) * block_size
    return min(primary_token_capacity, max_model_len)


def exact_attention_query_capacity(
    max_num_seqs: int,
    num_speculative_tokens: int,
) -> int:
    """Return the maximum query count of one RetroSpec exact-attention pass."""
    dimensions = {
        "max_num_seqs": max_num_seqs,
        "num_speculative_tokens": num_speculative_tokens,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    # Sparse/expanded verification contains at most num_speculative_tokens per
    # request. Full verification additionally computes one target correction
    # token, so it defines the shared workspace upper bound.
    return max_num_seqs * (num_speculative_tokens + 1)


def exact_attention_source_token_capacity(
    primary_token_capacity: int,
    cluster_page_slot_capacity: int,
    page_size: int,
) -> int:
    """Convert exact-attention metadata slots to source-token capacity."""
    if primary_token_capacity < 0:
        raise ValueError("primary_token_capacity must be non-negative")
    if cluster_page_slot_capacity < 0:
        raise ValueError("cluster_page_slot_capacity must be non-negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    return max(
        primary_token_capacity + cluster_page_slot_capacity * page_size,
        1,
    )


def exact_attention_partition_capacity(max_num_source_tokens: int) -> int:
    """Return the power-of-two partition capacity for exact attention."""
    if max_num_source_tokens <= 0:
        raise ValueError("max_num_source_tokens must be positive")

    num_partitions = (
        max_num_source_tokens + EXACT_ATTENTION_PARTITION_SIZE - 1
    ) // EXACT_ATTENTION_PARTITION_SIZE
    return 1 << (num_partitions - 1).bit_length()


def exact_attention_workspace_size_bytes(
    max_num_queries: int,
    num_query_heads: int,
    head_size: int,
    dtype_size: int,
    partition_capacity: int,
) -> int:
    """Return the complete exact-attention workspace allocation size."""
    dimensions = {
        "max_num_queries": max_num_queries,
        "num_query_heads": num_query_heads,
        "head_size": head_size,
        "dtype_size": dtype_size,
        "partition_capacity": partition_capacity,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    partial_output_bytes = (
        max_num_queries * num_query_heads * partition_capacity * head_size * dtype_size
    )
    partial_statistics_bytes = (
        max_num_queries * num_query_heads * partition_capacity * 2 * 4
    )
    # Cluster-prefix and native-suffix states coexist until their final LSE
    # merge, while the larger split-K partial workspace is reused serially.
    output_bytes = 2 * max_num_queries * num_query_heads * head_size * dtype_size
    output_lse_bytes = 2 * max_num_queries * num_query_heads * 4
    # One normalized output/max/sum state combines fixed-capacity partition
    # waves when a physically padded page descriptor exceeds the planned
    # single-wave width. This state is request-sized, not context-sized.
    accumulated_state_bytes = (
        max_num_queries * num_query_heads * (head_size * dtype_size + 2 * 4)
    )
    return (
        partial_output_bytes
        + partial_statistics_bytes
        + output_bytes
        + output_lse_bytes
        + accumulated_state_bytes
    )
