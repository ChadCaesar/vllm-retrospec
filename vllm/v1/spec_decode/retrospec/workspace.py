# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

EXACT_ATTENTION_PARTITION_SIZE = 1024


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
    output_bytes = max_num_queries * num_query_heads * head_size * dtype_size
    output_lse_bytes = max_num_queries * num_query_heads * 4
    return (
        partial_output_bytes
        + partial_statistics_bytes
        + output_bytes
        + output_lse_bytes
    )
