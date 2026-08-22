# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec


@dataclass(frozen=True)
class RetroSpecLongContextCapacity:
    native_working_set_tokens: int
    native_num_blocks: int
    native_memory_bytes: int
    auxiliary_memory_bytes: int

    @property
    def total_memory_bytes(self) -> int:
        return self.native_memory_bytes + self.auxiliary_memory_bytes


def uses_retrospec_cpu_offload(vllm_config: VllmConfig) -> bool:
    config = vllm_config.speculative_config
    return (
        config is not None
        and config.method == "retrospec"
        and config.retrospec_index_mode == "segmented_cluster"
        and config.retrospec_cache_mode == "cpu_offload"
    )


def get_retrospec_native_working_set_tokens(
    vllm_config: VllmConfig,
    block_size: int,
) -> int:
    config = vllm_config.speculative_config
    if config is None or config.method != "retrospec":
        raise ValueError("RetroSpec capacity requires a RetroSpec configuration")
    if config.num_speculative_tokens is None:
        raise ValueError("RetroSpec requires num_speculative_tokens")

    scheduler_config = vllm_config.scheduler_config
    max_prefill_chunk = scheduler_config.max_num_batched_tokens
    long_prefill_threshold = scheduler_config.long_prefill_token_threshold
    if long_prefill_threshold > 0:
        max_prefill_chunk = min(max_prefill_chunk, long_prefill_threshold)

    num_recent_blocks = cdiv(config.num_speculative_tokens, block_size) + 1

    # Newly scheduled prefill tokens cannot be retired until the current model
    # execution has built and committed every complete segment they contain.
    working_set_tokens = (
        block_size
        + config.retrospec_index_segment_size
        + num_recent_blocks * block_size
        + max_prefill_chunk
        + config.num_speculative_tokens
    )
    working_set_tokens = cdiv(working_set_tokens, block_size) * block_size

    return min(working_set_tokens, vllm_config.model_config.max_model_len)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(value, 1) - 1).bit_length()


def _get_attention_specs(
    kv_cache_specs: Mapping[str, KVCacheSpec],
) -> tuple[AttentionSpec, ...]:
    specs = tuple(kv_cache_specs.values())
    if not specs:
        raise ValueError("RetroSpec requires at least one KV cache layer")
    if not all(isinstance(spec, AttentionSpec) for spec in specs):
        raise NotImplementedError(
            "RetroSpec long-context capacity supports attention KV caches only"
        )

    attention_specs = tuple(spec for spec in specs if isinstance(spec, AttentionSpec))
    block_sizes = {spec.block_size for spec in attention_specs}
    if len(block_sizes) != 1:
        raise NotImplementedError(
            "RetroSpec long-context capacity requires one KV block size"
        )

    return attention_specs


def _get_indexed_token_capacity(
    vllm_config: VllmConfig,
    block_size: int,
) -> int:
    config = vllm_config.speculative_config
    assert config is not None
    assert config.num_speculative_tokens is not None

    max_model_len = vllm_config.model_config.max_model_len
    num_recent_blocks = cdiv(config.num_speculative_tokens, block_size) + 1

    full_block_count = max_model_len // block_size
    stable_end_block = max(full_block_count - num_recent_blocks, 1)
    indexable_tokens = (stable_end_block - 1) * block_size

    complete_segments = indexable_tokens // config.retrospec_index_segment_size
    return complete_segments * config.retrospec_index_segment_size


def build_retrospec_long_context_capacity(
    vllm_config: VllmConfig,
    kv_cache_specs: Mapping[str, KVCacheSpec],
) -> RetroSpecLongContextCapacity:
    if not uses_retrospec_cpu_offload(vllm_config):
        raise ValueError(
            "RetroSpec long-context capacity requires CPU-backed segmented storage"
        )

    config = vllm_config.speculative_config
    assert config is not None
    assert config.num_speculative_tokens is not None

    attention_specs = _get_attention_specs(kv_cache_specs)
    block_size = attention_specs[0].block_size

    segment_size = config.retrospec_index_segment_size
    tokens_per_cluster = config.retrospec_blocks_per_cluster * block_size

    if segment_size % block_size != 0:
        raise ValueError("retrospec_index_segment_size must be divisible by block_size")
    if segment_size % tokens_per_cluster != 0:
        raise ValueError(
            "retrospec_index_segment_size must be divisible by "
            "retrospec_blocks_per_cluster * block_size"
        )

    native_tokens = get_retrospec_native_working_set_tokens(
        vllm_config,
        block_size,
    )

    # Block zero is vLLM's physical null block and cannot store request KV.
    native_num_blocks = cdiv(native_tokens, block_size) + 1
    native_memory_bytes = sum(
        native_num_blocks * spec.page_size_bytes for spec in attention_specs
    )

    indexed_tokens = _get_indexed_token_capacity(vllm_config, block_size)
    num_clusters = indexed_tokens // tokens_per_cluster

    all_layer_token_bytes = sum(
        spec.real_page_size_bytes // block_size for spec in attention_specs
    )

    # Segment records retain centroid K/V, while the active packed index owns
    # one additional GPU copy of the same summaries.
    cluster_summary_bytes = 2 * num_clusters * all_layer_token_bytes

    # Sum(ceil(cluster_size / block_size)) is bounded by the raw page count
    # plus one partially filled page for every cluster.
    cluster_pages_per_head = cdiv(indexed_tokens, block_size) + num_clusters

    effective_cache_ratio = config.retrospec_cache_ratio
    if effective_cache_ratio == 0.0:
        effective_cache_ratio = min(config.retrospec_retrieval_ratio * 3.0, 1.0)

    resident_pages_per_head = ceil(cluster_pages_per_head * effective_cache_ratio)
    resident_cache_bytes = sum(
        resident_pages_per_head * spec.real_page_size_bytes for spec in attention_specs
    )

    # Cluster IDs, page IDs, page counts, token counts and packed masks remain
    # on GPU together with the centroid vectors.
    cluster_metadata_bytes = sum(
        num_clusters * spec.num_kv_heads * 32 for spec in attention_specs
    )

    max_model_len = vllm_config.model_config.max_model_len
    max_full_verify_workspace = 0
    max_cluster_build_workspace = 0

    scheduler_config = vllm_config.scheduler_config
    max_prefill_chunk = scheduler_config.max_num_batched_tokens
    if scheduler_config.long_prefill_token_threshold > 0:
        max_prefill_chunk = min(
            max_prefill_chunk,
            scheduler_config.long_prefill_token_threshold,
        )

    cluster_build_tokens = min(indexed_tokens, segment_size + max_prefill_chunk)

    for spec in attention_specs:
        num_kv_heads = spec.num_kv_heads
        per_head_page_bytes = spec.real_page_size_bytes // num_kv_heads
        per_head_token_bytes = per_head_page_bytes // block_size

        # Full-verification pages and the continuous exact-attention input each
        # use one layer-shared growable power-of-two arena.
        transfer_pages = _next_power_of_two(
            max(64, num_kv_heads * cluster_pages_per_head)
        )
        transfer_buffer_bytes = transfer_pages * per_head_page_bytes

        execution_tokens = _next_power_of_two(max(1024, num_kv_heads * max_model_len))
        execution_buffer_bytes = execution_tokens * per_head_token_bytes

        max_full_verify_workspace = max(
            max_full_verify_workspace,
            transfer_buffer_bytes + execution_buffer_bytes,
        )

        # GPU k-means temporarily retains token K/V, assignments, centroids
        # and float accumulation buffers.
        layer_token_bytes = spec.real_page_size_bytes // block_size
        max_cluster_build_workspace = max(
            max_cluster_build_workspace,
            4 * cluster_build_tokens * layer_token_bytes,
        )

    num_query_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    max_kv_heads = max(spec.num_kv_heads for spec in attention_specs)

    # Float logits/scores plus top-k values and indices, shared across layers.
    selection_workspace_bytes = num_clusters * (4 * num_query_heads + 24 * max_kv_heads)

    phase_workspace_bytes = max(
        max_full_verify_workspace + selection_workspace_bytes,
        max_cluster_build_workspace,
    )

    auxiliary_memory_bytes = (
        cluster_summary_bytes
        + resident_cache_bytes
        + cluster_metadata_bytes
        + phase_workspace_bytes
    )

    # Cover allocator rounding, CUDA events and small metadata tensors.
    auxiliary_memory_bytes = ceil(auxiliary_memory_bytes * 1.1)

    return RetroSpecLongContextCapacity(
        native_working_set_tokens=native_tokens,
        native_num_blocks=native_num_blocks,
        native_memory_bytes=native_memory_bytes,
        auxiliary_memory_bytes=auxiliary_memory_bytes,
    )
