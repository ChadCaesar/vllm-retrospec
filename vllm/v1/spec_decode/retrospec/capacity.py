# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec

from .workspace import (
    exact_attention_partition_capacity,
    exact_attention_primary_token_capacity,
    exact_attention_source_token_capacity,
)

_GPU_INDEX_CLUSTER_METADATA_BYTES = 28
_GPU_INDEX_PAGE_METADATA_BYTES = 12
_GPU_INDEX_REQUEST_SCALAR_BYTES = 44
_GPU_INDEX_ADMISSION_SAFETY_FACTOR = 1.10


@dataclass(frozen=True)
class RetroSpecGPUIndexFootprint:
    cluster_capacity: int
    page_capacity: int


@dataclass(frozen=True)
class RetroSpecLongContextCapacity:
    native_working_set_tokens: int
    native_num_blocks: int
    native_memory_bytes: int
    auxiliary_memory_bytes: int

    @property
    def total_memory_bytes(self) -> int:
        return self.native_memory_bytes + self.auxiliary_memory_bytes


def is_retrospec_long_context_enabled(vllm_config: VllmConfig) -> bool:
    config = vllm_config.speculative_config
    if config is None or config.method != "retrospec":
        return False
    # The GPU-native branch retains the complete native KV cache. It cannot
    # use the smaller offload working-set capacity or retire native blocks.
    return False


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
    max_model_len = vllm_config.model_config.max_model_len

    max_prefill_chunk = scheduler_config.max_num_batched_tokens
    long_prefill_threshold = scheduler_config.long_prefill_token_threshold
    if long_prefill_threshold > 0:
        max_prefill_chunk = min(max_prefill_chunk, long_prefill_threshold)

    num_recent_blocks = cdiv(config.num_speculative_tokens, block_size) + 1

    max_unindexed_segment_tokens = max(
        config.retrospec_index_segment_size,
        config.retrospec_index_update_interval,
    )

    # Size the native KV allocation for one active request. The scheduler and
    # request residency managers admit additional requests from the remaining
    # runtime budget instead of reserving max_num_seqs copies at startup.
    per_request_steady_tokens = (
        block_size
        + max_unindexed_segment_tokens
        + num_recent_blocks * block_size
        + config.num_speculative_tokens
    )
    per_request_steady_tokens = min(per_request_steady_tokens, max_model_len)

    working_set_tokens = min(
        per_request_steady_tokens + max_prefill_chunk,
        max_model_len,
    )
    working_set_tokens = cdiv(working_set_tokens, block_size) * block_size

    maximum_rounded_tokens = cdiv(max_model_len, block_size) * block_size
    return min(working_set_tokens, maximum_rounded_tokens)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(value, 1) - 1).bit_length()


def _get_indexed_token_capacity(
    vllm_config: VllmConfig,
    block_size: int,
) -> int:
    return _get_indexed_token_capacity_for_length(
        vllm_config,
        vllm_config.model_config.max_model_len,
        block_size,
    )


def _get_indexed_token_capacity_for_length(
    vllm_config: VllmConfig, context_len: int, block_size: int
) -> int:
    if context_len <= 0:
        raise ValueError("context_len must be positive")

    config = vllm_config.speculative_config
    assert config is not None
    assert config.num_speculative_tokens is not None

    context_len = min(context_len, vllm_config.model_config.max_model_len)
    num_recent_blocks = cdiv(config.num_speculative_tokens, block_size) + 1

    full_block_count = context_len // block_size
    stable_end_block = max(full_block_count - num_recent_blocks, 1)
    return (stable_end_block - 1) * block_size


def _get_scheduler_attention_specs(
    kv_cache_config: KVCacheConfig,
) -> tuple[tuple[int, AttentionSpec], ...]:
    specs: list[tuple[int, AttentionSpec]] = []
    for group in kv_cache_config.kv_cache_groups:
        num_layers = len(group.layer_names)
        if num_layers == 0:
            continue

        spec = group.kv_cache_spec
        if not isinstance(spec, AttentionSpec):
            raise NotImplementedError(
                "RetroSpec GPU index admission supports attention KV caches only"
            )
        specs.append((num_layers, spec))

    if not specs:
        raise ValueError("RetroSpec GPU index admission requires attention layers")
    return tuple(specs)


def get_retrospec_gpu_index_descriptor_bytes(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> int:
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    return sum(
        num_layers
        * max_num_seqs
        * (_GPU_INDEX_REQUEST_SCALAR_BYTES + 4 * spec.num_kv_heads)
        for num_layers, spec in _get_scheduler_attention_specs(kv_cache_config)
    )


def estimate_retrospec_gpu_index_footprint(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    max_context_tokens: int,
) -> RetroSpecGPUIndexFootprint:
    specs = _get_scheduler_attention_specs(kv_cache_config)
    block_sizes = {spec.block_size for _, spec in specs}
    if len(block_sizes) != 1:
        raise NotImplementedError(
            "RetroSpec GPU index admission requires one KV block size"
        )

    block_size = next(iter(block_sizes))
    config = vllm_config.speculative_config
    assert config is not None

    tokens_per_cluster = config.retrospec_blocks_per_cluster * block_size
    indexed_tokens = _get_indexed_token_capacity_for_length(
        vllm_config, max_context_tokens, block_size
    )
    indexed_tokens = indexed_tokens // tokens_per_cluster * tokens_per_cluster
    if indexed_tokens == 0:
        return RetroSpecGPUIndexFootprint(0, 0)

    num_clusters = indexed_tokens // tokens_per_cluster
    return RetroSpecGPUIndexFootprint(
        cluster_capacity=_next_power_of_two(num_clusters),
        page_capacity=_next_power_of_two(
            cdiv(indexed_tokens, block_size) + num_clusters
        ),
    )


def estimate_retrospec_gpu_index_arena_bytes(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    footprints: Iterable[RetroSpecGPUIndexFootprint],
) -> int:
    nonempty = tuple(
        footprint
        for footprint in footprints
        if footprint.cluster_capacity and footprint.page_capacity
    )
    if not nonempty:
        return 0

    cluster_capacity = _next_power_of_two(
        max(64, sum(item.cluster_capacity for item in nonempty))
    )
    page_capacity = _next_power_of_two(
        max(64, sum(item.page_capacity for item in nonempty))
    )

    arena_bytes = get_retrospec_gpu_index_descriptor_bytes(vllm_config, kv_cache_config)
    for num_layers, spec in _get_scheduler_attention_specs(kv_cache_config):
        arena_bytes += (
            num_layers
            * spec.num_kv_heads
            * (
                cluster_capacity
                * (
                    _GPU_INDEX_CLUSTER_METADATA_BYTES
                    + 2 * spec.head_size * get_dtype_size(spec.dtype)
                )
                + page_capacity * _GPU_INDEX_PAGE_METADATA_BYTES
            )
        )

    return ceil(arena_bytes * _GPU_INDEX_ADMISSION_SAFETY_FACTOR)


def get_retrospec_exact_attention_source_token_capacity(
    vllm_config: VllmConfig,
    block_size: int,
) -> int:
    """Return the maximum physical source-token slots per request."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    config = vllm_config.speculative_config
    if config is None or config.method != "retrospec":
        raise ValueError("RetroSpec capacity requires a RetroSpec configuration")
    if config.num_speculative_tokens is None:
        raise ValueError("RetroSpec requires num_speculative_tokens")

    indexed_tokens = _get_indexed_token_capacity(vllm_config, block_size)
    tokens_per_cluster = config.retrospec_blocks_per_cluster * block_size
    num_clusters = cdiv(indexed_tokens, tokens_per_cluster)

    # Every cluster may leave one partially occupied page. The first term
    # covers the indexed tokens and the second bounds cluster tail pages.
    max_cluster_pages = cdiv(indexed_tokens, block_size) + num_clusters

    # The plan table retains a fixed primary width even after most of those
    # slots become masked by later index publication. Capacity must describe
    # that physical metadata layout rather than only its valid logical suffix.
    max_primary_tokens = exact_attention_primary_token_capacity(
        max_model_len=vllm_config.model_config.max_model_len,
        prefill_segment_size=config.retrospec_index_segment_size,
        generation_update_interval=config.retrospec_index_update_interval,
        num_speculative_tokens=config.num_speculative_tokens,
        block_size=block_size,
    )
    return exact_attention_source_token_capacity(
        max_primary_tokens,
        max_cluster_pages,
        block_size,
    )


def get_retrospec_exact_attention_partition_capacity(
    vllm_config: VllmConfig,
    block_size: int,
) -> int:
    max_num_source_tokens = get_retrospec_exact_attention_source_token_capacity(
        vllm_config, block_size
    )
    return exact_attention_partition_capacity(max_num_source_tokens)


def build_retrospec_long_context_capacity(
    vllm_config: VllmConfig,
    kv_cache_specs: Mapping[str, KVCacheSpec],
) -> RetroSpecLongContextCapacity:
    # Keep the former import and failure behavior for callers of this helper.
    # GPU-native RetroSpec retains the full native KV cache.
    raise ValueError("RetroSpec long-context capacity requires RetroSpec")
