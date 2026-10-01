# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU-native capacity contracts.

The branch keeps complete target KV on GPU. Offload working-set and
CPU-staging capacity tests do not describe its runtime memory policy.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.spec_decode.retrospec.capacity import (
    RetroSpecGPUIndexFootprint,
    estimate_retrospec_gpu_index_arena_bytes,
    estimate_retrospec_gpu_index_footprint,
    get_retrospec_gpu_index_descriptor_bytes,
    is_retrospec_long_context_enabled,
)

pytestmark = pytest.mark.cpu_test


def make_config(
    *,
    max_model_len: int = 65536,
    max_num_seqs: int = 1,
    **overrides: Any,
) -> VllmConfig:
    spec_values = {
        "method": "retrospec",
        "num_speculative_tokens": 64,
        "retrospec_index_segment_size": 8192,
        "retrospec_index_update_interval": 1024,
        "retrospec_blocks_per_cluster": 1,
        "retrospec_retrieval_ratio": 0.018,
        **overrides,
    }
    return cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SimpleNamespace(**spec_values),
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=1024,
                long_prefill_token_threshold=0,
                enable_chunked_prefill=True,
                disable_hybrid_kv_cache_manager=False,
            ),
            model_config=SimpleNamespace(
                max_model_len=max_model_len,
                original_max_model_len=max_model_len,
                get_num_attention_heads=lambda _: 8,
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        ),
    )


def make_kv_cache_specs(num_layers: int = 2) -> dict[str, FullAttentionSpec]:
    return {
        f"layer.{layer_index}": FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        )
        for layer_index in range(num_layers)
    }


def make_scheduler_kv_cache_config(num_layers: int = 2) -> KVCacheConfig:
    spec = next(iter(make_kv_cache_specs(num_layers=1).values()))
    return KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"layer.{index}" for index in range(num_layers)],
                kv_cache_spec=spec,
            )
        ],
    )


def test_gpu_native_never_uses_offload_working_set_capacity():
    config = make_config(max_model_len=65536)
    assert not is_retrospec_long_context_enabled(config)
    config.model_config.max_model_len = 8192
    assert not is_retrospec_long_context_enabled(config)
    config.speculative_config.method = "ngram"
    assert not is_retrospec_long_context_enabled(config)


def test_gpu_index_footprint_matches_stable_cluster_and_page_layout():
    config = make_config(max_model_len=2048)
    footprint = estimate_retrospec_gpu_index_footprint(
        config, make_scheduler_kv_cache_config(), max_context_tokens=1024
    )
    assert footprint == RetroSpecGPUIndexFootprint(64, 128)


def test_gpu_index_footprint_is_empty_before_stable_prefix_exists():
    config = make_config(max_model_len=2048)
    footprint = estimate_retrospec_gpu_index_footprint(
        config, make_scheduler_kv_cache_config(), max_context_tokens=80
    )
    assert footprint == RetroSpecGPUIndexFootprint(0, 0)


def test_gpu_index_descriptor_bytes_match_worker_tensor_layout():
    config = make_config(max_num_seqs=4)
    descriptor_bytes = get_retrospec_gpu_index_descriptor_bytes(
        config, make_scheduler_kv_cache_config(num_layers=2)
    )
    assert descriptor_bytes == 2 * 4 * (44 + 4 * 2)


def test_gpu_index_arena_projects_packed_growth():
    config = make_config(max_num_seqs=4)
    kv_cache_config = make_scheduler_kv_cache_config(num_layers=2)
    footprint = RetroSpecGPUIndexFootprint(64, 128)

    one_request = estimate_retrospec_gpu_index_arena_bytes(
        config, kv_cache_config, (footprint,)
    )
    two_requests = estimate_retrospec_gpu_index_arena_bytes(
        config, kv_cache_config, (footprint, footprint)
    )
    three_requests = estimate_retrospec_gpu_index_arena_bytes(
        config, kv_cache_config, (footprint, footprint, footprint)
    )
    assert one_request < two_requests < three_requests
    assert three_requests > 2 * one_request


def test_gpu_index_arena_ignores_empty_request_footprints():
    config = make_config()
    kv_cache_config = make_scheduler_kv_cache_config()
    footprint = RetroSpecGPUIndexFootprint(64, 128)
    expected = estimate_retrospec_gpu_index_arena_bytes(
        config, kv_cache_config, (footprint,)
    )
    actual = estimate_retrospec_gpu_index_arena_bytes(
        config,
        kv_cache_config,
        (RetroSpecGPUIndexFootprint(0, 0), footprint),
    )
    assert actual == expected
    assert estimate_retrospec_gpu_index_arena_bytes(config, kv_cache_config, ()) == 0


def test_long_prompt_requires_complete_native_kv_capacity():
    config = make_config(max_model_len=65536)
    spec = next(iter(make_kv_cache_specs(num_layers=1).values()))
    full_context_blocks = config.model_config.max_model_len // spec.block_size

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(config, [{"layer.0": spec}], [1000 * spec.page_size_bytes])

    available_blocks = full_context_blocks + 128
    kv_cache_configs = get_kv_cache_configs(
        config, [{"layer.0": spec}], [available_blocks * spec.page_size_bytes]
    )
    assert kv_cache_configs[0].num_blocks == available_blocks
    assert kv_cache_configs[0].kv_cache_tensors[0].size == (
        available_blocks * spec.page_size_bytes
    )


def test_multiple_scheduler_slots_do_not_reduce_native_kv_allocation():
    spec = next(iter(make_kv_cache_specs(num_layers=1).values()))
    available_blocks = 5000
    memory = available_blocks * spec.page_size_bytes
    single = get_kv_cache_configs(
        make_config(max_num_seqs=1), [{"layer.0": spec}], [memory]
    )[0]
    multiple = get_kv_cache_configs(
        make_config(max_num_seqs=2), [{"layer.0": spec}], [memory]
    )[0]
    assert single.num_blocks == multiple.num_blocks == available_blocks


def test_max_concurrency_uses_complete_native_context():
    config = make_config(max_model_len=65536)
    spec = next(iter(make_kv_cache_specs(num_layers=1).values()))
    kv_cache_config = KVCacheConfig(
        num_blocks=8192,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    assert get_max_concurrency_for_kv_cache_config(config, kv_cache_config) == 2.0
