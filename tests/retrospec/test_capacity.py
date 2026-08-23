# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    RetroSpecLongContextCapacity,
    build_retrospec_long_context_capacity,
    get_retrospec_native_working_set_tokens,
    uses_retrospec_cpu_offload,
)

pytestmark = pytest.mark.cpu_test


def make_capacity_config(
    *,
    max_model_len: int = 65536,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 4096,
    long_prefill_token_threshold: int = 0,
    **overrides: Any,
) -> VllmConfig:
    spec_values = {
        "method": "retrospec",
        "num_speculative_tokens": 64,
        "retrospec_index_mode": "segmented_cluster",
        "retrospec_cache_mode": "cpu_offload",
        "retrospec_index_segment_size": 8192,
        "retrospec_index_update_interval": 1024,
        "retrospec_blocks_per_cluster": 1,
        "retrospec_retrieval_ratio": 0.018,
        "retrospec_cache_ratio": 0.0,
        **overrides,
    }
    return cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SimpleNamespace(**spec_values),
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                long_prefill_token_threshold=long_prefill_token_threshold,
            ),
            model_config=SimpleNamespace(
                max_model_len=max_model_len,
                get_num_attention_heads=lambda _: 8,
            ),
            parallel_config=SimpleNamespace(),
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


def make_engine_config(
    *,
    max_model_len: int = 65536,
    max_num_seqs: int = 1,
    enable_chunked_prefill: bool = True,
) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=max_model_len,
                original_max_model_len=max_model_len,
                get_num_attention_heads=lambda _: 8,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=1024,
                long_prefill_token_threshold=0,
                enable_chunked_prefill=enable_chunked_prefill,
                disable_hybrid_kv_cache_manager=False,
            ),
            speculative_config=SimpleNamespace(
                method="retrospec",
                num_speculative_tokens=64,
                retrospec_index_mode="segmented_cluster",
                retrospec_cache_mode="cpu_offload",
                retrospec_index_segment_size=8192,
                retrospec_index_update_interval=1024,
                retrospec_blocks_per_cluster=1,
                retrospec_retrieval_ratio=0.018,
                retrospec_cache_ratio=0.0,
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        ),
    )


def test_cpu_offload_capacity_mode_detection():
    config = make_capacity_config()
    assert uses_retrospec_cpu_offload(config)

    config.speculative_config.retrospec_cache_mode = "gpu_reference"
    assert not uses_retrospec_cpu_offload(config)

    config.speculative_config.retrospec_cache_mode = "cpu_offload"
    config.speculative_config.retrospec_index_mode = "block_mean"
    assert not uses_retrospec_cpu_offload(config)


def test_native_working_set_includes_unretired_prefill_chunk():
    config = make_capacity_config()

    # sink 16 + segment 8192 + recent 80 + prefill 4096 + lookahead 64
    assert get_retrospec_native_working_set_tokens(config, 16) == 12448


def test_native_working_set_honors_long_prefill_threshold():
    config = make_capacity_config(long_prefill_token_threshold=1024)
    assert get_retrospec_native_working_set_tokens(config, 16) == 9376


def test_native_working_set_uses_larger_generation_update_interval():
    config = make_capacity_config(retrospec_index_update_interval=16384)

    assert get_retrospec_native_working_set_tokens(config, 16) == 20640


def test_native_working_set_scales_request_state_but_shares_prefill_chunk():
    config = make_capacity_config(max_num_seqs=2)

    # 2 * (sink 16 + segment 8192 + recent 80 + lookahead 64)
    # + one batch-wide prefill chunk 4096.
    assert get_retrospec_native_working_set_tokens(config, 16) == 20800


def test_native_working_set_is_capped_by_max_model_len():
    config = make_capacity_config(max_model_len=4096)
    assert get_retrospec_native_working_set_tokens(config, 16) == 4096


def test_capacity_reserves_null_block_and_auxiliary_buffers():
    config = make_capacity_config()
    specs = make_kv_cache_specs()
    capacity = build_retrospec_long_context_capacity(config, specs)

    assert capacity.native_working_set_tokens == 12448
    assert capacity.native_num_blocks == 779
    assert capacity.native_memory_bytes == sum(
        capacity.native_num_blocks * spec.page_size_bytes for spec in specs.values()
    )
    assert capacity.auxiliary_memory_bytes > 0
    assert capacity.total_memory_bytes > capacity.native_memory_bytes


def test_capacity_reserves_indices_and_pages_for_all_resident_requests():
    single = build_retrospec_long_context_capacity(
        make_capacity_config(max_num_seqs=1), make_kv_cache_specs()
    )
    multiple = build_retrospec_long_context_capacity(
        make_capacity_config(max_num_seqs=2), make_kv_cache_specs()
    )

    assert multiple.native_working_set_tokens == 20800
    assert multiple.native_num_blocks > single.native_num_blocks
    assert multiple.auxiliary_memory_bytes > single.auxiliary_memory_bytes


def test_capacity_rejects_unaligned_segment_size():
    config = make_capacity_config(retrospec_index_segment_size=1000)

    with pytest.raises(ValueError, match="divisible by block_size"):
        build_retrospec_long_context_capacity(config, make_kv_cache_specs())


def test_capacity_rejects_unaligned_generation_update_interval():
    config = make_capacity_config(retrospec_index_update_interval=1000)

    with pytest.raises(ValueError, match="retrospec_index_update_interval"):
        build_retrospec_long_context_capacity(config, make_kv_cache_specs())


def test_kv_config_uses_native_working_set_when_full_kv_does_not_fit(monkeypatch):
    config = make_engine_config()
    spec = next(iter(make_kv_cache_specs().values()))
    capacity = RetroSpecLongContextCapacity(
        native_working_set_tokens=144,
        native_num_blocks=10,
        native_memory_bytes=10 * spec.page_size_bytes,
        auxiliary_memory_bytes=spec.page_size_bytes,
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.capacity.build_retrospec_long_context_capacity",
        lambda *_: capacity,
    )

    available_memory = capacity.total_memory_bytes + spec.page_size_bytes
    kv_cache_configs = get_kv_cache_configs(
        config,
        [{"layer.0": spec}],
        [available_memory],
    )

    assert kv_cache_configs[0].num_blocks == capacity.native_num_blocks
    assert kv_cache_configs[0].kv_cache_tensors[0].size == (
        capacity.native_memory_bytes
    )


def test_kv_config_uses_estimated_long_context_capacity():
    config = make_engine_config()
    specs = make_kv_cache_specs(num_layers=32)
    capacity = build_retrospec_long_context_capacity(config, specs)
    available_memory = (
        capacity.total_memory_bytes + next(iter(specs.values())).page_size_bytes
    )

    kv_cache_configs = get_kv_cache_configs(config, [specs], [available_memory])

    assert kv_cache_configs[0].num_blocks == capacity.native_num_blocks
    assert sum(tensor.size for tensor in kv_cache_configs[0].kv_cache_tensors) == (
        capacity.native_memory_bytes
    )


def test_kv_config_keeps_normal_capacity_when_full_kv_fits():
    config = make_engine_config(max_model_len=16, max_num_seqs=2)
    spec = next(iter(make_kv_cache_specs().values()))
    available_memory = 10 * spec.page_size_bytes

    kv_cache_configs = get_kv_cache_configs(
        config,
        [{"layer.0": spec}],
        [available_memory],
    )

    assert kv_cache_configs[0].num_blocks == 10
    assert kv_cache_configs[0].kv_cache_tensors[0].size == available_memory


def test_long_context_concurrency_uses_native_working_set():
    config = make_capacity_config()
    spec = next(iter(make_kv_cache_specs().values()))
    capacity = build_retrospec_long_context_capacity(config, {"layer.0": spec})
    kv_cache_config = KVCacheConfig(
        num_blocks=2 * capacity.native_num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    assert get_max_concurrency_for_kv_cache_config(config, kv_cache_config) == 2.0


def test_long_context_capacity_rejects_disabled_chunked_prefill():
    config = make_engine_config(
        max_num_seqs=2,
        enable_chunked_prefill=False,
    )
    spec = next(iter(make_kv_cache_specs().values()))

    with pytest.raises(ValueError, match="enable_chunked_prefill=True"):
        get_kv_cache_configs(config, [{"layer.0": spec}], [spec.page_size_bytes])


def test_long_context_capacity_accepts_multiple_resident_requests():
    # The auxiliary request-level and packed indices make 64K full KV cheaper
    # than long-context mode for this synthetic 32-layer model. Use a context
    # where full KV still exceeds the complete RetroSpec working set.
    config = make_engine_config(max_model_len=131072, max_num_seqs=2)
    specs = make_kv_cache_specs(num_layers=32)
    capacity = build_retrospec_long_context_capacity(config, specs)
    available_memory = (
        capacity.total_memory_bytes + next(iter(specs.values())).page_size_bytes
    )

    kv_cache_configs = get_kv_cache_configs(config, [specs], [available_memory])

    assert kv_cache_configs[0].num_blocks == capacity.native_num_blocks


def test_long_context_capacity_checks_auxiliary_reserve(monkeypatch):
    config = make_engine_config()
    spec = next(iter(make_kv_cache_specs().values()))
    capacity = RetroSpecLongContextCapacity(
        native_working_set_tokens=144,
        native_num_blocks=10,
        native_memory_bytes=10 * spec.page_size_bytes,
        auxiliary_memory_bytes=10 * spec.page_size_bytes,
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.capacity.build_retrospec_long_context_capacity",
        lambda *_: capacity,
    )

    with pytest.raises(ValueError, match="auxiliary RetroSpec buffers"):
        get_kv_cache_configs(
            config,
            [{"layer.0": spec}],
            [capacity.total_memory_bytes - 1],
        )
