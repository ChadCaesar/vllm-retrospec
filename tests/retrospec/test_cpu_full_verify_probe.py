# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from benchmarks.retrospec.benchmark_cpu_full_verify import (
    AppendOnlyLayer,
    copy_cache_to_cpu,
    grouped_decode_attention,
    make_cache,
)


def test_append_only_kv_preserves_storage_and_bounds():
    layer = AppendOnlyLayer(5)
    keys = torch.arange(24).reshape(1, 2, 3, 4).float()
    values = keys + 100
    layer.update(keys, values)
    pointer = layer.keys.data_ptr()
    new_keys = torch.full((1, 2, 2, 4), 42.0)
    actual_k, actual_v = layer.update(new_keys, new_keys + 100)
    assert layer.keys.data_ptr() == pointer
    assert layer.get_seq_length() == 5
    torch.testing.assert_close(actual_k, torch.cat([keys, new_keys], dim=2))
    torch.testing.assert_close(actual_v, actual_k + 100)
    with pytest.raises(ValueError, match="KV capacity exceeded"):
        layer.update(new_keys, new_keys)
    assert layer.get_seq_length() == 5
    layer.crop(-3)
    assert layer.get_seq_length() == 2
    actual_k, _ = layer.update(new_keys, new_keys)
    torch.testing.assert_close(actual_k[..., :2, :], keys[..., :2, :])
    layer.reset()
    assert layer.get_seq_length() == 0
    assert layer.keys.data_ptr() == pointer


@pytest.mark.parametrize("config_class", [LlamaConfig, Qwen2Config])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grouped", [False, True])
def test_complete_model_cache_and_handoff(config_class, dtype, grouped):
    previous_threads = torch.get_num_threads()
    original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
    torch.set_num_threads(2)
    try:
        if grouped:
            ALL_ATTENTION_FUNCTIONS.register("sdpa", grouped_decode_attention)
        torch.manual_seed(0)
        config = config_class(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        config._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(config).to(dtype).eval()
        ids = torch.randint(0, 128, (1, 12))
        cache = make_cache(config, 16)
        with torch.inference_mode():
            for start, end in [(0, 4), (4, 7), (7, 12)]:
                actual = model(
                    ids[:, start:end],
                    past_key_values=cache,
                    cache_position=torch.arange(start, end),
                    use_cache=True,
                ).logits
                reference = model(ids[:, :end], use_cache=False).logits[:, start:]
                torch.testing.assert_close(
                    actual,
                    reference,
                    rtol=0.01 if dtype == torch.bfloat16 else 1e-4,
                    atol=0.005 if dtype == torch.bfloat16 else 1e-5,
                )
            copied = copy_cache_to_cpu(cache, config, 16)
            for source, target in zip(cache.layers, copied.layers, strict=True):
                assert target.get_seq_length() == 12
                assert source.keys.data_ptr() != target.keys.data_ptr()
                assert torch.equal(source.keys[..., :12, :], target.keys[..., :12, :])
            next_id = actual[:, -1].argmax(-1).reshape(1, 1)
            original = model(next_id, past_key_values=cache).logits
            resumed = model(next_id, past_key_values=copied).logits
            torch.testing.assert_close(original, resumed, rtol=0, atol=0)
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original_attention)
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("groups", [4, 7])
def test_grouped_decode_matches_expanded_head_attention(dtype, groups):
    torch.manual_seed(0)
    query = torch.randn(2, 2 * groups, 1, 16, dtype=dtype)
    # Valid KV is a view into a larger capacity, as in the real probe.
    key = torch.randn(2, 2, 64, 16, dtype=dtype)[..., :23, :]
    value = torch.randn(2, 2, 64, 16, dtype=dtype)[..., :23, :]
    actual, _ = grouped_decode_attention(None, query, key, value, None, scaling=0.25)
    reference = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(groups, dim=1),
        value.repeat_interleave(groups, dim=1),
        scale=0.25,
    ).transpose(1, 2)
    torch.testing.assert_close(
        actual,
        reference,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        atol=0.002 if dtype == torch.bfloat16 else 1e-6,
    )
