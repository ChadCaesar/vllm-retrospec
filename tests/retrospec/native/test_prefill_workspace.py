# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.spec_decode.retrospec.prefill import (
    RetroSpecLayerPrefillWorkspace,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


class PermutedKVBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (1, 0, 2, 3, 4)


def make_spec(*, page_size_padded: int | None = None) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        head_size_v=8,
        dtype=torch.float16,
        page_size_padded=page_size_padded,
    )


def make_workspace() -> RetroSpecLayerPrefillWorkspace:
    return RetroSpecLayerPrefillWorkspace(
        device=torch.device("cuda"),
        backend=PermutedKVBackend,
        kv_cache_spec=make_spec(),
        cache_dtype="auto",
    )


def test_workspace_prepares_backend_layout_and_tile_descriptors():
    workspace = make_workspace()
    workspace.prepare(33)

    assert workspace.num_tokens == 33
    assert workspace.capacity_tokens == 48
    assert workspace.kv_cache.shape == (2, 3, 16, 2, 8)
    assert workspace.kv_cache.permute(1, 0, 2, 3, 4).is_contiguous()

    workspace.begin_layer("model.layers.0.self_attn.attn")
    tile = workspace.tile(16, 33)

    assert tile.kv_cache.data_ptr() == workspace.kv_cache.data_ptr()
    assert tile.block_table.dtype == torch.int32
    assert tile.slot_mapping.dtype == torch.int64
    assert tile.block_table.tolist() == [[0, 1, 2]]
    assert tile.slot_mapping.tolist() == list(range(16, 33))
    assert tile.num_scheduled_tokens == 17
    assert tile.context_len == 33

    workspace.end_layer()
    workspace.close()


def test_workspace_reuses_capacity_and_grows_only_when_required():
    workspace = make_workspace()
    workspace.prepare(33)
    original_ptr = workspace.kv_cache.data_ptr()

    workspace.prepare(31)
    assert workspace.num_tokens == 31
    assert workspace.capacity_tokens == 48
    assert workspace.kv_cache.data_ptr() == original_ptr

    workspace.prepare(65)
    assert workspace.num_tokens == 65
    assert workspace.capacity_tokens == 80
    assert workspace.kv_cache.shape[1] == 5

    workspace.close()


def test_workspace_temporarily_binds_attention_cache_and_restores_it():
    workspace = make_workspace()
    workspace.prepare(16)
    workspace.begin_layer("model.layers.0.self_attn.attn")

    original_cache = torch.empty(1, device="cuda")
    attention_layer = SimpleNamespace(kv_cache=[original_cache])

    with workspace.bind_layer(
        "model.layers.0.self_attn.attn",
        attention_layer,
    ):
        assert attention_layer.kv_cache == [workspace.kv_cache]

    assert attention_layer.kv_cache == [original_cache]
    workspace.end_layer()
    workspace.close()


def test_workspace_waits_for_explicit_reuse_event():
    workspace = make_workspace()
    workspace.prepare(16)
    workspace.begin_layer("model.layers.0.self_attn.attn")

    producer_stream = torch.cuda.Stream()
    producer_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer_stream):
        workspace.kv_cache.fill_(3)
        reuse_ready_event = torch.cuda.Event()
        reuse_ready_event.record(producer_stream)

    workspace.end_layer(reuse_ready_event)
    workspace.begin_layer("model.layers.1.self_attn.attn")
    torch.cuda.current_stream().synchronize()

    torch.testing.assert_close(
        workspace.kv_cache,
        torch.full_like(workspace.kv_cache, 3),
    )
    workspace.end_layer()
    workspace.close()


def test_workspace_rejects_invalid_lifecycle_operations():
    workspace = make_workspace()

    with pytest.raises(RuntimeError, match="must be prepared"):
        workspace.begin_layer("layer.0")
    with pytest.raises(ValueError, match="must be positive"):
        workspace.prepare(0)

    workspace.prepare(16)
    workspace.begin_layer("layer.0")

    with pytest.raises(RuntimeError, match="active layer-prefill workspace"):
        workspace.prepare(32)
    with pytest.raises(RuntimeError, match="already owned"):
        workspace.begin_layer("layer.1")
    with pytest.raises(ValueError, match="exceeds"):
        workspace.tile(0, 17)
    with pytest.raises(RuntimeError, match="Cannot close"):
        workspace.close()

    workspace.abort_layer()
    workspace.close()


def test_workspace_rejects_padded_kv_pages():
    spec = make_spec()
    padded_spec = make_spec(page_size_padded=spec.real_page_size_bytes + 16)

    with pytest.raises(NotImplementedError, match="padded KV pages"):
        RetroSpecLayerPrefillWorkspace(
            device=torch.device("cuda"),
            backend=PermutedKVBackend,
            kv_cache_spec=padded_spec,
            cache_dtype="auto",
        )
