# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.spec_decode.retrospec.attention import RetroSpecSparseAttention
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionSelection


def make_controller() -> RetroSpecSparseAttention:
    config = cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="retrospec",
                num_speculative_tokens=2,
                retrospec_retrieval_ratio=0.25,
                retrospec_estimation_ratio=0.5,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            cache_config=SimpleNamespace(block_size=2),
        ),
    )
    return RetroSpecSparseAttention(config, torch.device("cpu"))


def mark_installed(controller: RetroSpecSparseAttention) -> None:
    controller.original_forwards["layer"] = cast(
        tuple[FlashAttentionImpl, Any],
        (object(), Mock()),
    )


def test_draft_context_and_step_average_hit_attention():
    controller = make_controller()
    mark_installed(controller)

    with controller.draft_context():
        assert controller.enabled
        controller.begin_step(torch.tensor([True, False]))
        controller.hit_attn_sum[:2].copy_(torch.tensor([1.4, 2.0]))
        controller.hit_attn_layer_count = 2

        hit_attn = controller.end_step()

        assert hit_attn.tolist() == pytest.approx([0.7, 1.0])
        assert not controller.step_active

    assert not controller.enabled
    assert controller.active_mask is None


def test_draft_context_restores_state_after_exception():
    controller = make_controller()
    mark_installed(controller)

    with (
        pytest.raises(RuntimeError, match="model failure"),
        controller.draft_context(),
    ):
        controller.begin_step(torch.tensor([True]))
        raise RuntimeError("model failure")

    assert not controller.enabled
    assert not controller.step_active
    assert controller.active_mask is None
    assert controller.batch_size == 0


def test_forward_uses_original_attention_outside_draft():
    controller = make_controller()
    original_forward = Mock(return_value=torch.tensor([3.0]))
    layer = Mock()
    query = torch.tensor([1.0])
    key = torch.tensor([2.0])
    value = torch.tensor([3.0])
    kv_cache = torch.tensor([4.0])
    metadata = cast(Any, object())
    output = torch.tensor([5.0])

    result = controller.forward(
        original_forward,
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )

    assert result.tolist() == [3.0]
    original_forward.assert_called_once_with(
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
        None,
        None,
    )


def test_estimation_attention_weights_centroids_by_token_count():
    controller = make_controller()
    impl = cast(
        FlashAttentionImpl,
        SimpleNamespace(scale=1.0),
    )
    selection = RetroSpecAttentionSelection(
        exact_block_table=torch.empty(2, 0, dtype=torch.int32),
        exact_seq_lens=torch.zeros(2, dtype=torch.int32),
        estimation_keys=torch.zeros(2, 2, 1, 1),
        estimation_values=torch.tensor(
            [
                [[[[2.0]]], [[[4.0]]]],
                [[[[8.0]]], [[[9.0]]]],
            ]
        ).view(2, 2, 1, 1),
        estimation_token_counts=torch.tensor(
            [[1, 3], [0, 0]],
            dtype=torch.int32,
        ),
        hit_attn=torch.ones(2),
    )

    output, lse = controller._run_estimation_attention(
        impl,
        torch.zeros(2, 1, 1),
        selection,
    )

    assert output[0, 0, 0].item() == pytest.approx(3.5)
    assert output[1, 0, 0].item() == pytest.approx(0.0)
    assert lse[0, 0].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
    assert torch.isneginf(lse[0, 1])


def test_end_step_requires_completed_attention_layer():
    controller = make_controller()
    mark_installed(controller)

    with controller.draft_context():
        controller.begin_step(torch.tensor([True]))
        with pytest.raises(RuntimeError, match="No attention layer ran"):
            controller.end_step()
