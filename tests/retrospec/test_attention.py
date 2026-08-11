# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.spec_decode.retrospec.attention import (
    RetroSpecAttentionMode,
    RetroSpecSparseAttention,
)
from vllm.v1.spec_decode.retrospec.index import (
    RetroSpecAttentionLevel,
    RetroSpecAttentionSelection,
    RetroSpecSelectionPlan,
)


def make_controller(
    index_mode: str = "block_mean",
) -> RetroSpecSparseAttention:
    config = cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="retrospec",
                num_speculative_tokens=2,
                retrospec_retrieval_ratio=0.25,
                retrospec_estimation_ratio=0.5,
                retrospec_index_mode=index_mode,
                retrospec_index_segment_size=4,
                retrospec_blocks_per_cluster=1,
                retrospec_kmeans_iterations=2,
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


def make_plan(batch_size: int, width: int = 0) -> RetroSpecSelectionPlan:
    indices = torch.zeros(batch_size, width, dtype=torch.int64)
    mask = torch.zeros(batch_size, width, dtype=torch.bool)
    return RetroSpecSelectionPlan(
        sparse_exact_indices=indices,
        sparse_exact_mask=mask,
        sparse_estimation_indices=indices,
        sparse_estimation_mask=mask,
        expanded_exact_indices=indices,
        expanded_exact_mask=mask,
        expanded_estimation_indices=indices,
        expanded_estimation_mask=mask,
        valid_token_counts=torch.zeros(batch_size, width, dtype=torch.int64),
        sparse_attn=torch.ones(batch_size),
        expanded_attn=torch.ones(batch_size),
    )


def make_selection(batch_size: int = 2) -> RetroSpecAttentionSelection:
    return RetroSpecAttentionSelection(
        exact_block_table=torch.empty(batch_size, 0, dtype=torch.int32),
        exact_seq_lens=torch.zeros(batch_size, dtype=torch.int32),
        estimation_keys=torch.empty(batch_size, 0, 1, 1),
        estimation_values=torch.empty(batch_size, 0, 1, 1),
        estimation_token_counts=torch.empty(batch_size, 0, dtype=torch.int32),
        attention_mass=torch.ones(batch_size),
        plan=make_plan(batch_size),
    )


def test_proposal_context_and_step_average_attention_mass():
    controller = make_controller()
    mark_installed(controller)

    with controller.proposal_context(["request"]):
        assert controller.in_proposal
        controller.begin_step(
            RetroSpecAttentionMode.DRAFT, 0, torch.tensor([True, False])
        )
        controller.attention_mass_sum[:2].copy_(torch.tensor([1.4, 2.0]))
        controller.attention_mass_layer_count = 2

        attention_mass = controller.end_step()

        assert attention_mass.tolist() == pytest.approx([0.7, 1.0])
        assert controller.mode == RetroSpecAttentionMode.PASSTHROUGH
        assert not controller.step_active

    assert not controller.in_proposal
    assert controller.active_mask is None


def test_proposal_context_restores_state_and_plans_after_exception():
    controller = make_controller()
    mark_installed(controller)

    with (
        pytest.raises(RuntimeError, match="model failure"),
        controller.proposal_context(["request"]),
    ):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, torch.tensor([True]))
        controller.selection_plans[0]["layer"] = make_plan(1)
        raise RuntimeError("model failure")

    assert not controller.in_proposal
    assert controller.mode == RetroSpecAttentionMode.PASSTHROUGH
    assert not controller.step_active
    assert controller.active_mask is None
    assert controller.batch_size == 0
    assert controller.selection_plans == [{}, {}]


def test_proposal_context_cannot_nest_and_step_requires_context():
    controller = make_controller()
    mark_installed(controller)

    with pytest.raises(RuntimeError, match="inside proposal_context"):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, torch.tensor([True]))

    with (
        controller.proposal_context(["request"]),
        pytest.raises(RuntimeError, match="cannot be nested"),
        controller.proposal_context(["request"]),
    ):
        pass


def test_forward_uses_original_attention_outside_proposal():
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
        "layer",
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
        layer, query, key, value, kv_cache, metadata, output, None, None
    )


def test_passthrough_forward_builds_segmented_index_after_target_attention():
    controller = make_controller("segmented_cluster")
    mark_installed(controller)
    events: list[str] = []

    def original_forward(*_args):
        events.append("forward")
        return torch.tensor([3.0])

    original_build = controller.index.build_or_update

    def build_or_update(**kwargs):
        events.append("index")
        return original_build(**kwargs)

    controller.index.build_or_update = Mock(side_effect=build_or_update)
    kv_cache = torch.empty(2, 5, 2, 1, 1)
    for block_id in range(5):
        kv_cache[:, block_id].fill_(float(block_id))
    metadata = SimpleNamespace(
        block_table=torch.arange(5, dtype=torch.int32).view(1, -1)
    )

    with controller.prefill_index_context(["request"], [10], [0]):
        result = controller.forward(
            "layer",
            original_forward,
            Mock(),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            kv_cache,
            metadata,
        )

    assert result.tolist() == [3.0]
    assert events == ["forward", "index"]
    assert not controller.needs_index_update("request", 10)
    assert not controller.prefill_index_active
    assert controller.prefill_request_ids == ()
    assert controller.prefill_seq_lens == ()
    assert controller.prefill_build_rows == ()


def test_prefill_index_context_restores_state_after_exception():
    controller = make_controller("segmented_cluster")

    with (
        pytest.raises(RuntimeError, match="prefill failure"),
        controller.prefill_index_context(["request"], [10], [0]),
    ):
        raise RuntimeError("prefill failure")

    assert not controller.prefill_index_active
    assert controller.prefill_request_ids == ()
    assert controller.prefill_seq_lens == ()
    assert controller.prefill_build_rows == ()


def test_estimation_attention_weights_centroids_by_token_count():
    controller = make_controller()
    impl = cast(FlashAttentionImpl, SimpleNamespace(scale=1.0))
    selection = RetroSpecAttentionSelection(
        exact_block_table=torch.empty(2, 0, dtype=torch.int32),
        exact_seq_lens=torch.zeros(2, dtype=torch.int32),
        estimation_keys=torch.zeros(2, 2, 1, 1, dtype=torch.bfloat16),
        estimation_values=torch.tensor(
            [
                [[[[2.0]]], [[[4.0]]]],
                [[[[8.0]]], [[[9.0]]]],
            ]
        )
        .view(2, 2, 1, 1)
        .to(torch.bfloat16),
        estimation_token_counts=torch.tensor([[1, 3], [0, 0]], dtype=torch.int32),
        attention_mass=torch.ones(2),
        plan=make_plan(2),
    )

    output, lse = controller._run_estimation_attention(
        impl, torch.zeros(2, 1, 1, dtype=torch.bfloat16), selection
    )

    assert output[0, 0, 0].item() == pytest.approx(3.5)
    assert output[1, 0, 0].item() == pytest.approx(0.0)
    assert lse[0, 0].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
    assert torch.isneginf(lse[0, 1])


def test_verification_reuses_draft_selection_plan_without_reranking():
    controller = make_controller()
    mark_installed(controller)
    selection = make_selection(batch_size=1)
    controller.index.select = Mock(return_value=selection)
    controller.index.materialize = Mock(return_value=selection)
    controller._run_exact_attention = Mock(
        return_value=(torch.zeros(1, 1, 1), torch.zeros(1, 1))
    )
    controller._run_estimation_attention = Mock(
        return_value=(torch.zeros(1, 1, 1), torch.zeros(1, 1))
    )

    impl = cast(FlashAttentionImpl, SimpleNamespace(scale=1.0))
    layer = cast(torch.nn.Module, SimpleNamespace())
    query = torch.zeros(1, 1, 1)
    kv_cache = torch.zeros(2, 1, 2, 1, 1)
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
    )
    output = torch.zeros(1, 1, 1)

    with (
        patch("vllm.v1.spec_decode.retrospec.attention.merge_attn_states"),
        controller.proposal_context(["request"]),
    ):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, torch.tensor([True]))
        controller._sparse_forward(
            "layer", impl, layer, query, kv_cache, metadata, output
        )
        controller.end_step()

        controller.begin_step(
            RetroSpecAttentionMode.EXPANDED_VERIFY,
            0,
            torch.tensor([True]),
        )
        controller._sparse_forward(
            "layer", impl, layer, query, kv_cache, metadata, output
        )
        controller.end_step()

        controller.index.select.assert_called_once()
        controller.index.materialize.assert_called_once()
        materialize_args = controller.index.materialize.call_args.args
        assert materialize_args[0] is selection.plan
        assert materialize_args[1] == RetroSpecAttentionLevel.EXPANDED
        assert torch.equal(materialize_args[2], kv_cache[0])
        assert torch.equal(materialize_args[3], kv_cache[1])
        assert materialize_args[4] is metadata.block_table


def test_end_step_requires_completed_attention_layer():
    controller = make_controller()
    mark_installed(controller)

    with controller.proposal_context(["request"]):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, torch.tensor([True]))
        with pytest.raises(RuntimeError, match="No attention layer ran"):
            controller.end_step()
