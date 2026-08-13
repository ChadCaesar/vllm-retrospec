# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.spec_decode.retrospec.attention import (
    RetroSpecAttentionMode,
    RetroSpecSparseAttention,
)
from vllm.v1.spec_decode.retrospec.execution import RetroSpecExactExecution
from vllm.v1.spec_decode.retrospec.index import (
    RetroSpecAttentionLevel,
    RetroSpecAttentionSelection,
    RetroSpecSelectionPlan,
)
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
    RetroSpecTokenAttentionSelection,
    RetroSpecTokenSelectionPlan,
)


def make_controller(
    index_mode: str = "block_mean",
    cache_mode: str = "gpu_reference",
    cache_ratio: float = 0.0,
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
                retrospec_cache_mode=cache_mode,
                retrospec_cache_ratio=cache_ratio,
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


@pytest.mark.parametrize(
    (
        "cache_mode",
        "cache_ratio",
        "pin_memory_available",
        "expected_pin_memory",
        "expected_cache_ratio",
    ),
    [
        ("gpu_reference", 0.0, True, False, 0.0),
        ("cpu_offload", 0.0, False, False, 0.75),
        ("cpu_offload", 0.4, True, True, 0.4),
    ],
)
def test_segmented_attention_configures_cluster_backing_store(
    cache_mode: str,
    cache_ratio: float,
    pin_memory_available: bool,
    expected_pin_memory: bool,
    expected_cache_ratio: float,
):
    with patch(
        "vllm.v1.spec_decode.retrospec.attention.is_pin_memory_available",
        return_value=pin_memory_available,
    ):
        controller = make_controller(
            index_mode="segmented_cluster",
            cache_mode=cache_mode,
            cache_ratio=cache_ratio,
        )

    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)
    assert controller.index.cluster_store.storage_mode == cache_mode
    assert controller.index.cluster_store.pin_memory is expected_pin_memory
    assert controller.index.cluster_store.cache_ratio == pytest.approx(
        expected_cache_ratio
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


def make_token_plan(
    batch_size: int,
    num_kv_heads: int,
    exact_width: int,
    estimation_width: int,
) -> RetroSpecTokenSelectionPlan:
    exact_indices = torch.zeros(
        batch_size,
        num_kv_heads,
        exact_width,
        dtype=torch.int64,
    )
    exact_mask = torch.zeros_like(exact_indices, dtype=torch.bool)
    estimation_keys = torch.zeros(
        batch_size,
        num_kv_heads,
        estimation_width,
        1,
    )
    estimation_counts = torch.zeros(
        batch_size,
        num_kv_heads,
        estimation_width,
        dtype=torch.int32,
    )
    page_ids = torch.empty(
        batch_size,
        num_kv_heads,
        0,
        0,
        dtype=torch.int64,
    )
    page_token_counts = torch.empty_like(page_ids, dtype=torch.int32)

    return RetroSpecTokenSelectionPlan(
        layer_name="layer",
        primary_exact_token_indices=exact_indices,
        primary_exact_token_mask=exact_mask,
        sparse_exact_page_ids=page_ids,
        sparse_exact_page_token_counts=page_token_counts,
        sparse_estimation_keys=estimation_keys,
        sparse_estimation_values=estimation_keys.clone(),
        sparse_estimation_token_counts=estimation_counts,
        expanded_exact_page_ids=page_ids,
        expanded_exact_page_token_counts=page_token_counts,
        expanded_estimation_keys=estimation_keys,
        expanded_estimation_values=estimation_keys.clone(),
        expanded_estimation_token_counts=estimation_counts,
        sparse_attn=torch.ones(batch_size),
        expanded_attn=torch.ones(batch_size),
    )


def make_exact_execution(
    keys: torch.Tensor,
    values: torch.Tensor,
    token_mask: torch.Tensor,
) -> RetroSpecExactExecution:
    batch_size, num_kv_heads, max_exact_seq_len, head_size = keys.shape
    exact_seq_lens = token_mask.sum(dim=2, dtype=torch.int32).reshape(-1)
    cu_seqlens_k = torch.zeros(
        exact_seq_lens.numel() + 1,
        dtype=torch.int32,
        device=keys.device,
    )
    torch.cumsum(exact_seq_lens, dim=0, out=cu_seqlens_k[1:])
    packed_keys = keys[token_mask].view(-1, 1, head_size).contiguous()
    packed_values = values[token_mask].view(-1, 1, head_size).contiguous()

    return RetroSpecExactExecution(
        keys=packed_keys,
        values=packed_values,
        exact_seq_lens=exact_seq_lens,
        cu_seqlens_q=torch.arange(
            exact_seq_lens.numel() + 1,
            dtype=torch.int32,
            device=keys.device,
        ),
        cu_seqlens_k=cu_seqlens_k,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        max_exact_seq_len=max_exact_seq_len,
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


def test_grouped_reference_attention_keeps_kv_heads_independent():
    controller = make_controller("segmented_cluster")
    impl = cast(FlashAttentionImpl, SimpleNamespace(scale=1.0))
    query = torch.zeros(1, 4, 1, dtype=torch.bfloat16)
    keys = torch.zeros(1, 2, 2, 1, dtype=torch.bfloat16)
    values = torch.tensor(
        [[[[2.0], [4.0]], [[10.0], [20.0]]]],
        dtype=torch.bfloat16,
    )
    token_counts = torch.ones(1, 2, 2, dtype=torch.int32)

    output, lse = controller._run_grouped_reference_attention(
        impl,
        query,
        keys,
        values,
        token_counts,
    )

    assert output[0, :, 0].tolist() == pytest.approx([3.0, 3.0, 15.0, 15.0])
    assert lse[:, 0].tolist() == pytest.approx(
        [torch.log(torch.tensor(2.0)).item()] * 4
    )


def test_grouped_flash_exact_attention_handles_an_empty_batch():
    impl = cast(
        FlashAttentionImpl,
        SimpleNamespace(scale=1.0, vllm_flash_attn_version=2),
    )
    query = torch.empty(0, 4, 8, dtype=torch.bfloat16)
    keys = torch.empty(0, 2, 3, 8, dtype=torch.bfloat16)
    values = torch.empty_like(keys)
    token_mask = torch.empty(0, 2, 3, dtype=torch.bool)

    execution = make_exact_execution(keys, values, token_mask)
    output, lse = RetroSpecSparseAttention._run_grouped_flash_exact_attention(
        impl, query, execution
    )

    assert output.shape == query.shape
    assert lse.shape == (4, 0)


def test_grouped_flash_exact_attention_handles_no_selected_tokens():
    impl = cast(
        FlashAttentionImpl,
        SimpleNamespace(scale=1.0, vllm_flash_attn_version=2),
    )
    query = torch.ones(2, 4, 8, dtype=torch.bfloat16)
    keys = torch.empty(2, 2, 0, 8, dtype=torch.bfloat16)
    values = torch.ones_like(keys)
    token_mask = torch.empty(2, 2, 0, dtype=torch.bool)

    execution = make_exact_execution(keys, values, token_mask)
    output, lse = RetroSpecSparseAttention._run_grouped_flash_exact_attention(
        impl, query, execution
    )

    assert not output.any()
    assert torch.isneginf(lse).all()


def test_token_exact_attention_uses_reference_fallback_on_cpu():
    controller = make_controller("segmented_cluster")
    impl = cast(
        FlashAttentionImpl,
        SimpleNamespace(scale=1.0, vllm_flash_attn_version=2),
    )
    plan = make_token_plan(
        batch_size=1,
        num_kv_heads=2,
        exact_width=2,
        estimation_width=0,
    )
    selection = RetroSpecTokenAttentionSelection(
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.full((1, 2), 2, dtype=torch.int32),
        estimation_keys=torch.empty(1, 2, 0, 8, dtype=torch.bfloat16),
        estimation_values=torch.empty(1, 2, 0, 8, dtype=torch.bfloat16),
        estimation_token_counts=torch.empty(1, 2, 0, dtype=torch.int32),
        attention_mass=torch.ones(1),
        plan=plan,
    )
    query = torch.ones(1, 4, 8, dtype=torch.bfloat16)
    expected = (
        torch.ones_like(query),
        torch.zeros(4, 1, dtype=torch.float32),
    )
    reference_exact = (
        torch.ones(1, 2, 2, 8, dtype=torch.bfloat16),
        torch.ones(1, 2, 2, 8, dtype=torch.bfloat16),
        torch.ones(1, 2, 2, dtype=torch.bool),
    )

    with (
        patch.object(
            controller.index,
            "materialize_exact_reference",
            return_value=reference_exact,
        ) as materialize_reference,
        patch.object(
            controller,
            "_run_grouped_reference_attention",
            return_value=expected,
        ) as reference_attention,
        patch.object(
            controller,
            "_run_grouped_flash_exact_attention",
        ) as flash_attention,
    ):
        result = controller._run_exact_attention(
            impl,
            Mock(),
            query,
            torch.empty(0),
            torch.empty(0),
            cast(
                FlashAttentionMetadata,
                SimpleNamespace(block_table=torch.empty(1, 0)),
            ),
            selection,
        )

    assert result is expected
    materialize_reference.assert_called_once()
    reference_attention.assert_called_once()
    flash_attention.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_flash_exact_attention_matches_reference_on_cuda():
    device = torch.device("cuda")
    torch.manual_seed(0)

    batch_size = 2
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 64
    max_num_vectors = 5
    scale = head_size**-0.5

    impl = cast(
        FlashAttentionImpl,
        SimpleNamespace(scale=scale, vllm_flash_attn_version=2),
    )
    query = torch.randn(
        batch_size,
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    keys = torch.randn(
        batch_size,
        num_kv_heads,
        max_num_vectors,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    values = torch.randn_like(keys)
    token_mask = torch.tensor(
        [
            [[True, True, True, False, False], [False] * 5],
            [[True, True, True, True, False], [True, False, False, False, False]],
        ],
        dtype=torch.bool,
        device=device,
    )

    execution = make_exact_execution(keys, values, token_mask)
    output, lse = RetroSpecSparseAttention._run_grouped_flash_exact_attention(
        impl, query, execution
    )
    reference_output, reference_lse = (
        RetroSpecSparseAttention._run_grouped_reference_attention(
            impl,
            query,
            keys,
            values,
            token_mask.to(torch.int32),
        )
    )

    torch.testing.assert_close(output, reference_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, reference_lse, atol=1e-4, rtol=1e-4)
    assert not output[0, 2:].any()
    assert torch.isneginf(lse[2:, 0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_flash_exact_attention_handles_zero_lengths_on_cuda():
    device = torch.device("cuda")
    batch_size = 2
    num_kv_heads = 2
    num_query_heads = 4
    head_size = 64
    max_exact_seq_len = 3
    num_sequences = batch_size * num_kv_heads
    impl = cast(
        FlashAttentionImpl,
        SimpleNamespace(scale=1.0, vllm_flash_attn_version=2),
    )
    execution = RetroSpecExactExecution(
        keys=torch.empty(
            num_sequences * max_exact_seq_len,
            1,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        values=torch.empty(
            num_sequences * max_exact_seq_len,
            1,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        exact_seq_lens=torch.zeros(num_sequences, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.arange(num_sequences + 1, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.zeros(num_sequences + 1, dtype=torch.int32, device=device),
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        max_exact_seq_len=max_exact_seq_len,
    )

    output, lse = RetroSpecSparseAttention._run_grouped_flash_exact_attention(
        impl,
        torch.ones(
            batch_size,
            num_query_heads,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        execution,
    )

    assert not output.any()
    assert torch.isneginf(lse).all()


def test_token_estimation_attention_uses_per_head_cluster_sizes():
    controller = make_controller("segmented_cluster")
    impl = cast(FlashAttentionImpl, SimpleNamespace(scale=1.0))
    plan = make_token_plan(
        batch_size=1,
        num_kv_heads=2,
        exact_width=0,
        estimation_width=2,
    )
    selection = RetroSpecTokenAttentionSelection(
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.zeros(1, 2, dtype=torch.int32),
        estimation_keys=torch.zeros(1, 2, 2, 1, dtype=torch.bfloat16),
        estimation_values=torch.tensor(
            [[[[2.0], [4.0]], [[10.0], [20.0]]]],
            dtype=torch.bfloat16,
        ),
        estimation_token_counts=torch.tensor(
            [[[1, 3], [3, 1]]],
            dtype=torch.int32,
        ),
        attention_mass=torch.ones(1),
        plan=plan,
    )

    output, lse = controller._run_estimation_attention(
        impl,
        torch.zeros(1, 4, 1, dtype=torch.bfloat16),
        selection,
    )

    assert output[0, :, 0].tolist() == pytest.approx([3.5, 3.5, 12.5, 12.5])
    assert lse[:, 0].tolist() == pytest.approx(
        [torch.log(torch.tensor(4.0)).item()] * 4
    )


def test_get_grouped_estimation_converts_block_layout():
    selection = RetroSpecAttentionSelection(
        exact_block_table=torch.empty(1, 0, dtype=torch.int32),
        exact_seq_lens=torch.zeros(1, dtype=torch.int32),
        estimation_keys=torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2),
        estimation_values=torch.arange(12, 24, dtype=torch.float32).view(1, 3, 2, 2),
        estimation_token_counts=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        attention_mass=torch.ones(1),
        plan=make_plan(1),
    )

    keys, values, counts = RetroSpecSparseAttention._get_grouped_estimation(selection)

    assert keys.shape == (1, 2, 3, 2)
    assert keys.dtype == torch.float32
    assert values.shape == keys.shape
    assert counts.tolist() == [[[1, 2, 3], [1, 2, 3]]]
    assert keys[0, 0, :, 0].tolist() == [0, 4, 8]
    assert keys[0, 1, :, 0].tolist() == [2, 6, 10]


def test_get_grouped_estimation_keeps_token_layout():
    plan = make_token_plan(1, 2, exact_width=0, estimation_width=3)
    selection = RetroSpecTokenAttentionSelection(
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.zeros(1, 2, dtype=torch.int32),
        estimation_keys=torch.randn(1, 2, 3, 4),
        estimation_values=torch.randn(1, 2, 3, 4),
        estimation_token_counts=torch.ones(1, 2, 3, dtype=torch.int32),
        attention_mass=torch.ones(1),
        plan=plan,
    )

    keys, values, counts = RetroSpecSparseAttention._get_grouped_estimation(selection)

    assert keys is selection.estimation_keys
    assert values is selection.estimation_values
    assert counts is selection.estimation_token_counts


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
