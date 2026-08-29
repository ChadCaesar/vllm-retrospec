# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
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
from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecResolvedClusterPages,
)
from vllm.v1.spec_decode.retrospec.execution import (
    RetroSpecExactKVSource,
)
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecFullVerificationPlan,
    RetroSpecSegmentedTokenIndex,
    RetroSpecTokenAttentionSelection,
    RetroSpecTokenSelectionPlan,
)


def make_controller(
    cache_ratio: float = 0.0,
    max_pending_cluster_builds: int = 2,
    cpu_page_slab_size_mib: int = 1,
    max_pinned_memory: float = 0.0625,
    max_gpu_index_memory: float = 0.125,
    stats_interval_seconds: float = 0.0,
) -> RetroSpecSparseAttention:
    config = cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="retrospec",
                num_speculative_tokens=2,
                retrospec_retrieval_ratio=0.25,
                retrospec_estimation_ratio=0.5,
                retrospec_index_segment_size=4,
                retrospec_index_update_interval=2,
                retrospec_blocks_per_cluster=1,
                retrospec_kmeans_iterations=2,
                retrospec_max_pending_cluster_builds=max_pending_cluster_builds,
                retrospec_cpu_page_slab_size_mib=cpu_page_slab_size_mib,
                retrospec_max_pinned_memory=max_pinned_memory,
                retrospec_max_gpu_index_memory=max_gpu_index_memory,
                retrospec_cache_ratio=cache_ratio,
                retrospec_stats_interval_seconds=stats_interval_seconds,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=4,
                max_num_batched_tokens=32,
            ),
            cache_config=SimpleNamespace(block_size=2),
            model_config=SimpleNamespace(max_model_len=64),
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
        "cache_ratio",
        "pin_memory_available",
        "expected_pin_memory",
        "expected_cache_ratio",
    ),
    [
        (0.0, False, False, 0.75),
        (0.4, True, False, 0.4),
    ],
)
def test_segmented_attention_configures_cluster_backing_store(
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
            cache_ratio=cache_ratio,
        )

    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)
    assert controller.index.cluster_store.pin_memory is expected_pin_memory
    assert controller.index.cluster_store.cache_ratio == pytest.approx(
        expected_cache_ratio
    )
    assert controller.index.max_pending_cluster_builds == 2
    assert controller.index.cluster_store.cpu_page_slab_bytes == 1 << 20
    assert controller.index.cluster_store.max_pinned_memory_bytes == 64 << 20
    assert controller.index._gpu_index_residency.max_gpu_index_memory_bytes == 128 << 20


def test_segmented_attention_configures_pending_cluster_build_limit():
    controller = make_controller(
        max_pending_cluster_builds=4,
    )

    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)
    assert controller.index.max_pending_cluster_builds == 4


def test_exact_attention_workspace_covers_mixed_verification_batch():
    controller = make_controller()

    assert controller.max_parallel_tokens == 8
    assert controller.max_verification_tokens == 32
    assert controller.exact_attention_workspace.max_num_queries == 32


def test_segmented_attention_shares_enabled_performance_stats():
    controller = make_controller(stats_interval_seconds=5.0)

    assert controller.performance_stats.enabled
    assert controller.index.performance_stats is controller.performance_stats
    assert (
        controller.index.cluster_store.performance_stats is controller.performance_stats
    )


def test_segmented_attention_does_not_instrument_deep_paths_by_default():
    controller = make_controller()

    assert not controller.performance_stats.enabled
    assert controller.index.performance_stats is None
    assert controller.index.cluster_store.performance_stats is None


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
    cluster_ids = torch.empty(
        batch_size,
        num_kv_heads,
        0,
        dtype=torch.int64,
    )

    return RetroSpecTokenSelectionPlan(
        layer_name="layer",
        primary_exact_token_indices=exact_indices,
        primary_exact_token_mask=exact_mask,
        sparse_exact_cluster_ids=cluster_ids,
        sparse_exact_page_ids=page_ids,
        sparse_exact_page_token_counts=page_token_counts,
        sparse_estimation_keys=estimation_keys,
        sparse_estimation_values=estimation_keys.clone(),
        sparse_estimation_token_counts=estimation_counts,
        expanded_exact_cluster_ids=cluster_ids,
        expanded_exact_page_ids=page_ids,
        expanded_exact_page_token_counts=page_token_counts,
        expanded_estimation_keys=estimation_keys,
        expanded_estimation_values=estimation_keys.clone(),
        expanded_estimation_token_counts=estimation_counts,
        sparse_attn=torch.ones(batch_size),
        expanded_attn=torch.ones(batch_size),
    )


def make_plan(batch_size: int, width: int = 0) -> RetroSpecTokenSelectionPlan:
    return make_token_plan(
        batch_size,
        num_kv_heads=1,
        exact_width=width,
        estimation_width=0,
    )


def make_selection(batch_size: int = 2) -> RetroSpecTokenAttentionSelection:
    plan = make_plan(batch_size)
    return RetroSpecTokenAttentionSelection(
        exact_cluster_ids=plan.sparse_exact_cluster_ids,
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.zeros(batch_size, 1, dtype=torch.int32),
        estimation_keys=plan.sparse_estimation_keys,
        estimation_values=plan.sparse_estimation_values,
        estimation_token_counts=plan.sparse_estimation_token_counts,
        attention_mass=torch.ones(batch_size),
        plan=plan,
        resolved_pages=None,
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
    controller = make_controller()
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

    with controller.index_update_context(["request"], [10], [True], [True], [0]):
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
    assert controller.index.build_or_update.call_args.kwargs["defer_cpu_store"] is True
    assert controller.index.build_or_update.call_args.kwargs["is_prefill"] == (True,)
    assert not controller.needs_index_update("request", 10, True, True)
    assert not controller.index_update_active
    assert controller.index_update_request_ids == ()
    assert controller.index_update_seq_lens == ()
    assert controller.index_update_is_prefill == ()
    assert controller.index_update_prefill_complete == ()
    assert controller.index_update_build_rows == ()


def test_prefill_completion_marks_each_request_for_first_draft_warmup():
    controller = make_controller()
    mark_installed(controller)
    controller.index.build_or_update = Mock()
    controller.index.flush_staged_updates = Mock()
    controller.index.mark_first_draft_warmup = Mock()

    layer = SimpleNamespace()
    query = torch.arange(5, dtype=torch.float32).view(5, 1, 1)
    kv_cache = torch.ones(2, 4, 2, 1, 1)
    metadata = SimpleNamespace(
        block_table=torch.arange(4, dtype=torch.int32).repeat(2, 1),
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    with controller.index_update_context(
        ["first", "second"],
        [6, 4],
        [True, True],
        [True, True],
        [0, 1],
    ):
        controller.forward(
            "layer",
            Mock(return_value=torch.tensor([3.0])),
            layer,
            query,
            torch.ones_like(query),
            torch.ones_like(query),
            kv_cache,
            metadata,
        )

    controller.index.mark_first_draft_warmup.assert_called_once_with(
        ("first", "second"), ("layer",)
    )
    assert controller.index_update_request_ids == ()
    assert controller.index_update_build_rows == ()


def test_index_update_context_restores_state_after_exception():
    controller = make_controller()
    discard_staged_updates = Mock(
        wraps=controller.index.discard_staged_updates,
    )
    controller.index.discard_staged_updates = discard_staged_updates

    with (
        pytest.raises(RuntimeError, match="prefill failure"),
        controller.index_update_context(["request"], [10], [True], [True], [0]),
    ):
        raise RuntimeError("prefill failure")

    discard_staged_updates.assert_called_once_with()
    assert not controller.index_update_active
    assert controller.index_update_request_ids == ()
    assert controller.index_update_seq_lens == ()
    assert controller.index_update_is_prefill == ()
    assert controller.index_update_prefill_complete == ()
    assert controller.index_update_build_rows == ()


def test_index_update_context_restores_state_after_flush_failure():
    controller = make_controller()
    controller.index.flush_staged_updates = Mock(
        side_effect=RuntimeError("flush failure")
    )

    with (
        pytest.raises(RuntimeError, match="flush failure"),
        controller.index_update_context(["request"], [10], [True], [True], [0]),
    ):
        pass

    assert not controller.index_update_active
    assert controller.index_update_request_ids == ()
    assert controller.index_update_seq_lens == ()
    assert controller.index_update_is_prefill == ()
    assert controller.index_update_prefill_complete == ()
    assert controller.index_update_build_rows == ()


def test_generation_index_context_forwards_generation_phase():
    controller = make_controller()
    mark_installed(controller)
    controller.index.build_or_update = Mock()
    kv_cache = torch.ones(2, 8, 2, 1, 1)
    metadata = SimpleNamespace(
        block_table=torch.arange(8, dtype=torch.int32).view(1, -1)
    )

    with controller.index_update_context(["request"], [12], [False], [False], [0]):
        controller.forward(
            "layer",
            Mock(return_value=torch.tensor([3.0])),
            Mock(),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            kv_cache,
            metadata,
        )

    call_kwargs = controller.index.build_or_update.call_args.kwargs
    assert call_kwargs["seq_lens"] == (12,)
    assert call_kwargs["is_prefill"] == (False,)
    assert call_kwargs["prefill_complete"] == (False,)
    assert call_kwargs["rows"] == (0,)


def test_full_verification_context_prepares_and_restores_attention_state():
    controller = make_controller()
    mark_installed(controller)
    controller.index.prepare_full_verification = Mock(
        wraps=controller.index.prepare_full_verification
    )

    with controller.full_verification_context(
        request_ids=["request"],
        context_lens=[5],
        query_lens=[3],
    ):
        assert controller.mode == RetroSpecAttentionMode.FULL_VERIFY
        assert controller.full_verification_batch is not None
        assert controller.full_verification_batch.request_ids == ("request",)
        assert controller.full_verification_batch.context_lens == (5,)
        assert controller.full_verification_batch.query_lens == (3,)

    controller.index.prepare_full_verification.assert_called_once_with(
        ("request",),
        (5,),
        ("layer",),
    )
    assert controller.mode == RetroSpecAttentionMode.PASSTHROUGH
    assert controller.full_verification_batch is None


def test_full_verification_context_restores_state_after_exception():
    controller = make_controller()
    mark_installed(controller)

    with (
        pytest.raises(RuntimeError, match="verification failure"),
        controller.full_verification_context(
            request_ids=["request"],
            context_lens=[0],
            query_lens=[1],
        ),
    ):
        raise RuntimeError("verification failure")

    assert controller.mode == RetroSpecAttentionMode.PASSTHROUGH
    assert controller.full_verification_batch is None


def test_proposal_context_rejects_excess_residency_without_state_leak():
    controller = make_controller()
    mark_installed(controller)
    request_ids = [f"request-{index}" for index in range(5)]

    with (
        pytest.raises(RuntimeError, match="exceeds max_num_seqs"),
        controller.proposal_context(request_ids),
    ):
        pass

    assert not controller.in_proposal
    assert controller.proposal_request_ids == ()
    assert controller.index._gpu_index_residency.active_request_ids == ()


def test_attention_reports_only_new_fully_stored_retirement_ranges():
    controller = make_controller()
    mark_installed(controller)
    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)
    controller.index.get_fully_stored_indexed_end = Mock(return_value=6)

    assert controller.take_kv_cache_retirement_ranges(["request"]) == [
        ("request", 1, 3)
    ]
    assert controller.has_retired_kv_blocks(["request"])
    assert controller.take_kv_cache_retirement_ranges(["request"]) == []

    controller.index.get_fully_stored_indexed_end.return_value = 10
    assert controller.take_kv_cache_retirement_ranges(["request"]) == [
        ("request", 3, 5)
    ]

    controller.remove_requests(["request"])
    assert not controller.has_retired_kv_blocks(["request"])


def test_full_verification_rejects_rollback_behind_retired_boundary():
    controller = make_controller()
    mark_installed(controller)
    controller._retired_block_ends["request"] = 3

    with (
        pytest.raises(RuntimeError, match="behind retired KV boundary 6"),
        controller.full_verification_context(
            request_ids=["request"],
            context_lens=[5],
            query_lens=[1],
        ),
    ):
        pass


def test_forward_dispatches_full_verification_and_updates_index():
    controller = make_controller()
    mark_installed(controller)
    impl = object.__new__(FlashAttentionImpl)
    layer = SimpleNamespace(impl=impl)
    query = torch.ones(1, 1, 1)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    kv_cache = torch.ones(2, 1, 1, 1, 1)
    metadata = cast(
        FlashAttentionMetadata,
        SimpleNamespace(block_table=torch.zeros(1, 1, dtype=torch.int32)),
    )
    output = torch.empty_like(query)
    expected = torch.full_like(output, 3)
    original_forward = Mock()
    controller._full_verification_forward = Mock(return_value=expected)
    controller._maybe_update_index = Mock()

    with controller.full_verification_context(
        request_ids=["request"],
        context_lens=[0],
        query_lens=[1],
    ):
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

    assert result is expected
    original_forward.assert_not_called()
    controller._full_verification_forward.assert_called_once_with(
        "layer",
        impl,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )
    controller._maybe_update_index.assert_called_once_with("layer", kv_cache, metadata)


def test_estimation_attention_weights_centroids_by_token_count():
    controller = make_controller()
    impl = cast(FlashAttentionImpl, SimpleNamespace(scale=1.0))
    plan = make_token_plan(2, 1, exact_width=0, estimation_width=2)
    selection = RetroSpecTokenAttentionSelection(
        exact_cluster_ids=plan.sparse_exact_cluster_ids,
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.zeros(2, 1, dtype=torch.int32),
        estimation_keys=torch.zeros(2, 1, 2, 1, dtype=torch.bfloat16),
        estimation_values=torch.tensor(
            [
                [[[[2.0], [4.0]]]],
                [[[[8.0], [9.0]]]],
            ],
            dtype=torch.bfloat16,
        ).view(2, 1, 2, 1),
        estimation_token_counts=torch.tensor([[[1, 3]], [[0, 0]]], dtype=torch.int32),
        attention_mass=torch.ones(2),
        plan=plan,
        resolved_pages=None,
    )

    output, lse = controller._run_estimation_attention(
        impl, torch.zeros(2, 1, 1, dtype=torch.bfloat16), selection
    )

    assert output[0, 0, 0].item() == pytest.approx(3.5)
    assert output[1, 0, 0].item() == pytest.approx(0.0)
    assert lse[0, 0].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
    assert torch.isneginf(lse[0, 1])


def test_grouped_reference_attention_keeps_kv_heads_independent():
    controller = make_controller()
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


def test_token_exact_attention_uses_reference_fallback_on_cpu():
    controller = make_controller()
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
        exact_cluster_ids=plan.sparse_exact_cluster_ids,
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.full((1, 2), 2, dtype=torch.int32),
        estimation_keys=torch.empty(1, 2, 0, 8, dtype=torch.bfloat16),
        estimation_values=torch.empty(1, 2, 0, 8, dtype=torch.bfloat16),
        estimation_token_counts=torch.empty(1, 2, 0, dtype=torch.int32),
        attention_mass=torch.ones(1),
        plan=plan,
        resolved_pages=None,
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
            controller.exact_attention_workspace,
            "run",
        ) as exact_attention,
    ):
        result = controller._run_exact_attention(
            impl,
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
    exact_attention.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_reference_fallback_updates_resident_cache_after_materialization():
    controller = make_controller(
        cache_ratio=0.5,
    )
    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)
    controller.mode = RetroSpecAttentionMode.SPARSE_VERIFY

    device = torch.device("cuda")
    cluster_ids = torch.tensor([[[0]]], dtype=torch.int64, device=device)
    page_ids = torch.tensor([[[[0]]]], dtype=torch.int64, device=device)
    page_counts = torch.ones_like(page_ids, dtype=torch.int32)
    base_plan = make_token_plan(1, 1, exact_width=0, estimation_width=0)
    plan = replace(
        base_plan,
        sparse_exact_cluster_ids=cluster_ids,
        sparse_exact_page_ids=page_ids,
        sparse_exact_page_token_counts=page_counts,
    )
    selection = RetroSpecTokenAttentionSelection(
        exact_cluster_ids=cluster_ids,
        exact_page_ids=page_ids,
        exact_page_token_counts=page_counts,
        exact_token_counts=torch.ones(1, 1, dtype=torch.int32, device=device),
        estimation_keys=plan.sparse_estimation_keys,
        estimation_values=plan.sparse_estimation_values,
        estimation_token_counts=plan.sparse_estimation_token_counts,
        attention_mass=torch.ones(1, device=device),
        plan=plan,
        resolved_pages=None,
    )
    exact_keys = torch.ones(1, 1, 1, 1, device=device)
    reference_exact = (
        exact_keys,
        exact_keys.clone(),
        torch.ones(1, 1, 1, dtype=torch.bool, device=device),
    )
    expected = (
        torch.ones(1, 1, 1, device=device),
        torch.zeros(1, 1, device=device),
    )
    call_order: list[str] = []

    with (
        patch.object(
            controller.index,
            "materialize_exact_reference",
            side_effect=lambda *_: call_order.append("materialize") or reference_exact,
        ),
        patch.object(
            controller.index.cluster_store,
            "admit_resident_clusters",
            side_effect=lambda **_: call_order.append("admit"),
        ) as admit,
        patch.object(
            controller,
            "_run_grouped_reference_attention",
            side_effect=lambda *_: call_order.append("attention") or expected,
        ),
    ):
        result = controller._run_exact_attention(
            cast(FlashAttentionImpl, SimpleNamespace()),
            torch.ones(1, 1, 1, dtype=torch.float32, device=device),
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            cast(
                FlashAttentionMetadata,
                SimpleNamespace(
                    block_table=torch.empty(1, 0, dtype=torch.int32, device=device)
                ),
            ),
            selection,
        )

    assert result is expected
    admit.assert_called_once_with(
        layer_name="layer", cluster_ids=cluster_ids, page_ids=page_ids
    )
    assert call_order == ["materialize", "admit", "attention"]


def test_full_verification_source_skips_resolution_without_cluster_pages():
    controller = make_controller()
    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)

    plan = RetroSpecFullVerificationPlan(
        layer_name="layer",
        primary_exact_token_indices=torch.tensor([[[0, 1, 2]]]),
        primary_exact_token_mask=torch.ones(1, 1, 3, dtype=torch.bool),
        exact_page_ids=torch.empty(1, 1, 1, 0, dtype=torch.int64),
        exact_page_ids_cpu=torch.empty(1, 1, 1, 0, dtype=torch.int64),
        exact_page_token_counts=torch.empty(1, 1, 1, 0, dtype=torch.int32),
        exact_token_counts=torch.tensor([[3]], dtype=torch.int32),
    )
    controller.index.cluster_store.resolve_full_verification_blocks = Mock()
    key_cache = torch.zeros(2, 2, 1, 1)
    value_cache = key_cache.clone()
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    source, resolved_pages = controller._resolve_exact_kv_source(
        plan,
        key_cache,
        value_cache,
        block_table,
    )

    assert resolved_pages is None
    resolve_full = controller.index.cluster_store.resolve_full_verification_blocks
    resolve_full.assert_not_called()
    assert source.primary.key_cache is key_cache
    assert source.primary.value_cache is value_cache
    assert source.primary.token_indices is plan.primary_exact_token_indices
    assert source.primary.token_mask is plan.primary_exact_token_mask
    assert source.page_token_counts is plan.exact_page_token_counts
    assert source.resident_pages is None
    assert source.staging_pages is None


def test_full_verification_source_resolves_all_cluster_pages():
    controller = make_controller()
    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)

    page_ids = torch.tensor([[[[2], [3]]]], dtype=torch.int64)
    page_counts = torch.tensor([[[[2], [1]]]], dtype=torch.int32)
    plan = RetroSpecFullVerificationPlan(
        layer_name="layer",
        primary_exact_token_indices=torch.tensor([[[0, 4]]]),
        primary_exact_token_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        exact_page_ids=page_ids,
        exact_page_ids_cpu=page_ids.cpu(),
        exact_page_token_counts=page_counts,
        exact_token_counts=torch.tensor([[5]], dtype=torch.int32),
    )
    resident_page_ids = torch.tensor([[[[0], [-1]]]], dtype=torch.int64)
    staging_page_ids = torch.tensor([[[[-1], [0]]]], dtype=torch.int64)
    resident_keys = torch.zeros(1, 2, 1)
    resident_values = resident_keys.clone()
    staging_keys = torch.ones(1, 2, 1)
    staging_values = staging_keys.clone()
    staging_ready_event = Mock()
    resolved = RetroSpecResolvedClusterPages(
        resident_page_ids=resident_page_ids,
        staging_page_ids=staging_page_ids,
        resident_key_pages=resident_keys,
        resident_value_pages=resident_values,
        staging_key_pages=staging_keys,
        staging_value_pages=staging_values,
        hit_cluster_mask=torch.tensor([[[True, False]]]),
        miss_cluster_mask=torch.tensor([[[False, True]]]),
        hit_gate_ready_mask=torch.zeros(1, 1, 2, dtype=torch.bool),
        resident_ready_event=None,
        staging_ready_event=staging_ready_event,
    )
    controller.index.cluster_store.resolve_full_verification_blocks = Mock(
        return_value=resolved
    )
    key_cache = torch.zeros(3, 2, 1, 1)
    value_cache = key_cache.clone()
    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    source, resolved_pages = controller._resolve_exact_kv_source(
        plan,
        key_cache,
        value_cache,
        block_table,
    )

    assert resolved_pages is resolved
    resolve_full = controller.index.cluster_store.resolve_full_verification_blocks
    resolve_full.assert_called_once_with(
        layer_name="layer",
        logical_page_ids=page_ids,
        logical_page_ids_cpu=plan.exact_page_ids_cpu,
    )
    assert source.primary.token_indices is plan.primary_exact_token_indices
    assert source.primary.token_mask is plan.primary_exact_token_mask
    assert source.page_token_counts is page_counts
    assert source.resident_pages is not None
    assert source.staging_pages is not None
    assert source.resident_pages.page_ids is resident_page_ids
    assert source.staging_pages.page_ids is staging_page_ids
    assert source.staging_pages.ready_event is staging_ready_event

    controller.index.cluster_store.resolve_full_verification_blocks.reset_mock()
    prefetched_plan = replace(plan, resolved_pages=resolved)
    _, prefetched_pages = controller._resolve_exact_kv_source(
        prefetched_plan,
        key_cache,
        value_cache,
        block_table,
    )
    assert prefetched_pages is resolved
    resolve_full.assert_not_called()


def test_token_estimation_attention_uses_per_head_cluster_sizes():
    controller = make_controller()
    impl = cast(FlashAttentionImpl, SimpleNamespace(scale=1.0))
    plan = make_token_plan(
        batch_size=1,
        num_kv_heads=2,
        exact_width=0,
        estimation_width=2,
    )
    selection = RetroSpecTokenAttentionSelection(
        exact_cluster_ids=plan.sparse_exact_cluster_ids,
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
        resolved_pages=None,
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


def test_get_grouped_estimation_keeps_token_layout():
    plan = make_token_plan(1, 2, exact_width=0, estimation_width=3)
    selection = RetroSpecTokenAttentionSelection(
        exact_cluster_ids=plan.sparse_exact_cluster_ids,
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.zeros(1, 2, dtype=torch.int32),
        estimation_keys=torch.randn(1, 2, 3, 4),
        estimation_values=torch.randn(1, 2, 3, 4),
        estimation_token_counts=torch.ones(1, 2, 3, dtype=torch.int32),
        attention_mass=torch.ones(1),
        plan=plan,
        resolved_pages=None,
    )

    keys, values, counts = RetroSpecSparseAttention._get_grouped_estimation(selection)

    assert keys is selection.estimation_keys
    assert values is selection.estimation_values
    assert counts is selection.estimation_token_counts


@pytest.mark.parametrize(
    ("pre_resolved", "attention_mode", "expect_admission"),
    [
        (False, RetroSpecAttentionMode.SPARSE_VERIFY, True),
        (True, RetroSpecAttentionMode.SPARSE_VERIFY, True),
        (True, RetroSpecAttentionMode.DRAFT, False),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_attention_resolves_resident_and_staging_pages(
    pre_resolved: bool,
    attention_mode: RetroSpecAttentionMode,
    expect_admission: bool,
):
    controller = make_controller(
        cache_ratio=0.5,
    )
    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)
    controller.mode = attention_mode

    device = torch.device("cuda")
    cluster_ids = torch.tensor([[[0, 1]]], dtype=torch.int64, device=device)
    page_ids = torch.tensor([[[[0], [1]]]], dtype=torch.int64, device=device)
    page_counts = torch.tensor([[[[2], [1]]]], dtype=torch.int32, device=device)
    plan = RetroSpecTokenSelectionPlan(
        layer_name="layer",
        primary_exact_token_indices=torch.empty(
            1, 1, 0, dtype=torch.int64, device=device
        ),
        primary_exact_token_mask=torch.empty(1, 1, 0, dtype=torch.bool, device=device),
        sparse_exact_cluster_ids=cluster_ids,
        sparse_exact_page_ids=page_ids,
        sparse_exact_page_token_counts=page_counts,
        sparse_estimation_keys=torch.empty(1, 1, 0, 1, device=device),
        sparse_estimation_values=torch.empty(1, 1, 0, 1, device=device),
        sparse_estimation_token_counts=torch.empty(
            1, 1, 0, dtype=torch.int32, device=device
        ),
        expanded_exact_cluster_ids=cluster_ids,
        expanded_exact_page_ids=page_ids,
        expanded_exact_page_token_counts=page_counts,
        expanded_estimation_keys=torch.empty(1, 1, 0, 1, device=device),
        expanded_estimation_values=torch.empty(1, 1, 0, 1, device=device),
        expanded_estimation_token_counts=torch.empty(
            1, 1, 0, dtype=torch.int32, device=device
        ),
        sparse_attn=torch.ones(1, device=device),
        expanded_attn=torch.ones(1, device=device),
    )
    selection = RetroSpecTokenAttentionSelection(
        exact_cluster_ids=cluster_ids,
        exact_page_ids=page_ids,
        exact_page_token_counts=page_counts,
        exact_token_counts=torch.tensor([[3]], dtype=torch.int32, device=device),
        estimation_keys=plan.sparse_estimation_keys,
        estimation_values=plan.sparse_estimation_values,
        estimation_token_counts=plan.sparse_estimation_token_counts,
        attention_mass=torch.ones(1, device=device),
        plan=plan,
        resolved_pages=None,
    )

    resident_page_ids = torch.tensor([[[[0], [-1]]]], dtype=torch.int64, device=device)
    staging_page_ids = torch.tensor([[[[-1], [0]]]], dtype=torch.int64, device=device)
    resident_keys = torch.zeros(1, 2, 1, dtype=torch.float16, device=device)
    resident_values = resident_keys.clone()
    staging_keys = torch.ones(1, 2, 1, dtype=torch.float16, device=device)
    staging_values = staging_keys.clone()
    resident_ready_event = torch.cuda.Event()
    resolved = RetroSpecResolvedClusterPages(
        resident_page_ids=resident_page_ids,
        staging_page_ids=staging_page_ids,
        resident_key_pages=resident_keys,
        resident_value_pages=resident_values,
        staging_key_pages=staging_keys,
        staging_value_pages=staging_values,
        hit_cluster_mask=torch.tensor([[[True, False]]], device=device),
        miss_cluster_mask=torch.tensor([[[False, True]]], device=device),
        hit_gate_ready_mask=torch.ones(1, 1, 2, dtype=torch.bool, device=device),
        resident_ready_event=resident_ready_event,
    )
    if pre_resolved:
        selection = replace(selection, resolved_pages=resolved)

    controller.index.cluster_store.resolve_cluster_blocks = Mock(return_value=resolved)

    call_order: list[str] = []
    controller.index.cluster_store.admit_staged_clusters = Mock(
        side_effect=lambda **_: call_order.append("admit"),
    )
    expected_output = (
        torch.zeros(1, 1, 1, dtype=torch.float16, device=device),
        torch.zeros(1, 1, dtype=torch.float32, device=device),
    )
    run = Mock(
        side_effect=lambda *_args, **_kwargs: call_order.append("attention")
        or expected_output
    )
    controller.exact_attention_workspace = SimpleNamespace(run=run)

    key_cache = torch.zeros(1, 2, 1, 1, dtype=torch.float16, device=device)
    value_cache = key_cache.clone()
    result = controller._run_exact_attention(
        cast(FlashAttentionImpl, SimpleNamespace(scale=1.0)),
        torch.zeros(1, 1, 1, dtype=torch.float16, device=device),
        key_cache,
        value_cache,
        cast(
            FlashAttentionMetadata,
            SimpleNamespace(
                block_table=torch.zeros(1, 1, dtype=torch.int32, device=device)
            ),
        ),
        selection,
    )

    assert result is expected_output
    if pre_resolved:
        controller.index.cluster_store.resolve_cluster_blocks.assert_not_called()
    else:
        controller.index.cluster_store.resolve_cluster_blocks.assert_called_once_with(
            layer_name="layer",
            cluster_ids=cluster_ids,
            logical_page_ids=page_ids,
            mode="verification",
        )
    if expect_admission:
        controller.index.cluster_store.admit_staged_clusters.assert_called_once_with(
            layer_name="layer",
            cluster_ids=cluster_ids,
            logical_page_ids=page_ids,
            staging_page_ids=staging_page_ids,
            staging_key_pages=staging_keys,
            staging_value_pages=staging_values,
        )
        assert call_order == ["attention", "admit"]
    else:
        controller.index.cluster_store.admit_staged_clusters.assert_not_called()
        assert call_order == ["attention"]
    source = run.call_args.args[0]
    assert isinstance(source, RetroSpecExactKVSource)
    assert source.primary.key_cache is key_cache
    assert source.primary.value_cache is value_cache
    assert source.primary.token_indices is plan.primary_exact_token_indices
    assert source.primary.token_mask is plan.primary_exact_token_mask
    assert source.page_token_counts is page_counts
    assert source.resident_pages is not None
    assert source.staging_pages is not None

    resident_source = source.resident_pages
    staging_source = source.staging_pages
    assert resident_source.page_ids is resident_page_ids
    assert resident_source.key_pages is resident_keys
    assert resident_source.value_pages is resident_values
    assert resident_source.ready_event is resident_ready_event
    assert staging_source.page_ids is staging_page_ids
    assert staging_source.key_pages is staging_keys
    assert staging_source.value_pages is staging_values
    assert staging_source.ready_event is None


def test_verification_reuses_draft_selection_plan_without_reranking():
    controller = make_controller()
    mark_installed(controller)
    selection = make_selection(batch_size=1)
    controller.index.select_segmented = Mock(return_value=selection)
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

        controller.index.select_segmented.assert_called_once()
        controller.index.materialize.assert_called_once()
        materialize_args = controller.index.materialize.call_args.args
        assert materialize_args[0] is selection.plan
        assert materialize_args[1] == RetroSpecAttentionLevel.EXPANDED
        assert torch.equal(materialize_args[2], kv_cache[0])
        assert torch.equal(materialize_args[3], kv_cache[1])
        assert materialize_args[4] is metadata.block_table


def test_segmented_draft_prefetches_sparse_plan_after_attention():
    controller = make_controller(
        cache_ratio=0.5,
    )
    mark_installed(controller)
    assert isinstance(controller.index, RetroSpecSegmentedTokenIndex)

    plan = make_token_plan(2, num_kv_heads=1, exact_width=0, estimation_width=0)
    selection = RetroSpecTokenAttentionSelection(
        exact_cluster_ids=plan.sparse_exact_cluster_ids,
        exact_page_ids=plan.sparse_exact_page_ids,
        exact_page_token_counts=plan.sparse_exact_page_token_counts,
        exact_token_counts=torch.zeros(2, 1, dtype=torch.int32),
        estimation_keys=plan.sparse_estimation_keys,
        estimation_values=plan.sparse_estimation_values,
        estimation_token_counts=plan.sparse_estimation_token_counts,
        attention_mass=torch.ones(2),
        plan=plan,
        resolved_pages=None,
    )
    controller.index.select_segmented = Mock(return_value=selection)

    call_order: list[str] = []
    controller._run_exact_attention = Mock(
        side_effect=lambda *args: call_order.append("exact")
        or (torch.zeros(2, 1, 1), torch.zeros(2, 1))
    )
    controller._run_estimation_attention = Mock(
        side_effect=lambda *args: call_order.append("estimation")
        or (torch.zeros(2, 1, 1), torch.zeros(2, 1))
    )
    controller.index.prefetch_sparse_verification = Mock(
        side_effect=lambda **kwargs: call_order.append("prefetch")
    )
    controller.index.complete_first_draft_warmup = Mock()

    impl = cast(FlashAttentionImpl, SimpleNamespace(scale=1.0))
    layer = cast(torch.nn.Module, SimpleNamespace())
    query = torch.zeros(2, 1, 1)
    kv_cache = torch.zeros(2, 2, 2, 1, 1)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=1,
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
    )
    output = torch.zeros(2, 1, 1)
    active_mask = torch.tensor([True, False])

    with (
        patch("vllm.v1.spec_decode.retrospec.attention.merge_attn_states"),
        controller.proposal_context(["request-0", "request-1"]),
    ):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, active_mask)
        controller._sparse_forward(
            "layer", impl, layer, query, kv_cache, metadata, output
        )
        controller.end_step()

    assert call_order == ["exact", "estimation", "prefetch"]
    controller.index.prefetch_sparse_verification.assert_called_once_with(
        selection=selection,
        active_mask=active_mask,
    )
    assert controller.index.select_segmented.call_args.kwargs["warm_first_draft"]
    controller.index.complete_first_draft_warmup.assert_called_once_with(
        ("request-0", "request-1"), ("layer",), active_mask
    )


def test_parallel_verification_gathers_segmented_token_plan_rows():
    controller = make_controller()
    mark_installed(controller)
    step_zero = replace(
        make_token_plan(2, num_kv_heads=1, exact_width=2, estimation_width=1),
        primary_exact_token_indices=torch.tensor([[[0, 1]], [[2, 3]]]),
        sparse_attn=torch.tensor([0.1, 0.2]),
    )
    step_one = replace(
        make_token_plan(2, num_kv_heads=1, exact_width=2, estimation_width=1),
        primary_exact_token_indices=torch.tensor([[[4, 5]], [[6, 7]]]),
        sparse_attn=torch.tensor([0.3, 0.4]),
    )

    with controller.proposal_context(["request-0", "request-1"]):
        controller.selection_plans[0]["layer"] = step_zero
        controller.selection_plans[1]["layer"] = step_one
        controller.begin_parallel_step(
            RetroSpecAttentionMode.EXPANDED_VERIFY,
            request_indices=torch.tensor([1, 0, 1], dtype=torch.int64),
            token_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        )

        plan = controller._gather_parallel_plan("layer")
        assert isinstance(plan, RetroSpecTokenSelectionPlan)
        assert plan.primary_exact_token_indices.tolist() == [
            [[2, 3]],
            [[4, 5]],
            [[6, 7]],
        ]
        assert plan.sparse_attn.tolist() == pytest.approx([0.2, 0.3, 0.4])

        controller.attention_mass_layer_count = 1
        controller.end_step()


def test_parallel_verification_rejects_missing_token_plan():
    controller = make_controller()
    mark_installed(controller)

    with controller.proposal_context(["request"]):
        controller.selection_plans[0]["layer"] = make_token_plan(
            1, num_kv_heads=1, exact_width=1, estimation_width=1
        )
        controller.begin_parallel_step(
            RetroSpecAttentionMode.SPARSE_VERIFY,
            request_indices=torch.tensor([0], dtype=torch.int64),
            token_indices=torch.tensor([1], dtype=torch.int64),
        )

        with pytest.raises(RuntimeError, match="selection plan is missing"):
            controller._gather_parallel_plan("layer")


def test_end_step_requires_completed_attention_layer():
    controller = make_controller()
    mark_installed(controller)

    with controller.proposal_context(["request"]):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, torch.tensor([True]))
        with pytest.raises(RuntimeError, match="No attention layer ran"):
            controller.end_step()
