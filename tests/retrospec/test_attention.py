# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Controller contracts shared by the GPU-native RetroSpec attention path.

Kernel arithmetic and GPU index layout are covered by test_gpu_native.py.
CPU page, resident-cache, and transfer-ring contracts belonged to the
offload implementation and are intentionally absent from this branch.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.retrospec.attention import (
    RetroSpecAttentionMode,
    RetroSpecSparseAttention,
)

pytestmark = pytest.mark.cpu_test


def make_controller(
    *, replay_mode: str = "off", max_num_seqs: int = 4, num_speculative_tokens: int = 8
) -> RetroSpecSparseAttention:
    config = cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="retrospec",
                num_speculative_tokens=num_speculative_tokens,
                retrospec_retrieval_ratio=0.125,
                retrospec_estimation_ratio=0.25,
                retrospec_sparse_verify_exact_fraction=0.875,
                retrospec_index_segment_size=64,
                retrospec_index_update_interval=32,
                retrospec_blocks_per_cluster=1,
                retrospec_kmeans_iterations=2,
                retrospec_hit_attn_threshold=None,
                retrospec_retrieval_attn_threshold=None,
                retrospec_expanded_attn_threshold=None,
                retrospec_replay_mode=replay_mode,
                retrospec_stats_interval_seconds=0.0,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        ),
    )
    return RetroSpecSparseAttention(config, torch.device("cpu"))


def mark_installed(controller: RetroSpecSparseAttention) -> None:
    controller.original_forwards["layer"] = cast(Any, (object(), Mock()))


def test_gpu_native_controller_uses_native_kv_and_shared_statistics():
    controller = make_controller(max_num_seqs=2, num_speculative_tokens=4)

    assert not controller.uses_full_verification_offload
    assert not controller.selection_provenance_enabled
    assert controller.max_parallel_tokens == 8
    assert controller.index.performance_stats is controller.performance_stats
    assert controller.index.sparse_verify_exact_fraction == 0.875
    assert not controller.has_retired_kv_blocks(["request"])
    assert controller.take_kv_cache_retirement_ranges(["request"]) == []


def test_gpu_native_controller_rejects_legacy_resident_replay():
    with pytest.raises(ValueError, match="does not support resident replay"):
        make_controller(replay_mode="trace")


def test_proposal_context_requires_installed_attention_and_restores_state():
    controller = make_controller()
    with (
        pytest.raises(RuntimeError, match="installed before proposing"),
        controller.proposal_context(["request"]),
    ):
        pass

    mark_installed(controller)
    with (
        patch.object(controller.index, "begin_proposal") as begin,
        patch.object(controller.index, "end_proposal") as end,
        patch.object(controller.index, "flush_sparse_verification_prefetch") as flush,
    ):
        with (
            pytest.raises(LookupError, match="test failure"),
            controller.proposal_context(["request"], [32]),
        ):
            assert controller.in_proposal
            assert controller.proposal_request_ids == ("request",)
            assert controller.proposal_context_lens == (32,)
            controller.set_proposal_round(2)
            with (
                pytest.raises(RuntimeError, match="cannot be nested"),
                controller.proposal_context(["other"]),
            ):
                pass
            raise LookupError("test failure")

        begin.assert_called_once_with(("request",))
        flush.assert_called_once_with()
        end.assert_called_once_with()
    assert not controller.in_proposal
    assert controller.proposal_request_ids == ()
    assert controller.proposal_context_lens == ()
    assert controller.proposal_round == 0
    assert controller.mode == RetroSpecAttentionMode.PASSTHROUGH


@pytest.mark.parametrize("context_lens", [[-1], [1, 2]])
def test_proposal_context_rejects_invalid_lengths(context_lens: list[int]):
    controller = make_controller()
    mark_installed(controller)
    with (
        pytest.raises(ValueError),
        controller.proposal_context(["request"], context_lens),
    ):
        pass
    assert not controller.in_proposal


def test_proposal_round_is_scoped_and_monotonic():
    controller = make_controller()
    mark_installed(controller)
    with pytest.raises(RuntimeError, match="only inside proposal_context"):
        controller.set_proposal_round(1)

    with controller.proposal_context(["request"]):
        controller.set_proposal_round(1)
        controller.set_proposal_round(3)
        assert controller.proposal_round == 3
        with pytest.raises(ValueError, match="monotonic"):
            controller.set_proposal_round(2)
        with pytest.raises(ValueError, match="positive"):
            controller.set_proposal_round(0)


def test_draft_step_statistics_reset_after_completion():
    controller = make_controller()
    mark_installed(controller)
    active = torch.tensor([True, False])

    with controller.proposal_context(["first", "second"]):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, active)
        with pytest.raises(RuntimeError, match="still active"):
            controller.begin_step(RetroSpecAttentionMode.DRAFT, 1, active)
        controller.attention_mass_sum[:2].copy_(torch.tensor([1.5, 0.5]))
        controller.attention_mass_layer_count = 2
        torch.testing.assert_close(controller.end_step(), torch.tensor([0.75, 0.25]))
        assert not controller.step_active
        assert controller.mode == RetroSpecAttentionMode.PASSTHROUGH
        assert controller.active_mask is None


def test_draft_step_rejects_invalid_mask_and_missing_attention_layer():
    controller = make_controller()
    mark_installed(controller)
    with controller.proposal_context(["request"]):
        with pytest.raises(ValueError, match="one-dimensional boolean"):
            controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, torch.ones(1))
        with pytest.raises(ValueError, match="outside the speculative token range"):
            controller.begin_step(
                RetroSpecAttentionMode.DRAFT,
                controller.num_speculative_tokens,
                torch.ones(1, dtype=torch.bool),
            )
        controller.begin_step(
            RetroSpecAttentionMode.DRAFT, 0, torch.ones(1, dtype=torch.bool)
        )
        with pytest.raises(RuntimeError, match="No attention layer ran"):
            controller.end_step_statistics()
        controller.abort_step()
        assert not controller.step_active


def test_parallel_verification_tracks_request_and_token_rows():
    controller = make_controller()
    mark_installed(controller)
    request_indices = torch.tensor([0, 1, 0], dtype=torch.int32)
    token_indices = torch.tensor([0, 2, 3], dtype=torch.int32)

    with controller.proposal_context(["first", "second"]):
        with pytest.raises(ValueError, match="split ordinary and bonus"):
            controller.begin_parallel_step(
                RetroSpecAttentionMode.SPARSE_VERIFY,
                request_indices,
                token_indices,
                bonus_start_index=3,
            )
        controller.begin_parallel_step(
            RetroSpecAttentionMode.SPARSE_VERIFY,
            request_indices,
            token_indices,
            bonus_start_index=2,
        )
        assert controller.parallel_request_indices is request_indices
        assert controller.parallel_token_indices is token_indices
        assert controller.parallel_bonus_start_index == 2
        controller.attention_mass_sum[:3].fill_(0.5)
        controller.attention_mass_layer_count = 1
        with patch.object(
            controller.index, "end_indexed_verification_transaction"
        ) as end:
            torch.testing.assert_close(controller.end_step(), torch.full((3,), 0.5))
            end.assert_called_once_with()
        assert controller.parallel_request_indices is None
        assert controller.parallel_bonus_start_index is None


def test_index_update_context_flushes_and_clears_metadata():
    controller = make_controller()
    with patch.object(controller.index, "flush_staged_updates") as flush:
        with controller.index_update_context(["request"], [64], [True], [True], [0]):
            assert controller.index_update_active
            assert controller.index_update_request_ids == ("request",)
            assert controller.index_update_build_rows == (0,)
            with (
                pytest.raises(RuntimeError, match="cannot be nested"),
                controller.index_update_context(["request"], [64], [True], [True], [0]),
            ):
                pass
        flush.assert_called_once_with()
    assert not controller.index_update_active
    assert controller.index_update_request_ids == ()
    assert controller.index_update_build_rows == ()


def test_index_update_context_discards_staged_changes_on_failure():
    controller = make_controller()
    with patch.object(controller.index, "discard_staged_updates") as discard:
        with (
            pytest.raises(LookupError, match="test failure"),
            controller.index_update_context(["request"], [64], [True], [True], [0]),
        ):
            raise LookupError("test failure")
        discard.assert_called_once_with()
    assert not controller.index_update_active


@pytest.mark.parametrize(
    ("seq_lens", "is_prefill", "prefill_complete", "build_rows", "message"),
    [
        ([32, 64], [True], [True], [0], "seq_lens must match"),
        ([32], [True, False], [True], [0], "is_prefill must match"),
        ([32], [True], [True, False], [0], "prefill_complete must match"),
        ([32], [False], [True], [0], "requires is_prefill"),
        ([32], [True], [True], [0, 0], "build_rows must be unique"),
    ],
)
def test_index_update_context_validates_descriptors(
    seq_lens: list[int],
    is_prefill: list[bool],
    prefill_complete: list[bool],
    build_rows: list[int],
    message: str,
):
    controller = make_controller()
    with (
        pytest.raises(ValueError, match=message),
        controller.index_update_context(
            ["request"], seq_lens, is_prefill, prefill_complete, build_rows
        ),
    ):
        pass


def test_passthrough_forward_updates_gpu_index_after_original_attention():
    controller = make_controller()
    expected = torch.tensor([1.0])
    original_forward = Mock(return_value=expected)
    query = torch.zeros(1, 1, 1)
    kv_cache = torch.zeros(2, 1, 1)
    metadata = SimpleNamespace(block_table=torch.zeros(1, 1, dtype=torch.int32))

    with (
        patch.object(controller.index, "build_or_update") as update,
        patch.object(controller.index, "flush_staged_updates") as flush,
    ):
        with controller.index_update_context(["request"], [64], [True], [True], [0]):
            result = controller.forward(
                "layer",
                original_forward,
                object(),
                query,
                query,
                query,
                kv_cache,
                cast(Any, metadata),
                expected,
            )
            update.assert_called_once()
            assert update.call_args.kwargs["request_ids"] == ("request",)
            assert update.call_args.kwargs["prefill_complete"] == (True,)
        flush.assert_called_once_with()

    assert result is expected
    original_forward.assert_called_once()
    assert not controller.index_update_active


def test_draft_without_cluster_pages_uses_original_attention():
    controller = make_controller()
    mark_installed(controller)
    query = torch.zeros(1, 1, 4)
    kv_cache = torch.zeros(2, 1, 4)
    output = torch.zeros_like(query)
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    original_forward = Mock(return_value=output)

    with (
        controller.proposal_context(["request"]),
        patch.object(controller.index, "has_cluster_pages", return_value=False),
    ):
        controller.begin_step(RetroSpecAttentionMode.DRAFT, 0, torch.tensor([True]))
        result = controller._sparse_forward(
            "layer",
            original_forward,
            cast(Any, SimpleNamespace(scale=1.0)),
            cast(Any, object()),
            query,
            query,
            query,
            kv_cache,
            cast(Any, metadata),
            output,
        )
        assert result is output
        torch.testing.assert_close(controller.end_step(), torch.ones(1))
    original_forward.assert_called_once()


def test_parallel_sparse_verification_passes_request_rows_to_native_index():
    controller = make_controller()
    mark_installed(controller)
    request_indices = torch.tensor([1, 0], dtype=torch.int32)
    token_indices = torch.tensor([2, 3], dtype=torch.int32)
    query = torch.zeros(2, 1, 4)
    kv_cache = torch.zeros(2, 1, 4)
    output = torch.zeros_like(query)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=1,
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        seq_lens=torch.tensor([3, 4], dtype=torch.int32),
    )
    original_forward = Mock()

    with (
        controller.proposal_context(["first", "second"]),
        patch.object(controller.index, "has_cluster_pages", return_value=True),
        patch.object(
            controller.index, "forward", return_value=torch.tensor([0.4, 0.8])
        ) as native_forward,
    ):
        controller.begin_parallel_step(
            RetroSpecAttentionMode.SPARSE_VERIFY, request_indices, token_indices
        )
        result = controller._sparse_forward(
            "layer",
            original_forward,
            cast(Any, SimpleNamespace(scale=1.0)),
            cast(Any, object()),
            query,
            query,
            query,
            kv_cache,
            cast(Any, metadata),
            output,
        )
        assert result is output
        assert native_forward.call_args.kwargs["request_indices"] is request_indices
        assert native_forward.call_args.kwargs["token_indices"] is token_indices
        assert native_forward.call_args.kwargs["sparse_verify"] is True
        torch.testing.assert_close(controller.end_step(), torch.tensor([0.4, 0.8]))
    original_forward.assert_not_called()


def test_native_full_verification_has_no_transfer_or_retirement():
    controller = make_controller()
    mark_installed(controller)
    with (
        pytest.raises(RuntimeError, match="original vLLM attention"),
        controller.full_verification_context(["request"], [32], [1]),
    ):
        pass
    with controller.proposal_context(["request"]):
        assert not controller.maybe_prime_full_verification(8)
    with pytest.raises(RuntimeError, match="requires a proposal"):
        controller.maybe_prime_full_verification(8)


def test_legacy_layer_major_prefill_is_not_used_by_gpu_native_mode():
    controller = make_controller()
    with pytest.raises(NotImplementedError, match="ordinary chunked prefill"):
        controller.commit_layer_major_prefill("request", ["layer"])


def test_install_and_uninstall_restore_original_attention_forward():
    controller = make_controller()
    original_forward = Mock()
    impl = SimpleNamespace(forward=original_forward)
    layer = SimpleNamespace(impl=impl)
    with patch.object(controller, "_validate_layer", return_value=impl):
        controller.install({"layer": cast(Any, layer)})
    assert impl.forward is controller.forward_wrappers["layer"]
    controller.uninstall()
    assert impl.forward is original_forward
    assert not controller.original_forwards
