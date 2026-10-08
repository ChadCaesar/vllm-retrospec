# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from tests.retrospec.support.proposer import (
    disable_pin_memory_for_cpu_tests as disable_pin_memory_for_cpu_tests,
)
from tests.retrospec.support.proposer import (
    make_runner,
    make_vllm_config,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.retrospec import (
    RetroSpecPipelineStage,
    RetroSpecProposer,
    transition_trace,
)
from vllm.v1.spec_decode.retrospec.decision import RetroSpecMetrics
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


def test_close_uninstalls_sparse_attention_once():
    proposer = RetroSpecProposer.__new__(RetroSpecProposer)
    proposer.sparse_attention = Mock()
    proposer._closed = False

    proposer.close()
    proposer.close()

    proposer.sparse_attention.uninstall.assert_called_once_with()


def test_retrospec_proposer_initialization():
    vllm_config = make_vllm_config()
    runner = make_runner()
    device = torch.device("cpu")

    proposer = RetroSpecProposer(vllm_config, device, runner)

    assert proposer.vllm_config is vllm_config
    assert proposer.device == device
    assert proposer.runner is runner
    assert proposer.model is None
    assert proposer.num_speculative_tokens == 4
    assert proposer.max_batch_size == 8
    assert proposer.pipeline_protocol.max_batch_size == 8
    assert proposer.pipeline_protocol.device == device
    assert proposer.policy.max_draft_tokens == 4
    assert proposer.transition_tracer.enabled is False
    assert proposer.state.max_batch_size == 8
    assert proposer.state.device == device
    assert proposer.attn_metadata_builder is None
    assert proposer.attn_layer_names == []
    assert proposer._cudagraph_registration_failure == "uninitialized"


def test_draft_transition_trace_records_only_stopping_requests(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_max_draft_tokens=2,
            retrospec_trace_transitions=True,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.state.begin_batch(2)
    proposer.state.add_draft_counts(torch.tensor([1, 2], dtype=torch.int32))
    proposer.positions[:2].copy_(torch.tensor([10, 20]))
    projected_pending_counts = proposer.state.draft_counts.clone()
    generation_limit_reached = torch.zeros(2, dtype=torch.bool)
    index_update_required = torch.zeros(2, dtype=torch.bool)
    decision = proposer.policy.evaluate(
        current_stage=RetroSpecStage.DRAFT,
        request_stages=proposer.state.stage,
        metrics=RetroSpecMetrics(hit_attn=torch.ones(2)),
        draft_counts=proposer.state.draft_counts,
        pending_counts=projected_pending_counts,
        active_mask=proposer.state.active_mask,
        generation_limit_reached=generation_limit_reached,
        index_update_required=index_update_required,
    )
    transition_mask = decision.next_stage == int(RetroSpecStage.SPARSE_VERIFY)
    logger_info = Mock()
    monkeypatch.setattr(transition_trace.logger, "info", logger_info)

    proposer._trace_draft_transition(
        request_ids=["request-0", "request-1"],
        proposal_round=4,
        transition_mask=transition_mask,
        decision=decision,
        draft_margin=None,
        hit_attn=torch.ones(2),
        projected_pending_counts=projected_pending_counts,
        generation_limit_reached=generation_limit_reached,
        index_update_required=index_update_required,
    )

    payload = json.loads(logger_info.call_args.args[1])
    assert payload["proposal_round"] == 4
    assert payload["records"][0]["request_id"] == "request-1"
    assert payload["records"][0]["position"] == 21
    assert payload["records"][0]["reason_names"] == ["MAX_DRAFT_TOKENS"]


def test_initialize_cudagraph_keys_defers_dynamic_layout_capture():
    dispatcher = Mock()
    dispatcher.register_piecewise_cudagraph_sizes.return_value = (
        1,
        2,
        4,
        8,
        16,
        32,
    )
    dispatcher.has_piecewise_cudagraph_namespace.return_value = True
    runner_input_ids = torch.empty(32, dtype=torch.int32)
    runner_positions = torch.empty(32, dtype=torch.int64)
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(
            cudagraph_dispatcher=dispatcher,
            input_ids=SimpleNamespace(gpu=runner_input_ids),
            positions=SimpleNamespace(gpu=runner_positions),
        ),
    )

    proposer.initialize_cudagraph_keys(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        [1, 2, 4, 8, 16, 64],
        32,
    )

    dispatcher.register_piecewise_cudagraph_sizes.assert_not_called()
    dispatcher.has_piecewise_cudagraph_namespace.assert_not_called()
    assert proposer._cudagraph_registration_failure == "gpu_native_dynamic_layout"


def test_initialize_cudagraph_keys_records_piecewise_disabled():
    dispatcher = Mock()
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(cudagraph_dispatcher=dispatcher),
    )

    proposer.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, [1, 2, 4], 4)

    dispatcher.register_piecewise_cudagraph_sizes.assert_not_called()
    assert proposer._cudagraph_registration_failure == "gpu_native_dynamic_layout"


def test_initialize_cudagraph_keys_skips_unsupported_data_parallelism():
    dispatcher = Mock()
    config = make_vllm_config(enforce_eager=False)
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=2,
    )
    proposer = RetroSpecProposer(
        config,
        torch.device("cpu"),
        make_runner(cudagraph_dispatcher=dispatcher),
    )

    proposer.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE, [1, 2, 4], 4)

    dispatcher.register_piecewise_cudagraph_sizes.assert_not_called()
    assert proposer._cudagraph_registration_failure == "gpu_native_dynamic_layout"


def test_initialize_cudagraph_keys_requires_runner_input_workspace():
    dispatcher = Mock()
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(cudagraph_dispatcher=dispatcher),
    )

    proposer.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE, [1, 2, 4], 4)

    dispatcher.register_piecewise_cudagraph_sizes.assert_not_called()
    assert proposer._cudagraph_registration_failure == "gpu_native_dynamic_layout"


def test_pipeline_graph_receive_destination_uses_runner_workspace():
    workspace = IntermediateTensors(
        {
            "hidden_states": torch.empty(8, 4),
            "residual": torch.empty(8, 4),
        }
    )
    destination = workspace[:4]
    slice_workspace = Mock(return_value=destination)
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(
            intermediate_tensors=workspace,
            sync_and_slice_intermediate_tensors=slice_workspace,
        ),
    )
    stage = RetroSpecPipelineStage(1, 2, 1, 2)

    result = proposer._get_pipeline_receive_destination(
        stage, num_tokens=4, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )

    assert result is destination
    slice_workspace.assert_called_once_with(
        4, intermediate_tensors=None, sync_self=False
    )


def test_pipeline_receive_destination_is_unused_for_first_or_eager_stage():
    slice_workspace = Mock()
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(
            intermediate_tensors=None,
            sync_and_slice_intermediate_tensors=slice_workspace,
        ),
    )

    assert (
        proposer._get_pipeline_receive_destination(
            RetroSpecPipelineStage(0, 2, 0, 1),
            num_tokens=4,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        is None
    )
    assert (
        proposer._get_pipeline_receive_destination(
            RetroSpecPipelineStage(1, 2, 1, 2),
            num_tokens=4,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        is None
    )
    slice_workspace.assert_not_called()


def test_pipeline_graph_receive_destination_requires_runner_workspace():
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(intermediate_tensors=None),
    )

    with pytest.raises(RuntimeError, match="requires the runner"):
        proposer._get_pipeline_receive_destination(
            RetroSpecPipelineStage(1, 2, 1, 2),
            num_tokens=4,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )


def test_pipeline_stage_model_receives_into_graph_workspace():
    workspace = IntermediateTensors(
        {
            "hidden_states": torch.empty(4, 4),
            "residual": torch.empty(4, 4),
        }
    )
    slice_workspace = Mock(return_value=workspace)
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(
            intermediate_tensors=workspace,
            sync_and_slice_intermediate_tensors=slice_workspace,
        ),
    )
    proposer.pipeline_stage = RetroSpecPipelineStage(1, 2, 1, 2)
    proposer.pipeline_protocol.receive_model_input = Mock(return_value=workspace)
    proposer.model = Mock(return_value=torch.ones(4, 4))
    proposer.performance_stats.cpu_timer = Mock(return_value=nullcontext())
    proposer.performance_stats.cuda_timer = Mock(return_value=nullcontext())
    positions = torch.arange(4)

    output = proposer._run_pipeline_stage_model(
        torch.zeros(4, dtype=torch.int32),
        positions,
        num_tokens=4,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        stage_name="draft",
    )

    proposer.pipeline_protocol.receive_model_input.assert_called_once_with(
        proposer.pipeline_stage, 4, workspace
    )
    model_kwargs = proposer.model.call_args.kwargs
    assert model_kwargs["input_ids"] is None
    assert model_kwargs["positions"] is positions
    assert model_kwargs["intermediate_tensors"] is workspace
    assert model_kwargs["inputs_embeds"] is None
    proposer.performance_stats.cpu_timer.assert_called_once_with(
        "draft_pipeline_receive_wall"
    )
    proposer.performance_stats.cuda_timer.assert_called_once_with(
        "draft_pipeline_receive"
    )
    torch.testing.assert_close(output, torch.ones(4, 4))


def test_pipeline_stage_model_times_nonfinal_send_only():
    output = IntermediateTensors(
        {
            "hidden_states": torch.ones(4, 4),
            "residual": torch.ones(4, 4),
        }
    )
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    proposer.pipeline_stage = RetroSpecPipelineStage(0, 2, 0, 1)
    proposer.pipeline_protocol.receive_model_input = Mock(return_value=None)
    proposer.pipeline_protocol.send_model_output = Mock()
    proposer.model = Mock(return_value=output)
    proposer.performance_stats.cpu_timer = Mock(return_value=nullcontext())
    proposer.performance_stats.cuda_timer = Mock(return_value=nullcontext())

    result = proposer._run_pipeline_stage_model(
        torch.zeros(4, dtype=torch.int32),
        torch.arange(4),
        num_tokens=4,
        cudagraph_mode=CUDAGraphMode.NONE,
        stage_name="sparse_verify",
    )

    assert result is None
    proposer.performance_stats.cpu_timer.assert_called_once_with(
        "sparse_verify_pipeline_send_wall"
    )
    proposer.performance_stats.cuda_timer.assert_called_once_with(
        "sparse_verify_pipeline_send"
    )
    proposer.pipeline_protocol.send_model_output.assert_called_once_with(
        proposer.pipeline_stage, output, 4
    )


def test_piecewise_model_inputs_preserve_eager_views():
    dispatcher = Mock()
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(cudagraph_dispatcher=dispatcher),
    )
    input_ids = torch.tensor([3, 4], dtype=torch.int32)
    positions = torch.tensor([7, 8], dtype=torch.int64)
    slot_mapping = proposer._slot_mapping[:2]
    slot_mapping.copy_(torch.tensor([11, 12]))

    result = proposer._prepare_piecewise_model_inputs(
        input_ids,
        positions,
        slot_mapping,
        proposer._slot_mapping,
        "draft",
    )

    assert result[0] is input_ids
    assert result[1] is positions
    assert result[2] is slot_mapping
    assert result[3] == CUDAGraphMode.NONE
    assert result[4] == BatchDescriptor(2)
    dispatcher.dispatch_piecewise_cudagraph.assert_not_called()


def test_piecewise_model_inputs_keep_eager_views_for_dynamic_layout():
    dispatcher = Mock(
        dispatch_piecewise_cudagraph=Mock(
            return_value=(CUDAGraphMode.PIECEWISE, BatchDescriptor(4))
        )
    )
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(cudagraph_dispatcher=dispatcher),
    )
    proposer._cudagraph_registration_failure = None
    input_ids = torch.tensor([3, 4, 5], dtype=torch.int32)
    positions = torch.tensor([7, 8, 9], dtype=torch.int64)
    slot_mapping = proposer._slot_mapping[:3]
    slot_mapping.copy_(torch.tensor([11, 12, 13]))

    result = proposer._prepare_piecewise_model_inputs(
        input_ids,
        positions,
        slot_mapping,
        proposer._slot_mapping,
        "draft",
    )

    graph_input_ids, graph_positions, graph_slots, mode, descriptor = result
    assert graph_input_ids is input_ids
    assert graph_positions is positions
    assert graph_slots is slot_mapping
    assert graph_slots.tolist() == [11, 12, 13]
    assert mode == CUDAGraphMode.NONE
    assert descriptor == BatchDescriptor(3)
    dispatcher.dispatch_piecewise_cudagraph.assert_not_called()


def test_piecewise_model_inputs_fall_back_when_bucket_exceeds_capacity():
    dispatcher = Mock(
        dispatch_piecewise_cudagraph=Mock(
            return_value=(CUDAGraphMode.PIECEWISE, BatchDescriptor(16))
        )
    )
    proposer = RetroSpecProposer(
        make_vllm_config(enforce_eager=False),
        torch.device("cpu"),
        make_runner(cudagraph_dispatcher=dispatcher),
    )
    proposer._cudagraph_registration_failure = None
    input_ids = torch.tensor([3, 4, 5], dtype=torch.int32)
    positions = torch.tensor([7, 8, 9], dtype=torch.int64)
    slot_mapping = proposer._slot_mapping[:3]

    result = proposer._prepare_piecewise_model_inputs(
        input_ids,
        positions,
        slot_mapping,
        proposer._slot_mapping,
        "draft",
    )

    assert result[0] is input_ids
    assert result[1] is positions
    assert result[2] is slot_mapping
    assert result[3] == CUDAGraphMode.NONE
    assert result[4] == BatchDescriptor(3)


def test_retrospec_proposer_loads_target_model():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    target_model = Mock()

    install = Mock()
    proposer.sparse_attention.install = install
    attention_layer = Mock()

    layer_model = SimpleNamespace(start_layer=0, end_layer=1)
    with (
        patch(
            "vllm.v1.spec_decode.retrospec.proposer.get_layers_from_vllm_config",
            return_value={"model.layers.0.self_attn.attn": attention_layer},
        ),
        patch(
            "vllm.v1.spec_decode.retrospec.proposer.resolve_retrospec_layer_model",
            return_value=layer_model,
        ),
    ):
        proposer.load_model(target_model)

    assert proposer.model is target_model
    assert proposer.attn_layer_names == ["model.layers.0.self_attn.attn"]
    assert proposer.pipeline_stage == RetroSpecPipelineStage(0, 1, 0, 1)
    install.assert_called_once_with({"model.layers.0.self_attn.attn": attention_layer})


@pytest.mark.parametrize(
    ("config", "runner", "message"),
    [
        (
            make_vllm_config(disable_padded_drafter_batch=True),
            make_runner(),
            "padded drafter batches",
        ),
        (
            make_vllm_config(async_scheduling=True),
            make_runner(),
            "async scheduling",
        ),
        (
            make_vllm_config(),
            make_runner(supports_mm_inputs=True),
            "multimodal models",
        ),
        (
            make_vllm_config(),
            make_runner(uses_mrope=True),
            "one-dimensional RoPE",
        ),
    ],
)
def test_retrospec_proposer_rejects_unsupported_features(
    config: VllmConfig,
    runner: Any,
    message: str,
):
    with pytest.raises(NotImplementedError, match=message):
        RetroSpecProposer(config, torch.device("cpu"), runner)


def test_retrospec_proposer_delegates_phase_aware_index_updates():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    expected = nullcontext()
    proposer.sparse_attention.needs_index_update = Mock(return_value=True)
    proposer.sparse_attention.index_update_context = Mock(return_value=expected)

    assert proposer.needs_index_update("request", 12, False, False)
    context = proposer.index_update_context(
        request_ids=["request"],
        seq_lens=[12],
        is_prefill=[False],
        prefill_complete=[False],
        build_rows=[0],
    )

    assert context is expected
    proposer.sparse_attention.needs_index_update.assert_called_once_with(
        "request", 12, False, False
    )
    proposer.sparse_attention.index_update_context.assert_called_once_with(
        ["request"], [12], [False], [False], [0]
    )


def test_retrospec_proposer_resolves_kv_cache_group():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.attn_layer_names = ["layer"]
    kv_cache_config = cast(
        KVCacheConfig,
        SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=["other"]),
                SimpleNamespace(layer_names=["layer"]),
            ]
        ),
    )
    proposer.validate_same_kv_cache_group(kv_cache_config)
    assert proposer.kv_cache_group_id == 1
