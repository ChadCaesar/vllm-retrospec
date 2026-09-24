# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import CUDAGraphMode, SpeculativeConfig, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.retrospec import (
    RetroSpecAttentionMassStats,
    RetroSpecAttentionMode,
    RetroSpecPipelineStage,
    RetroSpecProposer,
    transition_trace,
)
from vllm.v1.spec_decode.retrospec.decision import RetroSpecMetrics
from vllm.v1.spec_decode.retrospec.proposer import (
    RetroSpecParallelVerificationOutput,
    RetroSpecVerificationResult,
)
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


@pytest.fixture(autouse=True)
def disable_pin_memory_for_cpu_tests(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.is_pin_memory_available",
        lambda: False,
    )
    pp_group = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        is_last_rank=True,
        last_rank=0,
        device_group=None,
        all_reduce=lambda tensor: tensor,
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group", lambda: pp_group
    )


def make_vllm_config(
    *,
    max_model_len: int = 16,
    async_scheduling: bool = False,
    **spec_overrides: Any,
) -> VllmConfig:
    spec_values = {
        "method": "retrospec",
        "num_speculative_tokens": 4,
        "retrospec_max_draft_tokens": 4,
        **spec_overrides,
    }
    return cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SpeculativeConfig(**spec_values),
            scheduler_config=SimpleNamespace(
                max_num_seqs=8,
                async_scheduling=async_scheduling,
            ),
            model_config=SimpleNamespace(
                dtype=torch.float32,
                max_model_len=max_model_len,
                get_hidden_size=Mock(return_value=4),
            ),
            cache_config=SimpleNamespace(block_size=4),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                data_parallel_size=1,
            ),
        ),
    )


def make_runner(**overrides: Any) -> Any:
    values = {
        "supports_mm_inputs": False,
        "uses_mrope": False,
        "uses_xdrope_dim": 0,
        **overrides,
    }
    return SimpleNamespace(**values)


def make_common_metadata(seq_lens: list[int]) -> CommonAttentionMetadata:
    metadata = SimpleNamespace(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        batch_size=lambda: len(seq_lens),
    )
    return cast(CommonAttentionMetadata, metadata)


def make_sampling_metadata(*, all_greedy: bool) -> SamplingMetadata:
    max_batch_size = 8
    return SamplingMetadata(
        temperature=None if all_greedy else torch.ones(max_batch_size),
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(max_batch_size),
        presence_penalties=torch.zeros(max_batch_size),
        repetition_penalties=torch.ones(max_batch_size),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_close_uninstalls_sparse_attention_once():
    proposer = RetroSpecProposer.__new__(RetroSpecProposer)
    proposer.sparse_attention = Mock()
    proposer._closed = False

    proposer.close()
    proposer.close()

    proposer.sparse_attention.uninstall.assert_called_once_with()


def run_proposal(
    proposer: RetroSpecProposer,
    next_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    common_attn_metadata: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
    request_ids: list[str] | None = None,
    committed_positions: list[int] | None = None,
    proposal_active_mask: torch.Tensor | None = None,
    remaining_generation_tokens: list[int] | None = None,
    valid_sampled_tokens_count: torch.Tensor | None = None,
) -> list[list[int]]:
    initialize_single_pipeline_stage(proposer)
    batch_size = common_attn_metadata.batch_size()
    if request_ids is None:
        request_ids = [f"request-{index}" for index in range(batch_size)]
    if committed_positions is None:
        committed_positions = common_attn_metadata.seq_lens.tolist()
    if proposal_active_mask is None:
        proposal_active_mask = torch.ones(batch_size, dtype=torch.bool)
    if remaining_generation_tokens is None:
        remaining_generation_tokens = [proposer.num_speculative_tokens] * batch_size
    if valid_sampled_tokens_count is None:
        valid_sampled_tokens_count = torch.zeros(batch_size, dtype=torch.int32)

    return proposer.propose(
        request_ids=request_ids,
        committed_positions=committed_positions,
        next_token_ids=next_token_ids,
        sampling_metadata=sampling_metadata,
        common_attn_metadata=common_attn_metadata,
        proposal_active_mask=proposal_active_mask,
        remaining_generation_tokens=remaining_generation_tokens,
        valid_sampled_tokens_count=valid_sampled_tokens_count,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )


def initialize_single_pipeline_stage(proposer: RetroSpecProposer) -> None:
    if proposer.pipeline_stage is None:
        num_layers = max(len(proposer.attn_layer_names), 1)
        proposer.pipeline_stage = RetroSpecPipelineStage(0, 1, 0, num_layers)


def attention_stats(size: int) -> RetroSpecAttentionMassStats:
    return RetroSpecAttentionMassStats(torch.ones(size), layer_count=1)


def mock_proposal_execution(
    proposer: RetroSpecProposer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(
        proposer,
        "_verify_draft_tokens",
        lambda *args: RetroSpecVerificationResult(
            verified_counts=proposer.state.draft_counts.clone(),
            require_full=proposer.state.active_mask.clone(),
        ),
    )


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


def test_retrospec_proposer_uses_gpu_native_full_verification():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )

    assert not proposer.uses_full_verification_offload


def test_retrospec_proposer_delegates_full_verification_context():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    expected = nullcontext()
    proposer.sparse_attention.full_verification_context = Mock(return_value=expected)

    result = proposer.full_verification_context(
        request_ids=["request"],
        context_lens=[5],
        query_lens=[2],
    )

    assert result is expected
    proposer.sparse_attention.full_verification_context.assert_called_once_with(
        ["request"],
        [5],
        [2],
    )


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


def test_retrospec_proposer_attaches_kv_cache_group_to_retirement():
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
    proposer.sparse_attention.take_kv_cache_retirement_ranges = Mock(
        return_value=[("request", 1, 4)]
    )

    retirements = proposer.take_kv_cache_retirements(["request"])

    assert len(retirements) == 1
    assert retirements[0].request_id == "request"
    assert retirements[0].kv_cache_group_id == 1
    assert retirements[0].start_block == 1
    assert retirements[0].end_block == 4


def test_propose_rejects_random_sampling():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )

    with pytest.raises(NotImplementedError, match="greedy decoding only"):
        run_proposal(
            proposer,
            torch.tensor([1], dtype=torch.int32),
            make_sampling_metadata(all_greedy=False),
            make_common_metadata([1]),
        )


def test_propose_keeps_partial_prefill_rows_idle(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_max_draft_tokens=1,
            retrospec_stats_interval_seconds=3600.0,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    observed_masks: list[list[bool]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        observed_masks.append(active_mask.tolist())
        return (
            torch.tensor([10, 11], dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([8, 4]),
        proposal_active_mask=torch.tensor([False, True]),
    )

    assert observed_masks == [[False, True]]
    assert result == [[], [11]]
    assert proposer.state.stage.tolist() == [
        int(RetroSpecStage.IDLE),
        int(RetroSpecStage.FULL_VERIFY),
    ]
    assert proposer.state.active_mask.tolist() == [False, True]
    stats = proposer.performance_stats
    proposal_requests_index = stats._gpu_counter_indices["proposal_requests"]
    assert stats._gpu_counters[proposal_requests_index].item() == 1
    assert stats._gpu_histograms["draft_to_sparse_tokens"].tolist() == [0, 1, 0, 0, 0]
    assert stats._cpu_counters["first_proposal_requests"] == 1
    assert stats._cpu_times["first_proposal_batch_wall"][1] == 1
    assert proposer._last_proposed_counts == {"request-1": 1}


def test_propose_stops_requests_independently_on_draft_margin(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_draft_margin_threshold=1.0),
        torch.device("cpu"),
        make_runner(),
    )
    expected_masks = [
        [True, True, True],
        [True, False, True],
        [False, False, True],
    ]
    margins = [
        [2.0, 0.5, 2.0],
        [0.5, 0.0, 2.0],
        [0.0, 0.0, 0.5],
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        assert batch_size == 3
        assert active_mask.tolist() == expected_masks[draft_index]
        token_base = 10 * (draft_index + 1)
        token_ids = torch.tensor(
            [token_base + i for i in range(batch_size)], dtype=torch.int32
        )
        draft_margin = torch.tensor(margins[draft_index])
        hit_attn = torch.ones(batch_size)
        return token_ids, draft_margin, hit_attn

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1, 1]),
    )

    assert result == [[10, 20], [11], [12, 22, 32]]


def test_propose_stops_at_configured_max_draft_tokens(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=3),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1]),
    )

    assert result == [[1, 2, 3], [1, 2, 3]]


def test_propose_respects_per_request_generation_limit(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(max_model_len=6),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        token_base = 10 * (draft_index + 1)
        return (
            torch.tensor(
                [token_base + i for i in range(batch_size)], dtype=torch.int32
            ),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([4, 3, 5]),
    )

    assert result == [[10], [11, 21], []]


def test_propose_respects_per_request_output_budget(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    observed_masks: list[list[bool]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        observed_masks.append(active_mask.tolist())
        return (
            torch.tensor([10, 11, 12], dtype=torch.int32) + draft_index * 10,
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1, 1]),
        remaining_generation_tokens=[1, 3, 0],
        valid_sampled_tokens_count=torch.tensor([1, 1, 0], dtype=torch.int32),
    )

    assert observed_masks == [[False, True, False], [False, True, False]]
    assert result == [[], [11, 21], []]


def test_propose_rolls_back_rejected_tokens_before_drafting(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(max_model_len=6),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.tensor([draft_index + 1], dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([5]),
        num_rejected_tokens_gpu=torch.tensor([2], dtype=torch.int32),
    )

    assert result == [[1, 2]]


def test_run_draft_step_preserves_attention_seq_lens_dtype(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
    )

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert metadata.seq_lens.dtype == common_attn_metadata.seq_lens.dtype
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            return torch.zeros((input_ids.shape[0], 4))

        def compute_logits(self, hidden_states):
            return torch.tensor([[0.0, 1.0]])

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = lambda **kwargs: SimpleNamespace(
        sampled_token_ids=torch.tensor([[1]], dtype=torch.int32)
    )
    proposer.sparse_attention.begin_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(1)
    )
    proposer.input_ids[0] = 1
    proposer.positions[0] = 3
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    initialize_single_pipeline_stage(proposer)
    proposer._run_draft_step(
        batch_size=1,
        draft_index=0,
        common_attn_metadata=common_attn_metadata,
        active_mask=torch.tensor([True]),
        sampling_metadata=make_sampling_metadata(all_greedy=True),
    )

    proposer.sparse_attention.begin_step.assert_called_once()
    proposer.sparse_attention.end_step_statistics.assert_called_once_with()


def test_model_step_sanitizes_input_ids_for_inactive_rows(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 4], dtype=torch.int64),
    )

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [7, 0]
            return torch.zeros((input_ids.shape[0], 4))

        def compute_logits(self, hidden_states):
            return torch.zeros((hidden_states.shape[0], 2))

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = lambda **kwargs: SimpleNamespace(
        sampled_token_ids=torch.ones(2, 1, dtype=torch.int32)
    )
    proposer.sparse_attention.begin_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(2)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    initialize_single_pipeline_stage(proposer)
    proposer._run_model_step(
        batch_size=2,
        step_index=1,
        input_ids=torch.tensor([7, -1], dtype=torch.int32),
        positions=torch.tensor([3, 3], dtype=torch.int64),
        active_mask=torch.tensor([True, False]),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=RetroSpecAttentionMode.SPARSE_VERIFY,
        compute_margin=False,
    )


def test_model_step_uses_unpadded_dynamic_layout_without_padding_policy_rows(
    monkeypatch,
):
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
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 4], dtype=torch.int64),
    )

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert metadata.num_actual_tokens == 2
            assert metadata.slot_mapping.shape == (2,)
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [7, 0]
            assert positions.tolist() == [3, 0]
            return torch.zeros((2, 4))

        def compute_logits(self, hidden_states):
            assert hidden_states.shape == (2, 4)
            return torch.zeros((2, 2))

    forward_context_kwargs: list[dict[str, Any]] = []

    def fake_forward_context(*args, **kwargs):
        forward_context_kwargs.append(kwargs)
        return nullcontext()

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = lambda **kwargs: SimpleNamespace(
        sampled_token_ids=torch.ones(2, 1, dtype=torch.int32)
    )
    proposer.sparse_attention.begin_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(2)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.set_forward_context",
        fake_forward_context,
    )

    initialize_single_pipeline_stage(proposer)
    proposer._run_model_step(
        batch_size=2,
        step_index=1,
        input_ids=torch.tensor([7, -1], dtype=torch.int32),
        positions=torch.tensor([3, 3], dtype=torch.int64),
        active_mask=torch.tensor([True, False]),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=RetroSpecAttentionMode.DRAFT,
        compute_margin=False,
    )

    context = forward_context_kwargs[0]
    assert context["num_tokens"] == 2
    assert context["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    assert context["batch_descriptor"] is None
    slot_mapping = context["slot_mapping"]["model.layers.0.self_attn.attn"]
    assert slot_mapping.tolist() == [3, -1]
    active_mask = proposer.sparse_attention.begin_step.call_args.args[2]
    assert active_mask.tolist() == [True, False]


def test_propose_stops_requests_independently_on_hit_attention(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_hit_attn_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    expected_masks = [
        [True, True],
        [True, False],
    ]
    hit_attn_values = [
        [0.8, 0.4],
        [0.3, 1.0],
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        assert active_mask.tolist() == expected_masks[draft_index]
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.tensor(hit_attn_values[draft_index]),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1]),
    )

    assert result == [[1, 2], [1]]


def initialize_verification(
    proposer: RetroSpecProposer,
    draft_token_ids: torch.Tensor,
    draft_counts: torch.Tensor,
    pending_counts: torch.Tensor | None = None,
) -> None:
    batch_size = draft_token_ids.shape[0]
    proposer.sparse_attention.maybe_prime_full_verification = Mock(return_value=False)
    proposer.state.begin_batch(batch_size)
    proposer.index_update_state.begin_batch(
        [f"request-{index}" for index in range(batch_size)],
        [1] * batch_size,
    )
    proposer.state.add_draft_counts(draft_counts)
    if pending_counts is not None:
        proposer.state.set_pending_counts(pending_counts)
    proposer._draft_token_ids[:batch_size].copy_(draft_token_ids)
    proposer.proposal_input_ids[:batch_size].copy_(
        torch.arange(7, 7 + batch_size, dtype=torch.int32)
    )
    proposer.proposal_start_positions[:batch_size].fill_(1)


def make_parallel_verification_output(
    request_indices: list[int],
    token_indices: list[int],
    token_ids: list[int],
    margin: list[float] | None = None,
    attention_mass: list[float] | None = None,
) -> RetroSpecParallelVerificationOutput:
    if attention_mass is None:
        attention_mass = [1.0] * len(request_indices)
    return RetroSpecParallelVerificationOutput(
        request_indices=torch.tensor(request_indices, dtype=torch.int64),
        token_indices=torch.tensor(token_indices, dtype=torch.int64),
        token_ids=torch.tensor(token_ids, dtype=torch.int32),
        margin=None if margin is None else torch.tensor(margin),
        attention_mass=torch.tensor(attention_mass),
    )


def test_sparse_bonus_pairs_append_only_with_capacity_and_budget():
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    proposer._sparse_bonus_enabled = True
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1], [11, 21, 31, -1]], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32),
        pending_counts=torch.tensor([0, 2], dtype=torch.int32),
    )
    starts = proposer.state.pending_counts.clone()
    active = proposer.state.active_mask.clone()
    requests, steps = proposer._build_verification_pairs(
        2, starts, proposer.state.draft_counts, active
    )
    combined_requests, combined_steps, bonus_requests = (
        proposer._append_sparse_bonus_pairs(
            2,
            requests,
            steps,
            starts,
            proposer.state.draft_counts,
            active,
            make_sampling_metadata(all_greedy=True),
        )
    )

    assert combined_requests.tolist() == [0, 0, 1, 0]
    assert combined_steps.tolist() == [0, 1, 2, 2]
    assert bonus_requests.tolist() == [0]

    proposer._sparse_bonus_enabled = False
    ordinary_requests, ordinary_steps, no_bonus = proposer._append_sparse_bonus_pairs(
        2,
        requests,
        steps,
        starts,
        proposer.state.draft_counts,
        active,
        make_sampling_metadata(all_greedy=True),
    )
    assert ordinary_requests.shape[0] == 3
    assert ordinary_steps.shape[0] == 3
    assert no_bonus.numel() == 0

    proposer._sparse_bonus_enabled = True
    proposer.max_parallel_tokens = 3
    _, _, no_capacity = proposer._append_sparse_bonus_pairs(
        2,
        requests,
        steps,
        starts,
        proposer.state.draft_counts,
        active,
        make_sampling_metadata(all_greedy=True),
    )
    assert no_capacity.numel() == 0
    proposer.max_parallel_tokens = (
        proposer.max_batch_size * proposer.num_speculative_tokens
    )

    _, _, processed_sampling = proposer._append_sparse_bonus_pairs(
        2,
        requests,
        steps,
        starts,
        proposer.state.draft_counts,
        active,
        replace(make_sampling_metadata(all_greedy=True), no_penalties=False),
    )
    assert processed_sampling.numel() == 0


def test_sparse_bonus_is_discarded_for_request_with_verification_boundary(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    proposer._sparse_bonus_enabled = True
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1], [11, 21, -1, -1]], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
    )
    observed: list[tuple[list[int], list[int], int | None]] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
        bonus_start_index=None,
    ):
        if attention_mode == RetroSpecAttentionMode.EXPANDED_VERIFY:
            return make_parallel_verification_output([1], [1], [99])
        observed.append(
            (request_indices.tolist(), token_indices.tolist(), bonus_start_index)
        )
        return make_parallel_verification_output(
            request_indices.tolist(),
            token_indices.tolist(),
            [10, 20, 11, 99, 30, 31],
        )

    monkeypatch.setattr(
        proposer, "_run_parallel_verification", fake_run_parallel_verification
    )
    verification = proposer._verify_draft_tokens(
        2,
        ["request-0", "request-1"],
        1,
        torch.zeros(2, dtype=torch.int32),
        make_common_metadata([1, 1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed == [([0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 2], 4)]
    assert verification.verified_counts.tolist() == [2, 2]
    assert verification.bonus_mask is not None
    assert verification.bonus_mask.tolist() == [True, False]
    assert verification.bonus_token_ids is not None
    assert verification.bonus_token_ids.tolist() == [30, 31]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_proposal_token_budget_subtracts_target_output_and_clamps(device):
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())

    budgets = proposer._prepare_proposal_token_budgets(
        [0, 2, 9],
        torch.tensor([1, 1, 2], dtype=torch.int32, device=device),
    )

    assert budgets.data_ptr() == proposer._proposal_token_budgets.gpu.data_ptr()
    assert budgets.tolist() == [0, 1, 4]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_verification_pair_compaction_stays_on_device(device):
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    round_starts = torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=device)
    draft_counts = torch.tensor([3, 2, 1, 4], dtype=torch.int32, device=device)
    active = torch.tensor([True, False, True, True], device=device)

    request_indices, token_indices = proposer._build_verification_pairs(
        4, round_starts, draft_counts, active
    )

    assert request_indices.device.type == device.type
    assert token_indices.device.type == device.type
    assert (
        request_indices.data_ptr() == proposer._verification_request_indices.data_ptr()
    )
    assert token_indices.data_ptr() == proposer._verification_token_indices.data_ptr()
    assert request_indices.tolist() == [0, 0, 0, 2, 3, 3, 3, 3]
    assert token_indices.tolist() == [0, 1, 2, 2, 0, 1, 2, 3]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_verification_compaction_reuses_fixed_output(device):
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    output = torch.empty(8, dtype=torch.int64, device=device)

    selected = proposer._compact_mask_indices(
        torch.tensor(
            [True, False, True, True, False, False, True, False],
            device=device,
        ),
        output,
    )

    assert selected.device.type == device.type
    assert selected.data_ptr() == output.data_ptr()
    assert selected.tolist() == [0, 2, 3, 6]

    empty = proposer._compact_mask_indices(
        torch.zeros(8, dtype=torch.bool, device=device),
        output,
    )
    assert empty.numel() == 0


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_dynamic_draft_control_skips_empty_columns_and_reuses_workspace(device):
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    proposer.state.begin_batch(
        4,
        torch.tensor([True, True, True, False], device=device),
    )
    proposer.state.set_pending_counts(
        torch.tensor([2, 0, 1, 0], dtype=torch.int32, device=device)
    )
    proposer.state.add_draft_counts(
        torch.tensor([4, 4, 4, 0], dtype=torch.int32, device=device)
    )
    proposer.positions[:4].copy_(
        torch.tensor([1, 1, 15, 1], dtype=torch.int64, device=device)
    )

    round_mask, round_starts = proposer._begin_draft_round(4)

    assert round_mask.data_ptr() == proposer._draft_round_mask.data_ptr()
    assert round_starts.data_ptr() == proposer._draft_round_start_counts.data_ptr()
    assert round_mask.tolist() == [True, True, False, False]
    assert round_starts.tolist() == [2, 0, 1, 0]
    assert proposer.state.draft_counts.tolist() == [0, 0, 0, 0]

    token_index, step_mask = proposer._prepare_next_draft_step(
        4, round_mask, round_starts
    )
    assert token_index == 0
    assert step_mask.data_ptr() == proposer._draft_stage_mask.data_ptr()
    assert step_mask.tolist() == [False, True, False, False]

    proposer.state.add_draft_counts(step_mask.to(torch.int32))
    proposer.state.set_stage(step_mask, RetroSpecStage.FULL_VERIFY)
    token_index, step_mask = proposer._prepare_next_draft_step(
        4, round_mask, round_starts
    )
    assert token_index == 2
    assert step_mask.tolist() == [True, False, False, False]

    proposer.state.set_stage(step_mask, RetroSpecStage.FULL_VERIFY)
    token_index, step_mask = proposer._prepare_next_draft_step(
        4, round_mask, round_starts
    )
    assert token_index == proposer.num_speculative_tokens
    assert not step_mask.any()


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_parallel_sampling_uses_one_raw_argmax_for_plain_greedy(device):
    sampler = Mock()
    proposer = RetroSpecProposer(
        make_vllm_config(), device, make_runner(sampler=sampler)
    )
    logits = torch.tensor(
        [[4.0, 4.0, 1.0], [3.0, 2.0, 1.0], [0.0, 1.0, 5.0]], device=device
    )
    metadata = make_sampling_metadata(all_greedy=True)
    reference = Sampler()(
        logits=logits.clone(), sampling_metadata=metadata
    ).sampled_token_ids.view(-1)

    sampled = proposer._sample_parallel_logits(
        batch_size=2,
        logits=logits,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64, device=device),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64, device=device),
        sampling_metadata=metadata,
        output=proposer._sparse_sampled_token_ids,
    )

    torch.testing.assert_close(sampled, reference)
    assert sampled.data_ptr() == proposer._sparse_sampled_token_ids.data_ptr()
    assert proposer._verification_step_logits is None
    sampler.assert_not_called()


def test_parallel_sampling_builds_pair_metadata_once_for_indexable_constraints(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm.v1.sample.ops.penalties.is_pin_memory_available", lambda: False
    )
    captured_metadata: list[SamplingMetadata] = []
    sampler = Sampler()
    sampler.pin_memory = False

    def sample(*, logits, sampling_metadata):
        captured_metadata.append(sampling_metadata)
        return sampler(logits=logits, sampling_metadata=sampling_metadata)

    proposer = RetroSpecProposer(
        make_vllm_config(), torch.device("cpu"), make_runner(sampler=sample)
    )
    allowed_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [True, False, False, False],
        ]
    )
    metadata = replace(
        make_sampling_metadata(all_greedy=True),
        no_penalties=False,
        prompt_token_ids=torch.tensor([[1, 2], [3, 0], [1, 3]]),
        frequency_penalties=torch.tensor([0.1, 0.2, 0.3]),
        presence_penalties=torch.tensor([0.4, 0.5, 0.6]),
        repetition_penalties=torch.tensor([1.1, 1.2, 1.3]),
        output_token_ids=[[1], [2], [3]],
        allowed_token_ids_mask=allowed_mask,
        bad_words_token_ids={0: [[1]], 2: [[2]]},
    )
    request_indices = torch.tensor([2, 0, 2], dtype=torch.int64)
    token_indices = torch.tensor([0, 1, 1], dtype=torch.int64)
    logits = torch.tensor(
        [[0.0, 1.0, 3.0, 2.0], [4.0, 1.0, 0.0, 2.0], [0.0, 5.0, 1.0, 2.0]]
    )
    reference_sampler = Sampler()
    reference_sampler.pin_memory = False
    reference_proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(sampler=reference_sampler),
    )
    reference_output = reference_proposer._sparse_sampled_token_ids[: logits.shape[0]]
    sampler_calls = reference_proposer._sample_parallel_logits_by_position(
        batch_size=3,
        logits=logits.clone(),
        request_indices=request_indices,
        token_indices=token_indices,
        sampling_metadata=metadata,
        sampled_token_ids=reference_output,
    )

    sampled = proposer._sample_parallel_logits(
        batch_size=3,
        logits=logits.clone(),
        request_indices=request_indices,
        token_indices=token_indices,
        sampling_metadata=metadata,
        output=proposer._sparse_sampled_token_ids,
    )

    assert sampler_calls == 2
    torch.testing.assert_close(sampled, reference_output)
    assert len(captured_metadata) == 1
    pair_metadata = captured_metadata[0]
    assert pair_metadata.max_num_logprobs is None
    assert pair_metadata.output_token_ids == [[3], [1], [3]]
    assert pair_metadata.bad_words_token_ids == {
        0: [[2]],
        1: [[1]],
        2: [[2]],
    }
    assert pair_metadata.prompt_token_ids.tolist() == [[1, 3], [1, 2], [1, 3]]
    assert pair_metadata.frequency_penalties.tolist() == pytest.approx([0.3, 0.1, 0.3])
    assert pair_metadata.presence_penalties.tolist() == pytest.approx([0.6, 0.4, 0.6])
    assert pair_metadata.repetition_penalties.tolist() == pytest.approx([1.3, 1.1, 1.3])
    assert torch.equal(pair_metadata.allowed_token_ids_mask, allowed_mask[[2, 0, 2]])


def test_parallel_sampling_retains_position_fallback_for_indexed_processors():
    sampling_calls: list[torch.Tensor] = []

    def sample(*, logits, sampling_metadata):
        sampling_calls.append(logits.clone())
        return SimpleNamespace(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    processors = LogitsProcessors()
    processors.non_argmax_invariant.append(Mock())
    metadata = replace(make_sampling_metadata(all_greedy=True), logitsprocs=processors)
    proposer = RetroSpecProposer(
        make_vllm_config(), torch.device("cpu"), make_runner(sampler=sample)
    )
    logits = torch.tensor([[0.0, 4.0, 1.0], [3.0, 2.0, 1.0], [0.0, 1.0, 5.0]])

    sampled = proposer._sample_parallel_logits(
        batch_size=2,
        logits=logits,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        sampling_metadata=metadata,
        output=proposer._sparse_sampled_token_ids,
    )

    assert sampled.tolist() == [1, 0, 2]
    assert len(sampling_calls) == 2
    assert all(call.shape == (2, 3) for call in sampling_calls)
    assert proposer._verification_step_logits is not None


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_first_verification_boundary_reduces_by_request(device):
    proposer = RetroSpecProposer(make_vllm_config(), device, make_runner())
    request_indices = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64, device=device)
    boundary_mask = torch.tensor(
        [False, True, False, False, True, False], device=device
    )

    first = proposer._find_first_boundary_indices(4, request_indices, boundary_mask)

    assert first.device.type == device.type
    assert first.data_ptr() == proposer._verification_first_boundaries.data_ptr()
    assert first.tolist() == [1, 4, 6, 6]


def test_propose_stops_draft_at_index_update_boundary(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=4),
        torch.device("cpu"),
        make_runner(),
    )
    observed_indices: list[int] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        observed_indices.append(draft_index)
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1]),
        committed_positions=[1],
    )

    assert observed_indices == [0, 1, 2, 3]
    assert result == [[1, 2, 3, 4]]


def test_sparse_verification_requires_full_at_index_update_boundary(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=4),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, 40]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
    )
    observed_rows: list[tuple[list[int], list[int]]] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        assert attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
        observed_rows.append((list(request_indices), list(token_indices)))
        return make_parallel_verification_output(
            list(request_indices), list(token_indices), [10, 20, 30, 40]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    flush_prefetch = Mock()
    prime_full_verification = Mock(return_value=False)
    monkeypatch.setattr(
        proposer.sparse_attention,
        "flush_sparse_verification_prefetch",
        flush_prefetch,
    )
    monkeypatch.setattr(
        proposer.sparse_attention,
        "maybe_prime_full_verification",
        prime_full_verification,
    )

    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed_rows == [([0, 0, 0, 0], [0, 1, 2, 3])]
    flush_prefetch.assert_called_once_with()
    prime_full_verification.assert_called_once_with(4)
    assert verification.verified_counts.tolist() == [4]
    assert verification.require_full.tolist() == [True]


def test_sparse_full_trigger_skips_expanded_verification(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.max_model_len = 2
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    observed_modes: list[RetroSpecAttentionMode] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        observed_modes.append(attention_mode)
        return make_parallel_verification_output(
            list(request_indices),
            list(token_indices),
            [10, 20],
            margin=[0.1, 0.1],
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed_modes == [RetroSpecAttentionMode.SPARSE_VERIFY]
    assert verification.verified_counts.tolist() == [1]
    assert verification.require_full.tolist() == [True]


def test_remove_requests_resets_proposer_index_update_boundary():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=4),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.index_update_state.begin_batch(["request"], [10])

    proposer.remove_requests({"request"})
    proposer.index_update_state.begin_batch(["request"], [100])

    assert proposer.index_update_state.next_update_positions.tolist() == [104]


def test_finished_request_records_terminal_proposal_outcome_and_flushes():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._seen_proposal_request_ids.update(("finished", "running"))
    proposer._last_proposed_counts.update({"finished": 4, "running": 3})
    proposer.performance_stats.flush = Mock()

    proposer.record_previous_proposal_outcomes(
        ("finished", "running"),
        (3, 4),
        {"finished"},
    )

    counters = proposer.performance_stats._cpu_counters
    assert counters["terminal_proposal_tokens"] == 4
    assert counters["terminal_committed_proposal_tokens"] == 2
    assert counters["terminal_wasted_proposal_tokens"] == 2
    assert proposer._last_proposed_counts == {"running": 3}
    assert proposer._last_committed_proposal_counts == {"running": 3}
    assert proposer._seen_proposal_request_ids == {"running"}
    proposer.performance_stats.flush.assert_called_once_with("request_finished")


def test_preempted_request_preserves_first_proposal_tracking():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._seen_proposal_request_ids.add("preempted")
    proposer._last_proposed_counts["preempted"] = 4
    proposer.performance_stats.flush = Mock()

    proposer.record_previous_proposal_outcomes(("preempted",), (2,), set())

    assert proposer._last_proposed_counts == {"preempted": 4}
    assert proposer._seen_proposal_request_ids == {"preempted"}
    proposer.performance_stats.flush.assert_not_called()


def test_finished_request_without_sample_count_clears_request_id_lifecycle():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._seen_proposal_request_ids.add("reused")
    proposer._last_proposed_counts["reused"] = 4
    proposer.performance_stats.flush = Mock()

    proposer.record_previous_proposal_outcomes((), (), {"reused"})

    assert "reused" not in proposer._last_proposed_counts
    assert "reused" not in proposer._seen_proposal_request_ids
    proposer.performance_stats.flush.assert_called_once_with("request_finished")


def test_sync_bookkeeping_records_committed_proposal_before_finish():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._last_proposed_counts.update({"first": 4, "second": 3})

    proposer.record_sampled_proposal_outcomes(("first", "second"), (3, 1))

    assert proposer._last_committed_proposal_counts == {"first": 2, "second": 0}


def test_verified_proposal_outcomes_use_consumed_lengths_not_next_proposal():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._last_proposed_counts.update({"first": 64, "second": 64})

    proposer.record_verified_proposal_outcomes((4, 8, 0), (3, 9, 1))

    counters = proposer.performance_stats._cpu_counters
    assert counters["proposal_verified_rounds"] == 2
    assert counters["proposal_verified_tokens"] == 12
    assert counters["proposal_accepted_tokens"] == 10
    assert counters["proposal_rejected_tokens"] == 2
    assert counters["proposal_fully_accepted"] == 1


def test_verified_proposal_outcomes_reject_mismatched_rows():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_stats_interval_seconds=60.0),
        torch.device("cpu"),
        make_runner(),
    )

    with pytest.raises(ValueError, match="equal length"):
        proposer.record_verified_proposal_outcomes((4,), (3, 2))

    with pytest.raises(ValueError, match="equal length"):
        proposer.record_verified_proposal_outcomes((4,), (3,), ("first", "second"))


def test_full_verify_feedback_limits_only_low_acceptance_request():
    proposer = RetroSpecProposer(
        make_vllm_config(
            max_model_len=128,
            num_speculative_tokens=64,
            retrospec_max_draft_tokens=8,
            retrospec_stats_interval_seconds=0.0,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._feedback_horizon_enabled = True
    proposer.record_verified_proposal_outcomes((64, 64), (15, 33), ("low", "high"))

    budgets = proposer._prepare_proposal_token_budgets(
        [100, 100], torch.ones(2, dtype=torch.int32), ("low", "high")
    )

    assert budgets.tolist() == [16, 64]
    assert proposer._feedback_horizons == {"low": 16}
    proposer.remove_requests({"low"})
    assert "low" not in proposer._feedback_horizons


def test_full_verify_feedback_recovers_after_three_fully_accepted_rounds():
    proposer = RetroSpecProposer(
        make_vllm_config(
            max_model_len=128,
            num_speculative_tokens=64,
            retrospec_max_draft_tokens=8,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    proposer._feedback_horizon_enabled = True
    proposer.record_verified_proposal_outcomes((64,), (15,), ("request",))
    proposer.record_verified_proposal_outcomes((16,), (17,), ("request",))
    proposer.record_verified_proposal_outcomes((16,), (9,), ("request",))
    assert proposer._feedback_recovery["request"] == 0

    for _ in range(3):
        proposer.record_verified_proposal_outcomes((16,), (17,), ("request",))

    assert "request" not in proposer._feedback_horizons
    budgets = proposer._prepare_proposal_token_budgets(
        [100], torch.ones(1, dtype=torch.int32), ("request",)
    )
    assert budgets.tolist() == [64]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "tp_size,pp_size,host_enabled", [(1, 1, True), (2, 1, False), (1, 2, False)]
)
def test_full_verify_feedback_uses_device_state_for_multiple_ranks(
    tp_size, pp_size, host_enabled
):
    config = make_vllm_config(
        max_model_len=128,
        num_speculative_tokens=64,
        retrospec_max_draft_tokens=8,
    )
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = pp_size
    proposer = RetroSpecProposer(config, torch.device("cuda"), make_runner())

    assert proposer._feedback_horizon_enabled is host_enabled
    assert proposer._feedback_device_enabled is not host_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_device_feedback_tracks_reordered_requests_and_clears_reused_slot():
    config = make_vllm_config(
        max_model_len=128, num_speculative_tokens=64, retrospec_max_draft_tokens=8
    )
    config.parallel_config.tensor_parallel_size = 2
    device = torch.device("cuda", torch.cuda.current_device())
    proposer = RetroSpecProposer(config, device, make_runner())
    sampled = torch.ones(2, dtype=torch.int32, device=device)

    initial = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("low", "high")
    )
    assert initial.tolist() == [64, 64]
    proposer._update_device_feedback_horizon(
        ("low", "high"),
        (64, 64),
        torch.tensor([15, 33], dtype=torch.int32, device=device),
    )
    reordered = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("high", "low")
    )
    assert reordered.tolist() == [64, 16]

    for _ in range(3):
        proposer._update_device_feedback_horizon(
            ("high", "low"),
            (0, 16),
            torch.tensor([1, 17], dtype=torch.int32, device=device),
        )
    recovered = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("high", "low")
    )
    assert recovered.tolist() == [64, 64]

    proposer._update_device_feedback_horizon(
        ("high", "low"),
        (0, 64),
        torch.tensor([1, 15], dtype=torch.int32, device=device),
    )
    proposer.remove_requests({"low"})
    reused = proposer._prepare_proposal_token_budgets(
        [100, 100], sampled, ("high", "new")
    )
    assert reused.tolist() == [64, 64]


def test_verify_unchanged_sparse_tokens_keeps_complete_prefix(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    observed_rows: list[tuple[list[int], list[int]]] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        assert attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
        observed_rows.append((list(request_indices), list(token_indices)))
        return make_parallel_verification_output(
            list(request_indices), list(token_indices), [10, 20, 30]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [3]
    assert not verification.require_full.any()
    assert proposer._draft_token_ids[0, :3].tolist() == [10, 20, 30]
    assert observed_rows == [([0, 0, 0], [0, 1, 2])]
    assert proposer.state.stage.tolist() == [int(RetroSpecStage.DRAFT)]


def test_sparse_token_change_is_corrected_and_truncates_prefix(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    observed_modes: list[RetroSpecAttentionMode] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        observed_modes.append(attention_mode)
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            token_ids = [11, 99, 98]
        else:
            token_ids = [11]
        return make_parallel_verification_output(
            list(request_indices), list(token_indices), token_ids
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert not verification.require_full.any()
    assert proposer._draft_token_ids[0, :3].tolist() == [11, 20, 30]
    assert observed_modes == [
        RetroSpecAttentionMode.SPARSE_VERIFY,
        RetroSpecAttentionMode.EXPANDED_VERIFY,
    ]


def test_expanded_verification_preserves_sparse_boundary_across_shared_workspace(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, -1, -1, -1]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        shared_token_ids = proposer.pipeline_protocol._model_token_ids[:1]
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            shared_token_ids.fill_(10)
            margin = [0.1]
        else:
            shared_token_ids.fill_(99)
            margin = None
        return RetroSpecParallelVerificationOutput(
            request_indices=request_indices,
            token_indices=token_indices,
            token_ids=shared_token_ids,
            margin=None if margin is None else torch.tensor(margin),
            attention_mass=torch.ones(1),
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )

    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert verification.require_full.tolist() == [True]
    assert proposer._draft_token_ids[0, 0].item() == 99


def test_expanded_verification_passes_or_stops_requests_independently(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_sparse_margin_threshold=0.5,
            retrospec_expanded_margin_threshold=0.5,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1], [11, 21, 31, -1]], dtype=torch.int32),
        torch.tensor([3, 3], dtype=torch.int32),
    )
    observed_rows: list[tuple[RetroSpecAttentionMode, list[int], list[int]]] = []
    compaction_lengths: list[int] = []
    compact_mask_indices = proposer._compact_mask_indices

    def track_compaction(mask, output):
        compaction_lengths.append(mask.shape[0])
        return compact_mask_indices(mask, output)

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        request_indices = list(request_indices)
        token_indices = list(token_indices)
        observed_rows.append((attention_mode, request_indices, token_indices))
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return make_parallel_verification_output(
                request_indices,
                token_indices,
                [10, 20, 30, 11, 21, 31],
                margin=[0.1] * 6,
            )

        return make_parallel_verification_output(
            request_indices, token_indices, [10, 11], margin=[0.9, 0.1]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    monkeypatch.setattr(proposer, "_compact_mask_indices", track_compaction)
    verification = proposer._verify_draft_tokens(
        2,
        ["request-0", "request-1"],
        1,
        torch.zeros(2, dtype=torch.int32),
        make_common_metadata([1, 1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1, 1]
    assert verification.require_full.tolist() == [False, True]
    assert observed_rows == [
        (RetroSpecAttentionMode.SPARSE_VERIFY, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]),
        (RetroSpecAttentionMode.EXPANDED_VERIFY, [0, 1], [0, 0]),
    ]
    assert compaction_lengths == [8, 2]
    assert proposer.state.stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.FULL_VERIFY),
    ]


def test_verification_trace_records_request_boundaries(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_sparse_margin_threshold=0.5,
            retrospec_expanded_margin_threshold=0.5,
            retrospec_trace_transitions=True,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1], [11, 21, -1, -1]], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
    )

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return make_parallel_verification_output(
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [10, 20, 11, 21],
                margin=[0.9, 0.9, 0.1, 0.9],
            )
        return make_parallel_verification_output([1], [0], [11], margin=[0.1])

    logger_info = Mock()
    monkeypatch.setattr(
        proposer, "_run_parallel_verification", fake_run_parallel_verification
    )
    monkeypatch.setattr(transition_trace.logger, "info", logger_info)

    verification = proposer._verify_draft_tokens(
        2,
        ["request-0", "request-1"],
        3,
        torch.zeros(2, dtype=torch.int32),
        make_common_metadata([1, 1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [2, 1]
    payloads = [json.loads(call.args[1]) for call in logger_info.call_args_list]
    assert [payload["phase"] for payload in payloads] == [
        "sparse_boundary",
        "sparse_complete",
        "expanded_boundary",
    ]
    assert all(payload["proposal_round"] == 3 for payload in payloads)
    assert payloads[0]["records"][0]["request_id"] == "request-1"
    assert payloads[0]["records"][0]["reason_names"] == ["SPARSE_MARGIN"]
    assert payloads[1]["records"][0]["request_id"] == "request-0"
    assert payloads[1]["records"][0]["next_stage_name"] == "DRAFT"
    assert payloads[2]["records"][0]["request_id"] == "request-1"
    assert payloads[2]["records"][0]["reason_names"] == ["EXPANDED_MARGIN"]


def test_expanded_token_change_replaces_current_token_before_truncation(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return make_parallel_verification_output(
                [0, 0], [0, 1], [10, 20], margin=[0.1, 0.9]
            )
        return make_parallel_verification_output([0], [0], [99])

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert verification.require_full.tolist() == [True]
    assert proposer._draft_token_ids[0, 0].item() == 99


def test_verify_only_processes_current_logical_draft_interval(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, 40]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        pending_counts=torch.tensor([2], dtype=torch.int32),
    )
    observed_rows: list[tuple[list[int], list[int]]] = []

    def fake_run_parallel_verification(
        batch_size,
        request_indices,
        token_indices,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        assert attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
        observed_rows.append((list(request_indices), list(token_indices)))
        return make_parallel_verification_output(
            list(request_indices), list(token_indices), [30, 40]
        )

    monkeypatch.setattr(
        proposer,
        "_run_parallel_verification",
        fake_run_parallel_verification,
    )
    verification = proposer._verify_draft_tokens(
        1,
        ["request-0"],
        1,
        torch.tensor([2], dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed_rows == [([0, 0], [2, 3])]
    assert verification.verified_counts.tolist() == [2]
    assert verification.require_full.tolist() == [True]


def test_sparse_bonus_seeds_next_round_without_committing_early(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=2),
        torch.device("cpu"),
        make_runner(),
    )
    draft_calls: list[tuple[int, int, int]] = []
    verified_rounds: list[tuple[int, int, list[int]]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        draft_calls.append(
            (
                draft_index,
                int(proposer.input_ids[0]),
                int(proposer.positions[0]),
            )
        )
        return (
            torch.tensor([draft_index + 1], dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    def fake_verify(
        batch_size,
        request_ids,
        proposal_round,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        verified_rounds.append(
            (
                int(round_start_counts[0]),
                int(proposer.state.draft_counts[0]),
                proposer._draft_token_ids[0].tolist(),
            )
        )
        if proposal_round == 1:
            return RetroSpecVerificationResult(
                verified_counts=torch.tensor([2], dtype=torch.int32),
                require_full=torch.tensor([False]),
                bonus_mask=torch.tensor([True]),
                bonus_token_ids=torch.tensor([99], dtype=torch.int32),
            )
        return RetroSpecVerificationResult(
            verified_counts=torch.tensor([2], dtype=torch.int32),
            require_full=torch.tensor([True]),
        )

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2]),
    )

    assert draft_calls == [(0, 7, 2), (1, 1, 3), (3, 99, 5)]
    assert verified_rounds == [(0, 2, [1, 2, -1, -1]), (2, 2, [1, 2, 99, 4])]
    assert result == [[1, 2, 99, 4]]


def test_propose_accumulates_multiple_draft_rounds(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_max_draft_tokens=2,
            retrospec_stats_interval_seconds=3600.0,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    round_starts: list[list[int]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    def fake_verify(
        batch_size,
        request_ids,
        proposal_round,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        assert request_ids == ["request-0"]
        assert proposal_round == len(round_starts) + 1
        round_starts.append(round_start_counts.tolist())
        verified_counts = proposer.state.draft_counts.clone()
        require_full = (
            round_start_counts + verified_counts >= proposer.policy.pending_limit
        )
        return RetroSpecVerificationResult(verified_counts, require_full)

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2]),
    )

    assert round_starts == [[0], [2]]
    assert result == [[1, 2, 3, 4]]
    assert proposer.state.pending_counts.tolist() == [4]
    assert proposer.state.stage.tolist() == [int(RetroSpecStage.FULL_VERIFY)]

    stats = proposer.performance_stats
    gpu_counters = {
        name: int(stats._gpu_counters[index].item())
        for name, index in stats._gpu_counter_indices.items()
    }
    assert stats._cpu_counters["proposal_calls"] == 1
    assert gpu_counters == {
        "proposal_requests": 1,
        "draft_round_requests": 2,
        "draft_tokens": 4,
        "sparse_bonus_admitted": 0,
        "feedback_horizon_reductions": 0,
        "feedback_horizon_restores": 0,
        "verified_tokens": 4,
        "proposed_tokens": 4,
        "resident_cluster_hits": 0,
        "resident_cluster_misses": 0,
        "draft_compact_resident_pages": 0,
        "draft_compact_selected_clusters": 0,
        "resident_bound_direct_hits": 0,
        "resident_hash_fallback_lookups": 0,
        "resident_hash_fallback_hits": 0,
        "resident_hash_fallback_misses": 0,
        "resident_hash_probe_steps": 0,
        "resident_hash_max_probe": 0,
        "resident_binding_invalidations": 0,
        "verification_lookup_clusters": 0,
        "verification_resident_hits": 0,
        "verification_resident_misses": 0,
    }
    assert stats._cpu_times["proposal_wall"][1] == 1


def test_propose_handles_different_round_offsets_in_one_buffer(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=2),
        torch.device("cpu"),
        make_runner(),
    )
    round_starts: list[list[int]] = []
    draft_calls: list[tuple[int, list[bool], list[int]]] = []
    verified_by_round = [
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        draft_calls.append(
            (
                draft_index,
                active_mask.tolist(),
                proposer.positions[:batch_size].tolist(),
            )
        )
        return (
            torch.tensor(
                [draft_index * 10 + row for row in range(batch_size)],
                dtype=torch.int32,
            ),
            None,
            torch.ones(batch_size),
        )

    def fake_verify(
        batch_size,
        request_ids,
        proposal_round,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        assert request_ids == ["request-0", "request-1"]
        assert proposal_round == len(round_starts) + 1
        round_index = len(round_starts)
        round_starts.append(round_start_counts.tolist())
        verified_counts = verified_by_round[round_index]
        require_full = (
            round_start_counts + verified_counts >= proposer.policy.pending_limit
        )
        return RetroSpecVerificationResult(verified_counts, require_full)

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7, 8], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2, 4]),
    )

    assert round_starts == [[0, 0], [1, 2], [3, 4]]
    assert [(index, mask) for index, mask, _ in draft_calls] == [
        (0, [True, True]),
        (1, [True, True]),
        (1, [True, False]),
        (2, [True, True]),
        (3, [False, True]),
        (3, [True, False]),
    ]
    assert draft_calls[2][2] == [3, 6]
    assert draft_calls[-1][2] == [5, 8]
    assert result == [[0, 10, 20, 30], [1, 11, 21, 31]]
    assert proposer.state.pending_counts.tolist() == [4, 4]


@pytest.mark.parametrize(
    ("attention_mode", "expect_margin"),
    [
        (RetroSpecAttentionMode.SPARSE_VERIFY, True),
        (RetroSpecAttentionMode.EXPANDED_VERIFY, False),
    ],
)
def test_parallel_verification_flattens_tokens_and_preserves_sampling_rows(
    monkeypatch,
    attention_mode,
    expect_margin,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=6,
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 13], dtype=torch.int64),
    )
    proposer.proposal_start_positions[:2].copy_(torch.tensor([3, 5]))
    proposer.proposal_input_ids[:2].copy_(torch.tensor([7, 8]))
    proposer._draft_token_ids[:2, :2].copy_(torch.tensor([[10, 20], [11, 21]]))

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert draft_index == 0
            assert metadata.num_reqs == 3
            assert metadata.query_start_loc.tolist() == [0, 1, 2, 3]
            assert metadata.seq_lens.tolist() == [6, 5, 7]
            assert metadata.block_table_tensor.tolist() == [[2, 3], [0, 1], [2, 3]]
            assert metadata.slot_mapping.tolist() == [13, 4, 14]
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [8, 10, 11]
            assert positions.tolist() == [5, 4, 6]
            return torch.zeros((3, 4))

        def compute_logits(self, hidden_states):
            return torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0], [0.0, 1.0, 4.0]])

    sampling_calls: list[torch.Tensor] = []

    def sample(*, logits, sampling_metadata):
        sampling_calls.append(logits.clone())
        return SimpleNamespace(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = sample
    proposer.sparse_attention.begin_parallel_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(3)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    initialize_single_pipeline_stage(proposer)
    result = proposer._run_parallel_verification(
        batch_size=2,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=attention_mode,
    )

    assert result.request_indices.tolist() == [1, 0, 1]
    assert result.token_indices.tolist() == [0, 1, 1]
    assert result.token_ids.tolist() == [1, 0, 2]
    assert (
        result.token_ids.data_ptr()
        == proposer.pipeline_protocol._model_token_ids.data_ptr()
    )
    assert (result.margin is not None) is expect_margin
    if result.margin is not None:
        assert result.margin.tolist() == [1.0, 2.0, 3.0]
    assert len(sampling_calls) == 0
    proposer.sparse_attention.begin_parallel_step.assert_called_once()
    begin_args = proposer.sparse_attention.begin_parallel_step.call_args.args
    assert begin_args[0] == attention_mode
    assert torch.equal(begin_args[1], torch.tensor([1, 0, 1], dtype=torch.int64))
    assert torch.equal(begin_args[2], torch.tensor([0, 1, 1], dtype=torch.int64))
    proposer.sparse_attention.end_step_statistics.assert_called_once_with()


def test_parallel_verification_uses_unpadded_dynamic_layout(monkeypatch):
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
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=6,
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 13], dtype=torch.int64),
    )
    proposer.proposal_start_positions[:2].copy_(torch.tensor([3, 5]))
    proposer.proposal_input_ids[:2].copy_(torch.tensor([7, 8]))
    proposer._draft_token_ids[:2, :2].copy_(torch.tensor([[10, 20], [11, 21]]))

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert metadata.num_actual_tokens == 3
            assert metadata.slot_mapping.tolist() == [13, 4, 14]
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert intermediate_tensors is None
            assert input_ids.tolist() == [8, 10, 11]
            assert positions.tolist() == [5, 4, 6]
            return torch.zeros((3, 4))

        def compute_logits(self, hidden_states):
            assert hidden_states.shape == (3, 4)
            return torch.tensor([[0.0, 2.0], [3.0, 1.0], [0.0, 4.0]])

    forward_context_kwargs: list[dict[str, Any]] = []

    def fake_forward_context(*args, **kwargs):
        forward_context_kwargs.append(kwargs)
        return nullcontext()

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.sparse_attention.begin_parallel_step = Mock()
    proposer.sparse_attention.end_step_statistics = Mock(
        return_value=attention_stats(3)
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.set_forward_context",
        fake_forward_context,
    )

    initialize_single_pipeline_stage(proposer)
    result = proposer._run_parallel_verification(
        batch_size=2,
        request_indices=torch.tensor([1, 0, 1], dtype=torch.int64),
        token_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=RetroSpecAttentionMode.EXPANDED_VERIFY,
    )

    assert result.request_indices.tolist() == [1, 0, 1]
    assert result.token_indices.tolist() == [0, 1, 1]
    assert result.token_ids.tolist() == [1, 0, 1]
    context = forward_context_kwargs[0]
    assert context["num_tokens"] == 3
    assert context["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    assert context["batch_descriptor"] is None
    slot_mapping = context["slot_mapping"]["model.layers.0.self_attn.attn"]
    assert slot_mapping.tolist() == [13, 4, 14]
    begin_args = proposer.sparse_attention.begin_parallel_step.call_args.args
    assert torch.equal(begin_args[1], torch.tensor([1, 0, 1], dtype=torch.int64))
    assert torch.equal(begin_args[2], torch.tensor([0, 1, 1], dtype=torch.int64))
