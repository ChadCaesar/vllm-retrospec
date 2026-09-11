# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.ray_executor import RayDistributedExecutor
from vllm.v1.spec_decode.retrospec.pipeline import (
    RetroSpecAttentionMassStats,
    RetroSpecPipelineControlState,
    RetroSpecPipelineModelOutput,
    RetroSpecPipelineProtocol,
    RetroSpecPipelineStage,
)

pytestmark = pytest.mark.cpu_test


class FakeLayerModel:
    def __init__(self, start_layer: int, end_layer: int) -> None:
        self.start_layer = start_layer
        self.end_layer = end_layer

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids[:, None].expand(-1, 3).to(torch.float32)


def make_pp_group(
    *,
    rank: int = 0,
    world_size: int = 1,
    is_last_rank: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        is_last_rank=is_last_rank,
        last_rank=world_size - 1,
        device_group=object(),
        all_reduce=Mock(side_effect=lambda tensor: tensor),
        recv_into=Mock(),
        send=Mock(),
    )


def make_protocol(
    *,
    max_batch_size: int = 4,
    max_parallel_tokens: int = 16,
    max_sampled_tokens: int = 5,
    tensor_parallel_size: int = 1,
    enable_sp: bool = False,
) -> RetroSpecPipelineProtocol:
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp),
            splitting_ops=[],
            use_inductor_graph_partition=False,
            compile_sizes=None,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            get_hidden_size=Mock(return_value=4),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tensor_parallel_size),
    )
    return RetroSpecPipelineProtocol(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        max_batch_size=max_batch_size,
        max_parallel_tokens=max_parallel_tokens,
        max_sampled_tokens=max_sampled_tokens,
    )


def make_executor_concurrency_view(executor_cls, method: str | None):
    executor = object.__new__(executor_cls)
    executor.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    executor.scheduler_config = SimpleNamespace(async_scheduling=False)
    executor.vllm_config = SimpleNamespace(
        speculative_config=None if method is None else SimpleNamespace(method=method)
    )
    return executor


@pytest.mark.parametrize("executor_cls", [MultiprocExecutor, RayDistributedExecutor])
def test_retrospec_pipeline_uses_pipeline_depth(executor_cls):
    executor = make_executor_concurrency_view(executor_cls, "retrospec")

    assert executor.max_concurrent_batches == 2


@pytest.mark.parametrize("executor_cls", [MultiprocExecutor, RayDistributedExecutor])
@pytest.mark.parametrize("method", [None, "ngram"])
def test_non_retrospec_pipeline_retains_pipeline_batch_concurrency(
    executor_cls, method
):
    executor = make_executor_concurrency_view(executor_cls, method)

    assert executor.max_concurrent_batches == 2


@pytest.mark.parametrize("use_retrospec_pp_batch_queue", [False, True])
def test_engine_uses_tagged_drafts_for_retrospec_pp(
    use_retrospec_pp_batch_queue: bool,
):
    engine = EngineCore.__new__(EngineCore)
    engine.async_scheduling = False
    engine.use_spec_decode = True
    engine.use_retrospec_pp_batch_queue = use_retrospec_pp_batch_queue
    engine.model_executor = Mock()
    engine.model_executor.take_draft_token_ids.return_value = None
    engine.scheduler = Mock()

    engine.post_step(model_executed=True)

    if use_retrospec_pp_batch_queue:
        engine.model_executor.take_draft_token_ids.assert_not_called()
    else:
        engine.model_executor.take_draft_token_ids.assert_called_once_with()


def test_engine_does_not_submit_empty_retrospec_pp_batch():
    engine = EngineCore.__new__(EngineCore)
    engine._scheduler_paused = False
    engine.batch_queue_size = 2
    engine.use_retrospec_pp_batch_queue = True
    engine.is_ec_producer = False
    engine.is_pooling_model = False
    engine.scheduler = Mock()
    engine.scheduler.has_requests.return_value = True
    engine.scheduler.schedule.return_value = SchedulerOutput.make_empty()
    expected_outputs = {0: Mock()}
    engine.scheduler.update_from_output.return_value = expected_outputs
    engine.model_executor = Mock()
    engine.log_error_detail = Mock(return_value=nullcontext())
    engine.log_iteration_details = Mock(return_value=nullcontext())
    engine._process_aborts_queue = Mock()

    completed_output = Mock()
    completed_future = Future()
    completed_future.set_result(completed_output)
    completed_scheduler_output = SchedulerOutput.make_empty()
    completed_scheduler_output.num_scheduled_tokens = {"request": 1}
    completed_scheduler_output.total_num_scheduled_tokens = 1
    engine.batch_queue = deque(
        [(completed_future, completed_scheduler_output, Mock())], maxlen=2
    )

    outputs, model_executed = engine.step_with_batch_queue()

    assert outputs is expected_outputs
    assert not model_executed
    engine.model_executor.execute_model.assert_not_called()
    engine.model_executor.sample_tokens.assert_not_called()
    engine.scheduler.update_from_output.assert_called_once_with(
        completed_scheduler_output, completed_output
    )


def test_pipeline_stage_describes_local_layer_range():
    pp_group = make_pp_group(rank=1, world_size=3, is_last_rank=False)
    protocol = make_protocol()
    model = FakeLayerModel(start_layer=4, end_layer=7)

    with patch(
        "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
        return_value=pp_group,
    ):
        stage = protocol.describe_stage(model, ["layer.4", "layer.5", "layer.6"])

    assert stage == RetroSpecPipelineStage(
        rank=1,
        world_size=3,
        start_layer=4,
        end_layer=7,
    )
    assert not stage.is_first
    assert not stage.is_last
    assert stage.num_layers == 3


def test_pipeline_stage_rejects_mismatched_attention_layers():
    protocol = make_protocol()
    model = FakeLayerModel(start_layer=2, end_layer=4)

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=make_pp_group(),
        ),
        pytest.raises(RuntimeError, match="attention-layer count"),
    ):
        protocol.describe_stage(model, ["layer.2"])


def test_first_pipeline_stage_embeds_prompt_tokens():
    protocol = make_protocol()
    model = FakeLayerModel(start_layer=0, end_layer=2)
    stage = RetroSpecPipelineStage(0, 2, 0, 2)
    token_ids = torch.tensor([2, 4, 6], dtype=torch.int64)

    hidden_states = protocol.prepare_layer_prefill_input(
        stage=stage,
        layer_model=model,
        prompt_token_ids=token_ids,
        intermediate_tensors=None,
        prompt_num_tokens=3,
    )

    torch.testing.assert_close(
        hidden_states,
        torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]]),
    )


def test_nonfirst_pipeline_stage_consumes_and_emits_contiguous_hidden_states():
    protocol = make_protocol()
    model = FakeLayerModel(start_layer=2, end_layer=4)
    stage = RetroSpecPipelineStage(1, 3, 2, 4)
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4).T
    intermediate = IntermediateTensors({protocol._HIDDEN_STATES_KEY: hidden_states})

    received = protocol.prepare_layer_prefill_input(
        stage=stage,
        layer_model=model,
        prompt_token_ids=None,
        intermediate_tensors=intermediate,
        prompt_num_tokens=4,
    )
    output = protocol.make_layer_prefill_output(stage, received)

    assert received.data_ptr() == hidden_states.data_ptr()
    assert output[protocol._HIDDEN_STATES_KEY].is_contiguous()
    torch.testing.assert_close(output[protocol._HIDDEN_STATES_KEY], hidden_states)


def test_attention_mass_reduction_weights_pipeline_stages_by_layer_count():
    protocol = make_protocol()
    pp_group = make_pp_group(rank=0, world_size=2, is_last_rank=False)

    def add_remote_stage(reduction: torch.Tensor) -> torch.Tensor:
        result = reduction.clone()
        result[:2].add_(torch.tensor([9.0, 10.0]))
        result[2].add_(3.0)
        return result

    pp_group.all_reduce.side_effect = add_remote_stage
    stats = RetroSpecAttentionMassStats(
        value_sum=torch.tensor([1.0, 4.0]),
        layer_count=2,
    )

    with patch(
        "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
        return_value=pp_group,
    ):
        attention_mass = protocol.reduce_attention_mass(stats)

    torch.testing.assert_close(attention_mass, torch.tensor([2.0, 2.8]))
    reduced_input = pp_group.all_reduce.call_args.args[0]
    torch.testing.assert_close(reduced_input, torch.tensor([1.0, 4.0, 2.0]))


def test_final_pipeline_stage_packs_fixed_control_workspace():
    protocol = make_protocol()
    pp_group = make_pp_group(rank=1, world_size=2, is_last_rank=True)
    state = RetroSpecPipelineControlState(
        token_ids=torch.tensor([11, 12], dtype=torch.int32),
        stages=torch.tensor([1, 3], dtype=torch.int8),
        draft_counts=torch.tensor([2, 4], dtype=torch.int32),
        pending_counts=torch.tensor([5, 6], dtype=torch.int32),
        active_mask=torch.tensor([True, False]),
    )

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        patch("torch.distributed.broadcast") as broadcast,
    ):
        output = protocol.broadcast_control_state(2, state)

    assert broadcast.call_count == 2
    assert output.token_ids.tolist() == [11, 12]
    assert output.stages.tolist() == [1, 3]
    assert output.draft_counts.tolist() == [2, 4]
    assert output.pending_counts.tolist() == [5, 6]
    assert output.active_mask.tolist() == [True, False]
    assert output.token_ids.data_ptr() == protocol._integer_control.data_ptr()


def test_nonfinal_pipeline_stage_receives_control_workspace():
    protocol = make_protocol()
    pp_group = make_pp_group(rank=0, world_size=2, is_last_rank=False)

    def receive_control(tensor: torch.Tensor, **_kwargs) -> None:
        if tensor.dtype == torch.bool:
            tensor.copy_(torch.tensor([True, False]))
        else:
            tensor.copy_(
                torch.tensor(
                    [[21, 1, 2, 3], [22, 4, 5, 6]],
                    dtype=torch.int32,
                )
            )

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        patch("torch.distributed.broadcast", side_effect=receive_control),
    ):
        output = protocol.broadcast_control_state(2, None)

    assert output.token_ids.tolist() == [21, 22]
    assert output.stages.tolist() == [1, 4]
    assert output.draft_counts.tolist() == [2, 5]
    assert output.pending_counts.tolist() == [3, 6]
    assert output.active_mask.tolist() == [True, False]


def test_attention_mass_workspace_supports_parallel_verification_rows():
    protocol = make_protocol(max_batch_size=2, max_parallel_tokens=8)
    stats = RetroSpecAttentionMassStats(
        value_sum=torch.arange(8, dtype=torch.float32),
        layer_count=2,
    )

    with patch(
        "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
        return_value=make_pp_group(),
    ):
        attention_mass = protocol.reduce_attention_mass(stats)

    torch.testing.assert_close(attention_mass, stats.value_sum / 2)


def test_final_pipeline_stage_broadcasts_padded_target_samples():
    protocol = make_protocol(max_sampled_tokens=5)
    pp_group = make_pp_group(rank=1, world_size=2, is_last_rank=True)
    sampled_token_ids = torch.tensor([[11, 12, -1], [21, -1, -1]], dtype=torch.int32)

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        patch("torch.distributed.broadcast") as broadcast,
    ):
        output = protocol.broadcast_target_sampled_token_ids(2, sampled_token_ids)

    assert broadcast.call_count == 1
    assert output.tolist() == [
        [11, 12, -1, -1, -1],
        [21, -1, -1, -1, -1],
    ]


@pytest.mark.parametrize("compute_margin", [False, True])
def test_final_pipeline_stage_broadcasts_model_output(compute_margin: bool):
    protocol = make_protocol()
    pp_group = make_pp_group(rank=1, world_size=2, is_last_rank=True)
    token_ids = torch.tensor([31, 32, 33], dtype=torch.int32)
    margin = torch.tensor([0.1, 0.2, 0.3]) if compute_margin else None

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        patch("torch.distributed.broadcast") as broadcast,
    ):
        output = protocol.broadcast_model_output(
            num_tokens=3,
            token_ids=token_ids,
            margin=margin,
            compute_margin=compute_margin,
        )

    assert isinstance(output, RetroSpecPipelineModelOutput)
    assert output.token_ids.tolist() == [31, 32, 33]
    if compute_margin:
        assert output.margin is not None
        torch.testing.assert_close(output.margin, margin)
        assert broadcast.call_count == 2
    else:
        assert output.margin is None
        assert broadcast.call_count == 1


def test_nonfinal_pipeline_stage_receives_model_input():
    protocol = make_protocol()
    pp_group = make_pp_group(rank=1, world_size=3, is_last_rank=False)

    def receive_tensor(tensor: torch.Tensor) -> None:
        value = (pp_group.recv_into.call_count - 1) % 2 + 1
        tensor.fill_(float(value))

    pp_group.recv_into.side_effect = receive_tensor
    stage = RetroSpecPipelineStage(1, 3, 2, 4)

    with patch(
        "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
        return_value=pp_group,
    ):
        output = protocol.receive_model_input(stage, num_tokens=3)
        hidden_ptr = output["hidden_states"].data_ptr()
        residual_ptr = output["residual"].data_ptr()
        reused = protocol.receive_model_input(stage, num_tokens=3)

    assert isinstance(output, IntermediateTensors)
    torch.testing.assert_close(output["hidden_states"], torch.ones(3, 4))
    torch.testing.assert_close(output["residual"], torch.full((3, 4), 2.0))
    assert reused is output
    assert reused["hidden_states"].data_ptr() == hidden_ptr
    assert reused["residual"].data_ptr() == residual_ptr
    assert pp_group.recv_into.call_count == 4


def test_nonfinal_pipeline_stage_sends_model_output():
    protocol = make_protocol()
    pp_group = make_pp_group(rank=0, world_size=2, is_last_rank=False)
    stage = RetroSpecPipelineStage(0, 2, 0, 2)
    output = IntermediateTensors(
        {
            "hidden_states": torch.ones(3, 4),
            "residual": torch.full((3, 4), 2.0),
        }
    )

    with patch(
        "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
        return_value=pp_group,
    ):
        protocol.send_model_output(stage, output, num_tokens=3)

    assert pp_group.send.call_count == 2
    torch.testing.assert_close(
        pp_group.send.call_args_list[0].args[0], output["hidden_states"]
    )
    torch.testing.assert_close(
        pp_group.send.call_args_list[1].args[0], output["residual"]
    )


def test_pipeline_proposal_tp_transport_sends_shards_and_gathers_into_workspace():
    protocol = make_protocol(tensor_parallel_size=2)
    pp_group = make_pp_group(rank=1, world_size=2, is_last_rank=True)
    tp_group = SimpleNamespace(
        rank_in_group=1,
        world_size=2,
        all_gather_into_tensor=Mock(),
    )
    stage = RetroSpecPipelineStage(0, 2, 0, 2)
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    residual = hidden_states + 20
    output = IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_tp_group",
            return_value=tp_group,
        ),
    ):
        protocol.send_model_output(stage, output, num_tokens=3)

    assert pp_group.send.call_count == 2
    torch.testing.assert_close(
        pp_group.send.call_args_list[0].args[0], hidden_states.reshape(2, -1)[1]
    )
    torch.testing.assert_close(
        pp_group.send.call_args_list[1].args[0], residual.reshape(2, -1)[1]
    )

    def receive_shard(tensor: torch.Tensor) -> None:
        tensor.fill_(float(pp_group.recv_into.call_count))

    def gather_tensor(output_tensor: torch.Tensor, input_tensor: torch.Tensor) -> None:
        output_tensor.copy_(torch.cat((input_tensor, input_tensor + 10)))

    pp_group.recv_into.reset_mock()
    pp_group.recv_into.side_effect = receive_shard
    tp_group.all_gather_into_tensor.side_effect = gather_tensor
    stage = RetroSpecPipelineStage(1, 2, 2, 4)

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_tp_group",
            return_value=tp_group,
        ),
    ):
        received = protocol.receive_model_input(stage, num_tokens=3)

    assert pp_group.recv_into.call_count == 2
    assert tp_group.all_gather_into_tensor.call_count == 2
    torch.testing.assert_close(
        received["hidden_states"].view(-1),
        torch.tensor([1.0] * 6 + [11.0] * 6),
    )
    torch.testing.assert_close(
        received["residual"].view(-1),
        torch.tensor([2.0] * 6 + [12.0] * 6),
    )


def test_pipeline_sequence_parallel_residual_is_not_gathered():
    protocol = make_protocol(tensor_parallel_size=2, enable_sp=True)
    pp_group = make_pp_group(rank=0, world_size=2, is_last_rank=False)
    tp_group = SimpleNamespace(rank_in_group=0, world_size=2)
    stage = RetroSpecPipelineStage(0, 2, 0, 2)
    hidden_states = torch.arange(16, dtype=torch.float32).view(4, 4)
    residual = torch.arange(8, dtype=torch.float32).view(2, 4)
    output = IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_tp_group",
            return_value=tp_group,
        ),
    ):
        protocol.send_model_output(stage, output, num_tokens=4)

    torch.testing.assert_close(
        pp_group.send.call_args_list[0].args[0], hidden_states.reshape(2, -1)[0]
    )
    torch.testing.assert_close(pp_group.send.call_args_list[1].args[0], residual)


def test_pipeline_proposal_transport_rejects_dynamic_schema():
    protocol = make_protocol()
    pp_group = make_pp_group(rank=0, world_size=2, is_last_rank=False)
    stage = RetroSpecPipelineStage(0, 2, 0, 2)
    output = IntermediateTensors(
        {
            "hidden_states": torch.ones(3, 4),
            "residual": torch.ones(3, 4),
            "extra": torch.ones(3, 4),
        }
    )

    with (
        patch(
            "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group",
            return_value=pp_group,
        ),
        pytest.raises(RuntimeError, match="must contain exactly"),
    ):
        protocol.send_model_output(stage, output, num_tokens=3)
