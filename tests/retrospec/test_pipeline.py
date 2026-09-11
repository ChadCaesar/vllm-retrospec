# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.spec_decode.retrospec.pipeline import (
    RetroSpecAttentionMassStats,
    RetroSpecPipelineControlState,
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
    )


def test_pipeline_stage_describes_local_layer_range():
    pp_group = make_pp_group(rank=1, world_size=3, is_last_rank=False)
    protocol = RetroSpecPipelineProtocol(torch.device("cpu"), max_batch_size=4)
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
    protocol = RetroSpecPipelineProtocol(torch.device("cpu"), max_batch_size=4)
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
    protocol = RetroSpecPipelineProtocol(torch.device("cpu"), max_batch_size=4)
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
    protocol = RetroSpecPipelineProtocol(torch.device("cpu"), max_batch_size=4)
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
    protocol = RetroSpecPipelineProtocol(torch.device("cpu"), max_batch_size=4)
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
    protocol = RetroSpecPipelineProtocol(torch.device("cpu"), max_batch_size=4)
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
    protocol = RetroSpecPipelineProtocol(torch.device("cpu"), max_batch_size=4)
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
