# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.model_executor.models.llama import LlamaModel
from vllm.v1.spec_decode.retrospec.prefill import (
    resolve_retrospec_layer_model,
)

pytestmark = pytest.mark.cpu_test


class AddLayer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.received_scale: float | None = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        self.received_scale = scale
        if residual is None:
            residual = hidden_states
        return hidden_states + self.value * scale, residual


class FinalNorm(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states + residual, residual


class TrackingLlamaModel(LlamaModel):
    def forward_layer(
        self,
        layer_index: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **extra_layer_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.executed_layer_indices.append(layer_index)
        return super().forward_layer(
            layer_index,
            positions,
            hidden_states,
            residual,
            **extra_layer_kwargs,
        )


def make_llama_model(*, start_layer: int = 0, end_layer: int = 2) -> TrackingLlamaModel:
    model = TrackingLlamaModel.__new__(TrackingLlamaModel)
    nn.Module.__init__(model)
    model.start_layer = start_layer
    model.end_layer = end_layer
    model.layers = nn.ModuleList([AddLayer(1.0), AddLayer(2.0), AddLayer(3.0)])
    model.norm = FinalNorm()
    model.aux_hidden_state_layers = tuple()
    model.executed_layer_indices: list[int] = []
    model.do_not_compile = True
    return model


def test_forward_layer_uses_global_index_and_forwards_kwargs():
    model = make_llama_model(start_layer=1, end_layer=3)
    positions = torch.arange(2)
    hidden_states = torch.ones(2, 4)

    output, residual = model.forward_layer(1, positions, hidden_states, None, scale=2.0)

    torch.testing.assert_close(output, hidden_states + 4.0)
    torch.testing.assert_close(residual, hidden_states)
    assert model.layers[1].received_scale == 2.0


@pytest.mark.parametrize("layer_index", [0, 3])
def test_forward_layer_rejects_layer_outside_pipeline_range(layer_index: int):
    model = make_llama_model(start_layer=1, end_layer=3)

    with pytest.raises(ValueError, match="not owned by this pipeline rank"):
        model.forward_layer(
            layer_index,
            torch.arange(1),
            torch.ones(1, 4),
            None,
        )


def test_regular_forward_uses_forward_layer_without_changing_output():
    model = make_llama_model()
    positions = torch.arange(2)
    inputs_embeds = torch.ones(2, 4)
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    with patch("vllm.model_executor.models.llama.get_pp_group", return_value=pp_group):
        output = model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )

    torch.testing.assert_close(output, inputs_embeds * 2 + 3.0)
    assert model.executed_layer_indices == [0, 1]


def test_resolve_retrospec_layer_model_accepts_inner_and_wrapped_model():
    layer_model = make_llama_model()
    wrapper = nn.Module()
    wrapper.model = layer_model

    assert resolve_retrospec_layer_model(layer_model) is layer_model
    assert resolve_retrospec_layer_model(wrapper) is layer_model


def test_resolve_retrospec_layer_model_rejects_unsupported_model():
    with pytest.raises(TypeError, match="requires a model exposing"):
        resolve_retrospec_layer_model(nn.Identity())
