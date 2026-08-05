# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.retrospec import RetroSpecProposer


def make_vllm_config() -> VllmConfig:
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=64,
    )
    return cast(
        VllmConfig,
        SimpleNamespace(speculative_config=speculative_config),
    )


def test_retrospec_proposer_initialization():
    vllm_config = make_vllm_config()
    runner = Mock()
    device = torch.device("cpu")

    proposer = RetroSpecProposer(vllm_config, device, runner)

    assert proposer.vllm_config is vllm_config
    assert proposer.device == device
    assert proposer.runner is runner
    assert proposer.model is None
    assert proposer.num_speculative_tokens == 64


def test_retrospec_proposer_loads_target_model():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        Mock(),
    )
    target_model = Mock()

    proposer.load_model(target_model)

    assert proposer.model is target_model


def test_retrospec_proposer_scaffold_fails_explicitly():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        Mock(),
    )

    with pytest.raises(NotImplementedError, match="next commit"):
        proposer.propose()
