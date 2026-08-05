# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class RetroSpecProposer:
    """
    Self-speculative proposer backed by RetroSpec attention.
    This initial scaffold only registers the proposer with vLLM.
    Draft execution is implemented in the next integration commit.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "GPUModelRunner",
    ) -> None:
        config = vllm_config.speculative_config
        assert config is not None
        assert config.method == "retrospec"

        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        self.model = None
        self.num_speculative_tokens = config.num_speculative_tokens

    def load_model(self, target_model) -> None:
        self.model = target_model

    def propose(self, *args, **kwargs):
        raise NotImplementedError(
            "RetroSpec proposal execution will be added in the next commit."
        )
