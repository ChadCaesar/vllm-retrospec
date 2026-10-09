# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetroSpecVerificationResult:
    verified_counts: torch.Tensor
    require_full: torch.Tensor
    bonus_mask: torch.Tensor | None = None
    bonus_token_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class RetroSpecParallelVerificationOutput:
    request_indices: torch.Tensor
    token_indices: torch.Tensor
    token_ids: torch.Tensor
    margin: torch.Tensor | None
    attention_mass: torch.Tensor
