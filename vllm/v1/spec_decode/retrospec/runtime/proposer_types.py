# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetroSpecVerificationResult:
    verified_counts: torch.Tensor
    require_full: torch.Tensor


@dataclass(frozen=True)
class RetroSpecParallelVerificationOutput:
    request_indices: torch.Tensor
    token_indices: torch.Tensor
    token_ids: torch.Tensor
    margin: torch.Tensor | None
    attention_mass: torch.Tensor


# Keep the original qualified class paths for pickles and diagnostics.
RetroSpecVerificationResult.__module__ = "vllm.v1.spec_decode.retrospec.proposer"
RetroSpecParallelVerificationOutput.__module__ = (
    "vllm.v1.spec_decode.retrospec.proposer"
)
