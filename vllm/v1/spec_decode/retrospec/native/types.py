# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _NativeLayerRecord:
    indexed_end: int
    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    token_indices: torch.Tensor
    cluster_offsets: torch.Tensor
    key_storage: torch.Tensor | None = None
    value_storage: torch.Tensor | None = None
    count_storage: torch.Tensor | None = None
    token_storage: torch.Tensor | None = None
    offset_storage: torch.Tensor | None = None


@dataclass(frozen=True)
class _NativeBatchLayer:
    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    token_indices: torch.Tensor
    cluster_offsets: torch.Tensor
    indexed_ends: torch.Tensor
    request_slots: torch.Tensor | None = None
    quantized_keys: torch.Tensor | None = None
    key_scales: torch.Tensor | None = None


@dataclass(frozen=True)
class _NativeRankedPlan:
    ranked: torch.Tensor
    candidate_counts: torch.Tensor
    sparse_mass: torch.Tensor
    verification_mass: torch.Tensor
    expanded_mass: torch.Tensor


@dataclass(frozen=True)
class _NativePlanWorkspace:
    ranked: torch.Tensor
    candidate_counts: torch.Tensor
    sparse_mass: torch.Tensor
    verification_mass: torch.Tensor
    expanded_mass: torch.Tensor
    scores: torch.Tensor
    topk_values: torch.Tensor


@dataclass(frozen=True)
class _NativeAttentionWorkspace:
    output: torch.Tensor
    maximum: torch.Tensor
    denominator: torch.Tensor
