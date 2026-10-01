# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility exports for GPU-native index and sparse-attention internals."""

from vllm.v1.spec_decode.retrospec.native.kernels.attention import (
    _merge_native_attention_partitions,
    _native_grouped_sparse_attention_kernel,
    _native_sparse_attention_kernel,
)
from vllm.v1.spec_decode.retrospec.native.kernels.build import (
    _scatter_cluster_tokens_kernel,
)
from vllm.v1.spec_decode.retrospec.native.kernels.rank import (
    _cluster_mass_kernel,
    _cluster_rank_dot_kernel,
    _cluster_rank_int8_kernel,
    _cluster_scores_kernel,
)
from vllm.v1.spec_decode.retrospec.native.types import (
    _NativeAttentionWorkspace,
    _NativeBatchLayer,
    _NativeLayerRecord,
    _NativePlanWorkspace,
    _NativeRankedPlan,
)

__all__ = [
    "_NativeLayerRecord",
    "_NativeBatchLayer",
    "_NativeRankedPlan",
    "_NativePlanWorkspace",
    "_NativeAttentionWorkspace",
    "_scatter_cluster_tokens_kernel",
    "_cluster_rank_dot_kernel",
    "_cluster_rank_int8_kernel",
    "_cluster_scores_kernel",
    "_cluster_mass_kernel",
    "_native_sparse_attention_kernel",
    "_native_grouped_sparse_attention_kernel",
    "_merge_native_attention_partitions",
]
