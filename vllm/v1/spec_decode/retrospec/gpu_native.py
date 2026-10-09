# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU-only clustered index and sparse attention for RetroSpec.

The native vLLM KV blocks remain the sole exact-token storage. The reverse
cluster layout contains logical token indices, not copies of K/V vectors.
"""

from contextlib import nullcontext

import torch

from vllm.v1.spec_decode.retrospec.native.kernels import (
    _cluster_mass_kernel,
    _cluster_rank_dot_kernel,
    _cluster_rank_int8_kernel,
    _cluster_scores_kernel,
    _merge_native_attention_partitions,
    _native_grouped_sparse_attention_kernel,
    _native_sparse_attention_kernel,
    _NativeAttentionWorkspace,
    _NativeBatchLayer,
    _NativeLayerRecord,
    _NativePlanWorkspace,
    _NativeRankedPlan,
    _scatter_cluster_tokens_kernel,
)

from .index import RetroSpecIndexBase
from .performance import RetroSpecPerformanceStats

__all__ = [
    "RetroSpecGPUNativeIndex",
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


from vllm.v1.spec_decode.retrospec.native.attention import RetroSpecNativeAttentionMixin
from vllm.v1.spec_decode.retrospec.native.index import RetroSpecNativeIndexMixin
from vllm.v1.spec_decode.retrospec.native.rank import RetroSpecNativeRankMixin


class RetroSpecGPUNativeIndex(
    RetroSpecNativeIndexMixin,
    RetroSpecNativeRankMixin,
    RetroSpecNativeAttentionMixin,
    RetroSpecIndexBase,
):
    """Per-request GPU summaries and reverse logical-token layouts."""

    def __init__(
        self,
        *,
        block_size: int,
        num_speculative_tokens: int,
        retrieval_ratio: float,
        estimation_ratio: float,
        prefill_segment_size_tokens: int,
        generation_update_interval: int,
        blocks_per_cluster: int,
        num_kmeans_iterations: int,
        draft_rank_dtype: str = "native",
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        super().__init__(
            block_size, num_speculative_tokens, retrieval_ratio, estimation_ratio
        )
        self.num_speculative_tokens = num_speculative_tokens
        self.prefill_segment_size_tokens = prefill_segment_size_tokens
        self.generation_update_interval = generation_update_interval
        self.tokens_per_cluster = blocks_per_cluster * block_size
        self.num_kmeans_iterations = num_kmeans_iterations
        if draft_rank_dtype not in ("int8", "native"):
            raise ValueError("draft_rank_dtype must be 'int8' or 'native'")
        self.draft_rank_dtype = draft_rank_dtype
        self.performance_stats = performance_stats
        self._records: dict[str, dict[str, _NativeLayerRecord]] = {}
        self._staged: dict[tuple[str, str], _NativeLayerRecord | None] = {}
        self._request_ids: tuple[str, ...] = ()
        self._batch_layers: dict[str, _NativeBatchLayer] = {}
        self._active_cluster_counts: dict[str, int] = {}
        self._workspaces: dict[str, _NativeBatchLayer] = {}
        self._workspace_generations: dict[str, int] = {}
        self._arena_slots: dict[str, dict[str, int]] = {}
        self._arena_free_slots: dict[str, list[int]] = {}
        self._arena_versions: dict[str, dict[str, _NativeLayerRecord]] = {}
        self._plans: dict[str, dict[int, _NativeRankedPlan]] = {}
        self._plan_workspaces: dict[str, _NativePlanWorkspace] = {}
        self._active_plan_workspaces: dict[str, _NativePlanWorkspace] = {}
        self._logit_workspaces: dict[str, torch.Tensor] = {}
        self._shared_attention_workspace: _NativeAttentionWorkspace | None = None
        self._request_slots: torch.Tensor | None = None

    @property
    def has_staged_updates(self) -> bool:
        return bool(self._staged)

    def _cuda_timer(self, name: str):
        if self.performance_stats is None:
            return nullcontext()
        return self.performance_stats.cuda_timer(name)

    def close(self) -> None:
        self._records.clear()
        self._staged.clear()
        self._batch_layers.clear()
        self._workspaces.clear()
        self._workspace_generations.clear()
        self._arena_slots.clear()
        self._arena_free_slots.clear()
        self._arena_versions.clear()
        self._plans.clear()
        self._plan_workspaces.clear()
        self._active_plan_workspaces.clear()
        self._logit_workspaces.clear()
        self._request_slots = None
