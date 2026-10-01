# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.retrospec.legacy.attention.exact import (
    RetroSpecExecutionExactMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.attention.kernels import (
    RetroSpecCompactExactPageTable,
    RetroSpecCompactKVSource,
    RetroSpecEstimationKVSource,
    RetroSpecExactKVSource,
    RetroSpecExactPageKVSource,
    RetroSpecExactPrimaryKVSource,
    RetroSpecFullVerificationKVSource,
    RetroSpecRankedDraftKVSource,
    _accumulate_exact_partition_wave_kernel,
    _multi_source_exact_partition_kernel,
    _parallel_cluster_prefix_kernel,
    _parallel_native_suffix_kernel,
    _ranked_draft_attention_partition_kernel,
    _reduce_exact_partitions_kernel,
    _reduce_proposal_partitions_kernel,
)
from vllm.v1.spec_decode.retrospec.legacy.attention.proposal import (
    RetroSpecExecutionProposalMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.attention.validation import (
    RetroSpecExecutionValidationMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.attention.verification import (
    RetroSpecExecutionVerificationMixin,
)

from ..workspace import EXACT_ATTENTION_PARTITION_SIZE

_EXACT_ATTENTION_BLOCK_TOKENS = 64
_PARALLEL_FULL_BLOCK_QUERIES = 16
_PARALLEL_FULL_BLOCK_TOKENS = 64
_PARALLEL_FULL_NATIVE_BLOCK_TOKENS = 32
_PARALLEL_FULL_NUM_SPLITS = 8
__all__ = [
    "EXACT_ATTENTION_PARTITION_SIZE",
    "RetroSpecExactPrimaryKVSource",
    "RetroSpecExactPageKVSource",
    "RetroSpecCompactExactPageTable",
    "RetroSpecExactKVSource",
    "RetroSpecEstimationKVSource",
    "RetroSpecRankedDraftKVSource",
    "RetroSpecCompactKVSource",
    "RetroSpecFullVerificationKVSource",
    "_multi_source_exact_partition_kernel",
    "_ranked_draft_attention_partition_kernel",
    "_accumulate_exact_partition_wave_kernel",
    "_reduce_exact_partitions_kernel",
    "_reduce_proposal_partitions_kernel",
    "_parallel_cluster_prefix_kernel",
    "_parallel_native_suffix_kernel",
    "RetroSpecExactAttentionWorkspace",
]


class RetroSpecExactAttentionWorkspace(
    RetroSpecExecutionValidationMixin,
    RetroSpecExecutionVerificationMixin,
    RetroSpecExecutionExactMixin,
    RetroSpecExecutionProposalMixin,
):
    """Reusable workspace for partitioned attention over three KV sources."""

    def __init__(
        self,
        page_size: int,
        max_num_queries: int,
        partition_capacity: int,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if max_num_queries <= 0:
            raise ValueError("max_num_queries must be positive")
        if partition_capacity <= 0:
            raise ValueError("partition_capacity must be positive")
        if partition_capacity & (partition_capacity - 1):
            raise ValueError("partition_capacity must be a power of two")

        self.page_size = page_size
        self.max_num_queries = max_num_queries
        self._partition_capacity = partition_capacity
        self._configuration: tuple[torch.dtype, torch.device, int, int] | None = None
        self._partial_output: torch.Tensor | None = None
        self._partial_max: torch.Tensor | None = None
        self._partial_sum: torch.Tensor | None = None
        self._accumulated_output: torch.Tensor | None = None
        self._accumulated_max: torch.Tensor | None = None
        self._accumulated_sum: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._output_lse: torch.Tensor | None = None
        self._secondary_output: torch.Tensor | None = None
        self._secondary_output_lse: torch.Tensor | None = None

    def _ensure_output_workspace(self, query: torch.Tensor) -> None:
        _, num_query_heads, head_size = query.shape
        configuration = (query.dtype, query.device, num_query_heads, head_size)
        if self._configuration is not None and self._configuration != configuration:
            raise ValueError("Exact-attention workspace configuration changed")
        if self._configuration is not None:
            return
        output = torch.empty(
            self.max_num_queries,
            num_query_heads,
            head_size,
            dtype=query.dtype,
            device=query.device,
        )
        output_lse = torch.empty(
            num_query_heads * self.max_num_queries,
            dtype=torch.float32,
            device=query.device,
        )
        self._output = output
        self._output_lse = output_lse
        self._secondary_output = torch.empty_like(output)
        self._secondary_output_lse = torch.empty_like(output_lse)
        self._configuration = configuration

    def _ensure_workspace(self, query: torch.Tensor) -> None:
        self._ensure_output_workspace(query)
        if self._partial_output is not None:
            return

        _, num_query_heads, head_size = query.shape
        partial_shape = (
            self.max_num_queries,
            num_query_heads,
            self._partition_capacity,
            head_size,
        )
        stats_shape = partial_shape[:-1]
        self._partial_output = torch.empty(
            partial_shape, dtype=query.dtype, device=query.device
        )
        self._partial_max = torch.empty(
            stats_shape, dtype=torch.float32, device=query.device
        )
        self._partial_sum = torch.empty_like(self._partial_max)
        accumulated_shape = (
            self.max_num_queries,
            num_query_heads,
            1,
            head_size,
        )
        accumulated_stats_shape = accumulated_shape[:-1]
        self._accumulated_output = torch.empty(
            accumulated_shape, dtype=query.dtype, device=query.device
        )
        self._accumulated_max = torch.empty(
            accumulated_stats_shape, dtype=torch.float32, device=query.device
        )
        self._accumulated_sum = torch.empty_like(self._accumulated_max)
