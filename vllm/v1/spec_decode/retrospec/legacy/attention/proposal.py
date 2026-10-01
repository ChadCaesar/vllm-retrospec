# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.legacy.attention.kernels import (
    RetroSpecEstimationKVSource,
    RetroSpecExactKVSource,
    RetroSpecRankedDraftKVSource,
    _accumulate_exact_partition_wave_kernel,
    _ranked_draft_attention_partition_kernel,
    _reduce_exact_partitions_kernel,
    _reduce_proposal_partitions_kernel,
)
from vllm.v1.spec_decode.retrospec.workspace import (
    EXACT_ATTENTION_PARTITION_SIZE,
)

_EXACT_ATTENTION_BLOCK_TOKENS = 64
_PARALLEL_FULL_BLOCK_QUERIES = 16
_PARALLEL_FULL_BLOCK_TOKENS = 64
_PARALLEL_FULL_NATIVE_BLOCK_TOKENS = 32
_PARALLEL_FULL_NUM_SPLITS = 8


class RetroSpecExecutionProposalMixin:
    def _launch_ranked_draft_partitions(
        self,
        source: RetroSpecRankedDraftKVSource,
        query: torch.Tensor,
        scale: float,
        num_kv_heads: int,
        max_primary_tokens: int,
        retrieval_width: int,
        estimation_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_queries, num_query_heads, head_size = query.shape
        max_pages = source.resident_table_page_slots.shape[1]
        source_capacity = (
            max_primary_tokens
            + retrieval_width * max_pages * self.page_size
            + estimation_width
            + retrieval_width
        )
        total_partitions = triton.cdiv(source_capacity, EXACT_ATTENTION_PARTITION_SIZE)
        self._ensure_workspace(query)
        assert self._partial_output is not None
        assert self._partial_max is not None
        assert self._partial_sum is not None
        assert self._accumulated_output is not None
        assert self._accumulated_max is not None
        assert self._accumulated_sum is not None

        primary = source.primary
        block_d = triton.next_power_of_2(head_size)
        partition_start = 0
        wave_index = 0
        while partition_start < total_partitions:
            wave_partitions = min(
                self._partition_capacity, total_partitions - partition_start
            )
            _ranked_draft_attention_partition_kernel[
                (num_queries, num_query_heads, wave_partitions)
            ](
                query,
                primary.key_cache,
                primary.value_cache,
                primary.block_table,
                primary.token_indices,
                primary.token_mask,
                source.request_slot_ids,
                source.ranked_cluster_indices,
                source.candidate_counts,
                source.resident_bucket_ids,
                source.cluster_keys,
                source.cluster_values,
                source.cluster_token_counts,
                source.cluster_page_starts,
                source.cluster_page_counts,
                source.page_token_counts,
                source.cluster_offsets,
                source.page_offsets,
                source.resident_table_page_slots,
                source.resident_key_pages,
                source.resident_value_pages,
                self._partial_output,
                self._partial_max,
                self._partial_sum,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                primary.key_cache.stride(0),
                primary.key_cache.stride(1),
                primary.key_cache.stride(2),
                primary.key_cache.stride(3),
                primary.value_cache.stride(0),
                primary.value_cache.stride(1),
                primary.value_cache.stride(2),
                primary.value_cache.stride(3),
                primary.block_table.stride(0),
                primary.block_table.stride(1),
                source.resident_key_pages.stride(0),
                source.resident_key_pages.stride(1),
                source.resident_key_pages.stride(2),
                source.resident_value_pages.stride(0),
                source.resident_value_pages.stride(1),
                source.resident_value_pages.stride(2),
                self._partial_output.stride(0),
                self._partial_output.stride(1),
                self._partial_output.stride(2),
                self._partial_output.stride(3),
                self._partial_max.stride(0),
                self._partial_max.stride(1),
                self._partial_max.stride(2),
                scale,
                partition_start,
                NUM_KV_HEADS=num_kv_heads,
                QUERIES_PER_KV_HEAD=num_query_heads // num_kv_heads,
                MAX_PRIMARY_TOKENS=max_primary_tokens,
                RETRIEVAL_WIDTH=retrieval_width,
                ESTIMATION_WIDTH=estimation_width,
                MAX_PAGES=max_pages,
                PAGE_SIZE=self.page_size,
                CLUSTER_CAPACITY=source.cluster_keys.shape[1],
                PAGE_CAPACITY=source.page_token_counts.shape[1],
                RESIDENT_PAGE_STRIDE=source.resident_table_page_slots.stride(0),
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
                PARTITION_SIZE=EXACT_ATTENTION_PARTITION_SIZE,
                BLOCK_TOKENS=_EXACT_ATTENTION_BLOCK_TOKENS,
                RANKING_WIDTH=source.ranked_cluster_indices.shape[2],
                RETRIEVAL_RATIO=source.retrieval_ratio,
                ESTIMATION_RATIO=source.estimation_ratio,
            )
            partition_start += wave_partitions
            if total_partitions <= self._partition_capacity:
                return (
                    self._partial_output,
                    self._partial_max,
                    self._partial_sum,
                    wave_partitions,
                )

            _accumulate_exact_partition_wave_kernel[(num_queries, num_query_heads)](
                self._partial_output,
                self._partial_max,
                self._partial_sum,
                self._accumulated_output,
                self._accumulated_max,
                self._accumulated_sum,
                self._partial_output.stride(0),
                self._partial_output.stride(1),
                self._partial_output.stride(2),
                self._partial_output.stride(3),
                self._partial_max.stride(0),
                self._partial_max.stride(1),
                self._partial_max.stride(2),
                self._accumulated_output.stride(0),
                self._accumulated_output.stride(1),
                self._accumulated_output.stride(3),
                self._accumulated_max.stride(0),
                self._accumulated_max.stride(1),
                self._accumulated_max.stride(2),
                wave_partitions,
                RESET=wave_index == 0,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
            )
            wave_index += 1

        return (
            self._accumulated_output,
            self._accumulated_max,
            self._accumulated_sum,
            1,
        )

    def run_ranked_draft_proposal(
        self,
        source: RetroSpecRankedDraftKVSource,
        query: torch.Tensor,
        scale: float,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run DRAFT attention directly from ranked index and resident buckets."""
        (
            num_kv_heads,
            max_primary_tokens,
            retrieval_width,
            estimation_width,
        ) = self._validate_ranked_draft_source(source, query, output)
        partial_output, partial_max, partial_sum, num_partitions = (
            self._launch_ranked_draft_partitions(
                source,
                query,
                scale,
                num_kv_heads,
                max_primary_tokens,
                retrieval_width,
                estimation_width,
            )
        )
        num_queries, num_query_heads, head_size = query.shape
        if num_queries == 0:
            return output

        assert self._output_lse is not None
        output_lse = self._output_lse[: num_queries * num_query_heads].view(
            num_query_heads, num_queries
        )
        _reduce_exact_partitions_kernel[(num_queries, num_query_heads)](
            partial_output,
            partial_max,
            partial_sum,
            output,
            output_lse,
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_output.stride(3),
            partial_max.stride(0),
            partial_max.stride(1),
            partial_max.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output_lse.stride(0),
            output_lse.stride(1),
            num_partitions,
            HEAD_SIZE=head_size,
            BLOCK_D=triton.next_power_of_2(head_size),
        )
        return output

    def run_proposal(
        self,
        source: RetroSpecExactKVSource,
        estimation: RetroSpecEstimationKVSource,
        query: torch.Tensor,
        scale: float,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused exact and weighted-estimation proposal attention."""
        num_kv_heads, max_primary_tokens, max_page_slots = self._validate_source(
            source, query, request_indices=None
        )
        num_estimation_vectors = self._validate_estimation_source(
            estimation, query, num_kv_heads
        )
        if output.shape != query.shape:
            raise ValueError("Proposal output shape must match query")
        if output.dtype != query.dtype:
            raise ValueError("Proposal output dtype must match query")
        if output.device != query.device:
            raise ValueError("Proposal output must use the query device")

        num_queries, num_query_heads, head_size = query.shape
        partial_output, partial_max, partial_sum, num_partitions = (
            self._launch_exact_partitions(
                source,
                query,
                scale,
                None,
                num_kv_heads,
                max_primary_tokens,
                max_page_slots,
            )
        )
        if num_queries == 0:
            return output

        plan_row_mapping = (
            estimation.keys
            if estimation.plan_row_indices is None
            else estimation.plan_row_indices
        )
        block_d = triton.next_power_of_2(head_size)
        _reduce_proposal_partitions_kernel[(num_queries, num_query_heads)](
            query,
            plan_row_mapping,
            estimation.keys,
            estimation.values,
            estimation.token_counts,
            partial_output,
            partial_max,
            partial_sum,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            estimation.keys.stride(0),
            estimation.keys.stride(1),
            estimation.keys.stride(2),
            estimation.keys.stride(3),
            estimation.values.stride(0),
            estimation.values.stride(1),
            estimation.values.stride(2),
            estimation.values.stride(3),
            estimation.token_counts.stride(0),
            estimation.token_counts.stride(1),
            estimation.token_counts.stride(2),
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_output.stride(3),
            partial_max.stride(0),
            partial_max.stride(1),
            partial_max.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            scale,
            num_partitions,
            num_estimation_vectors,
            USE_PLAN_ROWS=estimation.plan_row_indices is not None,
            QUERIES_PER_KV_HEAD=num_query_heads // num_kv_heads,
            HEAD_SIZE=head_size,
            BLOCK_M=16,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return output
