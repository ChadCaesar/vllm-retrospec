# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.legacy.attention.kernels import (
    RetroSpecExactKVSource,
    _accumulate_exact_partition_wave_kernel,
    _multi_source_exact_partition_kernel,
    _reduce_exact_partitions_kernel,
)
from vllm.v1.spec_decode.retrospec.workspace import (
    EXACT_ATTENTION_PARTITION_SIZE,
    exact_attention_source_token_capacity,
)

_EXACT_ATTENTION_BLOCK_TOKENS = 64
_PARALLEL_FULL_BLOCK_QUERIES = 16
_PARALLEL_FULL_BLOCK_TOKENS = 64
_PARALLEL_FULL_NATIVE_BLOCK_TOKENS = 32
_PARALLEL_FULL_NUM_SPLITS = 8


class RetroSpecExecutionExactMixin:
    def _launch_exact_partitions(
        self,
        source: RetroSpecExactKVSource,
        query: torch.Tensor,
        scale: float,
        request_indices: torch.Tensor | None,
        num_kv_heads: int,
        max_primary_tokens: int,
        max_page_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_queries, num_query_heads, head_size = query.shape
        num_source_tokens = exact_attention_source_token_capacity(
            max_primary_tokens,
            max_page_slots,
            self.page_size,
        )
        total_partitions = triton.cdiv(
            num_source_tokens, EXACT_ATTENTION_PARTITION_SIZE
        )
        self._ensure_workspace(query)

        assert self._partial_output is not None
        assert self._partial_max is not None
        assert self._partial_sum is not None
        assert self._accumulated_output is not None
        assert self._accumulated_max is not None
        assert self._accumulated_sum is not None
        if num_queries == 0:
            return (
                self._partial_output,
                self._partial_max,
                self._partial_sum,
                min(total_partitions, self._partition_capacity),
            )

        primary = source.primary
        current_stream = torch.cuda.current_stream(query.device)
        for exact_source in (
            primary,
            source.resident_pages,
            source.staging_pages,
        ):
            if exact_source is not None and exact_source.ready_event is not None:
                current_stream.wait_event(exact_source.ready_event)

        resident = source.resident_pages
        staging = source.staging_pages
        dummy_page_ids = source.page_token_counts
        compact_page_counts = (
            source.page_token_counts
            if source.compact_pages is None
            else source.compact_pages.page_counts
        )
        dummy_key_pages = primary.key_cache[:, :, 0, :]
        dummy_value_pages = primary.value_cache[:, :, 0, :]
        resident_page_ids = dummy_page_ids if resident is None else resident.page_ids
        resident_key_pages = dummy_key_pages if resident is None else resident.key_pages
        resident_value_pages = (
            dummy_value_pages if resident is None else resident.value_pages
        )
        staging_page_ids = dummy_page_ids if staging is None else staging.page_ids
        staging_key_pages = dummy_key_pages if staging is None else staging.key_pages
        staging_value_pages = (
            dummy_value_pages if staging is None else staging.value_pages
        )
        request_mapping = (
            primary.token_indices if request_indices is None else request_indices
        )
        plan_row_mapping = (
            primary.token_indices
            if source.plan_row_indices is None
            else source.plan_row_indices
        )
        block_d = triton.next_power_of_2(head_size)

        partition_start = 0
        wave_index = 0
        while partition_start < total_partitions:
            wave_partitions = min(
                self._partition_capacity,
                total_partitions - partition_start,
            )
            _multi_source_exact_partition_kernel[
                (num_queries, num_query_heads, wave_partitions)
            ](
                query,
                request_mapping,
                plan_row_mapping,
                primary.key_cache,
                primary.value_cache,
                primary.block_table,
                primary.token_indices,
                primary.token_mask,
                source.page_token_counts,
                compact_page_counts,
                resident_page_ids,
                resident_key_pages,
                resident_value_pages,
                staging_page_ids,
                staging_key_pages,
                staging_value_pages,
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
                resident_key_pages.stride(0),
                resident_key_pages.stride(1),
                resident_key_pages.stride(2),
                resident_value_pages.stride(0),
                resident_value_pages.stride(1),
                resident_value_pages.stride(2),
                staging_key_pages.stride(0),
                staging_key_pages.stride(1),
                staging_key_pages.stride(2),
                staging_value_pages.stride(0),
                staging_value_pages.stride(1),
                staging_value_pages.stride(2),
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
                MAX_PAGE_SLOTS=max_page_slots,
                PAGE_SIZE=self.page_size,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
                PARTITION_SIZE=EXACT_ATTENTION_PARTITION_SIZE,
                BLOCK_TOKENS=_EXACT_ATTENTION_BLOCK_TOKENS,
                IDENTITY_REQUESTS=request_indices is None,
                USE_PLAN_ROWS=source.plan_row_indices is not None,
                HAS_RESIDENT=resident is not None,
                HAS_STAGING=staging is not None,
                COMPACT_PAGES=source.compact_pages is not None,
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

    def run(
        self,
        source: RetroSpecExactKVSource,
        query: torch.Tensor,
        scale: float,
        request_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run exact attention without materializing a contiguous KV copy."""
        num_kv_heads, max_primary_tokens, max_page_slots = self._validate_source(
            source, query, request_indices
        )
        num_queries, num_query_heads, head_size = query.shape
        partial_output, partial_max, partial_sum, num_partitions = (
            self._launch_exact_partitions(
                source,
                query,
                scale,
                request_indices,
                num_kv_heads,
                max_primary_tokens,
                max_page_slots,
            )
        )

        assert self._output is not None
        assert self._output_lse is not None

        output = self._output[:num_queries]
        num_output_lse_elements = num_query_heads * num_queries
        output_lse = self._output_lse[:num_output_lse_elements].view(
            num_query_heads, num_queries
        )
        if num_queries == 0:
            return output, output_lse

        block_d = triton.next_power_of_2(head_size)
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
            BLOCK_D=block_d,
        )
        return output, output_lse
