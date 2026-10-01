# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.legacy.attention.kernels import (
    RetroSpecFullVerificationKVSource,
    _parallel_cluster_prefix_kernel,
    _parallel_native_suffix_kernel,
    _reduce_exact_partitions_kernel,
)

_EXACT_ATTENTION_BLOCK_TOKENS = 64
_PARALLEL_FULL_BLOCK_QUERIES = 16
_PARALLEL_FULL_BLOCK_TOKENS = 64
_PARALLEL_FULL_NATIVE_BLOCK_TOKENS = 32
_PARALLEL_FULL_NUM_SPLITS = 8


class RetroSpecExecutionVerificationMixin:
    def run_parallel_full_verification(
        self,
        source: RetroSpecFullVerificationKVSource,
        query: torch.Tensor,
        local_keys: torch.Tensor,
        local_values: torch.Tensor,
        scale: float,
        query_start_loc: torch.Tensor,
        max_query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run specialized clustered-prefix and causal native-suffix attention."""
        if query.ndim != 3:
            raise ValueError("Query must have shape [queries, query_heads, head_size]")
        if query.device.type != "cuda":
            raise ValueError("Parallel full verification requires CUDA tensors")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Parallel full verification requires FP16 or BF16")
        if query.shape[0] > self.max_num_queries:
            raise ValueError("Query count exceeds exact-attention workspace capacity")
        if local_keys.shape != local_values.shape or local_keys.ndim != 3:
            raise ValueError("Local KV must have shape [queries, kv_heads, head_size]")
        if local_keys.shape[0] != query.shape[0]:
            raise ValueError("Local KV must contain one entry per query")
        if local_keys.dtype != query.dtype or local_keys.device != query.device:
            raise ValueError("Local KV layout must match the query")
        if query_start_loc.ndim != 1 or query_start_loc.device != query.device:
            raise ValueError("query_start_loc must be a one-dimensional CUDA tensor")
        if query_start_loc.dtype not in (torch.int32, torch.int64):
            raise ValueError("query_start_loc must be integral")
        if max_query_len <= 0 and query.shape[0] > 0:
            raise ValueError("max_query_len must be positive for non-empty queries")

        primary = source.primary
        if primary.key_cache.shape != primary.value_cache.shape:
            raise ValueError("Primary key and value cache shapes must match")
        if primary.key_cache.ndim != 4 or primary.key_cache.shape[1] != self.page_size:
            raise ValueError("Primary KV cache has an invalid page layout")
        if primary.key_cache.dtype != query.dtype:
            raise ValueError("Primary KV dtype does not match query")
        if primary.token_indices.shape != primary.token_mask.shape:
            raise ValueError("Primary token indices and mask shapes must match")
        if primary.token_indices.ndim != 3 or primary.token_mask.dtype != torch.bool:
            raise ValueError("Primary token metadata has an invalid layout")
        batch_size, num_kv_heads, max_primary_tokens = primary.token_indices.shape
        if query_start_loc.shape != (batch_size + 1,):
            raise ValueError("query_start_loc must contain one boundary per request")
        if primary.block_table.shape[0] != batch_size:
            raise ValueError("Block table batch size does not match primary metadata")
        if primary.key_cache.shape[2:] != (num_kv_heads, query.shape[2]):
            raise ValueError("Primary KV shape does not match query metadata")
        if local_keys.shape[1:] != (num_kv_heads, query.shape[2]):
            raise ValueError("Local KV shape does not match primary metadata")
        if query.shape[1] % num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads")
        primary_tensors = (
            primary.key_cache,
            primary.value_cache,
            primary.block_table,
            primary.token_indices,
            primary.token_mask,
        )
        if any(tensor.device != query.device for tensor in primary_tensors):
            raise ValueError("All primary KV tensors must use the query device")

        clustered = source.clustered
        if clustered is not None:
            if clustered.key_tokens.shape != clustered.value_tokens.shape:
                raise ValueError("Clustered key and value token shapes must match")
            if clustered.key_tokens.ndim != 2:
                raise ValueError("Clustered KV must have shape [tokens, head_size]")
            if clustered.key_tokens.shape[1] != query.shape[2]:
                raise ValueError("Clustered KV head size does not match query")
            if clustered.token_offsets.shape != (batch_size, num_kv_heads):
                raise ValueError("Clustered token offsets have an invalid shape")
            if clustered.token_counts.shape != (batch_size, num_kv_heads):
                raise ValueError("Clustered token counts have an invalid shape")
            cluster_tensors = (
                clustered.key_tokens,
                clustered.value_tokens,
                clustered.token_offsets,
                clustered.token_counts,
            )
            if any(tensor.device != query.device for tensor in cluster_tensors):
                raise ValueError("All clustered KV tensors must use the query device")
            if clustered.key_tokens.dtype != query.dtype:
                raise ValueError("Clustered KV dtype does not match query")
            if clustered.token_offsets.dtype not in (torch.int32, torch.int64):
                raise ValueError("Clustered token offsets must be integral")
            if clustered.token_counts.dtype not in (torch.int32, torch.int64):
                raise ValueError("Clustered token counts must be integral")

        num_queries, num_query_heads, head_size = query.shape
        num_splits = min(_PARALLEL_FULL_NUM_SPLITS, self._partition_capacity)
        self._ensure_workspace(query)
        assert self._partial_output is not None
        assert self._partial_max is not None
        assert self._partial_sum is not None
        assert self._output is not None
        assert self._output_lse is not None
        assert self._secondary_output is not None
        assert self._secondary_output_lse is not None

        cluster_output = self._output[:num_queries]
        cluster_lse = self._output_lse[: num_query_heads * num_queries].view(
            num_query_heads, num_queries
        )
        native_output = self._secondary_output[:num_queries]
        native_lse = self._secondary_output_lse[: num_query_heads * num_queries].view(
            num_query_heads, num_queries
        )
        if num_queries == 0:
            return cluster_output, cluster_lse, native_output, native_lse

        current_stream = torch.cuda.current_stream(query.device)
        if primary.ready_event is not None:
            current_stream.wait_event(primary.ready_event)
        if clustered is not None and clustered.ready_event is not None:
            current_stream.wait_event(clustered.ready_event)

        block_d = triton.next_power_of_2(head_size)
        query_tiles = triton.cdiv(max_query_len, _PARALLEL_FULL_BLOCK_QUERIES)
        launch_grid = (batch_size, num_query_heads, query_tiles * num_splits)

        if clustered is None or clustered.max_tokens_per_head == 0:
            cluster_output.zero_()
            cluster_lse.fill_(float("-inf"))
        else:
            _parallel_cluster_prefix_kernel[launch_grid](
                query,
                query_start_loc,
                clustered.key_tokens,
                clustered.value_tokens,
                clustered.token_offsets,
                clustered.token_counts,
                self._partial_output,
                self._partial_max,
                self._partial_sum,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                clustered.key_tokens.stride(0),
                clustered.key_tokens.stride(1),
                clustered.value_tokens.stride(0),
                clustered.value_tokens.stride(1),
                clustered.token_offsets.stride(0),
                clustered.token_offsets.stride(1),
                self._partial_output.stride(0),
                self._partial_output.stride(1),
                self._partial_output.stride(2),
                self._partial_output.stride(3),
                self._partial_max.stride(0),
                self._partial_max.stride(1),
                self._partial_max.stride(2),
                scale,
                NUM_KV_HEADS=num_kv_heads,
                QUERIES_PER_KV_HEAD=num_query_heads // num_kv_heads,
                MAX_CLUSTER_TOKENS=clustered.max_tokens_per_head,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
                BLOCK_M=_PARALLEL_FULL_BLOCK_QUERIES,
                BLOCK_N=_PARALLEL_FULL_BLOCK_TOKENS,
                NUM_SPLITS=num_splits,
                num_warps=4,
                num_stages=2,
            )
            _reduce_exact_partitions_kernel[(num_queries, num_query_heads)](
                self._partial_output,
                self._partial_max,
                self._partial_sum,
                cluster_output,
                cluster_lse,
                self._partial_output.stride(0),
                self._partial_output.stride(1),
                self._partial_output.stride(2),
                self._partial_output.stride(3),
                self._partial_max.stride(0),
                self._partial_max.stride(1),
                self._partial_max.stride(2),
                cluster_output.stride(0),
                cluster_output.stride(1),
                cluster_output.stride(2),
                cluster_lse.stride(0),
                cluster_lse.stride(1),
                num_splits,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
            )

        _parallel_native_suffix_kernel[launch_grid](
            query,
            local_keys,
            local_values,
            query_start_loc,
            primary.key_cache,
            primary.value_cache,
            primary.block_table,
            primary.token_indices,
            primary.token_mask,
            self._partial_output,
            self._partial_max,
            self._partial_sum,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            local_keys.stride(0),
            local_keys.stride(1),
            local_keys.stride(2),
            local_values.stride(0),
            local_values.stride(1),
            local_values.stride(2),
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
            self._partial_output.stride(0),
            self._partial_output.stride(1),
            self._partial_output.stride(2),
            self._partial_output.stride(3),
            self._partial_max.stride(0),
            self._partial_max.stride(1),
            self._partial_max.stride(2),
            scale,
            NUM_KV_HEADS=num_kv_heads,
            QUERIES_PER_KV_HEAD=num_query_heads // num_kv_heads,
            MAX_PRIMARY_TOKENS=max_primary_tokens,
            MAX_QUERY_LEN=max_query_len,
            PAGE_SIZE=self.page_size,
            HEAD_SIZE=head_size,
            BLOCK_D=block_d,
            BLOCK_M=_PARALLEL_FULL_BLOCK_QUERIES,
            BLOCK_N=_PARALLEL_FULL_NATIVE_BLOCK_TOKENS,
            NUM_SPLITS=num_splits,
            num_warps=4,
            num_stages=2,
        )
        _reduce_exact_partitions_kernel[(num_queries, num_query_heads)](
            self._partial_output,
            self._partial_max,
            self._partial_sum,
            native_output,
            native_lse,
            self._partial_output.stride(0),
            self._partial_output.stride(1),
            self._partial_output.stride(2),
            self._partial_output.stride(3),
            self._partial_max.stride(0),
            self._partial_max.stride(1),
            self._partial_max.stride(2),
            native_output.stride(0),
            native_output.stride(1),
            native_output.stride(2),
            native_lse.stride(0),
            native_lse.stride(1),
            num_splits,
            HEAD_SIZE=head_size,
            BLOCK_D=block_d,
        )
        return cluster_output, cluster_lse, native_output, native_lse
