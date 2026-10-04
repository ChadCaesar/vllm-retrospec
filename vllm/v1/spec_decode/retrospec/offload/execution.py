# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.triton_utils import triton

from ..workspace import (
    EXACT_ATTENTION_PARTITION_SIZE,
    exact_attention_source_token_capacity,
)
from .execution_kernels import (
    _accumulate_exact_partition_wave_kernel,
    _multi_source_exact_partition_kernel,
    _parallel_cluster_prefix_kernel,
    _parallel_native_suffix_kernel,
    _ranked_draft_attention_partition_kernel,
    _reduce_exact_partitions_kernel,
    _reduce_proposal_partitions_kernel,
)

_EXACT_ATTENTION_BLOCK_TOKENS = 64
_PARALLEL_FULL_BLOCK_QUERIES = 16
_PARALLEL_FULL_BLOCK_TOKENS = 64
_PARALLEL_FULL_NATIVE_BLOCK_TOKENS = 32
_PARALLEL_FULL_NUM_SPLITS = 8


@dataclass(frozen=True)
class RetroSpecExactPrimaryKVSource:
    """Primary exact KV and the logical-token metadata used to address it."""

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    token_indices: torch.Tensor
    token_mask: torch.Tensor
    ready_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class RetroSpecExactPageKVSource:
    """One resolved GPU page source for clustered exact KV."""

    key_pages: torch.Tensor
    value_pages: torch.Tensor
    page_ids: torch.Tensor
    ready_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class RetroSpecCompactExactPageTable:
    """Valid row lengths for a compact exact-page table."""

    page_counts: torch.Tensor


@dataclass(frozen=True)
class RetroSpecExactKVSource:
    """GPU-visible descriptors for native, resident and staging exact KV."""

    primary: RetroSpecExactPrimaryKVSource
    page_token_counts: torch.Tensor
    resident_pages: RetroSpecExactPageKVSource | None = None
    staging_pages: RetroSpecExactPageKVSource | None = None
    plan_row_indices: torch.Tensor | None = None
    compact_pages: RetroSpecCompactExactPageTable | None = None


@dataclass(frozen=True)
class RetroSpecEstimationKVSource:
    """Weighted cluster summaries consumed by fused proposal attention."""

    keys: torch.Tensor
    values: torch.Tensor
    token_counts: torch.Tensor
    plan_row_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class RetroSpecRankedDraftKVSource:
    """GPU index and resident-table views for one ranked DRAFT selection."""

    primary: RetroSpecExactPrimaryKVSource
    request_slot_ids: torch.Tensor
    ranked_cluster_indices: torch.Tensor
    candidate_counts: torch.Tensor
    resident_bucket_ids: torch.Tensor
    sparse_retrieval_width: int
    sparse_estimation_width: int
    retrieval_ratio: float
    estimation_ratio: float
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_page_starts: torch.Tensor
    cluster_page_counts: torch.Tensor
    page_token_counts: torch.Tensor
    cluster_offsets: torch.Tensor
    page_offsets: torch.Tensor
    resident_table_page_slots: torch.Tensor
    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor


@dataclass(frozen=True)
class RetroSpecCompactKVSource:
    """Token-contiguous clustered KV staged for full verification."""

    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    token_offsets: torch.Tensor
    token_counts: torch.Tensor
    max_tokens_per_head: int
    ready_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class RetroSpecFullVerificationKVSource:
    """Specialized cluster-prefix and native-suffix full-verify sources."""

    primary: RetroSpecExactPrimaryKVSource
    clustered: RetroSpecCompactKVSource | None = None


class RetroSpecExactAttentionWorkspace:
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

    def _validate_source(
        self,
        source: RetroSpecExactKVSource,
        query: torch.Tensor,
        request_indices: torch.Tensor | None,
    ) -> tuple[int, int, int]:
        if query.ndim != 3:
            raise ValueError("Query must have shape [queries, query_heads, head_size]")
        if query.device.type != "cuda":
            raise ValueError("Triton exact attention requires CUDA tensors")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Triton exact attention requires FP16 or BF16 queries")
        if query.shape[0] > self.max_num_queries:
            raise ValueError("Query count exceeds exact-attention workspace capacity")

        primary = source.primary
        if primary.key_cache.shape != primary.value_cache.shape:
            raise ValueError("Primary key and value cache shapes must match")
        if primary.key_cache.ndim != 4:
            raise ValueError(
                "Primary KV cache must have shape "
                "[blocks, block_size, kv_heads, head_size]"
            )
        if primary.key_cache.shape[1] != self.page_size:
            raise ValueError("Primary KV block size does not match page size")
        if primary.key_cache.dtype != query.dtype:
            raise ValueError("Primary KV dtype does not match query")
        if primary.token_indices.shape != primary.token_mask.shape:
            raise ValueError("Primary token indices and mask shapes must match")
        if primary.token_indices.ndim != 3:
            raise ValueError(
                "Primary token metadata must have shape [batch, kv_heads, tokens]"
            )
        if primary.token_mask.dtype != torch.bool:
            raise ValueError("Primary token mask must be boolean")
        compact_pages = source.compact_pages
        if compact_pages is None:
            if source.page_token_counts.ndim != 4:
                raise ValueError(
                    "Padded page metadata must have shape "
                    "[batch, kv_heads, clusters, pages]"
                )
        elif source.page_token_counts.ndim != 3:
            raise ValueError(
                "Compact page metadata must have shape [batch, kv_heads, pages]"
            )
        if primary.block_table.ndim != 2:
            raise ValueError("Block table must have shape [batch, blocks]")

        metadata_rows, num_kv_heads, max_primary_tokens = primary.token_indices.shape
        plan_row_indices = source.plan_row_indices
        resolved_rows = metadata_rows if plan_row_indices is None else query.shape[0]
        expected_page_prefix = (
            (metadata_rows, num_kv_heads)
            if compact_pages is None
            else (resolved_rows, num_kv_heads)
        )
        if source.page_token_counts.shape[:2] != expected_page_prefix:
            raise ValueError("Exact page metadata uses the wrong row namespace")
        if plan_row_indices is None and primary.block_table.shape[0] != metadata_rows:
            raise ValueError("Block table batch size does not match exact metadata")
        if primary.key_cache.shape[2:] != (num_kv_heads, query.shape[2]):
            raise ValueError("Primary KV shape does not match query metadata")
        if query.shape[1] % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        integral_tensors = (
            primary.block_table,
            primary.token_indices,
            source.page_token_counts,
        )
        if any(
            tensor.dtype not in (torch.int32, torch.int64)
            for tensor in integral_tensors
        ):
            raise ValueError("Exact-attention metadata must be integral")

        tensors = [
            primary.key_cache,
            primary.value_cache,
            primary.block_table,
            primary.token_indices,
            primary.token_mask,
            source.page_token_counts,
        ]
        if compact_pages is not None:
            if compact_pages.page_counts.shape != expected_page_prefix:
                raise ValueError("Compact page counts do not match page metadata")
            if compact_pages.page_counts.dtype not in (torch.int32, torch.int64):
                raise ValueError("Compact page counts must be integral")
            tensors.append(compact_pages.page_counts)
        if plan_row_indices is not None:
            if plan_row_indices.shape != (query.shape[0],):
                raise ValueError("Plan rows must contain one entry per query")
            if plan_row_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("Plan rows must be integral")
            tensors.append(plan_row_indices)
        if any(tensor.device != query.device for tensor in tensors):
            raise ValueError("All exact-attention tensors must be on the query device")

        if request_indices is None:
            if query.shape[0] != primary.block_table.shape[0]:
                raise ValueError(
                    "Identity request mapping requires one query per request"
                )
        else:
            if request_indices.shape != (query.shape[0],):
                raise ValueError("request_indices must contain one entry per query")
            if request_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("request_indices must be integral")
            if request_indices.device != query.device:
                raise ValueError("request_indices must be on the query device")

        max_page_slots = source.page_token_counts.shape[2]
        if compact_pages is None:
            max_page_slots *= source.page_token_counts.shape[3]
        if (
            max_page_slots
            and source.resident_pages is None
            and source.staging_pages is None
        ):
            raise RuntimeError(
                "Exact page metadata requires a resident or staging source"
            )

        expected_page_shape = source.page_token_counts.shape
        if plan_row_indices is not None and compact_pages is None:
            expected_page_shape = (
                query.shape[0],
                *source.page_token_counts.shape[1:],
            )
        for page_source in (source.resident_pages, source.staging_pages):
            if page_source is None:
                continue
            if page_source.page_ids.shape != expected_page_shape:
                raise ValueError("Exact page IDs and token-count shapes must match")
            if page_source.page_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Exact page IDs must be integral")
            if page_source.key_pages.shape != page_source.value_pages.shape:
                raise ValueError("Exact key and value page shapes must match")
            if page_source.key_pages.ndim != 3:
                raise ValueError(
                    "Exact pages must have shape [pages, page_size, head_size]"
                )
            if page_source.key_pages.shape[1:] != (self.page_size, query.shape[2]):
                raise ValueError("Exact page shape does not match query")
            if (
                page_source.key_pages.dtype != query.dtype
                or page_source.value_pages.dtype != query.dtype
            ):
                raise ValueError("Exact page dtype does not match query")
            if any(
                tensor.device != query.device
                for tensor in (
                    page_source.page_ids,
                    page_source.key_pages,
                    page_source.value_pages,
                )
            ):
                raise ValueError("Exact pages must be on the query device")

        return num_kv_heads, max_primary_tokens, max_page_slots

    def _validate_estimation_source(
        self,
        source: RetroSpecEstimationKVSource,
        query: torch.Tensor,
        num_kv_heads: int,
    ) -> int:
        if source.keys.shape != source.values.shape:
            raise ValueError("Estimation key and value shapes must match")
        if source.keys.ndim != 4:
            raise ValueError(
                "Estimation KV must have shape [rows, num_kv_heads, vectors, head_size]"
            )
        if source.token_counts.shape != source.keys.shape[:3]:
            raise ValueError("Estimation token counts do not match estimation KV")
        if source.keys.shape[1] != num_kv_heads:
            raise ValueError("Estimation KV-head count does not match exact KV")
        if source.keys.shape[3] != query.shape[2]:
            raise ValueError("Estimation head size does not match query")
        if query.shape[1] % num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads")

        if source.keys.dtype != source.values.dtype:
            raise ValueError("Estimation keys and values must have the same dtype")
        if source.keys.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Estimation KV must use a floating-point dtype")
        if source.token_counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("Estimation token counts must be integral")

        plan_row_indices = source.plan_row_indices
        if plan_row_indices is None:
            if source.keys.shape[0] != query.shape[0]:
                raise ValueError("Estimation rows must match query rows")
        else:
            if plan_row_indices.shape != (query.shape[0],):
                raise ValueError("Plan rows must contain one entry per query")
            if plan_row_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("Plan rows must be integral")

        tensors = [source.keys, source.values, source.token_counts]
        if plan_row_indices is not None:
            tensors.append(plan_row_indices)
        if any(tensor.device != query.device for tensor in tensors):
            raise ValueError("Estimation tensors must use the query device")

        return source.keys.shape[2]

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

    def _validate_ranked_draft_source(
        self,
        source: RetroSpecRankedDraftKVSource,
        query: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        if query.ndim != 3 or query.device.type != "cuda":
            raise ValueError("Ranked DRAFT attention requires CUDA query rows")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Ranked DRAFT attention requires FP16 or BF16")
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("Ranked DRAFT output must match query")
        if output.device != query.device:
            raise ValueError("Ranked DRAFT output must use the query device")
        if query.shape[0] > self.max_num_queries:
            raise ValueError("Query count exceeds exact-attention capacity")

        primary = source.primary
        if primary.key_cache.shape != primary.value_cache.shape:
            raise ValueError("Primary key and value cache shapes must match")
        if primary.key_cache.ndim != 4:
            raise ValueError("Primary KV cache must have four dimensions")
        if primary.key_cache.dtype != query.dtype:
            raise ValueError("Primary KV dtype must match query")
        if primary.key_cache.shape[1] != self.page_size:
            raise ValueError("Primary KV block size does not match page size")
        if primary.token_indices.shape != primary.token_mask.shape:
            raise ValueError("Primary token descriptors must have equal shapes")
        if primary.token_indices.ndim != 3:
            raise ValueError("Primary token descriptors must be three-dimensional")
        if primary.token_mask.dtype != torch.bool:
            raise ValueError("Primary token mask must be boolean")

        batch_size, num_kv_heads, max_primary_tokens = primary.token_indices.shape
        if batch_size != query.shape[0]:
            raise ValueError("Ranked DRAFT requires one query per request")
        if query.shape[1] % num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads")
        if primary.key_cache.shape[2:] != (num_kv_heads, query.shape[2]):
            raise ValueError("Primary KV shape does not match query")
        if primary.block_table.shape[0] != batch_size:
            raise ValueError("Block table batch does not match query")

        ranked_shape = source.ranked_cluster_indices.shape
        retrieval_shape = (
            batch_size,
            num_kv_heads,
            source.sparse_retrieval_width,
        )
        if source.resident_bucket_ids.shape != retrieval_shape:
            raise ValueError("Resident buckets do not match retrieval clusters")
        if ranked_shape[:2] != (batch_size, num_kv_heads):
            raise ValueError("Ranked journal rows do not match query")
        if source.candidate_counts.shape != (batch_size, num_kv_heads):
            raise ValueError("Candidate counts do not match query")
        if ranked_shape[2] < (
            source.sparse_retrieval_width + source.sparse_estimation_width
        ):
            raise ValueError("Ranked journal is narrower than DRAFT selection")
        if source.request_slot_ids.shape != (batch_size,):
            raise ValueError("Request slots do not match query")
        if source.ranked_cluster_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Ranked cluster indices must be integral")
        if source.candidate_counts.dtype != torch.int32:
            raise ValueError("Candidate counts must use int32")
        if source.resident_bucket_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Resident buckets must be integral")

        if source.cluster_keys.shape != source.cluster_values.shape:
            raise ValueError("Cluster key and value summaries must match")
        if source.cluster_keys.dtype != query.dtype:
            raise ValueError("Cluster summary dtype must match query")
        if source.cluster_keys.ndim != 3:
            raise ValueError("Cluster summaries must have three dimensions")
        if source.cluster_keys.shape[0] != num_kv_heads:
            raise ValueError("Cluster summary KV heads do not match query")
        if source.cluster_keys.shape[2] != query.shape[2]:
            raise ValueError("Cluster summary head size does not match query")
        cluster_shape = source.cluster_keys.shape[:2]
        for tensor in (
            source.cluster_token_counts,
            source.cluster_page_starts,
            source.cluster_page_counts,
        ):
            if tensor.shape != cluster_shape:
                raise ValueError("Cluster descriptor shape is inconsistent")
        if source.page_token_counts.ndim != 2:
            raise ValueError("Page token counts must have two dimensions")
        if source.page_token_counts.shape[0] != num_kv_heads:
            raise ValueError("Page descriptors do not match KV heads")
        if source.cluster_offsets.ndim != 1 or source.page_offsets.ndim != 1:
            raise ValueError("Request offsets must be one-dimensional")
        if source.resident_table_page_slots.ndim != 2:
            raise ValueError("Resident page table must have two dimensions")
        if source.resident_key_pages.shape != source.resident_value_pages.shape:
            raise ValueError("Resident key and value pages must match")
        if source.resident_key_pages.ndim != 3:
            raise ValueError("Resident pages must have three dimensions")
        if source.resident_key_pages.dtype != query.dtype:
            raise ValueError("Resident page dtype must match query")
        if source.resident_key_pages.shape[1:] != (
            self.page_size,
            query.shape[2],
        ):
            raise ValueError("Resident page shape does not match query")

        tensors = (
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
        )
        if any(tensor.device != query.device for tensor in tensors):
            raise ValueError("Ranked DRAFT tensors must use the query device")
        return (
            num_kv_heads,
            max_primary_tokens,
            source.sparse_retrieval_width,
            source.sparse_estimation_width,
        )

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


# Preserve the original qualified path for serialized classes.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = __name__.replace(".offload.", ".")
del _legacy_type
