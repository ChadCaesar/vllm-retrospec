# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.retrospec.legacy.attention.kernels import (
    RetroSpecEstimationKVSource,
    RetroSpecExactKVSource,
    RetroSpecRankedDraftKVSource,
)

_EXACT_ATTENTION_BLOCK_TOKENS = 64
_PARALLEL_FULL_BLOCK_QUERIES = 16
_PARALLEL_FULL_BLOCK_TOKENS = 64
_PARALLEL_FULL_NATIVE_BLOCK_TOKENS = 32
_PARALLEL_FULL_NUM_SPLITS = 8


class RetroSpecExecutionValidationMixin:
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
