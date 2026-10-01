# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from math import ceil

import torch

from vllm.v1.spec_decode.retrospec.constants import RESIDENT_CLUSTER_SCORE_TILE_SIZE
from vllm.v1.spec_decode.retrospec.legacy.cluster_scoring import (
    reduce_grouped_cluster_scores,
    score_resident_clusters,
)
from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecResidentBatchView,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.types import (
    _ClusterSelectionWorkspace,
    _PackedClusterZones,
)


class _RetroSpecSegmentedTokenIndexScoringMixin:
    def _validate_resident_index(
        self,
        layer_name: str,
        request_ids: Sequence[str],
    ) -> None:
        layer_indices = self._indices.get(layer_name, {})
        for request_id in request_ids:
            record = layer_indices.get(request_id)
            expected_num_clusters = 0 if record is None else record.num_clusters
            resident_num_clusters = self._gpu_index_residency.get_num_clusters(
                layer_name, request_id
            )
            if resident_num_clusters != expected_num_clusters:
                raise RuntimeError(
                    "CPU and GPU RetroSpec cluster counts are inconsistent"
                )

            if record is not None and record.segments:
                resident_indexed_end = self._gpu_index_residency.get_indexed_end(
                    layer_name, request_id
                )
                if resident_indexed_end != record.indexed_end:
                    raise RuntimeError(
                        "CPU and GPU RetroSpec indexed prefixes are inconsistent"
                    )

    def _get_resident_view(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        key_cache: torch.Tensor,
    ) -> RetroSpecResidentBatchView:
        request_ids = tuple(request_ids)
        self._validate_resident_index(layer_name, request_ids)

        view = self._gpu_index_residency.get_active_view(
            layer_name, request_ids, key_cache.device
        )
        arena = view.arena
        if arena is not None:
            if arena.cluster_keys.device != key_cache.device:
                raise RuntimeError(
                    "Resident RetroSpec arena and attention KV use different devices"
                )
            if arena.cluster_keys.dtype != key_cache.dtype:
                raise RuntimeError(
                    "Resident RetroSpec arena and attention KV use different dtypes"
                )
            if arena.cluster_keys.shape[0] != key_cache.shape[2]:
                raise RuntimeError("Resident RetroSpec arena changed KV-head count")
            if arena.cluster_keys.shape[2] != key_cache.shape[3]:
                raise RuntimeError("Resident RetroSpec arena changed head size")
        return view

    @staticmethod
    def _get_resident_indexed_bounds(
        view: RetroSpecResidentBatchView,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = view.request_slot_ids.shape[0]
        if view.arena is None:
            zeros = torch.zeros(batch_size, dtype=torch.int64, device=device)
            valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return zeros, zeros, valid

        valid_requests = view.request_slot_ids >= 0
        safe_slots = view.request_slot_ids.clamp_min(0)
        indexed_starts = view.arena.indexed_starts.index_select(0, safe_slots)
        indexed_ends = view.arena.indexed_ends.index_select(0, safe_slots)
        return indexed_starts, indexed_ends, valid_requests

    def _build_token_layout(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_num_tokens = block_table.shape[1] * self.block_size
        logical_token_ids = torch.arange(
            max_num_tokens,
            dtype=torch.int64,
            device=block_table.device,
        )

        bounded_seq_lens = seq_lens.to(torch.int64).clamp(
            min=1,
            max=max_num_tokens,
        )
        valid_token_mask = logical_token_ids.unsqueeze(0) < bounded_seq_lens.unsqueeze(
            1
        )

        valid_block_counts = torch.div(
            bounded_seq_lens + self.block_size - 1,
            self.block_size,
            rounding_mode="floor",
        )
        recent_start_blocks = (valid_block_counts - self.num_recent_blocks).clamp_min(0)

        logical_block_ids = torch.div(
            logical_token_ids,
            self.block_size,
            rounding_mode="floor",
        )

        forced_exact_mask = valid_token_mask & (
            (logical_block_ids.unsqueeze(0) == 0)
            | (logical_block_ids.unsqueeze(0) >= recent_start_blocks.unsqueeze(1))
        )

        return logical_token_ids, valid_token_mask, forced_exact_mask

    @staticmethod
    def _compute_cluster_logits(
        query: torch.Tensor,
        cluster_keys: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute raw grouped query-centroid dot products."""
        if query.ndim != 3:
            raise ValueError(
                "query must have shape [batch, num_query_heads, head_size]"
            )
        if cluster_keys.ndim != 4:
            raise ValueError(
                "cluster_keys must have shape "
                "[batch, num_kv_heads, num_clusters, head_size]"
            )

        batch_size, num_query_heads, head_size = query.shape
        (
            key_batch_size,
            num_kv_heads,
            num_clusters,
            key_head_size,
        ) = cluster_keys.shape

        if key_batch_size != batch_size:
            raise ValueError("Cluster key batch size does not match query")
        if key_head_size != head_size:
            raise ValueError("Cluster key head size does not match query")
        if num_kv_heads <= 0:
            raise ValueError("Cluster keys must contain at least one KV head")
        if num_clusters <= 0:
            raise ValueError("Cluster keys must contain at least one cluster slot")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )
        if cluster_keys.device != query.device:
            raise ValueError("Query and cluster keys must be on one device")

        queries_per_kv = num_query_heads // num_kv_heads
        output_shape = (
            batch_size,
            num_kv_heads,
            queries_per_kv,
            num_clusters,
        )

        if output is not None:
            if output.shape != output_shape:
                raise ValueError("Cluster-logit output has an unexpected shape")
            if output.dtype != torch.float32:
                raise ValueError("Cluster-logit output must use float32")
            if output.device != query.device:
                raise ValueError("Cluster-logit output must be on the query device")
            if not output.is_contiguous():
                raise ValueError("Cluster-logit output must be contiguous")

        grouped_query = query.reshape(
            batch_size * num_kv_heads,
            queries_per_kv,
            head_size,
        )
        flattened_cluster_keys = cluster_keys.reshape(
            batch_size * num_kv_heads,
            num_clusters,
            head_size,
        )
        transposed_cluster_keys = flattened_cluster_keys.transpose(1, 2)

        flat_output = None
        if output is not None:
            flat_output = output.view(
                batch_size * num_kv_heads,
                queries_per_kv,
                num_clusters,
            )

        can_use_tensor_core_bmm = (
            query.device.type == "cuda"
            and query.dtype == cluster_keys.dtype
            and query.dtype in (torch.float16, torch.bfloat16)
        )

        if can_use_tensor_core_bmm:
            if flat_output is None:
                flat_logits = torch.bmm(
                    grouped_query,
                    transposed_cluster_keys,
                    out_dtype=torch.float32,
                )
            else:
                torch.bmm(
                    grouped_query,
                    transposed_cluster_keys,
                    out_dtype=torch.float32,
                    out=flat_output,
                )
                flat_logits = flat_output
        else:
            grouped_query_float = grouped_query.float()
            cluster_keys_float = transposed_cluster_keys.float()

            if flat_output is None:
                flat_logits = torch.bmm(
                    grouped_query_float,
                    cluster_keys_float,
                )
            else:
                torch.bmm(
                    grouped_query_float,
                    cluster_keys_float,
                    out=flat_output,
                )
                flat_logits = flat_output

        if output is not None:
            return output
        return flat_logits.view(output_shape)

    @staticmethod
    def _reduce_cluster_scores_reference(
        logits: torch.Tensor,
        cluster_mask: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Reference grouped softmax and GQA probability reduction."""
        logits.mul_(scale)
        logits.add_(torch.log(cluster_token_counts.clamp_min(1).float()).unsqueeze(2))
        logits.masked_fill_(
            ~cluster_mask.unsqueeze(2),
            float("-inf"),
        )

        has_clusters = cluster_mask.any(dim=2)
        safe_logits = torch.where(
            has_clusters[:, :, None, None],
            logits,
            torch.zeros_like(logits),
        )

        probabilities = torch.softmax(
            safe_logits,
            dim=3,
        )
        probabilities.masked_fill_(
            ~cluster_mask.unsqueeze(2),
            0.0,
        )

        return probabilities.mean(dim=2)

    @classmethod
    def _score_clusters(
        cls,
        query: torch.Tensor,
        cluster_keys: torch.Tensor,
        cluster_mask: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Score packed clusters for reference and non-resident execution."""
        logits = cls._compute_cluster_logits(query, cluster_keys)

        if logits.device.type == "cuda":
            return reduce_grouped_cluster_scores(
                logits,
                cluster_mask,
                cluster_token_counts,
                scale,
            )

        return cls._reduce_cluster_scores_reference(
            logits,
            cluster_mask,
            cluster_token_counts,
            scale,
        )

    def _maximum_zone_widths(
        self,
        num_clusters: int,
    ) -> tuple[int, int, int]:
        """Return synchronization-free capacities for cluster zones."""
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")

        max_retrieval = min(
            ceil(num_clusters * self.retrieval_ratio),
            num_clusters,
        )
        max_estimation = min(
            ceil(num_clusters * self.estimation_ratio),
            num_clusters - max_retrieval,
        )
        max_total_compute = max_retrieval + max_estimation
        max_expanded_retrieval = min(
            max_retrieval * 2,
            max_total_compute,
        )

        return (
            max_retrieval,
            max_estimation,
            max_expanded_retrieval,
        )

    def _maximum_prefill_warmup_width(self, num_clusters: int) -> int:
        max_retrieval, _, _ = self._maximum_zone_widths(num_clusters)
        return min(
            max_retrieval * self.prefill_warmup_multiplier,
            num_clusters,
        )

    def _get_cluster_selection_workspace(
        self,
        query: torch.Tensor,
        num_kv_heads: int,
        num_clusters: int,
        prefill_hint: bool = False,
    ) -> _ClusterSelectionWorkspace:
        """Return a reusable CUDA workspace for cluster selection."""
        if query.device.type != "cuda":
            raise ValueError("Cluster selection workspace requires CUDA")

        batch_size, num_query_heads, _ = query.shape
        if num_kv_heads <= 0:
            raise ValueError("Cluster keys must contain at least one KV head")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        queries_per_kv = num_query_heads // num_kv_heads
        (
            max_retrieval,
            max_estimation,
            _,
        ) = self._maximum_zone_widths(num_clusters)
        max_total_compute = max_retrieval + max_estimation
        max_ranked_clusters = (
            self._maximum_prefill_warmup_width(num_clusters)
            if prefill_hint
            else max_total_compute
        )

        num_tiles = (
            num_clusters + RESIDENT_CLUSTER_SCORE_TILE_SIZE - 1
        ) // RESIDENT_CLUSTER_SCORE_TILE_SIZE
        scores_shape = (
            batch_size,
            num_kv_heads,
            num_clusters,
        )
        lse_shape = (
            batch_size,
            num_kv_heads,
            queries_per_kv,
        )
        tile_shape = (*lse_shape, num_tiles)
        tile_count_shape = (batch_size, num_kv_heads, num_tiles)
        topk_shape = (
            batch_size,
            num_kv_heads,
            max_ranked_clusters,
        )

        workspace = (
            self._prefill_hint_selection_workspace
            if prefill_hint
            else self._cluster_selection_workspace
        )
        if (
            workspace is not None
            and workspace.scores.device == query.device
            and workspace.scores.shape == scores_shape
            and workspace.softmax_lse.shape == lse_shape
            and workspace.tile_max.shape == tile_shape
            and workspace.tile_sum.shape == tile_shape
            and workspace.tile_candidate_counts.shape == tile_count_shape
            and workspace.candidate_counts.shape == scores_shape[:2]
            and workspace.topk_values.shape == topk_shape
            and workspace.topk_indices.shape == topk_shape
        ):
            return workspace

        workspace = _ClusterSelectionWorkspace(
            scores=torch.empty(
                scores_shape,
                dtype=torch.float32,
                device=query.device,
            ),
            softmax_lse=torch.empty(
                lse_shape,
                dtype=torch.float32,
                device=query.device,
            ),
            tile_max=torch.empty(
                tile_shape,
                dtype=torch.float32,
                device=query.device,
            ),
            tile_sum=torch.empty(tile_shape, dtype=torch.float32, device=query.device),
            tile_candidate_counts=torch.empty(
                tile_count_shape, dtype=torch.int32, device=query.device
            ),
            candidate_counts=torch.empty(
                scores_shape[:2],
                dtype=torch.int32,
                device=query.device,
            ),
            topk_values=torch.empty(
                topk_shape,
                dtype=torch.float32,
                device=query.device,
            ),
            topk_indices=torch.empty(
                topk_shape,
                dtype=torch.int64,
                device=query.device,
            ),
        )
        if prefill_hint:
            self._prefill_hint_selection_workspace = workspace
        else:
            self._cluster_selection_workspace = workspace
        return workspace

    def _score_resident_view(
        self,
        query: torch.Tensor,
        view: RetroSpecResidentBatchView,
        scale: float,
        num_kv_heads: int,
        prefill_hint: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        _ClusterSelectionWorkspace | None,
    ]:
        workspace = None
        if query.device.type == "cuda":
            workspace = self._get_cluster_selection_workspace(
                query,
                num_kv_heads,
                view.max_num_clusters,
                prefill_hint=prefill_hint,
            )

        arena = view.arena
        if arena is None:
            shape = (query.shape[0], num_kv_heads, view.max_num_clusters)
            if workspace is None:
                scores = torch.full(
                    shape, float("-inf"), dtype=torch.float32, device=query.device
                )
                candidate_counts = torch.zeros(
                    shape[:2], dtype=torch.int32, device=query.device
                )
            else:
                scores = workspace.scores.fill_(float("-inf"))
                candidate_counts = workspace.candidate_counts.zero_()
            return scores, candidate_counts, workspace

        if workspace is not None:
            scores = score_resident_clusters(
                query=query,
                cluster_keys=arena.cluster_keys,
                cluster_ids=arena.cluster_ids,
                cluster_token_counts=arena.cluster_token_counts,
                cluster_offsets=arena.cluster_offsets,
                num_clusters=arena.num_clusters,
                request_slot_ids=view.request_slot_ids,
                scale=scale,
                output=workspace.scores,
                softmax_lse=workspace.softmax_lse,
                tile_max=workspace.tile_max,
                tile_sum=workspace.tile_sum,
                tile_candidate_counts=workspace.tile_candidate_counts,
                candidate_counts=workspace.candidate_counts,
            )
            return scores, workspace.candidate_counts, workspace

        safe_slots = view.request_slot_ids.clamp_min(0)
        request_num_clusters = arena.num_clusters.index_select(0, safe_slots)
        request_cluster_offsets = arena.cluster_offsets.index_select(0, safe_slots)
        local_cluster_indices = torch.arange(
            view.max_num_clusters, dtype=torch.int64, device=query.device
        )
        absolute_cluster_indices = (
            request_cluster_offsets[:, None, None]
            + local_cluster_indices[None, None, :]
        )
        absolute_cluster_indices.clamp_(min=0, max=arena.cluster_ids.shape[1] - 1)
        head_indices = torch.arange(
            num_kv_heads, dtype=torch.int64, device=query.device
        )[None, :, None]
        packed_keys = arena.cluster_keys[head_indices, absolute_cluster_indices]
        packed_ids = arena.cluster_ids[head_indices, absolute_cluster_indices]
        packed_counts = arena.cluster_token_counts[
            head_indices, absolute_cluster_indices
        ]
        cluster_mask = (
            (view.request_slot_ids >= 0)[:, None, None]
            & (
                local_cluster_indices[None, None, :]
                < request_num_clusters[:, None, None]
            )
            & (packed_ids >= 0)
            & (packed_counts > 0)
        )
        scores = self._score_clusters(
            query, packed_keys, cluster_mask, packed_counts, scale
        )
        scores = scores.masked_fill(~cluster_mask, float("-inf"))
        candidate_counts = cluster_mask.sum(dim=2, dtype=torch.int32)
        return scores, candidate_counts, None

    def _rank_cluster_scores(
        self,
        cluster_scores: torch.Tensor,
        workspace: _ClusterSelectionWorkspace,
        ranking_width: int | None = None,
        output_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank clusters and optionally publish into a persistent journal."""
        if cluster_scores is not workspace.scores:
            raise ValueError("Workspace scores do not match cluster scores")

        num_clusters = cluster_scores.shape[2]
        if ranking_width is None:
            max_retrieval, max_estimation, _ = self._maximum_zone_widths(num_clusters)
            ranking_width = max_retrieval + max_estimation
        if ranking_width <= 0 or ranking_width > num_clusters:
            raise ValueError("Cluster ranking width is outside the valid range")
        ranked_values = workspace.topk_values[:, :, :ranking_width]
        if output_indices is None:
            if workspace.topk_indices.shape[2] < ranking_width:
                raise ValueError("Workspace top-k capacity is too small")
            ranked_indices = workspace.topk_indices[:, :, :ranking_width]
        else:
            expected_shape = (*cluster_scores.shape[:2], ranking_width)
            if output_indices.shape != expected_shape:
                raise ValueError("Persistent ranked journal has the wrong shape")
            if output_indices.dtype != torch.int64:
                raise ValueError("Persistent ranked journal must use int64")
            if output_indices.device != cluster_scores.device:
                raise ValueError("Persistent ranked journal uses the wrong device")
            ranked_indices = output_indices
        torch.topk(
            workspace.scores,
            k=ranking_width,
            dim=2,
            largest=True,
            sorted=True,
            out=(ranked_values, ranked_indices),
        )
        return ranked_values, ranked_indices

    @staticmethod
    def _slice_rank_range(
        ranked_indices: torch.Tensor,
        start_counts: torch.Tensor,
        end_counts: torch.Tensor,
        output_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a variable rank interval into a fixed-width tensor."""
        if output_width < 0:
            raise ValueError("output_width must be non-negative")

        if output_width == 0:
            return (
                ranked_indices[..., :0].contiguous(),
                torch.empty_like(
                    ranked_indices[..., :0],
                    dtype=torch.bool,
                ),
            )

        rank_offsets = torch.arange(
            output_width,
            dtype=torch.int64,
            device=ranked_indices.device,
        )
        rank_positions = start_counts.unsqueeze(-1) + rank_offsets

        selected_mask = rank_positions < end_counts.unsqueeze(-1)
        safe_rank_positions = rank_positions.clamp(
            min=0,
            max=ranked_indices.shape[-1] - 1,
        )
        selected_indices = ranked_indices.gather(
            dim=2,
            index=safe_rank_positions,
        )

        return (
            selected_indices.contiguous(),
            selected_mask.contiguous(),
        )

    def _select_prefill_warmup(
        self,
        ranked_indices: torch.Tensor,
        candidate_counts: torch.Tensor,
        view: RetroSpecResidentBatchView,
        active_mask: torch.Tensor,
        warmup_page_budgets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a bounded final-prefill resident-cache seed."""
        batch_size, num_kv_heads = candidate_counts.shape
        if active_mask.shape != (batch_size,):
            raise ValueError("Prefill warmup mask has an unexpected shape")
        if warmup_page_budgets.shape != (batch_size, num_kv_heads):
            raise ValueError("Prefill warmup page budgets have an unexpected shape")
        if active_mask.device != ranked_indices.device:
            raise ValueError("Prefill warmup mask must use the rank device")
        if warmup_page_budgets.device != ranked_indices.device:
            raise ValueError("Prefill warmup page budgets must use the rank device")

        num_clusters = view.max_num_clusters
        max_retrieval, _, _ = self._maximum_zone_widths(num_clusters)
        max_warmup = self._maximum_prefill_warmup_width(num_clusters)
        counts = candidate_counts.to(torch.int64)
        retrieval_counts = torch.ceil(counts.float() * self.retrieval_ratio).to(
            torch.int64
        )
        retrieval_counts = torch.minimum(retrieval_counts, counts)
        warmup_counts = torch.minimum(
            retrieval_counts * self.prefill_warmup_multiplier,
            counts,
        )
        zero_counts = torch.zeros_like(retrieval_counts)
        warmup_indices, warmup_mask = self._slice_rank_range(
            ranked_indices,
            zero_counts,
            warmup_counts,
            max_warmup,
        )

        if view.arena is None or max_warmup == 0 or max_retrieval == 0:
            warmup_mask.zero_()
            return warmup_indices, warmup_mask

        arena = view.arena
        safe_slots = view.request_slot_ids.clamp_min(0)
        request_cluster_offsets = arena.cluster_offsets.index_select(0, safe_slots)
        local_cluster_indices = torch.arange(
            view.max_num_clusters,
            dtype=torch.int64,
            device=view.request_slot_ids.device,
        )
        absolute_cluster_indices = (
            request_cluster_offsets[:, None, None]
            + local_cluster_indices[None, None, :]
        )
        absolute_cluster_indices.clamp_(
            min=0, max=arena.cluster_page_counts.shape[1] - 1
        )
        head_indices = torch.arange(
            arena.cluster_page_counts.shape[0],
            dtype=torch.int64,
            device=view.request_slot_ids.device,
        )[None, :, None]
        cluster_page_counts = arena.cluster_page_counts[
            head_indices, absolute_cluster_indices
        ]
        ranked_page_counts = cluster_page_counts.gather(2, warmup_indices)
        cumulative_pages = ranked_page_counts.cumsum(dim=2)
        warmup_mask &= cumulative_pages <= warmup_page_budgets.unsqueeze(-1)
        warmup_mask &= active_mask[:, None, None]
        warmup_mask &= view.request_slot_ids[:, None, None] >= 0
        return warmup_indices, warmup_mask

    def _get_prefill_hint_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        stream = self._prefill_hint_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._prefill_hint_streams[device] = stream
        return stream

    def _get_prefill_hint_view(
        self,
        layer_name: str,
        request_ids: tuple[str, ...],
        query: torch.Tensor,
    ) -> RetroSpecResidentBatchView:
        self._validate_resident_index(layer_name, request_ids)
        view = self._gpu_index_residency.get_active_view(
            layer_name,
            request_ids,
            query.device,
        )
        arena = view.arena
        if arena is None:
            return view
        if arena.cluster_keys.device != query.device:
            raise RuntimeError("Prefill hint and cluster index use different devices")
        if arena.cluster_keys.dtype != query.dtype:
            raise RuntimeError("Prefill hint and cluster index use different dtypes")
        if arena.cluster_keys.shape[2] != query.shape[2]:
            raise RuntimeError("Prefill hint and cluster index head sizes differ")
        return view

    @staticmethod
    def _gather_ranked_cluster_handles(
        view: RetroSpecResidentBatchView,
        ranked_indices: torch.Tensor,
        ranked_mask: torch.Tensor,
    ) -> torch.Tensor:
        if ranked_indices.shape != ranked_mask.shape:
            raise ValueError("Ranked cluster indices and mask must match")

        output = torch.full(
            ranked_indices.shape,
            -1,
            dtype=torch.int64,
            device=ranked_indices.device,
        )
        if view.arena is None or ranked_indices.shape[2] == 0:
            return output

        arena = view.arena
        safe_slots = view.request_slot_ids.clamp_min(0)
        request_offsets = arena.cluster_offsets.index_select(0, safe_slots)
        absolute_indices = request_offsets[:, None, None] + ranked_indices
        absolute_indices.clamp_(min=0, max=arena.cluster_ids.shape[1] - 1)
        head_indices = torch.arange(
            ranked_indices.shape[1],
            dtype=torch.int64,
            device=ranked_indices.device,
        )[None, :, None].expand_as(ranked_indices)

        handles = arena.cluster_ids[head_indices, absolute_indices]
        valid = (
            ranked_mask & (view.request_slot_ids[:, None, None] >= 0) & (handles >= 0)
        )
        output.copy_(handles)
        output.masked_fill_(~valid, -1)
        return output

    def _prefetch_final_prefill_query(
        self,
        request_ids: tuple[str, ...],
        layer_name: str,
        query: torch.Tensor,
        scale: float,
    ) -> bool:
        if query.ndim != 3 or query.shape[0] != len(request_ids):
            raise ValueError("Prefill hint query batch does not match request IDs")

        view = self._get_prefill_hint_view(layer_name, request_ids, query)
        arena = view.arena
        if arena is None:
            return False

        num_kv_heads = arena.cluster_keys.shape[0]
        if query.shape[1] % num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads")

        cluster_scores, candidate_counts, workspace = self._score_resident_view(
            query=query,
            view=view,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefill_hint=True,
        )
        if workspace is None:
            return False

        warmup_width = self._maximum_prefill_warmup_width(view.max_num_clusters)
        _, ranked_indices = self._rank_cluster_scores(
            cluster_scores,
            workspace,
            ranking_width=warmup_width,
        )
        group_targets = self.cluster_store.resident_group_target_pages(
            layer_name,
            request_ids,
            num_kv_heads,
        )
        page_budgets = torch.tensor(
            group_targets,
            dtype=torch.int64,
            device=query.device,
        )
        page_budgets = torch.div(
            page_budgets + 1,
            2,
            rounding_mode="floor",
        )
        active_mask = torch.ones(
            len(request_ids),
            dtype=torch.bool,
            device=query.device,
        )
        warmup_indices, warmup_mask = self._select_prefill_warmup(
            ranked_indices=ranked_indices,
            candidate_counts=candidate_counts,
            view=view,
            active_mask=active_mask,
            warmup_page_budgets=page_budgets,
        )
        cluster_handles = self._gather_ranked_cluster_handles(
            view,
            warmup_indices,
            warmup_mask,
        )
        access_kinds = warmup_mask.to(torch.int8)
        access_kinds.mul_(2)

        self.cluster_store.reserve_resident_access_ring(
            query.device,
            cluster_handles.numel(),
        )
        return self.cluster_store.prefetch_resident_clusters(
            layer_name=layer_name,
            cluster_ids=cluster_handles,
            access_kinds=access_kinds,
            source="prefill_hint",
        )

    def prefetch_final_prefill_queries(
        self,
        request_ids: Sequence[str],
        query_hints: Mapping[str, tuple[torch.Tensor, float]],
    ) -> int:
        """Seed resident pages without delaying the first draft step."""
        if not self.cluster_store.pin_memory or not query_hints:
            return 0
        if self._proposal_active:
            raise RuntimeError("Prefill hints cannot start during a proposal")

        request_ids = tuple(request_ids)
        devices = {query.device for query, _ in query_hints.values()}
        if len(devices) != 1:
            raise ValueError("All prefill hints must use one CUDA device")
        device = next(iter(devices))
        if device.type != "cuda":
            return 0

        producer_event = torch.cuda.Event()
        producer_event.record(torch.cuda.current_stream(device))
        hint_stream = self._get_prefill_hint_stream(device)
        submitted = 0

        self._gpu_index_residency.activate(request_ids)
        try:
            with torch.cuda.stream(hint_stream):
                hint_stream.wait_event(producer_event)
                for layer_name, (query, scale) in query_hints.items():
                    query.record_stream(hint_stream)
                    with self._cuda_timer("prefill_hint_selection"):
                        queued = self._prefetch_final_prefill_query(
                            request_ids=request_ids,
                            layer_name=layer_name,
                            query=query,
                            scale=scale,
                        )
                    submitted += int(queued)
        finally:
            self._gpu_index_residency.deactivate()

        if self.performance_stats is not None:
            self.performance_stats.add_counter("prefill_hint_layers", len(query_hints))
            self.performance_stats.add_counter("prefill_hint_records", submitted)
        return submitted

    def _select_cluster_zones(
        self,
        cluster_scores: torch.Tensor,
        candidate_counts: torch.Tensor,
        view: RetroSpecResidentBatchView,
        workspace: _ClusterSelectionWorkspace | None = None,
    ) -> _PackedClusterZones:
        """Rank relevant clusters once and return compact zone indices."""
        if cluster_scores.ndim != 3:
            raise ValueError(
                "Cluster scores must have shape [batch, num_kv_heads, num_clusters]"
            )
        if candidate_counts.shape != cluster_scores.shape[:2]:
            raise ValueError("Candidate counts do not match cluster scores")

        _, _, num_clusters = cluster_scores.shape
        (
            max_retrieval,
            max_estimation,
            max_expanded_retrieval,
        ) = self._maximum_zone_widths(num_clusters)
        max_total_compute = max_retrieval + max_estimation
        ranking_width = max_total_compute

        candidate_counts = candidate_counts.to(torch.int64)

        retrieval_counts = torch.ceil(
            candidate_counts.float() * self.retrieval_ratio
        ).to(torch.int64)
        retrieval_counts = torch.minimum(
            retrieval_counts,
            candidate_counts,
        )

        estimation_counts = torch.ceil(
            candidate_counts.float() * self.estimation_ratio
        ).to(torch.int64)
        estimation_counts = torch.minimum(
            estimation_counts,
            candidate_counts - retrieval_counts,
        )

        total_compute_counts = retrieval_counts + estimation_counts
        expanded_retrieval_counts = torch.minimum(
            retrieval_counts * 2,
            total_compute_counts,
        )

        if workspace is None:
            topk_values, ranked_indices = torch.topk(
                cluster_scores,
                k=ranking_width,
                dim=2,
                largest=True,
                sorted=True,
            )
        else:
            if cluster_scores is not workspace.scores:
                raise ValueError("Workspace scores do not match cluster scores")
            if workspace.topk_indices.shape[2] < ranking_width:
                raise ValueError("Workspace top-k capacity is too small")

            topk_values = workspace.topk_values[:, :, :ranking_width]
            topk_indices = workspace.topk_indices[:, :, :ranking_width]

            torch.topk(
                workspace.scores,
                k=ranking_width,
                dim=2,
                largest=True,
                sorted=True,
                out=(topk_values, topk_indices),
            )
            ranked_indices = topk_indices

        zero_counts = torch.zeros_like(retrieval_counts)

        (
            sparse_retrieval_indices,
            sparse_retrieval_mask,
        ) = self._slice_rank_range(
            ranked_indices,
            zero_counts,
            retrieval_counts,
            max_retrieval,
        )
        (
            sparse_estimation_indices,
            sparse_estimation_mask,
        ) = self._slice_rank_range(
            ranked_indices,
            retrieval_counts,
            total_compute_counts,
            max_estimation,
        )
        (
            expanded_retrieval_indices,
            expanded_retrieval_mask,
        ) = self._slice_rank_range(
            ranked_indices,
            zero_counts,
            expanded_retrieval_counts,
            max_expanded_retrieval,
        )
        sparse_retrieval_scores = topk_values[:, :, :max_retrieval].contiguous()
        sparse_retrieval_scores.masked_fill_(~sparse_retrieval_mask, 0.0)
        expanded_retrieval_scores = topk_values[
            :, :, :max_expanded_retrieval
        ].contiguous()
        expanded_retrieval_scores.masked_fill_(~expanded_retrieval_mask, 0.0)
        (
            expanded_estimation_indices,
            expanded_estimation_mask,
        ) = self._slice_rank_range(
            ranked_indices,
            expanded_retrieval_counts,
            total_compute_counts,
            max_estimation,
        )

        return _PackedClusterZones(
            sparse_retrieval_indices=sparse_retrieval_indices,
            sparse_retrieval_mask=sparse_retrieval_mask,
            sparse_retrieval_scores=sparse_retrieval_scores,
            sparse_estimation_indices=sparse_estimation_indices,
            sparse_estimation_mask=sparse_estimation_mask,
            expanded_retrieval_indices=expanded_retrieval_indices,
            expanded_retrieval_mask=expanded_retrieval_mask,
            expanded_retrieval_scores=expanded_retrieval_scores,
            expanded_estimation_indices=expanded_estimation_indices,
            expanded_estimation_mask=expanded_estimation_mask,
        )
