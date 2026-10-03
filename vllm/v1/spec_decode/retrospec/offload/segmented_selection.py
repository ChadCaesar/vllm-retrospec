# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from math import ceil, prod

import torch

from ..workspace import exact_attention_primary_token_capacity
from .cluster_scoring import (
    RESIDENT_CLUSTER_SCORE_TILE_SIZE,
    reduce_grouped_cluster_scores,
    score_resident_clusters,
)
from .index import RetroSpecAttentionLevel
from .index_residency import (
    RetroSpecResidentBatchView,
)
from .segmented_types import (
    RetroSpecIndexedTokenAttentionSelection,
    RetroSpecRankedSelectionPlan,
    RetroSpecTokenSelectionPlan,
    _ClusterSelectionWorkspace,
    _DraftSelectionScratch,
    _IndexedVerificationWorkspace,
    _PackedClusterZones,
    _PreparedIndexedVerificationLayer,
    _SelectionPlanTable,
    _SelectionStepWorkspace,
)
from .selection_kernels import (
    emit_primary_exact_token_plan,
    gather_resident_estimation,
    gather_resident_exact_pages,
    pack_ranked_verification_plan,
)


class RetroSpecSegmentSelectionMixin:
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

    @staticmethod
    def _pack_bounded_mask_indices(
        mask: torch.Tensor,
        output_width: int,
        output_indices: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
        topk_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack True positions using a synchronization-free upper bound."""
        if output_width < 0:
            raise ValueError("output_width must be non-negative")
        supplied_outputs = (output_indices, output_mask, topk_order)
        if any(output is not None for output in supplied_outputs) and not all(
            output is not None for output in supplied_outputs
        ):
            raise ValueError("Bounded packing outputs must be supplied together")

        max_num_items = mask.shape[-1]
        leading_shape = mask.shape[:-1]
        active_width = min(output_width, max_num_items)

        if output_indices is None:
            output_indices = torch.empty(
                (*leading_shape, active_width),
                dtype=torch.int64,
                device=mask.device,
            )
            output_mask = torch.empty_like(output_indices, dtype=torch.bool)
            topk_order = torch.empty_like(output_indices)
        else:
            expected_shape = (*leading_shape, output_width)
            if output_indices.shape != expected_shape:
                raise ValueError("Bounded packing index output has an invalid shape")
            if output_mask is None or output_mask.shape != expected_shape:
                raise ValueError("Bounded packing mask output has an invalid shape")
            if topk_order is None or topk_order.shape != expected_shape:
                raise ValueError("Bounded packing scratch output has an invalid shape")
            output_indices.zero_()
            output_mask.zero_()

        assert output_mask is not None
        assert topk_order is not None
        if active_width == 0:
            return output_indices, output_mask

        logical_indices = torch.arange(
            max_num_items,
            dtype=torch.int64,
            device=mask.device,
        )
        sentinel = torch.full(
            (),
            max_num_items,
            dtype=torch.int64,
            device=mask.device,
        )

        candidates = torch.where(
            mask,
            logical_indices,
            sentinel,
        )

        packed_indices = output_indices[..., :active_width]
        packed_order = topk_order[..., :active_width]
        torch.topk(
            candidates,
            k=active_width,
            dim=-1,
            largest=False,
            sorted=True,
            out=(packed_indices, packed_order),
        )

        packed_mask = output_mask[..., :active_width]
        torch.lt(packed_indices, max_num_items, out=packed_mask)
        packed_indices.clamp_(
            min=0,
            max=max_num_items - 1,
        )

        return output_indices, output_mask

    @staticmethod
    def _sum_selected_scores(
        selected_scores: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sum probability mass already aligned with a ranked cluster zone."""
        if selected_scores.shape != selected_mask.shape:
            raise ValueError("Selected cluster scores and mask must match")

        if selected_scores.shape[2] == 0:
            return torch.zeros(
                selected_scores.shape[:2],
                dtype=selected_scores.dtype,
                device=selected_scores.device,
            )

        return selected_scores.masked_fill(~selected_mask, 0.0).sum(dim=2)

    def _get_draft_selection_scratch(
        self,
        batch_size: int,
        num_kv_heads: int,
        sparse_retrieval_width: int,
        device: torch.device,
    ) -> _DraftSelectionScratch:
        scratch = self._draft_selection_scratch
        matches = scratch is not None and scratch.matches(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            sparse_retrieval_width=sparse_retrieval_width,
            device=device,
        )
        if matches:
            return scratch

        batch_capacity = batch_size
        compatible_scratch = (
            scratch is not None
            and scratch.draft_exact_cluster_handles.shape[1] == num_kv_heads
            and scratch.draft_exact_cluster_handles.device == device
        )
        if compatible_scratch:
            assert scratch is not None
            batch_capacity = max(
                batch_capacity,
                scratch.draft_exact_cluster_handles.shape[0],
            )
            old_retrieval_width = scratch.draft_exact_cluster_handles.shape[2]
            sparse_retrieval_width = max(sparse_retrieval_width, old_retrieval_width)
        scratch = _DraftSelectionScratch.allocate(
            batch_capacity=batch_capacity,
            num_kv_heads=num_kv_heads,
            sparse_retrieval_width=sparse_retrieval_width,
            device=device,
        )
        self._draft_selection_scratch = scratch
        return scratch

    def _get_indexed_verification_workspace(
        self,
        pair_capacity: int,
        num_kv_heads: int,
        exact_width: int,
        estimation_width: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _IndexedVerificationWorkspace:
        workspace = self._indexed_verification_workspace
        matches = workspace is not None and workspace.matches(
            pair_capacity=pair_capacity,
            num_kv_heads=num_kv_heads,
            exact_width=exact_width,
            estimation_width=estimation_width,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        if matches:
            return workspace

        if workspace is not None:
            pair_capacity = max(pair_capacity, workspace.keys.shape[0])
            exact_width = max(exact_width, workspace.exact_cluster_indices.shape[2])
            estimation_width = max(estimation_width, workspace.keys.shape[2])
        workspace = _IndexedVerificationWorkspace.allocate(
            pair_capacity=pair_capacity,
            num_kv_heads=num_kv_heads,
            exact_width=exact_width,
            estimation_width=estimation_width,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        self._indexed_verification_workspace = workspace
        return workspace

    def _get_selection_plan_step(
        self,
        layer_name: str,
        plan_slot: int,
        view: RetroSpecResidentBatchView,
        batch_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[
        RetroSpecRankedSelectionPlan,
        _SelectionStepWorkspace,
        _SelectionPlanTable,
    ]:
        if not 0 <= plan_slot < self.num_speculative_tokens:
            raise ValueError("plan_slot is outside the speculative range")

        sparse_retrieval_width, sparse_estimation_width, expanded_retrieval_width = (
            self._maximum_zone_widths(view.max_num_clusters)
        )
        primary_exact_width = exact_attention_primary_token_capacity(
            max_model_len=self.max_model_len,
            prefill_segment_size=self.prefill_segment_size_tokens,
            generation_update_interval=self.generation_update_interval,
            num_speculative_tokens=self.num_speculative_tokens,
            block_size=self.block_size,
        )
        max_access_width = sparse_retrieval_width
        if device.type == "cuda":
            self.cluster_store.reserve_resident_access_ring(
                device,
                batch_size * num_kv_heads * max_access_width,
            )
            self.cluster_store.reserve_verification_resolve_workspace(
                device,
                batch_size
                * self.num_speculative_tokens
                * num_kv_heads
                * expanded_retrieval_width,
                view.max_pages_per_cluster,
                batch_size * self.num_speculative_tokens * num_kv_heads,
                batch_size
                * self.num_speculative_tokens
                * num_kv_heads
                * expanded_retrieval_width
                * view.max_pages_per_cluster,
            )

        table = self._selection_plan_tables.get(layer_name)
        matches = table is not None and table.matches(
            num_steps=self.num_speculative_tokens,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            primary_exact_width=primary_exact_width,
            sparse_retrieval_width=sparse_retrieval_width,
            prefetch_width=max_access_width,
            expanded_retrieval_width=expanded_retrieval_width,
            sparse_estimation_width=sparse_estimation_width,
            max_pages_per_cluster=view.max_pages_per_cluster,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        if not matches:
            if layer_name in self._selection_plan_written_layers:
                raise RuntimeError(
                    "Selection-plan shape changed during an active proposal"
                )
            batch_capacity = batch_size
            if table is not None:
                batch_capacity = max(batch_capacity, table.batch_capacity)
            table = _SelectionPlanTable.allocate(
                layer_name=layer_name,
                num_steps=self.num_speculative_tokens,
                batch_capacity=batch_capacity,
                num_kv_heads=num_kv_heads,
                primary_exact_width=primary_exact_width,
                sparse_retrieval_width=sparse_retrieval_width,
                prefetch_width=max_access_width,
                expanded_retrieval_width=expanded_retrieval_width,
                sparse_estimation_width=sparse_estimation_width,
                max_pages_per_cluster=view.max_pages_per_cluster,
                head_size=head_size,
                dtype=dtype,
                device=device,
            )
            self._selection_plan_tables[layer_name] = table

        assert table is not None
        scratch = self._get_draft_selection_scratch(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            sparse_retrieval_width=sparse_retrieval_width,
            device=device,
        )
        plan = table.ranked_plan(plan_slot, batch_size)
        workspace = table.step_workspace(plan_slot, batch_size, scratch)
        return plan, workspace, table

    @staticmethod
    def _build_resident_exact_cluster_selection(
        view: RetroSpecResidentBatchView,
        packed_cluster_indices: torch.Tensor,
        packed_cluster_mask: torch.Tensor,
        selected_cluster_ids: torch.Tensor | None = None,
        selected_page_ids: torch.Tensor | None = None,
        selected_page_token_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, max_selected = packed_cluster_indices.shape
        max_pages = view.max_pages_per_cluster
        device = packed_cluster_indices.device
        output_tensors = (
            selected_cluster_ids,
            selected_page_ids,
            selected_page_token_counts,
        )
        if any(output is None for output in output_tensors):
            if not all(output is None for output in output_tensors):
                raise ValueError("Exact selection outputs must be supplied together")
            selected_cluster_ids = torch.empty(
                (batch_size, num_kv_heads, max_selected),
                dtype=torch.int64,
                device=device,
            )
            selected_page_ids = torch.empty(
                (batch_size, num_kv_heads, max_selected, max_pages),
                dtype=torch.int64,
                device=device,
            )
            selected_page_token_counts = torch.empty(
                selected_page_ids.shape, dtype=torch.int32, device=device
            )

        assert selected_cluster_ids is not None
        assert selected_page_ids is not None
        assert selected_page_token_counts is not None
        if view.arena is None or max_selected == 0 or max_pages == 0:
            selected_cluster_ids.fill_(-1)
            selected_page_ids.fill_(-1)
            selected_page_token_counts.zero_()
            return selected_cluster_ids, selected_page_ids, selected_page_token_counts

        arena = view.arena
        if device.type == "cuda":
            gather_resident_exact_pages(
                cluster_ids=arena.cluster_ids,
                cluster_page_starts=arena.cluster_page_starts,
                cluster_page_counts=arena.cluster_page_counts,
                page_ids=arena.page_ids,
                page_token_counts=arena.page_token_counts,
                cluster_offsets=arena.cluster_offsets,
                page_offsets=arena.page_offsets,
                request_slot_ids=view.request_slot_ids,
                selected_indices=packed_cluster_indices,
                selected_mask=packed_cluster_mask,
                output_cluster_ids=selected_cluster_ids,
                output_page_ids=selected_page_ids,
                output_page_token_counts=selected_page_token_counts,
            )
            return selected_cluster_ids, selected_page_ids, selected_page_token_counts

        slots = view.request_slot_ids.clamp_min(0)
        head_indices = torch.arange(num_kv_heads, dtype=torch.int64, device=device)[
            None, :, None
        ].expand_as(packed_cluster_indices)
        valid_clusters = packed_cluster_mask & (
            view.request_slot_ids[:, None, None] >= 0
        )
        request_cluster_offsets = arena.cluster_offsets.index_select(0, slots)
        absolute_cluster_indices = (
            request_cluster_offsets[:, None, None] + packed_cluster_indices
        )
        absolute_cluster_indices.clamp_(min=0, max=arena.cluster_ids.shape[1] - 1)
        gathered_cluster_ids = arena.cluster_ids[head_indices, absolute_cluster_indices]
        valid_clusters &= gathered_cluster_ids >= 0
        selected_cluster_ids.copy_(gathered_cluster_ids)
        selected_cluster_ids.masked_fill_(~valid_clusters, -1)

        page_starts = arena.cluster_page_starts[head_indices, absolute_cluster_indices]
        page_counts = arena.cluster_page_counts[head_indices, absolute_cluster_indices]
        page_offsets = torch.arange(max_pages, dtype=torch.int64, device=device)
        request_page_offsets = arena.page_offsets.index_select(0, slots)
        flat_page_indices = (
            request_page_offsets[:, None, None, None]
            + page_starts.unsqueeze(-1)
            + page_offsets
        )
        valid_pages = valid_clusters.unsqueeze(-1) & (
            page_offsets < page_counts.unsqueeze(-1)
        )
        flat_page_indices.clamp_(min=0, max=arena.page_ids.shape[1] - 1)
        page_heads = head_indices.unsqueeze(-1).expand_as(flat_page_indices)
        selected_page_ids.copy_(arena.page_ids[page_heads, flat_page_indices])
        selected_page_token_counts.copy_(
            arena.page_token_counts[page_heads, flat_page_indices]
        )
        selected_page_ids.masked_fill_(~valid_pages, -1)
        selected_page_token_counts.masked_fill_(~valid_pages, 0)

        return selected_cluster_ids, selected_page_ids, selected_page_token_counts

    @staticmethod
    def _build_resident_estimation_selection(
        view: RetroSpecResidentBatchView,
        packed_indices: torch.Tensor,
        packed_mask: torch.Tensor,
        head_size: int,
        dtype: torch.dtype,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, max_selected_clusters = packed_indices.shape
        output_shape = (
            batch_size,
            num_kv_heads,
            max_selected_clusters,
            head_size,
        )
        output_tensors = (keys, values, counts)
        if any(output is None for output in output_tensors):
            if not all(output is None for output in output_tensors):
                raise ValueError("Estimation outputs must be supplied together")
            keys = torch.empty(output_shape, dtype=dtype, device=packed_indices.device)
            values = torch.empty_like(keys)
            counts = torch.empty(
                output_shape[:-1], dtype=torch.int32, device=packed_indices.device
            )

        assert keys is not None
        assert values is not None
        assert counts is not None
        if view.arena is None or max_selected_clusters == 0:
            keys.zero_()
            values.zero_()
            counts.zero_()
            return keys, values, counts

        arena = view.arena
        if packed_indices.device.type == "cuda":
            gather_resident_estimation(
                cluster_keys=arena.cluster_keys,
                cluster_values=arena.cluster_values,
                cluster_token_counts=arena.cluster_token_counts,
                cluster_offsets=arena.cluster_offsets,
                request_slot_ids=view.request_slot_ids,
                selected_indices=packed_indices,
                selected_mask=packed_mask,
                output_keys=keys,
                output_values=values,
                output_token_counts=counts,
            )
            return keys, values, counts

        slots = view.request_slot_ids.clamp_min(0)
        head_indices = torch.arange(
            num_kv_heads, dtype=torch.int64, device=packed_indices.device
        )[None, :, None].expand_as(packed_indices)
        valid = packed_mask & (view.request_slot_ids[:, None, None] >= 0)
        request_cluster_offsets = arena.cluster_offsets.index_select(0, slots)
        absolute_indices = request_cluster_offsets[:, None, None] + packed_indices
        absolute_indices.clamp_(min=0, max=arena.cluster_ids.shape[1] - 1)
        keys.copy_(arena.cluster_keys[head_indices, absolute_indices])
        values.copy_(arena.cluster_values[head_indices, absolute_indices])
        counts.copy_(arena.cluster_token_counts[head_indices, absolute_indices])
        keys.masked_fill_(~valid.unsqueeze(-1), 0.0)
        values.masked_fill_(~valid.unsqueeze(-1), 0.0)
        counts.masked_fill_(~valid, 0)

        return keys, values, counts

    def _prepare_plan_step(
        self,
        layer_name: str,
        plan_slot: int,
        seq_lens: torch.Tensor,
        indexed_starts: torch.Tensor,
        indexed_ends: torch.Tensor,
        indexed_requests: torch.Tensor,
        max_num_tokens: int,
        view: RetroSpecResidentBatchView,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[
        RetroSpecRankedSelectionPlan,
        _SelectionStepWorkspace,
        _SelectionPlanTable,
    ]:
        plan, output_workspace, table = self._get_selection_plan_step(
            layer_name=layer_name,
            plan_slot=plan_slot,
            view=view,
            batch_size=seq_lens.shape[0],
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=seq_lens.device,
        )

        emit_primary_exact_token_plan(
            seq_lens=seq_lens,
            indexed_starts=indexed_starts,
            indexed_ends=indexed_ends,
            indexed_requests=indexed_requests,
            num_kv_heads=num_kv_heads,
            max_num_tokens=max_num_tokens,
            block_size=self.block_size,
            num_recent_blocks=self.num_recent_blocks,
            output_indices=plan.primary_exact_token_indices,
            output_mask=plan.primary_exact_token_mask,
        )
        return plan, output_workspace, table

    def _publish_plan_step(
        self,
        layer_name: str,
        plan_slot: int,
        active_mask: torch.Tensor,
        table: _SelectionPlanTable,
    ) -> None:
        table.valid_rows[plan_slot, : active_mask.shape[0]].copy_(active_mask)
        self._selection_plan_written_layers.add(layer_name)

    def _make_reference_plan(
        self,
        layer_name: str,
        plan_slot: int,
        active_mask: torch.Tensor,
        forced_exact_mask: torch.Tensor,
        cluster_zones: _PackedClusterZones,
        ranked_indices: torch.Tensor,
        candidate_counts: torch.Tensor,
        sparse_attn: torch.Tensor,
        expanded_attn: torch.Tensor,
        view: RetroSpecResidentBatchView,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[RetroSpecTokenSelectionPlan, _SelectionStepWorkspace]:
        ranked_plan, output_workspace, table = self._get_selection_plan_step(
            layer_name=layer_name,
            plan_slot=plan_slot,
            view=view,
            batch_size=forced_exact_mask.shape[0],
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=forced_exact_mask.device,
        )
        if ranked_indices.shape != ranked_plan.ranked_cluster_indices.shape:
            raise ValueError("Reference ranking does not match the plan journal")
        ranked_plan.ranked_cluster_indices.copy_(ranked_indices)
        ranked_plan.candidate_counts.copy_(candidate_counts)

        per_head_forced_exact = forced_exact_mask.unsqueeze(1).expand(
            -1, num_kv_heads, -1
        )
        packed_indices, packed_mask = self._pack_bounded_mask_indices(
            per_head_forced_exact,
            ranked_plan.primary_exact_token_indices.shape[-1],
        )
        ranked_plan.primary_exact_token_indices.zero_()
        ranked_plan.primary_exact_token_mask.zero_()
        packed_width = packed_indices.shape[-1]
        ranked_plan.primary_exact_token_indices[..., :packed_width].copy_(
            packed_indices
        )
        ranked_plan.primary_exact_token_mask[..., :packed_width].copy_(packed_mask)

        ranked_plan.request_slot_ids.copy_(view.request_slot_ids)
        if view.arena is None:
            ranked_plan.request_slot_generations.fill_(-1)
        else:
            valid_slots = view.request_slot_ids >= 0
            safe_slots = view.request_slot_ids.clamp_min(0)
            ranked_plan.request_slot_generations.copy_(
                view.arena.generations.index_select(0, safe_slots)
            )
            ranked_plan.request_slot_generations.masked_fill_(~valid_slots, -1)

        sparse_exact = cluster_zones.sparse_retrieval_indices.masked_fill(
            ~cluster_zones.sparse_retrieval_mask, -1
        )
        sparse_estimation = cluster_zones.sparse_estimation_indices.masked_fill(
            ~cluster_zones.sparse_estimation_mask, -1
        )
        expanded_exact = cluster_zones.expanded_retrieval_indices.masked_fill(
            ~cluster_zones.expanded_retrieval_mask, -1
        )
        expanded_estimation = cluster_zones.expanded_estimation_indices.masked_fill(
            ~cluster_zones.expanded_estimation_mask, -1
        )
        ranked_plan.sparse_attn.copy_(sparse_attn)
        ranked_plan.expanded_attn.copy_(expanded_attn)
        self._publish_plan_step(layer_name, plan_slot, active_mask, table)
        plan = RetroSpecTokenSelectionPlan(
            layer_name=layer_name,
            request_slot_ids=ranked_plan.request_slot_ids,
            request_slot_generations=ranked_plan.request_slot_generations,
            primary_exact_token_indices=ranked_plan.primary_exact_token_indices,
            primary_exact_token_mask=ranked_plan.primary_exact_token_mask,
            sparse_exact_cluster_indices=sparse_exact,
            sparse_estimation_cluster_indices=sparse_estimation,
            expanded_exact_cluster_indices=expanded_exact,
            expanded_estimation_cluster_indices=expanded_estimation,
            sparse_attn=ranked_plan.sparse_attn,
            expanded_attn=ranked_plan.expanded_attn,
        )
        return plan, output_workspace

    @staticmethod
    def _flatten_plan_rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten(0, 1)

    def _materialize_estimation_selection(
        self,
        layer_name: str,
        cluster_indices: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
        head_size: int,
        dtype: torch.dtype,
        plan_row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_rows = (
            cluster_indices.shape[0]
            if plan_row_indices is None
            else plan_row_indices.shape[0]
        )
        num_kv_heads = cluster_indices.shape[1]
        estimation_width = cluster_indices.shape[2]
        workspace = self._get_indexed_verification_workspace(
            pair_capacity=num_rows,
            num_kv_heads=num_kv_heads,
            exact_width=0,
            estimation_width=estimation_width,
            head_size=head_size,
            dtype=dtype,
            device=cluster_indices.device,
        )

        cluster_shape = (num_rows, num_kv_heads, estimation_width)
        summary_shape = (*cluster_shape, head_size)
        num_clusters = prod(cluster_shape)
        num_summary_items = prod(summary_shape)
        selected_indices = workspace.estimation_cluster_indices.view(-1)[
            :num_clusters
        ].view(cluster_shape)
        selected_mask = workspace.estimation_cluster_mask.view(-1)[:num_clusters].view(
            cluster_shape
        )
        packed_slots = workspace.request_slot_ids[:num_rows]
        packed_generations = workspace.request_slot_generations[:num_rows]

        if plan_row_indices is None:
            selected_indices.copy_(cluster_indices)
            packed_slots.copy_(request_slot_ids)
            packed_generations.copy_(request_slot_generations)
        else:
            torch.index_select(
                cluster_indices,
                0,
                plan_row_indices,
                out=selected_indices,
            )
            request_rows = plan_row_indices.remainder(request_slot_ids.shape[0])
            torch.index_select(
                request_slot_ids,
                0,
                request_rows,
                out=packed_slots,
            )
            torch.index_select(
                request_slot_generations,
                0,
                request_rows,
                out=packed_generations,
            )

        torch.ge(selected_indices, 0, out=selected_mask)
        view = self._gpu_index_residency.get_active_view(
            layer_name,
            self._proposal_request_ids,
            cluster_indices.device,
        )
        arena = view.arena
        if arena is not None:
            valid_slots = packed_slots >= 0
            safe_slots = packed_slots.clamp_min(0)
            current_generations = arena.generations.index_select(0, safe_slots)
            descriptors_valid = ~valid_slots | (
                current_generations == packed_generations
            )
            torch._assert_async(
                descriptors_valid.all(),
                f"Stale RetroSpec request descriptor for layer {layer_name!r}",
            )

        row_view = RetroSpecResidentBatchView(
            arena=arena,
            request_slot_ids=packed_slots,
            max_num_clusters=view.max_num_clusters,
            max_pages_per_cluster=view.max_pages_per_cluster,
            max_num_pages=view.max_num_pages,
        )
        keys = workspace.keys.view(-1)[:num_summary_items].view(summary_shape)
        values = workspace.values.view(-1)[:num_summary_items].view(summary_shape)
        counts = workspace.token_counts.view(-1)[:num_clusters].view(cluster_shape)
        self._build_resident_estimation_selection(
            row_view,
            selected_indices,
            selected_mask,
            head_size,
            dtype,
            keys,
            values,
            counts,
        )
        return selected_indices, keys, values, counts

    def get_indexed_selection(
        self,
        layer_name: str,
        level: RetroSpecAttentionLevel,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        prepared_exact: _PreparedIndexedVerificationLayer | None = None,
    ) -> RetroSpecIndexedTokenAttentionSelection:
        table = self._selection_plan_tables.get(layer_name)
        if table is None:
            raise RuntimeError(
                f"No draft selection plan exists for layer {layer_name!r}"
            )
        if request_indices.shape != token_indices.shape:
            raise ValueError("Indexed plan indices must have equal shapes")
        if request_indices.device != table.valid_rows.device:
            raise ValueError("Indexed request indices use the wrong device")
        if token_indices.device != table.valid_rows.device:
            raise ValueError("Indexed token indices use the wrong device")

        if level == RetroSpecAttentionLevel.SPARSE:
            exact_width = table.sparse_retrieval_width
            expanded = False
            attention_mass = table.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            exact_width = table.expanded_retrieval_width
            expanded = True
            attention_mass = table.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        num_pairs = request_indices.numel()
        num_kv_heads = table.ranked_cluster_indices.shape[2]
        estimation_width = table.sparse_estimation_width
        workspace = self._get_indexed_verification_workspace(
            pair_capacity=num_pairs,
            num_kv_heads=num_kv_heads,
            exact_width=0 if prepared_exact is not None else exact_width,
            estimation_width=estimation_width,
            head_size=table.head_size,
            dtype=table.dtype,
            device=request_indices.device,
        )

        plan_rows = workspace.plan_row_indices[:num_pairs]
        plan_valid_rows = workspace.plan_valid_rows[:num_pairs]
        packed_slots = workspace.request_slot_ids[:num_pairs]
        packed_generations = workspace.request_slot_generations[:num_pairs]
        exact_shape = (
            (num_pairs, num_kv_heads, 0)
            if prepared_exact is not None
            else (num_pairs, num_kv_heads, exact_width)
        )
        estimation_shape = (num_pairs, num_kv_heads, estimation_width)
        num_exact = prod(exact_shape)
        num_estimation = prod(estimation_shape)
        packed_exact = workspace.exact_cluster_indices.view(-1)[:num_exact].view(
            exact_shape
        )
        packed_estimation = workspace.estimation_cluster_indices.view(-1)[
            :num_estimation
        ].view(estimation_shape)
        packed_estimation_mask = workspace.estimation_cluster_mask.view(-1)[
            :num_estimation
        ].view(estimation_shape)
        packed_attention = workspace.attention_mass[:num_pairs]

        flatten = self._flatten_plan_rows
        pack_ranked_verification_plan(
            request_indices=request_indices,
            token_indices=token_indices,
            valid_rows=table.valid_rows,
            request_slot_ids=table.request_slot_ids,
            request_slot_generations=table.request_slot_generations,
            ranked_cluster_indices=flatten(table.ranked_cluster_indices),
            candidate_counts=flatten(table.candidate_counts),
            attention_mass=attention_mass.view(-1),
            output_plan_row_indices=plan_rows,
            output_plan_valid_rows=plan_valid_rows,
            output_request_slot_ids=packed_slots,
            output_request_slot_generations=packed_generations,
            output_exact_cluster_indices=packed_exact,
            output_estimation_cluster_indices=packed_estimation,
            output_estimation_cluster_mask=packed_estimation_mask,
            output_attention_mass=packed_attention,
            retrieval_ratio=self.retrieval_ratio,
            estimation_ratio=self.estimation_ratio,
            expanded=expanded,
        )

        if prepared_exact is not None:
            if prepared_exact.layer_name != layer_name:
                raise RuntimeError("Prepared verification layer does not match")
            plan_rows = prepared_exact.plan_row_indices
            plan_valid_rows = prepared_exact.plan_valid_rows
            packed_slots = prepared_exact.request_slot_ids
            packed_generations = prepared_exact.request_slot_generations
            packed_exact = prepared_exact.exact_cluster_indices
            packed_attention = prepared_exact.attention_mass

        view = self._gpu_index_residency.get_active_view(
            layer_name, self._proposal_request_ids, request_indices.device
        )
        summary_shape = (*estimation_shape, table.head_size)
        num_summary = prod(summary_shape)
        keys = workspace.keys.view(-1)[:num_summary].view(summary_shape)
        values = workspace.values.view(-1)[:num_summary].view(summary_shape)
        counts = workspace.token_counts.view(-1)[:num_estimation].view(estimation_shape)
        if view.arena is None:
            keys.zero_()
            values.zero_()
            counts.zero_()
        else:
            gather_resident_estimation(
                cluster_keys=view.arena.cluster_keys,
                cluster_values=view.arena.cluster_values,
                cluster_token_counts=view.arena.cluster_token_counts,
                cluster_offsets=view.arena.cluster_offsets,
                request_slot_ids=packed_slots,
                selected_indices=packed_estimation,
                selected_mask=packed_estimation_mask,
                output_keys=keys,
                output_values=values,
                output_token_counts=counts,
            )
        return RetroSpecIndexedTokenAttentionSelection(
            layer_name=layer_name,
            plan_row_indices=plan_rows,
            plan_valid_rows=plan_valid_rows,
            request_slot_ids=packed_slots,
            request_slot_generations=packed_generations,
            primary_exact_token_indices=flatten(table.primary_exact_token_indices),
            primary_exact_token_mask=flatten(table.primary_exact_token_mask),
            exact_cluster_indices=packed_exact,
            estimation_cluster_indices=packed_estimation,
            estimation_keys=keys,
            estimation_values=values,
            estimation_token_counts=counts,
            attention_mass=packed_attention,
        )
