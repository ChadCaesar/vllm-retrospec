# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from vllm.v1.spec_decode.retrospec.legacy.segmented.build import (
    _RetroSpecSegmentedTokenIndexBuildMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.planning import (
    _RetroSpecSegmentedTokenIndexPlanningMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.scoring import (
    _RetroSpecSegmentedTokenIndexScoringMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.selection import (
    _RetroSpecSegmentedTokenIndexSelectionMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.types import (
    RetroSpecFullVerificationPlan,
    RetroSpecIndexedTokenAttentionSelection,
    RetroSpecRankedDraftAttentionSelection,
    RetroSpecRankedSelectionPlan,
    RetroSpecTokenAttentionSelection,
    RetroSpecTokenSelectionPlan,
    _ClusterSelectionWorkspace,
    _CompletedRequestLayerSegment,
    _DraftSelectionScratch,
    _IndexedVerificationTransaction,
    _IndexedVerificationWorkspace,
    _PackedClusterZones,
    _PrefetchedFullVerificationLayer,
    _PreparedIndexedVerificationLayer,
    _PrimedFullVerificationPipeline,
    _RequestLayerIndex,
    _RequestLayerSegment,
    _SelectionPlanTable,
    _SelectionStepWorkspace,
    _StagedRequestLayerSegment,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.verification import (
    _RetroSpecSegmentedTokenIndexVerificationMixin,
)

from ..clustering import segmented_kmeans
from ..index import RetroSpecAttentionLevel, RetroSpecIndexBase
from ..performance import RetroSpecPerformanceStats
from .cluster_store import (
    RetroSpecClusterPageStore,
)
from .index_residency import (
    RetroSpecGPUIndexResidencyManager,
)
from .pinned_memory import RetroSpecPinnedMemoryManager
from .resident_cache import RetroSpecResidentReadLease
from .selection_provenance import (
    RetroSpecReplayMode,
    RetroSpecSelectionProvenanceTracer,
)

__all__ = [
    "segmented_kmeans",
    "RetroSpecTokenSelectionPlan",
    "RetroSpecRankedSelectionPlan",
    "RetroSpecTokenAttentionSelection",
    "RetroSpecRankedDraftAttentionSelection",
    "RetroSpecIndexedTokenAttentionSelection",
    "_PreparedIndexedVerificationLayer",
    "_IndexedVerificationTransaction",
    "RetroSpecFullVerificationPlan",
    "_RequestLayerSegment",
    "_CompletedRequestLayerSegment",
    "_StagedRequestLayerSegment",
    "_RequestLayerIndex",
    "_PrefetchedFullVerificationLayer",
    "_PrimedFullVerificationPipeline",
    "_PackedClusterZones",
    "_ClusterSelectionWorkspace",
    "_SelectionStepWorkspace",
    "_DraftSelectionScratch",
    "_SelectionPlanTable",
    "_IndexedVerificationWorkspace",
    "RetroSpecSegmentedTokenIndex",
]


class RetroSpecSegmentedTokenIndex(
    _RetroSpecSegmentedTokenIndexBuildMixin,
    _RetroSpecSegmentedTokenIndexScoringMixin,
    _RetroSpecSegmentedTokenIndexPlanningMixin,
    _RetroSpecSegmentedTokenIndexSelectionMixin,
    _RetroSpecSegmentedTokenIndexVerificationMixin,
    RetroSpecIndexBase,
):
    """Token-level segmented index backed by private cluster KV pages."""

    _FULL_VERIFY_PRIME_DEPTH = 2
    _FULL_VERIFY_PRIME_BOOTSTRAP_OUTCOMES = 8
    _FULL_VERIFY_PRIME_MIN_ADOPTION_RATE = 0.5
    _FULL_VERIFY_PRIME_EMA_ALPHA = 0.25
    _FULL_VERIFY_PRIME_REPROBE_INTERVAL = 8

    def __init__(
        self,
        block_size: int,
        num_speculative_tokens: int,
        retrieval_ratio: float,
        estimation_ratio: float,
        prefill_segment_size_tokens: int,
        generation_update_interval: int,
        blocks_per_cluster: int,
        num_kmeans_iterations: int,
        max_model_len: int,
        max_pending_cluster_builds: int = 2,
        cpu_page_build_workers: int = 4,
        full_verify_gather_workers: int = 4,
        cache_ratio: float = 0.0,
        pin_memory: bool = False,
        max_resident_requests: int = 1,
        prefill_warmup_multiplier: int = 4,
        cpu_page_initial_slab_bytes: int | None = None,
        cpu_page_slab_bytes: int = 1 << 20,
        max_pinned_memory_bytes: int = 64 << 20,
        max_gpu_index_memory_bytes: int = 4 << 30,
        replay_mode: RetroSpecReplayMode = "off",
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        super().__init__(
            block_size=block_size,
            num_speculative_tokens=num_speculative_tokens,
            retrieval_ratio=retrieval_ratio,
            estimation_ratio=estimation_ratio,
        )

        if prefill_segment_size_tokens % block_size != 0:
            raise ValueError(
                "prefill_segment_size_tokens must be divisible by block_size"
            )
        if generation_update_interval % block_size != 0:
            raise ValueError(
                "generation_update_interval must be divisible by block_size"
            )
        if blocks_per_cluster <= 0:
            raise ValueError("blocks_per_cluster must be positive")
        if num_kmeans_iterations <= 0:
            raise ValueError("num_kmeans_iterations must be positive")
        if max_pending_cluster_builds <= 0:
            raise ValueError("max_pending_cluster_builds must be positive")
        if cpu_page_build_workers <= 0:
            raise ValueError("cpu_page_build_workers must be positive")
        if full_verify_gather_workers <= 0:
            raise ValueError("full_verify_gather_workers must be positive")
        if prefill_warmup_multiplier <= 0:
            raise ValueError("prefill_warmup_multiplier must be positive")
        if cpu_page_slab_bytes <= 0:
            raise ValueError("cpu_page_slab_bytes must be positive")
        if cpu_page_initial_slab_bytes is None:
            cpu_page_initial_slab_bytes = min(8 << 20, cpu_page_slab_bytes)
        if cpu_page_initial_slab_bytes <= 0:
            raise ValueError("cpu_page_initial_slab_bytes must be positive")
        if cpu_page_initial_slab_bytes > cpu_page_slab_bytes:
            raise ValueError(
                "cpu_page_initial_slab_bytes must not exceed cpu_page_slab_bytes"
            )
        if max_pinned_memory_bytes <= 0:
            raise ValueError("max_pinned_memory_bytes must be positive")
        if max_gpu_index_memory_bytes <= 0:
            raise ValueError("max_gpu_index_memory_bytes must be positive")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

        tokens_per_cluster = blocks_per_cluster * block_size
        if prefill_segment_size_tokens % tokens_per_cluster != 0:
            raise ValueError(
                "prefill_segment_size_tokens must be divisible by "
                "blocks_per_cluster * block_size"
            )
        if generation_update_interval % tokens_per_cluster != 0:
            raise ValueError(
                "generation_update_interval must be divisible by "
                "blocks_per_cluster * block_size"
            )

        self.prefill_segment_size_tokens = prefill_segment_size_tokens
        self.generation_update_interval = generation_update_interval
        self.num_speculative_tokens = num_speculative_tokens
        self.tokens_per_cluster = tokens_per_cluster
        self.num_kmeans_iterations = num_kmeans_iterations
        self.max_pending_cluster_builds = max_pending_cluster_builds
        self.prefill_warmup_multiplier = prefill_warmup_multiplier
        self.max_model_len = max_model_len
        self.performance_stats = performance_stats
        self.replay_mode = replay_mode
        self.selection_provenance = RetroSpecSelectionProvenanceTracer(replay_mode)
        if replay_mode == "ready_selected" and not pin_memory:
            raise ValueError("ready_selected replay requires pinned CPU staging memory")
        effective_cache_ratio = cache_ratio
        if cache_ratio == 0.0:
            # RetroInfer uses three sparse retrieval zones when an explicit
            # cache ratio is not supplied.
            effective_cache_ratio = min(
                retrieval_ratio * 3.0,
                1.0,
            )

        self._pinned_memory = RetroSpecPinnedMemoryManager(
            enabled=pin_memory,
            max_bytes=max_pinned_memory_bytes,
        )
        self._gpu_index_residency = RetroSpecGPUIndexResidencyManager(
            max_resident_requests=max_resident_requests,
            max_gpu_index_memory_bytes=max_gpu_index_memory_bytes,
            pinned_memory=self._pinned_memory,
            max_summary_slots=max_pending_cluster_builds,
            performance_stats=performance_stats,
        )
        self.cluster_store = RetroSpecClusterPageStore(
            page_size=block_size,
            cache_ratio=effective_cache_ratio,
            cpu_page_initial_slab_bytes=cpu_page_initial_slab_bytes,
            cpu_page_slab_bytes=cpu_page_slab_bytes,
            max_pending_cluster_builds=max_pending_cluster_builds,
            cpu_page_build_workers=cpu_page_build_workers,
            full_verify_gather_workers=full_verify_gather_workers,
            performance_stats=performance_stats,
            pinned_memory=self._pinned_memory,
            gpu_index_residency=self._gpu_index_residency,
        )

        # layer_name -> request_id -> token-level index
        self._indices: dict[str, dict[str, _RequestLayerIndex]] = {}
        self._full_verification_revision_counter = 1

        self._proposal_active = False
        self._proposal_request_ids: tuple[str, ...] = ()
        self._proposal_read_leases: list[RetroSpecResidentReadLease] = []
        self._proposal_index_revisions: dict[tuple[str, str], int] = {}
        self._proposal_resident_frozen = False

        # Shared across model layers. Selection results are copied into each
        # plan before the workspace is reused.
        self._cluster_selection_workspace: _ClusterSelectionWorkspace | None = None
        self._prefill_hint_selection_workspace: _ClusterSelectionWorkspace | None = None
        self._prefill_hint_streams: dict[torch.device, torch.cuda.Stream] = {}

        # One contiguous table is retained per layer. Its first dimensions are
        # [draft_step, request], so verification can gather packed pair rows
        # without walking per-step Python dictionaries.
        self._selection_plan_tables: dict[str, _SelectionPlanTable] = {}
        self._selection_plan_written_layers: set[str] = set()
        self._draft_selection_scratch: _DraftSelectionScratch | None = None
        self._indexed_verification_workspace: _IndexedVerificationWorkspace | None = (
            None
        )
        self._indexed_verification_exact_workspaces: dict[
            tuple[str, RetroSpecAttentionLevel], _IndexedVerificationWorkspace
        ] = {}
        self._indexed_verification_transaction: (
            _IndexedVerificationTransaction | None
        ) = None

        # CPU-offload construction is staged during layer execution and
        # committed after the complete prefill attention context.
        self._staged_segments: list[_StagedRequestLayerSegment] = []
        self._staged_segment_keys: set[tuple[str, str]] = set()

        # CPU-backed cluster pages are built by one serialized background
        # worker. The executor lives for one staged index transaction and is
        # closed by flush_staged_updates() or discard_staged_updates().
        self._cluster_build_executor: ThreadPoolExecutor | None = None
        self._pending_cluster_builds: deque[Future[_CompletedRequestLayerSegment]] = (
            deque()
        )

        # Full verification retains only the current and next layer layouts.
        # Staging the next layer before returning the current plan overlaps its
        # H2D copy with the current layer's attention and MLP computation.
        self._full_verification_pipeline_active = False
        self._full_verification_request_ids: tuple[str, ...] = ()
        self._full_verification_layers: tuple[tuple[str, int], ...] = ()
        self._full_verification_layer_cursor = 0
        self._full_verification_next_layer_index = 0
        self._full_verification_device: torch.device | None = None
        self._full_verification_prefetched: deque[_PrefetchedFullVerificationLayer] = (
            deque()
        )
        self._primed_full_verification: _PrimedFullVerificationPipeline | None = None
        self._full_verify_prime_outcomes = 0
        self._full_verify_prime_adoption_ema: float | None = None
        self._full_verify_prime_skipped_opportunities = 0

    def select_segmented(
        self,
        request_ids: Sequence[str],
        layer_name: str,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        active_mask: torch.Tensor,
        scale: float,
        plan_slot: int = 0,
        proposal_round: int = 0,
    ) -> RetroSpecTokenAttentionSelection | RetroSpecRankedDraftAttentionSelection:
        self._validate_inputs(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            active_mask,
        )

        if tuple(request_ids) != self._proposal_request_ids:
            raise RuntimeError(
                "Segmented token index request order does not match proposal order"
            )
        index_revisions = self._get_proposal_index_revisions(layer_name, request_ids)

        with self._cuda_timer("draft_selection_layout"):
            view = self._get_resident_view(layer_name, request_ids, key_cache)
            indexed_starts, indexed_ends, indexed_requests = (
                self._get_resident_indexed_bounds(view, block_table.device)
            )
            max_num_tokens = block_table.shape[1] * self.block_size

        num_kv_heads = key_cache.shape[2]
        with self._cuda_timer("draft_cluster_score"):
            cluster_scores, candidate_counts, workspace = self._score_resident_view(
                query, view, scale, num_kv_heads
            )

        direct_cuda_plan = workspace is not None and view.arena is not None
        ranked_values = None
        ranked_indices = None
        cluster_zones = None
        plan_table = None
        plan_valid_rows = None
        capture_request_descriptors = False
        if direct_cuda_plan:
            with self._cuda_timer("draft_plan_prepare"):
                plan, output_workspace, plan_table = self._prepare_plan_step(
                    layer_name=layer_name,
                    plan_slot=plan_slot,
                    seq_lens=seq_lens,
                    indexed_starts=indexed_starts,
                    indexed_ends=indexed_ends,
                    indexed_requests=indexed_requests,
                    max_num_tokens=max_num_tokens,
                    view=view,
                    num_kv_heads=num_kv_heads,
                    head_size=key_cache.shape[3],
                    dtype=key_cache.dtype,
                )
            with self._cuda_timer("draft_cluster_topk"):
                ranked_values, ranked_indices = self._rank_cluster_scores(
                    cluster_scores,
                    workspace,
                    ranking_width=plan.ranked_cluster_indices.shape[2],
                    output_indices=plan.ranked_cluster_indices,
                )
                plan.candidate_counts.copy_(candidate_counts)
            plan_valid_rows = plan_table.valid_rows[plan_slot, : active_mask.shape[0]]
            capture_request_descriptors = (
                layer_name not in self._selection_plan_written_layers
            )
        else:
            with self._cuda_timer("draft_selection_layout"):
                logical_token_ids, valid_token_mask, sink_recent_mask = (
                    self._build_token_layout(block_table, seq_lens)
                )
                forced_exact_mask = sink_recent_mask.clone()
                forced_exact_mask |= valid_token_mask & (
                    ~indexed_requests.unsqueeze(1)
                    | (logical_token_ids.unsqueeze(0) < indexed_starts.unsqueeze(1))
                    | (logical_token_ids.unsqueeze(0) >= indexed_ends.unsqueeze(1))
                )
            with self._cuda_timer("draft_cluster_topk"):
                cluster_zones = self._select_cluster_zones(
                    cluster_scores,
                    candidate_counts,
                    view=view,
                    workspace=workspace,
                )
                max_retrieval, max_estimation, _ = self._maximum_zone_widths(
                    cluster_scores.shape[2]
                )
                _, ranked_indices = torch.topk(
                    cluster_scores,
                    k=max_retrieval + max_estimation,
                    dim=2,
                    largest=True,
                    sorted=True,
                )

            sparse_attn_by_head = self._sum_selected_scores(
                cluster_zones.sparse_retrieval_scores,
                cluster_zones.sparse_retrieval_mask,
            )
            expanded_attn_by_head = self._sum_selected_scores(
                cluster_zones.expanded_retrieval_scores,
                cluster_zones.expanded_retrieval_mask,
            )
            has_clusters_by_head = candidate_counts > 0
            sparse_attn_by_head = torch.where(
                has_clusters_by_head,
                sparse_attn_by_head,
                torch.ones_like(sparse_attn_by_head),
            )
            expanded_attn_by_head = torch.where(
                has_clusters_by_head,
                expanded_attn_by_head,
                torch.ones_like(expanded_attn_by_head),
            )
            sparse_attn = sparse_attn_by_head.mean(dim=1)
            expanded_attn = expanded_attn_by_head.mean(dim=1)
            sparse_attn = torch.where(
                active_mask,
                sparse_attn,
                torch.ones_like(sparse_attn),
            )
            expanded_attn = torch.where(
                active_mask,
                expanded_attn,
                torch.ones_like(expanded_attn),
            )

            with self._cuda_timer("draft_plan_build"):
                plan, output_workspace = self._make_reference_plan(
                    layer_name=layer_name,
                    plan_slot=plan_slot,
                    active_mask=active_mask,
                    forced_exact_mask=forced_exact_mask,
                    cluster_zones=cluster_zones,
                    ranked_indices=ranked_indices,
                    candidate_counts=candidate_counts,
                    sparse_attn=sparse_attn,
                    expanded_attn=expanded_attn,
                    view=view,
                    num_kv_heads=num_kv_heads,
                    head_size=key_cache.shape[3],
                    dtype=key_cache.dtype,
                )

        resolve_timer = (
            "draft_bucket_resolve"
            if direct_cuda_plan
            else "draft_reference_materialize"
        )
        with (
            self._cpu_timer(f"{resolve_timer}_wall"),
            self._cuda_timer(f"{resolve_timer}/{layer_name}"),
        ):
            selection = self._materialize_draft_selection(
                request_ids=request_ids,
                query=query,
                proposal_round=proposal_round,
                draft_step=plan_slot,
                index_revisions=index_revisions,
                plan=plan,
                output_workspace=output_workspace,
                view=view,
                active_mask=active_mask,
                ranked_values=ranked_values,
                ranked_indices=ranked_indices,
                candidate_counts=candidate_counts,
                plan_valid_rows=plan_valid_rows,
                capture_request_descriptors=capture_request_descriptors,
            )
        if plan_table is not None:
            self._selection_plan_written_layers.add(layer_name)
        return selection

    def materialize(
        self,
        plan: RetroSpecTokenSelectionPlan,
        level: RetroSpecAttentionLevel,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecTokenAttentionSelection:
        # These parameters remain in the common index interface. Exact KV is now
        # gathered later by the reusable execution buffer.
        del key_cache, value_cache, block_table

        return self._materialize_token_selection(plan, level)
