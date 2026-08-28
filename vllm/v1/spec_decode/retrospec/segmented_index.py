# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil

import torch

from .cluster_scoring import reduce_grouped_cluster_scores, score_resident_clusters
from .cluster_store import (
    RetroSpecClusterBlockTable,
    RetroSpecClusterPageStore,
    RetroSpecResolvedClusterPages,
    RetroSpecStagedClusterInput,
)
from .clustering import segmented_kmeans
from .index import RetroSpecAttentionLevel, RetroSpecIndexBase
from .index_residency import (
    RetroSpecClusterSummary,
    RetroSpecGPUIndexResidencyManager,
    RetroSpecResidentBatchView,
    RetroSpecResidentSegment,
    RetroSpecStagedClusterSummary,
)
from .performance import RetroSpecPerformanceStats
from .selection_kernels import (
    gather_resident_estimation,
    gather_resident_exact_pages,
)


@dataclass(frozen=True)
class RetroSpecTokenSelectionPlan:
    layer_name: str

    primary_exact_token_indices: torch.Tensor
    primary_exact_token_mask: torch.Tensor

    sparse_exact_cluster_ids: torch.Tensor
    sparse_exact_page_ids: torch.Tensor
    sparse_exact_page_token_counts: torch.Tensor
    sparse_estimation_keys: torch.Tensor
    sparse_estimation_values: torch.Tensor
    sparse_estimation_token_counts: torch.Tensor

    expanded_exact_cluster_ids: torch.Tensor
    expanded_exact_page_ids: torch.Tensor
    expanded_exact_page_token_counts: torch.Tensor
    expanded_estimation_keys: torch.Tensor
    expanded_estimation_values: torch.Tensor
    expanded_estimation_token_counts: torch.Tensor

    sparse_attn: torch.Tensor
    expanded_attn: torch.Tensor


@dataclass(frozen=True)
class RetroSpecTokenAttentionSelection:
    exact_cluster_ids: torch.Tensor
    exact_page_ids: torch.Tensor
    exact_page_token_counts: torch.Tensor
    exact_token_counts: torch.Tensor

    estimation_keys: torch.Tensor
    estimation_values: torch.Tensor
    estimation_token_counts: torch.Tensor

    attention_mass: torch.Tensor
    plan: RetroSpecTokenSelectionPlan
    resolved_pages: RetroSpecResolvedClusterPages | None

    @property
    def hit_attn(self) -> torch.Tensor:
        return self.attention_mass


@dataclass(frozen=True)
class RetroSpecFullVerificationPlan:
    """Exact committed-prefix layout used by target full verification.

    Clustered stable tokens reference the existing cluster page store.
    Tokens outside complete clustered segments remain primary references into
    the active vLLM KV cache. No second full-prefix KV cache is created.
    """

    layer_name: str

    primary_exact_token_indices: torch.Tensor
    primary_exact_token_mask: torch.Tensor

    exact_page_ids: torch.Tensor
    exact_page_ids_cpu: torch.Tensor
    exact_page_token_counts: torch.Tensor
    exact_token_counts: torch.Tensor
    resolved_pages: RetroSpecResolvedClusterPages | None = None


@dataclass(frozen=True)
class _RequestLayerSegment:
    indexed_start: int
    indexed_end: int

    cluster_start: int
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_blocks: RetroSpecClusterBlockTable


@dataclass(frozen=True)
class _StagedRequestLayerSegment:
    layer_name: str
    request_id: str

    indexed_start: int
    indexed_end: int
    cluster_start: int

    cluster_summary: RetroSpecStagedClusterSummary
    build_future: Future[RetroSpecClusterBlockTable]


@dataclass
class _RequestLayerIndex:
    segments: list[_RequestLayerSegment]
    num_clusters: int
    indexed_end: int
    full_verification_page_ids_cpu: torch.Tensor | None


@dataclass(frozen=True)
class _FullVerificationPageLayout:
    page_ids: torch.Tensor
    page_ids_cpu: torch.Tensor
    page_token_counts: torch.Tensor


@dataclass(frozen=True)
class _PrefetchedFullVerificationLayer:
    layer_name: str
    layout: _FullVerificationPageLayout
    resolved_pages: RetroSpecResolvedClusterPages | None


@dataclass(frozen=True)
class _PackedClusterZones:
    sparse_retrieval_indices: torch.Tensor
    sparse_retrieval_mask: torch.Tensor

    sparse_estimation_indices: torch.Tensor
    sparse_estimation_mask: torch.Tensor

    expanded_retrieval_indices: torch.Tensor
    expanded_retrieval_mask: torch.Tensor

    expanded_estimation_indices: torch.Tensor
    expanded_estimation_mask: torch.Tensor

    first_draft_warmup_indices: torch.Tensor
    first_draft_warmup_mask: torch.Tensor


@dataclass(frozen=True)
class _ClusterSelectionWorkspace:
    logits: torch.Tensor
    scores: torch.Tensor
    softmax_lse: torch.Tensor
    ranking_scores: torch.Tensor
    candidate_counts: torch.Tensor
    topk_values: torch.Tensor
    topk_indices: torch.Tensor


@dataclass(frozen=True)
class _SelectionOutputWorkspace:
    sparse_exact_cluster_ids: torch.Tensor
    sparse_exact_page_ids: torch.Tensor
    sparse_exact_page_token_counts: torch.Tensor
    expanded_exact_cluster_ids: torch.Tensor
    expanded_exact_page_ids: torch.Tensor
    expanded_exact_page_token_counts: torch.Tensor
    draft_exact_page_token_counts: torch.Tensor

    draft_estimation_keys: torch.Tensor
    draft_estimation_values: torch.Tensor
    draft_estimation_token_counts: torch.Tensor
    expanded_estimation_keys: torch.Tensor
    expanded_estimation_values: torch.Tensor
    expanded_estimation_token_counts: torch.Tensor

    sparse_estimation_width: int
    sparse_retrieval_width: int

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        num_kv_heads: int,
        sparse_retrieval_width: int,
        expanded_retrieval_width: int,
        sparse_estimation_width: int,
        max_pages_per_cluster: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "_SelectionOutputWorkspace":
        sparse_cluster_shape = (
            batch_size,
            num_kv_heads,
            sparse_retrieval_width,
        )
        expanded_cluster_shape = (
            batch_size,
            num_kv_heads,
            expanded_retrieval_width,
        )
        sparse_page_shape = (*sparse_cluster_shape, max_pages_per_cluster)
        expanded_page_shape = (*expanded_cluster_shape, max_pages_per_cluster)
        draft_estimation_width = sparse_estimation_width + sparse_retrieval_width
        draft_estimation_shape = (
            batch_size,
            num_kv_heads,
            draft_estimation_width,
            head_size,
        )
        expanded_estimation_shape = (
            batch_size,
            num_kv_heads,
            sparse_estimation_width,
            head_size,
        )

        return cls(
            sparse_exact_cluster_ids=torch.empty(
                sparse_cluster_shape, dtype=torch.int64, device=device
            ),
            sparse_exact_page_ids=torch.empty(
                sparse_page_shape, dtype=torch.int64, device=device
            ),
            sparse_exact_page_token_counts=torch.empty(
                sparse_page_shape, dtype=torch.int32, device=device
            ),
            expanded_exact_cluster_ids=torch.empty(
                expanded_cluster_shape, dtype=torch.int64, device=device
            ),
            expanded_exact_page_ids=torch.empty(
                expanded_page_shape, dtype=torch.int64, device=device
            ),
            expanded_exact_page_token_counts=torch.empty(
                expanded_page_shape, dtype=torch.int32, device=device
            ),
            draft_exact_page_token_counts=torch.empty(
                sparse_page_shape, dtype=torch.int32, device=device
            ),
            draft_estimation_keys=torch.empty(
                draft_estimation_shape, dtype=dtype, device=device
            ),
            draft_estimation_values=torch.empty(
                draft_estimation_shape, dtype=dtype, device=device
            ),
            draft_estimation_token_counts=torch.empty(
                draft_estimation_shape[:-1], dtype=torch.int32, device=device
            ),
            expanded_estimation_keys=torch.empty(
                expanded_estimation_shape, dtype=dtype, device=device
            ),
            expanded_estimation_values=torch.empty(
                expanded_estimation_shape, dtype=dtype, device=device
            ),
            expanded_estimation_token_counts=torch.empty(
                expanded_estimation_shape[:-1],
                dtype=torch.int32,
                device=device,
            ),
            sparse_estimation_width=sparse_estimation_width,
            sparse_retrieval_width=sparse_retrieval_width,
        )

    def matches(
        self,
        batch_size: int,
        num_kv_heads: int,
        sparse_retrieval_width: int,
        expanded_retrieval_width: int,
        sparse_estimation_width: int,
        max_pages_per_cluster: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            self.sparse_exact_cluster_ids.shape
            == (batch_size, num_kv_heads, sparse_retrieval_width)
            and self.expanded_exact_cluster_ids.shape
            == (batch_size, num_kv_heads, expanded_retrieval_width)
            and self.sparse_exact_page_ids.shape[-1] == max_pages_per_cluster
            and self.draft_estimation_keys.shape
            == (
                batch_size,
                num_kv_heads,
                sparse_estimation_width + sparse_retrieval_width,
                head_size,
            )
            and self.expanded_estimation_keys.shape
            == (
                batch_size,
                num_kv_heads,
                sparse_estimation_width,
                head_size,
            )
            and self.draft_estimation_keys.dtype == dtype
            and self.draft_estimation_keys.device == device
        )


class RetroSpecSegmentedTokenIndex(RetroSpecIndexBase):
    """Token-level segmented index backed by private cluster KV pages."""

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
        cache_ratio: float = 0.0,
        pin_memory: bool = False,
        max_resident_requests: int = 1,
        first_draft_warmup_multiplier: int = 4,
        cpu_page_slab_bytes: int = 1 << 20,
        max_pinned_memory_bytes: int = 64 << 20,
        max_gpu_index_memory_bytes: int = 4 << 30,
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
        if first_draft_warmup_multiplier <= 0:
            raise ValueError("first_draft_warmup_multiplier must be positive")
        if cpu_page_slab_bytes <= 0:
            raise ValueError("cpu_page_slab_bytes must be positive")
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
        self.first_draft_warmup_multiplier = first_draft_warmup_multiplier
        self.performance_stats = performance_stats
        effective_cache_ratio = cache_ratio
        if cache_ratio == 0.0:
            # RetroInfer uses three sparse retrieval zones when an explicit
            # cache ratio is not supplied.
            effective_cache_ratio = min(
                retrieval_ratio * 3.0,
                1.0,
            )

        self.cluster_store = RetroSpecClusterPageStore(
            page_size=block_size,
            pin_memory=pin_memory,
            cache_ratio=effective_cache_ratio,
            cpu_page_slab_bytes=cpu_page_slab_bytes,
            max_pinned_memory_bytes=max_pinned_memory_bytes,
            max_pending_cluster_builds=max_pending_cluster_builds,
            performance_stats=performance_stats,
        )

        self._gpu_index_residency = RetroSpecGPUIndexResidencyManager(
            pin_memory=pin_memory,
            max_resident_requests=max_resident_requests,
            max_gpu_index_memory_bytes=max_gpu_index_memory_bytes,
        )

        # layer_name -> request_id -> token-level index
        self._indices: dict[str, dict[str, _RequestLayerIndex]] = {}

        self._proposal_active = False
        self._proposal_request_ids: tuple[str, ...] = ()

        # request_id -> layers that still need one query-guided resident
        # admission after the first real draft ranking.
        self._first_draft_warm_layers_by_request: dict[str, set[str]] = {}

        # Shared across model layers. Selection results are copied into each
        # plan before the workspace is reused.
        self._cluster_selection_workspace: _ClusterSelectionWorkspace | None = None

        # One output slot is retained for each layer and speculative step.
        # Plans stay live until parallel verification finishes, so slots may
        # be reused across proposals but not across steps in one proposal.
        self._selection_output_workspaces: dict[
            str, list[_SelectionOutputWorkspace]
        ] = {}
        self._selection_output_workspace_cursors: dict[str, int] = {}

        # CPU-offload construction is staged during layer execution and
        # committed after the complete prefill attention context.
        self._staged_segments: list[_StagedRequestLayerSegment] = []
        self._staged_segment_keys: set[tuple[str, str]] = set()

        # CPU-backed cluster pages are built by one serialized background
        # worker. The executor lives for one staged index transaction and is
        # closed by flush_staged_updates() or discard_staged_updates().
        self._cluster_build_executor: ThreadPoolExecutor | None = None
        self._pending_cluster_builds: deque[Future[RetroSpecClusterBlockTable]] = (
            deque()
        )

        # Full verification retains only the current and next layer layouts.
        # Staging the next layer before returning the current plan overlaps its
        # H2D copy with the current layer's attention and MLP computation.
        self._full_verification_pipeline_active = False
        self._full_verification_request_ids: tuple[str, ...] = ()
        self._full_verification_layers: tuple[tuple[str, int], ...] = ()
        self._full_verification_layer_cursor = 0
        self._full_verification_device: torch.device | None = None
        self._full_verification_prefetched: _PrefetchedFullVerificationLayer | None = (
            None
        )

    def _stable_indexed_end(self, seq_len: int) -> int:
        """Return the exclusive end of tokens that may leave native GPU KV."""
        full_block_count = seq_len // self.block_size

        # Block zero remains the exact sink. Recent complete blocks and the
        # current partial block remain in the native exact-attention zone.
        stable_end_block = max(
            full_block_count - self.num_recent_blocks,
            1,
        )
        return stable_end_block * self.block_size

    def _segment_size_for_phase(self, is_prefill: bool) -> int:
        if is_prefill:
            return self.prefill_segment_size_tokens
        return self.generation_update_interval

    def _desired_indexed_end(
        self,
        seq_len: int,
        record: _RequestLayerIndex | None,
        is_prefill: bool,
        prefill_complete: bool = False,
    ) -> int:
        """Return the next complete prefill or generation index boundary."""
        stable_end = self._stable_indexed_end(seq_len)
        segment_size = self._segment_size_for_phase(is_prefill)

        if record is None or stable_end < record.indexed_end:
            indexed_start = self.block_size
        else:
            indexed_start = record.indexed_end

        available_tokens = max(stable_end - indexed_start, 0)
        if is_prefill and prefill_complete:
            complete_clusters = available_tokens // self.tokens_per_cluster
            return indexed_start + complete_clusters * self.tokens_per_cluster

        complete_segments = available_tokens // segment_size
        return indexed_start + complete_segments * segment_size

    def _clustering_phases(
        self,
        num_tokens: int,
        is_prefill: bool,
        prefill_complete: bool,
    ) -> tuple[tuple[int, int], ...]:
        """Split one update into regular segments and an adaptive prefill tail.

        Each tuple contains ``(num_phase_tokens, segment_size)``. Standard
        prefill segments retain the configured target size. Once prefill is
        complete, the remaining cluster-aligned tail becomes one shorter
        segment instead of waiting for generation updates.
        """
        if num_tokens <= 0:
            return ()

        segment_size = self._segment_size_for_phase(is_prefill)
        if not is_prefill or not prefill_complete:
            if num_tokens % segment_size != 0:
                raise RuntimeError(
                    "New indexed region must contain complete phase-specific segments"
                )
            return tuple(
                (segment_size, segment_size) for _ in range(num_tokens // segment_size)
            )

        regular_tokens = num_tokens // segment_size * segment_size
        tail_tokens = num_tokens - regular_tokens
        phases: list[tuple[int, int]] = []

        phases.extend(
            (segment_size, segment_size) for _ in range(regular_tokens // segment_size)
        )
        if tail_tokens:
            if tail_tokens % self.tokens_per_cluster != 0:
                raise RuntimeError(
                    "Adaptive prefill tail must contain complete clusters"
                )
            phases.append((tail_tokens, tail_tokens))

        return tuple(phases)

    def needs_update(
        self,
        request_id: str,
        seq_len: int,
        layer_names: Sequence[str],
        is_prefill: bool,
        prefill_complete: bool = False,
    ) -> bool:
        if prefill_complete and not is_prefill:
            raise ValueError("prefill_complete requires is_prefill")

        for layer_name in layer_names:
            layer_indices = self._indices.get(layer_name)
            record = None if layer_indices is None else layer_indices.get(request_id)
            desired_end = self._desired_indexed_end(
                seq_len, record, is_prefill, prefill_complete
            )

            if record is None or record.indexed_end != desired_end:
                return True

        return False

    def get_fully_stored_indexed_end(
        self,
        request_id: str,
        layer_names: Sequence[str],
    ) -> int:
        """Return the token boundary stored successfully by every layer."""
        layer_names = tuple(layer_names)
        if not layer_names:
            return self.block_size

        indexed_ends: list[int] = []
        for layer_name in layer_names:
            record = self._indices.get(layer_name, {}).get(request_id)
            if record is None:
                return self.block_size
            indexed_ends.append(record.indexed_end)

        return min(indexed_ends)

    def remove_requests(self, request_ids: Sequence[str]) -> None:
        self.cluster_store.wait_for_resident_prefetches()
        request_ids = tuple(request_ids)
        self._gpu_index_residency.invalidate_requests(request_ids)

        for request_id in request_ids:
            self._first_draft_warm_layers_by_request.pop(request_id, None)

        for layer_name, layer_indices in self._indices.items():
            for request_id in request_ids:
                record = layer_indices.pop(request_id, None)
                if record is None:
                    continue

                self._free_record(layer_name, record)

    def prepare_full_verification(
        self,
        request_ids: Sequence[str],
        context_lens: Sequence[int],
        layer_names: Sequence[str],
    ) -> None:
        """Roll back clustered state that extends past committed context."""
        request_ids = tuple(request_ids)
        context_lens = tuple(int(context_len) for context_len in context_lens)

        if len(context_lens) != len(request_ids):
            raise ValueError("context_lens must match request_ids")
        if any(context_len < 0 for context_len in context_lens):
            raise ValueError("Full-verification context lengths must be non-negative")
        if self.has_staged_updates:
            raise RuntimeError(
                "Cannot prepare full verification while index updates are staged"
            )

        self.cluster_store.wait_for_resident_prefetches(layer_names)

        for layer_name in layer_names:
            layer_indices = self._indices.get(layer_name)
            if layer_indices is None:
                continue

            layer_changed = False
            for request_id, context_len in zip(request_ids, context_lens):
                record = layer_indices.get(request_id)
                if (
                    record is None
                    or not record.segments
                    or record.indexed_end <= context_len
                ):
                    continue

                self._free_record(layer_name, record)
                layer_indices[request_id] = self._empty_index()
                self._gpu_index_residency.discard_request_layer(layer_name, request_id)
                layer_changed = True

            if layer_changed:
                self._gpu_index_residency.invalidate_active_view(layer_name)

    def mark_first_draft_warmup(
        self,
        request_ids: Sequence[str],
        layer_names: Sequence[str],
    ) -> None:
        if not self.cluster_store.pin_memory:
            return

        layer_names = tuple(layer_names)
        if not layer_names:
            return

        for request_id in request_ids:
            self._first_draft_warm_layers_by_request.setdefault(
                request_id,
                set(),
            ).update(layer_names)

    def _get_first_draft_warmup_mask(
        self,
        request_ids: Sequence[str],
        layer_name: str,
        active_mask: torch.Tensor,
        warm_first_draft: bool,
    ) -> torch.Tensor | None:
        if not warm_first_draft or not self.cluster_store.pin_memory:
            return None

        pending_rows = [
            layer_name in self._first_draft_warm_layers_by_request.get(request_id, ())
            for request_id in request_ids
        ]
        if not any(pending_rows):
            return None

        pending_mask = torch.tensor(
            pending_rows,
            dtype=torch.bool,
            device=active_mask.device,
        )
        return pending_mask & active_mask

    def complete_first_draft_warmup(
        self,
        request_ids: Sequence[str],
        layer_names: Sequence[str],
        active_mask: torch.Tensor,
    ) -> None:
        """Consume warmup state after every active request finishes its draft."""
        completed_layers = set(layer_names)
        has_pending = any(
            completed_layers
            & self._first_draft_warm_layers_by_request.get(request_id, set())
            for request_id in request_ids
        )
        if not has_pending:
            return

        active_rows = active_mask.detach().to(device="cpu").tolist()

        for request_id, active in zip(request_ids, active_rows):
            if not active:
                continue

            pending_layers = self._first_draft_warm_layers_by_request.get(request_id)
            if pending_layers is None:
                continue

            pending_layers.difference_update(completed_layers)
            if not pending_layers:
                self._first_draft_warm_layers_by_request.pop(request_id, None)

    def begin_proposal(self, request_ids: Sequence[str]) -> None:
        if self._proposal_active:
            raise RuntimeError("Segmented token index proposal is already active")
        if self._staged_segments:
            raise RuntimeError(
                "Cannot begin a proposal before staged index updates are flushed"
            )
        request_ids = tuple(request_ids)
        self._gpu_index_residency.activate(request_ids)
        self._selection_output_workspace_cursors.clear()
        self._proposal_active = True
        self._proposal_request_ids = request_ids

    def end_proposal(self) -> None:
        if not self._proposal_active:
            raise RuntimeError("Segmented token index proposal is not active")

        try:
            self._gpu_index_residency.deactivate()
        finally:
            self._proposal_active = False
            self._proposal_request_ids = ()

    def begin_full_verification_residency(
        self,
        request_ids: Sequence[str],
    ) -> None:
        if self._proposal_active:
            raise RuntimeError("Full-verification residency cannot overlap a proposal")
        if self.has_staged_updates:
            raise RuntimeError(
                "Full-verification residency cannot begin with staged index updates"
            )

        self._gpu_index_residency.activate(request_ids)

    def end_full_verification_residency(self) -> None:
        self._gpu_index_residency.deactivate()

    def _empty_index(
        self,
        indexed_end: int | None = None,
    ) -> _RequestLayerIndex:
        if indexed_end is None:
            indexed_end = self.block_size

        return _RequestLayerIndex(
            segments=[],
            num_clusters=0,
            indexed_end=indexed_end,
            full_verification_page_ids_cpu=None,
        )

    def _free_record(
        self,
        layer_name: str,
        record: _RequestLayerIndex,
    ) -> None:
        for segment in record.segments:
            self.cluster_store.free(layer_name, segment.cluster_blocks)

    @property
    def has_staged_updates(self) -> bool:
        return bool(self._staged_segments)

    def _get_cluster_build_executor(self) -> ThreadPoolExecutor:
        if self._cluster_build_executor is None:
            self._cluster_build_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="retrospec-cluster-page",
            )

        return self._cluster_build_executor

    def _submit_cluster_build(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        staged_clusters: RetroSpecStagedClusterInput,
    ) -> Future[RetroSpecClusterBlockTable]:
        executor = self._get_cluster_build_executor()
        build_future = executor.submit(
            self.cluster_store.store_staged_clusters,
            layer_name=layer_name,
            request_id=request_id,
            cluster_start=cluster_start,
            staged=staged_clusters,
        )
        self._pending_cluster_builds.append(build_future)
        if self.performance_stats is not None:
            self.performance_stats.observe_peak(
                "cluster_build_queue_depth",
                len(self._pending_cluster_builds),
            )
        return build_future

    def _wait_for_cluster_build_slot(self) -> None:
        """Bound queued builds before allocating another pinned staging input."""
        while self._pending_cluster_builds and self._pending_cluster_builds[0].done():
            self._pending_cluster_builds.popleft().result()

        if len(self._pending_cluster_builds) < self.max_pending_cluster_builds:
            return

        self._pending_cluster_builds.popleft().result()

    def _stage_clustering_phase(
        self,
        layer_name: str,
        request_id: str,
        indexed_start: int,
        cluster_start: int,
        segment_size: int,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> int:
        """Overlap one bounded segment's D2H copy, k-means, and CPU build."""
        self._wait_for_cluster_build_slot()
        staged_token_kv = self.cluster_store.stage_token_kv(token_keys, token_values)

        try:
            kmeans_timer = (
                None
                if self.performance_stats is None
                else self.performance_stats.start_cuda_timer("segmented_kmeans")
            )
            phase_result = segmented_kmeans(
                token_keys=token_keys,
                token_values=token_values,
                segment_size=segment_size,
                items_per_cluster=self.tokens_per_cluster,
                num_iterations=self.num_kmeans_iterations,
            )
            if self.performance_stats is not None:
                self.performance_stats.stop_cuda_timer(kmeans_timer)
                self.performance_stats.add_counter(
                    "indexed_token_layers", token_keys.shape[1]
                )
                self.performance_stats.add_counter(
                    "cluster_slots_built", phase_result.cluster_sizes.numel()
                )
        except BaseException:
            self.cluster_store.discard_staged_token_kv(staged_token_kv)
            raise

        try:
            staged_summary = self._gpu_index_residency.stage_cluster_summary(
                phase_result.cluster_keys,
                phase_result.cluster_values,
                phase_result.cluster_sizes,
            )
        except BaseException:
            self.cluster_store.discard_staged_token_kv(staged_token_kv)
            raise

        try:
            staged_clusters = self.cluster_store.finish_stage_clusters(
                staged_token_kv,
                phase_result.assignments,
                phase_result.cluster_sizes,
                phase_result.token_offsets_in_cluster,
            )
        except BaseException:
            self.cluster_store.discard_staged_token_kv(staged_token_kv)
            self._gpu_index_residency.discard_cluster_summary(staged_summary)
            raise

        try:
            build_future = self._submit_cluster_build(
                layer_name=layer_name,
                request_id=request_id,
                cluster_start=cluster_start,
                staged_clusters=staged_clusters,
            )
        except BaseException:
            self.cluster_store.discard_staged_clusters(staged_clusters)
            self._gpu_index_residency.discard_cluster_summary(staged_summary)
            raise

        indexed_end = indexed_start + token_keys.shape[1]
        self._staged_segments.append(
            _StagedRequestLayerSegment(
                layer_name=layer_name,
                request_id=request_id,
                indexed_start=indexed_start,
                indexed_end=indexed_end,
                cluster_start=cluster_start,
                cluster_summary=staged_summary,
                build_future=build_future,
            )
        )
        return phase_result.cluster_sizes.shape[1]

    def _release_built_segments(
        self,
        built_segments: Sequence[
            tuple[
                _StagedRequestLayerSegment,
                RetroSpecClusterSummary,
                RetroSpecClusterBlockTable,
            ]
        ],
    ) -> None:
        for staged_segment, _, cluster_blocks in built_segments:
            self.cluster_store.free(staged_segment.layer_name, cluster_blocks)

    @staticmethod
    def _append_full_verification_page_descriptor(
        record: _RequestLayerIndex,
        page_ids: torch.Tensor,
    ) -> None:
        """Append one segment to the persistent CPU full-verify descriptor."""
        if page_ids.device.type != "cpu":
            raise ValueError("Full-verification page descriptors must reside on CPU")
        if page_ids.ndim < 2:
            raise ValueError("Cluster page IDs must include a KV-head dimension")

        num_kv_heads = page_ids.shape[0]
        previous = record.full_verification_page_ids_cpu
        if previous is not None and previous.shape[0] != num_kv_heads:
            raise RuntimeError("Full-verification descriptor changed KV-head count")

        per_head_pages: list[torch.Tensor] = []
        for head_index in range(num_kv_heads):
            new_pages = page_ids[head_index].reshape(-1)
            new_pages = new_pages[new_pages >= 0]
            if previous is not None:
                old_pages = previous[head_index]
                old_pages = old_pages[old_pages >= 0]
                new_pages = torch.cat((old_pages, new_pages))
            per_head_pages.append(new_pages)

        max_num_pages = max((pages.numel() for pages in per_head_pages), default=0)
        descriptor = torch.full(
            (num_kv_heads, max_num_pages),
            -1,
            dtype=torch.int64,
            device="cpu",
        )
        for head_index, pages in enumerate(per_head_pages):
            descriptor[head_index, : pages.numel()].copy_(pages)
        record.full_verification_page_ids_cpu = descriptor

    def _publish_built_segments(
        self,
        built_segments: Sequence[
            tuple[
                _StagedRequestLayerSegment,
                RetroSpecClusterSummary,
                RetroSpecClusterBlockTable,
            ]
        ],
    ) -> None:
        """Atomically publish CPU records and persistent GPU index segments."""
        pending_records: dict[tuple[str, str], _RequestLayerIndex] = {}
        resident_segments: list[RetroSpecResidentSegment] = []

        for staged_segment, summary, cluster_blocks in built_segments:
            key = (staged_segment.layer_name, staged_segment.request_id)
            record = pending_records.get(key)

            if record is None:
                current_record = self._indices.get(
                    staged_segment.layer_name,
                    {},
                ).get(staged_segment.request_id)

                if current_record is None:
                    record = self._empty_index()
                else:
                    record = _RequestLayerIndex(
                        segments=list(current_record.segments),
                        num_clusters=current_record.num_clusters,
                        indexed_end=current_record.indexed_end,
                        full_verification_page_ids_cpu=(
                            None
                            if current_record.full_verification_page_ids_cpu is None
                            else current_record.full_verification_page_ids_cpu.clone()
                        ),
                    )

                pending_records[key] = record

            if record.indexed_end != staged_segment.indexed_start:
                raise RuntimeError(
                    "Built RetroSpec segment no longer follows the indexed prefix"
                )
            if record.num_clusters != staged_segment.cluster_start:
                raise RuntimeError(
                    "Built RetroSpec segment cluster offset is no longer current"
                )

            block_metadata = self.cluster_store.get_cluster_block_metadata(
                layer_name=staged_segment.layer_name,
                cluster_ids=cluster_blocks.cluster_ids,
                device=torch.device("cpu"),
            )
            resident_segments.append(
                self._gpu_index_residency.build_resident_segment(
                    layer_name=staged_segment.layer_name,
                    request_id=staged_segment.request_id,
                    indexed_start=staged_segment.indexed_start,
                    indexed_end=staged_segment.indexed_end,
                    cluster_start=staged_segment.cluster_start,
                    staged_summary=staged_segment.cluster_summary,
                    cluster_ids=cluster_blocks.cluster_ids,
                    cluster_page_ids=block_metadata.page_ids,
                    cluster_page_token_counts=block_metadata.page_token_counts,
                )
            )
            self._append_full_verification_page_descriptor(
                record,
                block_metadata.page_ids,
            )

            record.segments.append(
                _RequestLayerSegment(
                    indexed_start=staged_segment.indexed_start,
                    indexed_end=staged_segment.indexed_end,
                    cluster_start=staged_segment.cluster_start,
                    cluster_keys=summary.cluster_keys,
                    cluster_values=summary.cluster_values,
                    cluster_token_counts=summary.cluster_token_counts,
                    cluster_blocks=cluster_blocks,
                )
            )
            record.num_clusters += summary.cluster_token_counts.shape[1]
            record.indexed_end = staged_segment.indexed_end

        # Construct replacement mappings before publishing self._indices so a
        # validation failure cannot expose only a subset of the transaction.
        new_indices = dict(self._indices)
        changed_layers: dict[str, dict[str, _RequestLayerIndex]] = {}

        for (layer_name, request_id), record in pending_records.items():
            layer_indices = changed_layers.get(layer_name)
            if layer_indices is None:
                layer_indices = dict(self._indices.get(layer_name, {}))
                changed_layers[layer_name] = layer_indices

            layer_indices[request_id] = record

        for layer_name, layer_indices in changed_layers.items():
            new_indices[layer_name] = layer_indices

        self._gpu_index_residency.publish_resident_segments(resident_segments)
        self._indices = new_indices

    def flush_staged_updates(self) -> None:
        """Wait for background page builds and publish them atomically."""
        staged_segments = self._staged_segments
        executor = self._cluster_build_executor

        self._staged_segments = []
        self._staged_segment_keys.clear()
        self._cluster_build_executor = None
        self._pending_cluster_builds.clear()

        if not staged_segments:
            if executor is not None:
                executor.shutdown(wait=True)
            return

        if executor is None:
            raise RuntimeError(
                "Staged RetroSpec segments have no cluster build executor"
            )

        built_segments: list[
            tuple[
                _StagedRequestLayerSegment,
                RetroSpecClusterSummary,
                RetroSpecClusterBlockTable,
            ]
        ] = []
        first_error: BaseException | None = None

        try:
            for staged_segment in staged_segments:
                summary: RetroSpecClusterSummary | None = None
                cluster_blocks: RetroSpecClusterBlockTable | None = None

                try:
                    summary = self._gpu_index_residency.finish_cluster_summary(
                        staged_segment.cluster_summary
                    )
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc

                try:
                    cluster_blocks = staged_segment.build_future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc

                if summary is not None and cluster_blocks is not None:
                    built_segments.append((staged_segment, summary, cluster_blocks))
                elif cluster_blocks is not None:
                    try:
                        self.cluster_store.free(
                            staged_segment.layer_name,
                            cluster_blocks,
                        )
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc

            if first_error is not None:
                self._release_built_segments(built_segments)
                raise first_error

            try:
                self._publish_built_segments(built_segments)
            except BaseException:
                self._release_built_segments(built_segments)
                raise
        finally:
            executor.shutdown(wait=True)

    def discard_staged_updates(self) -> None:
        """Drain background builds and release unpublished cluster pages."""
        staged_segments = self._staged_segments
        executor = self._cluster_build_executor

        self._staged_segments = []
        self._staged_segment_keys.clear()
        self._cluster_build_executor = None
        self._pending_cluster_builds.clear()

        cleanup_error: BaseException | None = None

        try:
            for staged_segment in staged_segments:
                try:
                    self._gpu_index_residency.discard_cluster_summary(
                        staged_segment.cluster_summary
                    )
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc

                try:
                    cluster_blocks = staged_segment.build_future.result()
                except BaseException:
                    # A failed store_clusters() call releases allocations made
                    # before it raises. The original prefill error takes priority.
                    continue

                try:
                    self.cluster_store.free(
                        staged_segment.layer_name,
                        cluster_blocks,
                    )
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        if cleanup_error is not None:
            raise cleanup_error

    def build_or_update(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        is_prefill: Sequence[bool],
        rows: Sequence[int],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        defer_cpu_store: bool = False,
        prefill_complete: Sequence[bool] | None = None,
    ) -> None:
        """Cluster stable tokens and stage or store private cluster pages."""
        if len(request_ids) != len(seq_lens):
            raise ValueError("request_ids and seq_lens must have equal length")
        if len(request_ids) != len(is_prefill):
            raise ValueError("request_ids and is_prefill must have equal length")
        if prefill_complete is None:
            prefill_complete = (False,) * len(request_ids)
        if len(request_ids) != len(prefill_complete):
            raise ValueError("request_ids and prefill_complete must have equal length")
        if block_table.shape[0] != len(request_ids):
            raise ValueError("block_table batch size does not match request_ids")
        if key_cache.shape != value_cache.shape:
            raise ValueError("key_cache and value_cache must have equal shapes")
        if key_cache.shape[1] != self.block_size:
            raise ValueError(
                f"KV cache block size {key_cache.shape[1]} does not match "
                f"configured block size {self.block_size}"
            )
        if len(rows) != len(set(rows)):
            raise ValueError("RetroSpec index build rows must be unique")

        layer_indices = self._indices.setdefault(layer_name, {})
        layer_changed = False

        for row in rows:
            if not 0 <= row < len(request_ids):
                raise IndexError("RetroSpec index build row is out of range")

            request_id = request_ids[row]
            seq_len = seq_lens[row]
            request_is_prefill = bool(is_prefill[row])
            request_prefill_complete = bool(prefill_complete[row])
            if request_prefill_complete and not request_is_prefill:
                raise ValueError("prefill_complete requires is_prefill")

            record = layer_indices.get(request_id)
            desired_end = self._desired_indexed_end(
                seq_len,
                record,
                request_is_prefill,
                request_prefill_complete,
            )

            staged_key = (layer_name, request_id)
            if staged_key in self._staged_segment_keys:
                raise RuntimeError(
                    "A RetroSpec request/layer segment is already staged"
                )

            if record is not None and desired_end < record.indexed_end:
                self._gpu_index_residency.discard_request_layer(layer_name, request_id)
                self._free_record(layer_name, record)
                record = self._empty_index()
                layer_indices[request_id] = record
                layer_changed = True

            indexed_start = self.block_size if record is None else record.indexed_end

            if desired_end <= indexed_start:
                if record is None:
                    layer_indices[request_id] = self._empty_index()
                    layer_changed = True
                continue

            num_new_tokens = desired_end - indexed_start
            clustering_phases = self._clustering_phases(
                num_new_tokens,
                request_is_prefill,
                request_prefill_complete,
            )

            num_kv_heads = key_cache.shape[2]
            head_size = key_cache.shape[3]
            cluster_start = 0 if record is None else record.num_clusters
            self._staged_segment_keys.add(staged_key)
            phase_start = indexed_start

            for phase_tokens, phase_segment_size in clustering_phases:
                phase_end = phase_start + phase_tokens
                first_logical_block = phase_start // self.block_size
                logical_block_end = phase_end // self.block_size
                logical_block_ids = torch.arange(
                    first_logical_block,
                    logical_block_end,
                    dtype=torch.int64,
                    device=block_table.device,
                )
                physical_block_ids = (
                    block_table[row].index_select(0, logical_block_ids).to(torch.int64)
                )
                key_blocks = key_cache.index_select(0, physical_block_ids)
                value_blocks = value_cache.index_select(0, physical_block_ids)
                token_keys = (
                    key_blocks.reshape(phase_tokens, num_kv_heads, head_size)
                    .transpose(0, 1)
                    .contiguous()
                )
                token_values = (
                    value_blocks.reshape(phase_tokens, num_kv_heads, head_size)
                    .transpose(0, 1)
                    .contiguous()
                )

                num_phase_clusters = self._stage_clustering_phase(
                    layer_name=layer_name,
                    request_id=request_id,
                    indexed_start=phase_start,
                    cluster_start=cluster_start,
                    segment_size=phase_segment_size,
                    token_keys=token_keys,
                    token_values=token_values,
                )
                phase_start = phase_end
                cluster_start += num_phase_clusters

            if phase_start != desired_end:
                raise RuntimeError("Clustering phases do not cover the update")
            if not defer_cpu_store:
                self.flush_staged_updates()

        if layer_changed:
            self._gpu_index_residency.invalidate_active_view(layer_name)

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
        workspace: _ClusterSelectionWorkspace | None = None,
    ) -> torch.Tensor:
        """Score clusters with a fused CUDA reduction when available."""
        logits = cls._compute_cluster_logits(
            query,
            cluster_keys,
            None if workspace is None else workspace.logits,
        )

        if logits.device.type == "cuda":
            return reduce_grouped_cluster_scores(
                logits,
                cluster_mask,
                cluster_token_counts,
                scale,
                None if workspace is None else workspace.scores,
                None if workspace is None else workspace.softmax_lse,
                None if workspace is None else workspace.ranking_scores,
            )

        if workspace is not None:
            raise ValueError("Cluster selection workspace is CUDA-only")

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

    def _maximum_first_draft_warmup_width(self, num_clusters: int) -> int:
        max_retrieval, _, _ = self._maximum_zone_widths(num_clusters)
        return min(
            max_retrieval * self.first_draft_warmup_multiplier,
            num_clusters,
        )

    def _get_cluster_selection_workspace(
        self,
        query: torch.Tensor,
        num_kv_heads: int,
        num_clusters: int,
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
        max_first_draft_warmup = self._maximum_first_draft_warmup_width(num_clusters)
        max_ranked_clusters = max(max_total_compute, max_first_draft_warmup)

        logits_shape = (
            batch_size,
            num_kv_heads,
            queries_per_kv,
            num_clusters,
        )
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
        topk_shape = (
            batch_size,
            num_kv_heads,
            max_ranked_clusters,
        )

        workspace = self._cluster_selection_workspace
        if (
            workspace is not None
            and workspace.logits.device == query.device
            and workspace.logits.shape == logits_shape
            and workspace.scores.shape == scores_shape
            and workspace.softmax_lse.shape == lse_shape
            and workspace.ranking_scores.shape == scores_shape
            and workspace.candidate_counts.shape == scores_shape[:2]
            and workspace.topk_values.shape == topk_shape
            and workspace.topk_indices.shape == topk_shape
        ):
            return workspace

        workspace = _ClusterSelectionWorkspace(
            logits=torch.empty(
                logits_shape,
                dtype=torch.float32,
                device=query.device,
            ),
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
            ranking_scores=torch.empty(
                scores_shape,
                dtype=torch.float32,
                device=query.device,
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
        self._cluster_selection_workspace = workspace
        return workspace

    def _score_resident_view(
        self,
        query: torch.Tensor,
        view: RetroSpecResidentBatchView,
        scale: float,
        num_kv_heads: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        _ClusterSelectionWorkspace | None,
    ]:
        workspace = None
        if query.device.type == "cuda":
            workspace = self._get_cluster_selection_workspace(
                query, num_kv_heads, view.max_num_clusters
            )

        arena = view.arena
        if arena is None:
            shape = (query.shape[0], num_kv_heads, view.max_num_clusters)
            if workspace is None:
                scores = torch.zeros(shape, dtype=torch.float32, device=query.device)
                ranking_scores = torch.full_like(scores, float("-inf"))
                candidate_counts = torch.zeros(
                    shape[:2], dtype=torch.int32, device=query.device
                )
            else:
                scores = workspace.scores.zero_()
                ranking_scores = workspace.ranking_scores.fill_(float("-inf"))
                candidate_counts = workspace.candidate_counts.zero_()
            return scores, ranking_scores, candidate_counts, workspace

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
                logits=workspace.logits,
                output=workspace.scores,
                softmax_lse=workspace.softmax_lse,
                ranking_output=workspace.ranking_scores,
                candidate_counts=workspace.candidate_counts,
            )
            return (
                scores,
                workspace.ranking_scores,
                workspace.candidate_counts,
                workspace,
            )

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
        ranking_scores = scores.masked_fill(~cluster_mask, float("-inf"))
        candidate_counts = cluster_mask.sum(dim=2, dtype=torch.int32)
        return scores, ranking_scores, candidate_counts, None

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

    def _select_cluster_zones(
        self,
        cluster_scores: torch.Tensor,
        ranking_scores: torch.Tensor,
        candidate_counts: torch.Tensor,
        view: RetroSpecResidentBatchView,
        first_draft_warmup_mask: torch.Tensor | None = None,
        warmup_page_budgets: torch.Tensor | None = None,
        workspace: _ClusterSelectionWorkspace | None = None,
    ) -> _PackedClusterZones:
        """Rank relevant clusters once and return compact zone indices."""
        if cluster_scores.ndim != 3:
            raise ValueError(
                "Cluster scores must have shape [batch, num_kv_heads, num_clusters]"
            )
        if ranking_scores.shape != cluster_scores.shape:
            raise ValueError("Ranking scores and cluster scores must match")
        if candidate_counts.shape != cluster_scores.shape[:2]:
            raise ValueError("Candidate counts do not match cluster scores")

        batch_size, num_kv_heads, num_clusters = cluster_scores.shape
        (
            max_retrieval,
            max_estimation,
            max_expanded_retrieval,
        ) = self._maximum_zone_widths(num_clusters)
        max_total_compute = max_retrieval + max_estimation
        max_warmup = self._maximum_first_draft_warmup_width(num_clusters)

        warmup_enabled = first_draft_warmup_mask is not None
        if warmup_enabled:
            if first_draft_warmup_mask.shape != (batch_size,):
                raise ValueError("First-draft warmup mask has an unexpected shape")
            if first_draft_warmup_mask.device != cluster_scores.device:
                raise ValueError("First-draft warmup mask must use the score device")
            if warmup_page_budgets is None:
                raise ValueError("First-draft warmup requires page budgets")
            if warmup_page_budgets.shape != (batch_size, num_kv_heads):
                raise ValueError("First-draft page budgets have an unexpected shape")
            if warmup_page_budgets.device != cluster_scores.device:
                raise ValueError("First-draft page budgets must use the score device")

        ranking_width = (
            max(max_total_compute, max_warmup) if warmup_enabled else max_total_compute
        )

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
            ranked_indices = torch.topk(
                ranking_scores,
                k=ranking_width,
                dim=2,
                largest=True,
                sorted=True,
            ).indices
        else:
            if cluster_scores is not workspace.scores:
                raise ValueError(
                    "Workspace ranking scores do not belong to cluster scores"
                )
            if ranking_scores is not workspace.ranking_scores:
                raise ValueError("Workspace ranking output does not match")
            if workspace.ranking_scores.shape != cluster_scores.shape:
                raise ValueError(
                    "Workspace ranking-score shape does not match cluster scores"
                )
            if workspace.topk_indices.shape[2] < ranking_width:
                raise ValueError("Workspace top-k capacity is too small")

            topk_values = workspace.topk_values[:, :, :ranking_width]
            topk_indices = workspace.topk_indices[:, :, :ranking_width]

            torch.topk(
                workspace.ranking_scores,
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
        (
            expanded_estimation_indices,
            expanded_estimation_mask,
        ) = self._slice_rank_range(
            ranked_indices,
            expanded_retrieval_counts,
            total_compute_counts,
            max_estimation,
        )

        if not warmup_enabled:
            first_draft_warmup_indices = ranked_indices[:, :, :0].contiguous()
            first_draft_warmup_cluster_mask = torch.empty_like(
                first_draft_warmup_indices,
                dtype=torch.bool,
            )
        else:
            assert first_draft_warmup_mask is not None
            assert warmup_page_budgets is not None
            warmup_counts = torch.minimum(
                retrieval_counts * self.first_draft_warmup_multiplier,
                candidate_counts,
            )
            (
                first_draft_warmup_indices,
                first_draft_warmup_cluster_mask,
            ) = self._slice_rank_range(
                ranked_indices,
                zero_counts,
                warmup_counts,
                max_warmup,
            )

            if view.arena is None or max_warmup == 0:
                first_draft_warmup_cluster_mask.zero_()
            else:
                arena = view.arena
                safe_slots = view.request_slot_ids.clamp_min(0)
                request_cluster_offsets = arena.cluster_offsets.index_select(
                    0, safe_slots
                )
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
                ranked_page_counts = cluster_page_counts.gather(
                    2,
                    first_draft_warmup_indices,
                )
                cumulative_pages = ranked_page_counts.cumsum(dim=2)
                first_draft_warmup_cluster_mask &= (
                    cumulative_pages <= warmup_page_budgets.unsqueeze(-1)
                )
                first_draft_warmup_cluster_mask &= first_draft_warmup_mask[
                    :, None, None
                ]
                first_draft_warmup_cluster_mask &= (
                    view.request_slot_ids[:, None, None] >= 0
                )

        return _PackedClusterZones(
            sparse_retrieval_indices=sparse_retrieval_indices,
            sparse_retrieval_mask=sparse_retrieval_mask,
            sparse_estimation_indices=sparse_estimation_indices,
            sparse_estimation_mask=sparse_estimation_mask,
            expanded_retrieval_indices=expanded_retrieval_indices,
            expanded_retrieval_mask=expanded_retrieval_mask,
            expanded_estimation_indices=expanded_estimation_indices,
            expanded_estimation_mask=expanded_estimation_mask,
            first_draft_warmup_indices=first_draft_warmup_indices,
            first_draft_warmup_mask=first_draft_warmup_cluster_mask,
        )

    @staticmethod
    def _pack_bounded_mask_indices(
        mask: torch.Tensor,
        output_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack True positions using a synchronization-free upper bound."""
        if output_width < 0:
            raise ValueError("output_width must be non-negative")

        max_num_items = mask.shape[-1]
        leading_shape = mask.shape[:-1]
        output_width = min(output_width, max_num_items)

        if output_width == 0:
            empty_shape = (*leading_shape, 0)
            return (
                torch.empty(
                    empty_shape,
                    dtype=torch.int64,
                    device=mask.device,
                ),
                torch.empty(
                    empty_shape,
                    dtype=torch.bool,
                    device=mask.device,
                ),
            )

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

        packed_indices = torch.topk(
            candidates,
            k=output_width,
            dim=-1,
            largest=False,
            sorted=True,
        ).values

        packed_mask = packed_indices < max_num_items
        packed_indices.clamp_(
            min=0,
            max=max_num_items - 1,
        )

        return (
            packed_indices.contiguous(),
            packed_mask.contiguous(),
        )

    @staticmethod
    def _sum_selected_scores(
        cluster_scores: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sum probability mass represented by a compact retrieval zone."""
        if selected_indices.shape != selected_mask.shape:
            raise ValueError("Selected cluster indices and mask must match")

        if selected_indices.shape[2] == 0:
            return torch.zeros(
                cluster_scores.shape[:2],
                dtype=cluster_scores.dtype,
                device=cluster_scores.device,
            )

        selected_scores = cluster_scores.gather(
            dim=2,
            index=selected_indices,
        )
        selected_scores.masked_fill_(
            ~selected_mask,
            0.0,
        )
        return selected_scores.sum(dim=2)

    def _get_selection_output_workspace(
        self,
        layer_name: str,
        view: RetroSpecResidentBatchView,
        batch_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _SelectionOutputWorkspace:
        if device.type != "cuda":
            raise ValueError("Selection output workspace requires CUDA")

        sparse_retrieval_width, sparse_estimation_width, expanded_retrieval_width = (
            self._maximum_zone_widths(view.max_num_clusters)
        )
        next_cursor = self._selection_output_workspace_cursors.get(layer_name, 0)
        # A proposal may contain several sparse-verification rounds. Only the
        # current round's plans remain live, so recycle the fixed speculative
        # slots instead of growing the workspace or rejecting a later round.
        cursor = next_cursor % self.num_speculative_tokens

        layer_workspaces = self._selection_output_workspaces.setdefault(layer_name, [])
        if cursor == len(layer_workspaces):
            layer_workspaces.append(
                _SelectionOutputWorkspace.allocate(
                    batch_size=batch_size,
                    num_kv_heads=num_kv_heads,
                    sparse_retrieval_width=sparse_retrieval_width,
                    expanded_retrieval_width=expanded_retrieval_width,
                    sparse_estimation_width=sparse_estimation_width,
                    max_pages_per_cluster=view.max_pages_per_cluster,
                    head_size=head_size,
                    dtype=dtype,
                    device=device,
                )
            )

        workspace = layer_workspaces[cursor]
        if not workspace.matches(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            sparse_retrieval_width=sparse_retrieval_width,
            expanded_retrieval_width=expanded_retrieval_width,
            sparse_estimation_width=sparse_estimation_width,
            max_pages_per_cluster=view.max_pages_per_cluster,
            head_size=head_size,
            dtype=dtype,
            device=device,
        ):
            workspace = _SelectionOutputWorkspace.allocate(
                batch_size=batch_size,
                num_kv_heads=num_kv_heads,
                sparse_retrieval_width=sparse_retrieval_width,
                expanded_retrieval_width=expanded_retrieval_width,
                sparse_estimation_width=sparse_estimation_width,
                max_pages_per_cluster=view.max_pages_per_cluster,
                head_size=head_size,
                dtype=dtype,
                device=device,
            )
            layer_workspaces[cursor] = workspace

        self._selection_output_workspace_cursors[layer_name] = next_cursor + 1
        return workspace

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

    def _make_plan(
        self,
        layer_name: str,
        forced_exact_mask: torch.Tensor,
        cluster_zones: _PackedClusterZones,
        sparse_attn: torch.Tensor,
        expanded_attn: torch.Tensor,
        view: RetroSpecResidentBatchView,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[RetroSpecTokenSelectionPlan, _SelectionOutputWorkspace | None]:
        output_workspace = None
        if forced_exact_mask.device.type == "cuda":
            output_workspace = self._get_selection_output_workspace(
                layer_name=layer_name,
                view=view,
                batch_size=forced_exact_mask.shape[0],
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                device=forced_exact_mask.device,
            )

        per_head_forced_exact = forced_exact_mask.unsqueeze(1).expand(
            -1,
            num_kv_heads,
            -1,
        )

        # With an up-to-date segmented index, the exact primary zone consists
        # of the sink block, an incomplete segment, recent blocks and the
        # current partial block. This expression is a synchronization-free
        # upper bound for that union.
        max_unindexed_segment_tokens = max(
            self.prefill_segment_size_tokens,
            self.generation_update_interval,
        )
        max_primary_exact_tokens = min(
            forced_exact_mask.shape[1],
            max_unindexed_segment_tokens
            + (self.num_recent_blocks + 1) * self.block_size,
        )

        (
            primary_exact_token_indices,
            primary_exact_token_mask,
        ) = self._pack_bounded_mask_indices(
            per_head_forced_exact,
            max_primary_exact_tokens,
        )

        (
            sparse_exact_cluster_ids,
            sparse_exact_page_ids,
            sparse_exact_page_token_counts,
        ) = self._build_resident_exact_cluster_selection(
            view,
            cluster_zones.sparse_retrieval_indices,
            cluster_zones.sparse_retrieval_mask,
            None
            if output_workspace is None
            else output_workspace.sparse_exact_cluster_ids,
            None
            if output_workspace is None
            else output_workspace.sparse_exact_page_ids,
            None
            if output_workspace is None
            else output_workspace.sparse_exact_page_token_counts,
        )

        (
            expanded_exact_cluster_ids,
            expanded_exact_page_ids,
            expanded_exact_page_token_counts,
        ) = self._build_resident_exact_cluster_selection(
            view,
            cluster_zones.expanded_retrieval_indices,
            cluster_zones.expanded_retrieval_mask,
            None
            if output_workspace is None
            else output_workspace.expanded_exact_cluster_ids,
            None
            if output_workspace is None
            else output_workspace.expanded_exact_page_ids,
            None
            if output_workspace is None
            else output_workspace.expanded_exact_page_token_counts,
        )

        sparse_estimation_keys = None
        sparse_estimation_values = None
        sparse_estimation_token_counts = None
        if output_workspace is not None:
            sparse_width = output_workspace.sparse_estimation_width
            sparse_estimation_keys = output_workspace.draft_estimation_keys[
                :, :, :sparse_width
            ]
            sparse_estimation_values = output_workspace.draft_estimation_values[
                :, :, :sparse_width
            ]
            sparse_estimation_token_counts = (
                output_workspace.draft_estimation_token_counts[:, :, :sparse_width]
            )

        (
            sparse_estimation_keys,
            sparse_estimation_values,
            sparse_estimation_token_counts,
        ) = self._build_resident_estimation_selection(
            view,
            cluster_zones.sparse_estimation_indices,
            cluster_zones.sparse_estimation_mask,
            head_size,
            dtype,
            sparse_estimation_keys,
            sparse_estimation_values,
            sparse_estimation_token_counts,
        )
        (
            expanded_estimation_keys,
            expanded_estimation_values,
            expanded_estimation_token_counts,
        ) = self._build_resident_estimation_selection(
            view,
            cluster_zones.expanded_estimation_indices,
            cluster_zones.expanded_estimation_mask,
            head_size,
            dtype,
            None
            if output_workspace is None
            else output_workspace.expanded_estimation_keys,
            None
            if output_workspace is None
            else output_workspace.expanded_estimation_values,
            None
            if output_workspace is None
            else output_workspace.expanded_estimation_token_counts,
        )

        return (
            RetroSpecTokenSelectionPlan(
                layer_name=layer_name,
                primary_exact_token_indices=primary_exact_token_indices,
                primary_exact_token_mask=primary_exact_token_mask,
                sparse_exact_cluster_ids=sparse_exact_cluster_ids,
                sparse_exact_page_ids=sparse_exact_page_ids,
                sparse_exact_page_token_counts=sparse_exact_page_token_counts,
                sparse_estimation_keys=sparse_estimation_keys,
                sparse_estimation_values=sparse_estimation_values,
                sparse_estimation_token_counts=sparse_estimation_token_counts,
                expanded_exact_cluster_ids=expanded_exact_cluster_ids,
                expanded_exact_page_ids=expanded_exact_page_ids,
                expanded_exact_page_token_counts=expanded_exact_page_token_counts,
                expanded_estimation_keys=expanded_estimation_keys,
                expanded_estimation_values=expanded_estimation_values,
                expanded_estimation_token_counts=expanded_estimation_token_counts,
                sparse_attn=sparse_attn,
                expanded_attn=expanded_attn,
            ),
            output_workspace,
        )

    def _gather_selected_tokens(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_kv_heads, max_num_tokens = token_indices.shape

        logical_block_ids = torch.div(
            token_indices,
            self.block_size,
            rounding_mode="floor",
        )
        block_offsets = token_indices % self.block_size

        expanded_block_table = block_table[:, None, :].expand(
            batch_size,
            num_kv_heads,
            -1,
        )
        physical_block_ids = expanded_block_table.gather(
            dim=2,
            index=logical_block_ids,
        ).to(torch.int64)

        head_ids = torch.arange(
            num_kv_heads,
            dtype=torch.int64,
            device=cache.device,
        )[None, :, None].expand(
            batch_size,
            num_kv_heads,
            max_num_tokens,
        )

        selected = cache[
            physical_block_ids,
            block_offsets,
            head_ids,
        ]
        selected.masked_fill_(~token_mask.unsqueeze(-1), 0.0)
        return selected.contiguous()

    def _materialize_draft_selection(
        self,
        plan: RetroSpecTokenSelectionPlan,
        output_workspace: _SelectionOutputWorkspace | None,
        view: RetroSpecResidentBatchView,
        cluster_zones: _PackedClusterZones,
        cluster_scores: torch.Tensor,
        has_clusters: torch.Tensor,
        head_size: int,
        dtype: torch.dtype,
        active_mask: torch.Tensor,
        include_pending_resident: bool,
    ) -> RetroSpecTokenAttentionSelection:
        """Use resident retrieval clusters and estimate selected cache misses."""
        if (
            plan.sparse_exact_page_ids.numel() == 0
            or plan.sparse_exact_page_ids.device.type != "cuda"
        ):
            return self._materialize_token_selection(
                plan,
                RetroSpecAttentionLevel.SPARSE,
            )
        if output_workspace is None:
            raise RuntimeError("CUDA draft selection requires an output workspace")

        resolve_mode = (
            "resident_pending" if include_pending_resident else "resident_only"
        )
        resolved_pages = self.cluster_store.resolve_cluster_blocks(
            layer_name=plan.layer_name,
            cluster_ids=plan.sparse_exact_cluster_ids,
            logical_page_ids=plan.sparse_exact_page_ids,
            mode=resolve_mode,
        )

        hit_cluster_mask = (
            cluster_zones.sparse_retrieval_mask & resolved_pages.hit_cluster_mask
        )
        miss_cluster_mask = (
            cluster_zones.sparse_retrieval_mask & resolved_pages.miss_cluster_mask
        )

        draft_exact_page_token_counts = output_workspace.draft_exact_page_token_counts
        draft_exact_page_token_counts.copy_(plan.sparse_exact_page_token_counts)
        draft_exact_page_token_counts.masked_fill_(
            ~hit_cluster_mask.unsqueeze(-1),
            0,
        )

        sparse_width = output_workspace.sparse_estimation_width
        retrieval_width = output_workspace.sparse_retrieval_width
        estimation_keys = output_workspace.draft_estimation_keys
        estimation_values = output_workspace.draft_estimation_values
        estimation_token_counts = output_workspace.draft_estimation_token_counts
        self._build_resident_estimation_selection(
            view,
            cluster_zones.sparse_retrieval_indices,
            miss_cluster_mask,
            head_size,
            dtype,
            estimation_keys[:, :, sparse_width : sparse_width + retrieval_width],
            estimation_values[:, :, sparse_width : sparse_width + retrieval_width],
            estimation_token_counts[
                :, :, sparse_width : sparse_width + retrieval_width
            ],
        )

        hit_attn_by_head = self._sum_selected_scores(
            cluster_scores,
            cluster_zones.sparse_retrieval_indices,
            hit_cluster_mask,
        )

        hit_gate_ready_by_head = (
            cluster_zones.sparse_retrieval_mask & resolved_pages.hit_gate_ready_mask
        ).any(dim=2)
        use_hit_attn = has_clusters & hit_gate_ready_by_head
        hit_attn_by_head = torch.where(
            use_hit_attn,
            hit_attn_by_head,
            torch.ones_like(hit_attn_by_head),
        )

        hit_attn = hit_attn_by_head.mean(dim=1)
        hit_attn = torch.where(
            active_mask,
            hit_attn,
            torch.ones_like(hit_attn),
        )

        primary_token_counts = plan.primary_exact_token_mask.sum(
            dim=2,
            dtype=torch.int32,
        )
        clustered_token_counts = draft_exact_page_token_counts.sum(
            dim=(2, 3),
            dtype=torch.int32,
        )
        exact_token_counts = (
            primary_token_counts + clustered_token_counts
        ).contiguous()

        return RetroSpecTokenAttentionSelection(
            exact_cluster_ids=plan.sparse_exact_cluster_ids,
            exact_page_ids=plan.sparse_exact_page_ids,
            exact_page_token_counts=draft_exact_page_token_counts,
            exact_token_counts=exact_token_counts,
            estimation_keys=estimation_keys,
            estimation_values=estimation_values,
            estimation_token_counts=estimation_token_counts,
            attention_mass=hit_attn,
            plan=plan,
            resolved_pages=resolved_pages,
        )

    def prefetch_sparse_verification(
        self,
        plan: RetroSpecTokenSelectionPlan,
        active_mask: torch.Tensor,
    ) -> None:
        """Asynchronously admit a draft plan's sparse pages into the GPU cache."""
        if not self.cluster_store.pin_memory:
            return

        cluster_ids = plan.sparse_exact_cluster_ids

        if active_mask.ndim != 1 or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a one-dimensional boolean tensor")
        if active_mask.shape != (cluster_ids.shape[0],):
            raise ValueError("active_mask does not match the selection batch size")
        if active_mask.device != cluster_ids.device:
            raise ValueError("active_mask and selection plan must use one device")
        if cluster_ids.numel() == 0 or plan.sparse_exact_page_ids.numel() == 0:
            return

        active_cluster_mask = active_mask[:, None, None]
        prefetch_cluster_ids = cluster_ids.masked_fill(~active_cluster_mask, -1)

        self.cluster_store.prefetch_resident_clusters(
            layer_name=plan.layer_name,
            cluster_ids=prefetch_cluster_ids,
        )

    def _materialize_token_selection(
        self,
        plan: RetroSpecTokenSelectionPlan,
        level: RetroSpecAttentionLevel,
    ) -> RetroSpecTokenAttentionSelection:
        if level == RetroSpecAttentionLevel.SPARSE:
            exact_cluster_ids = plan.sparse_exact_cluster_ids
            exact_page_ids = plan.sparse_exact_page_ids
            exact_page_token_counts = plan.sparse_exact_page_token_counts
            estimation_keys = plan.sparse_estimation_keys
            estimation_values = plan.sparse_estimation_values
            estimation_token_counts = plan.sparse_estimation_token_counts
            attention_mass = plan.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            exact_cluster_ids = plan.expanded_exact_cluster_ids
            exact_page_ids = plan.expanded_exact_page_ids
            exact_page_token_counts = plan.expanded_exact_page_token_counts
            estimation_keys = plan.expanded_estimation_keys
            estimation_values = plan.expanded_estimation_values
            estimation_token_counts = plan.expanded_estimation_token_counts
            attention_mass = plan.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        primary_token_counts = plan.primary_exact_token_mask.sum(
            dim=2,
            dtype=torch.int32,
        )
        clustered_token_counts = exact_page_token_counts.sum(
            dim=(2, 3),
            dtype=torch.int32,
        )
        exact_token_counts = (
            primary_token_counts + clustered_token_counts
        ).contiguous()

        return RetroSpecTokenAttentionSelection(
            exact_cluster_ids=exact_cluster_ids,
            exact_page_ids=exact_page_ids,
            exact_page_token_counts=exact_page_token_counts,
            exact_token_counts=exact_token_counts,
            estimation_keys=estimation_keys,
            estimation_values=estimation_values,
            estimation_token_counts=estimation_token_counts,
            attention_mass=attention_mass,
            plan=plan,
            resolved_pages=None,
        )

    def materialize_exact_reference(
        self,
        selection: RetroSpecTokenAttentionSelection,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Materialize exact KV for CPU tests and unsupported CUDA dtypes."""
        plan = selection.plan

        primary_keys = self._gather_selected_tokens(
            key_cache,
            block_table,
            plan.primary_exact_token_indices,
            plan.primary_exact_token_mask,
        )
        primary_values = self._gather_selected_tokens(
            value_cache,
            block_table,
            plan.primary_exact_token_indices,
            plan.primary_exact_token_mask,
        )

        batch_size, num_kv_heads = plan.primary_exact_token_indices.shape[:2]
        head_size = key_cache.shape[3]

        if selection.exact_page_ids.numel() == 0:
            clustered_keys = torch.empty(
                batch_size,
                num_kv_heads,
                0,
                head_size,
                dtype=key_cache.dtype,
                device=key_cache.device,
            )
            clustered_values = torch.empty_like(clustered_keys)
            clustered_mask = torch.empty(
                batch_size,
                num_kv_heads,
                0,
                dtype=torch.bool,
                device=key_cache.device,
            )
        else:
            clustered_keys, clustered_values, clustered_mask = (
                self.cluster_store.gather_pages(
                    plan.layer_name,
                    selection.exact_page_ids,
                    selection.exact_page_token_counts,
                )
            )

            if clustered_keys.device != key_cache.device:
                clustered_keys = clustered_keys.to(
                    device=key_cache.device,
                    non_blocking=False,
                )
                clustered_values = clustered_values.to(
                    device=value_cache.device,
                    non_blocking=False,
                )
                clustered_mask = clustered_mask.to(
                    device=key_cache.device,
                    non_blocking=False,
                )

        exact_keys = torch.cat(
            (primary_keys, clustered_keys),
            dim=2,
        ).contiguous()
        exact_values = torch.cat(
            (primary_values, clustered_values),
            dim=2,
        ).contiguous()
        exact_token_mask = torch.cat(
            (
                plan.primary_exact_token_mask,
                clustered_mask,
            ),
            dim=2,
        ).contiguous()

        return exact_keys, exact_values, exact_token_mask

    @staticmethod
    def _build_full_verification_pages(
        view: RetroSpecResidentBatchView,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = view.request_slot_ids.shape[0]
        device = view.request_slot_ids.device
        shape = (batch_size, num_kv_heads, 1, view.max_num_pages)
        page_ids = torch.full(shape, -1, dtype=torch.int64, device=device)
        page_token_counts = torch.zeros_like(page_ids, dtype=torch.int32)
        if view.arena is None or view.max_num_pages == 0:
            return page_ids, page_token_counts

        arena = view.arena
        slots = view.request_slot_ids.clamp_min(0)
        head_indices = torch.arange(num_kv_heads, dtype=torch.int64, device=device)[
            None, :, None
        ]
        local_page_indices = torch.arange(
            view.max_num_pages, dtype=torch.int64, device=device
        )[None, None, :]
        request_page_offsets = arena.page_offsets.index_select(0, slots)
        request_page_counts = arena.num_pages.index_select(0, slots)
        absolute_page_indices = request_page_offsets[:, None, None] + local_page_indices
        absolute_page_indices.clamp_(max=arena.page_ids.shape[1] - 1)
        valid = (view.request_slot_ids[:, None, None] >= 0) & (
            local_page_indices < request_page_counts.unsqueeze(-1)
        )
        page_ids[:, :, 0] = arena.page_ids[
            head_indices, absolute_page_indices
        ].masked_fill(~valid, -1)
        page_token_counts[:, :, 0] = arena.page_token_counts[
            head_indices, absolute_page_indices
        ].masked_fill(~valid, 0)
        return page_ids, page_token_counts

    def _pack_full_verification_page_ids_cpu(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        output_shape: torch.Size,
    ) -> torch.Tensor:
        page_ids_cpu = torch.full(
            output_shape,
            -1,
            dtype=torch.int64,
            device="cpu",
        )
        layer_indices = self._indices.get(layer_name, {})

        for row, request_id in enumerate(request_ids):
            record = layer_indices.get(request_id)
            if record is None or record.full_verification_page_ids_cpu is None:
                continue

            descriptor = record.full_verification_page_ids_cpu
            if descriptor.shape[0] != output_shape[1]:
                raise RuntimeError("Full-verification descriptor changed KV-head count")
            if descriptor.shape[1] > output_shape[3]:
                raise RuntimeError(
                    "Full-verification descriptor exceeds resident page layout"
                )
            page_ids_cpu[row, :, 0, : descriptor.shape[1]].copy_(descriptor)

        return page_ids_cpu

    def _prefetch_full_verification_layer(
        self,
        layer_index: int,
    ) -> _PrefetchedFullVerificationLayer:
        if self._full_verification_device is None:
            raise RuntimeError("Full-verification pipeline has no CUDA device")

        layer_name, num_kv_heads = self._full_verification_layers[layer_index]
        request_ids = self._full_verification_request_ids
        self._validate_resident_index(layer_name, request_ids)
        view = self._gpu_index_residency.get_active_view(
            layer_name,
            request_ids,
            self._full_verification_device,
        )
        page_ids, page_token_counts = self._build_full_verification_pages(
            view,
            num_kv_heads,
        )
        page_ids_cpu = self._pack_full_verification_page_ids_cpu(
            layer_name,
            request_ids,
            page_ids.shape,
        )
        layout = _FullVerificationPageLayout(
            page_ids=page_ids,
            page_ids_cpu=page_ids_cpu,
            page_token_counts=page_token_counts,
        )
        resolved_pages = None
        if view.max_num_pages > 0:
            resolved_pages = self.cluster_store.resolve_full_verification_blocks(
                layer_name=layer_name,
                logical_page_ids=page_ids,
                logical_page_ids_cpu=page_ids_cpu,
            )
        return _PrefetchedFullVerificationLayer(
            layer_name=layer_name,
            layout=layout,
            resolved_pages=resolved_pages,
        )

    def begin_full_verification_pipeline(
        self,
        request_ids: Sequence[str],
        layer_num_kv_heads: Mapping[str, int],
        device: torch.device,
    ) -> None:
        if self._full_verification_pipeline_active:
            raise RuntimeError("Full-verification pipeline is already active")
        if not layer_num_kv_heads:
            raise ValueError("Full-verification pipeline requires model layers")
        if device.type != "cuda":
            raise ValueError("Full-verification pipeline requires a CUDA device")

        layers = tuple(
            (layer_name, int(num_kv_heads))
            for layer_name, num_kv_heads in layer_num_kv_heads.items()
        )
        if any(num_kv_heads <= 0 for _, num_kv_heads in layers):
            raise ValueError("Full-verification KV-head counts must be positive")

        self._full_verification_pipeline_active = True
        self._full_verification_request_ids = tuple(request_ids)
        self._full_verification_layers = layers
        self._full_verification_layer_cursor = 0
        self._full_verification_device = device
        try:
            self._full_verification_prefetched = self._prefetch_full_verification_layer(
                0
            )
        except BaseException:
            self.end_full_verification_pipeline()
            raise

    def consume_full_verification_layer(
        self,
        layer_name: str,
    ) -> tuple[_FullVerificationPageLayout, RetroSpecResolvedClusterPages | None]:
        if not self._full_verification_pipeline_active:
            raise RuntimeError("Full-verification pipeline is not active")
        prefetched = self._full_verification_prefetched
        if prefetched is None:
            raise RuntimeError("Full-verification pipeline has no remaining layer")
        if prefetched.layer_name != layer_name:
            raise RuntimeError(
                "Full-verification layer order differs from the installed model"
            )

        next_cursor = self._full_verification_layer_cursor + 1
        next_prefetched = None
        if next_cursor < len(self._full_verification_layers):
            next_prefetched = self._prefetch_full_verification_layer(next_cursor)

        self._full_verification_layer_cursor = next_cursor
        self._full_verification_prefetched = next_prefetched
        return prefetched.layout, prefetched.resolved_pages

    def end_full_verification_pipeline(self) -> None:
        self._full_verification_pipeline_active = False
        self._full_verification_request_ids = ()
        self._full_verification_layers = ()
        self._full_verification_layer_cursor = 0
        self._full_verification_device = None
        self._full_verification_prefetched = None

    def build_full_verification_plan(
        self,
        request_ids: Sequence[str],
        layer_name: str,
        seq_lens: Sequence[int],
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecFullVerificationPlan:
        """Build an exact full-verification view over existing KV storage.

        Complete indexed segments are represented by every cluster page owned
        by the requests. Tokens outside those segments remain primary logical
        token references into the active vLLM KV cache.
        """
        request_ids = tuple(request_ids)
        seq_lens = tuple(int(seq_len) for seq_len in seq_lens)

        if any(staged.layer_name == layer_name for staged in self._staged_segments):
            raise RuntimeError(
                "Cannot build full verification because index updates are staged "
                "for this layer"
            )
        if key_cache.ndim != 4:
            raise ValueError(
                "KV cache must have shape [num_blocks, block_size, kv_heads, head_size]"
            )
        if key_cache.shape[1] != self.block_size:
            raise ValueError("KV cache block size does not match the index")
        if block_table.ndim != 2:
            raise ValueError("block_table must be two-dimensional")
        if block_table.shape[0] != len(request_ids):
            raise ValueError("block_table batch size does not match request_ids")
        if len(seq_lens) != len(request_ids):
            raise ValueError("request_ids and seq_lens must have equal length")
        if block_table.device != key_cache.device:
            raise ValueError("block_table and KV cache must use one device")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("block_table entries must be integral")

        max_num_tokens = block_table.shape[1] * self.block_size
        if any(seq_len < 0 for seq_len in seq_lens):
            raise ValueError("Full-verification context lengths must be non-negative")
        if any(seq_len > max_num_tokens for seq_len in seq_lens):
            raise ValueError(
                "Full-verification sequence length exceeds the block table"
            )

        layer_indices = self._indices.get(layer_name, {})
        primary_token_counts: list[int] = []

        for request_id, seq_len in zip(request_ids, seq_lens):
            record = layer_indices.get(request_id)

            if record is None or not record.segments:
                indexed_token_count = 0
            else:
                if record.indexed_end > seq_len:
                    raise RuntimeError(
                        "Full verification requires rolled-back cluster state "
                        "to be rebuilt first"
                    )
                indexed_token_count = (
                    record.indexed_end - record.segments[0].indexed_start
                )

            primary_token_counts.append(seq_len - indexed_token_count)

        view = self._get_resident_view(layer_name, request_ids, key_cache)

        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int64,
            device=block_table.device,
        )
        logical_token_ids, valid_token_mask, _ = self._build_token_layout(
            block_table,
            seq_lens_tensor,
        )

        # Every committed token not owned by a complete clustered segment is
        # part of the exact primary/steady zone.
        indexed_starts, indexed_ends, indexed_requests = (
            self._get_resident_indexed_bounds(view, block_table.device)
        )
        primary_token_mask = valid_token_mask & (
            ~indexed_requests.unsqueeze(1)
            | (logical_token_ids.unsqueeze(0) < indexed_starts.unsqueeze(1))
            | (logical_token_ids.unsqueeze(0) >= indexed_ends.unsqueeze(1))
        )
        num_kv_heads = key_cache.shape[2]
        per_head_primary_mask = primary_token_mask.unsqueeze(1).expand(
            -1,
            num_kv_heads,
            -1,
        )

        (
            primary_exact_token_indices,
            primary_exact_token_mask,
        ) = self._pack_bounded_mask_indices(
            per_head_primary_mask,
            max(primary_token_counts, default=0),
        )

        primary_exact_token_counts = primary_exact_token_mask.sum(
            dim=2,
            dtype=torch.int32,
        )
        resolved_pages = None
        if self._full_verification_pipeline_active:
            layout, resolved_pages = self.consume_full_verification_layer(layer_name)
            exact_page_ids = layout.page_ids
            exact_page_ids_cpu = layout.page_ids_cpu
            exact_page_token_counts = layout.page_token_counts
            if exact_page_ids.shape[:2] != (len(request_ids), num_kv_heads):
                raise RuntimeError(
                    "Prefetched full-verification layout changed batch shape"
                )
        else:
            exact_page_ids, exact_page_token_counts = (
                self._build_full_verification_pages(view, num_kv_heads)
            )
            exact_page_ids_cpu = self._pack_full_verification_page_ids_cpu(
                layer_name,
                request_ids,
                exact_page_ids.shape,
            )
        clustered_exact_token_counts = exact_page_token_counts.sum(
            dim=(2, 3),
            dtype=torch.int32,
        )
        exact_token_counts = (
            primary_exact_token_counts + clustered_exact_token_counts
        ).contiguous()
        return RetroSpecFullVerificationPlan(
            layer_name=layer_name,
            primary_exact_token_indices=primary_exact_token_indices,
            primary_exact_token_mask=primary_exact_token_mask,
            exact_page_ids=exact_page_ids,
            exact_page_ids_cpu=exact_page_ids_cpu,
            exact_page_token_counts=exact_page_token_counts,
            exact_token_counts=exact_token_counts,
            resolved_pages=resolved_pages,
        )

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
        warm_first_draft: bool = False,
    ) -> RetroSpecTokenAttentionSelection:
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

        view = self._get_resident_view(layer_name, request_ids, key_cache)

        logical_token_ids, valid_token_mask, forced_exact_mask = (
            self._build_token_layout(
                block_table,
                seq_lens,
            )
        )

        # All valid tokens not covered by a complete clustered segment remain
        # in the exact steady zone.
        indexed_starts, indexed_ends, indexed_requests = (
            self._get_resident_indexed_bounds(view, block_table.device)
        )
        forced_exact_mask |= valid_token_mask & (
            ~indexed_requests.unsqueeze(1)
            | (logical_token_ids.unsqueeze(0) < indexed_starts.unsqueeze(1))
            | (logical_token_ids.unsqueeze(0) >= indexed_ends.unsqueeze(1))
        )

        num_kv_heads = key_cache.shape[2]
        first_draft_warmup_mask = self._get_first_draft_warmup_mask(
            request_ids,
            layer_name,
            active_mask,
            warm_first_draft,
        )
        warmup_page_budgets = None
        if first_draft_warmup_mask is not None:
            group_targets = self.cluster_store.resident_group_target_pages(
                layer_name,
                request_ids,
                num_kv_heads,
            )
            warmup_page_budgets = torch.tensor(
                group_targets,
                dtype=torch.int64,
                device=query.device,
            )
            warmup_page_budgets = torch.div(
                warmup_page_budgets + 1,
                2,
                rounding_mode="floor",
            )

        (
            cluster_scores,
            ranking_scores,
            candidate_counts,
            workspace,
        ) = self._score_resident_view(
            query,
            view,
            scale,
            num_kv_heads,
        )

        cluster_zones = self._select_cluster_zones(
            cluster_scores,
            ranking_scores,
            candidate_counts,
            view=view,
            first_draft_warmup_mask=first_draft_warmup_mask,
            warmup_page_budgets=warmup_page_budgets,
            workspace=workspace,
        )

        sparse_attn_by_head = self._sum_selected_scores(
            cluster_scores,
            cluster_zones.sparse_retrieval_indices,
            cluster_zones.sparse_retrieval_mask,
        )
        expanded_attn_by_head = self._sum_selected_scores(
            cluster_scores,
            cluster_zones.expanded_retrieval_indices,
            cluster_zones.expanded_retrieval_mask,
        )

        has_clusters = candidate_counts > 0
        sparse_attn_by_head = torch.where(
            has_clusters,
            sparse_attn_by_head,
            torch.ones_like(sparse_attn_by_head),
        )
        expanded_attn_by_head = torch.where(
            has_clusters,
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

        plan, output_workspace = self._make_plan(
            layer_name=layer_name,
            forced_exact_mask=forced_exact_mask,
            cluster_zones=cluster_zones,
            sparse_attn=sparse_attn,
            expanded_attn=expanded_attn,
            view=view,
            num_kv_heads=num_kv_heads,
            head_size=key_cache.shape[3],
            dtype=key_cache.dtype,
        )

        include_pending_resident = False
        if first_draft_warmup_mask is not None:
            warmup_cluster_ids, warmup_page_ids, _ = (
                self._build_resident_exact_cluster_selection(
                    view,
                    cluster_zones.first_draft_warmup_indices,
                    cluster_zones.first_draft_warmup_mask,
                )
            )
            self.cluster_store.admit_resident_clusters(
                layer_name=layer_name,
                cluster_ids=warmup_cluster_ids,
                page_ids=warmup_page_ids,
            )
            include_pending_resident = True
        return self._materialize_draft_selection(
            plan=plan,
            output_workspace=output_workspace,
            view=view,
            cluster_zones=cluster_zones,
            cluster_scores=cluster_scores,
            has_clusters=has_clusters,
            head_size=key_cache.shape[3],
            dtype=key_cache.dtype,
            active_mask=active_mask,
            include_pending_resident=include_pending_resident,
        )

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
