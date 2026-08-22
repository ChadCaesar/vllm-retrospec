# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil

import torch

from .cluster_scoring import reduce_grouped_cluster_scores
from .cluster_store import (
    RetroSpecClusterBlockTable,
    RetroSpecClusterPageStore,
    RetroSpecClusterStorageMode,
    RetroSpecResolvedClusterPages,
    RetroSpecStagedClusterInput,
)
from .clustering import segmented_kmeans_assignments
from .index import (
    RetroSpecAttentionLevel,
    RetroSpecBlockIndex,
    RetroSpecSelectionPlan,
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

    exact_cluster_ids: torch.Tensor
    exact_page_ids: torch.Tensor
    exact_page_token_counts: torch.Tensor
    exact_token_counts: torch.Tensor


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

    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    build_future: Future[RetroSpecClusterBlockTable]


@dataclass
class _RequestLayerIndex:
    segments: list[_RequestLayerSegment]
    num_clusters: int
    indexed_end: int


@dataclass(frozen=True)
class _PackedSegmentedIndex:
    indexed_token_mask: torch.Tensor

    cluster_ids: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_mask: torch.Tensor

    cluster_page_ids: torch.Tensor
    cluster_page_token_counts: torch.Tensor


@dataclass(frozen=True)
class _PackedSegmentedIndexCacheEntry:
    request_ids: tuple[str, ...]
    max_num_tokens: int
    packed: _PackedSegmentedIndex


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


@dataclass(frozen=True)
class _ClusterSelectionWorkspace:
    logits: torch.Tensor
    scores: torch.Tensor
    softmax_lse: torch.Tensor
    ranking_scores: torch.Tensor
    topk_values: torch.Tensor
    topk_indices: torch.Tensor


class RetroSpecSegmentedTokenIndex(RetroSpecBlockIndex):
    """Token-level segmented index backed by private cluster KV pages."""

    def __init__(
        self,
        block_size: int,
        num_speculative_tokens: int,
        retrieval_ratio: float,
        estimation_ratio: float,
        segment_size_tokens: int,
        blocks_per_cluster: int,
        num_kmeans_iterations: int,
        max_pending_cluster_builds: int = 2,
        cache_mode: RetroSpecClusterStorageMode = "gpu_reference",
        cache_ratio: float = 0.0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__(
            block_size=block_size,
            num_speculative_tokens=num_speculative_tokens,
            retrieval_ratio=retrieval_ratio,
            estimation_ratio=estimation_ratio,
        )

        if segment_size_tokens % block_size != 0:
            raise ValueError("segment_size_tokens must be divisible by block_size")
        if blocks_per_cluster <= 0:
            raise ValueError("blocks_per_cluster must be positive")
        if num_kmeans_iterations <= 0:
            raise ValueError("num_kmeans_iterations must be positive")
        if max_pending_cluster_builds <= 0:
            raise ValueError("max_pending_cluster_builds must be positive")

        tokens_per_cluster = blocks_per_cluster * block_size
        if segment_size_tokens % tokens_per_cluster != 0:
            raise ValueError(
                "segment_size_tokens must be divisible by "
                "blocks_per_cluster * block_size"
            )

        self.segment_size_tokens = segment_size_tokens
        self.tokens_per_cluster = tokens_per_cluster
        self.num_kmeans_iterations = num_kmeans_iterations
        self.max_pending_cluster_builds = max_pending_cluster_builds
        effective_cache_ratio = cache_ratio
        if cache_mode == "cpu_offload" and cache_ratio == 0.0:
            # RetroInfer uses three sparse retrieval zones when an explicit
            # cache ratio is not supplied.
            effective_cache_ratio = min(
                retrieval_ratio * 3.0,
                1.0,
            )

        self.cluster_store = RetroSpecClusterPageStore(
            page_size=block_size,
            storage_mode=cache_mode,
            pin_memory=pin_memory,
            cache_ratio=effective_cache_ratio,
        )

        # layer_name -> request_id -> token-level index
        self._indices: dict[str, dict[str, _RequestLayerIndex]] = {}

        self._proposal_active = False
        self._proposal_request_ids: tuple[str, ...] = ()

        # Each layer retains only the most recently packed request batch.
        self._packed_index_cache: dict[str, _PackedSegmentedIndexCacheEntry] = {}

        # Shared across model layers. Selection results are copied into each
        # plan before the workspace is reused.
        self._cluster_selection_workspace: _ClusterSelectionWorkspace | None = None

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

    def _desired_indexed_end(self, seq_len: int) -> int:
        """Return the exclusive logical-token boundary covered by clustering."""
        full_block_count = seq_len // self.block_size

        # The first block remains an exact attention sink. Recent complete
        # blocks and the current partial block remain in the exact steady zone.
        stable_end_block = max(
            full_block_count - self.num_recent_blocks,
            1,
        )
        indexable_tokens = (stable_end_block - 1) * self.block_size

        complete_segments = indexable_tokens // self.segment_size_tokens
        return self.block_size + complete_segments * self.segment_size_tokens

    def needs_update(
        self,
        request_id: str,
        seq_len: int,
        layer_names: Sequence[str],
    ) -> bool:
        desired_end = self._desired_indexed_end(seq_len)

        for layer_name in layer_names:
            layer_indices = self._indices.get(layer_name)
            if layer_indices is None or request_id not in layer_indices:
                return True
            if layer_indices[request_id].indexed_end != desired_end:
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
        request_ids = tuple(request_ids)

        for layer_name, layer_indices in self._indices.items():
            layer_changed = False

            for request_id in request_ids:
                record = layer_indices.pop(request_id, None)
                if record is None:
                    continue

                self._free_record(layer_name, record)
                layer_changed = True

            if layer_changed:
                self._packed_index_cache.pop(layer_name, None)

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
                layer_changed = True

            if layer_changed:
                self._packed_index_cache.pop(layer_name, None)

    def begin_proposal(self, request_ids: Sequence[str]) -> None:
        if self._proposal_active:
            raise RuntimeError("Segmented token index proposal is already active")
        if self._staged_segments:
            raise RuntimeError(
                "Cannot begin a proposal before staged index updates are flushed"
            )

        self._proposal_active = True
        self._proposal_request_ids = tuple(request_ids)

    def end_proposal(self) -> None:
        if not self._proposal_active:
            raise RuntimeError("Segmented token index proposal is not active")

        self._proposal_active = False
        self._proposal_request_ids = ()

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
        if not self.cluster_store.is_cpu_backed:
            raise RuntimeError(
                "Background cluster construction requires CPU-backed storage"
            )

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
        return build_future

    def _wait_for_cluster_build_slot(self) -> None:
        """Bound queued builds before allocating another pinned staging input."""
        while self._pending_cluster_builds and self._pending_cluster_builds[0].done():
            self._pending_cluster_builds.popleft().result()

        if len(self._pending_cluster_builds) < self.max_pending_cluster_builds:
            return

        self._pending_cluster_builds.popleft().result()

    def _release_built_segments(
        self,
        built_segments: Sequence[
            tuple[_StagedRequestLayerSegment, RetroSpecClusterBlockTable]
        ],
    ) -> None:
        for staged_segment, cluster_blocks in built_segments:
            self.cluster_store.free(staged_segment.layer_name, cluster_blocks)

    def _publish_built_segments(
        self,
        built_segments: Sequence[
            tuple[_StagedRequestLayerSegment, RetroSpecClusterBlockTable]
        ],
    ) -> None:
        """Atomically publish one complete staged index transaction."""
        pending_records: dict[tuple[str, str], _RequestLayerIndex] = {}

        for staged_segment, cluster_blocks in built_segments:
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

            record.segments.append(
                _RequestLayerSegment(
                    indexed_start=staged_segment.indexed_start,
                    indexed_end=staged_segment.indexed_end,
                    cluster_start=staged_segment.cluster_start,
                    cluster_keys=staged_segment.cluster_keys,
                    cluster_values=staged_segment.cluster_values,
                    cluster_token_counts=staged_segment.cluster_token_counts,
                    cluster_blocks=cluster_blocks,
                )
            )
            record.num_clusters += staged_segment.cluster_token_counts.shape[1]
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

        self._indices = new_indices

        for layer_name in changed_layers:
            self._packed_index_cache.pop(layer_name, None)

    def _append_segment(
        self,
        layer_name: str,
        request_id: str,
        indexed_start: int,
        indexed_end: int,
        cluster_start: int,
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        cluster_blocks: RetroSpecClusterBlockTable,
    ) -> None:
        layer_indices = self._indices.setdefault(layer_name, {})
        record = layer_indices.get(request_id)

        if record is None:
            record = self._empty_index()
            layer_indices[request_id] = record

        if record.indexed_end != indexed_start:
            raise RuntimeError(
                "Staged RetroSpec segment no longer follows the indexed prefix"
            )
        if record.num_clusters != cluster_start:
            raise RuntimeError(
                "Staged RetroSpec segment cluster offset is no longer current"
            )

        record.segments.append(
            _RequestLayerSegment(
                indexed_start=indexed_start,
                indexed_end=indexed_end,
                cluster_start=cluster_start,
                cluster_keys=cluster_keys,
                cluster_values=cluster_values,
                cluster_token_counts=cluster_token_counts,
                cluster_blocks=cluster_blocks,
            )
        )
        record.num_clusters += cluster_token_counts.shape[1]
        record.indexed_end = indexed_end
        self._packed_index_cache.pop(layer_name, None)

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
            tuple[_StagedRequestLayerSegment, RetroSpecClusterBlockTable]
        ] = []
        first_error: BaseException | None = None

        try:
            # Resolve every future even after one fails. A later task may have
            # completed successfully and own unpublished CPU pages.
            for staged_segment in staged_segments:
                try:
                    cluster_blocks = staged_segment.build_future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    continue

                built_segments.append((staged_segment, cluster_blocks))

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

    @staticmethod
    def _cluster_means(
        vectors: torch.Tensor,
        assignments: torch.Tensor,
        cluster_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce [heads, tokens, dim] vectors into per-head cluster means."""
        num_kv_heads, _, head_size = vectors.shape
        num_clusters = cluster_counts.shape[1]

        expanded_assignments = assignments.unsqueeze(-1).expand(
            -1,
            -1,
            head_size,
        )

        cluster_sums = torch.zeros(
            num_kv_heads,
            num_clusters,
            head_size,
            dtype=torch.float32,
            device=vectors.device,
        )
        cluster_sums.scatter_add_(
            dim=1,
            index=expanded_assignments,
            src=vectors.float(),
        )

        return (
            cluster_sums / cluster_counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
        ).to(vectors.dtype)

    def build_or_update(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        rows: Sequence[int],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        defer_cpu_store: bool = False,
    ) -> None:
        """Cluster stable tokens and stage or store private cluster pages."""
        if len(request_ids) != len(seq_lens):
            raise ValueError("request_ids and seq_lens must have equal length")
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
            desired_end = self._desired_indexed_end(seq_len)

            staged_key = (layer_name, request_id)
            if staged_key in self._staged_segment_keys:
                raise RuntimeError(
                    "A RetroSpec request/layer segment is already staged"
                )

            record = layer_indices.get(request_id)
            if record is not None and desired_end < record.indexed_end:
                self._packed_index_cache.pop(layer_name, None)
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
            if num_new_tokens % self.segment_size_tokens != 0:
                raise RuntimeError(
                    "New indexed region must contain complete token segments"
                )

            first_logical_block = indexed_start // self.block_size
            logical_block_end = desired_end // self.block_size

            logical_block_ids = torch.arange(
                first_logical_block,
                logical_block_end,
                dtype=torch.int64,
                device=block_table.device,
            )
            physical_block_ids = (
                block_table[row]
                .index_select(
                    0,
                    logical_block_ids,
                )
                .to(torch.int64)
            )

            key_blocks = key_cache.index_select(
                0,
                physical_block_ids,
            )
            value_blocks = value_cache.index_select(
                0,
                physical_block_ids,
            )

            num_kv_heads = key_cache.shape[2]
            head_size = key_cache.shape[3]

            token_keys = (
                key_blocks.reshape(
                    num_new_tokens,
                    num_kv_heads,
                    head_size,
                )
                .transpose(0, 1)
                .contiguous()
            )
            token_values = (
                value_blocks.reshape(
                    num_new_tokens,
                    num_kv_heads,
                    head_size,
                )
                .transpose(0, 1)
                .contiguous()
            )

            staged_token_kv = None
            if self.cluster_store.is_cpu_backed:
                self._wait_for_cluster_build_slot()
                # Start token-KV D2H before clustering. Both streams only read
                # the token tensors, so transfer and k-means can safely overlap.
                staged_token_kv = self.cluster_store.stage_token_kv(
                    token_keys,
                    token_values,
                )

            try:
                local_assignments, cluster_token_counts = segmented_kmeans_assignments(
                    features=token_keys,
                    segment_size=self.segment_size_tokens,
                    items_per_cluster=self.tokens_per_cluster,
                    num_iterations=self.num_kmeans_iterations,
                )

                cluster_keys = self._cluster_means(
                    token_keys,
                    local_assignments,
                    cluster_token_counts,
                )
                cluster_values = self._cluster_means(
                    token_values,
                    local_assignments,
                    cluster_token_counts,
                )
            except BaseException:
                if staged_token_kv is not None:
                    self.cluster_store.discard_staged_token_kv(staged_token_kv)
                raise

            cluster_start = 0 if record is None else record.num_clusters

            if self.cluster_store.is_cpu_backed:
                assert staged_token_kv is not None

                try:
                    staged_clusters = self.cluster_store.finish_stage_clusters(
                        staged_token_kv,
                        local_assignments,
                        cluster_token_counts,
                    )
                except BaseException:
                    self.cluster_store.discard_staged_token_kv(staged_token_kv)
                    raise

                try:
                    build_future = self._submit_cluster_build(
                        layer_name=layer_name,
                        request_id=request_id,
                        cluster_start=cluster_start,
                        staged_clusters=staged_clusters,
                    )
                except BaseException:
                    # Metadata D2H may already be queued. Wait before returning
                    # its pinned staging slot to the shared pool.
                    self.cluster_store.discard_staged_clusters(staged_clusters)
                    raise

                self._staged_segments.append(
                    _StagedRequestLayerSegment(
                        layer_name=layer_name,
                        request_id=request_id,
                        indexed_start=indexed_start,
                        indexed_end=desired_end,
                        cluster_start=cluster_start,
                        cluster_keys=cluster_keys,
                        cluster_values=cluster_values,
                        cluster_token_counts=cluster_token_counts,
                        build_future=build_future,
                    )
                )
                self._staged_segment_keys.add(staged_key)
                if not defer_cpu_store:
                    self.flush_staged_updates()
                continue

            cluster_blocks = self.cluster_store.store_clusters(
                layer_name=layer_name,
                request_id=request_id,
                cluster_start=cluster_start,
                token_keys=token_keys,
                token_values=token_values,
                assignments=local_assignments,
                cluster_token_counts=cluster_token_counts,
            )
            self._append_segment(
                layer_name=layer_name,
                request_id=request_id,
                indexed_start=indexed_start,
                indexed_end=desired_end,
                cluster_start=cluster_start,
                cluster_keys=cluster_keys,
                cluster_values=cluster_values,
                cluster_token_counts=cluster_token_counts,
                cluster_blocks=cluster_blocks,
            )

        if layer_changed:
            self._packed_index_cache.pop(layer_name, None)

    def _pack_indices(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> _PackedSegmentedIndex:
        batch_size, max_num_blocks = block_table.shape
        max_num_tokens = max_num_blocks * self.block_size
        request_ids = tuple(request_ids)

        if len(request_ids) != batch_size:
            raise ValueError("request_ids batch size does not match block_table")

        cached = self._packed_index_cache.get(layer_name)
        if (
            cached is not None
            and cached.request_ids == request_ids
            and cached.max_num_tokens == max_num_tokens
        ):
            return cached.packed

        layer_indices = self._indices.get(layer_name, {})
        records = [layer_indices.get(request_id) for request_id in request_ids]

        max_num_clusters = max(
            1,
            max(
                (
                    record.num_clusters if record is not None else 0
                    for record in records
                ),
                default=0,
            ),
        )
        max_pages_per_cluster = max(
            (
                self.cluster_store.max_pages_per_cluster(
                    layer_name, segment.cluster_blocks.cluster_ids
                )
                for record in records
                if record is not None
                for segment in record.segments
            ),
            default=0,
        )

        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        indexed_token_mask = torch.zeros(
            batch_size,
            max_num_tokens,
            dtype=torch.bool,
            device=key_cache.device,
        )

        cluster_ids = torch.full(
            (
                batch_size,
                num_kv_heads,
                max_num_clusters,
            ),
            -1,
            dtype=torch.int64,
            device=key_cache.device,
        )

        cluster_keys = torch.zeros(
            batch_size,
            num_kv_heads,
            max_num_clusters,
            head_size,
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
        cluster_values = torch.zeros_like(cluster_keys)
        cluster_token_counts = torch.zeros(
            batch_size,
            num_kv_heads,
            max_num_clusters,
            dtype=torch.int32,
            device=key_cache.device,
        )
        cluster_mask = torch.zeros(
            batch_size,
            num_kv_heads,
            max_num_clusters,
            dtype=torch.bool,
            device=key_cache.device,
        )

        cluster_page_ids = torch.full(
            (
                batch_size,
                num_kv_heads,
                max_num_clusters,
                max_pages_per_cluster,
            ),
            -1,
            dtype=torch.int64,
            device=key_cache.device,
        )
        cluster_page_token_counts = torch.zeros(
            cluster_page_ids.shape,
            dtype=torch.int32,
            device=key_cache.device,
        )

        for row, record in enumerate(records):
            if record is None:
                continue

            for segment in record.segments:
                indexed_end = min(
                    segment.indexed_end,
                    max_num_tokens,
                )
                if indexed_end > segment.indexed_start:
                    indexed_token_mask[
                        row,
                        segment.indexed_start : indexed_end,
                    ] = True

                cluster_start = segment.cluster_start
                cluster_end = cluster_start + segment.cluster_keys.shape[1]

                packed_cluster_ids = cluster_ids[row, :, cluster_start:cluster_end]
                packed_cluster_token_counts = cluster_token_counts[
                    row, :, cluster_start:cluster_end
                ]

                packed_cluster_ids.copy_(
                    segment.cluster_blocks.cluster_ids, non_blocking=False
                )
                cluster_keys[row, :, cluster_start:cluster_end].copy_(
                    segment.cluster_keys
                )
                cluster_values[row, :, cluster_start:cluster_end].copy_(
                    segment.cluster_values
                )
                packed_cluster_token_counts.copy_(segment.cluster_token_counts)
                cluster_mask[row, :, cluster_start:cluster_end].copy_(
                    (packed_cluster_ids >= 0) & (packed_cluster_token_counts > 0)
                )

                block_metadata = self.cluster_store.get_cluster_block_metadata(
                    layer_name=layer_name,
                    cluster_ids=segment.cluster_blocks.cluster_ids,
                    device=torch.device("cpu"),
                )
                segment_page_width = block_metadata.page_ids.shape[2]

                cluster_page_ids[
                    row, :, cluster_start:cluster_end, :segment_page_width
                ].copy_(block_metadata.page_ids, non_blocking=False)
                cluster_page_token_counts[
                    row, :, cluster_start:cluster_end, :segment_page_width
                ].copy_(block_metadata.page_token_counts, non_blocking=False)

        packed = _PackedSegmentedIndex(
            indexed_token_mask=indexed_token_mask,
            cluster_ids=cluster_ids,
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            cluster_token_counts=cluster_token_counts,
            cluster_mask=cluster_mask,
            cluster_page_ids=cluster_page_ids,
            cluster_page_token_counts=(cluster_page_token_counts),
        )
        self._packed_index_cache[layer_name] = _PackedSegmentedIndexCacheEntry(
            request_ids=request_ids,
            max_num_tokens=max_num_tokens,
            packed=packed,
        )
        return packed

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

    def _get_cluster_selection_workspace(
        self,
        query: torch.Tensor,
        cluster_keys: torch.Tensor,
    ) -> _ClusterSelectionWorkspace:
        """Return a reusable CUDA workspace for cluster selection."""
        if query.device.type != "cuda":
            raise ValueError("Cluster selection workspace requires CUDA")
        if cluster_keys.device != query.device:
            raise ValueError("Query and cluster keys must be on one CUDA device")

        batch_size, num_query_heads, _ = query.shape
        num_kv_heads = cluster_keys.shape[1]
        num_clusters = cluster_keys.shape[2]

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
            max_total_compute,
        )

        workspace = self._cluster_selection_workspace
        if (
            workspace is not None
            and workspace.logits.device == query.device
            and workspace.logits.shape == logits_shape
            and workspace.scores.shape == scores_shape
            and workspace.softmax_lse.shape == lse_shape
            and workspace.ranking_scores.shape == scores_shape
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
        cluster_mask: torch.Tensor,
        workspace: _ClusterSelectionWorkspace | None = None,
    ) -> _PackedClusterZones:
        """Rank relevant clusters once and return compact zone indices."""
        if cluster_scores.shape != cluster_mask.shape:
            raise ValueError("Cluster scores and mask must have equal shapes")
        if cluster_scores.ndim != 3:
            raise ValueError(
                "Cluster scores must have shape [batch, num_kv_heads, num_clusters]"
            )

        num_clusters = cluster_scores.shape[2]
        (
            max_retrieval,
            max_estimation,
            max_expanded_retrieval,
        ) = self._maximum_zone_widths(num_clusters)

        max_total_compute = max_retrieval + max_estimation

        candidate_counts = cluster_mask.sum(
            dim=2,
            dtype=torch.int64,
        )

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
            ranking_scores = cluster_scores.masked_fill(
                ~cluster_mask,
                float("-inf"),
            )
            ranked_indices = torch.topk(
                ranking_scores,
                k=max_total_compute,
                dim=2,
                largest=True,
                sorted=True,
            ).indices
        else:
            if cluster_scores is not workspace.scores:
                raise ValueError(
                    "Workspace ranking scores do not belong to cluster scores"
                )
            expected_topk_shape = (
                cluster_scores.shape[0],
                cluster_scores.shape[1],
                max_total_compute,
            )
            if workspace.ranking_scores.shape != cluster_scores.shape:
                raise ValueError(
                    "Workspace ranking-score shape does not match cluster scores"
                )
            if workspace.topk_values.shape != expected_topk_shape:
                raise ValueError(
                    "Workspace top-k value shape does not match selection width"
                )
            if workspace.topk_indices.shape != expected_topk_shape:
                raise ValueError(
                    "Workspace top-k index shape does not match selection width"
                )

            torch.topk(
                workspace.ranking_scores,
                k=max_total_compute,
                dim=2,
                largest=True,
                sorted=True,
                out=(
                    workspace.topk_values,
                    workspace.topk_indices,
                ),
            )
            ranked_indices = workspace.topk_indices

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

        return _PackedClusterZones(
            sparse_retrieval_indices=sparse_retrieval_indices,
            sparse_retrieval_mask=sparse_retrieval_mask,
            sparse_estimation_indices=sparse_estimation_indices,
            sparse_estimation_mask=sparse_estimation_mask,
            expanded_retrieval_indices=expanded_retrieval_indices,
            expanded_retrieval_mask=expanded_retrieval_mask,
            expanded_estimation_indices=expanded_estimation_indices,
            expanded_estimation_mask=expanded_estimation_mask,
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

    @staticmethod
    def _build_exact_cluster_selection(
        cluster_ids: torch.Tensor,
        cluster_page_ids: torch.Tensor,
        cluster_page_token_counts: torch.Tensor,
        packed_cluster_indices: torch.Tensor,
        packed_cluster_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, _, max_pages = cluster_page_ids.shape
        max_selected_clusters = packed_cluster_indices.shape[2]

        selected_cluster_ids = cluster_ids.gather(
            dim=2,
            index=packed_cluster_indices,
        )

        page_indices = packed_cluster_indices.unsqueeze(-1).expand(
            batch_size,
            num_kv_heads,
            max_selected_clusters,
            max_pages,
        )

        selected_page_ids = cluster_page_ids.gather(
            dim=2,
            index=page_indices,
        )
        selected_page_token_counts = cluster_page_token_counts.gather(
            dim=2,
            index=page_indices,
        )

        valid_clusters = packed_cluster_mask & (selected_cluster_ids >= 0)
        selected_cluster_ids.masked_fill_(~valid_clusters, -1)

        valid_pages = (
            valid_clusters.unsqueeze(-1)
            & (selected_page_ids >= 0)
            & (selected_page_token_counts > 0)
        )
        selected_page_ids.masked_fill_(~valid_pages, -1)
        selected_page_token_counts.masked_fill_(~valid_pages, 0)

        return (
            selected_cluster_ids.contiguous(),
            selected_page_ids.contiguous(),
            selected_page_token_counts.contiguous(),
        )

    @staticmethod
    def _build_estimation_selection(
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        packed_indices: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, max_num_clusters, head_size = cluster_keys.shape
        del max_num_clusters

        max_selected_clusters = packed_indices.shape[2]
        vector_indices = packed_indices.unsqueeze(-1).expand(
            batch_size,
            num_kv_heads,
            max_selected_clusters,
            head_size,
        )

        estimation_keys = cluster_keys.gather(2, vector_indices)
        estimation_values = cluster_values.gather(2, vector_indices)
        estimation_token_counts = cluster_token_counts.gather(
            2,
            packed_indices,
        )

        estimation_keys.masked_fill_(
            ~packed_mask.unsqueeze(-1),
            0.0,
        )
        estimation_values.masked_fill_(
            ~packed_mask.unsqueeze(-1),
            0.0,
        )
        estimation_token_counts.masked_fill_(~packed_mask, 0)

        return (
            estimation_keys.contiguous(),
            estimation_values.contiguous(),
            estimation_token_counts.to(torch.int32).contiguous(),
        )

    def _make_plan(
        self,
        layer_name: str,
        forced_exact_mask: torch.Tensor,
        cluster_zones: _PackedClusterZones,
        sparse_attn: torch.Tensor,
        expanded_attn: torch.Tensor,
        packed: _PackedSegmentedIndex,
    ) -> RetroSpecTokenSelectionPlan:
        num_kv_heads = packed.cluster_keys.shape[1]
        per_head_forced_exact = forced_exact_mask.unsqueeze(1).expand(
            -1,
            num_kv_heads,
            -1,
        )

        # With an up-to-date segmented index, the exact primary zone consists
        # of the sink block, an incomplete segment, recent blocks and the
        # current partial block. This expression is a synchronization-free
        # upper bound for that union.
        max_primary_exact_tokens = min(
            forced_exact_mask.shape[1],
            self.segment_size_tokens + (self.num_recent_blocks + 1) * self.block_size,
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
        ) = self._build_exact_cluster_selection(
            packed.cluster_ids,
            packed.cluster_page_ids,
            packed.cluster_page_token_counts,
            cluster_zones.sparse_retrieval_indices,
            cluster_zones.sparse_retrieval_mask,
        )

        (
            expanded_exact_cluster_ids,
            expanded_exact_page_ids,
            expanded_exact_page_token_counts,
        ) = self._build_exact_cluster_selection(
            packed.cluster_ids,
            packed.cluster_page_ids,
            packed.cluster_page_token_counts,
            cluster_zones.expanded_retrieval_indices,
            cluster_zones.expanded_retrieval_mask,
        )

        (
            sparse_estimation_keys,
            sparse_estimation_values,
            sparse_estimation_token_counts,
        ) = self._build_estimation_selection(
            packed.cluster_keys,
            packed.cluster_values,
            packed.cluster_token_counts,
            cluster_zones.sparse_estimation_indices,
            cluster_zones.sparse_estimation_mask,
        )
        (
            expanded_estimation_keys,
            expanded_estimation_values,
            expanded_estimation_token_counts,
        ) = self._build_estimation_selection(
            packed.cluster_keys,
            packed.cluster_values,
            packed.cluster_token_counts,
            cluster_zones.expanded_estimation_indices,
            cluster_zones.expanded_estimation_mask,
        )

        return RetroSpecTokenSelectionPlan(
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
        packed: _PackedSegmentedIndex,
        cluster_zones: _PackedClusterZones,
        cluster_scores: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> RetroSpecTokenAttentionSelection:
        """Materialize a cache-aware sparse selection for the draft stage.

        GPU-reference storage computes every sparse retrieval cluster exactly.

        CPU-offload storage computes only resident retrieval clusters exactly.
        Selected cache misses are represented by their key/value centroids and
        merged into the normal sparse estimation zone.
        """
        if (
            not self.cluster_store.is_cpu_backed
            or plan.sparse_exact_page_ids.numel() == 0
        ):
            return self._materialize_token_selection(
                plan,
                RetroSpecAttentionLevel.SPARSE,
            )

        resolved_pages = self.cluster_store.resolve_cluster_blocks(
            layer_name=plan.layer_name,
            cluster_ids=plan.sparse_exact_cluster_ids,
            logical_page_ids=plan.sparse_exact_page_ids,
            mode="resident_only",
        )

        hit_cluster_mask = (
            cluster_zones.sparse_retrieval_mask & resolved_pages.hit_cluster_mask
        )
        miss_cluster_mask = (
            cluster_zones.sparse_retrieval_mask & resolved_pages.miss_cluster_mask
        )

        draft_exact_page_token_counts = plan.sparse_exact_page_token_counts.clone()
        draft_exact_page_token_counts.masked_fill_(
            ~hit_cluster_mask.unsqueeze(-1),
            0,
        )

        (
            miss_estimation_keys,
            miss_estimation_values,
            miss_estimation_token_counts,
        ) = self._build_estimation_selection(
            packed.cluster_keys,
            packed.cluster_values,
            packed.cluster_token_counts,
            cluster_zones.sparse_retrieval_indices,
            miss_cluster_mask,
        )

        estimation_keys = torch.cat(
            (
                plan.sparse_estimation_keys,
                miss_estimation_keys,
            ),
            dim=2,
        ).contiguous()
        estimation_values = torch.cat(
            (
                plan.sparse_estimation_values,
                miss_estimation_values,
            ),
            dim=2,
        ).contiguous()
        estimation_token_counts = torch.cat(
            (
                plan.sparse_estimation_token_counts,
                miss_estimation_token_counts,
            ),
            dim=2,
        ).contiguous()

        hit_attn_by_head = self._sum_selected_scores(
            cluster_scores,
            cluster_zones.sparse_retrieval_indices,
            hit_cluster_mask,
        )

        has_clusters = packed.cluster_mask.any(dim=2)
        hit_attn_by_head = torch.where(
            has_clusters,
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
        if not self.cluster_store.is_cpu_backed or not self.cluster_store.pin_memory:
            return

        cluster_ids = plan.sparse_exact_cluster_ids
        page_ids = plan.sparse_exact_page_ids

        if active_mask.ndim != 1 or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a one-dimensional boolean tensor")
        if active_mask.shape != (cluster_ids.shape[0],):
            raise ValueError("active_mask does not match the selection batch size")
        if active_mask.device != cluster_ids.device:
            raise ValueError("active_mask and selection plan must use one device")
        if page_ids.numel() == 0:
            return

        active_cluster_mask = active_mask[:, None, None]
        active_page_mask = active_mask[:, None, None, None]

        prefetch_cluster_ids = cluster_ids.masked_fill(~active_cluster_mask, -1)
        prefetch_page_ids = page_ids.masked_fill(~active_page_mask, -1)

        self.cluster_store.admit_resident_clusters(
            layer_name=plan.layer_name,
            cluster_ids=prefetch_cluster_ids,
            page_ids=prefetch_page_ids,
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
                indexed_token_count = record.indexed_end - self.block_size

            primary_token_counts.append(seq_len - indexed_token_count)

        packed = self._pack_indices(
            layer_name,
            request_ids,
            key_cache,
            block_table,
        )

        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int64,
            device=block_table.device,
        )
        _, valid_token_mask, _ = self._build_token_layout(
            block_table,
            seq_lens_tensor,
        )

        # Every committed token not owned by a complete clustered segment is
        # part of the exact primary/steady zone.
        primary_token_mask = valid_token_mask & ~packed.indexed_token_mask
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
        clustered_exact_token_counts = packed.cluster_page_token_counts.sum(
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
            exact_cluster_ids=packed.cluster_ids,
            exact_page_ids=packed.cluster_page_ids,
            exact_page_token_counts=packed.cluster_page_token_counts,
            exact_token_counts=exact_token_counts,
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

        packed = self._pack_indices(
            layer_name,
            request_ids,
            key_cache,
            block_table,
        )

        _, valid_token_mask, forced_exact_mask = self._build_token_layout(
            block_table,
            seq_lens,
        )

        # All valid tokens not covered by a complete clustered segment remain
        # in the exact steady zone.
        forced_exact_mask |= valid_token_mask & ~packed.indexed_token_mask

        workspace = None
        if query.device.type == "cuda":
            workspace = self._get_cluster_selection_workspace(
                query,
                packed.cluster_keys,
            )

        cluster_scores = self._score_clusters(
            query,
            packed.cluster_keys,
            packed.cluster_mask,
            packed.cluster_token_counts,
            scale,
            workspace,
        )

        cluster_zones = self._select_cluster_zones(
            cluster_scores,
            packed.cluster_mask,
            workspace,
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

        has_clusters = packed.cluster_mask.any(dim=2)
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

        plan = self._make_plan(
            layer_name=layer_name,
            forced_exact_mask=forced_exact_mask,
            cluster_zones=cluster_zones,
            sparse_attn=sparse_attn,
            expanded_attn=expanded_attn,
            packed=packed,
        )

        return self._materialize_draft_selection(
            plan=plan,
            packed=packed,
            cluster_zones=cluster_zones,
            cluster_scores=cluster_scores,
            active_mask=active_mask,
        )

    def materialize(
        self,
        plan: RetroSpecSelectionPlan | RetroSpecTokenSelectionPlan,
        level: RetroSpecAttentionLevel,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecTokenAttentionSelection:
        if not isinstance(plan, RetroSpecTokenSelectionPlan):
            raise TypeError("Segmented token index requires a token selection plan")

        # These parameters remain in the common index interface. Exact KV is now
        # gathered later by the reusable execution buffer.
        del key_cache, value_cache, block_table

        return self._materialize_token_selection(plan, level)
