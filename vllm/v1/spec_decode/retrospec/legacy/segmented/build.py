# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext

import torch

from vllm.v1.spec_decode.retrospec.clustering import segmented_kmeans
from vllm.v1.spec_decode.retrospec.legacy.cluster_store import (
    RetroSpecClusterBlockTable,
    RetroSpecFullVerificationDescriptor,
    RetroSpecStagedClusterInput,
)
from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecClusterSummary,
    RetroSpecResidentSegment,
    RetroSpecStagedClusterSummary,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.types import (
    _CompletedRequestLayerSegment,
    _RequestLayerIndex,
    _RequestLayerSegment,
    _StagedRequestLayerSegment,
)


class _RetroSpecSegmentedTokenIndexBuildMixin:
    def _cuda_timer(self, name: str) -> AbstractContextManager[None]:
        if self.performance_stats is None:
            return nullcontext()
        return self.performance_stats.cuda_timer(name)

    def _cpu_timer(self, name: str) -> AbstractContextManager[None]:
        if self.performance_stats is None:
            return nullcontext()
        return self.performance_stats.cpu_timer(name)

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
        request_ids = tuple(request_ids)
        self.end_indexed_verification_transaction()
        self._discard_primed_full_verification_for_requests(request_ids, wait=True)
        self.cluster_store.wait_for_resident_prefetches()
        self._gpu_index_residency.invalidate_requests(request_ids)

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
        # Sparse miss admission may overlap boundary handling and proposal
        # teardown, but it must not compete with the full-verification H2D
        # pipeline for PCIe bandwidth or mutate resident state underneath it.
        self.cluster_store.wait_for_verification_admissions()
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

        rollbacks: list[tuple[str, str]] = []
        for layer_name in layer_names:
            layer_indices = self._indices.get(layer_name, {})
            for request_id, context_len in zip(request_ids, context_lens):
                record = layer_indices.get(request_id)
                if (
                    record is not None
                    and record.segments
                    and record.indexed_end > context_len
                ):
                    rollbacks.append((layer_name, request_id))

        if not rollbacks:
            return

        rollback_request_ids = tuple({request_id for _, request_id in rollbacks})
        self._discard_primed_full_verification_for_requests(
            rollback_request_ids, wait=True
        )
        self.cluster_store.wait_for_resident_prefetches(layer_names)
        rollback_keys = set(rollbacks)

        for layer_name in layer_names:
            layer_indices = self._indices.get(layer_name)
            if layer_indices is None:
                continue

            layer_changed = False
            for request_id, context_len in zip(request_ids, context_lens):
                if (layer_name, request_id) not in rollback_keys:
                    continue
                record = layer_indices.get(request_id)
                assert record is not None

                self._free_record(layer_name, record)
                layer_indices[request_id] = self._empty_index()
                self._gpu_index_residency.discard_request_layer(layer_name, request_id)
                layer_changed = True

            if layer_changed:
                self._gpu_index_residency.invalidate_active_view(layer_name)

    def has_cluster_pages(
        self,
        layer_name: str,
        request_ids: Sequence[str],
    ) -> bool:
        """Return whether any request owns committed clusters for one layer."""
        layer_indices = self._indices.get(layer_name, {})
        return any(
            (record := layer_indices.get(request_id)) is not None
            and record.num_clusters > 0
            for request_id in request_ids
        )

    def _capture_proposal_index_revisions(self, request_ids: Sequence[str]) -> None:
        self._proposal_index_revisions.clear()
        if not self.selection_provenance.enabled:
            return

        for layer_name, layer_indices in self._indices.items():
            for request_id in request_ids:
                record = layer_indices.get(request_id)
                self._proposal_index_revisions[(layer_name, request_id)] = (
                    -1 if record is None else record.revision
                )

    def _get_proposal_index_revisions(
        self, layer_name: str, request_ids: Sequence[str]
    ) -> tuple[int, ...]:
        if not self.selection_provenance.enabled:
            return ()

        layer_indices = self._indices.get(layer_name, {})
        revisions: list[int] = []
        for request_id in request_ids:
            record = layer_indices.get(request_id)
            current = -1 if record is None else record.revision
            expected = self._proposal_index_revisions.get((layer_name, request_id), -1)
            if current != expected:
                raise RuntimeError(
                    "RetroSpec logical cluster index changed inside a proposal: "
                    f"{layer_name!r}, {request_id!r}, {expected} -> {current}"
                )
            revisions.append(current)
        return tuple(revisions)

    def begin_proposal(self, request_ids: Sequence[str]) -> None:
        if self._proposal_active:
            raise RuntimeError("Segmented token index proposal is already active")
        if self._staged_segments:
            raise RuntimeError(
                "Cannot begin a proposal before staged index updates are flushed"
            )
        request_ids = tuple(request_ids)
        self._capture_proposal_index_revisions(request_ids)
        try:
            if self.replay_mode == "freeze_resident":
                self.cluster_store.begin_resident_replay(tuple(self._indices))
                self._proposal_resident_frozen = True
            self._gpu_index_residency.activate(request_ids)
        except BaseException:
            try:
                if self._proposal_resident_frozen:
                    self.cluster_store.end_resident_replay()
            finally:
                self._proposal_resident_frozen = False
                self._proposal_index_revisions.clear()
            raise
        for table in self._selection_plan_tables.values():
            table.valid_rows.zero_()
        self._selection_plan_written_layers.clear()
        self._proposal_read_leases.clear()
        self._proposal_active = True
        self._proposal_request_ids = request_ids

    def end_proposal(self) -> None:
        if not self._proposal_active:
            raise RuntimeError("Segmented token index proposal is not active")

        self.end_indexed_verification_transaction()
        try:
            self._gpu_index_residency.deactivate()
        finally:
            try:
                for lease in self._proposal_read_leases:
                    lease.release()
                self._proposal_read_leases.clear()
                self._selection_plan_written_layers.clear()
                self._proposal_index_revisions.clear()
                self._proposal_active = False
                self._proposal_request_ids = ()
            finally:
                if self._proposal_resident_frozen:
                    self.cluster_store.end_resident_replay()
                    self._proposal_resident_frozen = False

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

    def _allocate_full_verification_revision(self) -> int:
        revision = self._full_verification_revision_counter
        self._full_verification_revision_counter += 1
        return revision

    def _empty_index(
        self,
        indexed_end: int | None = None,
    ) -> _RequestLayerIndex:
        if indexed_end is None:
            indexed_end = self.block_size

        return _RequestLayerIndex(
            revision=self._allocate_full_verification_revision(),
            segments=[],
            num_clusters=0,
            indexed_end=indexed_end,
            full_verification_descriptor=None,
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

    def close(self) -> None:
        try:
            self.end_indexed_verification_transaction()
            if self.has_staged_updates:
                self.discard_staged_updates()
        finally:
            try:
                if self._full_verification_pipeline_active:
                    self.end_full_verification_pipeline()
                self._discard_primed_full_verification(wait=True, record_outcome=False)
            finally:
                try:
                    for stream in self._prefill_hint_streams.values():
                        stream.synchronize()
                    self._prefill_hint_streams.clear()
                    self._prefill_hint_selection_workspace = None
                    self.cluster_store.close()
                finally:
                    self._gpu_index_residency.close()

        self._pinned_memory.assert_empty()

    def _get_cluster_build_executor(self) -> ThreadPoolExecutor:
        if self._cluster_build_executor is None:
            self._cluster_build_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="retrospec-cluster-page",
            )

        return self._cluster_build_executor

    def _finish_cluster_build(
        self,
        layer_name: str,
        request_id: str,
        indexed_start: int,
        indexed_end: int,
        cluster_start: int,
        staged_summary: RetroSpecStagedClusterSummary,
        staged_clusters: RetroSpecStagedClusterInput,
    ) -> _CompletedRequestLayerSegment:
        try:
            cluster_summary = self._gpu_index_residency.finish_cluster_summary(
                staged_summary
            )
        except BaseException:
            self.cluster_store.discard_staged_clusters(staged_clusters)
            raise

        if self.selection_provenance.enabled:
            staged_clusters.wait()
            self.selection_provenance.record_index_segment(
                request_id=request_id,
                layer_name=layer_name,
                indexed_start=indexed_start,
                indexed_end=indexed_end,
                cluster_start=cluster_start,
                assignments=staged_clusters.assignments,
                cluster_sizes=cluster_summary.cluster_token_counts,
                cluster_keys=cluster_summary.cluster_keys,
                cluster_values=cluster_summary.cluster_values,
                token_offsets_in_cluster=staged_clusters.token_offsets_in_cluster,
            )

        cluster_blocks = self.cluster_store.store_staged_clusters(
            layer_name=layer_name,
            request_id=request_id,
            cluster_start=cluster_start,
            staged=staged_clusters,
        )
        return _CompletedRequestLayerSegment(
            cluster_summary=cluster_summary,
            cluster_blocks=cluster_blocks,
        )

    def _submit_cluster_build(
        self,
        layer_name: str,
        request_id: str,
        indexed_start: int,
        indexed_end: int,
        cluster_start: int,
        staged_summary: RetroSpecStagedClusterSummary,
        staged_clusters: RetroSpecStagedClusterInput,
    ) -> Future[_CompletedRequestLayerSegment]:
        executor = self._get_cluster_build_executor()
        build_future = executor.submit(
            self._finish_cluster_build,
            layer_name,
            request_id,
            indexed_start,
            indexed_end,
            cluster_start,
            staged_summary,
            staged_clusters,
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
    ) -> tuple[int, torch.cuda.Event | None]:
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

        indexed_end = indexed_start + token_keys.shape[1]
        try:
            build_future = self._submit_cluster_build(
                layer_name=layer_name,
                request_id=request_id,
                indexed_start=indexed_start,
                indexed_end=indexed_end,
                cluster_start=cluster_start,
                staged_summary=staged_summary,
                staged_clusters=staged_clusters,
            )
        except BaseException:
            self.cluster_store.discard_staged_clusters(staged_clusters)
            self._gpu_index_residency.discard_cluster_summary(staged_summary)
            raise

        self._staged_segments.append(
            _StagedRequestLayerSegment(
                layer_name=layer_name,
                request_id=request_id,
                indexed_start=indexed_start,
                indexed_end=indexed_end,
                cluster_start=cluster_start,
                resident_summary=staged_summary.resident_summary,
                build_future=build_future,
            )
        )
        return phase_result.cluster_sizes.shape[1], staged_clusters.ready_event

    def has_staged_request_layer(self, layer_name: str, request_id: str) -> bool:
        return (layer_name, request_id) in self._staged_segment_keys

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
        descriptor: RetroSpecFullVerificationDescriptor,
    ) -> None:
        """Append immutable valid-token ranges for one completed segment."""
        previous = record.full_verification_descriptor
        record.full_verification_descriptor = (
            descriptor if previous is None else previous.append(descriptor)
        )

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
        changed_request_ids = tuple(
            {staged_segment.request_id for staged_segment, _, _ in built_segments}
        )
        self._discard_primed_full_verification_for_requests(
            changed_request_ids, wait=False
        )
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
                        revision=self._allocate_full_verification_revision(),
                        segments=list(current_record.segments),
                        num_clusters=current_record.num_clusters,
                        indexed_end=current_record.indexed_end,
                        full_verification_descriptor=(
                            current_record.full_verification_descriptor
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

            block_metadata = cluster_blocks.page_metadata
            resident_segments.append(
                self._gpu_index_residency.build_resident_segment(
                    layer_name=staged_segment.layer_name,
                    request_id=staged_segment.request_id,
                    indexed_start=staged_segment.indexed_start,
                    indexed_end=staged_segment.indexed_end,
                    cluster_start=staged_segment.cluster_start,
                    resident_summary=staged_segment.resident_summary,
                    cluster_token_counts_cpu=summary.cluster_token_counts,
                    cluster_ids=cluster_blocks.cluster_ids,
                    cluster_page_ids=block_metadata.page_ids,
                    cluster_page_token_counts=block_metadata.page_token_counts,
                )
            )
            self._append_full_verification_page_descriptor(
                record,
                cluster_blocks.full_verification_descriptor,
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
        for segment in resident_segments:
            if segment.cluster_keys.device.type != "cuda":
                continue
            self.cluster_store.republish_resident_table_bindings(
                layer_name=segment.layer_name,
                cluster_ids=segment.cluster_ids_cpu,
                stream=torch.cuda.current_stream(segment.cluster_keys.device),
            )
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
                try:
                    completed = staged_segment.build_future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    continue

                built_segments.append(
                    (
                        staged_segment,
                        completed.cluster_summary,
                        completed.cluster_blocks,
                    )
                )

            if first_error is not None:
                self._release_built_segments(built_segments)
                raise first_error

            publish_context = (
                nullcontext()
                if self.performance_stats is None
                else self.performance_stats.cpu_timer("cluster_publish_wall")
            )
            try:
                with publish_context:
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
                    completed = staged_segment.build_future.result()
                except BaseException:
                    # A failed store_clusters() call releases allocations made
                    # before it raises. The original prefill error takes priority.
                    continue

                try:
                    self.cluster_store.free(
                        staged_segment.layer_name,
                        completed.cluster_blocks,
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
    ) -> torch.cuda.Event | None:
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
        last_ready_event: torch.cuda.Event | None = None

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
                self._discard_primed_full_verification_for_requests(
                    (request_id,), wait=True
                )
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

                num_phase_clusters, ready_event = self._stage_clustering_phase(
                    layer_name=layer_name,
                    request_id=request_id,
                    indexed_start=phase_start,
                    cluster_start=cluster_start,
                    segment_size=phase_segment_size,
                    token_keys=token_keys,
                    token_values=token_values,
                )
                if ready_event is not None:
                    last_ready_event = ready_event
                phase_start = phase_end
                cluster_start += num_phase_clusters

            if phase_start != desired_end:
                raise RuntimeError("Clustering phases do not cover the update")
            if not defer_cpu_store:
                self.flush_staged_updates()

        if layer_changed:
            self._gpu_index_residency.invalidate_active_view(layer_name)
        return last_ready_event
