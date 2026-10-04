# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Mapping
from time import perf_counter

import torch

from .cluster_identity import RetroSpecClusterGroup
from .resident_cache_types import (
    RetroSpecCompactResidentPageAccess,
    RetroSpecCompactVerificationPageAccess,
    RetroSpecRankedDraftResidentAccess,
    RetroSpecResidentPageAccess,
    RetroSpecResidentReadLease,
    _ClusterId,
)
from .resident_kernels import (
    lookup_resident_handles,
    resolve_compact_draft_pages,
    resolve_compact_verification_pages,
    resolve_ranked_draft_buckets,
)


class _ResidentLookupMixin:
    """GPU and CPU resident lookup paths for the shared cache."""

    def lookup(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        touch: bool = True,
        include_pending: bool = True,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Resolve selected cluster blocks against resident GPU slots.

        When include_pending is false, clusters whose H2D copies are incomplete
        are reported as misses. Draft attention can then continue with centroid
        estimation instead of waiting for verification prefetches.
        """
        if not include_pending:
            self._reap_completed_copy_batches()

        (
            cluster_ids_cpu,
            page_ids_cpu,
            parsed_cluster_ids,
            parsed_cluster_pages,
            valid_positions,
        ) = self._parse_clusters(
            cluster_ids,
            page_ids,
            allocated_cluster_ids,
            allocated_page_ids,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )
        self._validate_cluster_groups(parsed_cluster_ids, cluster_groups)

        cache_page_ids_cpu = torch.full_like(page_ids_cpu, -1)
        flat_cache_page_ids = cache_page_ids_cpu.reshape(
            len(parsed_cluster_ids),
            page_ids_cpu.shape[-1],
        )

        hit_cluster_mask_cpu = torch.zeros(
            cluster_ids_cpu.shape,
            dtype=torch.bool,
        )
        miss_cluster_mask_cpu = torch.zeros_like(hit_cluster_mask_cpu)
        hit_gate_ready_mask_cpu = torch.zeros_like(hit_cluster_mask_cpu)

        flat_hit_mask = hit_cluster_mask_cpu.reshape(-1)
        flat_miss_mask = miss_cluster_mask_cpu.reshape(-1)
        flat_hit_gate_ready_mask = hit_gate_ready_mask_cpu.reshape(-1)
        hit_clusters: set[_ClusterId] = set()

        for cluster_index, (
            cluster_id,
            logical_pages,
            positions,
        ) in enumerate(
            zip(
                parsed_cluster_ids,
                parsed_cluster_pages,
                valid_positions,
            )
        ):
            if cluster_id is None:
                continue

            group = cluster_groups[cluster_id]
            flat_hit_gate_ready_mask[cluster_index] = self._is_group_hit_gate_ready(
                group
            )

            resident_slots = self._cluster_to_slots.get(cluster_id)
            pending = cluster_id in self._pending_cluster_events

            if resident_slots is None or (pending and not include_pending):
                flat_miss_mask[cluster_index] = True
                continue

            resident_group = self._cluster_to_group.get(cluster_id)
            if resident_group != cluster_groups[cluster_id]:
                raise RuntimeError(
                    "Resident cluster group does not match requested ownership"
                )

            resident_pages = self._cluster_to_pages[cluster_id]
            if resident_pages != logical_pages:
                raise RuntimeError(
                    "Resident cluster pages do not match the requested descriptor"
                )

            flat_hit_mask[cluster_index] = True
            hit_clusters.add(cluster_id)

            for position, slot_id in zip(positions, resident_slots):
                flat_cache_page_ids[cluster_index, position] = slot_id

        if touch:
            requested_clusters = self._priority_ordered_clusters(
                parsed_cluster_ids,
                cluster_ids_cpu.shape,
            )

            # Touch low-priority entries first so rank-zero clusters finish as MRU.
            for cluster_id in reversed(requested_clusters):
                if cluster_id in hit_clusters:
                    self._touch_cluster(cluster_id)

        ready_event = self._pending_event_for_clusters(hit_clusters)

        return RetroSpecResidentPageAccess(
            cache_page_ids=cache_page_ids_cpu.to(
                device=page_ids.device,
                non_blocking=False,
            ),
            hit_cluster_mask=hit_cluster_mask_cpu.to(
                device=cluster_ids.device,
                non_blocking=False,
            ),
            miss_cluster_mask=miss_cluster_mask_cpu.to(
                device=cluster_ids.device,
                non_blocking=False,
            ),
            hit_gate_ready_mask=hit_gate_ready_mask_cpu.to(
                device=cluster_ids.device,
                non_blocking=False,
            ),
            logical_page_ids_cpu=page_ids_cpu,
            miss_cluster_mask_cpu=miss_cluster_mask_cpu,
            ready_event=ready_event,
        )

    def lookup_gpu(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        active_mask: torch.Tensor | None,
        cache_page_ids: torch.Tensor,
        hit_cluster_mask: torch.Tensor,
        miss_cluster_mask: torch.Tensor,
        hit_gate_ready_mask: torch.Tensor,
        access_kinds: torch.Tensor,
        plan_row_indices: torch.Tensor | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Resolve handles without synchronizing or parsing on the CPU."""
        if cluster_ids.device != self.device or page_ids.device != self.device:
            raise ValueError("GPU resident lookup tensors must use the cache device")
        if page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Cluster IDs and logical pages do not match")
        output_batch = (
            cluster_ids.shape[0]
            if plan_row_indices is None
            else plan_row_indices.shape[0]
        )
        output_cluster_shape = (output_batch, *cluster_ids.shape[1:])
        output_page_shape = (*output_cluster_shape, page_ids.shape[-1])
        if cache_page_ids.shape != output_page_shape:
            raise ValueError("Resident page output has the wrong indexed shape")
        for output in (
            hit_cluster_mask,
            miss_cluster_mask,
            hit_gate_ready_mask,
            access_kinds,
        ):
            if output.shape != output_cluster_shape:
                raise ValueError("Resident lookup output has the wrong indexed shape")

        self._gpu_access_lock.acquire()
        try:
            self._ensure_handle_table(page_ids.shape[-1])
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            lookup_resident_handles(
                cluster_handles=cluster_ids,
                logical_page_ids=page_ids,
                active_mask=active_mask,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                output_page_slots=cache_page_ids,
                output_hit_mask=hit_cluster_mask,
                output_miss_mask=miss_cluster_mask,
                output_hit_gate_ready=hit_gate_ready_mask,
                output_access_kinds=access_kinds,
                plan_row_indices=plan_row_indices,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecResidentPageAccess(
            cache_page_ids=cache_page_ids,
            hit_cluster_mask=hit_cluster_mask,
            miss_cluster_mask=miss_cluster_mask,
            hit_gate_ready_mask=hit_gate_ready_mask,
            logical_page_ids_cpu=None,
            miss_cluster_mask_cpu=None,
            ready_event=None,
            access_kinds=access_kinds,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )

    def lookup_ranked_compact_draft_gpu(
        self,
        *,
        ranked_values: torch.Tensor,
        candidate_counts: torch.Tensor,
        arena_resident_table_buckets: torch.Tensor,
        arena_cluster_page_starts: torch.Tensor,
        arena_cluster_page_counts: torch.Tensor,
        arena_page_ids: torch.Tensor,
        arena_page_token_counts: torch.Tensor,
        arena_cluster_offsets: torch.Tensor,
        arena_page_offsets: torch.Tensor,
        request_slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
        retrieval_ratio: float,
        estimation_ratio: float,
        expanded_retrieval_width: int,
        max_pages_per_cluster: int,
        fallback_token_counts: torch.Tensor,
        sparse_cluster_indices: torch.Tensor,
        cluster_handles: torch.Tensor,
        cache_page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
        page_counts: torch.Tensor,
        clustered_token_counts: torch.Tensor,
        attention_mass: torch.Tensor,
        hit_attention_by_head: torch.Tensor,
        selected_cluster_counts: torch.Tensor,
        hit_cluster_counts: torch.Tensor,
        miss_cluster_counts: torch.Tensor,
        hit_gate_ready: torch.Tensor,
        miss_cluster_ids: torch.Tensor,
        miss_positions: torch.Tensor,
        miss_count: torch.Tensor,
        sparse_attention: torch.Tensor,
        expanded_attention: torch.Tensor,
        emit_misses: bool = True,
        statistics_buffer: torch.Tensor | None = None,
        statistics_indices: tuple[int, ...] | None = None,
    ) -> RetroSpecCompactResidentPageAccess:
        """Resolve ranked DRAFT rows without logical-page intermediates."""
        if ranked_values.device != self.device:
            raise ValueError("Ranked draft lookup must use the cache device")

        self._gpu_access_lock.acquire()
        try:
            self._ensure_handle_table(max_pages_per_cluster)
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            resolve_compact_draft_pages(
                ranked_values=ranked_values,
                candidate_counts=candidate_counts,
                arena_resident_table_buckets=arena_resident_table_buckets,
                arena_cluster_page_starts=arena_cluster_page_starts,
                arena_cluster_page_counts=arena_cluster_page_counts,
                arena_page_ids=arena_page_ids,
                arena_page_token_counts=arena_page_token_counts,
                arena_cluster_offsets=arena_cluster_offsets,
                arena_page_offsets=arena_page_offsets,
                request_slot_ids=request_slot_ids,
                active_mask=active_mask,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                retrieval_ratio=retrieval_ratio,
                estimation_ratio=estimation_ratio,
                expanded_retrieval_width=expanded_retrieval_width,
                max_pages_per_cluster=max_pages_per_cluster,
                fallback_token_counts=fallback_token_counts,
                sparse_cluster_indices=sparse_cluster_indices,
                cluster_handles=cluster_handles,
                output_page_slots=cache_page_ids,
                output_page_token_counts=page_token_counts,
                output_page_counts=page_counts,
                output_clustered_token_counts=clustered_token_counts,
                output_attention=attention_mass,
                output_hit_attention_by_head=hit_attention_by_head,
                output_selected_counts=selected_cluster_counts,
                output_hit_counts=hit_cluster_counts,
                output_miss_counts=miss_cluster_counts,
                output_gate_ready=hit_gate_ready,
                output_miss_handles=miss_cluster_ids,
                output_miss_positions=miss_positions,
                output_miss_count=miss_count,
                sparse_attention=sparse_attention,
                expanded_attention=expanded_attention,
                emit_misses=emit_misses,
                statistics_buffer=statistics_buffer,
                statistics_indices=statistics_indices,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecCompactResidentPageAccess(
            cache_page_ids=cache_page_ids,
            page_token_counts=page_token_counts,
            page_counts=page_counts,
            clustered_token_counts=clustered_token_counts,
            attention_mass=attention_mass,
            selected_cluster_counts=selected_cluster_counts,
            hit_cluster_counts=hit_cluster_counts,
            miss_cluster_counts=miss_cluster_counts,
            hit_gate_ready=hit_gate_ready,
            access_kinds=None,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )

    def lookup_ranked_draft_gpu(
        self,
        *,
        ranked_values: torch.Tensor,
        ranked_indices: torch.Tensor,
        candidate_counts: torch.Tensor,
        arena_cluster_ids: torch.Tensor,
        arena_resident_table_buckets: torch.Tensor,
        arena_cluster_token_counts: torch.Tensor,
        arena_cluster_page_counts: torch.Tensor,
        arena_cluster_offsets: torch.Tensor,
        arena_generations: torch.Tensor,
        request_slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
        retrieval_ratio: float,
        estimation_ratio: float,
        expanded_retrieval_width: int,
        max_pages_per_cluster: int,
        plan_valid_rows: torch.Tensor,
        output_request_slot_ids: torch.Tensor,
        output_request_slot_generations: torch.Tensor,
        cluster_handles: torch.Tensor,
        resident_bucket_ids: torch.Tensor,
        clustered_token_counts: torch.Tensor,
        attention_mass: torch.Tensor,
        hit_attention_by_head: torch.Tensor,
        selected_cluster_counts: torch.Tensor,
        hit_cluster_counts: torch.Tensor,
        miss_cluster_counts: torch.Tensor,
        hit_gate_ready: torch.Tensor,
        miss_cluster_ids: torch.Tensor,
        miss_positions: torch.Tensor,
        miss_count: torch.Tensor,
        sparse_attention: torch.Tensor,
        expanded_attention: torch.Tensor,
        capture_request_descriptors: bool,
        emit_misses: bool = True,
        statistics_buffer: torch.Tensor | None = None,
        statistics_indices: tuple[int, ...] | None = None,
    ) -> RetroSpecRankedDraftResidentAccess:
        """Resolve ranked DRAFT clusters without compact page intermediates."""
        if ranked_values.device != self.device:
            raise ValueError("Ranked DRAFT lookup must use the cache device")

        stats = self.performance_stats
        lock_started_at = (
            perf_counter() if stats is not None and stats.enabled else None
        )
        self._gpu_access_lock.acquire()
        if lock_started_at is not None:
            stats.record_cpu_time(
                "draft_ranked_resident_lock_wait_wall",
                perf_counter() - lock_started_at,
            )
        try:
            self._ensure_handle_table(max_pages_per_cluster)
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            resolve_ranked_draft_buckets(
                ranked_values=ranked_values,
                ranked_indices=ranked_indices,
                candidate_counts=candidate_counts,
                arena_cluster_ids=arena_cluster_ids,
                arena_resident_table_buckets=arena_resident_table_buckets,
                arena_cluster_token_counts=arena_cluster_token_counts,
                arena_cluster_page_counts=arena_cluster_page_counts,
                arena_cluster_offsets=arena_cluster_offsets,
                arena_generations=arena_generations,
                request_slot_ids=request_slot_ids,
                active_mask=active_mask,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                retrieval_ratio=retrieval_ratio,
                estimation_ratio=estimation_ratio,
                expanded_retrieval_width=expanded_retrieval_width,
                max_pages_per_cluster=max_pages_per_cluster,
                output_valid_rows=plan_valid_rows,
                output_request_slot_ids=output_request_slot_ids,
                output_request_slot_generations=(output_request_slot_generations),
                output_cluster_handles=cluster_handles,
                output_resident_buckets=resident_bucket_ids,
                output_clustered_token_counts=clustered_token_counts,
                output_attention=attention_mass,
                output_hit_attention_by_head=hit_attention_by_head,
                output_selected_counts=selected_cluster_counts,
                output_hit_counts=hit_cluster_counts,
                output_miss_counts=miss_cluster_counts,
                output_gate_ready=hit_gate_ready,
                output_miss_handles=miss_cluster_ids,
                output_miss_positions=miss_positions,
                output_miss_count=miss_count,
                sparse_attention=sparse_attention,
                expanded_attention=expanded_attention,
                capture_request_descriptors=capture_request_descriptors,
                emit_misses=emit_misses,
                statistics_buffer=statistics_buffer,
                statistics_indices=statistics_indices,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecRankedDraftResidentAccess(
            cluster_handles=cluster_handles,
            resident_bucket_ids=resident_bucket_ids,
            clustered_token_counts=clustered_token_counts,
            attention_mass=attention_mass,
            selected_cluster_counts=selected_cluster_counts,
            hit_cluster_counts=hit_cluster_counts,
            miss_cluster_counts=miss_cluster_counts,
            hit_gate_ready=hit_gate_ready,
            resident_table_page_counts=self._handle_table_page_counts,
            resident_table_page_slots=self._handle_table_page_slots,
            resident_key_pages=self.key_pages,
            resident_value_pages=self.value_pages,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )

    def lookup_compact_verification_gpu(
        self,
        selected_cluster_indices: torch.Tensor,
        plan_valid_rows: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
        arena_cluster_ids: torch.Tensor,
        arena_cluster_page_starts: torch.Tensor,
        arena_cluster_page_counts: torch.Tensor,
        arena_page_ids: torch.Tensor,
        arena_page_token_counts: torch.Tensor,
        arena_cluster_offsets: torch.Tensor,
        arena_page_offsets: torch.Tensor,
        arena_generations: torch.Tensor,
        resident_page_ids: torch.Tensor,
        staging_page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
        page_counts: torch.Tensor,
        selected_cluster_counts: torch.Tensor,
        hit_cluster_counts: torch.Tensor,
        miss_cluster_counts: torch.Tensor,
        unique_cluster_ids: torch.Tensor,
        unique_logical_page_ids: torch.Tensor,
        unique_page_counts: torch.Tensor,
        miss_hash_buckets: torch.Tensor,
        miss_unique_indices: torch.Tensor,
        miss_output_page_offsets: torch.Tensor,
        miss_count: torch.Tensor,
        unique_miss_count: torch.Tensor,
        miss_table_handles: torch.Tensor,
        miss_table_unique_indices: torch.Tensor,
        invalid_descriptor_count: torch.Tensor,
    ) -> RetroSpecCompactVerificationPageAccess:
        """Resolve verification selections directly from the resident arena."""
        if selected_cluster_indices.device != self.device:
            raise ValueError("Compact verification lookup uses the wrong device")

        self._gpu_access_lock.acquire()
        try:
            self._ensure_handle_table(unique_logical_page_ids.shape[1])
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            resolve_compact_verification_pages(
                selected_cluster_indices=selected_cluster_indices,
                plan_valid_rows=plan_valid_rows,
                request_slot_ids=request_slot_ids,
                request_slot_generations=request_slot_generations,
                arena_cluster_ids=arena_cluster_ids,
                arena_cluster_page_starts=arena_cluster_page_starts,
                arena_cluster_page_counts=arena_cluster_page_counts,
                arena_page_ids=arena_page_ids,
                arena_page_token_counts=arena_page_token_counts,
                arena_cluster_offsets=arena_cluster_offsets,
                arena_page_offsets=arena_page_offsets,
                arena_generations=arena_generations,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                output_resident_page_ids=resident_page_ids,
                output_staging_page_ids=staging_page_ids,
                output_page_token_counts=page_token_counts,
                output_page_counts=page_counts,
                output_selected_counts=selected_cluster_counts,
                output_hit_counts=hit_cluster_counts,
                output_miss_counts=miss_cluster_counts,
                output_miss_hash_buckets=miss_hash_buckets,
                output_miss_unique_indices=miss_unique_indices,
                output_miss_page_offsets=miss_output_page_offsets,
                output_miss_count=miss_count,
                output_unique_handles=unique_cluster_ids,
                output_unique_logical_page_ids=unique_logical_page_ids,
                output_unique_page_counts=unique_page_counts,
                output_unique_miss_count=unique_miss_count,
                miss_table_handles=miss_table_handles,
                miss_table_unique_indices=miss_table_unique_indices,
                output_invalid_descriptor_count=invalid_descriptor_count,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecCompactVerificationPageAccess(
            resident_page_ids=resident_page_ids,
            staging_page_ids=staging_page_ids,
            page_token_counts=page_token_counts,
            page_counts=page_counts,
            selected_cluster_counts=selected_cluster_counts,
            hit_cluster_counts=hit_cluster_counts,
            miss_cluster_counts=miss_cluster_counts,
            unique_cluster_ids=unique_cluster_ids,
            unique_logical_page_ids=unique_logical_page_ids,
            unique_page_counts=unique_page_counts,
            miss_unique_indices=miss_unique_indices,
            miss_output_page_offsets=miss_output_page_offsets,
            miss_count=miss_count,
            unique_miss_count=unique_miss_count,
            invalid_descriptor_count=invalid_descriptor_count,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )
