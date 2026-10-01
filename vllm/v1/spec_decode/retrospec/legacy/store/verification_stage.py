# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import perf_counter

import torch

from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecResidentLayerArena,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_cache import (
    RetroSpecRankedDraftResidentAccess,
)
from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    RetroSpecCompactResolvedClusterPages,
    RetroSpecRankedDraftResolvedClusters,
    _LayerClusterPagePool,
    _PinnedVerificationMissSlot,
    _VerificationResolveGPUArena,
)


class _RetroSpecClusterPageStoreVerificationStageMixin:
    def _get_verification_metadata_stream(
        self, device: torch.device
    ) -> torch.cuda.Stream:
        device = self._canonical_cuda_device(device)
        stream = self._verification_metadata_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._verification_metadata_streams[device] = stream
        return stream

    def reserve_verification_resolve_workspace(
        self,
        device: torch.device,
        cluster_capacity: int,
        max_pages: int,
        row_capacity: int,
        page_capacity: int,
    ) -> None:
        """Reserve compact records from the actual active selection width."""
        if cluster_capacity <= 0:
            return
        device = self._canonical_cuda_device(device)

        with self._verification_resolve_lock:
            slots = self._verification_miss_slots.setdefault(
                device,
                [
                    _PinnedVerificationMissSlot(self._pinned_memory)
                    for _ in range(self._verification_resolve_ring_size)
                ],
            )
            arenas = self._verification_gpu_arenas.setdefault(
                device,
                [
                    _VerificationResolveGPUArena()
                    for _ in range(self._verification_resolve_ring_size)
                ],
            )
            if len(slots) < self._verification_resolve_ring_size:
                slots.extend(
                    _PinnedVerificationMissSlot(self._pinned_memory)
                    for _ in range(self._verification_resolve_ring_size - len(slots))
                )
                arenas.extend(
                    _VerificationResolveGPUArena()
                    for _ in range(self._verification_resolve_ring_size - len(arenas))
                )
            preallocated_slots = min(self._VERIFICATION_MISS_RING_SIZE, len(slots))
            for slot in slots[:preallocated_slots]:
                if slot.in_use and (
                    cluster_capacity > slot.capacity or max_pages > slot.max_pages
                ):
                    raise RuntimeError(
                        "Cannot grow an active verification-miss metadata slot"
                    )
                if not slot.in_use:
                    slot.reserve_capacity(cluster_capacity, max_pages)
            for arena in arenas[:preallocated_slots]:
                arena.reserve_capacity(
                    cluster_capacity,
                    max_pages,
                    row_capacity,
                    page_capacity,
                    device,
                )

    def _acquire_verification_resolve_workspace(
        self,
        device: torch.device,
        cluster_capacity: int,
        max_pages: int,
        row_capacity: int,
        page_capacity: int,
    ) -> tuple[
        _PinnedVerificationMissSlot,
        _VerificationResolveGPUArena,
    ]:
        self.reserve_verification_resolve_workspace(
            device,
            cluster_capacity,
            max_pages,
            row_capacity,
            page_capacity,
        )
        device = self._canonical_cuda_device(device)

        with self._verification_resolve_lock:
            cursor = self._verification_resolve_cursors.get(device, 0)
            self._verification_resolve_cursors[device] = (
                cursor + 1
            ) % self._verification_resolve_ring_size
            slot = self._verification_miss_slots[device][cursor]
            arena = self._verification_gpu_arenas[device][cursor]
            if slot.in_use:
                raise RuntimeError("Verification-miss metadata ring is exhausted")
            slot.reserve_capacity(cluster_capacity, max_pages)
            arena.reserve_capacity(
                cluster_capacity,
                max_pages,
                row_capacity,
                page_capacity,
                device,
            )
            slot.in_use = True

        if slot.reuse_ready_event is not None:
            slot.reuse_ready_event.synchronize()
            slot.reuse_ready_event = None
        return slot, arena

    def _release_verification_miss_slot(
        self,
        slot: _PinnedVerificationMissSlot,
        reuse_ready_event: torch.cuda.Event | None,
    ) -> None:
        with self._verification_resolve_lock:
            if not slot.in_use:
                raise RuntimeError("Verification-miss metadata slot was already free")
            slot.reuse_ready_event = reuse_ready_event
            slot.in_use = False

    def _stage_verification_miss_pages(
        self,
        pool: _LayerClusterPagePool,
        page_ids_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        num_pages = page_ids_cpu.numel()
        if num_pages == 0:
            shape = (0, pool.page_size, pool.head_size)
            key_pages = torch.empty(
                shape, dtype=pool.dtype, device=pool.metadata_device
            )
            return key_pages, torch.empty_like(key_pages), None

        staging_shape = (num_pages, pool.page_size, pool.head_size)
        staging_key_pages = torch.empty(
            staging_shape, dtype=pool.dtype, device=pool.metadata_device
        )
        staging_value_pages = torch.empty_like(staging_key_pages)
        transfer_buffer = self._get_full_verification_buffer(pool)
        page_capacity = transfer_buffer.cpu_slot_capacity(pool)
        transfer_stream = self._get_resident_prefetch_stream(pool.metadata_device)
        staging_key_pages.record_stream(transfer_stream)
        staging_value_pages.record_stream(transfer_stream)
        h2d_timer = (
            None
            if self.performance_stats is None
            else self.performance_stats.start_cuda_timer(
                "verification_miss_h2d", transfer_stream
            )
        )

        page_start = 0
        while page_start < num_pages:
            page_end = min(page_start + page_capacity, num_pages)
            gather_started_at = (
                perf_counter()
                if self.performance_stats is not None and self.performance_stats.enabled
                else None
            )
            cpu_keys, cpu_values, cpu_slot = transfer_buffer.stage_cpu_pages(
                pool, page_ids_cpu[page_start:page_end]
            )
            if gather_started_at is not None:
                self.performance_stats.record_cpu_time(
                    "verification_miss_cpu_gather",
                    perf_counter() - gather_started_at,
                )

            try:
                with torch.cuda.stream(transfer_stream):
                    staging_key_pages[page_start:page_end].copy_(
                        cpu_keys, non_blocking=self.pin_memory
                    )
                    staging_value_pages[page_start:page_end].copy_(
                        cpu_values, non_blocking=self.pin_memory
                    )
                    chunk_ready_event = torch.cuda.Event()
                    chunk_ready_event.record(transfer_stream)
            except BaseException:
                transfer_stream.synchronize()
                transfer_buffer.release_cpu_slot(cpu_slot, None)
                raise
            transfer_buffer.release_cpu_slot(cpu_slot, chunk_ready_event)
            page_start = page_end

        with torch.cuda.stream(transfer_stream):
            if self.performance_stats is not None:
                self.performance_stats.stop_cuda_timer(h2d_timer, transfer_stream)
            staging_ready_event = torch.cuda.Event()
            staging_ready_event.record(transfer_stream)

        if self.performance_stats is not None:
            self.performance_stats.add_counter("verification_miss_pages", num_pages)
            self.performance_stats.add_counter(
                "verification_miss_h2d_bytes",
                staging_key_pages.nbytes + staging_value_pages.nbytes,
            )
            self.performance_stats.observe_peak(
                "verification_miss_staging_bytes",
                staging_key_pages.nbytes + staging_value_pages.nbytes,
            )
        return staging_key_pages, staging_value_pages, staging_ready_event

    def _validate_unique_verification_miss_metadata(
        self,
        layer_name: str,
        max_pages: int,
        slot: _PinnedVerificationMissSlot,
        num_unique_misses: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            slot.unique_cluster_id_storage is None
            or slot.unique_logical_page_id_storage is None
            or slot.unique_page_count_storage is None
            or slot.unique_staging_start_storage is None
        ):
            raise RuntimeError("Unique verification-miss CPU storage is unavailable")

        cluster_ids = slot.unique_cluster_id_storage[:num_unique_misses].tolist()
        page_counts = slot.unique_page_count_storage[:num_unique_misses].tolist()
        logical_page_ids = slot.unique_logical_page_id_storage[
            :num_unique_misses, :max_pages
        ]

        descriptors = self._cluster_block_descriptors.get(layer_name)
        if descriptors is None:
            raise RuntimeError(f"No cluster descriptors exist for {layer_name!r}")
        allocated_cluster_ids = self._get_allocated_cluster_ids(layer_name)

        unique_page_ids: list[int] = []
        for unique_index, cluster_id_value in enumerate(cluster_ids):
            cluster_id = int(cluster_id_value)
            if cluster_id not in allocated_cluster_ids:
                raise RuntimeError(
                    f"Verification selected stale cluster handle {cluster_id}"
                )
            descriptor = descriptors[cluster_id]
            page_count = int(page_counts[unique_index])
            if page_count < 0 or page_count > max_pages:
                raise RuntimeError(
                    "Verification descriptor exceeds the packed page-table width"
                )
            emitted_pages = tuple(
                int(page_id)
                for page_id in logical_page_ids[unique_index, :page_count].tolist()
            )
            if descriptor.page_ids != emitted_pages:
                raise RuntimeError(
                    f"Verification selected stale pages for cluster {cluster_id}"
                )
            slot.unique_staging_start_storage[unique_index] = len(unique_page_ids)
            unique_page_ids.extend(emitted_pages)

        cluster_ids_cpu = torch.tensor(cluster_ids, dtype=torch.int64)
        metadata = self._materialize_cluster_block_metadata_cpu(
            layer_name, cluster_ids_cpu, page_width=max_pages
        )
        staging_page_ids_cpu = torch.full_like(metadata.page_ids, -1)
        for unique_index, page_count_value in enumerate(page_counts):
            page_count = int(page_count_value)
            staging_start = int(slot.unique_staging_start_storage[unique_index].item())
            staging_page_ids_cpu[unique_index, :page_count] = torch.arange(
                staging_start, staging_start + page_count, dtype=torch.int64
            )

        return (
            cluster_ids_cpu,
            metadata.page_ids,
            staging_page_ids_cpu,
            torch.tensor(unique_page_ids, dtype=torch.int64),
        )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            self._reap_verification_admissions(wait=True)
            self.wait_for_resident_prefetches()
        finally:
            self._resident_prefetch_executor.shutdown(wait=True)

        for buffer in self._full_verification_buffers.values():
            buffer.close()
        self._full_verification_buffers.clear()

        for slots in self._pinned_staging_slots.values():
            for slot in slots:
                if slot.in_use:
                    raise RuntimeError(
                        "Cannot close an active cluster-build staging slot"
                    )
                slot.release_storage()
        self._pinned_staging_slots.clear()

        for slots in self._resident_prefetch_slots.values():
            for slot in slots:
                if slot.in_use:
                    raise RuntimeError(
                        "Cannot close an active resident-prefetch staging slot"
                    )
                slot.release_storage()
        self._resident_prefetch_slots.clear()
        self._resident_prefetch_deferred.clear()
        self._resident_prefetch_last_metadata_events.clear()

        for slots in self._verification_miss_slots.values():
            for slot in slots:
                if slot.in_use:
                    raise RuntimeError(
                        "Cannot close an active verification-miss metadata slot"
                    )
                if slot.reuse_ready_event is not None:
                    slot.reuse_ready_event.synchronize()
                slot.release_storage()
        self._verification_miss_slots.clear()
        self._verification_gpu_arenas.clear()
        self._verification_resolve_cursors.clear()
        self._verification_metadata_streams.clear()

    def resolve_ranked_compact_draft_cluster_blocks(
        self,
        *,
        layer_name: str,
        ranked_values: torch.Tensor,
        candidate_counts: torch.Tensor,
        arena: RetroSpecResidentLayerArena,
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
    ) -> RetroSpecCompactResolvedClusterPages:
        """Resolve ranked DRAFT rows directly into compact resident pages."""
        if ranked_values.device.type != "cuda":
            raise ValueError("Ranked compact draft lookup requires CUDA")

        resident_cache = self._get_resident_cache_for_lookup(layer_name)

        access = resident_cache.lookup_ranked_compact_draft_gpu(
            ranked_values=ranked_values,
            candidate_counts=candidate_counts,
            arena_resident_table_buckets=arena.resident_table_buckets,
            arena_cluster_page_starts=arena.cluster_page_starts,
            arena_cluster_page_counts=arena.cluster_page_counts,
            arena_page_ids=arena.page_ids,
            arena_page_token_counts=arena.page_token_counts,
            arena_cluster_offsets=arena.cluster_offsets,
            arena_page_offsets=arena.page_offsets,
            request_slot_ids=request_slot_ids,
            active_mask=active_mask,
            retrieval_ratio=retrieval_ratio,
            estimation_ratio=estimation_ratio,
            expanded_retrieval_width=expanded_retrieval_width,
            max_pages_per_cluster=max_pages_per_cluster,
            fallback_token_counts=fallback_token_counts,
            sparse_cluster_indices=sparse_cluster_indices,
            cluster_handles=cluster_handles,
            cache_page_ids=cache_page_ids,
            page_token_counts=page_token_counts,
            page_counts=page_counts,
            clustered_token_counts=clustered_token_counts,
            attention_mass=attention_mass,
            hit_attention_by_head=hit_attention_by_head,
            selected_cluster_counts=selected_cluster_counts,
            hit_cluster_counts=hit_cluster_counts,
            miss_cluster_counts=miss_cluster_counts,
            hit_gate_ready=hit_gate_ready,
            miss_cluster_ids=miss_cluster_ids,
            miss_positions=miss_positions,
            miss_count=miss_count,
            sparse_attention=sparse_attention,
            expanded_attention=expanded_attention,
            emit_misses=emit_misses,
            statistics_buffer=self._draft_resolve_counter_buffer,
            statistics_indices=self._draft_resolve_counter_indices,
        )

        return RetroSpecCompactResolvedClusterPages(
            resident_page_ids=access.cache_page_ids,
            page_token_counts=access.page_token_counts,
            page_counts=access.page_counts,
            clustered_token_counts=access.clustered_token_counts,
            attention_mass=access.attention_mass,
            selected_cluster_counts=access.selected_cluster_counts,
            hit_cluster_counts=access.hit_cluster_counts,
            miss_cluster_counts=access.miss_cluster_counts,
            hit_gate_ready=access.hit_gate_ready,
            resident_key_pages=resident_cache.key_pages,
            resident_value_pages=resident_cache.value_pages,
            read_lease=access.read_lease,
        )

    def resolve_ranked_draft_clusters(
        self,
        *,
        layer_name: str,
        ranked_values: torch.Tensor,
        ranked_indices: torch.Tensor,
        candidate_counts: torch.Tensor,
        arena: RetroSpecResidentLayerArena,
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
    ) -> RetroSpecRankedDraftResolvedClusters:
        """Resolve ranked DRAFT clusters without expanding their page tables."""
        if ranked_values.device.type != "cuda":
            raise ValueError("Ranked DRAFT lookup requires CUDA")

        resident_cache = self._get_resident_cache_for_lookup(layer_name)

        access: RetroSpecRankedDraftResidentAccess = (
            resident_cache.lookup_ranked_draft_gpu(
                ranked_values=ranked_values,
                ranked_indices=ranked_indices,
                candidate_counts=candidate_counts,
                arena_cluster_ids=arena.cluster_ids,
                arena_resident_table_buckets=arena.resident_table_buckets,
                arena_cluster_token_counts=arena.cluster_token_counts,
                arena_cluster_page_counts=arena.cluster_page_counts,
                arena_cluster_offsets=arena.cluster_offsets,
                arena_generations=arena.generations,
                request_slot_ids=request_slot_ids,
                active_mask=active_mask,
                retrieval_ratio=retrieval_ratio,
                estimation_ratio=estimation_ratio,
                expanded_retrieval_width=expanded_retrieval_width,
                max_pages_per_cluster=max_pages_per_cluster,
                plan_valid_rows=plan_valid_rows,
                output_request_slot_ids=output_request_slot_ids,
                output_request_slot_generations=(output_request_slot_generations),
                cluster_handles=cluster_handles,
                resident_bucket_ids=resident_bucket_ids,
                clustered_token_counts=clustered_token_counts,
                attention_mass=attention_mass,
                hit_attention_by_head=hit_attention_by_head,
                selected_cluster_counts=selected_cluster_counts,
                hit_cluster_counts=hit_cluster_counts,
                miss_cluster_counts=miss_cluster_counts,
                hit_gate_ready=hit_gate_ready,
                miss_cluster_ids=miss_cluster_ids,
                miss_positions=miss_positions,
                miss_count=miss_count,
                sparse_attention=sparse_attention,
                expanded_attention=expanded_attention,
                capture_request_descriptors=capture_request_descriptors,
                emit_misses=emit_misses,
                statistics_buffer=self._draft_resolve_counter_buffer,
                statistics_indices=self._draft_resolve_counter_indices,
            )
        )
        return RetroSpecRankedDraftResolvedClusters(
            cluster_handles=access.cluster_handles,
            resident_bucket_ids=access.resident_bucket_ids,
            clustered_token_counts=access.clustered_token_counts,
            attention_mass=access.attention_mass,
            selected_cluster_counts=access.selected_cluster_counts,
            hit_cluster_counts=access.hit_cluster_counts,
            miss_cluster_counts=access.miss_cluster_counts,
            hit_gate_ready=access.hit_gate_ready,
            resident_table_page_counts=access.resident_table_page_counts,
            resident_table_page_slots=access.resident_table_page_slots,
            resident_key_pages=access.resident_key_pages,
            resident_value_pages=access.resident_value_pages,
            read_lease=access.read_lease,
        )
