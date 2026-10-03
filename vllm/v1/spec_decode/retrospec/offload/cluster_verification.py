# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from time import perf_counter

import torch

from .cluster_store_support import (
    RetroSpecClusterResolveMode,
    RetroSpecCompactResolvedClusterPages,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationStaging,
    RetroSpecFullVerificationTicket,
    RetroSpecRankedDraftResolvedClusters,
    RetroSpecResolvedClusterPages,
    RetroSpecVerificationMissAdmission,
    RetroSpecVerificationResolveRequest,
    _PinnedVerificationMissSlot,
    _SubmittedVerificationResolve,
    _VerificationResolveGPUArena,
)
from .index_residency import (
    RetroSpecResidentLayerArena,
)
from .page_pool import (
    _LayerClusterPagePool,
)
from .resident_cache import (
    RetroSpecCompactVerificationPageAccess,
    RetroSpecRankedDraftResidentAccess,
)
from .resident_kernels import (
    scatter_compact_staging_page_ids,
)


class RetroSpecClusterVerificationMixin:
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

    def resolve_verification_cluster_blocks(
        self,
        layer_name: str,
        selected_cluster_indices: torch.Tensor,
        plan_valid_rows: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
        arena: RetroSpecResidentLayerArena,
        max_pages_per_cluster: int,
    ) -> RetroSpecCompactVerificationResolvedPages:
        """Resolve one verification layer through the batched implementation."""
        resolved = self.resolve_verification_cluster_batch(
            (
                RetroSpecVerificationResolveRequest(
                    layer_name=layer_name,
                    selected_cluster_indices=selected_cluster_indices,
                    plan_valid_rows=plan_valid_rows,
                    request_slot_ids=request_slot_ids,
                    request_slot_generations=request_slot_generations,
                    arena=arena,
                    max_pages_per_cluster=max_pages_per_cluster,
                ),
            )
        )
        return resolved[layer_name]

    @staticmethod
    def _validate_verification_resolve_request(
        request: RetroSpecVerificationResolveRequest,
    ) -> None:
        selected_cluster_indices = request.selected_cluster_indices
        plan_valid_rows = request.plan_valid_rows
        if selected_cluster_indices.device.type != "cuda":
            raise ValueError("GPU verification lookup requires CUDA")
        if selected_cluster_indices.ndim != 3:
            raise ValueError("Verification cluster indices must be three-dimensional")
        if plan_valid_rows.ndim != 1 or plan_valid_rows.dtype != torch.bool:
            raise ValueError("plan_valid_rows must be one-dimensional and boolean")
        if plan_valid_rows.device != selected_cluster_indices.device:
            raise ValueError("Plan validity must use the lookup device")
        if request.request_slot_ids.shape != request.request_slot_generations.shape:
            raise ValueError("Request slot descriptors must have equal shapes")
        if request.max_pages_per_cluster < 0:
            raise ValueError("max_pages_per_cluster must be non-negative")
        if plan_valid_rows.shape != (selected_cluster_indices.shape[0],):
            raise ValueError("Plan validity must contain one entry per query")

    def _submit_verification_cluster_blocks(
        self,
        request: RetroSpecVerificationResolveRequest,
    ) -> _SubmittedVerificationResolve:
        self._validate_verification_resolve_request(request)
        layer_name = request.layer_name
        selected_cluster_indices = request.selected_cluster_indices
        max_pages_per_cluster = request.max_pages_per_cluster

        with self._resident_state_lock:
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)

        current_stream = torch.cuda.current_stream(selected_cluster_indices.device)
        resident_cache.wait_for_pending_copies(current_stream)
        num_queries = selected_cluster_indices.shape[0]
        num_kv_heads = selected_cluster_indices.shape[1]
        retrieval_width = selected_cluster_indices.shape[2]
        page_width = retrieval_width * max_pages_per_cluster
        row_capacity = num_queries * num_kv_heads
        cluster_capacity = row_capacity * retrieval_width
        page_capacity = row_capacity * page_width

        slot: _PinnedVerificationMissSlot | None = None
        access: RetroSpecCompactVerificationPageAccess | None = None
        try:
            slot, resolve_arena = self._acquire_verification_resolve_workspace(
                selected_cluster_indices.device,
                cluster_capacity,
                max_pages_per_cluster,
                row_capacity,
                page_capacity,
            )
            required = (
                resolve_arena.unique_cluster_ids,
                resolve_arena.unique_logical_page_ids,
                resolve_arena.unique_page_counts,
                resolve_arena.unique_staging_starts,
                resolve_arena.miss_hash_buckets,
                resolve_arena.miss_unique_indices,
                resolve_arena.miss_output_page_offsets,
                resolve_arena.miss_count,
                resolve_arena.unique_miss_count,
                resolve_arena.invalid_descriptor_count,
                resolve_arena.miss_table_handles,
                resolve_arena.miss_table_unique_indices,
                resolve_arena.resident_page_ids,
                resolve_arena.staging_page_ids,
                resolve_arena.page_token_counts,
                resolve_arena.row_page_counts,
                resolve_arena.selected_cluster_counts,
                resolve_arena.hit_cluster_counts,
                resolve_arena.miss_cluster_counts,
            )
            if any(tensor is None for tensor in required):
                raise RuntimeError("Verification compact arena is unavailable")

            assert resolve_arena.unique_cluster_ids is not None
            assert resolve_arena.unique_logical_page_ids is not None
            assert resolve_arena.unique_page_counts is not None
            assert resolve_arena.unique_staging_starts is not None
            assert resolve_arena.miss_hash_buckets is not None
            assert resolve_arena.miss_unique_indices is not None
            assert resolve_arena.miss_output_page_offsets is not None
            assert resolve_arena.miss_count is not None
            assert resolve_arena.unique_miss_count is not None
            assert resolve_arena.invalid_descriptor_count is not None
            assert resolve_arena.miss_table_handles is not None
            assert resolve_arena.miss_table_unique_indices is not None
            assert resolve_arena.resident_page_ids is not None
            assert resolve_arena.staging_page_ids is not None
            assert resolve_arena.page_token_counts is not None
            assert resolve_arena.row_page_counts is not None
            assert resolve_arena.selected_cluster_counts is not None
            assert resolve_arena.hit_cluster_counts is not None
            assert resolve_arena.miss_cluster_counts is not None

            page_shape = (num_queries, num_kv_heads, page_width)
            row_shape = (num_queries, num_kv_heads)
            resident_page_ids = resolve_arena.resident_page_ids[:page_capacity].view(
                page_shape
            )
            staging_page_ids = resolve_arena.staging_page_ids[:page_capacity].view(
                page_shape
            )
            page_token_counts = resolve_arena.page_token_counts[:page_capacity].view(
                page_shape
            )
            page_counts = resolve_arena.row_page_counts[:row_capacity].view(row_shape)
            selected_counts = resolve_arena.selected_cluster_counts[:row_capacity].view(
                row_shape
            )
            hit_counts = resolve_arena.hit_cluster_counts[:row_capacity].view(row_shape)
            miss_counts = resolve_arena.miss_cluster_counts[:row_capacity].view(
                row_shape
            )

            lookup_timer = (
                None
                if self.performance_stats is None
                else self.performance_stats.start_cuda_timer(
                    "verification_gpu_lookup", current_stream
                )
            )
            access = resident_cache.lookup_compact_verification_gpu(
                selected_cluster_indices=selected_cluster_indices,
                plan_valid_rows=request.plan_valid_rows,
                request_slot_ids=request.request_slot_ids,
                request_slot_generations=request.request_slot_generations,
                arena_cluster_ids=request.arena.cluster_ids,
                arena_cluster_page_starts=request.arena.cluster_page_starts,
                arena_cluster_page_counts=request.arena.cluster_page_counts,
                arena_page_ids=request.arena.page_ids,
                arena_page_token_counts=request.arena.page_token_counts,
                arena_cluster_offsets=request.arena.cluster_offsets,
                arena_page_offsets=request.arena.page_offsets,
                arena_generations=request.arena.generations,
                resident_page_ids=resident_page_ids,
                staging_page_ids=staging_page_ids,
                page_token_counts=page_token_counts,
                page_counts=page_counts,
                selected_cluster_counts=selected_counts,
                hit_cluster_counts=hit_counts,
                miss_cluster_counts=miss_counts,
                unique_cluster_ids=resolve_arena.unique_cluster_ids[:cluster_capacity],
                unique_logical_page_ids=resolve_arena.unique_logical_page_ids[
                    :cluster_capacity, :max_pages_per_cluster
                ],
                unique_page_counts=resolve_arena.unique_page_counts[:cluster_capacity],
                miss_hash_buckets=resolve_arena.miss_hash_buckets[:cluster_capacity],
                miss_unique_indices=resolve_arena.miss_unique_indices[
                    :cluster_capacity
                ],
                miss_output_page_offsets=resolve_arena.miss_output_page_offsets[
                    :cluster_capacity
                ],
                miss_count=resolve_arena.miss_count,
                unique_miss_count=resolve_arena.unique_miss_count,
                miss_table_handles=resolve_arena.miss_table_handles,
                miss_table_unique_indices=resolve_arena.miss_table_unique_indices,
                invalid_descriptor_count=resolve_arena.invalid_descriptor_count,
            )
            if self.performance_stats is not None:
                self.performance_stats.stop_cuda_timer(lookup_timer, current_stream)
                self.performance_stats.add_gpu_counter(
                    "verification_lookup_clusters", access.selected_cluster_counts
                )
                self.performance_stats.add_gpu_counter(
                    "verification_resident_hits", access.hit_cluster_counts
                )
                self.performance_stats.add_gpu_counter(
                    "verification_resident_misses", access.miss_cluster_counts
                )

            pinned_required = (
                slot.unique_cluster_id_storage,
                slot.unique_logical_page_id_storage,
                slot.unique_page_count_storage,
                slot.unique_staging_start_storage,
                slot.miss_count_storage,
                slot.unique_miss_count_storage,
                slot.invalid_descriptor_count_storage,
            )
            if any(tensor is None for tensor in pinned_required):
                raise RuntimeError("Verification pinned compact slot is unavailable")
            assert slot.unique_cluster_id_storage is not None
            assert slot.unique_logical_page_id_storage is not None
            assert slot.unique_page_count_storage is not None
            assert slot.unique_staging_start_storage is not None
            assert slot.miss_count_storage is not None
            assert slot.unique_miss_count_storage is not None
            assert slot.invalid_descriptor_count_storage is not None

            lookup_ready_event = torch.cuda.Event()
            lookup_ready_event.record(current_stream)
            return _SubmittedVerificationResolve(
                layer_name=layer_name,
                pool=pool,
                resident_cache=resident_cache,
                slot=slot,
                resolve_arena=resolve_arena,
                access=access,
                cluster_capacity=cluster_capacity,
                max_pages_per_cluster=max_pages_per_cluster,
                lookup_ready_event=lookup_ready_event,
            )
        except BaseException:
            if slot is not None:
                current_stream.synchronize()
                self._release_verification_miss_slot(slot, None)
            if access is not None:
                access.read_lease.release()
            raise

    def _finalize_verification_cluster_blocks(
        self,
        submission: _SubmittedVerificationResolve,
        num_misses: int,
        num_unique_misses: int,
    ) -> RetroSpecCompactVerificationResolvedPages:
        slot = submission.slot
        access = submission.access
        resident_cache = submission.resident_cache
        empty_pages = resident_cache.key_pages[:0]
        if num_misses == 0:
            self._release_verification_miss_slot(slot, None)
            return RetroSpecCompactVerificationResolvedPages(
                resident_page_ids=access.resident_page_ids,
                staging_page_ids=access.staging_page_ids,
                page_token_counts=access.page_token_counts,
                page_counts=access.page_counts,
                resident_key_pages=resident_cache.key_pages,
                resident_value_pages=resident_cache.value_pages,
                staging_key_pages=empty_pages,
                staging_value_pages=resident_cache.value_pages[:0],
                staging_ready_event=None,
                read_lease=access.read_lease,
            )

        descriptor_started_at = (
            perf_counter()
            if self.performance_stats is not None and self.performance_stats.enabled
            else None
        )
        (
            miss_cluster_ids_cpu,
            miss_page_ids_cpu,
            miss_staging_page_ids_cpu,
            unique_page_ids_cpu,
        ) = self._validate_unique_verification_miss_metadata(
            layer_name=submission.layer_name,
            max_pages=submission.max_pages_per_cluster,
            slot=slot,
            num_unique_misses=num_unique_misses,
        )
        if descriptor_started_at is not None:
            self.performance_stats.record_cpu_time(
                "verification_miss_descriptor_build",
                perf_counter() - descriptor_started_at,
            )

        assert submission.resolve_arena.unique_staging_starts is not None
        assert slot.unique_staging_start_storage is not None
        submission.resolve_arena.unique_staging_starts[:num_unique_misses].copy_(
            slot.unique_staging_start_storage[:num_unique_misses],
            non_blocking=self.pin_memory,
        )
        scatter_compact_staging_page_ids(
            miss_unique_indices=access.miss_unique_indices,
            miss_output_page_offsets=access.miss_output_page_offsets,
            unique_staging_starts=submission.resolve_arena.unique_staging_starts,
            unique_page_counts=access.unique_page_counts,
            num_misses=num_misses,
            num_unique_misses=num_unique_misses,
            max_pages=submission.max_pages_per_cluster,
            output_page_ids=access.staging_page_ids,
        )
        current_stream = torch.cuda.current_stream(access.resident_page_ids.device)
        slot_ready_event = torch.cuda.Event()
        slot_ready_event.record(current_stream)
        self._release_verification_miss_slot(slot, slot_ready_event)

        staging_key_pages, staging_value_pages, staging_ready_event = (
            self._stage_verification_miss_pages(submission.pool, unique_page_ids_cpu)
        )
        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "verification_unique_miss_clusters", num_unique_misses
            )
            self.performance_stats.add_counter(
                "verification_duplicate_miss_clusters",
                num_misses - num_unique_misses,
            )

        admission = RetroSpecVerificationMissAdmission(
            layer_name=submission.layer_name,
            cluster_ids_cpu=miss_cluster_ids_cpu,
            logical_page_ids_cpu=miss_page_ids_cpu,
            staging_page_ids_cpu=miss_staging_page_ids_cpu,
            staging_key_pages=staging_key_pages,
            staging_value_pages=staging_value_pages,
            staging_ready_event=staging_ready_event,
        )
        return RetroSpecCompactVerificationResolvedPages(
            resident_page_ids=access.resident_page_ids,
            staging_page_ids=access.staging_page_ids,
            page_token_counts=access.page_token_counts,
            page_counts=access.page_counts,
            resident_key_pages=resident_cache.key_pages,
            resident_value_pages=resident_cache.value_pages,
            staging_key_pages=staging_key_pages,
            staging_value_pages=staging_value_pages,
            staging_ready_event=staging_ready_event,
            read_lease=access.read_lease,
            miss_admission=admission,
        )

    def resolve_verification_cluster_batch(
        self,
        requests: Sequence[RetroSpecVerificationResolveRequest],
    ) -> dict[str, RetroSpecCompactVerificationResolvedPages]:
        """Resolve all verification layers with two cross-layer CPU waits."""
        requests = tuple(requests)
        if not requests:
            return {}
        devices = {request.selected_cluster_indices.device for request in requests}
        if len(devices) != 1:
            raise ValueError("Verification batch must use one CUDA device")
        layer_names = tuple(request.layer_name for request in requests)
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("Verification batch layer names must be unique")

        prepare_started_at = (
            perf_counter()
            if self.performance_stats is not None and self.performance_stats.enabled
            else None
        )
        self._reap_verification_admissions(wait=True)
        self.wait_for_resident_prefetches(layer_names)
        submissions: list[_SubmittedVerificationResolve] = []
        try:
            for request in requests:
                submissions.append(self._submit_verification_cluster_blocks(request))
            device = requests[0].selected_cluster_indices.device
            metadata_stream = self._get_verification_metadata_stream(device)
            with torch.cuda.stream(metadata_stream):
                for submission in submissions:
                    metadata_stream.wait_event(submission.lookup_ready_event)
                    slot = submission.slot
                    access = submission.access
                    assert slot.miss_count_storage is not None
                    assert slot.unique_miss_count_storage is not None
                    assert slot.invalid_descriptor_count_storage is not None
                    slot.miss_count_storage.copy_(
                        access.miss_count, non_blocking=self.pin_memory
                    )
                    slot.unique_miss_count_storage.copy_(
                        access.unique_miss_count, non_blocking=self.pin_memory
                    )
                    slot.invalid_descriptor_count_storage.copy_(
                        access.invalid_descriptor_count,
                        non_blocking=self.pin_memory,
                    )
                counts_ready_event = torch.cuda.Event()
                counts_ready_event.record(metadata_stream)

            count_wait_started_at = perf_counter()
            counts_ready_event.synchronize()
            if self.performance_stats is not None and self.performance_stats.enabled:
                self.performance_stats.record_cpu_time(
                    "verification_batch_count_wait",
                    perf_counter() - count_wait_started_at,
                )

            counts: dict[str, tuple[int, int]] = {}
            miss_submissions: list[_SubmittedVerificationResolve] = []
            metadata_bytes = 0
            for submission in submissions:
                slot = submission.slot
                assert slot.miss_count_storage is not None
                assert slot.unique_miss_count_storage is not None
                assert slot.invalid_descriptor_count_storage is not None
                invalid_descriptors = int(slot.invalid_descriptor_count_storage.item())
                if invalid_descriptors:
                    raise RuntimeError(
                        "Verification selected an invalid descriptor or miss mapping"
                    )
                num_misses = int(slot.miss_count_storage.item())
                num_unique_misses = int(slot.unique_miss_count_storage.item())
                if (
                    not 0
                    <= num_unique_misses
                    <= num_misses
                    <= submission.cluster_capacity
                ):
                    raise RuntimeError("GPU verification miss count is out of bounds")
                counts[submission.layer_name] = (num_misses, num_unique_misses)
                metadata_bytes += (
                    slot.miss_count_storage.element_size()
                    + slot.unique_miss_count_storage.element_size()
                    + slot.invalid_descriptor_count_storage.element_size()
                )
                if num_misses:
                    miss_submissions.append(submission)

            if miss_submissions:
                with torch.cuda.stream(metadata_stream):
                    for submission in miss_submissions:
                        _, num_unique_misses = counts[submission.layer_name]
                        slot = submission.slot
                        access = submission.access
                        max_pages = submission.max_pages_per_cluster
                        assert slot.unique_cluster_id_storage is not None
                        assert slot.unique_logical_page_id_storage is not None
                        assert slot.unique_page_count_storage is not None
                        slot.unique_cluster_id_storage[:num_unique_misses].copy_(
                            access.unique_cluster_ids[:num_unique_misses],
                            non_blocking=self.pin_memory,
                        )
                        slot.unique_logical_page_id_storage[
                            :num_unique_misses, :max_pages
                        ].copy_(
                            access.unique_logical_page_ids[
                                :num_unique_misses, :max_pages
                            ],
                            non_blocking=self.pin_memory,
                        )
                        slot.unique_page_count_storage[:num_unique_misses].copy_(
                            access.unique_page_counts[:num_unique_misses],
                            non_blocking=self.pin_memory,
                        )
                        metadata_bytes += num_unique_misses * (
                            slot.unique_cluster_id_storage.element_size()
                            + slot.unique_page_count_storage.element_size()
                            + max_pages
                            * slot.unique_logical_page_id_storage.element_size()
                        )
                    metadata_ready_event = torch.cuda.Event()
                    metadata_ready_event.record(metadata_stream)
                metadata_wait_started_at = perf_counter()
                metadata_ready_event.synchronize()
                if (
                    self.performance_stats is not None
                    and self.performance_stats.enabled
                ):
                    self.performance_stats.record_cpu_time(
                        "verification_batch_metadata_wait",
                        perf_counter() - metadata_wait_started_at,
                    )

            resolved = {
                submission.layer_name: self._finalize_verification_cluster_blocks(
                    submission, *counts[submission.layer_name]
                )
                for submission in submissions
            }
            if self.performance_stats is not None:
                self.performance_stats.add_counter(
                    "verification_miss_metadata_d2h_bytes", metadata_bytes
                )
                self.performance_stats.add_counter(
                    "verification_prepared_layers", len(submissions)
                )
                self.performance_stats.add_counter(
                    "verification_miss_layers", len(miss_submissions)
                )
            if prepare_started_at is not None:
                self.performance_stats.record_cpu_time(
                    "verification_batch_prepare_wall",
                    perf_counter() - prepare_started_at,
                )
            return resolved
        except BaseException:
            for submission in submissions:
                if submission.slot.in_use:
                    torch.cuda.current_stream(
                        submission.access.resident_page_ids.device
                    ).synchronize()
                    self._release_verification_miss_slot(submission.slot, None)
                submission.access.read_lease.release()
            raise

    def resolve_cluster_blocks(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        mode: RetroSpecClusterResolveMode = "verification",
    ) -> RetroSpecResolvedClusterPages:
        if mode == "verification":
            self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
            return self._resolve_cluster_blocks_locked(
                layer_name,
                cluster_ids,
                logical_page_ids,
                mode,
            )

    def _resolve_cluster_blocks_locked(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        mode: RetroSpecClusterResolveMode = "verification",
    ) -> RetroSpecResolvedClusterPages:
        """Resolve selected cluster blocks to physical GPU page sources.

        resident_only exposes only completed resident pages. resident_pending
        exposes pending resident pages and their ready event without staging
        other misses. verification additionally stages every selected miss.
        """
        if mode not in ("resident_only", "resident_pending", "verification"):
            raise ValueError(f"Unsupported RetroSpec cluster resolve mode: {mode}")
        if logical_page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Logical page-table shape does not match cluster IDs")
        if logical_page_ids.device != cluster_ids.device:
            raise ValueError("Cluster IDs and logical pages must use one device")

        cluster_ids_cpu, logical_page_ids_cpu = self._validate_cluster_blocks(
            layer_name,
            cluster_ids,
            logical_page_ids,
        )

        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        allocated_cluster_ids = self._get_allocated_cluster_ids(layer_name)

        cluster_groups = self._get_cluster_groups(
            layer_name,
            cluster_ids_cpu,
        )
        pool, resident_cache = self._get_or_create_resident_cache(layer_name)

        verification = mode == "verification"
        include_pending = mode != "resident_only"
        resident_access = resident_cache.lookup(
            cluster_ids=cluster_ids,
            page_ids=logical_page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=pool.allocated_page_ids,
            touch=True,
            include_pending=include_pending,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=logical_page_ids_cpu,
        )

        if self.performance_stats is not None and not verification:
            valid_clusters_cpu = cluster_ids_cpu >= 0
            miss_clusters_cpu = (
                valid_clusters_cpu & resident_access.miss_cluster_mask_cpu
            )
            num_misses = int(miss_clusters_cpu.sum().item())
            num_valid = int(valid_clusters_cpu.sum().item())
            self.performance_stats.add_counter(
                "resident_cluster_hits",
                num_valid - num_misses,
            )
            self.performance_stats.add_counter(
                "resident_cluster_misses",
                num_misses,
            )

        if verification:
            (
                staging_page_ids,
                staging_key_pages,
                staging_value_pages,
                staging_ready_event,
            ) = self._stage_missing_pages(
                pool,
                logical_page_ids,
                resident_access.miss_cluster_mask,
                resident_access.logical_page_ids_cpu,
                resident_access.miss_cluster_mask_cpu,
            )
            resident_ready_event = resident_access.ready_event
        else:
            staging_page_ids = torch.full_like(logical_page_ids, -1)
            staging_key_pages = resident_cache.key_pages[:0]
            staging_value_pages = resident_cache.value_pages[:0]
            resident_ready_event = (
                resident_access.ready_event if include_pending else None
            )
            staging_ready_event = None

        return RetroSpecResolvedClusterPages(
            resident_page_ids=resident_access.cache_page_ids,
            staging_page_ids=staging_page_ids,
            resident_key_pages=resident_cache.key_pages,
            resident_value_pages=resident_cache.value_pages,
            staging_key_pages=staging_key_pages,
            staging_value_pages=staging_value_pages,
            hit_cluster_mask=resident_access.hit_cluster_mask,
            miss_cluster_mask=resident_access.miss_cluster_mask,
            hit_gate_ready_mask=resident_access.hit_gate_ready_mask,
            resident_ready_event=resident_ready_event,
            staging_ready_event=staging_ready_event,
        )

    def build_full_verification_descriptor(
        self,
        layer_name: str,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
    ) -> RetroSpecFullVerificationDescriptor:
        """Build a persistent valid-token descriptor for one request segment."""
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )
        return pool.build_full_verification_descriptor(page_ids, page_token_counts)

    def resolve_full_verification_tokens(
        self,
        layer_name: str,
        descriptors: Sequence[RetroSpecFullVerificationDescriptor],
    ) -> RetroSpecFullVerificationStaging:
        """Synchronously resolve a full-verification staging request."""
        return self.submit_full_verification_tokens(
            layer_name=layer_name,
            descriptors=descriptors,
        ).result()

    def submit_full_verification_tokens(
        self,
        layer_name: str,
        descriptors: Sequence[RetroSpecFullVerificationDescriptor],
    ) -> RetroSpecFullVerificationTicket:
        """Submit native CPU gather and H2D staging without blocking."""
        self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
            pool = self._layer_pools.get(layer_name)
            if pool is None:
                raise RuntimeError(
                    f"No RetroSpec page pool exists for layer {layer_name!r}"
                )

            transfer_buffer = self._get_full_verification_buffer(pool)
            source = pool.snapshot_full_verification_sources()
            return transfer_buffer.submit(source=source, descriptors=descriptors)
