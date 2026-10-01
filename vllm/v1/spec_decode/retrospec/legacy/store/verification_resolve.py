# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from time import perf_counter

import torch

from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecResidentLayerArena,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_cache import (
    RetroSpecCompactVerificationPageAccess,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_kernels import (
    scatter_compact_staging_page_ids,
)
from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    RetroSpecClusterResolveMode,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecFullVerificationDescriptor,
    RetroSpecResolvedClusterPages,
    RetroSpecVerificationMissAdmission,
    RetroSpecVerificationResolveRequest,
    _PinnedVerificationMissSlot,
    _SubmittedVerificationResolve,
)


class _RetroSpecClusterPageStoreVerificationResolveMixin:
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
