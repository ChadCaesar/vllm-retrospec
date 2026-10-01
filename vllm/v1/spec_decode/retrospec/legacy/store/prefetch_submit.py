# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from time import perf_counter

import torch

from vllm import _custom_ops as ops
from vllm.v1.spec_decode.retrospec.legacy.resident_cache import (
    RetroSpecResidentClusterCache,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_kernels import (
    compact_resident_misses,
)
from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    RetroSpecResidentPrefetchInput,
    RetroSpecResidentPrefetchSource,
    _DeferredResidentPrefetchWave,
    _LayerClusterPagePool,
    _LayerPrefetchDescriptorArena,
    _PreparedResidentPrefetchRecord,
    _ResidentPrefetchWaveFuture,
    _StagedResidentPrefetchRecord,
    _StagedResidentPrefetchWave,
)


class _RetroSpecClusterPageStorePrefetchSubmitMixin:
    def _defer_resident_prefetch_wave(
        self,
        device: torch.device,
        records: tuple[RetroSpecResidentPrefetchInput, ...],
    ) -> None:
        source_ready_event = torch.cuda.Event()
        source_ready_event.record(torch.cuda.current_stream(device))
        superseded = 0
        with self._resident_prefetch_lock:
            previous = self._resident_prefetch_deferred.get(device)
            merged: dict[
                str, tuple[RetroSpecResidentPrefetchInput, torch.cuda.Event]
            ] = {}
            if previous is not None:
                merged.update(
                    (record.layer_name, (record, ready_event))
                    for record, ready_event in zip(
                        previous.records, previous.source_ready_events
                    )
                )
            for record in records:
                if record.layer_name in merged:
                    superseded += 1
                merged[record.layer_name] = (record, source_ready_event)
            if len(merged) > self._resident_prefetch_wave_max_records:
                raise RuntimeError("Deferred resident prefetch exceeds layer capacity")
            deferred = _DeferredResidentPrefetchWave(
                records=tuple(record for record, _ in merged.values()),
                source_ready_events=tuple(event for _, event in merged.values()),
            )
            self._resident_prefetch_deferred[device] = deferred

        if self.performance_stats is not None:
            self.performance_stats.add_counter("prefetch_waves_deferred")
            self.performance_stats.observe_peak(
                "prefetch_deferred_layer_records", len(deferred.records)
            )
            if previous is not None:
                self.performance_stats.add_counter("prefetch_waves_coalesced")
            if superseded:
                self.performance_stats.add_counter(
                    "prefetch_records_superseded", superseded
                )

    def _wait_for_one_resident_prefetch(self, device: torch.device) -> bool:
        selected: _ResidentPrefetchWaveFuture | None = None
        with self._resident_prefetch_lock:
            retained: deque[_ResidentPrefetchWaveFuture] = deque()
            while self._resident_prefetch_futures:
                wave = self._resident_prefetch_futures.popleft()
                if selected is None and wave.device == device:
                    selected = wave
                else:
                    retained.append(wave)
            self._resident_prefetch_futures = retained

        if selected is None:
            return False

        wait_started_at = (
            perf_counter()
            if self.performance_stats is not None and self.performance_stats.enabled
            else None
        )
        try:
            selected.future.result()
        finally:
            if self.performance_stats is not None:
                self.performance_stats.add_counter("prefetch_backpressure_waits")
                if wait_started_at is not None:
                    self.performance_stats.record_cpu_time(
                        "prefetch_backpressure_wait_wall",
                        perf_counter() - wait_started_at,
                    )
        return True

    def _try_submit_deferred_resident_prefetch(
        self,
        device: torch.device,
        wait_for_slot: bool,
    ) -> bool:
        device = self._canonical_cuda_device(device)
        while True:
            with self._resident_prefetch_lock:
                deferred = self._resident_prefetch_deferred.get(device)
            if deferred is None:
                return False

            slot = self._acquire_resident_prefetch_slot(device)
            if slot is None:
                if not wait_for_slot:
                    return False
                if not self._wait_for_one_resident_prefetch(device):
                    raise RuntimeError(
                        "Resident prefetch ring is occupied without an in-flight "
                        "same-device worker"
                    )
                continue

            with self._resident_prefetch_lock:
                current = self._resident_prefetch_deferred.get(device)
                if current is deferred:
                    self._resident_prefetch_deferred.pop(device)
                else:
                    current = None

            if current is None:
                self._release_resident_prefetch_slot(slot)
                continue

            try:
                self._submit_resident_prefetch_wave(
                    current.records, slot, current.source_ready_events
                )
            except BaseException:
                self._discard_resident_prefetch_records(current.records)
                raise
            return True

    def _auto_submit_deferred_resident_prefetch(
        self,
        device: torch.device,
    ) -> None:
        """Use a newly released command slot without blocking the worker."""
        submitted = self._try_submit_deferred_resident_prefetch(
            device, wait_for_slot=False
        )
        if submitted and self.performance_stats is not None:
            self.performance_stats.add_counter("prefetch_worker_auto_submits")

    def flush_resident_prefetch_commands(self) -> None:
        """Submit deferred commands and seal their GPU workspace lifetime."""
        while True:
            with self._resident_prefetch_lock:
                devices = tuple(self._resident_prefetch_deferred)
            if not devices:
                break
            for device in devices:
                self._try_submit_deferred_resident_prefetch(device, wait_for_slot=True)

        with self._resident_prefetch_lock:
            metadata_events = tuple(
                self._resident_prefetch_last_metadata_events.items()
            )
            self._resident_prefetch_last_metadata_events.clear()

        for device, metadata_event in metadata_events:
            with torch.cuda.device(device):
                torch.cuda.current_stream(device).wait_event(metadata_event)

    @torch.inference_mode()
    def _prepare_resident_prefetch_wave(
        self,
        records: tuple[_StagedResidentPrefetchRecord, ...],
    ) -> tuple[_PreparedResidentPrefetchRecord | None, ...]:
        pools: list[_LayerClusterPagePool] = []
        resident_caches: list[RetroSpecResidentClusterCache] = []
        descriptor_arenas: list[_LayerPrefetchDescriptorArena] = []
        page_capacities: list[int] = []
        with self._resident_state_lock:
            for record in records:
                pool, resident_cache = self._get_or_create_resident_cache(
                    record.layer_name
                )
                descriptor_arena = self._prefetch_descriptor_arenas.get(
                    record.layer_name
                )
                if descriptor_arena is None:
                    raise RuntimeError(
                        f"Missing prefetch descriptors for {record.layer_name!r}"
                    )
                resident_cache.reserve_prefetch_handle_states(descriptor_arena.capacity)
                transfer_buffer = self._get_full_verification_buffer(pool)
                pools.append(pool)
                resident_caches.append(resident_cache)
                descriptor_arenas.append(descriptor_arena)
                page_capacities.append(transfer_buffer.cpu_slot_capacity(pool))
            plan_started_at = (
                perf_counter()
                if self.performance_stats is not None and self.performance_stats.enabled
                else None
            )
            (
                cluster_ids,
                page_ids,
                source_page_ids,
                unique_page_ids,
                group_ids,
                statistics,
            ) = ops.retrospec_plan_prefetch_admissions(
                tuple(record.miss_cluster_ids_cpu for record in records),
                tuple(record.miss_positions_cpu for record in records),
                tuple(record.miss_count_cpu for record in records),
                tuple(record.num_groups for record in records),
                tuple(record.num_ranks for record in records),
                tuple(arena.page_ids for arena in descriptor_arenas),
                tuple(arena.page_counts for arena in descriptor_arenas),
                tuple(arena.group_ids for arena in descriptor_arenas),
                tuple(
                    cache.prefetch_handle_states(arena.capacity)
                    for cache, arena in zip(
                        resident_caches, descriptor_arenas, strict=True
                    )
                ),
                tuple(page_capacities),
            )

        if plan_started_at is not None:
            assert self.performance_stats is not None
            self.performance_stats.record_cpu_time(
                "prefetch_native_plan_wall", perf_counter() - plan_started_at
            )
        if self.performance_stats is not None:
            totals = statistics.sum(dim=0).tolist()
            counter_names = (
                "prefetch_raw_commands",
                "prefetch_miss_commands",
                "prefetch_stale_commands",
                "prefetch_skipped_pending_clusters",
                "prefetch_skipped_resident_clusters",
                "prefetch_planned_clusters",
                "prefetch_planned_pages",
                "prefetch_budget_stops",
            )
            for name, value in zip(counter_names, totals, strict=True):
                if value:
                    self.performance_stats.add_counter(name, value)
            duplicate_count = totals[0] - totals[1]
            if duplicate_count:
                self.performance_stats.add_counter(
                    "prefetch_duplicate_misses", duplicate_count
                )

        prepared: list[_PreparedResidentPrefetchRecord | None] = []
        for (
            record,
            pool,
            resident_cache,
            descriptor_arena,
            record_cluster_ids,
            record_page_ids,
            record_source_page_ids,
            record_unique_page_ids,
            record_group_ids,
        ) in zip(
            records,
            pools,
            resident_caches,
            descriptor_arenas,
            cluster_ids,
            page_ids,
            source_page_ids,
            unique_page_ids,
            group_ids,
            strict=True,
        ):
            if record_cluster_ids.numel() == 0:
                prepared.append(None)
                continue
            prepared.append(
                _PreparedResidentPrefetchRecord(
                    layer_name=record.layer_name,
                    pool=pool,
                    resident_cache=resident_cache,
                    cluster_ids_cpu=record_cluster_ids,
                    page_ids_cpu=record_page_ids,
                    source_page_ids_cpu=record_source_page_ids,
                    unique_page_ids_cpu=record_unique_page_ids,
                    cluster_groups=descriptor_arena.resolve_groups(
                        record_cluster_ids, record_group_ids
                    ),
                )
            )
        return tuple(prepared)

    @torch.inference_mode()
    def _process_prepared_resident_prefetch(
        self,
        prepared: _PreparedResidentPrefetchRecord,
        execution_stream: torch.cuda.Stream,
    ) -> None:
        stats = self.performance_stats
        gather_started_at = (
            perf_counter() if stats is not None and stats.enabled else None
        )
        try:
            (
                source_page_ids,
                source_key_pages,
                source_value_pages,
                transfer_buffer,
                transfer_slot,
            ) = self._stage_resident_pages(
                prepared.pool,
                prepared.source_page_ids_cpu,
                prepared.unique_page_ids_cpu,
            )
        finally:
            if gather_started_at is not None:
                stats.record_cpu_time(
                    "prefetch_page_gather_wall",
                    perf_counter() - gather_started_at,
                )

        admission_started_at = (
            perf_counter() if stats is not None and stats.enabled else None
        )
        try:
            prepare_started_at = (
                perf_counter() if stats is not None and stats.enabled else None
            )
            try:
                admission = prepared.resident_cache.prepare_staged_admission(
                    cluster_ids=prepared.cluster_ids_cpu,
                    page_ids=prepared.page_ids_cpu,
                    cluster_groups=prepared.cluster_groups,
                    staging_page_ids=source_page_ids,
                    staging_key_pages=source_key_pages,
                    staging_value_pages=source_value_pages,
                    cluster_ids_cpu=prepared.cluster_ids_cpu,
                    page_ids_cpu=prepared.page_ids_cpu,
                )
            finally:
                if prepare_started_at is not None:
                    stats.record_cpu_time(
                        "prefetch_resident_prepare_wall",
                        perf_counter() - prepare_started_at,
                    )

            capture_started_at = (
                perf_counter() if stats is not None and stats.enabled else None
            )
            try:
                with (
                    self._resident_state_lock,
                    prepared.resident_cache.mutation_guard(),
                    torch.cuda.device(prepared.pool.metadata_device),
                ):
                    lru_capture = (
                        prepared.resident_cache.capture_prepared_admission_lru(
                            prepared=admission,
                            allocated_cluster_ids=self._get_allocated_cluster_ids(
                                prepared.layer_name
                            ),
                            allocated_page_ids=prepared.pool.allocated_page_ids,
                            stream=execution_stream,
                        )
                    )
            finally:
                if capture_started_at is not None:
                    stats.record_cpu_time(
                        "prefetch_resident_lru_capture_wall",
                        perf_counter() - capture_started_at,
                    )

            resolve_started_at = (
                perf_counter() if stats is not None and stats.enabled else None
            )
            try:
                resolved_lru = (
                    None
                    if lru_capture is None
                    else prepared.resident_cache.resolve_lru_capture(lru_capture)
                )
            finally:
                if resolve_started_at is not None:
                    stats.record_cpu_time(
                        "prefetch_resident_lru_resolve_wall",
                        perf_counter() - resolve_started_at,
                    )

            commit_started_at = (
                perf_counter() if stats is not None and stats.enabled else None
            )
            try:
                with (
                    self._resident_state_lock,
                    prepared.resident_cache.mutation_guard(),
                    torch.cuda.device(prepared.pool.metadata_device),
                ):
                    access = prepared.resident_cache.admit_prepared_staged(
                        prepared=admission,
                        allocated_cluster_ids=self._get_allocated_cluster_ids(
                            prepared.layer_name
                        ),
                        allocated_page_ids=prepared.pool.allocated_page_ids,
                        mutation_stream=execution_stream,
                        lookup_after_admit=False,
                        resolved_lru=resolved_lru,
                    )
            finally:
                if commit_started_at is not None:
                    stats.record_cpu_time(
                        "prefetch_resident_commit_wall",
                        perf_counter() - commit_started_at,
                    )
        except BaseException:
            prepared.resident_cache.synchronize_pending_copies()
            if transfer_slot is not None:
                transfer_buffer.release_cpu_slot(transfer_slot, None)
            raise
        finally:
            if admission_started_at is not None:
                stats.record_cpu_time(
                    "prefetch_resident_admission_wall",
                    perf_counter() - admission_started_at,
                )
        if transfer_slot is not None:
            transfer_buffer.release_cpu_slot(transfer_slot, access.ready_event)

    @torch.inference_mode()
    def _finish_resident_prefetch_wave(
        self,
        staged: _StagedResidentPrefetchWave,
    ) -> None:
        stats = self.performance_stats
        background_started_at = (
            perf_counter() if stats is not None and stats.enabled else None
        )
        slot_released = False
        completed = False
        active_records: list[_StagedResidentPrefetchRecord] = []
        try:
            metadata_wait_started_at = (
                perf_counter() if stats is not None and stats.enabled else None
            )
            staged.metadata_ready_event.synchronize()
            if metadata_wait_started_at is not None:
                stats.record_cpu_time(
                    "prefetch_metadata_wait",
                    perf_counter() - metadata_wait_started_at,
                )

            for record in staged.records:
                if self._resident_prefetch_record_is_current(record):
                    active_records.append(record)
                else:
                    staged.progress.complete_layer(record.layer_name)
                    if stats is not None:
                        stats.add_counter("prefetch_superseded_records")
            prepared_records = (
                self._prepare_resident_prefetch_wave(tuple(active_records))
                if active_records
                else ()
            )
            # The prepared records still reference the pinned command slot.
            # Parse them before releasing the slot so a deferred wave cannot
            # overwrite the metadata while the CPU planner is reading it.
            self._release_resident_prefetch_slot(staged.slot)
            slot_released = True
            self._auto_submit_deferred_resident_prefetch(staged.device)

            for staged_record, prepared in zip(
                active_records, prepared_records, strict=True
            ):
                if prepared is not None:
                    self._process_prepared_resident_prefetch(
                        prepared, staged.execution_stream
                    )
                self._complete_resident_prefetch_record(staged_record)
                staged.progress.complete_layer(staged_record.layer_name)
            completed = True
        except BaseException as error:
            staged.progress.fail(error)
            raise
        finally:
            for record in active_records:
                self._complete_resident_prefetch_record(record)
            if stats is not None:
                stats.add_counter(
                    "prefetch_worker_completed"
                    if completed
                    else "prefetch_worker_failed"
                )
                if background_started_at is not None:
                    stats.record_cpu_time(
                        "prefetch_worker_wall",
                        perf_counter() - background_started_at,
                    )
            if not slot_released:
                self._release_resident_prefetch_slot(staged.slot)

    def _reap_resident_prefetches(
        self,
        layer_names: Sequence[str] | None = None,
        wait: bool = False,
    ) -> None:
        requested_layers = None if layer_names is None else frozenset(layer_names)
        completed: list[_ResidentPrefetchWaveFuture] = []
        layer_waits: list[tuple[_ResidentPrefetchWaveFuture, tuple[str, ...]]] = []
        with self._resident_prefetch_lock:
            retained: deque[_ResidentPrefetchWaveFuture] = deque()
            while self._resident_prefetch_futures:
                wave = self._resident_prefetch_futures.popleft()
                matches = requested_layers is None or not wave.layer_names.isdisjoint(
                    requested_layers
                )
                if not matches:
                    retained.append(wave)
                    continue
                if wave.future.done() or (wait and requested_layers is None):
                    completed.append(wave)
                    continue
                if wait:
                    matched_layers = tuple(
                        layer_name
                        for layer_name in requested_layers
                        if layer_name in wave.layer_names
                    )
                    layer_waits.append((wave, matched_layers))
                retained.append(wave)
            self._resident_prefetch_futures = retained

        layer_wait_count = sum(len(waited_layers) for _, waited_layers in layer_waits)
        if self.performance_stats is not None:
            self.performance_stats.add_counter("prefetch_reaped_tasks", len(completed))
            self.performance_stats.add_counter("prefetch_reaped_waves", len(completed))
            if wait:
                waited_waves = len(completed) + len(layer_waits)
                self.performance_stats.add_counter(
                    "prefetch_waited_tasks", waited_waves
                )
                self.performance_stats.add_counter(
                    "prefetch_waited_waves", waited_waves
                )
                self.performance_stats.add_counter(
                    "prefetch_layer_waits", layer_wait_count
                )

        wait_started_at = (
            perf_counter()
            if wait
            and (completed or layer_waits)
            and self.performance_stats is not None
            and self.performance_stats.enabled
            else None
        )
        try:
            for wave in completed:
                wave.future.result()
            for wave, waited_layers in layer_waits:
                wave.progress.wait_for(waited_layers)
        finally:
            if wait_started_at is not None:
                self.performance_stats.record_cpu_time(
                    "prefetch_wait_wall",
                    perf_counter() - wait_started_at,
                )

    def prefetch_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        access_kinds: torch.Tensor,
        source: RetroSpecResidentPrefetchSource = "draft",
    ) -> bool:
        """Compatibility wrapper for a one-layer resident-prefetch wave."""
        if self._closed:
            raise RuntimeError("RetroSpec cluster page store is closed")
        if cluster_ids.numel() == 0:
            return False
        if cluster_ids.ndim not in (1, 2, 3) or access_kinds.shape != cluster_ids.shape:
            raise ValueError("Resident prefetch inputs must have equal 1D--3D shapes")

        if cluster_ids.ndim == 1:
            compact_cluster_ids = cluster_ids.view(1, 1, -1)
            compact_access_kinds = access_kinds.view(1, 1, -1)
        elif cluster_ids.ndim == 2:
            compact_cluster_ids = cluster_ids.unsqueeze(0)
            compact_access_kinds = access_kinds.unsqueeze(0)
        else:
            compact_cluster_ids = cluster_ids
            compact_access_kinds = access_kinds

        miss_cluster_ids = torch.empty(
            cluster_ids.numel(), dtype=torch.int64, device=cluster_ids.device
        )
        miss_positions = torch.empty_like(miss_cluster_ids)
        miss_count = torch.empty(1, dtype=torch.int32, device=cluster_ids.device)
        compact_resident_misses(
            cluster_handles=compact_cluster_ids,
            miss_mask=compact_access_kinds == 2,
            output_handles=miss_cluster_ids,
            output_positions=miss_positions,
            output_count=miss_count,
        )
        return self.prefetch_resident_cluster_wave(
            (
                RetroSpecResidentPrefetchInput(
                    layer_name=layer_name,
                    miss_cluster_ids=miss_cluster_ids,
                    miss_positions=miss_positions,
                    miss_count=miss_count,
                    num_groups=(
                        compact_cluster_ids.shape[0] * compact_cluster_ids.shape[1]
                    ),
                    num_ranks=compact_cluster_ids.shape[2],
                    source=source,
                ),
            )
        )

    def prefetch_resident_cluster_wave(
        self,
        records: Sequence[RetroSpecResidentPrefetchInput],
    ) -> bool:
        """Queue or coalesce one draft step's cross-layer miss commands."""
        if self._closed:
            raise RuntimeError("RetroSpec cluster page store is closed")
        if self._resident_admission_frozen:
            return False
        records = tuple(records)
        if not records:
            return False
        if not self.pin_memory:
            return False
        device = self._validate_resident_prefetch_wave(records)
        records = self._stamp_resident_prefetch_records(records)
        if not records:
            return False
        try:
            layer_names = tuple(record.layer_name for record in records)
            self._reap_resident_prefetches(layer_names, wait=False)
            self._try_submit_deferred_resident_prefetch(device, wait_for_slot=False)
            slot = self._acquire_resident_prefetch_slot(device)
            if slot is None:
                self._defer_resident_prefetch_wave(device, records)
                return True

            self._submit_resident_prefetch_wave(records, slot, source_ready_events=None)
            return True
        except BaseException:
            self._discard_resident_prefetch_records(records)
            raise

    def wait_for_resident_prefetches(
        self,
        layer_names: Sequence[str] | None = None,
    ) -> None:
        self.flush_resident_prefetch_commands()
        self._reap_resident_prefetches(layer_names, wait=True)

    def synchronize_resident_prefetches(
        self,
        layer_names: Sequence[str],
    ) -> None:
        """Wait for background admission and resident H2D copies to finish."""
        layer_names = tuple(dict.fromkeys(layer_names))
        self.wait_for_resident_prefetches(layer_names)

        with self._resident_state_lock:
            resident_caches = tuple(
                resident_cache
                for layer_name in layer_names
                if (resident_cache := self._resident_caches.get(layer_name)) is not None
            )

        for resident_cache in resident_caches:
            resident_cache.synchronize_pending_copies()

    def begin_resident_replay(self, layer_names: Sequence[str]) -> None:
        """Drain resident mutations and freeze the physical source mapping."""
        if self._resident_admission_frozen:
            raise RuntimeError("Resident replay is already active")

        self.synchronize_resident_prefetches(layer_names)
        self._resident_admission_frozen = True

    def end_resident_replay(self) -> None:
        if not self._resident_admission_frozen:
            raise RuntimeError("Resident replay is not active")
        self._resident_admission_frozen = False
