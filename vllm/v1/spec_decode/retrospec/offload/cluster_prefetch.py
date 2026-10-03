# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter

import torch

from vllm import _custom_ops as ops

from .cluster_store_support import (
    _PREFETCH_SOURCE_PRIORITY,
    RetroSpecResidentPrefetchInput,
    RetroSpecResidentPrefetchSource,
    _DeferredResidentPrefetchWave,
    _LayerPrefetchDescriptorArena,
    _PinnedPageTransferSlot,
    _PinnedSelectionSlot,
    _PreparedResidentPrefetchRecord,
    _ResidentPrefetchWaveFuture,
    _ResidentPrefetchWaveProgress,
    _StagedResidentPrefetchRecord,
    _StagedResidentPrefetchWave,
)
from .page_pool import (
    _LayerClusterPagePool,
)
from .resident_cache import (
    RetroSpecResidentClusterCache,
)
from .resident_kernels import (
    compact_resident_misses,
)
from .verification_transfer import (
    _FullVerificationTransferBuffer,
)


class RetroSpecClusterPrefetchMixin:
    def _get_full_verification_buffer(
        self,
        pool: _LayerClusterPagePool,
    ) -> _FullVerificationTransferBuffer:
        device = pool.metadata_device

        if device.type != "cuda":
            raise RuntimeError("Full-verification transfer requires CUDA metadata")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        buffer = self._full_verification_buffers.get(device)
        if buffer is None:
            buffer = _FullVerificationTransferBuffer(
                page_size=self.page_size,
                device=device,
                max_pinned_memory_bytes=self.max_pinned_memory_bytes,
                pinned_memory=self._pinned_memory,
                gather_workers=self.full_verify_gather_workers,
                performance_stats=self.performance_stats,
            )
            self._full_verification_buffers[device] = buffer

        return buffer

    def _select_resident_staging_prefix(
        self,
        pool: _LayerClusterPagePool,
        cluster_ids_cpu: torch.Tensor,
        logical_page_ids_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a complete priority prefix that fits one H2D ring slot."""
        transfer_buffer = self._get_full_verification_buffer(pool)
        page_capacity = transfer_buffer.cpu_slot_capacity(pool)
        flat_cluster_ids = cluster_ids_cpu.reshape(-1)
        flat_page_ids = logical_page_ids_cpu.reshape(
            flat_cluster_ids.numel(), logical_page_ids_cpu.shape[-1]
        )

        if cluster_ids_cpu.ndim <= 1:
            priority_positions = range(flat_cluster_ids.numel())
        else:
            num_ranks = cluster_ids_cpu.shape[-1]
            num_groups = flat_cluster_ids.numel() // num_ranks
            priority_positions = (
                group_index * num_ranks + rank
                for rank in range(num_ranks)
                for group_index in range(num_groups)
            )

        selected_positions: list[int] = []
        selected_pages: set[int] = set()
        selected_clusters: set[int] = set()
        for position in priority_positions:
            cluster_id = int(flat_cluster_ids[position])
            if cluster_id < 0 or cluster_id in selected_clusters:
                continue
            cluster_pages = tuple(
                page_id for page_id in flat_page_ids[position].tolist() if page_id >= 0
            )
            new_pages = tuple(
                page_id for page_id in cluster_pages if page_id not in selected_pages
            )
            if len(selected_pages) + len(new_pages) > page_capacity:
                break
            selected_positions.append(position)
            selected_clusters.add(cluster_id)
            selected_pages.update(new_pages)

        selection = torch.tensor(selected_positions, dtype=torch.int64)
        return (
            flat_cluster_ids.index_select(0, selection),
            flat_page_ids.index_select(0, selection),
        )

    def _stage_resident_pages(
        self,
        pool: _LayerClusterPagePool,
        source_page_ids_cpu: torch.Tensor,
        unique_page_ids_cpu: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        _FullVerificationTransferBuffer,
        _PinnedPageTransferSlot | None,
    ]:
        if unique_page_ids_cpu is None:
            logical_page_ids_cpu = source_page_ids_cpu
            valid_page_mask = logical_page_ids_cpu >= 0
            valid_page_ids = logical_page_ids_cpu[valid_page_mask]
            unique_page_ids_cpu, inverse = torch.unique(
                valid_page_ids, sorted=False, return_inverse=True
            )
            source_page_ids_cpu = torch.full_like(logical_page_ids_cpu, -1)
            source_page_ids_cpu.masked_scatter_(valid_page_mask, inverse)

        transfer_buffer = self._get_full_verification_buffer(pool)
        num_pages = unique_page_ids_cpu.numel()
        if num_pages > transfer_buffer.cpu_slot_capacity(pool):
            raise RuntimeError(
                "RetroSpec resident staging selection exceeds one fixed H2D slot"
            )

        if not num_pages:
            empty_shape = (0, pool.page_size, pool.head_size)
            empty_keys = torch.empty(empty_shape, dtype=pool.dtype, device="cpu")
            return (
                source_page_ids_cpu,
                empty_keys,
                empty_keys.clone(),
                transfer_buffer,
                None,
            )

        staged_keys, staged_values, slot = transfer_buffer.stage_cpu_pages(
            pool, unique_page_ids_cpu
        )
        return (
            source_page_ids_cpu,
            staged_keys,
            staged_values,
            transfer_buffer,
            slot,
        )

    def _stage_missing_pages(
        self,
        pool: _LayerClusterPagePool,
        logical_page_ids: torch.Tensor,
        miss_cluster_mask: torch.Tensor,
        logical_page_ids_cpu: torch.Tensor,
        miss_cluster_mask_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        """Copy non-resident selected pages into a temporary GPU page pool."""
        if logical_page_ids.ndim < 1:
            raise ValueError("Logical page IDs must have at least one dimension")
        if miss_cluster_mask.shape != logical_page_ids.shape[:-1]:
            raise ValueError("Miss-cluster mask does not match logical page IDs")
        if pool.metadata_device.type != "cuda":
            raise RuntimeError("Temporary cluster-page staging requires CUDA metadata")

        if logical_page_ids_cpu.device.type != "cpu":
            raise ValueError("CPU logical page IDs must reside on CPU")
        if miss_cluster_mask_cpu.device.type != "cpu":
            raise ValueError("CPU miss mask must reside on CPU")
        if logical_page_ids_cpu.shape != logical_page_ids.shape:
            raise ValueError("CPU logical page IDs do not match the GPU layout")
        if miss_cluster_mask_cpu.shape != miss_cluster_mask.shape:
            raise ValueError("CPU miss mask does not match the GPU layout")

        missing_page_mask_cpu = miss_cluster_mask_cpu.unsqueeze(-1) & (
            logical_page_ids_cpu >= 0
        )

        missing_page_mask = miss_cluster_mask.unsqueeze(-1) & (logical_page_ids >= 0)
        staging_page_ids = torch.full_like(
            logical_page_ids,
            -1,
            dtype=torch.int64,
        )
        num_staging_pages = int(missing_page_mask_cpu.sum().item())

        if num_staging_pages:
            staging_page_ids.masked_scatter_(
                missing_page_mask,
                torch.arange(
                    num_staging_pages,
                    dtype=torch.int64,
                    device=pool.metadata_device,
                ),
            )

        staging_shape = (
            num_staging_pages,
            pool.page_size,
            pool.head_size,
        )
        staging_key_pages = torch.empty(
            staging_shape,
            dtype=pool.dtype,
            device=pool.metadata_device,
        )
        staging_value_pages = torch.empty_like(staging_key_pages)

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "verification_miss_pages", num_staging_pages
            )
            self.performance_stats.add_counter(
                "verification_miss_h2d_bytes",
                staging_key_pages.nbytes + staging_value_pages.nbytes,
            )

        staging_ready_event: torch.cuda.Event | None = None
        if num_staging_pages:
            missing_logical_page_ids = logical_page_ids_cpu[
                missing_page_mask_cpu
            ].contiguous()
            transfer_buffer = self._get_full_verification_buffer(pool)
            page_capacity = transfer_buffer.cpu_slot_capacity(pool)
            current_stream = torch.cuda.current_stream(pool.metadata_device)

            page_start = 0
            while page_start < num_staging_pages:
                page_end = min(page_start + page_capacity, num_staging_pages)
                gather_started_at = (
                    perf_counter()
                    if self.performance_stats is not None
                    and self.performance_stats.enabled
                    else None
                )
                cpu_keys, cpu_values, cpu_slot = transfer_buffer.stage_cpu_pages(
                    pool, missing_logical_page_ids[page_start:page_end]
                )
                if gather_started_at is not None:
                    self.performance_stats.record_cpu_time(
                        "verification_miss_cpu_gather",
                        perf_counter() - gather_started_at,
                    )

                h2d_timer = (
                    None
                    if self.performance_stats is None
                    else self.performance_stats.start_cuda_timer(
                        "verification_miss_h2d", current_stream
                    )
                )
                staging_key_pages[page_start:page_end].copy_(
                    cpu_keys, non_blocking=self.pin_memory
                )
                staging_value_pages[page_start:page_end].copy_(
                    cpu_values, non_blocking=self.pin_memory
                )
                if self.performance_stats is not None:
                    self.performance_stats.stop_cuda_timer(h2d_timer, current_stream)
                chunk_ready_event = torch.cuda.Event()
                chunk_ready_event.record(current_stream)
                transfer_buffer.release_cpu_slot(cpu_slot, chunk_ready_event)
                page_start = page_end

            staging_ready_event = torch.cuda.Event()
            staging_ready_event.record(current_stream)

        return (
            staging_page_ids,
            staging_key_pages,
            staging_value_pages,
            staging_ready_event,
        )

    def _get_resident_prefetch_stream(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream:
        device = self._canonical_cuda_device(device)
        stream = self._resident_prefetch_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._resident_prefetch_streams[device] = stream
        return stream

    def reserve_resident_access_ring(
        self,
        device: torch.device,
        capacity: int,
    ) -> None:
        """Reserve one layer record's share of the pinned wave ring."""
        if not self.pin_memory or capacity <= 0:
            return
        device = self._canonical_cuda_device(device)

        with self._resident_prefetch_lock:
            self._resident_access_record_capacity = max(
                self._resident_access_record_capacity,
                capacity,
            )
            wave_capacity = (
                self._resident_access_record_capacity
                * self._resident_prefetch_wave_max_records
            )
            slots = self._resident_prefetch_slots.setdefault(
                device,
                [
                    _PinnedSelectionSlot(pinned_memory=self._pinned_memory)
                    for _ in range(self._RESIDENT_PREFETCH_RING_SIZE)
                ],
            )
            for slot in slots:
                if not slot.in_use:
                    slot.reserve_capacity(
                        wave_capacity, self._resident_prefetch_wave_max_records
                    )

    def configure_resident_prefetch_wave(self, max_records: int) -> None:
        """Configure the maximum number of layer records in one draft wave."""
        if max_records <= 0:
            raise ValueError("Resident prefetch wave size must be positive")

        with self._resident_prefetch_lock:
            self._resident_prefetch_wave_max_records = max(
                self._resident_prefetch_wave_max_records, max_records
            )
            self._verification_resolve_ring_size = max(
                self._verification_resolve_ring_size, max_records
            )
            wave_capacity = (
                self._resident_access_record_capacity
                * self._resident_prefetch_wave_max_records
            )
            for slots in self._resident_prefetch_slots.values():
                for slot in slots:
                    if not slot.in_use:
                        slot.reserve_capacity(
                            wave_capacity, self._resident_prefetch_wave_max_records
                        )

    def _acquire_resident_prefetch_slot(
        self,
        device: torch.device,
    ) -> _PinnedSelectionSlot | None:
        device = self._canonical_cuda_device(device)

        with self._resident_prefetch_lock:
            wave_capacity = (
                self._resident_access_record_capacity
                * self._resident_prefetch_wave_max_records
            )
            slots = self._resident_prefetch_slots.setdefault(
                device,
                [
                    _PinnedSelectionSlot(pinned_memory=self._pinned_memory)
                    for _ in range(self._RESIDENT_PREFETCH_RING_SIZE)
                ],
            )
            for slot in slots:
                if not slot.in_use:
                    slot.reserve_capacity(
                        wave_capacity, self._resident_prefetch_wave_max_records
                    )
                    slot.in_use = True
                    return slot

        return None

    def _release_resident_prefetch_slot(
        self,
        slot: _PinnedSelectionSlot,
    ) -> None:
        with self._resident_prefetch_lock:
            if not slot.in_use:
                raise RuntimeError("Resident prefetch slot was already released")
            slot.in_use = False
            slot.reserve_capacity(
                self._resident_access_record_capacity
                * self._resident_prefetch_wave_max_records,
                self._resident_prefetch_wave_max_records,
            )

    def _validate_resident_prefetch_wave(
        self,
        records: tuple[RetroSpecResidentPrefetchInput, ...],
    ) -> torch.device:
        layer_names = tuple(record.layer_name for record in records)
        if len(layer_names) != len(set(layer_names)):
            raise ValueError("Resident prefetch wave layer names must be unique")
        if len(records) > self._resident_prefetch_wave_max_records:
            raise ValueError("Resident prefetch wave exceeds configured capacity")

        device = self._canonical_cuda_device(records[0].miss_cluster_ids.device)
        for record in records:
            cluster_ids = record.miss_cluster_ids
            positions = record.miss_positions
            count = record.miss_count
            if cluster_ids.numel() == 0:
                raise ValueError("Resident prefetch records must not be empty")
            if cluster_ids.device.type != "cuda":
                raise ValueError(
                    "Asynchronous resident prefetch requires CUDA cluster IDs"
                )
            if cluster_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Cluster IDs must use an integral dtype")
            if positions.shape != cluster_ids.shape:
                raise ValueError("Resident miss positions must match cluster IDs")
            if positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("Resident miss positions must be integral")
            if positions.device != cluster_ids.device:
                raise ValueError("Resident miss commands must use one device")
            if count.shape != (1,) or count.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("Resident miss count must be one integral value")
            if count.device != cluster_ids.device:
                raise ValueError("Resident miss count must use the command device")
            if record.num_groups <= 0 or record.num_ranks <= 0:
                raise ValueError("Resident prefetch layout must be positive")
            if record.source not in _PREFETCH_SOURCE_PRIORITY:
                raise ValueError("Unknown resident prefetch source")
            if self._canonical_cuda_device(cluster_ids.device) != device:
                raise ValueError("Resident prefetch wave must use one CUDA device")
        return device

    def _submit_resident_prefetch_wave(
        self,
        records: tuple[RetroSpecResidentPrefetchInput, ...],
        slot: _PinnedSelectionSlot,
        source_ready_events: Sequence[torch.cuda.Event] | None,
    ) -> None:
        device = self._canonical_cuda_device(records[0].miss_cluster_ids.device)
        stream = self._get_resident_prefetch_stream(device)
        progress = _ResidentPrefetchWaveProgress.create(
            tuple(record.layer_name for record in records)
        )
        ownership_transferred = False
        try:
            cpu_views = slot.reserve_wave(records)
            if source_ready_events is not None and len(source_ready_events) != len(
                records
            ):
                raise ValueError(
                    "Resident prefetch producer events must match layer records"
                )
            if source_ready_events is None:
                stream.wait_stream(torch.cuda.current_stream(device))

            with torch.cuda.stream(stream):
                for record_index, (
                    record,
                    (cluster_ids_cpu, positions_cpu, count_cpu),
                ) in enumerate(zip(records, cpu_views)):
                    if source_ready_events is not None:
                        stream.wait_event(source_ready_events[record_index])
                    cluster_ids_cpu.copy_(record.miss_cluster_ids, non_blocking=True)
                    positions_cpu.copy_(record.miss_positions, non_blocking=True)
                    count_cpu.copy_(record.miss_count, non_blocking=True)
                metadata_ready_event = torch.cuda.Event()
                metadata_ready_event.record(stream)

            for record in records:
                record.miss_cluster_ids.record_stream(stream)
                record.miss_positions.record_stream(stream)
                record.miss_count.record_stream(stream)

            staged = _StagedResidentPrefetchWave(
                records=tuple(
                    _StagedResidentPrefetchRecord(
                        layer_name=record.layer_name,
                        miss_cluster_ids_cpu=cluster_ids_cpu,
                        miss_positions_cpu=positions_cpu,
                        miss_count_cpu=count_cpu,
                        num_groups=record.num_groups,
                        num_ranks=record.num_ranks,
                        source=record.source,
                        sequence=record.sequence,
                    )
                    for record, (
                        cluster_ids_cpu,
                        positions_cpu,
                        count_cpu,
                    ) in zip(records, cpu_views)
                ),
                device=device,
                metadata_ready_event=metadata_ready_event,
                execution_stream=stream,
                progress=progress,
                slot=slot,
            )
            priority = max(
                _PREFETCH_SOURCE_PRIORITY[record.source] for record in records
            )
            future = self._resident_prefetch_executor.submit(
                priority, self._finish_resident_prefetch_wave, staged
            )
            ownership_transferred = True

            with self._resident_prefetch_lock:
                self._resident_prefetch_last_metadata_events[device] = (
                    metadata_ready_event
                )
                self._resident_prefetch_futures.append(
                    _ResidentPrefetchWaveFuture(
                        device=device,
                        layer_names=frozenset(record.layer_name for record in records),
                        progress=progress,
                        future=future,
                    )
                )

            if self.performance_stats is not None:
                self.performance_stats.add_counter("prefetch_submitted", len(records))
                self.performance_stats.add_counter("prefetch_waves_submitted")
                self.performance_stats.add_counter(
                    "prefetch_wave_records", len(records)
                )
                self.performance_stats.add_counter(
                    "prefetch_command_capacity",
                    sum(record.miss_cluster_ids.numel() for record in records),
                )
        except BaseException:
            if not ownership_transferred:
                stream.synchronize()
                self._release_resident_prefetch_slot(slot)
            raise

    def _stamp_resident_prefetch_records(
        self,
        records: tuple[RetroSpecResidentPrefetchInput, ...],
    ) -> tuple[RetroSpecResidentPrefetchInput, ...]:
        accepted: list[RetroSpecResidentPrefetchInput] = []
        with self._resident_prefetch_lock:
            for record in records:
                priority = _PREFETCH_SOURCE_PRIORITY[record.source]
                previous = self._resident_prefetch_latest.get(record.layer_name)
                if previous is not None and priority < previous[0]:
                    if self.performance_stats is not None:
                        self.performance_stats.add_counter(
                            "prefetch_lower_priority_dropped"
                        )
                    continue
                sequence = self._resident_prefetch_next_sequence
                self._resident_prefetch_next_sequence += 1
                stamped = replace(record, sequence=sequence)
                self._resident_prefetch_latest[record.layer_name] = (
                    priority,
                    sequence,
                )
                accepted.append(stamped)
        return tuple(accepted)

    def _resident_prefetch_record_is_current(
        self, record: _StagedResidentPrefetchRecord
    ) -> bool:
        priority = _PREFETCH_SOURCE_PRIORITY[record.source]
        with self._resident_prefetch_lock:
            return self._resident_prefetch_latest.get(record.layer_name) == (
                priority,
                record.sequence,
            )

    def _complete_resident_prefetch_record(
        self, record: _StagedResidentPrefetchRecord
    ) -> None:
        priority = _PREFETCH_SOURCE_PRIORITY[record.source]
        with self._resident_prefetch_lock:
            if self._resident_prefetch_latest.get(record.layer_name) == (
                priority,
                record.sequence,
            ):
                self._resident_prefetch_latest.pop(record.layer_name)

    def _discard_resident_prefetch_records(
        self, records: Sequence[RetroSpecResidentPrefetchInput]
    ) -> None:
        """Forget commands that failed before ownership reached the worker."""
        with self._resident_prefetch_lock:
            for record in records:
                priority = _PREFETCH_SOURCE_PRIORITY[record.source]
                if self._resident_prefetch_latest.get(record.layer_name) == (
                    priority,
                    record.sequence,
                ):
                    self._resident_prefetch_latest.pop(record.layer_name)

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
