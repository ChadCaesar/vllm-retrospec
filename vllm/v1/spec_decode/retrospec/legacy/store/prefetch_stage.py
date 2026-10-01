# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter

import torch

from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    _PREFETCH_SOURCE_PRIORITY,
    RetroSpecResidentPrefetchInput,
    _FullVerificationTransferBuffer,
    _LayerClusterPagePool,
    _PinnedPageTransferSlot,
    _PinnedSelectionSlot,
    _ResidentPrefetchWaveFuture,
    _ResidentPrefetchWaveProgress,
    _StagedResidentPrefetchRecord,
    _StagedResidentPrefetchWave,
)


class _RetroSpecClusterPageStorePrefetchStageMixin:
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
