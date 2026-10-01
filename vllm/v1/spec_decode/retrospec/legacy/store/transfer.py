# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Condition, Lock
from threading import Event as ThreadEvent
from time import perf_counter

import torch

from vllm import _custom_ops as ops
from vllm.v1.spec_decode.retrospec.legacy.pinned_memory import (
    RetroSpecPinnedMemoryManager,
)
from vllm.v1.spec_decode.retrospec.legacy.store.contracts import (
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationStaging,
    RetroSpecFullVerificationTicket,
)
from vllm.v1.spec_decode.retrospec.legacy.store.pool import (
    _FullVerificationGPUArena,
    _FullVerificationSourceSnapshot,
    _LayerClusterPagePool,
    _PinnedPageTransferSlot,
)
from vllm.v1.spec_decode.retrospec.performance import RetroSpecPerformanceStats


class _FullVerificationTransferBuffer:
    """Two reusable full-layer H2D page arenas for a CUDA device."""

    _MIN_CAPACITY = 64
    _RESIDENT_PREFETCH_RING_SIZE = 2

    def __init__(
        self,
        page_size: int,
        device: torch.device,
        max_pinned_memory_bytes: int,
        pinned_memory: RetroSpecPinnedMemoryManager,
        gather_workers: int,
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if device.type != "cuda":
            raise ValueError("Full-verification transfer buffer requires CUDA")
        if max_pinned_memory_bytes <= 0:
            raise ValueError("max_pinned_memory_bytes must be positive")
        if gather_workers <= 0:
            raise ValueError("gather_workers must be positive")

        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.page_size = page_size
        self.device = device
        self.performance_stats = performance_stats
        self.max_pinned_memory_bytes = max_pinned_memory_bytes
        self._pinned_memory = pinned_memory
        self.pin_memory = pinned_memory.enabled
        self.gather_workers = gather_workers

        self._gpu_arenas = [
            _FullVerificationGPUArena(),
            _FullVerificationGPUArena(),
        ]
        self._gpu_arena_cursor = 0

        self._transfer_stream = torch.cuda.Stream(device=device)
        self._cpu_slots: list[_PinnedPageTransferSlot] = []
        self._cpu_slot_layout: tuple[torch.dtype, int] | None = None
        self._cpu_slot_capacity = 0
        self._cpu_slot_cursor = 0
        self._cpu_slot_lock = Lock()
        self._cpu_slot_available = Condition(self._cpu_slot_lock)
        self._gather_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"retrospec-full-verify-{device.index}",
        )
        self._closed = False

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (max(value, 1) - 1).bit_length()

    @property
    def capacity(self) -> int:
        return max(arena.capacity for arena in self._gpu_arenas)

    def _release_old_storage(self, arena: _FullVerificationGPUArena) -> None:
        if arena.key_tokens is None or arena.value_tokens is None:
            return

        # The transfer stream already waits for the execution stream before
        # this method is called. Recording the old tensors on the transfer
        # stream prevents the CUDA allocator from recycling them too early.
        arena.key_tokens.record_stream(self._transfer_stream)
        arena.value_tokens.record_stream(self._transfer_stream)
        assert arena.token_offsets is not None
        assert arena.token_counts is not None
        arena.token_offsets.record_stream(self._transfer_stream)
        arena.token_counts.record_stream(self._transfer_stream)

    def _ensure_capacity(
        self,
        arena: _FullVerificationGPUArena,
        required_tokens: int,
        required_metadata: int,
        dtype: torch.dtype,
        head_size: int,
    ) -> None:
        if required_tokens < 0:
            raise ValueError("required_tokens must be non-negative")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        layout_changed = arena.dtype != dtype or arena.head_size != head_size
        if required_metadata < 0:
            raise ValueError("required_metadata must be non-negative")
        if (
            not layout_changed
            and required_tokens <= arena.capacity
            and required_metadata <= arena.metadata_capacity
        ):
            return

        self._release_old_storage(arena)

        arena.dtype = dtype
        arena.head_size = head_size
        arena.capacity = max(
            self._MIN_CAPACITY,
            self._next_power_of_two(required_tokens),
        )
        arena.metadata_capacity = max(
            self._MIN_CAPACITY,
            self._next_power_of_two(required_metadata),
        )

        shape = (arena.capacity, head_size)
        arena.key_tokens = torch.empty(shape, dtype=dtype, device=self.device)
        arena.value_tokens = torch.empty_like(arena.key_tokens)
        arena.token_offsets = torch.empty(
            arena.metadata_capacity, dtype=torch.int64, device=self.device
        )
        arena.token_counts = torch.empty(
            arena.metadata_capacity, dtype=torch.int32, device=self.device
        )

    def _ensure_cpu_slots(self, dtype: torch.dtype, head_size: int) -> None:
        layout = (dtype, head_size)
        with self._cpu_slot_lock:
            if self._cpu_slot_layout == layout:
                return
            if any(slot.in_use for slot in self._cpu_slots):
                raise RuntimeError(
                    "Cannot change the full-verification staging layout while "
                    "a pinned slot is active"
                )
            if self._cpu_slots:
                for slot in self._cpu_slots:
                    if slot.reuse_ready_event is not None:
                        slot.reuse_ready_event.synchronize()
                    self._pinned_memory.release(slot.key_pages)
                    self._pinned_memory.release(slot.value_pages)

            page_pair_bytes = 2 * self.page_size * head_size * dtype.itemsize
            h2d_budget = self.max_pinned_memory_bytes // 2
            self._cpu_slot_capacity = (
                h2d_budget // self._RESIDENT_PREFETCH_RING_SIZE // page_pair_bytes
            )
            if self._cpu_slot_capacity == 0:
                raise RuntimeError(
                    "retrospec_max_pinned_memory cannot hold one H2D page per ring slot"
                )
            shape = (self._cpu_slot_capacity, self.page_size, head_size)
            self._cpu_slots = []
            try:
                for _ in range(self._RESIDENT_PREFETCH_RING_SIZE):
                    key_pages = self._pinned_memory.empty(
                        shape, dtype, "full-verification-h2d-keys"
                    )
                    try:
                        value_pages = self._pinned_memory.empty(
                            shape, dtype, "full-verification-h2d-values"
                        )
                    except BaseException:
                        self._pinned_memory.release(key_pages)
                        raise
                    self._cpu_slots.append(
                        _PinnedPageTransferSlot(
                            key_pages=key_pages,
                            value_pages=value_pages,
                        )
                    )
            except BaseException:
                for slot in self._cpu_slots:
                    self._pinned_memory.release(slot.key_pages)
                    self._pinned_memory.release(slot.value_pages)
                self._cpu_slots.clear()
                raise
            self._cpu_slot_layout = layout
            self._cpu_slot_cursor = 0

    def close(self) -> None:
        with self._cpu_slot_available:
            if self._closed:
                return
            self._closed = True
            self._cpu_slot_available.notify_all()
        self._gather_executor.shutdown(wait=True)

        for slot in self._cpu_slots:
            if slot.in_use:
                raise RuntimeError(
                    "Cannot close an active full-verification H2D staging slot"
                )
            if slot.reuse_ready_event is not None:
                slot.reuse_ready_event.synchronize()
            self._pinned_memory.release(slot.key_pages)
            self._pinned_memory.release(slot.value_pages)
        self._cpu_slots.clear()
        self._cpu_slot_layout = None
        self._cpu_slot_capacity = 0

    def _acquire_cpu_slot(self) -> _PinnedPageTransferSlot:
        with self._cpu_slot_available:
            while True:
                if self._closed:
                    raise RuntimeError("Full-verification transfer buffer is closed")
                num_slots = len(self._cpu_slots)
                if num_slots == 0:
                    raise RuntimeError(
                        "RetroSpec pinned H2D staging ring is not initialized"
                    )

                slot = None
                for offset in range(num_slots):
                    slot_index = (self._cpu_slot_cursor + offset) % num_slots
                    candidate = self._cpu_slots[slot_index]
                    if candidate.in_use:
                        continue
                    slot = candidate
                    self._cpu_slot_cursor = (slot_index + 1) % num_slots
                    slot.in_use = True
                    break
                if slot is not None:
                    break
                self._cpu_slot_available.wait()

        if slot.reuse_ready_event is not None:
            slot.reuse_ready_event.synchronize()
            slot.reuse_ready_event = None
        return slot

    def stage_cpu_pages(
        self,
        pool: _LayerClusterPagePool,
        page_ids_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, _PinnedPageTransferSlot]:
        """Gather one bounded selection from pageable slabs into pinned CPU."""
        self._ensure_cpu_slots(pool.dtype, pool.head_size)
        page_ids_cpu = page_ids_cpu.reshape(-1).contiguous()
        num_pages = page_ids_cpu.numel()
        if num_pages > self._cpu_slot_capacity:
            raise RuntimeError(
                "RetroSpec resident H2D selection exceeds the fixed pinned staging "
                "slot; increase retrospec_max_pinned_memory"
            )

        slot = self._acquire_cpu_slot()
        staged_keys = slot.key_pages[:num_pages]
        staged_values = slot.value_pages[:num_pages]
        source = pool.snapshot_full_verification_sources()
        try:
            ops.retrospec_gather_cluster_pages(
                source.key_slabs,
                source.value_slabs,
                page_ids_cpu,
                pool.page_size,
                staged_keys,
                staged_values,
                self.gather_workers,
            )
        except BaseException:
            self.release_cpu_slot(slot, None)
            raise
        return staged_keys, staged_values, slot

    def cpu_slot_capacity(self, pool: _LayerClusterPagePool) -> int:
        self._ensure_cpu_slots(pool.dtype, pool.head_size)
        return self._cpu_slot_capacity

    def _stage_cpu_token_chunk(
        self,
        source: _FullVerificationSourceSnapshot,
        range_tables: tuple[torch.Tensor, ...],
        token_offsets_cpu: torch.Tensor,
        token_start: int,
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, _PinnedPageTransferSlot]:
        """Gather one compact token interval using the native CPU operator."""
        self._ensure_cpu_slots(source.dtype, source.head_size)
        token_capacity = self._cpu_slot_capacity * self.page_size
        if token_count > token_capacity:
            raise RuntimeError(
                "RetroSpec compact H2D selection exceeds the fixed pinned staging slot"
            )

        slot = self._acquire_cpu_slot()
        key_tokens = slot.key_pages.view(-1, source.head_size)[:token_count]
        value_tokens = slot.value_pages.view(-1, source.head_size)[:token_count]
        try:
            ops.retrospec_gather_compact_kv(
                source.key_slabs,
                source.value_slabs,
                range_tables,
                token_offsets_cpu,
                token_start,
                key_tokens,
                value_tokens,
                self.gather_workers,
            )
        except BaseException:
            self.release_cpu_slot(slot, None)
            raise
        return key_tokens, value_tokens, slot

    def release_cpu_slot(
        self,
        slot: _PinnedPageTransferSlot,
        reuse_ready_event: torch.cuda.Event | None,
    ) -> None:
        with self._cpu_slot_available:
            if not slot.in_use:
                raise RuntimeError("RetroSpec pinned H2D slot was already released")
            slot.reuse_ready_event = reuse_ready_event
            slot.in_use = False
            self._cpu_slot_available.notify()

    def submit(
        self,
        source: _FullVerificationSourceSnapshot,
        descriptors: Sequence[RetroSpecFullVerificationDescriptor],
    ) -> RetroSpecFullVerificationTicket:
        if self._closed:
            raise RuntimeError("Full-verification transfer buffer is closed")
        if not descriptors:
            raise ValueError("Full-verification staging requires request descriptors")

        execution_ready_event = torch.cuda.Event()
        execution_ready_event.record(torch.cuda.current_stream(self.device))
        cancel_event = ThreadEvent()
        future = self._gather_executor.submit(
            self._stage,
            source,
            tuple(descriptors),
            execution_ready_event,
            cancel_event,
        )
        return RetroSpecFullVerificationTicket(
            future=future,
            cancel_event=cancel_event,
        )

    def _stage(
        self,
        source: _FullVerificationSourceSnapshot,
        descriptors: tuple[RetroSpecFullVerificationDescriptor, ...],
        execution_ready_event: torch.cuda.Event,
        cancel_event: ThreadEvent,
    ) -> RetroSpecFullVerificationStaging:
        """Gather compact CPU ranges and enqueue one full-layer H2D copy."""
        if cancel_event.is_set():
            raise CancelledError

        num_kv_heads = descriptors[0].num_kv_heads
        if any(descriptor.num_kv_heads != num_kv_heads for descriptor in descriptors):
            raise ValueError("Full-verification descriptors changed KV-head count")

        token_counts_cpu = torch.stack(
            tuple(descriptor.head_token_counts_tensor for descriptor in descriptors),
            dim=0,
        ).contiguous()
        flat_counts = token_counts_cpu.reshape(-1).to(dtype=torch.int64)
        flat_offsets = torch.zeros_like(flat_counts)
        if flat_offsets.numel() > 1:
            flat_offsets[1:] = flat_counts.cumsum(0)[:-1]
        token_offsets_cpu = flat_offsets.view_as(token_counts_cpu).contiguous()
        num_tokens = int(flat_counts.sum().item())
        max_tokens_per_head = int(flat_counts.max().item()) if num_tokens else 0
        if sum(descriptor.num_tokens for descriptor in descriptors) != num_tokens:
            raise RuntimeError("Full-verification compact descriptor is inconsistent")

        range_tables = tuple(descriptor.range_table for descriptor in descriptors)
        if cancel_event.is_set():
            raise CancelledError
        self._ensure_cpu_slots(source.dtype, source.head_size)
        token_capacity = self._cpu_slot_capacity * self.page_size
        if cancel_event.is_set():
            raise CancelledError

        with torch.cuda.device(self.device):
            arena = self._gpu_arenas[self._gpu_arena_cursor]
            self._gpu_arena_cursor = (self._gpu_arena_cursor + 1) % len(
                self._gpu_arenas
            )
            ready_event = torch.cuda.Event()

            with torch.cuda.stream(self._transfer_stream):
                self._transfer_stream.wait_event(execution_ready_event)
                if cancel_event.is_set():
                    raise CancelledError
                self._ensure_capacity(
                    arena=arena,
                    required_tokens=num_tokens,
                    required_metadata=token_counts_cpu.numel(),
                    dtype=source.dtype,
                    head_size=source.head_size,
                )

                assert arena.key_tokens is not None
                assert arena.value_tokens is not None
                assert arena.token_offsets is not None
                assert arena.token_counts is not None

                staging_key_tokens = arena.key_tokens[:num_tokens]
                staging_value_tokens = arena.value_tokens[:num_tokens]
                staging_token_offsets = arena.token_offsets[
                    : token_counts_cpu.numel()
                ].view(token_counts_cpu.shape)
                staging_token_counts = arena.token_counts[
                    : token_counts_cpu.numel()
                ].view(token_counts_cpu.shape)
                staging_token_offsets.copy_(token_offsets_cpu)
                staging_token_counts.copy_(token_counts_cpu)

            transfer_timer = None
            gather_elapsed = 0.0
            transferred_tokens = 0
            cancelled = False
            token_start = 0
            while token_start < num_tokens:
                if cancel_event.is_set():
                    cancelled = True
                    break

                chunk_tokens = min(token_capacity, num_tokens - token_start)
                gather_started = perf_counter()
                cpu_keys, cpu_values, cpu_slot = self._stage_cpu_token_chunk(
                    source,
                    range_tables,
                    token_offsets_cpu,
                    token_start,
                    chunk_tokens,
                )
                gather_elapsed += perf_counter() - gather_started
                if cancel_event.is_set():
                    self.release_cpu_slot(cpu_slot, None)
                    cancelled = True
                    break

                token_end = token_start + chunk_tokens

                try:
                    with torch.cuda.stream(self._transfer_stream):
                        if (
                            transfer_timer is None
                            and self.performance_stats is not None
                        ):
                            transfer_timer = self.performance_stats.start_cuda_timer(
                                "full_verify_h2d", self._transfer_stream
                            )
                        staging_key_tokens[token_start:token_end].copy_(
                            cpu_keys, non_blocking=self.pin_memory
                        )
                        staging_value_tokens[token_start:token_end].copy_(
                            cpu_values, non_blocking=self.pin_memory
                        )
                        chunk_ready_event = torch.cuda.Event()
                        chunk_ready_event.record(self._transfer_stream)
                except BaseException:
                    self._transfer_stream.synchronize()
                    self.release_cpu_slot(cpu_slot, None)
                    raise

                self.release_cpu_slot(cpu_slot, chunk_ready_event)
                token_start = token_end
                transferred_tokens = token_end
                if cancel_event.is_set():
                    cancelled = True
                    break

            with torch.cuda.stream(self._transfer_stream):
                if self.performance_stats is not None:
                    transfer_bytes = (
                        transferred_tokens
                        * source.head_size
                        * source.dtype.itemsize
                        * 2
                    )
                    self.performance_stats.add_counter(
                        "full_verify_h2d_tokens", transferred_tokens
                    )
                    self.performance_stats.add_counter(
                        "full_verify_h2d_bytes", transfer_bytes
                    )
                    self.performance_stats.record_cpu_time(
                        "full_verify_cpu_gather", gather_elapsed
                    )
                    self.performance_stats.stop_cuda_timer(
                        transfer_timer, self._transfer_stream
                    )
                if not cancelled:
                    ready_event.record(self._transfer_stream)

        if cancelled:
            raise CancelledError

        return RetroSpecFullVerificationStaging(
            key_tokens=staging_key_tokens,
            value_tokens=staging_value_tokens,
            token_offsets=staging_token_offsets,
            token_counts=staging_token_counts,
            max_tokens_per_head=max_tokens_per_head,
            ready_event=ready_event,
        )
