# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetroSpecClusterSummary:
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor


@dataclass(frozen=True)
class RetroSpecStagedClusterSummary:
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    resident_summary: RetroSpecClusterSummary
    ready_event: torch.cuda.Event | None


@dataclass(frozen=True)
class RetroSpecResidentSegment:
    """Transient payload published into a layer-level GPU arena."""

    layer_name: str
    request_id: str

    indexed_start: int
    indexed_end: int
    cluster_start: int

    cluster_ids_cpu: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor

    cluster_page_ids_cpu: torch.Tensor
    cluster_page_token_counts_cpu: torch.Tensor
    cluster_page_counts_cpu: torch.Tensor


@dataclass
class RetroSpecResidentLayerArena:
    """Packed growable resident cluster index for one attention layer."""

    # Cluster storage is [num_kv_heads, cluster_capacity, ...].
    cluster_ids: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_page_starts: torch.Tensor
    cluster_page_counts: torch.Tensor

    # Page storage is [num_kv_heads, page_capacity].
    page_ids: torch.Tensor
    page_token_counts: torch.Tensor

    # Fixed request-slot descriptors point into the packed storage.
    cluster_offsets: torch.Tensor
    num_clusters: torch.Tensor
    page_offsets: torch.Tensor
    num_pages: torch.Tensor
    generations: torch.Tensor
    indexed_starts: torch.Tensor
    indexed_ends: torch.Tensor


@dataclass(frozen=True)
class RetroSpecResidentBatchView:
    """Map one active batch onto a persistent layer arena."""

    arena: RetroSpecResidentLayerArena | None
    request_slot_ids: torch.Tensor
    max_num_clusters: int
    max_pages_per_cluster: int
    max_num_pages: int


@dataclass(frozen=True)
class _ResidentRequestState:
    """CPU control metadata for one request in one layer arena."""

    slot: int
    generation: int
    indexed_start: int
    indexed_end: int
    cluster_offset: int
    cluster_capacity: int
    num_clusters: int
    page_offset: int
    page_capacity: int
    page_counts: tuple[int, ...]
    max_pages_per_cluster: int


class _FreeSpanAllocator:
    """Coalescing first-fit allocator for one packed tensor dimension."""

    def __init__(self) -> None:
        self.capacity = 0
        self._free_spans: list[tuple[int, int]] = []

    def allocate(self, size: int) -> int | None:
        if size <= 0:
            raise ValueError("Allocated span size must be positive")

        for index, (offset, span_size) in enumerate(self._free_spans):
            if span_size < size:
                continue
            if span_size == size:
                self._free_spans.pop(index)
            else:
                self._free_spans[index] = (offset + size, span_size - size)
            return offset
        return None

    def release(self, offset: int, size: int) -> None:
        if offset < 0 or size <= 0 or offset + size > self.capacity:
            raise ValueError("Released span is outside allocator capacity")

        self._free_spans.append((offset, size))
        self._free_spans.sort()
        merged: list[tuple[int, int]] = []
        for span_offset, span_size in self._free_spans:
            if not merged:
                merged.append((span_offset, span_size))
                continue

            previous_offset, previous_size = merged[-1]
            previous_end = previous_offset + previous_size
            if span_offset < previous_end:
                raise RuntimeError("Packed arena free spans overlap")
            if span_offset == previous_end:
                merged[-1] = (previous_offset, previous_size + span_size)
            else:
                merged.append((span_offset, span_size))
        self._free_spans = merged

    def extend(self, new_capacity: int) -> None:
        if new_capacity <= self.capacity:
            raise ValueError("Allocator capacity must grow")
        old_capacity = self.capacity
        self.capacity = new_capacity
        self.release(old_capacity, new_capacity - old_capacity)


@dataclass
class _PackedLayerState:
    arena: RetroSpecResidentLayerArena
    cluster_allocator: _FreeSpanAllocator
    page_allocator: _FreeSpanAllocator


@dataclass(frozen=True)
class _ResidentSpanTransaction:
    layer_state: _PackedLayerState
    new_cluster_span: tuple[int, int] | None = None
    old_cluster_span: tuple[int, int] | None = None
    new_page_span: tuple[int, int] | None = None
    old_page_span: tuple[int, int] | None = None


class RetroSpecGPUIndexResidencyManager:
    """Own request-slot layer arenas and active batch descriptors."""

    def __init__(
        self,
        pin_memory: bool,
        max_resident_requests: int,
        max_gpu_index_memory_bytes: int,
    ) -> None:
        if max_resident_requests <= 0:
            raise ValueError("max_resident_requests must be positive")
        if max_gpu_index_memory_bytes <= 0:
            raise ValueError("max_gpu_index_memory_bytes must be positive")

        self.pin_memory = pin_memory
        self.max_resident_requests = max_resident_requests
        self.max_gpu_index_memory_bytes = max_gpu_index_memory_bytes
        self._allocated_gpu_index_bytes = 0

        self._active_request_ids: tuple[str, ...] | None = None
        self._active_views: dict[str, RetroSpecResidentBatchView] = {}

        self._request_slots: dict[str, int] = {}
        self._slot_generations = [0] * max_resident_requests
        self._free_request_slots = list(reversed(range(max_resident_requests)))

        self._layer_arenas: dict[str, _PackedLayerState] = {}
        self._resident_states: dict[
            str,
            dict[str, _ResidentRequestState],
        ] = {}
        self._offload_streams: dict[torch.device, torch.cuda.Stream] = {}

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return () if self._active_request_ids is None else self._active_request_ids

    @property
    def resident_request_ids(self) -> tuple[str, ...]:
        request_ids = {
            request_id
            for layer_states in self._resident_states.values()
            for request_id in layer_states
        }
        return tuple(sorted(request_ids))

    @property
    def num_resident_requests(self) -> int:
        return len(self.resident_request_ids)

    @property
    def num_resident_layers(self) -> int:
        return sum(bool(states) for states in self._resident_states.values())

    def activate(self, request_ids: Sequence[str]) -> None:
        if self._active_request_ids is not None:
            raise RuntimeError("A RetroSpec GPU index residency set is already active")

        request_ids = tuple(request_ids)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("RetroSpec resident request IDs must be unique")
        if len(request_ids) > self.max_resident_requests:
            raise RuntimeError(
                "RetroSpec GPU index residency exceeds max_num_seqs: "
                f"{len(request_ids)} > {self.max_resident_requests}"
            )

        self._active_views.clear()
        self._active_request_ids = request_ids

    def deactivate(self) -> None:
        if self._active_request_ids is None:
            raise RuntimeError("No RetroSpec GPU index residency set is active")

        self._active_views.clear()
        self._active_request_ids = None

    def _validate_active_requests(self, request_ids: tuple[str, ...]) -> None:
        if self._active_request_ids is None:
            raise RuntimeError(
                "RetroSpec resident indices may be accessed only inside an "
                "active proposal or full-verification context"
            )
        if request_ids != self._active_request_ids:
            raise RuntimeError(
                "RetroSpec request order does not match the active GPU residency set"
            )

    def _get_or_allocate_request_slot(self, request_id: str) -> int:
        slot = self._request_slots.get(request_id)
        if slot is not None:
            return slot

        if not self._free_request_slots:
            raise RuntimeError(
                "RetroSpec persistent GPU index residency exceeds max_num_seqs"
            )

        slot = self._free_request_slots.pop()
        self._slot_generations[slot] += 1
        self._request_slots[request_id] = slot
        return slot

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (max(value, 1) - 1).bit_length()

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _reserve_gpu_index_bytes(self, num_bytes: int) -> None:
        projected = self._allocated_gpu_index_bytes + num_bytes
        if projected > self.max_gpu_index_memory_bytes:
            raise RuntimeError(
                "RetroSpec GPU index memory budget was exceeded: "
                f"{projected} > {self.max_gpu_index_memory_bytes} bytes"
            )
        self._allocated_gpu_index_bytes = projected

    def _get_or_create_arena(
        self,
        layer_name: str,
        segment: RetroSpecResidentSegment,
    ) -> _PackedLayerState:
        layer_state = self._layer_arenas.get(layer_name)
        if layer_state is not None:
            arena = layer_state.arena
            if arena.cluster_keys.device != segment.cluster_keys.device:
                raise RuntimeError("Resident layer arena changed device")
            if arena.cluster_keys.dtype != segment.cluster_keys.dtype:
                raise RuntimeError("Resident layer arena changed dtype")
            if arena.cluster_keys.shape[0] != segment.cluster_keys.shape[0]:
                raise RuntimeError("Resident layer arena changed KV-head count")
            if arena.cluster_keys.shape[2] != segment.cluster_keys.shape[2]:
                raise RuntimeError("Resident layer arena changed head size")
            return layer_state

        num_kv_heads, _, head_size = segment.cluster_keys.shape
        device = segment.cluster_keys.device
        dtype = segment.cluster_keys.dtype

        cluster_shape = (num_kv_heads, 0)
        summary_shape = (*cluster_shape, head_size)
        page_shape = (num_kv_heads, 0)
        request_shape = (self.max_resident_requests,)

        arena = RetroSpecResidentLayerArena(
            cluster_ids=torch.empty(cluster_shape, dtype=torch.int64, device=device),
            cluster_keys=torch.empty(summary_shape, dtype=dtype, device=device),
            cluster_values=torch.empty(summary_shape, dtype=dtype, device=device),
            cluster_token_counts=torch.empty(
                cluster_shape, dtype=torch.int32, device=device
            ),
            cluster_page_starts=torch.empty(
                cluster_shape, dtype=torch.int64, device=device
            ),
            cluster_page_counts=torch.empty(
                cluster_shape, dtype=torch.int32, device=device
            ),
            page_ids=torch.empty(page_shape, dtype=torch.int64, device=device),
            page_token_counts=torch.empty(page_shape, dtype=torch.int32, device=device),
            cluster_offsets=torch.zeros(
                request_shape, dtype=torch.int64, device=device
            ),
            num_clusters=torch.zeros(request_shape, dtype=torch.int32, device=device),
            page_offsets=torch.zeros(request_shape, dtype=torch.int64, device=device),
            num_pages=torch.zeros(
                (self.max_resident_requests, num_kv_heads),
                dtype=torch.int32,
                device=device,
            ),
            generations=torch.zeros(request_shape, dtype=torch.int64, device=device),
            indexed_starts=torch.zeros(request_shape, dtype=torch.int64, device=device),
            indexed_ends=torch.zeros(request_shape, dtype=torch.int64, device=device),
        )
        descriptor_bytes = sum(
            self._tensor_bytes(tensor)
            for tensor in (
                arena.cluster_offsets,
                arena.num_clusters,
                arena.page_offsets,
                arena.num_pages,
                arena.generations,
                arena.indexed_starts,
                arena.indexed_ends,
            )
        )
        self._reserve_gpu_index_bytes(descriptor_bytes)
        layer_state = _PackedLayerState(
            arena=arena,
            cluster_allocator=_FreeSpanAllocator(),
            page_allocator=_FreeSpanAllocator(),
        )
        self._layer_arenas[layer_name] = layer_state
        return layer_state

    def _grow_cluster_storage(
        self,
        layer_state: _PackedLayerState,
        required_span: int,
    ) -> None:
        arena = layer_state.arena
        allocator = layer_state.cluster_allocator
        old_capacity = allocator.capacity
        new_capacity = self._next_power_of_two(
            max(64, old_capacity * 2, old_capacity + required_span)
        )
        num_kv_heads, _, head_size = arena.cluster_keys.shape
        added_capacity = new_capacity - old_capacity
        added_bytes = (
            num_kv_heads
            * added_capacity
            * (24 + 2 * head_size * arena.cluster_keys.element_size())
        )
        self._reserve_gpu_index_bytes(added_bytes)

        try:
            cluster_ids = torch.empty(
                (num_kv_heads, new_capacity),
                dtype=arena.cluster_ids.dtype,
                device=arena.cluster_ids.device,
            )
            cluster_keys = torch.empty(
                (num_kv_heads, new_capacity, head_size),
                dtype=arena.cluster_keys.dtype,
                device=arena.cluster_keys.device,
            )
            cluster_values = torch.empty_like(cluster_keys)
            cluster_token_counts = torch.empty(
                (num_kv_heads, new_capacity),
                dtype=arena.cluster_token_counts.dtype,
                device=arena.cluster_token_counts.device,
            )
            cluster_page_starts = torch.empty(
                (num_kv_heads, new_capacity),
                dtype=arena.cluster_page_starts.dtype,
                device=arena.cluster_page_starts.device,
            )
            cluster_page_counts = torch.empty(
                (num_kv_heads, new_capacity),
                dtype=arena.cluster_page_counts.dtype,
                device=arena.cluster_page_counts.device,
            )
            if old_capacity:
                cluster_ids[:, :old_capacity].copy_(arena.cluster_ids)
                cluster_keys[:, :old_capacity].copy_(arena.cluster_keys)
                cluster_values[:, :old_capacity].copy_(arena.cluster_values)
                cluster_token_counts[:, :old_capacity].copy_(arena.cluster_token_counts)
                cluster_page_starts[:, :old_capacity].copy_(arena.cluster_page_starts)
                cluster_page_counts[:, :old_capacity].copy_(arena.cluster_page_counts)
        except BaseException:
            self._allocated_gpu_index_bytes -= added_bytes
            raise

        arena.cluster_ids = cluster_ids
        arena.cluster_keys = cluster_keys
        arena.cluster_values = cluster_values
        arena.cluster_token_counts = cluster_token_counts
        arena.cluster_page_starts = cluster_page_starts
        arena.cluster_page_counts = cluster_page_counts
        allocator.extend(new_capacity)

    def _grow_page_storage(
        self,
        layer_state: _PackedLayerState,
        required_span: int,
    ) -> None:
        arena = layer_state.arena
        allocator = layer_state.page_allocator
        old_capacity = allocator.capacity
        new_capacity = self._next_power_of_two(
            max(64, old_capacity * 2, old_capacity + required_span)
        )
        num_kv_heads = arena.page_ids.shape[0]
        added_capacity = new_capacity - old_capacity
        added_bytes = num_kv_heads * added_capacity * 12
        self._reserve_gpu_index_bytes(added_bytes)

        try:
            page_ids = torch.empty(
                (num_kv_heads, new_capacity),
                dtype=arena.page_ids.dtype,
                device=arena.page_ids.device,
            )
            page_token_counts = torch.empty(
                (num_kv_heads, new_capacity),
                dtype=arena.page_token_counts.dtype,
                device=arena.page_token_counts.device,
            )
            if old_capacity:
                page_ids[:, :old_capacity].copy_(arena.page_ids)
                page_token_counts[:, :old_capacity].copy_(arena.page_token_counts)
        except BaseException:
            self._allocated_gpu_index_bytes -= added_bytes
            raise

        arena.page_ids = page_ids
        arena.page_token_counts = page_token_counts
        allocator.extend(new_capacity)

    def _allocate_cluster_span(
        self,
        layer_state: _PackedLayerState,
        size: int,
    ) -> int:
        offset = layer_state.cluster_allocator.allocate(size)
        if offset is None:
            self._grow_cluster_storage(layer_state, size)
            offset = layer_state.cluster_allocator.allocate(size)
        assert offset is not None
        return offset

    def _allocate_page_span(
        self,
        layer_state: _PackedLayerState,
        size: int,
    ) -> int:
        offset = layer_state.page_allocator.allocate(size)
        if offset is None:
            self._grow_page_storage(layer_state, size)
            offset = layer_state.page_allocator.allocate(size)
        assert offset is not None
        return offset

    @staticmethod
    def _pad_page_descriptors(
        tensor: torch.Tensor,
        width: int,
        fill_value: int,
    ) -> torch.Tensor:
        if tensor.shape[-1] == width:
            return tensor

        padded = torch.full(
            (*tensor.shape[:-1], width),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[..., : tensor.shape[-1]].copy_(tensor)
        return padded

    def _coalesce_resident_segments(
        self,
        segments: Sequence[RetroSpecResidentSegment],
    ) -> list[RetroSpecResidentSegment]:
        grouped: dict[tuple[str, str], list[RetroSpecResidentSegment]] = {}
        for segment in segments:
            grouped.setdefault((segment.layer_name, segment.request_id), []).append(
                segment
            )

        merged_segments: list[RetroSpecResidentSegment] = []
        for grouped_segments in grouped.values():
            if len(grouped_segments) == 1:
                merged_segments.append(grouped_segments[0])
                continue

            first = grouped_segments[0]
            previous = first
            for segment in grouped_segments[1:]:
                if segment.indexed_start != previous.indexed_end:
                    raise RuntimeError(
                        "Resident segment does not follow the indexed token prefix"
                    )
                expected_cluster_start = (
                    previous.cluster_start + previous.cluster_ids_cpu.shape[1]
                )
                if segment.cluster_start != expected_cluster_start:
                    raise RuntimeError(
                        "Resident segment does not follow the cluster prefix"
                    )
                segment_summary_shape = (
                    segment.cluster_keys.shape[0],
                    segment.cluster_keys.shape[2],
                )
                first_summary_shape = (
                    first.cluster_keys.shape[0],
                    first.cluster_keys.shape[2],
                )
                if segment_summary_shape != first_summary_shape:
                    raise RuntimeError("Resident segment changed cluster summary shape")
                previous = segment

            page_width = max(
                segment.cluster_page_ids_cpu.shape[-1] for segment in grouped_segments
            )
            merged_segments.append(
                RetroSpecResidentSegment(
                    layer_name=first.layer_name,
                    request_id=first.request_id,
                    indexed_start=first.indexed_start,
                    indexed_end=grouped_segments[-1].indexed_end,
                    cluster_start=first.cluster_start,
                    cluster_ids_cpu=torch.cat(
                        [segment.cluster_ids_cpu for segment in grouped_segments],
                        dim=1,
                    ),
                    cluster_keys=torch.cat(
                        [segment.cluster_keys for segment in grouped_segments], dim=1
                    ),
                    cluster_values=torch.cat(
                        [segment.cluster_values for segment in grouped_segments], dim=1
                    ),
                    cluster_token_counts=torch.cat(
                        [segment.cluster_token_counts for segment in grouped_segments],
                        dim=1,
                    ),
                    cluster_page_ids_cpu=torch.cat(
                        [
                            self._pad_page_descriptors(
                                segment.cluster_page_ids_cpu, page_width, -1
                            )
                            for segment in grouped_segments
                        ],
                        dim=1,
                    ),
                    cluster_page_token_counts_cpu=torch.cat(
                        [
                            self._pad_page_descriptors(
                                segment.cluster_page_token_counts_cpu, page_width, 0
                            )
                            for segment in grouped_segments
                        ],
                        dim=1,
                    ),
                    cluster_page_counts_cpu=torch.cat(
                        [
                            segment.cluster_page_counts_cpu
                            for segment in grouped_segments
                        ],
                        dim=1,
                    ),
                )
            )

        return merged_segments

    def build_resident_segment(
        self,
        layer_name: str,
        request_id: str,
        indexed_start: int,
        indexed_end: int,
        cluster_start: int,
        staged_summary: RetroSpecStagedClusterSummary,
        cluster_ids: torch.Tensor,
        cluster_page_ids: torch.Tensor,
        cluster_page_token_counts: torch.Tensor,
    ) -> RetroSpecResidentSegment:
        if indexed_start < 0 or indexed_end <= indexed_start:
            raise ValueError("Resident segment token range is invalid")
        if cluster_start < 0:
            raise ValueError("Resident segment cluster offset must be non-negative")

        summary = staged_summary.resident_summary
        device = summary.cluster_keys.device

        if summary.cluster_keys.shape != summary.cluster_values.shape:
            raise ValueError("Resident cluster key/value shapes must match")
        if summary.cluster_keys.ndim != 3:
            raise ValueError(
                "Resident cluster summaries must have shape "
                "[num_kv_heads, num_clusters, head_size]"
            )
        if summary.cluster_token_counts.shape != summary.cluster_keys.shape[:2]:
            raise ValueError("Resident cluster counts do not match summaries")
        if cluster_ids.shape != summary.cluster_token_counts.shape:
            raise ValueError("Resident cluster IDs do not match cluster counts")
        if cluster_page_ids.shape != cluster_page_token_counts.shape:
            raise ValueError("Resident cluster page metadata shapes must match")
        if cluster_page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Resident cluster pages do not match cluster IDs")

        cluster_page_ids_cpu = cluster_page_ids.to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        cluster_page_token_counts_cpu = cluster_page_token_counts.to(
            device="cpu", dtype=torch.int32
        ).contiguous()
        cluster_ids_cpu = cluster_ids.to(device="cpu", dtype=torch.int64)
        cluster_token_counts_cpu = staged_summary.cluster_token_counts.to(
            device="cpu", dtype=torch.int32
        )
        valid_clusters = cluster_ids_cpu >= 0
        positive_clusters = cluster_token_counts_cpu > 0
        valid_pages = cluster_page_ids_cpu >= 0
        positive_pages = cluster_page_token_counts_cpu > 0

        if not torch.equal(valid_clusters, positive_clusters):
            raise ValueError(
                "Resident cluster IDs and token counts describe different clusters"
            )
        if not torch.equal(valid_pages, positive_pages):
            raise ValueError(
                "Resident cluster page IDs and token counts describe different pages"
            )

        cluster_page_counts_cpu = (cluster_page_ids_cpu >= 0).sum(
            dim=-1, dtype=torch.int32
        )
        if not torch.equal(cluster_page_counts_cpu > 0, valid_clusters):
            raise ValueError("Resident clusters and page descriptors do not match")
        if not torch.equal(
            cluster_page_token_counts_cpu.sum(dim=-1),
            cluster_token_counts_cpu,
        ):
            raise ValueError(
                "Resident cluster token counts do not match page descriptors"
            )

        resident_token_counts = summary.cluster_token_counts.to(
            device=device, dtype=torch.int32
        ).contiguous()

        return RetroSpecResidentSegment(
            layer_name=layer_name,
            request_id=request_id,
            indexed_start=indexed_start,
            indexed_end=indexed_end,
            cluster_start=cluster_start,
            cluster_ids_cpu=cluster_ids_cpu.contiguous(),
            cluster_keys=summary.cluster_keys.contiguous(),
            cluster_values=summary.cluster_values.contiguous(),
            cluster_token_counts=resident_token_counts,
            cluster_page_ids_cpu=cluster_page_ids_cpu,
            cluster_page_token_counts_cpu=cluster_page_token_counts_cpu,
            cluster_page_counts_cpu=cluster_page_counts_cpu.contiguous(),
        )

    def _write_resident_segment(
        self,
        layer_state: _PackedLayerState,
        segment: RetroSpecResidentSegment,
        slot: int,
        previous_state: _ResidentRequestState | None,
    ) -> tuple[_ResidentRequestState, _ResidentSpanTransaction]:
        arena = layer_state.arena
        num_kv_heads, num_segment_clusters = segment.cluster_ids_cpu.shape
        cluster_start = segment.cluster_start
        cluster_end = cluster_start + num_segment_clusters

        if previous_state is None:
            if cluster_start != 0:
                raise RuntimeError(
                    "The first resident segment must start at cluster zero"
                )
            page_counts = (0,) * num_kv_heads
            indexed_start = segment.indexed_start
            previous_indexed_end = segment.indexed_start
            previous_num_clusters = 0
            max_pages_per_cluster = 0
            generation = self._slot_generations[slot]
        else:
            if len(previous_state.page_counts) != num_kv_heads:
                raise RuntimeError("Resident segment changed KV-head count")
            page_counts = previous_state.page_counts
            indexed_start = previous_state.indexed_start
            previous_indexed_end = previous_state.indexed_end
            previous_num_clusters = previous_state.num_clusters
            max_pages_per_cluster = previous_state.max_pages_per_cluster
            generation = previous_state.generation

        if previous_indexed_end != segment.indexed_start:
            raise RuntimeError(
                "Resident segment does not follow the indexed token prefix"
            )
        if previous_num_clusters != cluster_start:
            raise RuntimeError("Resident segment does not follow the cluster prefix")

        next_page_counts = tuple(
            page_counts[head_index]
            + int(segment.cluster_page_counts_cpu[head_index].sum().item())
            for head_index in range(num_kv_heads)
        )
        required_cluster_capacity = self._next_power_of_two(cluster_end)
        required_page_capacity = self._next_power_of_two(max(next_page_counts))

        new_cluster_span = None
        old_cluster_span = None
        new_page_span = None
        old_page_span = None

        if previous_state is None:
            cluster_offset = self._allocate_cluster_span(
                layer_state, required_cluster_capacity
            )
            new_cluster_span = (cluster_offset, required_cluster_capacity)
            try:
                page_offset = self._allocate_page_span(
                    layer_state, required_page_capacity
                )
            except BaseException:
                layer_state.cluster_allocator.release(*new_cluster_span)
                raise
            new_page_span = (page_offset, required_page_capacity)
            cluster_capacity = required_cluster_capacity
            page_capacity = required_page_capacity
        else:
            cluster_offset = previous_state.cluster_offset
            cluster_capacity = previous_state.cluster_capacity
            page_offset = previous_state.page_offset
            page_capacity = previous_state.page_capacity

            if required_cluster_capacity > cluster_capacity:
                new_offset = self._allocate_cluster_span(
                    layer_state, required_cluster_capacity
                )
                new_cluster_span = (new_offset, required_cluster_capacity)
                old_cluster_span = (cluster_offset, cluster_capacity)
                try:
                    source = slice(
                        cluster_offset, cluster_offset + previous_num_clusters
                    )
                    destination = slice(new_offset, new_offset + previous_num_clusters)
                    arena.cluster_ids[:, destination].copy_(
                        arena.cluster_ids[:, source]
                    )
                    arena.cluster_keys[:, destination].copy_(
                        arena.cluster_keys[:, source]
                    )
                    arena.cluster_values[:, destination].copy_(
                        arena.cluster_values[:, source]
                    )
                    arena.cluster_token_counts[:, destination].copy_(
                        arena.cluster_token_counts[:, source]
                    )
                    arena.cluster_page_starts[:, destination].copy_(
                        arena.cluster_page_starts[:, source]
                    )
                    arena.cluster_page_counts[:, destination].copy_(
                        arena.cluster_page_counts[:, source]
                    )
                except BaseException:
                    layer_state.cluster_allocator.release(*new_cluster_span)
                    raise
                cluster_offset = new_offset
                cluster_capacity = required_cluster_capacity

            if required_page_capacity > page_capacity:
                try:
                    new_offset = self._allocate_page_span(
                        layer_state, required_page_capacity
                    )
                except BaseException:
                    if new_cluster_span is not None:
                        layer_state.cluster_allocator.release(*new_cluster_span)
                    raise
                new_page_span = (new_offset, required_page_capacity)
                old_page_span = (page_offset, page_capacity)
                try:
                    for head_index, head_page_count in enumerate(page_counts):
                        source = slice(page_offset, page_offset + head_page_count)
                        destination = slice(new_offset, new_offset + head_page_count)
                        arena.page_ids[head_index, destination].copy_(
                            arena.page_ids[head_index, source]
                        )
                        arena.page_token_counts[head_index, destination].copy_(
                            arena.page_token_counts[head_index, source]
                        )
                except BaseException:
                    layer_state.page_allocator.release(*new_page_span)
                    if new_cluster_span is not None:
                        layer_state.cluster_allocator.release(*new_cluster_span)
                    raise
                page_offset = new_offset
                page_capacity = required_page_capacity

        try:
            absolute_cluster_start = cluster_offset + cluster_start
            absolute_cluster_end = cluster_offset + cluster_end
            cluster_slice = slice(absolute_cluster_start, absolute_cluster_end)
            arena.cluster_ids[:, cluster_slice].copy_(segment.cluster_ids_cpu)
            arena.cluster_keys[:, cluster_slice].copy_(segment.cluster_keys)
            arena.cluster_values[:, cluster_slice].copy_(segment.cluster_values)
            arena.cluster_token_counts[:, cluster_slice].copy_(
                segment.cluster_token_counts
            )

            for head_index, page_start in enumerate(page_counts):
                segment_page_counts = segment.cluster_page_counts_cpu[head_index]
                page_starts_cpu = torch.empty(num_segment_clusters, dtype=torch.int64)
                if num_segment_clusters:
                    page_starts_cpu[0] = page_start
                if num_segment_clusters > 1:
                    torch.cumsum(
                        segment_page_counts[:-1].to(torch.int64),
                        dim=0,
                        out=page_starts_cpu[1:],
                    )
                    page_starts_cpu[1:].add_(page_start)

                arena.cluster_page_starts[head_index, cluster_slice].copy_(
                    page_starts_cpu, non_blocking=self.pin_memory
                )
                arena.cluster_page_counts[head_index, cluster_slice].copy_(
                    segment_page_counts, non_blocking=self.pin_memory
                )

                valid_pages = segment.cluster_page_ids_cpu[head_index] >= 0
                flat_page_ids = segment.cluster_page_ids_cpu[head_index].masked_select(
                    valid_pages
                )
                flat_page_token_counts = segment.cluster_page_token_counts_cpu[
                    head_index
                ].masked_select(valid_pages)

                absolute_page_start = page_offset + page_start
                absolute_page_end = absolute_page_start + flat_page_ids.numel()
                arena.page_ids[head_index, absolute_page_start:absolute_page_end].copy_(
                    flat_page_ids
                )
                arena.page_token_counts[
                    head_index, absolute_page_start:absolute_page_end
                ].copy_(flat_page_token_counts)
        except BaseException:
            if new_page_span is not None:
                layer_state.page_allocator.release(*new_page_span)
            if new_cluster_span is not None:
                layer_state.cluster_allocator.release(*new_cluster_span)
            raise

        segment_max_pages = int(segment.cluster_page_counts_cpu.max().item())
        state = _ResidentRequestState(
            slot=slot,
            generation=generation,
            indexed_start=indexed_start,
            indexed_end=segment.indexed_end,
            cluster_offset=cluster_offset,
            cluster_capacity=cluster_capacity,
            num_clusters=cluster_end,
            page_offset=page_offset,
            page_capacity=page_capacity,
            page_counts=next_page_counts,
            max_pages_per_cluster=max(max_pages_per_cluster, segment_max_pages),
        )
        transaction = _ResidentSpanTransaction(
            layer_state=layer_state,
            new_cluster_span=new_cluster_span,
            old_cluster_span=old_cluster_span,
            new_page_span=new_page_span,
            old_page_span=old_page_span,
        )
        return state, transaction

    @staticmethod
    def _rollback_span_transaction(transaction: _ResidentSpanTransaction) -> None:
        if transaction.new_page_span is not None:
            transaction.layer_state.page_allocator.release(*transaction.new_page_span)
        if transaction.new_cluster_span is not None:
            transaction.layer_state.cluster_allocator.release(
                *transaction.new_cluster_span
            )

    @staticmethod
    def _commit_span_transaction(transaction: _ResidentSpanTransaction) -> None:
        if transaction.old_page_span is not None:
            transaction.layer_state.page_allocator.release(*transaction.old_page_span)
        if transaction.old_cluster_span is not None:
            transaction.layer_state.cluster_allocator.release(
                *transaction.old_cluster_span
            )

    def publish_resident_segments(
        self,
        segments: Sequence[RetroSpecResidentSegment],
    ) -> None:
        if not segments:
            return

        incoming_request_ids = {segment.request_id for segment in segments}
        resident_request_ids = set(self.resident_request_ids)
        new_request_ids = resident_request_ids | incoming_request_ids
        if len(new_request_ids) > self.max_resident_requests:
            raise RuntimeError(
                "RetroSpec persistent GPU index residency exceeds max_num_seqs: "
                f"{len(new_request_ids)} > {self.max_resident_requests}"
            )

        projected_states = {
            layer_name: dict(layer_states)
            for layer_name, layer_states in self._resident_states.items()
        }
        changed_layers: set[str] = set()
        allocated_request_ids: list[str] = []
        transactions: list[_ResidentSpanTransaction] = []

        try:
            for segment in self._coalesce_resident_segments(segments):
                if segment.request_id not in self._request_slots:
                    allocated_request_ids.append(segment.request_id)
                slot = self._get_or_allocate_request_slot(segment.request_id)
                layer_state = self._get_or_create_arena(segment.layer_name, segment)
                layer_states = projected_states.setdefault(segment.layer_name, {})
                previous_state = layer_states.get(segment.request_id)
                state, transaction = self._write_resident_segment(
                    layer_state=layer_state,
                    segment=segment,
                    slot=slot,
                    previous_state=previous_state,
                )
                layer_states[segment.request_id] = state
                transactions.append(transaction)
                changed_layers.add(segment.layer_name)
        except BaseException:
            for transaction in reversed(transactions):
                self._rollback_span_transaction(transaction)
            for request_id in reversed(allocated_request_ids):
                slot = self._request_slots.pop(request_id, None)
                if slot is not None:
                    self._free_request_slots.append(slot)
            raise

        for transaction in transactions:
            self._commit_span_transaction(transaction)
        self._resident_states = projected_states
        for layer_name in changed_layers:
            arena = self._layer_arenas[layer_name].arena
            for state in self._resident_states[layer_name].values():
                arena.cluster_offsets[state.slot] = state.cluster_offset
                arena.num_clusters[state.slot] = state.num_clusters
                arena.page_offsets[state.slot] = state.page_offset
                arena.num_pages[state.slot].copy_(
                    torch.tensor(
                        state.page_counts,
                        dtype=torch.int32,
                        device=arena.num_pages.device,
                    )
                )
                arena.generations[state.slot] = state.generation
                arena.indexed_starts[state.slot] = state.indexed_start
                arena.indexed_ends[state.slot] = state.indexed_end
            self._active_views.pop(layer_name, None)

    def get_active_view(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        device: torch.device,
    ) -> RetroSpecResidentBatchView:
        request_ids = tuple(request_ids)
        self._validate_active_requests(request_ids)

        cached = self._active_views.get(layer_name)
        if cached is not None and cached.request_slot_ids.device == device:
            return cached

        layer_states = self._resident_states.get(layer_name, {})
        request_slots = [
            -1 if (state := layer_states.get(request_id)) is None else state.slot
            for request_id in request_ids
        ]
        layer_state = self._layer_arenas.get(layer_name)
        arena = (
            None
            if layer_state is None or all(slot < 0 for slot in request_slots)
            else layer_state.arena
        )
        request_slot_ids = torch.tensor(request_slots, dtype=torch.int64, device=device)
        max_num_clusters = max(
            (
                state.num_clusters
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )
        max_pages_per_cluster = max(
            (
                state.max_pages_per_cluster
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )
        max_num_pages = max(
            (
                max(state.page_counts, default=0)
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )

        view = RetroSpecResidentBatchView(
            arena=arena,
            request_slot_ids=request_slot_ids,
            max_num_clusters=max(max_num_clusters, 1),
            max_pages_per_cluster=max_pages_per_cluster,
            max_num_pages=max_num_pages,
        )
        self._active_views[layer_name] = view
        return view

    def get_num_clusters(self, layer_name: str, request_id: str) -> int:
        state = self._resident_states.get(layer_name, {}).get(request_id)
        return 0 if state is None else state.num_clusters

    def get_indexed_end(self, layer_name: str, request_id: str) -> int | None:
        state = self._resident_states.get(layer_name, {}).get(request_id)
        return None if state is None else state.indexed_end

    def invalidate_active_view(self, layer_name: str) -> None:
        self._active_views.pop(layer_name, None)

    def _release_request_state(
        self,
        layer_name: str,
        state: _ResidentRequestState,
    ) -> None:
        layer_state = self._layer_arenas[layer_name]
        layer_state.cluster_allocator.release(
            state.cluster_offset, state.cluster_capacity
        )
        layer_state.page_allocator.release(state.page_offset, state.page_capacity)

        arena = layer_state.arena
        arena.cluster_offsets[state.slot] = 0
        arena.num_clusters[state.slot] = 0
        arena.page_offsets[state.slot] = 0
        arena.num_pages[state.slot].zero_()
        arena.generations[state.slot] = 0
        arena.indexed_starts[state.slot] = 0
        arena.indexed_ends[state.slot] = 0

    def discard_request_layer(self, layer_name: str, request_id: str) -> None:
        layer_states = self._resident_states.get(layer_name)
        if layer_states is None:
            return

        state = layer_states.pop(request_id, None)
        if state is None:
            return

        self._release_request_state(layer_name, state)
        self._active_views.pop(layer_name, None)

    def invalidate_requests(self, request_ids: Sequence[str]) -> None:
        removed = set(request_ids)
        if not removed:
            return

        for layer_name, layer_states in self._resident_states.items():
            for request_id in removed:
                state = layer_states.pop(request_id, None)
                if state is None:
                    continue
                self._release_request_state(layer_name, state)

        self._active_views.clear()

        for request_id in removed:
            slot = self._request_slots.pop(request_id, None)
            if slot is not None:
                self._free_request_slots.append(slot)

    def _get_offload_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        stream = self._offload_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._offload_streams[device] = stream

        return stream

    def stage_cluster_summary(
        self,
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        cluster_token_counts: torch.Tensor,
    ) -> RetroSpecStagedClusterSummary:
        if cluster_keys.shape != cluster_values.shape:
            raise ValueError("Cluster key/value summary shapes must match")
        if cluster_keys.ndim != 3:
            raise ValueError(
                "Cluster summaries must have shape "
                "[num_kv_heads, num_clusters, head_size]"
            )
        if cluster_token_counts.shape != cluster_keys.shape[:2]:
            raise ValueError("Cluster token counts do not match cluster summaries")

        resident_summary = RetroSpecClusterSummary(
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            cluster_token_counts=cluster_token_counts,
        )

        host_keys = torch.empty_like(
            cluster_keys, device="cpu", pin_memory=self.pin_memory
        )
        host_values = torch.empty_like(
            cluster_values, device="cpu", pin_memory=self.pin_memory
        )
        host_counts = torch.empty_like(
            cluster_token_counts, device="cpu", pin_memory=self.pin_memory
        )

        if cluster_keys.device.type != "cuda":
            host_keys.copy_(cluster_keys)
            host_values.copy_(cluster_values)
            host_counts.copy_(cluster_token_counts)
            return RetroSpecStagedClusterSummary(
                cluster_keys=host_keys,
                cluster_values=host_values,
                cluster_token_counts=host_counts,
                resident_summary=resident_summary,
                ready_event=None,
            )

        if not self.pin_memory:
            host_keys.copy_(cluster_keys, non_blocking=False)
            host_values.copy_(cluster_values, non_blocking=False)
            host_counts.copy_(cluster_token_counts, non_blocking=False)
            return RetroSpecStagedClusterSummary(
                cluster_keys=host_keys,
                cluster_values=host_values,
                cluster_token_counts=host_counts,
                resident_summary=resident_summary,
                ready_event=None,
            )

        device = cluster_keys.device
        transfer_stream = self._get_offload_stream(device)
        current_stream = torch.cuda.current_stream(device)
        transfer_stream.wait_stream(current_stream)

        with torch.cuda.stream(transfer_stream):
            host_keys.copy_(cluster_keys, non_blocking=True)
            host_values.copy_(cluster_values, non_blocking=True)
            host_counts.copy_(cluster_token_counts, non_blocking=True)
            ready_event = torch.cuda.Event()
            ready_event.record(transfer_stream)

        return RetroSpecStagedClusterSummary(
            cluster_keys=host_keys,
            cluster_values=host_values,
            cluster_token_counts=host_counts,
            resident_summary=resident_summary,
            ready_event=ready_event,
        )

    @staticmethod
    def finish_cluster_summary(
        staged: RetroSpecStagedClusterSummary,
    ) -> RetroSpecClusterSummary:
        if staged.ready_event is not None:
            staged.ready_event.synchronize()

        return RetroSpecClusterSummary(
            cluster_keys=staged.cluster_keys,
            cluster_values=staged.cluster_values,
            cluster_token_counts=staged.cluster_token_counts,
        )

    @staticmethod
    def discard_cluster_summary(staged: RetroSpecStagedClusterSummary) -> None:
        if staged.ready_event is not None:
            staged.ready_event.synchronize()
