# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.v1.spec_decode.retrospec.legacy.residency.index.types import (
    RetroSpecResidentLayerArena,
    RetroSpecResidentSegment,
    _FreeSpanAllocator,
    _PackedLayerState,
)


class _IndexResidencyArenaMixin:
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
            resident_table_buckets=torch.empty(
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
            * (28 + 2 * head_size * arena.cluster_keys.element_size())
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
            resident_table_buckets = torch.full(
                (num_kv_heads, new_capacity),
                -1,
                dtype=torch.int32,
                device=arena.cluster_ids.device,
            )
            if old_capacity:
                cluster_ids[:, :old_capacity].copy_(arena.cluster_ids)
                cluster_keys[:, :old_capacity].copy_(arena.cluster_keys)
                cluster_values[:, :old_capacity].copy_(arena.cluster_values)
                cluster_token_counts[:, :old_capacity].copy_(arena.cluster_token_counts)
                cluster_page_starts[:, :old_capacity].copy_(arena.cluster_page_starts)
                cluster_page_counts[:, :old_capacity].copy_(arena.cluster_page_counts)
                resident_table_buckets[:, :old_capacity].copy_(
                    arena.resident_table_buckets
                )
        except BaseException:
            self._allocated_gpu_index_bytes -= added_bytes
            raise

        arena.cluster_ids = cluster_ids
        arena.cluster_keys = cluster_keys
        arena.cluster_values = cluster_values
        arena.cluster_token_counts = cluster_token_counts
        arena.cluster_page_starts = cluster_page_starts
        arena.cluster_page_counts = cluster_page_counts
        arena.resident_table_buckets = resident_table_buckets
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
