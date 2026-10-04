# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import torch

from .pinned_memory import RetroSpecPinnedMemoryManager


@dataclass(frozen=True)
class RetroSpecClusterSummary:
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor


@dataclass
class _PinnedSummarySlot:
    pinned_memory: RetroSpecPinnedMemoryManager

    key_storage: torch.Tensor | None = None
    value_storage: torch.Tensor | None = None
    count_storage: torch.Tensor | None = None
    in_use: bool = False

    def _reserve(
        self,
        storage: torch.Tensor | None,
        source: torch.Tensor,
        label: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required_numel = source.numel()
        if (
            storage is None
            or storage.dtype != source.dtype
            or storage.numel() < required_numel
        ):
            storage = self.pinned_memory.replace(
                storage,
                (required_numel,),
                source.dtype,
                label,
            )
        return storage, storage[:required_numel].view(source.shape)

    def reserve(
        self,
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        cluster_token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.in_use:
            raise RuntimeError("Pinned summary slot must be acquired before use")

        self.key_storage, host_keys = self._reserve(
            self.key_storage, cluster_keys, "cluster-summary-keys"
        )
        self.value_storage, host_values = self._reserve(
            self.value_storage, cluster_values, "cluster-summary-values"
        )
        self.count_storage, host_counts = self._reserve(
            self.count_storage, cluster_token_counts, "cluster-summary-counts"
        )
        return host_keys, host_values, host_counts

    def release_storage(self) -> None:
        self.pinned_memory.release(self.key_storage)
        self.pinned_memory.release(self.value_storage)
        self.pinned_memory.release(self.count_storage)
        self.key_storage = None
        self.value_storage = None
        self.count_storage = None


@dataclass(frozen=True)
class RetroSpecStagedClusterSummary:
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    resident_summary: RetroSpecClusterSummary
    ready_event: torch.cuda.Event | None
    staging_slot: _PinnedSummarySlot | None = field(
        default=None,
        repr=False,
        compare=False,
    )


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


@dataclass(frozen=True)
class RetroSpecResidentTableBinding:
    """Validated route from one stable cluster handle to an arena binding."""

    request_id: str
    kv_head_index: int
    local_cluster_index: int
    cluster_handle: int
    table_bucket: int


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
    resident_table_buckets: torch.Tensor

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


# Keep serialized class paths compatible with the original public module.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = __name__.replace(
            ".offload.index_residency_types", ".index_residency"
        )
del _legacy_type
