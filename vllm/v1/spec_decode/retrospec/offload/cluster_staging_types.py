# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from .pinned_memory import RetroSpecPinnedMemoryManager

RetroSpecResidentPrefetchSource = Literal["prefill_hint", "draft"]

_PREFETCH_SOURCE_PRIORITY: dict[RetroSpecResidentPrefetchSource, int] = {
    "prefill_hint": 0,
    "draft": 1,
}


@dataclass
class _PinnedStagingSlot:
    """Reusable pinned CPU buffers for one in-flight cluster build."""

    source_device: torch.device
    max_bytes: int
    pinned_memory: RetroSpecPinnedMemoryManager

    token_key_storage: torch.Tensor | None = None
    token_value_storage: torch.Tensor | None = None
    assignment_storage: torch.Tensor | None = None
    cluster_count_storage: torch.Tensor | None = None
    token_offset_storage: torch.Tensor | None = None

    in_use: bool = False

    def _retained_bytes(self, excluded: torch.Tensor | None = None) -> int:
        storages = (
            self.token_key_storage,
            self.token_value_storage,
            self.assignment_storage,
            self.cluster_count_storage,
            self.token_offset_storage,
        )
        return sum(
            storage.nbytes
            for storage in storages
            if storage is not None and storage is not excluded
        )

    def _reserve(
        self,
        storage: torch.Tensor | None,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required_numel = source.numel()
        if (
            storage is None
            or storage.dtype != source.dtype
            or storage.numel() < required_numel
        ):
            required_bytes = required_numel * source.element_size()
            if self._retained_bytes(excluded=storage) + required_bytes > self.max_bytes:
                raise RuntimeError(
                    "RetroSpec cluster-build staging exceeds its pinned-memory "
                    "slot budget; reduce the clustering segment size or increase "
                    "retrospec_max_pinned_memory"
                )
            storage = self.pinned_memory.replace(
                storage,
                (required_numel,),
                source.dtype,
                "cluster-build-staging",
            )

        view = storage[:required_numel].view(source.shape)
        return storage, view

    def reserve_token_kv(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.in_use:
            raise RuntimeError("Pinned staging slot must be acquired before use")

        self.token_key_storage, staged_token_keys = self._reserve(
            self.token_key_storage,
            token_keys,
        )
        self.token_value_storage, staged_token_values = self._reserve(
            self.token_value_storage,
            token_values,
        )
        return staged_token_keys, staged_token_values

    def reserve_cluster_metadata(
        self,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.in_use:
            raise RuntimeError("Pinned staging slot must be acquired before use")

        self.assignment_storage, staged_assignments = self._reserve(
            self.assignment_storage,
            assignments,
        )
        self.cluster_count_storage, staged_cluster_token_counts = self._reserve(
            self.cluster_count_storage,
            cluster_token_counts,
        )
        self.token_offset_storage, staged_token_offsets = self._reserve(
            self.token_offset_storage,
            token_offsets_in_cluster,
        )
        return (
            staged_assignments,
            staged_cluster_token_counts,
            staged_token_offsets,
        )

    def release_storage(self) -> None:
        self.pinned_memory.release(self.token_key_storage)
        self.pinned_memory.release(self.token_value_storage)
        self.pinned_memory.release(self.assignment_storage)
        self.pinned_memory.release(self.cluster_count_storage)
        self.pinned_memory.release(self.token_offset_storage)
        self.token_key_storage = None
        self.token_value_storage = None
        self.assignment_storage = None
        self.cluster_count_storage = None
        self.token_offset_storage = None


@dataclass(frozen=True)
class RetroSpecResidentPrefetchInput:
    """One layer's GPU-compacted resident miss commands."""

    layer_name: str
    miss_cluster_ids: torch.Tensor
    miss_positions: torch.Tensor
    miss_count: torch.Tensor
    num_groups: int
    num_ranks: int
    source: RetroSpecResidentPrefetchSource = "draft"
    sequence: int = 0


@dataclass
class _PinnedSelectionSlot:
    """Reusable pinned ring slot for compact resident miss commands."""

    pinned_memory: RetroSpecPinnedMemoryManager
    cluster_id_storage: torch.Tensor | None = None
    position_storage: torch.Tensor | None = None
    count_storage: torch.Tensor | None = None
    in_use: bool = False

    def reserve_capacity(self, required_numel: int, required_records: int) -> None:
        if required_numel <= 0:
            return
        if (
            self.cluster_id_storage is None
            or self.cluster_id_storage.numel() < required_numel
        ):
            self.cluster_id_storage = self.pinned_memory.replace(
                self.cluster_id_storage,
                (required_numel,),
                torch.int64,
                "resident-prefetch-cluster-ids",
            )
        if (
            self.position_storage is None
            or self.position_storage.numel() < required_numel
        ):
            self.position_storage = self.pinned_memory.replace(
                self.position_storage,
                (required_numel,),
                torch.int64,
                "resident-prefetch-miss-positions",
            )
        if self.count_storage is None or self.count_storage.numel() < required_records:
            self.count_storage = self.pinned_memory.replace(
                self.count_storage,
                (required_records,),
                torch.int32,
                "resident-prefetch-miss-counts",
            )

    def reserve_wave(
        self,
        records: Sequence[RetroSpecResidentPrefetchInput],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        if not self.in_use:
            raise RuntimeError("Pinned selection slot must be acquired before use")

        required_numel = sum(record.miss_cluster_ids.numel() for record in records)
        self.reserve_capacity(required_numel, len(records))
        if (
            self.cluster_id_storage is None
            or self.position_storage is None
            or self.count_storage is None
        ):
            raise RuntimeError("Resident miss-command storage is unavailable")

        views: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        offset = 0
        for record_index, record in enumerate(records):
            cluster_ids = record.miss_cluster_ids
            positions = record.miss_positions
            if positions.shape != cluster_ids.shape:
                raise ValueError("Resident miss positions must match cluster IDs")

            next_offset = offset + cluster_ids.numel()
            views.append(
                (
                    self.cluster_id_storage[offset:next_offset].view(cluster_ids.shape),
                    self.position_storage[offset:next_offset].view(positions.shape),
                    self.count_storage[record_index : record_index + 1],
                )
            )
            offset = next_offset

        return tuple(views)

    def release_storage(self) -> None:
        self.pinned_memory.release(self.cluster_id_storage)
        self.pinned_memory.release(self.position_storage)
        self.pinned_memory.release(self.count_storage)
        self.cluster_id_storage = None
        self.position_storage = None
        self.count_storage = None


@dataclass
class _PinnedVerificationMissSlot:
    """Pinned metadata for one GPU-unique verification miss batch."""

    pinned_memory: RetroSpecPinnedMemoryManager
    unique_cluster_id_storage: torch.Tensor | None = None
    unique_logical_page_id_storage: torch.Tensor | None = None
    unique_page_count_storage: torch.Tensor | None = None
    unique_staging_start_storage: torch.Tensor | None = None
    miss_count_storage: torch.Tensor | None = None
    unique_miss_count_storage: torch.Tensor | None = None
    invalid_descriptor_count_storage: torch.Tensor | None = None
    capacity: int = 0
    max_pages: int = 0
    in_use: bool = False
    reuse_ready_event: torch.cuda.Event | None = None

    def reserve_capacity(self, capacity: int, max_pages: int) -> None:
        if capacity <= self.capacity and max_pages <= self.max_pages:
            return
        capacity = max(capacity, self.capacity)
        max_pages = max(max_pages, self.max_pages)
        self.unique_cluster_id_storage = self.pinned_memory.replace(
            self.unique_cluster_id_storage,
            (capacity,),
            torch.int64,
            "verification-unique-cluster-ids",
        )
        self.unique_logical_page_id_storage = self.pinned_memory.replace(
            self.unique_logical_page_id_storage,
            (capacity, max_pages),
            torch.int64,
            "verification-unique-logical-page-ids",
        )
        self.unique_page_count_storage = self.pinned_memory.replace(
            self.unique_page_count_storage,
            (capacity,),
            torch.int32,
            "verification-unique-page-counts",
        )
        self.unique_staging_start_storage = self.pinned_memory.replace(
            self.unique_staging_start_storage,
            (capacity,),
            torch.int64,
            "verification-unique-staging-starts",
        )
        if self.miss_count_storage is None:
            self.miss_count_storage = self.pinned_memory.empty(
                (1,), torch.int32, "verification-miss-count"
            )
        if self.unique_miss_count_storage is None:
            self.unique_miss_count_storage = self.pinned_memory.empty(
                (1,), torch.int32, "verification-unique-miss-count"
            )
        if self.invalid_descriptor_count_storage is None:
            self.invalid_descriptor_count_storage = self.pinned_memory.empty(
                (1,), torch.int32, "verification-invalid-descriptor-count"
            )
        self.capacity = capacity
        self.max_pages = max_pages

    def release_storage(self) -> None:
        self.pinned_memory.release(self.unique_cluster_id_storage)
        self.pinned_memory.release(self.unique_logical_page_id_storage)
        self.pinned_memory.release(self.unique_page_count_storage)
        self.pinned_memory.release(self.unique_staging_start_storage)
        self.pinned_memory.release(self.miss_count_storage)
        self.pinned_memory.release(self.unique_miss_count_storage)
        self.pinned_memory.release(self.invalid_descriptor_count_storage)
        self.unique_cluster_id_storage = None
        self.unique_logical_page_id_storage = None
        self.unique_page_count_storage = None
        self.unique_staging_start_storage = None
        self.miss_count_storage = None
        self.unique_miss_count_storage = None
        self.invalid_descriptor_count_storage = None
        self.capacity = 0
        self.max_pages = 0


@dataclass
class _VerificationResolveGPUArena:
    """Reusable device records for one verification resident lookup."""

    unique_cluster_ids: torch.Tensor | None = None
    unique_logical_page_ids: torch.Tensor | None = None
    unique_page_counts: torch.Tensor | None = None
    unique_staging_starts: torch.Tensor | None = None
    miss_hash_buckets: torch.Tensor | None = None
    miss_unique_indices: torch.Tensor | None = None
    miss_output_page_offsets: torch.Tensor | None = None
    miss_count: torch.Tensor | None = None
    unique_miss_count: torch.Tensor | None = None
    invalid_descriptor_count: torch.Tensor | None = None
    miss_table_handles: torch.Tensor | None = None
    miss_table_unique_indices: torch.Tensor | None = None
    resident_page_ids: torch.Tensor | None = None
    staging_page_ids: torch.Tensor | None = None
    page_token_counts: torch.Tensor | None = None
    row_page_counts: torch.Tensor | None = None
    selected_cluster_counts: torch.Tensor | None = None
    hit_cluster_counts: torch.Tensor | None = None
    miss_cluster_counts: torch.Tensor | None = None
    capacity: int = 0
    max_pages: int = 0
    hash_capacity: int = 0
    row_capacity: int = 0
    page_capacity: int = 0

    def reserve_capacity(
        self,
        capacity: int,
        max_pages: int,
        row_capacity: int,
        page_capacity: int,
        device: torch.device,
    ) -> None:
        required_hash_capacity = 1 << (max(2, 2 * capacity) - 1).bit_length()
        if (
            capacity <= self.capacity
            and max_pages <= self.max_pages
            and required_hash_capacity <= self.hash_capacity
            and row_capacity <= self.row_capacity
            and page_capacity <= self.page_capacity
        ):
            return
        capacity = max(capacity, self.capacity)
        max_pages = max(max_pages, self.max_pages)
        hash_capacity = max(required_hash_capacity, self.hash_capacity)
        row_capacity = max(row_capacity, self.row_capacity)
        page_capacity = max(page_capacity, self.page_capacity)
        self.unique_cluster_ids = torch.empty(
            capacity, dtype=torch.int64, device=device
        )
        self.unique_logical_page_ids = torch.empty(
            (capacity, max_pages), dtype=torch.int64, device=device
        )
        self.unique_page_counts = torch.empty(
            capacity, dtype=torch.int32, device=device
        )
        self.unique_staging_starts = torch.empty(
            capacity, dtype=torch.int64, device=device
        )
        self.miss_hash_buckets = torch.empty(capacity, dtype=torch.int64, device=device)
        self.miss_unique_indices = torch.empty(
            capacity, dtype=torch.int32, device=device
        )
        self.miss_output_page_offsets = torch.empty(
            capacity, dtype=torch.int64, device=device
        )
        self.miss_count = torch.empty(1, dtype=torch.int32, device=device)
        self.unique_miss_count = torch.empty(1, dtype=torch.int32, device=device)
        self.invalid_descriptor_count = torch.empty(1, dtype=torch.int32, device=device)
        self.miss_table_handles = torch.empty(
            hash_capacity, dtype=torch.int64, device=device
        )
        self.miss_table_unique_indices = torch.empty(
            hash_capacity, dtype=torch.int32, device=device
        )
        self.resident_page_ids = torch.empty(
            page_capacity, dtype=torch.int64, device=device
        )
        self.staging_page_ids = torch.empty(
            page_capacity, dtype=torch.int64, device=device
        )
        self.page_token_counts = torch.empty(
            page_capacity, dtype=torch.int32, device=device
        )
        self.row_page_counts = torch.empty(
            row_capacity, dtype=torch.int32, device=device
        )
        self.selected_cluster_counts = torch.empty(
            row_capacity, dtype=torch.int32, device=device
        )
        self.hit_cluster_counts = torch.empty(
            row_capacity, dtype=torch.int32, device=device
        )
        self.miss_cluster_counts = torch.empty(
            row_capacity, dtype=torch.int32, device=device
        )
        self.capacity = capacity
        self.max_pages = max_pages
        self.hash_capacity = hash_capacity
        self.row_capacity = row_capacity
        self.page_capacity = page_capacity


@dataclass(frozen=True)
class _CPUPageSlab:
    key_pages: torch.Tensor
    value_pages: torch.Tensor


@dataclass(frozen=True)
class _FullVerificationSourceSnapshot:
    dtype: torch.dtype
    head_size: int
    key_slabs: tuple[torch.Tensor, ...]
    value_slabs: tuple[torch.Tensor, ...]


@dataclass
class _PinnedPageTransferSlot:
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    reuse_ready_event: torch.cuda.Event | None = None
    in_use: bool = False


# Keep the original qualified path for serialized classes.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = "vllm.v1.spec_decode.retrospec.cluster_store"
del _legacy_type
