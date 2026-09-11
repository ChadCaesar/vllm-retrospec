# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from math import ceil
from threading import Event as ThreadEvent
from threading import Lock, RLock
from time import perf_counter
from typing import Literal

import torch

from vllm import _custom_ops as ops

from .cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from .index_residency import RetroSpecResidentLayerArena
from .performance import RetroSpecPerformanceStats
from .pinned_memory import RetroSpecPinnedMemoryManager
from .resident_cache import (
    RetroSpecCompactVerificationPageAccess,
    RetroSpecResidentClusterCache,
    RetroSpecResidentPageAccess,
    RetroSpecResidentReadLease,
)
from .resident_kernels import (
    compact_resident_misses,
    scatter_compact_staging_page_ids,
)

RetroSpecClusterResolveMode = Literal[
    "resident_only",
    "resident_pending",
    "verification",
]


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
    """Pinned metadata for one GPU-compacted verification miss batch."""

    pinned_memory: RetroSpecPinnedMemoryManager
    cluster_id_storage: torch.Tensor | None = None
    logical_page_id_storage: torch.Tensor | None = None
    output_page_offset_storage: torch.Tensor | None = None
    source_page_count_storage: torch.Tensor | None = None
    staging_start_storage: torch.Tensor | None = None
    page_count_storage: torch.Tensor | None = None
    miss_count_storage: torch.Tensor | None = None
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
        self.cluster_id_storage = self.pinned_memory.replace(
            self.cluster_id_storage,
            (capacity,),
            torch.int64,
            "verification-miss-cluster-ids",
        )
        self.logical_page_id_storage = self.pinned_memory.replace(
            self.logical_page_id_storage,
            (capacity, max_pages),
            torch.int64,
            "verification-miss-logical-page-ids",
        )
        self.output_page_offset_storage = self.pinned_memory.replace(
            self.output_page_offset_storage,
            (capacity,),
            torch.int64,
            "verification-miss-output-page-offsets",
        )
        self.source_page_count_storage = self.pinned_memory.replace(
            self.source_page_count_storage,
            (capacity,),
            torch.int32,
            "verification-miss-source-page-counts",
        )
        self.staging_start_storage = self.pinned_memory.replace(
            self.staging_start_storage,
            (capacity,),
            torch.int64,
            "verification-miss-staging-starts",
        )
        self.page_count_storage = self.pinned_memory.replace(
            self.page_count_storage,
            (capacity,),
            torch.int32,
            "verification-miss-page-counts",
        )
        if self.miss_count_storage is None:
            self.miss_count_storage = self.pinned_memory.empty(
                (1,), torch.int32, "verification-miss-count"
            )
        if self.invalid_descriptor_count_storage is None:
            self.invalid_descriptor_count_storage = self.pinned_memory.empty(
                (1,), torch.int32, "verification-invalid-descriptor-count"
            )
        self.capacity = capacity
        self.max_pages = max_pages

    def release_storage(self) -> None:
        self.pinned_memory.release(self.cluster_id_storage)
        self.pinned_memory.release(self.logical_page_id_storage)
        self.pinned_memory.release(self.output_page_offset_storage)
        self.pinned_memory.release(self.source_page_count_storage)
        self.pinned_memory.release(self.staging_start_storage)
        self.pinned_memory.release(self.page_count_storage)
        self.pinned_memory.release(self.miss_count_storage)
        self.pinned_memory.release(self.invalid_descriptor_count_storage)
        self.cluster_id_storage = None
        self.logical_page_id_storage = None
        self.output_page_offset_storage = None
        self.source_page_count_storage = None
        self.staging_start_storage = None
        self.page_count_storage = None
        self.miss_count_storage = None
        self.invalid_descriptor_count_storage = None
        self.capacity = 0
        self.max_pages = 0


@dataclass
class _VerificationResolveGPUArena:
    """Reusable device records for one verification resident lookup."""

    cluster_ids: torch.Tensor | None = None
    logical_page_ids: torch.Tensor | None = None
    output_page_offsets: torch.Tensor | None = None
    source_page_counts: torch.Tensor | None = None
    staging_starts: torch.Tensor | None = None
    page_counts: torch.Tensor | None = None
    miss_count: torch.Tensor | None = None
    invalid_descriptor_count: torch.Tensor | None = None
    resident_page_ids: torch.Tensor | None = None
    staging_page_ids: torch.Tensor | None = None
    page_token_counts: torch.Tensor | None = None
    row_page_counts: torch.Tensor | None = None
    selected_cluster_counts: torch.Tensor | None = None
    hit_cluster_counts: torch.Tensor | None = None
    miss_cluster_counts: torch.Tensor | None = None
    capacity: int = 0
    max_pages: int = 0
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
        if (
            capacity <= self.capacity
            and max_pages <= self.max_pages
            and row_capacity <= self.row_capacity
            and page_capacity <= self.page_capacity
        ):
            return
        capacity = max(capacity, self.capacity)
        max_pages = max(max_pages, self.max_pages)
        row_capacity = max(row_capacity, self.row_capacity)
        page_capacity = max(page_capacity, self.page_capacity)
        self.cluster_ids = torch.empty(capacity, dtype=torch.int64, device=device)
        self.logical_page_ids = torch.empty(
            (capacity, max_pages), dtype=torch.int64, device=device
        )
        self.output_page_offsets = torch.empty(
            capacity, dtype=torch.int64, device=device
        )
        self.source_page_counts = torch.empty(
            capacity, dtype=torch.int32, device=device
        )
        self.staging_starts = torch.empty(capacity, dtype=torch.int64, device=device)
        self.page_counts = torch.empty(capacity, dtype=torch.int32, device=device)
        self.miss_count = torch.empty(1, dtype=torch.int32, device=device)
        self.invalid_descriptor_count = torch.empty(1, dtype=torch.int32, device=device)
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
        self.row_capacity = row_capacity
        self.page_capacity = page_capacity


@dataclass(frozen=True)
class _StagedResidentPrefetchRecord:
    """One layer's resident miss commands staged in pinned CPU memory."""

    layer_name: str
    miss_cluster_ids_cpu: torch.Tensor
    miss_positions_cpu: torch.Tensor
    miss_count_cpu: torch.Tensor
    num_groups: int
    num_ranks: int


@dataclass(frozen=True)
class _StagedResidentPrefetchWave:
    """One draft step's cross-layer resident access records."""

    records: tuple[_StagedResidentPrefetchRecord, ...]
    metadata_ready_event: torch.cuda.Event
    execution_stream: torch.cuda.Stream
    slot: _PinnedSelectionSlot = field(repr=False, compare=False)


@dataclass(frozen=True)
class _DeferredResidentPrefetchWave:
    """Latest GPU command wave retained while the pinned ring is full."""

    records: tuple[RetroSpecResidentPrefetchInput, ...]
    source_ready_event: torch.cuda.Event


@dataclass(frozen=True)
class _PreparedResidentPrefetchRecord:
    """CPU descriptors prepared for one layer's resident admission."""

    layer_name: str
    pool: "_LayerClusterPagePool"
    resident_cache: RetroSpecResidentClusterCache
    cluster_ids_cpu: torch.Tensor
    page_ids_cpu: torch.Tensor
    cluster_groups: dict[int, RetroSpecClusterGroup]


@dataclass(frozen=True)
class _ResidentPrefetchWaveFuture:
    """One background wave and the CUDA device whose ring slot it owns."""

    device: torch.device
    layer_names: frozenset[str]
    future: Future[None] = field(repr=False, compare=False)


@dataclass(frozen=True)
class RetroSpecClusterBlockTable:
    """Ownership handle for CPU-managed cluster blocks.

    cluster_ids has shape:

        [num_kv_heads, num_clusters]

    The handle deliberately does not expose backing page IDs. Page placement
    belongs to RetroSpecClusterPageStore and is materialized only when an active
    batch is packed for retrieval.
    """

    cluster_ids: torch.Tensor
    full_verification_descriptor: "RetroSpecFullVerificationDescriptor"


@dataclass(frozen=True)
class RetroSpecClusterBlockMetadata:
    """Materialized physical descriptors for selected cluster IDs.

    page_ids and page_token_counts have shape:

        [*, max_pages_per_cluster]

    The leading shape is identical to the input cluster-ID tensor.
    """

    page_ids: torch.Tensor
    page_token_counts: torch.Tensor


@dataclass(frozen=True)
class RetroSpecStagedTokenKV:
    """Token KV staged before GPU clustering starts.

    For pinned CPU offload, token KV is copied on the per-device offload
    stream while segmented k-means runs on the model execution stream.
    """

    token_keys: torch.Tensor
    token_values: torch.Tensor

    source_device: torch.device
    ready_event: torch.cuda.Event | None
    staging_slot: _PinnedStagingSlot | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def wait(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()


@dataclass(frozen=True)
class RetroSpecStagedClusterInput:
    """Complete CPU inputs required to construct cluster pages.

    ready_event is recorded after both token KV and clustering metadata have
    been copied to CPU. Waiting for it therefore makes every tensor in this
    structure safe for CPU page construction.
    """

    token_keys: torch.Tensor
    token_values: torch.Tensor
    assignments: torch.Tensor
    cluster_token_counts: torch.Tensor
    token_offsets_in_cluster: torch.Tensor

    metadata_device: torch.device
    ready_event: torch.cuda.Event | None
    staging_slot: _PinnedStagingSlot | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def wait(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()


@dataclass(frozen=True)
class _ClusterBlockDescriptor:
    """CPU metadata mapped from one layer-global stable cluster handle."""

    identity: RetroSpecClusterIdentity
    page_ids: tuple[int, ...]
    page_token_counts: tuple[int, ...]


@dataclass(frozen=True)
class RetroSpecCompactTokenRange:
    """One valid-token range in a layer's pageable CPU slab storage."""

    slab_id: int
    token_offset: int
    token_count: int


class RetroSpecFullVerificationDescriptor:
    """Persistent compact full-prefix layout for every KV head of a request."""

    def __init__(
        self,
        head_ranges: tuple[tuple[RetroSpecCompactTokenRange, ...], ...],
        head_token_counts: tuple[int, ...],
    ) -> None:
        if len(head_ranges) != len(head_token_counts):
            raise ValueError("Full-verification descriptor head counts differ")

        rows: list[tuple[int, int, int, int, int]] = []
        for head_index, (ranges, expected_count) in enumerate(
            zip(head_ranges, head_token_counts)
        ):
            head_token_offset = 0
            for token_range in ranges:
                if token_range.slab_id < 0:
                    raise ValueError("Compact range slab ID must be non-negative")
                if token_range.token_offset < 0:
                    raise ValueError("Compact range token offset must be non-negative")
                if token_range.token_count <= 0:
                    raise ValueError("Compact range token count must be positive")

                rows.append(
                    (
                        head_index,
                        token_range.slab_id,
                        token_range.token_offset,
                        token_range.token_count,
                        head_token_offset,
                    )
                )
                head_token_offset += token_range.token_count

            if head_token_offset != expected_count:
                raise ValueError(
                    "Full-verification descriptor token count is inconsistent"
                )

        range_table = (
            torch.tensor(rows, dtype=torch.int64, device="cpu")
            if rows
            else torch.empty((0, 5), dtype=torch.int64, device="cpu")
        )
        counts = torch.tensor(head_token_counts, dtype=torch.int32, device="cpu")
        self._set_tensor_representation(range_table, counts)
        self._head_ranges_cache = head_ranges
        self._head_token_counts_cache = head_token_counts

    def _set_tensor_representation(
        self,
        range_table: torch.Tensor,
        head_token_counts: torch.Tensor,
    ) -> None:
        if range_table.device.type != "cpu" or range_table.dtype != torch.int64:
            raise ValueError("Full-verification range table must be CPU int64")
        if range_table.ndim != 2 or range_table.shape[1] != 5:
            raise ValueError(
                "Full-verification range table must have shape [ranges, 5]"
            )
        if (
            head_token_counts.device.type != "cpu"
            or head_token_counts.dtype != torch.int32
        ):
            raise ValueError("Full-verification head counts must be CPU int32")
        if head_token_counts.ndim != 1:
            raise ValueError("Full-verification head counts must be one-dimensional")

        range_table = range_table.contiguous()
        head_token_counts = head_token_counts.contiguous()
        num_heads = head_token_counts.shape[0]

        if range_table.numel():
            head_ids = range_table[:, 0]
            if torch.any((head_ids < 0) | (head_ids >= num_heads)).item():
                raise ValueError("Full-verification range contains an invalid head")
            if torch.any(range_table[:, 1:4] < 0).item():
                raise ValueError("Full-verification range contains negative fields")
            if torch.any(range_table[:, 3] == 0).item():
                raise ValueError("Full-verification range must contain tokens")
            if torch.any(range_table[:, 4] < 0).item():
                raise ValueError("Full-verification output offset is negative")

            actual_counts = torch.zeros(num_heads, dtype=torch.int64, device="cpu")
            actual_counts.scatter_add_(0, head_ids, range_table[:, 3])
            if not torch.equal(actual_counts, head_token_counts.to(dtype=torch.int64)):
                raise ValueError(
                    "Full-verification range counts do not match head counts"
                )
        elif torch.any(head_token_counts != 0).item():
            raise ValueError("Empty range table has non-zero head counts")

        self.range_table = range_table
        self.head_token_counts_tensor = head_token_counts

    @classmethod
    def from_tensors(
        cls,
        range_table: torch.Tensor,
        head_token_counts: torch.Tensor,
    ) -> "RetroSpecFullVerificationDescriptor":
        descriptor = cls.__new__(cls)
        descriptor._set_tensor_representation(range_table, head_token_counts)
        descriptor._head_ranges_cache = None
        descriptor._head_token_counts_cache = None
        return descriptor

    @classmethod
    def empty(cls, num_kv_heads: int) -> "RetroSpecFullVerificationDescriptor":
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        return cls.from_tensors(
            torch.empty((0, 5), dtype=torch.int64, device="cpu"),
            torch.zeros(num_kv_heads, dtype=torch.int32, device="cpu"),
        )

    @property
    def num_kv_heads(self) -> int:
        return self.head_token_counts_tensor.shape[0]

    @property
    def num_tokens(self) -> int:
        return int(self.head_token_counts_tensor.sum().item())

    @property
    def head_token_counts(self) -> tuple[int, ...]:
        if self._head_token_counts_cache is None:
            self._head_token_counts_cache = tuple(
                self.head_token_counts_tensor.tolist()
            )
        return self._head_token_counts_cache

    @property
    def head_ranges(
        self,
    ) -> tuple[tuple[RetroSpecCompactTokenRange, ...], ...]:
        if self._head_ranges_cache is None:
            ranges: list[list[RetroSpecCompactTokenRange]] = [
                [] for _ in range(self.num_kv_heads)
            ]
            for (
                head_index,
                slab_id,
                token_offset,
                token_count,
                _,
            ) in self.range_table.tolist():
                ranges[head_index].append(
                    RetroSpecCompactTokenRange(
                        slab_id=slab_id,
                        token_offset=token_offset,
                        token_count=token_count,
                    )
                )
            self._head_ranges_cache = tuple(tuple(row) for row in ranges)
        return self._head_ranges_cache

    def append(
        self, other: "RetroSpecFullVerificationDescriptor"
    ) -> "RetroSpecFullVerificationDescriptor":
        if self.num_kv_heads != other.num_kv_heads:
            raise ValueError("Full-verification descriptors changed KV-head count")

        appended_ranges = other.range_table.clone()
        if appended_ranges.numel():
            appended_heads = appended_ranges[:, 0]
            appended_ranges[:, 4].add_(
                self.head_token_counts_tensor.index_select(0, appended_heads).to(
                    dtype=torch.int64
                )
            )

        return RetroSpecFullVerificationDescriptor.from_tensors(
            torch.cat((self.range_table, appended_ranges), dim=0),
            self.head_token_counts_tensor + other.head_token_counts_tensor,
        )


@dataclass(frozen=True)
class RetroSpecFullVerificationStaging:
    """Token-contiguous GPU staging used only by full verification."""

    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    token_offsets: torch.Tensor
    token_counts: torch.Tensor
    max_tokens_per_head: int
    ready_event: torch.cuda.Event | None


@dataclass(frozen=True)
class RetroSpecFullVerificationTicket:
    """One asynchronously prepared full-verification layer."""

    future: Future[RetroSpecFullVerificationStaging]
    cancel_event: ThreadEvent

    def result(self) -> RetroSpecFullVerificationStaging:
        return self.future.result()

    def ready(self) -> bool:
        if not self.future.done() or self.future.cancelled():
            return False

        try:
            staging = self.future.result()
        except BaseException:
            return False

        return staging.ready_event is None or staging.ready_event.query()

    def cancel(self, wait: bool = False) -> bool:
        self.cancel_event.set()
        cancelled = self.future.cancel()
        if wait and not cancelled:
            with suppress(CancelledError):
                self.future.result()
        return cancelled


@dataclass(frozen=True)
class RetroSpecResolvedClusterPages:
    """Physical GPU sources for one logical cluster-page selection.

    resident_page_ids and staging_page_ids have the same shape as the logical
    page table. A non-negative resident page ID indexes resident_key_pages and
    resident_value_pages. A non-negative staging page ID indexes the temporary
    staging tensors.

    hit_cluster_mask and miss_cluster_mask have the logical page table's
    leading shape, excluding the page dimension. Empty or padded clusters are
    false in both masks.

    hit_gate_ready_mask marks clusters whose request/head resident LRU has
    reached its soft page target. Cold groups remain protected from hit-based
    draft transitions until this mask becomes true.

    resident_ready_event is recorded on the resident-cache copy stream.
    staging_ready_event is recorded after a full-verification layer transfer.
    The execution stream must wait for the corresponding event before reading
    a page source.
    """

    resident_page_ids: torch.Tensor
    staging_page_ids: torch.Tensor

    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor

    staging_key_pages: torch.Tensor
    staging_value_pages: torch.Tensor

    hit_cluster_mask: torch.Tensor
    miss_cluster_mask: torch.Tensor
    hit_gate_ready_mask: torch.Tensor
    resident_ready_event: torch.cuda.Event | None
    staging_ready_event: torch.cuda.Event | None = None
    access_kinds: torch.Tensor | None = None
    read_lease: RetroSpecResidentReadLease | None = None
    miss_admission: "RetroSpecVerificationMissAdmission | None" = None


@dataclass(frozen=True)
class RetroSpecCompactResolvedClusterPages:
    """Compact row-local resident page source produced for one DRAFT layer."""

    resident_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    clustered_token_counts: torch.Tensor
    attention_mass: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    hit_gate_ready: torch.Tensor
    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor
    read_lease: RetroSpecResidentReadLease


@dataclass(frozen=True)
class RetroSpecCompactVerificationResolvedPages:
    """Query-row compact resident and staging pages for verification."""

    resident_page_ids: torch.Tensor
    staging_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor
    staging_key_pages: torch.Tensor
    staging_value_pages: torch.Tensor
    staging_ready_event: torch.cuda.Event | None
    read_lease: RetroSpecResidentReadLease
    miss_admission: "RetroSpecVerificationMissAdmission | None" = None


@dataclass(frozen=True)
class RetroSpecVerificationMissAdmission:
    """Compact CPU metadata for admitting verification misses after attention."""

    layer_name: str
    cluster_ids_cpu: torch.Tensor
    logical_page_ids_cpu: torch.Tensor
    staging_page_ids_cpu: torch.Tensor
    staging_key_pages: torch.Tensor
    staging_value_pages: torch.Tensor
    staging_ready_event: torch.cuda.Event | None


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


class _LayerClusterPagePool:
    """Geometrically growing pageable CPU slabs for one layer."""

    _PAGE_OFFSET_BITS = 32
    _PAGE_OFFSET_MASK = (1 << _PAGE_OFFSET_BITS) - 1

    def __init__(
        self,
        page_size: int,
        head_size: int,
        dtype: torch.dtype,
        storage_device: torch.device,
        metadata_device: torch.device,
        initial_slab_size_bytes: int,
        max_slab_size_bytes: int,
    ) -> None:
        self.page_size = page_size
        self.head_size = head_size
        self.dtype = dtype
        self.storage_device = storage_device
        self.metadata_device = metadata_device
        if storage_device.type != "cpu":
            raise ValueError("Cluster-page slabs must use CPU storage")
        if initial_slab_size_bytes <= 0:
            raise ValueError("initial_slab_size_bytes must be positive")
        if max_slab_size_bytes < initial_slab_size_bytes:
            raise ValueError(
                "max_slab_size_bytes must not be smaller than initial_slab_size_bytes"
            )

        self.pin_memory = False
        page_pair_bytes = 2 * page_size * head_size * dtype.itemsize
        self.initial_pages_per_slab = max(initial_slab_size_bytes // page_pair_bytes, 1)
        self.max_pages_per_slab = max(
            max_slab_size_bytes // page_pair_bytes,
            self.initial_pages_per_slab,
        )
        if self.max_pages_per_slab > self._PAGE_OFFSET_MASK + 1:
            raise ValueError("Cluster-page slab contains too many pages")

        self._slabs: list[_CPUPageSlab] = []
        self._slab_allocated_page_counts: list[int] = []

        # Allocation state remains on the CPU because page allocation and
        # request release are control-plane operations.
        self._free_page_ids: list[int] = []
        self._allocated_page_ids: set[int] = set()

    @classmethod
    def encode_page_id(cls, slab_id: int, page_offset: int) -> int:
        if slab_id < 0 or page_offset < 0:
            raise ValueError("Slab ID and page offset must be non-negative")
        if page_offset > cls._PAGE_OFFSET_MASK:
            raise ValueError("Page offset exceeds the encoded handle width")
        return (slab_id << cls._PAGE_OFFSET_BITS) | page_offset

    @classmethod
    def decode_page_id(cls, page_id: int) -> tuple[int, int]:
        if page_id < 0:
            raise ValueError("Cluster page ID must be non-negative")
        return page_id >> cls._PAGE_OFFSET_BITS, page_id & cls._PAGE_OFFSET_MASK

    @property
    def pages_per_slab(self) -> int:
        """Maximum slab capacity retained for compatibility and diagnostics."""
        return self.max_pages_per_slab

    def _next_slab_page_capacity(self) -> int:
        if not self._slabs:
            return self.initial_pages_per_slab
        return min(
            self._slabs[-1].key_pages.shape[0] * 2,
            self.max_pages_per_slab,
        )

    def _append_slab(self) -> None:
        slab_id = len(self._slabs)
        slab_capacity = self._next_slab_page_capacity()
        shape = (slab_capacity, self.page_size, self.head_size)
        self._slabs.append(
            _CPUPageSlab(
                key_pages=torch.empty(shape, dtype=self.dtype, device="cpu"),
                value_pages=torch.empty(shape, dtype=self.dtype, device="cpu"),
            )
        )
        self._slab_allocated_page_counts.append(0)
        self._free_page_ids.extend(
            self.encode_page_id(slab_id, page_offset)
            for page_offset in range(slab_capacity - 1, -1, -1)
        )

    @property
    def capacity(self) -> int:
        return sum(slab.key_pages.shape[0] for slab in self._slabs)

    @property
    def num_slabs(self) -> int:
        return len(self._slabs)

    @property
    def num_allocated_pages(self) -> int:
        return len(self._allocated_page_ids)

    @property
    def allocated_page_ids(self) -> set[int]:
        """Return allocator-owned IDs for internal membership checks."""
        return self._allocated_page_ids

    def snapshot_full_verification_sources(
        self,
    ) -> _FullVerificationSourceSnapshot:
        """Capture stable tensor references without copying slab contents."""
        return _FullVerificationSourceSnapshot(
            dtype=self.dtype,
            head_size=self.head_size,
            key_slabs=tuple(slab.key_pages for slab in self._slabs),
            value_slabs=tuple(slab.value_pages for slab in self._slabs),
        )

    def build_cluster_pages(
        self,
        allocated_page_ids: torch.Tensor,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
        num_workers: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        RetroSpecFullVerificationDescriptor,
    ]:
        page_ids, page_token_counts, range_table, head_token_counts = (
            ops.retrospec_build_cluster_pages(
                tuple(slab.key_pages for slab in self._slabs),
                tuple(slab.value_pages for slab in self._slabs),
                allocated_page_ids.contiguous(),
                token_keys.contiguous(),
                token_values.contiguous(),
                assignments.contiguous(),
                cluster_token_counts.contiguous(),
                token_offsets_in_cluster.contiguous(),
                self.page_size,
                num_workers,
            )
        )
        descriptor = RetroSpecFullVerificationDescriptor.from_tensors(
            range_table, head_token_counts
        )
        return page_ids, page_token_counts, descriptor

    def allocate(self, num_pages: int) -> torch.Tensor:
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages == 0:
            return torch.empty(
                0,
                dtype=torch.int64,
                device=self.storage_device,
            )

        page_ids: list[int] = []
        while len(page_ids) < num_pages:
            if not self._free_page_ids:
                self._append_slab()
            num_available = min(len(self._free_page_ids), num_pages - len(page_ids))
            page_ids.extend(self._free_page_ids.pop() for _ in range(num_available))

        for page_id in page_ids:
            if page_id in self._allocated_page_ids:
                raise RuntimeError(
                    f"RetroSpec cluster page {page_id} is already allocated"
                )
            slab_id, _ = self.decode_page_id(page_id)
            self._allocated_page_ids.add(page_id)
            self._slab_allocated_page_counts[slab_id] += 1

        return torch.tensor(
            page_ids,
            dtype=torch.int64,
            device=self.storage_device,
        )

    def free(self, page_ids: torch.Tensor) -> None:
        if page_ids.numel() == 0:
            return

        valid_page_ids = page_ids[page_ids >= 0]
        if valid_page_ids.numel() == 0:
            return

        # Request removal and index rollback are infrequent control-plane
        # operations, so synchronizing page IDs here is acceptable.
        unique_page_ids = set(valid_page_ids.detach().cpu().tolist())

        for page_id in unique_page_ids:
            if page_id not in self._allocated_page_ids:
                raise RuntimeError(f"RetroSpec cluster page {page_id} is not allocated")

        for page_id in sorted(unique_page_ids, reverse=True):
            slab_id, _ = self.decode_page_id(page_id)
            self._allocated_page_ids.remove(page_id)
            self._slab_allocated_page_counts[slab_id] -= 1
            self._free_page_ids.append(page_id)

        self._trim_empty_tail_slabs()

    def _trim_empty_tail_slabs(self) -> None:
        while self._slabs and self._slab_allocated_page_counts[-1] == 0:
            removed_slab_id = len(self._slabs) - 1
            self._free_page_ids = [
                page_id
                for page_id in self._free_page_ids
                if self.decode_page_id(page_id)[0] != removed_slab_id
            ]
            self._slabs.pop()
            self._slab_allocated_page_counts.pop()

    def write(
        self,
        page_ids: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
    ) -> None:
        expected_shape = (
            page_ids.numel(),
            self.page_size,
            self.head_size,
        )

        if key_pages.shape != expected_shape:
            raise ValueError("key_pages shape does not match allocated page count")
        if value_pages.shape != expected_shape:
            raise ValueError("value_pages shape does not match allocated page count")
        if key_pages.dtype != self.dtype:
            raise ValueError("Key-page dtype does not match the layer page pool")
        if value_pages.dtype != self.dtype:
            raise ValueError("Value-page dtype does not match the layer page pool")
        if page_ids.device != self.storage_device:
            raise ValueError("Page IDs must be on the backing-store device")
        if key_pages.device != self.storage_device:
            raise ValueError("Key pages must be on the backing-store device")
        if value_pages.device != self.storage_device:
            raise ValueError("Value pages must be on the backing-store device")

        slab_positions: dict[int, list[tuple[int, int]]] = {}
        for source_index, page_id in enumerate(page_ids.tolist()):
            if page_id not in self._allocated_page_ids:
                raise RuntimeError(f"RetroSpec cluster page {page_id} is not allocated")
            slab_id, page_offset = self.decode_page_id(page_id)
            slab_positions.setdefault(slab_id, []).append((source_index, page_offset))

        for slab_id, positions in slab_positions.items():
            slab = self._slabs[slab_id]
            source_ids, page_offsets = zip(*positions)
            source_index = torch.tensor(source_ids, dtype=torch.int64)
            slab_index = torch.tensor(page_offsets, dtype=torch.int64)
            slab.key_pages.index_copy_(
                0, slab_index, key_pages.index_select(0, source_index)
            )
            slab.value_pages.index_copy_(
                0, slab_index, value_pages.index_select(0, source_index)
            )

    def read_into(
        self,
        page_ids: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
    ) -> None:
        """Gather logical page handles into caller-owned CPU storage."""
        if page_ids.device.type != "cpu" or page_ids.dtype != torch.int64:
            raise ValueError("Page IDs must be CPU int64")
        expected_shape = (page_ids.numel(), self.page_size, self.head_size)
        if key_pages.shape != expected_shape or value_pages.shape != expected_shape:
            raise ValueError("Destination page storage has an invalid shape")
        if key_pages.device.type != "cpu" or value_pages.device.type != "cpu":
            raise ValueError("Destination page storage must reside on CPU")
        if key_pages.dtype != self.dtype or value_pages.dtype != self.dtype:
            raise ValueError("Destination page dtype does not match the pool")

        flat_page_ids = page_ids.reshape(-1).tolist()
        key_pages.zero_()
        value_pages.zero_()
        slab_positions: dict[int, list[tuple[int, int]]] = {}
        for destination_index, page_id in enumerate(flat_page_ids):
            if page_id < 0:
                continue
            if page_id not in self._allocated_page_ids:
                raise RuntimeError(f"RetroSpec cluster page {page_id} is not allocated")
            slab_id, page_offset = self.decode_page_id(page_id)
            slab_positions.setdefault(slab_id, []).append(
                (destination_index, page_offset)
            )

        for slab_id, positions in slab_positions.items():
            slab = self._slabs[slab_id]
            destination_ids, page_offsets = zip(*positions)
            destination_index = torch.tensor(destination_ids, dtype=torch.int64)
            slab_index = torch.tensor(page_offsets, dtype=torch.int64)
            key_pages.index_copy_(
                0, destination_index, slab.key_pages.index_select(0, slab_index)
            )
            value_pages.index_copy_(
                0, destination_index, slab.value_pages.index_select(0, slab_index)
            )

    def read(
        self,
        page_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.any(page_ids < -1).item():
            raise ValueError("Cluster page IDs must be at least -1")

        storage_page_ids = page_ids.to(
            device=self.storage_device,
            dtype=torch.int64,
        )
        output_shape = (
            *storage_page_ids.shape,
            self.page_size,
            self.head_size,
        )

        if storage_page_ids.numel() == 0:
            empty_keys = torch.empty(
                output_shape,
                dtype=self.dtype,
                device=self.storage_device,
            )
            return empty_keys, empty_keys.clone()

        flat_shape = (storage_page_ids.numel(), self.page_size, self.head_size)
        key_pages = torch.empty(flat_shape, dtype=self.dtype, device="cpu")
        value_pages = torch.empty_like(key_pages)
        self.read_into(storage_page_ids, key_pages, value_pages)
        return key_pages.view(output_shape), value_pages.view(output_shape)

    def build_full_verification_descriptor(
        self,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
    ) -> RetroSpecFullVerificationDescriptor:
        """Convert cluster pages into immutable valid-token slab ranges."""
        if page_ids.device.type != "cpu" or page_ids.dtype != torch.int64:
            raise ValueError("Full-verification page IDs must be CPU int64")
        if page_token_counts.device.type != "cpu":
            raise ValueError("Full-verification token counts must reside on CPU")
        if page_ids.shape != page_token_counts.shape or page_ids.ndim < 2:
            raise ValueError("Full-verification page metadata has an invalid shape")

        head_ranges: list[tuple[RetroSpecCompactTokenRange, ...]] = []
        head_token_counts: list[int] = []
        for head_index in range(page_ids.shape[0]):
            ranges: list[RetroSpecCompactTokenRange] = []
            flat_ids = page_ids[head_index].reshape(-1).tolist()
            flat_counts = page_token_counts[head_index].reshape(-1).tolist()
            for page_id, token_count in zip(flat_ids, flat_counts):
                if page_id < 0:
                    continue
                if page_id not in self._allocated_page_ids:
                    raise RuntimeError(
                        f"RetroSpec cluster page {page_id} is not allocated"
                    )
                if not 0 < token_count <= self.page_size:
                    raise ValueError("Cluster page token count is out of range")
                slab_id, page_offset = self.decode_page_id(page_id)
                token_range = RetroSpecCompactTokenRange(
                    slab_id=slab_id,
                    token_offset=page_offset * self.page_size,
                    token_count=token_count,
                )
                if ranges:
                    previous = ranges[-1]
                    previous_end = previous.token_offset + previous.token_count
                    if (
                        previous.slab_id == slab_id
                        and previous_end == token_range.token_offset
                    ):
                        ranges[-1] = RetroSpecCompactTokenRange(
                            slab_id=slab_id,
                            token_offset=previous.token_offset,
                            token_count=previous.token_count + token_count,
                        )
                        continue
                ranges.append(token_range)
            head_ranges.append(tuple(ranges))
            head_token_counts.append(
                sum(token_range.token_count for token_range in ranges)
            )

        return RetroSpecFullVerificationDescriptor(
            head_ranges=tuple(head_ranges),
            head_token_counts=tuple(head_token_counts),
        )


@dataclass
class _FullVerificationGPUArena:
    dtype: torch.dtype | None = None
    head_size: int | None = None
    capacity: int = 0
    metadata_capacity: int = 0
    key_tokens: torch.Tensor | None = None
    value_tokens: torch.Tensor | None = None
    token_offsets: torch.Tensor | None = None
    token_counts: torch.Tensor | None = None


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
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if device.type != "cuda":
            raise ValueError("Full-verification transfer buffer requires CUDA")
        if max_pinned_memory_bytes <= 0:
            raise ValueError("max_pinned_memory_bytes must be positive")

        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.page_size = page_size
        self.device = device
        self.performance_stats = performance_stats
        self.max_pinned_memory_bytes = max_pinned_memory_bytes
        self._pinned_memory = pinned_memory
        self.pin_memory = pinned_memory.enabled

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
        if self._closed:
            return

        self._closed = True
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
        with self._cpu_slot_lock:
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
            if slot is None:
                raise RuntimeError("RetroSpec pinned H2D staging ring is exhausted")

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
        num_pages = page_ids_cpu.numel()
        if num_pages > self._cpu_slot_capacity:
            raise RuntimeError(
                "RetroSpec resident H2D selection exceeds the fixed pinned staging "
                "slot; increase retrospec_max_pinned_memory"
            )

        slot = self._acquire_cpu_slot()
        staged_keys = slot.key_pages[:num_pages]
        staged_values = slot.value_pages[:num_pages]
        try:
            pool.read_into(page_ids_cpu.reshape(-1), staged_keys, staged_values)
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
        with self._cpu_slot_lock:
            if not slot.in_use:
                raise RuntimeError("RetroSpec pinned H2D slot was already released")
            slot.reuse_ready_event = reuse_ready_event
            slot.in_use = False

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


class RetroSpecClusterPageStore:
    """CPU cluster-page backing store with a bounded GPU resident cache."""

    _RESIDENT_PREFETCH_RING_SIZE = 2
    _VERIFICATION_MISS_RING_SIZE = 2

    def __init__(
        self,
        page_size: int,
        pin_memory: bool = False,
        cache_ratio: float = 0.0,
        cpu_page_initial_slab_bytes: int | None = None,
        cpu_page_slab_bytes: int = 1 << 20,
        max_pinned_memory_bytes: int = 64 << 20,
        max_pending_cluster_builds: int = 2,
        cpu_page_build_workers: int = 4,
        performance_stats: RetroSpecPerformanceStats | None = None,
        pinned_memory: RetroSpecPinnedMemoryManager | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if not 0.0 <= cache_ratio <= 1.0:
            raise ValueError("cache_ratio must be between zero and one")
        if cpu_page_slab_bytes <= 0:
            raise ValueError("cpu_page_slab_bytes must be positive")
        if cpu_page_initial_slab_bytes is None:
            cpu_page_initial_slab_bytes = min(8 << 20, cpu_page_slab_bytes)
        if cpu_page_initial_slab_bytes <= 0:
            raise ValueError("cpu_page_initial_slab_bytes must be positive")
        if cpu_page_initial_slab_bytes > cpu_page_slab_bytes:
            raise ValueError(
                "cpu_page_initial_slab_bytes must not exceed cpu_page_slab_bytes"
            )
        if max_pinned_memory_bytes <= 0:
            raise ValueError("max_pinned_memory_bytes must be positive")
        if max_pending_cluster_builds <= 0:
            raise ValueError("max_pending_cluster_builds must be positive")
        if cpu_page_build_workers <= 0:
            raise ValueError("cpu_page_build_workers must be positive")

        if pinned_memory is None:
            pinned_memory = RetroSpecPinnedMemoryManager(
                enabled=pin_memory,
                max_bytes=max_pinned_memory_bytes,
            )
        elif pin_memory and not pinned_memory.enabled:
            raise ValueError("pin_memory conflicts with the shared pinned manager")

        self.page_size = page_size
        self._pinned_memory = pinned_memory
        self.pin_memory = pinned_memory.enabled
        self.cache_ratio = cache_ratio
        self.cpu_page_initial_slab_bytes = cpu_page_initial_slab_bytes
        self.cpu_page_slab_bytes = cpu_page_slab_bytes
        self.max_pinned_memory_bytes = pinned_memory.max_bytes
        self.max_pending_cluster_builds = max_pending_cluster_builds
        self.cpu_page_build_workers = cpu_page_build_workers
        self.performance_stats = performance_stats

        self._layer_pools: dict[str, _LayerClusterPagePool] = {}
        self._resident_caches: dict[str, RetroSpecResidentClusterCache] = {}

        self._next_cluster_ids: dict[str, int] = {}
        self._allocated_cluster_ids: dict[str, set[int]] = {}
        self._cluster_block_descriptors: dict[
            str, dict[int, _ClusterBlockDescriptor]
        ] = {}

        # layer_name -> group -> number of owned CPU backing pages
        self._group_backing_page_counts: dict[
            str, dict[RetroSpecClusterGroup, int]
        ] = {}

        # One serialized D2H stream per CUDA device. Different model layers
        # share the stream so their staged CPU buffers are completed in enqueue
        # order without synchronizing the model execution stream.
        self._offload_streams: dict[torch.device, torch.cuda.Stream] = {}

        # Pinned D2H staging slots are shared by all model layers on one CUDA
        # device. The segmented index bounds the number simultaneously in use.
        self._pinned_staging_slots: dict[
            torch.device,
            list[_PinnedStagingSlot],
        ] = {}
        self._pinned_staging_lock = Lock()

        # Full verification transfers one complete layer at a time. Layers on
        # the same CUDA device reuse one growable page arena.
        self._full_verification_buffers: dict[
            torch.device, _FullVerificationTransferBuffer
        ] = {}

        self._resident_prefetch_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._resident_prefetch_slots: dict[
            torch.device, list[_PinnedSelectionSlot]
        ] = {}
        self._resident_access_record_capacity = 0
        self._resident_prefetch_wave_max_records = 1
        self._resident_prefetch_futures: deque[_ResidentPrefetchWaveFuture] = deque()
        self._resident_prefetch_deferred: dict[
            torch.device, _DeferredResidentPrefetchWave
        ] = {}
        self._resident_prefetch_last_metadata_events: dict[
            torch.device, torch.cuda.Event
        ] = {}
        self._resident_prefetch_lock = Lock()
        self._resident_prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="retrospec-resident-prefetch",
        )
        self._verification_miss_slots: dict[
            torch.device, list[_PinnedVerificationMissSlot]
        ] = {}
        self._verification_gpu_arenas: dict[
            torch.device, list[_VerificationResolveGPUArena]
        ] = {}
        self._verification_resolve_cursors: dict[torch.device, int] = {}
        self._verification_metadata_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._verification_resolve_lock = Lock()
        self._resident_state_lock = RLock()
        self._closed = False

    def _allocate_cluster_ids(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        cluster_token_counts: torch.Tensor,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        if cluster_start < 0:
            raise ValueError("cluster_start must be non-negative")

        cluster_token_counts_cpu = cluster_token_counts.detach().to(
            device="cpu", dtype=torch.int64
        )
        page_ids_cpu = page_ids.detach().to(device="cpu", dtype=torch.int64)
        page_token_counts_cpu = page_token_counts.detach().to(
            device="cpu", dtype=torch.int32
        )

        if cluster_token_counts_cpu.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if page_ids_cpu.shape != page_token_counts_cpu.shape:
            raise ValueError(
                "Cluster page IDs and page token counts must have equal shapes"
            )
        if page_ids_cpu.shape[:-1] != cluster_token_counts_cpu.shape:
            raise ValueError(
                "Cluster page metadata does not match cluster token counts"
            )

        num_kv_heads, num_clusters = cluster_token_counts_cpu.shape
        groups = tuple(
            RetroSpecClusterGroup(
                request_id=request_id,
                kv_head_index=head_index,
            )
            for head_index in range(num_kv_heads)
        )

        valid_clusters_cpu = cluster_token_counts_cpu > 0
        num_valid_clusters = int(valid_clusters_cpu.sum().item())

        cluster_ids_cpu = torch.full(
            cluster_token_counts_cpu.shape,
            -1,
            dtype=torch.int64,
            device="cpu",
        )

        # Stable handles remain unique and monotonic within one layer. They are
        # storage handles rather than local clustering labels.
        next_cluster_id = self._next_cluster_ids.get(layer_name, 0)
        cluster_id_end = next_cluster_id + num_valid_clusters

        if num_valid_clusters:
            cluster_ids_cpu.masked_scatter_(
                valid_clusters_cpu,
                torch.arange(
                    next_cluster_id,
                    cluster_id_end,
                    dtype=torch.int64,
                ),
            )

        flat_cluster_ids = cluster_ids_cpu.reshape(-1).tolist()
        flat_cluster_token_counts = cluster_token_counts_cpu.reshape(-1).tolist()
        flat_page_ids = page_ids_cpu.reshape(
            len(flat_cluster_ids), page_ids_cpu.shape[-1]
        ).tolist()
        flat_page_token_counts = page_token_counts_cpu.reshape(
            len(flat_cluster_ids), page_token_counts_cpu.shape[-1]
        ).tolist()

        new_descriptors: dict[int, _ClusterBlockDescriptor] = {}

        for flat_index, (
            cluster_id,
            cluster_token_count,
            page_row,
            page_count_row,
        ) in enumerate(
            zip(
                flat_cluster_ids,
                flat_cluster_token_counts,
                flat_page_ids,
                flat_page_token_counts,
            )
        ):
            head_index, cluster_index = divmod(flat_index, num_clusters)

            valid_pages = tuple(page_id for page_id in page_row if page_id >= 0)
            valid_page_token_counts = tuple(
                page_count
                for page_id, page_count in zip(page_row, page_count_row)
                if page_id >= 0
            )

            for page_id, page_count in zip(page_row, page_count_row):
                if (page_id >= 0) != (page_count > 0):
                    raise RuntimeError(
                        "Cluster page ID and token-count validity do not match"
                    )

            if cluster_id < 0:
                if cluster_token_count != 0 or valid_pages:
                    raise RuntimeError("An empty cluster cannot own backing pages")
                continue

            if not valid_pages:
                raise RuntimeError("A valid cluster must own at least one backing page")
            if sum(valid_page_token_counts) != cluster_token_count:
                raise RuntimeError(
                    "Cluster page token counts do not match cluster size"
                )

            new_descriptors[cluster_id] = _ClusterBlockDescriptor(
                identity=RetroSpecClusterIdentity(
                    group=groups[head_index],
                    local_cluster_id=cluster_start + cluster_index,
                ),
                page_ids=valid_pages,
                page_token_counts=valid_page_token_counts,
            )

        allocated = self._allocated_cluster_ids.setdefault(layer_name, set())
        descriptors = self._cluster_block_descriptors.setdefault(layer_name, {})
        group_page_counts = self._group_backing_page_counts.setdefault(
            layer_name,
            {},
        )

        if allocated.intersection(new_descriptors):
            raise RuntimeError("RetroSpec cluster ID allocator produced a duplicate ID")

        new_group_pages: dict[RetroSpecClusterGroup, int] = {}
        for descriptor in new_descriptors.values():
            group = descriptor.identity.group
            new_group_pages[group] = new_group_pages.get(group, 0) + len(
                descriptor.page_ids
            )

        allocated.update(new_descriptors)
        descriptors.update(new_descriptors)

        for group, num_pages in new_group_pages.items():
            group_page_counts[group] = group_page_counts.get(group, 0) + num_pages

        self._next_cluster_ids[layer_name] = cluster_id_end
        return cluster_ids_cpu

    def _free_cluster_ids(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
    ) -> None:
        allocated = self._allocated_cluster_ids.get(layer_name)
        descriptors = self._cluster_block_descriptors.get(layer_name)
        group_page_counts = self._group_backing_page_counts.get(layer_name)

        if allocated is None or descriptors is None or group_page_counts is None:
            raise RuntimeError(f"No RetroSpec clusters exist for layer {layer_name!r}")

        cluster_ids_cpu = cluster_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        released = set(cluster_ids_cpu[cluster_ids_cpu >= 0].tolist())

        released_group_pages: dict[RetroSpecClusterGroup, int] = {}

        for cluster_id in released:
            if cluster_id not in allocated:
                raise RuntimeError(f"RetroSpec cluster {cluster_id} is not allocated")

            descriptor = descriptors[cluster_id]
            group = descriptor.identity.group
            released_group_pages[group] = released_group_pages.get(group, 0) + len(
                descriptor.page_ids
            )

        for group, num_pages in released_group_pages.items():
            current_pages = group_page_counts.get(group)
            if current_pages is None or current_pages < num_pages:
                raise RuntimeError(
                    "RetroSpec group backing-page accounting is inconsistent"
                )

        allocated.difference_update(released)

        for cluster_id in released:
            del descriptors[cluster_id]

        for group, num_pages in released_group_pages.items():
            remaining_pages = group_page_counts[group] - num_pages
            if remaining_pages:
                group_page_counts[group] = remaining_pages
            else:
                del group_page_counts[group]

    def _get_allocated_cluster_ids(
        self,
        layer_name: str,
    ) -> set[int]:
        allocated = self._allocated_cluster_ids.get(layer_name)
        if allocated is None:
            raise RuntimeError(f"No RetroSpec clusters exist for layer {layer_name!r}")
        return allocated

    def _validate_cluster_ids(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        if cluster_ids.ndim < 1:
            raise ValueError("Cluster IDs must have at least one dimension")
        if cluster_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cluster IDs must use an integral dtype")

        cluster_ids_cpu = cluster_ids.detach().to(device="cpu", dtype=torch.int64)

        if torch.any(cluster_ids_cpu < -1).item():
            raise ValueError("Cluster IDs must be at least -1")

        descriptors = self._cluster_block_descriptors.get(layer_name)
        if descriptors is None:
            raise RuntimeError(f"No RetroSpec clusters exist for layer {layer_name!r}")

        for cluster_id in cluster_ids_cpu.reshape(-1).tolist():
            if cluster_id >= 0 and cluster_id not in descriptors:
                raise RuntimeError(f"RetroSpec cluster {cluster_id} is not allocated")

        return cluster_ids_cpu

    def get_cluster_identities(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
    ) -> dict[int, RetroSpecClusterIdentity]:
        """Return semantic identities for valid stable cluster handles."""
        cluster_ids_cpu = self._validate_cluster_ids(layer_name, cluster_ids)
        descriptors = self._cluster_block_descriptors[layer_name]

        ordered_cluster_ids = dict.fromkeys(
            cluster_id
            for cluster_id in cluster_ids_cpu.reshape(-1).tolist()
            if cluster_id >= 0
        )

        return {
            cluster_id: descriptors[cluster_id].identity
            for cluster_id in ordered_cluster_ids
        }

    def _get_cluster_groups(
        self,
        layer_name: str,
        cluster_ids_cpu: torch.Tensor,
    ) -> dict[int, RetroSpecClusterGroup]:
        """Map stable cluster handles to resident replacement domains."""
        descriptors = self._cluster_block_descriptors[layer_name]
        ordered_cluster_ids = dict.fromkeys(
            cluster_id
            for cluster_id in cluster_ids_cpu.reshape(-1).tolist()
            if cluster_id >= 0
        )

        return {
            cluster_id: descriptors[cluster_id].identity.group
            for cluster_id in ordered_cluster_ids
        }

    def _validate_cluster_blocks(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if page_ids.ndim != cluster_ids.ndim + 1:
            raise ValueError("Cluster pages must add one page dimension to cluster IDs")
        if page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Cluster ID and page-table shapes do not match")
        if page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cluster page IDs must use an integral dtype")
        if cluster_ids.device != page_ids.device:
            raise ValueError("Cluster IDs and page IDs must use one device")

        cluster_ids_cpu = self._validate_cluster_ids(layer_name, cluster_ids)
        metadata = self._materialize_cluster_block_metadata_cpu(
            layer_name,
            cluster_ids_cpu,
            page_width=page_ids.shape[-1],
        )
        if metadata.page_ids.shape != page_ids.shape:
            raise RuntimeError("Cluster-page descriptor shape is inconsistent")

        return cluster_ids_cpu, metadata.page_ids

    def max_pages_per_cluster(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
    ) -> int:
        cluster_ids_cpu = self._validate_cluster_ids(layer_name, cluster_ids)
        descriptors = self._cluster_block_descriptors[layer_name]

        return max(
            (
                len(descriptors[cluster_id].page_ids)
                for cluster_id in cluster_ids_cpu.reshape(-1).tolist()
                if cluster_id >= 0
            ),
            default=0,
        )

    def get_cluster_block_metadata(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        device: torch.device | None = None,
    ) -> RetroSpecClusterBlockMetadata:
        """Materialize CPU-owned block descriptors for an active selection."""
        cluster_ids_cpu = self._validate_cluster_ids(layer_name, cluster_ids)
        metadata = self._materialize_cluster_block_metadata_cpu(
            layer_name,
            cluster_ids_cpu,
        )
        target_device = cluster_ids.device if device is None else device

        if target_device.type == "cpu":
            return metadata

        return RetroSpecClusterBlockMetadata(
            page_ids=metadata.page_ids.to(
                device=target_device,
                non_blocking=self.pin_memory,
            ),
            page_token_counts=metadata.page_token_counts.to(
                device=target_device,
                non_blocking=self.pin_memory,
            ),
        )

    def _materialize_cluster_block_metadata_cpu(
        self,
        layer_name: str,
        cluster_ids_cpu: torch.Tensor,
        page_width: int | None = None,
    ) -> RetroSpecClusterBlockMetadata:
        if cluster_ids_cpu.device.type != "cpu":
            raise ValueError("CPU cluster IDs must reside on CPU")
        if cluster_ids_cpu.dtype != torch.int64:
            raise ValueError("CPU cluster IDs must use int64")

        descriptors = self._cluster_block_descriptors[layer_name]

        natural_page_width = max(
            (
                len(descriptors[cluster_id].page_ids)
                for cluster_id in cluster_ids_cpu.reshape(-1).tolist()
                if cluster_id >= 0
            ),
            default=0,
        )
        if page_width is None:
            max_pages = natural_page_width
        else:
            if page_width < natural_page_width:
                raise RuntimeError(
                    "Packed cluster-page width is smaller than the CPU descriptor"
                )
            max_pages = page_width
        output_shape = (*cluster_ids_cpu.shape, max_pages)

        page_ids_cpu = torch.full(
            output_shape,
            -1,
            dtype=torch.int64,
            device="cpu",
        )
        page_token_counts_cpu = torch.zeros(
            output_shape,
            dtype=torch.int32,
            device="cpu",
        )

        flat_cluster_ids = cluster_ids_cpu.reshape(-1).tolist()
        flat_page_ids = page_ids_cpu.reshape(len(flat_cluster_ids), max_pages)
        flat_page_token_counts = page_token_counts_cpu.reshape(
            len(flat_cluster_ids), max_pages
        )

        for row_index, cluster_id in enumerate(flat_cluster_ids):
            if cluster_id < 0:
                continue

            descriptor = descriptors[cluster_id]
            num_pages = len(descriptor.page_ids)

            flat_page_ids[row_index, :num_pages] = torch.tensor(
                descriptor.page_ids, dtype=torch.int64
            )
            flat_page_token_counts[row_index, :num_pages] = torch.tensor(
                descriptor.page_token_counts, dtype=torch.int32
            )

        return RetroSpecClusterBlockMetadata(
            page_ids=page_ids_cpu,
            page_token_counts=page_token_counts_cpu,
        )

    def num_allocated_clusters(
        self,
        layer_name: str,
    ) -> int:
        allocated = self._allocated_cluster_ids.get(layer_name)
        return 0 if allocated is None else len(allocated)

    def _get_or_create_pool(
        self,
        layer_name: str,
        vectors: torch.Tensor,
        metadata_device: torch.device | None = None,
    ) -> _LayerClusterPagePool:
        if vectors.ndim != 3:
            raise ValueError(
                "Cluster vectors must have shape [num_kv_heads, num_tokens, head_size]"
            )

        if metadata_device is None:
            metadata_device = vectors.device

        head_size = vectors.shape[2]
        storage_device = torch.device("cpu")
        pool = self._layer_pools.get(layer_name)

        if pool is None:
            pool = _LayerClusterPagePool(
                page_size=self.page_size,
                head_size=head_size,
                dtype=vectors.dtype,
                storage_device=storage_device,
                metadata_device=metadata_device,
                initial_slab_size_bytes=self.cpu_page_initial_slab_bytes,
                max_slab_size_bytes=self.cpu_page_slab_bytes,
            )
            self._layer_pools[layer_name] = pool
            return pool

        if pool.head_size != head_size:
            raise ValueError("Cluster vectors do not match the layer head size")
        if pool.dtype != vectors.dtype:
            raise ValueError("Cluster vectors do not match the layer KV dtype")
        if pool.storage_device != storage_device:
            raise ValueError("Cluster vectors do not match the layer storage device")
        if pool.metadata_device != metadata_device:
            raise ValueError("Cluster metadata device changed for an existing layer")

        return pool

    def _resident_target_capacity(
        self,
        pool: _LayerClusterPagePool,
    ) -> int:
        if pool.num_allocated_pages == 0:
            return 0

        return min(
            pool.num_allocated_pages,
            ceil(pool.num_allocated_pages * self.cache_ratio),
        )

    def _resident_group_targets(
        self,
        layer_name: str,
        capacity: int,
    ) -> dict[RetroSpecClusterGroup, int]:
        """Distribute layer capacity proportionally across backing-page owners."""
        group_page_counts = self._group_backing_page_counts.get(
            layer_name,
            {},
        )
        total_backing_pages = sum(group_page_counts.values())

        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )
        if total_backing_pages != pool.num_allocated_pages:
            raise RuntimeError(
                "RetroSpec group backing-page accounting does not match "
                "the layer page pool"
            )

        if total_backing_pages == 0:
            if capacity != 0:
                raise RuntimeError(
                    "A non-empty resident capacity has no backing-page owners"
                )
            return {}

        if capacity > total_backing_pages:
            raise RuntimeError("Resident capacity exceeds owned backing pages")

        targets: dict[RetroSpecClusterGroup, int] = {}
        remainders: list[tuple[int, RetroSpecClusterGroup]] = []

        for group, num_backing_pages in group_page_counts.items():
            weighted_pages = capacity * num_backing_pages
            target_pages, remainder = divmod(
                weighted_pages,
                total_backing_pages,
            )
            targets[group] = target_pages
            remainders.append((remainder, group))

        remaining_pages = capacity - sum(targets.values())
        ordered_remainders = sorted(
            remainders,
            key=lambda item: (
                -item[0],
                item[1].request_id,
                item[1].kv_head_index,
            ),
        )

        for _, group in ordered_remainders[:remaining_pages]:
            targets[group] += 1

        if sum(targets.values()) != capacity:
            raise RuntimeError("Resident group targets do not cover layer capacity")

        return targets

    def resident_group_target_pages(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        num_kv_heads: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the resident-page target for each request/KV-head group."""
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

        request_ids = tuple(request_ids)
        with self._resident_state_lock:
            pool = self._layer_pools.get(layer_name)
            if pool is None:
                return tuple((0,) * num_kv_heads for _ in request_ids)

            capacity = self._resident_target_capacity(pool)
            targets = self._resident_group_targets(layer_name, capacity)

            return tuple(
                tuple(
                    targets.get(
                        RetroSpecClusterGroup(
                            request_id=request_id,
                            kv_head_index=head_index,
                        ),
                        0,
                    )
                    for head_index in range(num_kv_heads)
                )
                for request_id in request_ids
            )

    def _resize_resident_cache(
        self,
        layer_name: str,
        pool: _LayerClusterPagePool,
    ) -> None:
        resident_cache = self._resident_caches.get(layer_name)
        if resident_cache is None:
            return

        capacity = self._resident_target_capacity(pool)
        group_targets = self._resident_group_targets(
            layer_name,
            capacity,
        )
        resident_cache.resize(
            capacity,
            group_targets=group_targets,
        )

    def _get_or_create_resident_cache(
        self,
        layer_name: str,
    ) -> tuple[
        _LayerClusterPagePool,
        RetroSpecResidentClusterCache,
    ]:
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )
        if pool.metadata_device.type != "cuda":
            raise RuntimeError("Resident cluster cache requires CUDA metadata")

        resident_cache = self._resident_caches.get(layer_name)
        if resident_cache is None:
            resident_cache = RetroSpecResidentClusterCache(
                page_size=self.page_size,
                head_size=pool.head_size,
                dtype=pool.dtype,
                device=pool.metadata_device,
            )
            self._resident_caches[layer_name] = resident_cache

        capacity = self._resident_target_capacity(pool)
        resident_cache.resize(
            capacity,
            group_targets=self._resident_group_targets(
                layer_name,
                capacity,
            ),
        )
        return pool, resident_cache

    @staticmethod
    def _validate_token_kv_input(
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> None:
        if token_keys.ndim != 3:
            raise ValueError(
                "token_keys must have shape [num_kv_heads, num_tokens, head_size]"
            )
        if token_values.shape != token_keys.shape:
            raise ValueError("token_keys and token_values must have equal shapes")
        if token_values.dtype != token_keys.dtype:
            raise ValueError("token_keys and token_values must have equal dtypes")
        if token_values.device != token_keys.device:
            raise ValueError("Token keys and values must be on one device")

    @classmethod
    def _validate_cluster_input_metadata(
        cls,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> None:
        cls._validate_token_kv_input(token_keys, token_values)

        token_shape = token_keys.shape[:2]
        if assignments.shape != token_shape:
            raise ValueError("assignments must have shape [num_kv_heads, num_tokens]")
        if token_offsets_in_cluster.shape != token_shape:
            raise ValueError(
                "token_offsets_in_cluster must have shape [num_kv_heads, num_tokens]"
            )
        if cluster_token_counts.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if cluster_token_counts.shape[0] != token_keys.shape[0]:
            raise ValueError(
                "cluster_token_counts KV-head count does not match token KV"
            )
        metadata = (
            ("assignments", assignments),
            ("cluster_token_counts", cluster_token_counts),
            ("token_offsets_in_cluster", token_offsets_in_cluster),
        )
        for name, tensor in metadata:
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must use an integral dtype")
            if tensor.device != token_keys.device:
                raise ValueError(f"{name} and token KV must be on one device")

    @staticmethod
    def _validate_staged_cluster_metadata(
        staged_token_kv: RetroSpecStagedTokenKV,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> None:
        token_shape = staged_token_kv.token_keys.shape[:2]

        if assignments.shape != token_shape:
            raise ValueError("assignments must have shape [num_kv_heads, num_tokens]")
        if token_offsets_in_cluster.shape != token_shape:
            raise ValueError(
                "token_offsets_in_cluster must have shape [num_kv_heads, num_tokens]"
            )
        if cluster_token_counts.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if cluster_token_counts.shape[0] != token_shape[0]:
            raise ValueError(
                "cluster_token_counts KV-head count does not match token KV"
            )
        metadata = (
            ("Assignments", assignments),
            ("Cluster counts", cluster_token_counts),
            ("Cluster token offsets", token_offsets_in_cluster),
        )
        for name, tensor in metadata:
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must use an integral dtype")
            if tensor.device != staged_token_kv.source_device:
                raise ValueError(f"{name} must remain on the token KV source device")

    @staticmethod
    def _canonical_cuda_device(device: torch.device) -> torch.device:
        if device.type != "cuda":
            raise ValueError("RetroSpec staging requires a CUDA source device")

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        return torch.device("cuda", device_index)

    def _get_offload_stream(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream:
        canonical_device = self._canonical_cuda_device(device)
        stream = self._offload_streams.get(canonical_device)

        if stream is None:
            stream = torch.cuda.Stream(device=canonical_device)
            self._offload_streams[canonical_device] = stream

        return stream

    def _acquire_pinned_staging_slot(
        self,
        source_device: torch.device,
    ) -> _PinnedStagingSlot:
        canonical_device = self._canonical_cuda_device(source_device)

        with self._pinned_staging_lock:
            slots = self._pinned_staging_slots.setdefault(canonical_device, [])

            for slot in slots:
                if not slot.in_use:
                    slot.in_use = True
                    return slot

            if len(slots) >= self.max_pending_cluster_builds:
                raise RuntimeError(
                    "No RetroSpec cluster-build staging slot is available"
                )

            slot = _PinnedStagingSlot(
                source_device=canonical_device,
                max_bytes=(
                    self.max_pinned_memory_bytes // 2 // self.max_pending_cluster_builds
                ),
                pinned_memory=self._pinned_memory,
                in_use=True,
            )
            slots.append(slot)
            return slot

    def _release_pinned_staging_slot(
        self,
        slot: _PinnedStagingSlot | None,
    ) -> None:
        if slot is None:
            return

        with self._pinned_staging_lock:
            if not slot.in_use:
                raise RuntimeError("Pinned staging slot has already been released")
            slot.in_use = False

    def discard_staged_token_kv(
        self,
        staged: RetroSpecStagedTokenKV,
    ) -> None:
        """Wait for an abandoned token-KV transfer and release its slot."""
        try:
            staged.wait()
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    def discard_staged_clusters(
        self,
        staged: RetroSpecStagedClusterInput,
    ) -> None:
        """Wait for abandoned cluster inputs and release their slot."""
        try:
            staged.wait()
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    def stage_token_kv(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> RetroSpecStagedTokenKV:
        """Start staging token KV before segmented clustering."""
        self._validate_token_kv_input(token_keys, token_values)
        source_device = token_keys.device

        if source_device.type == "cpu":
            return RetroSpecStagedTokenKV(
                token_keys=token_keys,
                token_values=token_values,
                source_device=source_device,
                ready_event=None,
            )

        if source_device.type != "cuda":
            raise ValueError("CPU-backed cluster staging requires CPU or CUDA inputs")

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "token_kv_d2h_bytes",
                token_keys.nbytes + token_values.nbytes,
            )

        if not self.pin_memory:
            return RetroSpecStagedTokenKV(
                token_keys=token_keys.to(device="cpu", non_blocking=False),
                token_values=token_values.to(device="cpu", non_blocking=False),
                source_device=source_device,
                ready_event=None,
            )

        staging_slot = self._acquire_pinned_staging_slot(source_device)
        offload_stream: torch.cuda.Stream | None = None

        try:
            staged_token_keys, staged_token_values = staging_slot.reserve_token_kv(
                token_keys,
                token_values,
            )

            offload_stream = self._get_offload_stream(source_device)
            current_stream = torch.cuda.current_stream(source_device)
            offload_stream.wait_stream(current_stream)

            ready_event = torch.cuda.Event()

            with torch.cuda.stream(offload_stream):
                staged_token_keys.copy_(token_keys, non_blocking=True)
                staged_token_values.copy_(token_values, non_blocking=True)
                ready_event.record(offload_stream)

            token_keys.record_stream(offload_stream)
            token_values.record_stream(offload_stream)
        except BaseException:
            if offload_stream is not None:
                offload_stream.synchronize()
            self._release_pinned_staging_slot(staging_slot)
            raise

        return RetroSpecStagedTokenKV(
            token_keys=staged_token_keys,
            token_values=staged_token_values,
            source_device=source_device,
            ready_event=ready_event,
            staging_slot=staging_slot,
        )

    def finish_stage_clusters(
        self,
        staged_token_kv: RetroSpecStagedTokenKV,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> RetroSpecStagedClusterInput:
        """Stage clustering metadata after GPU clustering finishes."""
        self._validate_staged_cluster_metadata(
            staged_token_kv,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )
        source_device = staged_token_kv.source_device

        if source_device.type == "cpu":
            return RetroSpecStagedClusterInput(
                token_keys=staged_token_kv.token_keys,
                token_values=staged_token_kv.token_values,
                assignments=assignments,
                cluster_token_counts=cluster_token_counts,
                token_offsets_in_cluster=token_offsets_in_cluster,
                metadata_device=source_device,
                ready_event=None,
            )

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "cluster_metadata_d2h_bytes",
                assignments.nbytes
                + cluster_token_counts.nbytes
                + token_offsets_in_cluster.nbytes,
            )

        if not self.pin_memory:
            return RetroSpecStagedClusterInput(
                token_keys=staged_token_kv.token_keys,
                token_values=staged_token_kv.token_values,
                assignments=assignments.to(device="cpu", non_blocking=False),
                cluster_token_counts=cluster_token_counts.to(
                    device="cpu",
                    non_blocking=False,
                ),
                token_offsets_in_cluster=token_offsets_in_cluster.to(
                    device="cpu",
                    non_blocking=False,
                ),
                metadata_device=source_device,
                ready_event=None,
            )

        staging_slot = staged_token_kv.staging_slot
        if staging_slot is None:
            raise RuntimeError("Pinned token KV does not own a staging slot")

        offload_stream = self._get_offload_stream(source_device)

        try:
            (
                staged_assignments,
                staged_cluster_token_counts,
                staged_token_offsets,
            ) = staging_slot.reserve_cluster_metadata(
                assignments,
                cluster_token_counts,
                token_offsets_in_cluster,
            )

            current_stream = torch.cuda.current_stream(source_device)

            # Token-KV copies were enqueued earlier on the same offload stream.
            # This wait delays only metadata D2H until clustering has completed.
            offload_stream.wait_stream(current_stream)

            ready_event = torch.cuda.Event()

            with torch.cuda.stream(offload_stream):
                staged_assignments.copy_(assignments, non_blocking=True)
                staged_cluster_token_counts.copy_(
                    cluster_token_counts,
                    non_blocking=True,
                )
                staged_token_offsets.copy_(
                    token_offsets_in_cluster,
                    non_blocking=True,
                )
                ready_event.record(offload_stream)

            assignments.record_stream(offload_stream)
            cluster_token_counts.record_stream(offload_stream)
            token_offsets_in_cluster.record_stream(offload_stream)
        except BaseException:
            # Ownership remains with staged_token_kv until this method returns.
            # Synchronize partially queued metadata copies before it is discarded.
            offload_stream.synchronize()
            raise

        return RetroSpecStagedClusterInput(
            token_keys=staged_token_kv.token_keys,
            token_values=staged_token_kv.token_values,
            assignments=staged_assignments,
            cluster_token_counts=staged_cluster_token_counts,
            token_offsets_in_cluster=staged_token_offsets,
            metadata_device=source_device,
            ready_event=ready_event,
            staging_slot=staging_slot,
        )

    def stage_clusters(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> RetroSpecStagedClusterInput:
        """Stage complete inputs when overlap is not controlled by the caller."""
        self._validate_cluster_input_metadata(
            token_keys,
            token_values,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )
        staged_token_kv = self.stage_token_kv(token_keys, token_values)

        try:
            return self.finish_stage_clusters(
                staged_token_kv,
                assignments,
                cluster_token_counts,
                token_offsets_in_cluster,
            )
        except BaseException:
            self.discard_staged_token_kv(staged_token_kv)
            raise

    def store_staged_clusters(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        staged: RetroSpecStagedClusterInput,
    ) -> RetroSpecClusterBlockTable:
        """Wait for staged D2H copies and construct CPU cluster pages."""
        wait_started_at = perf_counter()
        try:
            staged.wait()
            if self.performance_stats is not None:
                self.performance_stats.record_cpu_time(
                    "cluster_build_wait",
                    perf_counter() - wait_started_at,
                )

            build_started_at = perf_counter()
            result = self.store_clusters(
                layer_name=layer_name,
                request_id=request_id,
                cluster_start=cluster_start,
                token_keys=staged.token_keys,
                token_values=staged.token_values,
                assignments=staged.assignments,
                cluster_token_counts=staged.cluster_token_counts,
                token_offsets_in_cluster=staged.token_offsets_in_cluster,
                metadata_device=staged.metadata_device,
            )
            if self.performance_stats is not None:
                self.performance_stats.record_cpu_time(
                    "cluster_page_build",
                    perf_counter() - build_started_at,
                )
                self.performance_stats.add_counter("cluster_builds")
            return result
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    @staticmethod
    def _move_to_storage(
        tensor: torch.Tensor,
        storage_device: torch.device,
    ) -> torch.Tensor:
        if tensor.device == storage_device:
            return tensor

        return tensor.to(
            device=storage_device,
            non_blocking=False,
        )

    def store_clusters(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
        metadata_device: torch.device | None = None,
    ) -> RetroSpecClusterBlockTable:
        """Pack token KV into per-head, per-cluster backing pages."""
        if cluster_start < 0:
            raise ValueError("cluster_start must be non-negative")

        self._validate_cluster_input_metadata(
            token_keys,
            token_values,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )

        pool = self._get_or_create_pool(
            layer_name,
            token_keys,
            metadata_device=metadata_device,
        )
        storage_keys = self._move_to_storage(token_keys, pool.storage_device)
        storage_values = self._move_to_storage(token_values, pool.storage_device)
        storage_assignments = self._move_to_storage(
            assignments,
            pool.storage_device,
        )
        storage_cluster_counts = self._move_to_storage(
            cluster_token_counts, pool.storage_device
        )
        storage_token_offsets = self._move_to_storage(
            token_offsets_in_cluster,
            pool.storage_device,
        )

        if torch.any(storage_cluster_counts < 0).item():
            raise ValueError("cluster_token_counts must be non-negative")

        cluster_page_counts = torch.div(
            storage_cluster_counts.to(dtype=torch.int64) + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        total_pages = int(cluster_page_counts.sum().item())
        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "cluster_pages_built",
                total_pages,
            )

        allocated_page_ids = pool.allocate(total_pages)

        try:
            page_ids, page_token_counts, full_descriptor = pool.build_cluster_pages(
                allocated_page_ids=allocated_page_ids,
                token_keys=storage_keys,
                token_values=storage_values,
                assignments=storage_assignments,
                cluster_token_counts=storage_cluster_counts,
                token_offsets_in_cluster=storage_token_offsets,
                num_workers=self.cpu_page_build_workers,
            )
        except Exception:
            pool.free(allocated_page_ids)
            raise

        cluster_ids: torch.Tensor | None = None
        with self._resident_state_lock:
            try:
                cluster_ids = self._allocate_cluster_ids(
                    layer_name=layer_name,
                    request_id=request_id,
                    cluster_start=cluster_start,
                    cluster_token_counts=storage_cluster_counts,
                    page_ids=page_ids,
                    page_token_counts=page_token_counts,
                )
                self._resize_resident_cache(layer_name, pool)
            except Exception:
                if cluster_ids is not None:
                    self._free_cluster_ids(layer_name, cluster_ids)
                pool.free(allocated_page_ids)
                raise

        assert cluster_ids is not None
        return RetroSpecClusterBlockTable(
            cluster_ids=cluster_ids,
            full_verification_descriptor=full_descriptor,
        )

    def free(
        self,
        layer_name: str,
        block_table: RetroSpecClusterBlockTable,
    ) -> None:
        self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
            pool = self._layer_pools.get(layer_name)
            if pool is None:
                raise RuntimeError(
                    f"No RetroSpec page pool exists for layer {layer_name!r}"
                )

            block_metadata = self.get_cluster_block_metadata(
                layer_name=layer_name,
                cluster_ids=block_table.cluster_ids,
                device=torch.device("cpu"),
            )

            resident_cache = self._resident_caches.get(layer_name)
            if resident_cache is not None:
                with resident_cache.mutation_guard():
                    resident_cache.invalidate(block_table.cluster_ids)

            pool.free(block_metadata.page_ids)
            self._free_cluster_ids(layer_name, block_table.cluster_ids)
            self._resize_resident_cache(layer_name, pool)

    def gather_pages(
        self,
        layer_name: str,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather selected pages on the backing-store device."""
        if page_ids.shape != page_token_counts.shape:
            raise ValueError("page_ids and page_token_counts must have equal shapes")

        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        key_pages, value_pages = pool.read(page_ids)
        storage_page_token_counts = page_token_counts.to(
            device=pool.storage_device,
            dtype=torch.int32,
        )

        token_offsets = torch.arange(
            self.page_size,
            dtype=torch.int32,
            device=pool.storage_device,
        )
        token_mask = token_offsets.view(
            *((1,) * storage_page_token_counts.ndim),
            self.page_size,
        ) < storage_page_token_counts.unsqueeze(-1)

        batch_size, num_kv_heads = page_ids.shape[:2]

        exact_keys = key_pages.reshape(
            batch_size,
            num_kv_heads,
            -1,
            pool.head_size,
        )
        exact_values = value_pages.reshape_as(exact_keys)
        exact_token_mask = token_mask.reshape(
            batch_size,
            num_kv_heads,
            -1,
        )

        exact_keys.masked_fill_(
            ~exact_token_mask.unsqueeze(-1),
            0.0,
        )
        exact_values.masked_fill_(
            ~exact_token_mask.unsqueeze(-1),
            0.0,
        )

        return (
            exact_keys.contiguous(),
            exact_values.contiguous(),
            exact_token_mask.contiguous(),
        )

    def read_page_storage(
        self,
        layer_name: str,
        page_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read logical page handles from the layer's pageable CPU slabs."""
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        return pool.read(page_ids)

    def get_storage_device(
        self,
        layer_name: str,
    ) -> torch.device:
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        return pool.storage_device

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
        logical_page_ids_cpu: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        _FullVerificationTransferBuffer,
        _PinnedPageTransferSlot | None,
    ]:
        source_page_ids = torch.full_like(logical_page_ids_cpu, -1)
        transfer_buffer = self._get_full_verification_buffer(pool)
        unique_logical_pages = tuple(
            dict.fromkeys(
                page_id
                for page_id in logical_page_ids_cpu.reshape(-1).tolist()
                if page_id >= 0
            )
        )
        num_pages = len(unique_logical_pages)
        if num_pages > transfer_buffer.cpu_slot_capacity(pool):
            raise RuntimeError(
                "RetroSpec resident staging selection exceeds one fixed H2D slot"
            )

        if not num_pages:
            empty_shape = (0, pool.page_size, pool.head_size)
            empty_keys = torch.empty(empty_shape, dtype=pool.dtype, device="cpu")
            return (
                source_page_ids,
                empty_keys,
                empty_keys.clone(),
                transfer_buffer,
                None,
            )

        logical_pages = torch.tensor(unique_logical_pages, dtype=torch.int64)
        staged_keys, staged_values, slot = transfer_buffer.stage_cpu_pages(
            pool, logical_pages
        )
        staging_id_by_page = {
            page_id: staging_id
            for staging_id, page_id in enumerate(unique_logical_pages)
        }
        flat_source_page_ids = source_page_ids.reshape(-1)
        for position, page_id in enumerate(logical_page_ids_cpu.reshape(-1).tolist()):
            staging_id = staging_id_by_page.get(page_id)
            if staging_id is not None:
                flat_source_page_ids[position] = staging_id
        return source_page_ids, staged_keys, staged_values, transfer_buffer, slot

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
            if self._canonical_cuda_device(cluster_ids.device) != device:
                raise ValueError("Resident prefetch wave must use one CUDA device")
        return device

    def _submit_resident_prefetch_wave(
        self,
        records: tuple[RetroSpecResidentPrefetchInput, ...],
        slot: _PinnedSelectionSlot,
        source_ready_event: torch.cuda.Event | None,
    ) -> None:
        device = self._canonical_cuda_device(records[0].miss_cluster_ids.device)
        stream = self._get_resident_prefetch_stream(device)
        ownership_transferred = False
        try:
            cpu_views = slot.reserve_wave(records)
            if source_ready_event is None:
                stream.wait_stream(torch.cuda.current_stream(device))
            else:
                stream.wait_event(source_ready_event)

            with torch.cuda.stream(stream):
                for record, (cluster_ids_cpu, positions_cpu, count_cpu) in zip(
                    records, cpu_views
                ):
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
                    )
                    for record, (
                        cluster_ids_cpu,
                        positions_cpu,
                        count_cpu,
                    ) in zip(records, cpu_views)
                ),
                metadata_ready_event=metadata_ready_event,
                execution_stream=stream,
                slot=slot,
            )
            future = self._resident_prefetch_executor.submit(
                self._finish_resident_prefetch_wave, staged
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

    def _defer_resident_prefetch_wave(
        self,
        device: torch.device,
        records: tuple[RetroSpecResidentPrefetchInput, ...],
    ) -> None:
        source_ready_event = torch.cuda.Event()
        source_ready_event.record(torch.cuda.current_stream(device))
        deferred = _DeferredResidentPrefetchWave(
            records=records, source_ready_event=source_ready_event
        )
        with self._resident_prefetch_lock:
            previous = self._resident_prefetch_deferred.get(device)
            self._resident_prefetch_deferred[device] = deferred

        if self.performance_stats is not None:
            self.performance_stats.add_counter("prefetch_waves_deferred")
            if previous is not None:
                self.performance_stats.add_counter("prefetch_waves_coalesced")
                self.performance_stats.add_counter(
                    "prefetch_records_superseded", len(previous.records)
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

            self._submit_resident_prefetch_wave(
                current.records, slot, current.source_ready_event
            )
            return True

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
    ) -> tuple[_PreparedResidentPrefetchRecord, ...]:
        ordered_records, raw_counts = ops.retrospec_order_prefetch_misses(
            tuple(record.miss_cluster_ids_cpu for record in records),
            tuple(record.miss_positions_cpu for record in records),
            tuple(record.miss_count_cpu for record in records),
            tuple(record.num_groups for record in records),
            tuple(record.num_ranks for record in records),
        )
        raw_command_count = int(raw_counts.sum().item())
        unique_command_count = sum(record.numel() for record in ordered_records)

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "prefetch_miss_commands", unique_command_count
            )
            self.performance_stats.add_counter(
                "prefetch_duplicate_misses",
                raw_command_count - unique_command_count,
            )

        prepared: list[_PreparedResidentPrefetchRecord] = []
        stale_commands = 0
        with self._resident_state_lock:
            for staged, ordered_cluster_ids in zip(records, ordered_records):
                if ordered_cluster_ids.numel() == 0:
                    continue

                pool, resident_cache = self._get_or_create_resident_cache(
                    staged.layer_name
                )
                descriptors = self._cluster_block_descriptors.get(staged.layer_name, {})
                transfer_buffer = self._get_full_verification_buffer(pool)
                page_capacity = transfer_buffer.cpu_slot_capacity(pool)

                selected_cluster_ids: list[int] = []
                selected_descriptors: list[_ClusterBlockDescriptor] = []
                selected_pages: set[int] = set()
                for cluster_id in ordered_cluster_ids.tolist():
                    descriptor = descriptors.get(cluster_id)
                    if descriptor is None:
                        stale_commands += 1
                        continue

                    new_pages = tuple(
                        page_id
                        for page_id in descriptor.page_ids
                        if page_id not in selected_pages
                    )
                    if len(selected_pages) + len(new_pages) > page_capacity:
                        break

                    selected_cluster_ids.append(cluster_id)
                    selected_descriptors.append(descriptor)
                    selected_pages.update(new_pages)

                if not selected_cluster_ids:
                    continue

                max_pages = max(
                    len(descriptor.page_ids) for descriptor in selected_descriptors
                )
                cluster_ids_cpu = torch.tensor(selected_cluster_ids, dtype=torch.int64)
                page_ids_cpu = torch.full(
                    (len(selected_cluster_ids), max_pages),
                    -1,
                    dtype=torch.int64,
                )
                cluster_groups: dict[int, RetroSpecClusterGroup] = {}
                for row_index, (cluster_id, descriptor) in enumerate(
                    zip(selected_cluster_ids, selected_descriptors)
                ):
                    page_count = len(descriptor.page_ids)
                    page_ids_cpu[row_index, :page_count] = torch.tensor(
                        descriptor.page_ids, dtype=torch.int64
                    )
                    cluster_groups[cluster_id] = descriptor.identity.group

                prepared.append(
                    _PreparedResidentPrefetchRecord(
                        layer_name=staged.layer_name,
                        pool=pool,
                        resident_cache=resident_cache,
                        cluster_ids_cpu=cluster_ids_cpu,
                        page_ids_cpu=page_ids_cpu,
                        cluster_groups=cluster_groups,
                    )
                )

        if self.performance_stats is not None and stale_commands:
            self.performance_stats.add_counter(
                "prefetch_stale_commands", stale_commands
            )
        return tuple(prepared)

    @torch.inference_mode()
    def _process_prepared_resident_prefetch(
        self,
        prepared: _PreparedResidentPrefetchRecord,
        execution_stream: torch.cuda.Stream,
    ) -> None:
        (
            source_page_ids,
            source_key_pages,
            source_value_pages,
            transfer_buffer,
            transfer_slot,
        ) = self._stage_resident_pages(prepared.pool, prepared.page_ids_cpu)

        with (
            self._resident_state_lock,
            prepared.resident_cache.mutation_guard(),
            torch.cuda.device(prepared.pool.metadata_device),
        ):
            try:
                access = prepared.resident_cache.admit_staged(
                    cluster_ids=prepared.cluster_ids_cpu,
                    page_ids=prepared.page_ids_cpu,
                    cluster_groups=prepared.cluster_groups,
                    allocated_cluster_ids=self._get_allocated_cluster_ids(
                        prepared.layer_name
                    ),
                    allocated_page_ids=prepared.pool.allocated_page_ids,
                    staging_page_ids=source_page_ids,
                    staging_key_pages=source_key_pages,
                    staging_value_pages=source_value_pages,
                    cluster_ids_cpu=prepared.cluster_ids_cpu,
                    page_ids_cpu=prepared.page_ids_cpu,
                    mutation_stream=execution_stream,
                    lookup_after_admit=False,
                )
            except BaseException:
                prepared.resident_cache.synchronize_pending_copies()
                if transfer_slot is not None:
                    transfer_buffer.release_cpu_slot(transfer_slot, None)
                raise
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
        completed = False
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

            prepared_records = self._prepare_resident_prefetch_wave(staged.records)
            for prepared in prepared_records:
                self._process_prepared_resident_prefetch(
                    prepared, staged.execution_stream
                )
            completed = True
        finally:
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
            self._release_resident_prefetch_slot(staged.slot)

    def _reap_resident_prefetches(
        self,
        layer_names: Sequence[str] | None = None,
        wait: bool = False,
    ) -> None:
        requested_layers = None if layer_names is None else frozenset(layer_names)
        ready: list[_ResidentPrefetchWaveFuture] = []
        with self._resident_prefetch_lock:
            retained: deque[_ResidentPrefetchWaveFuture] = deque()
            while self._resident_prefetch_futures:
                wave = self._resident_prefetch_futures.popleft()
                matches = requested_layers is None or not wave.layer_names.isdisjoint(
                    requested_layers
                )
                if matches and (wait or wave.future.done()):
                    ready.append(wave)
                else:
                    retained.append(wave)
            self._resident_prefetch_futures = retained

        if self.performance_stats is not None:
            self.performance_stats.add_counter("prefetch_reaped_tasks", len(ready))
            self.performance_stats.add_counter("prefetch_reaped_waves", len(ready))
            if wait:
                self.performance_stats.add_counter("prefetch_waited_tasks", len(ready))
                self.performance_stats.add_counter("prefetch_waited_waves", len(ready))

        wait_started_at = (
            perf_counter()
            if wait
            and ready
            and self.performance_stats is not None
            and self.performance_stats.enabled
            else None
        )
        try:
            for wave in ready:
                wave.future.result()
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
        records = tuple(records)
        if not records:
            return False
        if not self.pin_memory:
            return False

        device = self._validate_resident_prefetch_wave(records)
        layer_names = tuple(record.layer_name for record in records)
        self._reap_resident_prefetches(layer_names, wait=False)
        self._try_submit_deferred_resident_prefetch(device, wait_for_slot=False)
        slot = self._acquire_resident_prefetch_slot(device)
        if slot is None:
            self._defer_resident_prefetch_wave(device, records)
            return True

        self._submit_resident_prefetch_wave(records, slot, source_ready_event=None)
        return True

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
                    for _ in range(self._VERIFICATION_MISS_RING_SIZE)
                ],
            )
            arenas = self._verification_gpu_arenas.setdefault(
                device,
                [
                    _VerificationResolveGPUArena()
                    for _ in range(self._VERIFICATION_MISS_RING_SIZE)
                ],
            )
            for slot in slots:
                if slot.in_use and (
                    cluster_capacity > slot.capacity or max_pages > slot.max_pages
                ):
                    raise RuntimeError(
                        "Cannot grow an active verification-miss metadata slot"
                    )
                if not slot.in_use:
                    slot.reserve_capacity(cluster_capacity, max_pages)
            for arena in arenas:
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
            ) % self._VERIFICATION_MISS_RING_SIZE
            slot = self._verification_miss_slots[device][cursor]
            arena = self._verification_gpu_arenas[device][cursor]
            if slot.in_use:
                raise RuntimeError("Verification-miss metadata ring is exhausted")
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

    def _build_verification_miss_metadata(
        self,
        layer_name: str,
        max_pages: int,
        slot: _PinnedVerificationMissSlot,
        num_misses: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            slot.cluster_id_storage is None
            or slot.logical_page_id_storage is None
            or slot.output_page_offset_storage is None
            or slot.source_page_count_storage is None
        ):
            raise RuntimeError("Verification-miss CPU input storage is unavailable")
        if slot.staging_start_storage is None or slot.page_count_storage is None:
            raise RuntimeError("Verification-miss CPU output storage is unavailable")

        cluster_ids = slot.cluster_id_storage[:num_misses].tolist()
        output_page_offsets = slot.output_page_offset_storage[:num_misses].tolist()
        source_page_counts = slot.source_page_count_storage[:num_misses].tolist()
        logical_page_ids = slot.logical_page_id_storage[:num_misses, :max_pages]
        record_order = sorted(
            range(num_misses), key=lambda index: output_page_offsets[index]
        )

        descriptors = self._cluster_block_descriptors.get(layer_name)
        if descriptors is None:
            raise RuntimeError(f"No cluster descriptors exist for {layer_name!r}")
        allocated_cluster_ids = self._get_allocated_cluster_ids(layer_name)

        ordered_cluster_ids: list[int] = []
        staging_start_by_cluster: dict[int, int] = {}
        unique_page_ids: list[int] = []
        for record_index in record_order:
            cluster_id = int(cluster_ids[record_index])
            if cluster_id not in allocated_cluster_ids:
                raise RuntimeError(
                    f"Verification selected stale cluster handle {cluster_id}"
                )
            descriptor = descriptors[cluster_id]
            page_count = int(source_page_counts[record_index])
            emitted_pages = tuple(
                int(page_id)
                for page_id in logical_page_ids[record_index, :page_count].tolist()
            )
            if descriptor.page_ids != emitted_pages:
                raise RuntimeError(
                    f"Verification selected stale pages for cluster {cluster_id}"
                )
            if page_count > max_pages:
                raise RuntimeError(
                    "Verification descriptor exceeds the packed page-table width"
                )
            staging_start = staging_start_by_cluster.get(cluster_id)
            if staging_start is None:
                staging_start = len(unique_page_ids)
                staging_start_by_cluster[cluster_id] = staging_start
                ordered_cluster_ids.append(cluster_id)
                unique_page_ids.extend(emitted_pages)
            slot.staging_start_storage[record_index] = staging_start
            slot.page_count_storage[record_index] = page_count

        cluster_ids_cpu = torch.tensor(ordered_cluster_ids, dtype=torch.int64)
        metadata = self._materialize_cluster_block_metadata_cpu(
            layer_name, cluster_ids_cpu, page_width=max_pages
        )
        staging_page_ids_cpu = torch.full_like(metadata.page_ids, -1)
        for row_index, cluster_id in enumerate(ordered_cluster_ids):
            descriptor = descriptors[cluster_id]
            staging_start = staging_start_by_cluster[cluster_id]
            page_count = len(descriptor.page_ids)
            staging_page_ids_cpu[row_index, :page_count] = torch.arange(
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

    def resolve_draft_cluster_blocks(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache_page_ids: torch.Tensor,
        hit_cluster_mask: torch.Tensor,
        miss_cluster_mask: torch.Tensor,
        hit_gate_ready_mask: torch.Tensor,
        access_kinds: torch.Tensor,
    ) -> RetroSpecResolvedClusterPages:
        """Resolve draft cluster handles entirely on the model device."""
        if cluster_ids.device.type != "cuda":
            raise ValueError("Draft resident lookup requires CUDA")
        if logical_page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Logical pages do not match cluster IDs")
        if active_mask.shape != (cluster_ids.shape[0],):
            raise ValueError("active_mask does not match the draft batch")

        with self._resident_state_lock:
            _, resident_cache = self._get_or_create_resident_cache(layer_name)

        access = resident_cache.lookup_gpu(
            cluster_ids=cluster_ids,
            page_ids=logical_page_ids,
            active_mask=active_mask,
            cache_page_ids=cache_page_ids,
            hit_cluster_mask=hit_cluster_mask,
            miss_cluster_mask=miss_cluster_mask,
            hit_gate_ready_mask=hit_gate_ready_mask,
            access_kinds=access_kinds,
        )
        if self.performance_stats is not None:
            self.performance_stats.add_gpu_counter(
                "resident_cluster_hits", access.hit_cluster_mask
            )
            self.performance_stats.add_gpu_counter(
                "resident_cluster_misses", access.miss_cluster_mask
            )
        return RetroSpecResolvedClusterPages(
            resident_page_ids=access.cache_page_ids,
            staging_page_ids=torch.full_like(logical_page_ids, -1),
            resident_key_pages=resident_cache.key_pages,
            resident_value_pages=resident_cache.value_pages,
            staging_key_pages=resident_cache.key_pages[:0],
            staging_value_pages=resident_cache.value_pages[:0],
            hit_cluster_mask=access.hit_cluster_mask,
            miss_cluster_mask=access.miss_cluster_mask,
            hit_gate_ready_mask=access.hit_gate_ready_mask,
            resident_ready_event=None,
            staging_ready_event=None,
            access_kinds=access.access_kinds,
            read_lease=access.read_lease,
        )

    def resolve_compact_draft_cluster_blocks(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        logical_page_token_counts: torch.Tensor,
        retrieval_scores: torch.Tensor,
        active_mask: torch.Tensor,
        has_clusters: torch.Tensor,
        fallback_token_counts: torch.Tensor,
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
        emit_misses: bool = True,
    ) -> RetroSpecCompactResolvedClusterPages:
        """Resolve DRAFT pages into a compact GPU-only descriptor."""
        if cluster_ids.device.type != "cuda":
            raise ValueError("Compact draft resident lookup requires CUDA")

        with self._resident_state_lock:
            _, resident_cache = self._get_or_create_resident_cache(layer_name)

        access = resident_cache.lookup_compact_draft_gpu(
            cluster_ids=cluster_ids,
            logical_page_ids=logical_page_ids,
            logical_page_token_counts=logical_page_token_counts,
            retrieval_scores=retrieval_scores,
            active_mask=active_mask,
            has_clusters=has_clusters,
            fallback_token_counts=fallback_token_counts,
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
            emit_misses=emit_misses,
        )
        if self.performance_stats is not None:
            self.performance_stats.add_gpu_counter(
                "resident_cluster_hits", access.hit_cluster_counts
            )
            self.performance_stats.add_gpu_counter(
                "resident_cluster_misses", access.miss_cluster_counts
            )
            self.performance_stats.add_gpu_counter(
                "draft_compact_resident_pages", access.page_counts
            )
            self.performance_stats.add_gpu_counter(
                "draft_compact_selected_clusters", access.selected_cluster_counts
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

    def resolve_verification_cluster_blocks(
        self,
        layer_name: str,
        selected_cluster_indices: torch.Tensor,
        plan_row_indices: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
        arena: RetroSpecResidentLayerArena,
        max_pages_per_cluster: int,
    ) -> RetroSpecCompactVerificationResolvedPages:
        """Resolve indexed verification plans into compact query-row pages."""
        if selected_cluster_indices.device.type != "cuda":
            raise ValueError("GPU verification lookup requires CUDA")
        if selected_cluster_indices.ndim != 3:
            raise ValueError("Verification cluster indices must be three-dimensional")
        if plan_row_indices.ndim != 1:
            raise ValueError("plan_row_indices must be one-dimensional")
        if plan_row_indices.device != selected_cluster_indices.device:
            raise ValueError("Indexed plan rows must use the lookup device")
        if request_slot_ids.shape != request_slot_generations.shape:
            raise ValueError("Request slot descriptors must have equal shapes")
        if max_pages_per_cluster < 0:
            raise ValueError("max_pages_per_cluster must be non-negative")

        self.wait_for_resident_prefetches((layer_name,))
        with self._resident_state_lock:
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)

        current_stream = torch.cuda.current_stream(selected_cluster_indices.device)
        resident_cache.wait_for_pending_copies(current_stream)
        num_queries = plan_row_indices.shape[0]
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
                resolve_arena.cluster_ids,
                resolve_arena.logical_page_ids,
                resolve_arena.output_page_offsets,
                resolve_arena.source_page_counts,
                resolve_arena.staging_starts,
                resolve_arena.page_counts,
                resolve_arena.miss_count,
                resolve_arena.invalid_descriptor_count,
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

            assert resolve_arena.cluster_ids is not None
            assert resolve_arena.logical_page_ids is not None
            assert resolve_arena.output_page_offsets is not None
            assert resolve_arena.source_page_counts is not None
            assert resolve_arena.staging_starts is not None
            assert resolve_arena.page_counts is not None
            assert resolve_arena.miss_count is not None
            assert resolve_arena.invalid_descriptor_count is not None
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
                plan_row_indices=plan_row_indices,
                request_slot_ids=request_slot_ids,
                request_slot_generations=request_slot_generations,
                arena_cluster_ids=arena.cluster_ids,
                arena_cluster_page_starts=arena.cluster_page_starts,
                arena_cluster_page_counts=arena.cluster_page_counts,
                arena_page_ids=arena.page_ids,
                arena_page_token_counts=arena.page_token_counts,
                arena_cluster_offsets=arena.cluster_offsets,
                arena_page_offsets=arena.page_offsets,
                arena_generations=arena.generations,
                resident_page_ids=resident_page_ids,
                staging_page_ids=staging_page_ids,
                page_token_counts=page_token_counts,
                page_counts=page_counts,
                selected_cluster_counts=selected_counts,
                hit_cluster_counts=hit_counts,
                miss_cluster_counts=miss_counts,
                miss_cluster_ids=resolve_arena.cluster_ids[:cluster_capacity],
                miss_logical_page_ids=resolve_arena.logical_page_ids[
                    :cluster_capacity, :max_pages_per_cluster
                ],
                miss_page_counts=resolve_arena.source_page_counts[:cluster_capacity],
                miss_output_page_offsets=resolve_arena.output_page_offsets[
                    :cluster_capacity
                ],
                miss_count=resolve_arena.miss_count,
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
                slot.cluster_id_storage,
                slot.logical_page_id_storage,
                slot.output_page_offset_storage,
                slot.source_page_count_storage,
                slot.miss_count_storage,
                slot.invalid_descriptor_count_storage,
            )
            if any(tensor is None for tensor in pinned_required):
                raise RuntimeError("Verification pinned compact slot is unavailable")
            assert slot.cluster_id_storage is not None
            assert slot.logical_page_id_storage is not None
            assert slot.output_page_offset_storage is not None
            assert slot.source_page_count_storage is not None
            assert slot.miss_count_storage is not None
            assert slot.invalid_descriptor_count_storage is not None

            lookup_ready_event = torch.cuda.Event()
            lookup_ready_event.record(current_stream)
            metadata_stream = self._get_verification_metadata_stream(
                selected_cluster_indices.device
            )
            with torch.cuda.stream(metadata_stream):
                metadata_stream.wait_event(lookup_ready_event)
                slot.cluster_id_storage[:cluster_capacity].copy_(
                    access.miss_cluster_ids[:cluster_capacity],
                    non_blocking=self.pin_memory,
                )
                slot.logical_page_id_storage[
                    :cluster_capacity, :max_pages_per_cluster
                ].copy_(
                    access.miss_logical_page_ids[
                        :cluster_capacity, :max_pages_per_cluster
                    ],
                    non_blocking=self.pin_memory,
                )
                slot.output_page_offset_storage[:cluster_capacity].copy_(
                    access.miss_output_page_offsets[:cluster_capacity],
                    non_blocking=self.pin_memory,
                )
                slot.source_page_count_storage[:cluster_capacity].copy_(
                    access.miss_page_counts[:cluster_capacity],
                    non_blocking=self.pin_memory,
                )
                slot.miss_count_storage.copy_(
                    access.miss_count, non_blocking=self.pin_memory
                )
                slot.invalid_descriptor_count_storage.copy_(
                    access.invalid_descriptor_count, non_blocking=self.pin_memory
                )
                metadata_ready_event = torch.cuda.Event()
                metadata_ready_event.record(metadata_stream)

            wait_started_at = (
                perf_counter()
                if self.performance_stats is not None and self.performance_stats.enabled
                else None
            )
            metadata_ready_event.synchronize()
            if wait_started_at is not None:
                self.performance_stats.record_cpu_time(
                    "verification_miss_metadata_wait",
                    perf_counter() - wait_started_at,
                )

            invalid_descriptors = int(slot.invalid_descriptor_count_storage.item())
            if invalid_descriptors:
                raise RuntimeError(
                    "Verification selected a stale request-slot descriptor"
                )
            num_misses = int(slot.miss_count_storage.item())
            if num_misses < 0 or num_misses > cluster_capacity:
                raise RuntimeError("GPU verification miss count is out of bounds")
            if self.performance_stats is not None:
                metadata_bytes = (
                    cluster_capacity
                    * (
                        slot.cluster_id_storage.element_size()
                        + slot.output_page_offset_storage.element_size()
                        + slot.source_page_count_storage.element_size()
                        + max_pages_per_cluster
                        * slot.logical_page_id_storage.element_size()
                    )
                    + slot.miss_count_storage.element_size()
                    + slot.invalid_descriptor_count_storage.element_size()
                )
                self.performance_stats.add_counter(
                    "verification_miss_metadata_d2h_bytes", metadata_bytes
                )

            empty_pages = resident_cache.key_pages[:0]
            if num_misses == 0:
                self._release_verification_miss_slot(slot, None)
                slot = None
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
            ) = self._build_verification_miss_metadata(
                layer_name=layer_name,
                max_pages=max_pages_per_cluster,
                slot=slot,
                num_misses=num_misses,
            )
            if descriptor_started_at is not None:
                self.performance_stats.record_cpu_time(
                    "verification_miss_descriptor_build",
                    perf_counter() - descriptor_started_at,
                )

            assert slot.staging_start_storage is not None
            assert slot.page_count_storage is not None
            resolve_arena.staging_starts[:num_misses].copy_(
                slot.staging_start_storage[:num_misses],
                non_blocking=self.pin_memory,
            )
            resolve_arena.page_counts[:num_misses].copy_(
                slot.page_count_storage[:num_misses], non_blocking=self.pin_memory
            )
            scatter_compact_staging_page_ids(
                miss_output_page_offsets=access.miss_output_page_offsets,
                staging_starts=resolve_arena.staging_starts,
                page_counts=resolve_arena.page_counts,
                num_misses=num_misses,
                max_pages=max_pages_per_cluster,
                output_page_ids=access.staging_page_ids,
            )
            slot_ready_event = torch.cuda.Event()
            slot_ready_event.record(current_stream)
            self._release_verification_miss_slot(slot, slot_ready_event)
            slot = None

            staging_key_pages, staging_value_pages, staging_ready_event = (
                self._stage_verification_miss_pages(pool, unique_page_ids_cpu)
            )
            if self.performance_stats is not None:
                num_unique_misses = miss_cluster_ids_cpu.numel()
                self.performance_stats.add_counter(
                    "verification_unique_miss_clusters", num_unique_misses
                )
                self.performance_stats.add_counter(
                    "verification_duplicate_miss_clusters",
                    num_misses - num_unique_misses,
                )

            admission = RetroSpecVerificationMissAdmission(
                layer_name=layer_name,
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
        except BaseException:
            if slot is not None:
                current_stream.synchronize()
                self._release_verification_miss_slot(slot, None)
            if access is not None:
                access.read_lease.release()
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

    def lookup_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        touch: bool = True,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
                layer_name,
                cluster_ids,
                page_ids,
            )
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)

            return resident_cache.lookup(
                cluster_ids=cluster_ids,
                page_ids=page_ids,
                cluster_groups=self._get_cluster_groups(
                    layer_name,
                    cluster_ids_cpu,
                ),
                allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
                allocated_page_ids=pool.allocated_page_ids,
                touch=touch,
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
            )

    def admit_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
                layer_name,
                cluster_ids,
                page_ids,
            )
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)
            selected_cluster_ids, selected_page_ids = (
                self._select_resident_staging_prefix(
                    pool, cluster_ids_cpu, page_ids_cpu
                )
            )
            (
                source_page_ids,
                source_key_pages,
                source_value_pages,
                transfer_buffer,
                transfer_slot,
            ) = self._stage_resident_pages(pool, selected_page_ids)

            try:
                with resident_cache.mutation_guard():
                    access = resident_cache.admit_staged(
                        cluster_ids=selected_cluster_ids,
                        page_ids=selected_page_ids,
                        cluster_groups=self._get_cluster_groups(
                            layer_name,
                            selected_cluster_ids,
                        ),
                        allocated_cluster_ids=self._get_allocated_cluster_ids(
                            layer_name
                        ),
                        allocated_page_ids=pool.allocated_page_ids,
                        staging_page_ids=source_page_ids,
                        staging_key_pages=source_key_pages,
                        staging_value_pages=source_value_pages,
                        cluster_ids_cpu=selected_cluster_ids,
                        page_ids_cpu=selected_page_ids,
                    )
            except BaseException:
                resident_cache.synchronize_pending_copies()
                if transfer_slot is not None:
                    transfer_buffer.release_cpu_slot(transfer_slot, None)
                raise
            if transfer_slot is not None:
                transfer_buffer.release_cpu_slot(transfer_slot, access.ready_event)
            return resident_cache.lookup(
                cluster_ids=cluster_ids,
                page_ids=page_ids,
                cluster_groups=self._get_cluster_groups(layer_name, cluster_ids_cpu),
                allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
                allocated_page_ids=pool.allocated_page_ids,
                touch=False,
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
            )

    def admit_staged_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
                layer_name,
                cluster_ids,
                logical_page_ids,
            )
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)

            with resident_cache.mutation_guard():
                return resident_cache.admit_staged(
                    cluster_ids=cluster_ids,
                    page_ids=logical_page_ids,
                    cluster_groups=self._get_cluster_groups(
                        layer_name,
                        cluster_ids_cpu,
                    ),
                    allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
                    allocated_page_ids=pool.allocated_page_ids,
                    staging_page_ids=staging_page_ids,
                    staging_key_pages=staging_key_pages,
                    staging_value_pages=staging_value_pages,
                    cluster_ids_cpu=cluster_ids_cpu,
                    page_ids_cpu=page_ids_cpu,
                )

    def admit_verification_misses(
        self,
        admission: RetroSpecVerificationMissAdmission | None,
    ) -> None:
        """Admit compact verification misses without re-reading GPU metadata."""
        if admission is None or admission.cluster_ids_cpu.numel() == 0:
            return

        with self._resident_state_lock:
            pool, resident_cache = self._get_or_create_resident_cache(
                admission.layer_name
            )
            cluster_groups = self._get_cluster_groups(
                admission.layer_name, admission.cluster_ids_cpu
            )
            with resident_cache.mutation_guard():
                resident_cache.admit_staged(
                    cluster_ids=admission.cluster_ids_cpu,
                    page_ids=admission.logical_page_ids_cpu,
                    cluster_groups=cluster_groups,
                    allocated_cluster_ids=self._get_allocated_cluster_ids(
                        admission.layer_name
                    ),
                    allocated_page_ids=pool.allocated_page_ids,
                    staging_page_ids=admission.staging_page_ids_cpu,
                    staging_key_pages=admission.staging_key_pages,
                    staging_value_pages=admission.staging_value_pages,
                    cluster_ids_cpu=admission.cluster_ids_cpu,
                    page_ids_cpu=admission.logical_page_ids_cpu,
                    reuse_ready_event=admission.staging_ready_event,
                    lookup_after_admit=False,
                )

    def get_resident_page_storage(
        self,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return resident GPU pages after waiting on pending H2D copies.

        The wait is inserted into the current CUDA stream and does not block the
        CPU thread.
        """
        _, resident_cache = self._get_or_create_resident_cache(layer_name)
        resident_cache.wait_for_pending_copies()

        return (
            resident_cache.key_pages,
            resident_cache.value_pages,
        )

    def resident_capacity(
        self,
        layer_name: str,
    ) -> int:
        _, resident_cache = self._get_or_create_resident_cache(layer_name)
        return resident_cache.capacity

    def num_resident_pages(
        self,
        layer_name: str,
    ) -> int:
        resident_cache = self._resident_caches.get(layer_name)
        return 0 if resident_cache is None else resident_cache.num_resident_pages

    def num_resident_clusters(
        self,
        layer_name: str,
    ) -> int:
        resident_cache = self._resident_caches.get(layer_name)
        return 0 if resident_cache is None else resident_cache.num_resident_clusters

    def num_resident_groups(
        self,
        layer_name: str,
    ) -> int:
        resident_cache = self._resident_caches.get(layer_name)
        return 0 if resident_cache is None else resident_cache.num_resident_groups

    def num_allocated_pages(
        self,
        layer_name: str,
    ) -> int:
        pool = self._layer_pools.get(layer_name)
        return 0 if pool is None else pool.num_allocated_pages
