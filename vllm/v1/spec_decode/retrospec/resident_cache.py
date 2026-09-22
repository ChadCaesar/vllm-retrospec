# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict, deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from threading import Lock
from time import perf_counter
from types import MappingProxyType

import torch

from vllm import _custom_ops as ops

from .cluster_identity import RetroSpecClusterGroup
from .performance import RetroSpecPerformanceStats
from .resident_kernels import (
    lookup_resident_handles,
    resolve_compact_draft_pages,
    resolve_compact_verification_pages,
    resolve_ranked_draft_buckets,
    update_resident_handles,
)

_ClusterId = int
_LogicalPages = tuple[int, ...]
RetroSpecResidentBindingPublisher = Callable[
    [tuple[int, ...], tuple[int, ...], torch.cuda.Stream], None
]

_PREFETCH_ABSENT = 0
_PREFETCH_PENDING = 1
_PREFETCH_RESIDENT = 2


def _resident_handle_hash(cluster_id: _ClusterId) -> int:
    """Mix every handle bit before masking into the GPU hash table."""
    value = ((cluster_id & 0xFFFFFFFF) ^ (cluster_id >> 32)) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


@dataclass(frozen=True)
class _PendingCopyBatch:
    """Keep asynchronous cache-copy sources alive until completion."""

    ready_event: torch.cuda.Event
    cluster_ids: tuple[_ClusterId, ...]
    source_key_pages: torch.Tensor
    source_value_pages: torch.Tensor


@dataclass
class _ResidentGroupState:
    """Resident replacement state for one request and KV head."""

    lru: OrderedDict[_ClusterId, None] = field(default_factory=OrderedDict)
    num_pages: int = 0


@dataclass(frozen=True)
class RetroSpecResidentPageAccess:
    """Result of resolving logical cluster pages against the GPU cache."""

    cache_page_ids: torch.Tensor
    hit_cluster_mask: torch.Tensor
    miss_cluster_mask: torch.Tensor
    hit_gate_ready_mask: torch.Tensor
    logical_page_ids_cpu: torch.Tensor | None
    miss_cluster_mask_cpu: torch.Tensor | None
    ready_event: torch.cuda.Event | None
    access_kinds: torch.Tensor | None = None
    read_lease: "RetroSpecResidentReadLease | None" = None


@dataclass(frozen=True)
class RetroSpecPreparedResidentAdmission:
    """Immutable CPU descriptors prepared before taking the mutation guard."""

    cluster_ids: torch.Tensor
    page_ids: torch.Tensor
    cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup]
    source_page_ids: torch.Tensor
    source_key_pages: torch.Tensor
    source_value_pages: torch.Tensor
    cluster_ids_cpu: torch.Tensor
    page_ids_cpu: torch.Tensor
    parsed_cluster_ids: tuple[_ClusterId | None, ...]
    parsed_cluster_pages: tuple[_LogicalPages, ...]
    valid_positions: tuple[tuple[int, ...], ...]
    requested_clusters: tuple[_ClusterId, ...]
    cluster_page_map: Mapping[_ClusterId, _LogicalPages]
    cluster_source_ids: Mapping[_ClusterId, tuple[int, ...]]
    referenced_cluster_ids: frozenset[_ClusterId]
    referenced_page_ids: frozenset[int]


@dataclass(frozen=True)
class RetroSpecResidentLruCapture:
    """Immutable resident descriptors captured while structure is stable."""

    structure_revision: int
    groups: tuple[RetroSpecClusterGroup, ...]
    epoch_table: torch.Tensor = field(repr=False, compare=False)
    stream: torch.cuda.Stream = field(repr=False, compare=False)


@dataclass(frozen=True)
class RetroSpecResolvedResidentLru:
    """CPU-resolved LRU epoch snapshot awaiting revision validation."""

    structure_revision: int
    groups: tuple[RetroSpecClusterGroup, ...]
    epochs: tuple[int, ...]


@dataclass(frozen=True)
class RetroSpecCompactResidentPageAccess:
    """Row-local compact resident pages and fused DRAFT statistics."""

    cache_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    clustered_token_counts: torch.Tensor
    attention_mass: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    hit_gate_ready: torch.Tensor
    access_kinds: torch.Tensor | None
    read_lease: "RetroSpecResidentReadLease"


@dataclass(frozen=True)
class RetroSpecRankedDraftResidentAccess:
    """Stable resident-table view for one ranked DRAFT selection."""

    cluster_handles: torch.Tensor
    resident_bucket_ids: torch.Tensor
    clustered_token_counts: torch.Tensor
    attention_mass: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    hit_gate_ready: torch.Tensor
    resident_table_page_counts: torch.Tensor
    resident_table_page_slots: torch.Tensor
    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor
    read_lease: "RetroSpecResidentReadLease"


@dataclass(frozen=True)
class RetroSpecCompactVerificationPageAccess:
    """Query-row compact resident pages and GPU-unique verification misses."""

    resident_page_ids: torch.Tensor
    staging_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    unique_cluster_ids: torch.Tensor
    unique_logical_page_ids: torch.Tensor
    unique_page_counts: torch.Tensor
    miss_unique_indices: torch.Tensor
    miss_output_page_offsets: torch.Tensor
    miss_count: torch.Tensor
    unique_miss_count: torch.Tensor
    invalid_descriptor_count: torch.Tensor
    read_lease: "RetroSpecResidentReadLease"


class RetroSpecResidentReadLease:
    """Keep resident slots stable until their attention kernels are submitted."""

    def __init__(self, lock: Lock) -> None:
        self._lock = lock
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._lock.release()


class RetroSpecResidentClusterCache:
    """Shared GPU page arena with group-scoped replacement state.

    Logical page IDs address the stable CPU backing store. Resident page IDs
    address slots in the shared key_pages and value_pages tensors.

    Every cluster belongs to one request/KV-head group. Each group owns an
    independent LRU and a soft resident-page target. Physical GPU slots remain
    shared by the whole layer, so unused group capacity may be borrowed.
    Admission and eviction always operate on complete clusters.
    """

    def __init__(
        self,
        page_size: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        performance_stats: RetroSpecPerformanceStats | None = None,
        binding_publisher: RetroSpecResidentBindingPublisher | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if head_size <= 0:
            raise ValueError("head_size must be positive")
        if device.type != "cuda":
            raise ValueError("Resident cluster cache requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.page_size = page_size
        self.head_size = head_size
        self.dtype = dtype
        self.device = device
        self.performance_stats = performance_stats
        self._binding_publisher = binding_publisher

        self.key_pages = torch.empty(
            0,
            page_size,
            head_size,
            dtype=dtype,
            device=device,
        )
        self.value_pages = torch.empty_like(self.key_pages)

        self._logical_capacity = 0
        self._physical_capacity = 0

        # Physical page and slot ownership remains layer-wide.
        self._cluster_to_pages: dict[_ClusterId, _LogicalPages] = {}
        self._cluster_to_slots: dict[_ClusterId, tuple[int, ...]] = {}
        self._page_to_cluster: dict[int, _ClusterId] = {}
        self._free_slots: set[int] = set()

        # Semantic ownership and recency are isolated by request/KV-head group.
        self._cluster_to_group: dict[_ClusterId, RetroSpecClusterGroup] = {}
        self._group_states: dict[RetroSpecClusterGroup, _ResidentGroupState] = {}

        # Soft resident-page targets. Physical slots remain shared and groups may
        # temporarily borrow unused capacity from one another.
        self._group_targets: dict[RetroSpecClusterGroup, int] = {}

        self._copy_stream = torch.cuda.Stream(device=device)
        self._pending_copy_batches: deque[_PendingCopyBatch] = deque()
        self._pending_cluster_events: dict[_ClusterId, torch.cuda.Event] = {}

        # GPU draft lookups and resident mutations share this short host-side
        # guard. It is held only while kernels are submitted, never while CUDA
        # work completes or CPU descriptors are parsed.
        self._gpu_access_lock = Lock()

        # Open-addressed GPU handle table. Stable cluster IDs are the keys;
        # versions protect readers from concurrent publication on the H2D
        # stream. CPU shadows are used only by the background mutation path.
        self._handle_table_capacity = 0
        self._handle_table_max_pages = 0
        self._handle_table_handles = torch.empty(0, dtype=torch.int64, device=device)
        self._handle_table_versions = torch.empty(0, dtype=torch.int32, device=device)
        self._handle_table_page_counts = torch.empty(
            0, dtype=torch.int32, device=device
        )
        self._handle_table_page_slots = torch.empty(
            (0, 0), dtype=torch.int32, device=device
        )
        self._handle_table_hit_gate_ready = torch.empty(
            0, dtype=torch.bool, device=device
        )
        self._handle_table_last_access_epochs = torch.empty(
            0, dtype=torch.int64, device=device
        )
        self._next_access_epoch = 1
        self._handle_to_bucket: dict[_ClusterId, int] = {}
        self._bucket_handles: list[int] = []
        self._handle_table_needs_rebuild = False
        self._prefetch_handle_states = torch.zeros(0, dtype=torch.uint8, device="cpu")
        self._structure_revision = 0

    @property
    def capacity(self) -> int:
        """Maximum number of resident pages currently permitted."""
        return self._logical_capacity

    @property
    def physical_capacity(self) -> int:
        """Number of physically allocated GPU page slots."""
        return self._physical_capacity

    @property
    def num_resident_pages(self) -> int:
        return len(self._page_to_cluster)

    @property
    def num_resident_clusters(self) -> int:
        return len(self._cluster_to_slots)

    @property
    def num_resident_groups(self) -> int:
        return len(self._group_states)

    @property
    def num_pending_copy_batches(self) -> int:
        """Number of submitted copy batches not yet reaped by the host."""
        self._reap_completed_copy_batches()
        return len(self._pending_copy_batches)

    def requires_resize(
        self,
        capacity: int,
        group_targets: Mapping[RetroSpecClusterGroup, int] | None = None,
    ) -> bool:
        targets = self._group_targets if group_targets is None else group_targets
        return capacity != self._logical_capacity or targets != self._group_targets

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 if value <= 1 else 1 << (value - 1).bit_length()

    def mutation_guard(self) -> Lock:
        """Return the short guard shared by GPU readers and cache mutations."""
        return self._gpu_access_lock

    def _bump_structure_revision(self) -> None:
        self._structure_revision += 1

    def partition_admission_candidates(
        self,
        cluster_ids: Collection[_ClusterId],
    ) -> tuple[tuple[_ClusterId, ...], int, int]:
        """Separate absent clusters from resident and pending clusters.

        The caller must hold ``mutation_guard()``. Input priority order is
        preserved in the returned admission candidates.
        """
        self._reap_completed_copy_batches()

        admission_candidates: list[_ClusterId] = []
        resident_count = 0
        pending_count = 0
        for cluster_id in cluster_ids:
            if cluster_id not in self._cluster_to_slots:
                admission_candidates.append(cluster_id)
            elif cluster_id in self._pending_cluster_events:
                pending_count += 1
            else:
                resident_count += 1

        return tuple(admission_candidates), resident_count, pending_count

    def reserve_prefetch_handle_states(self, required_capacity: int) -> None:
        if required_capacity < 0:
            raise ValueError("Prefetch handle-state capacity must be non-negative")
        if required_capacity <= self._prefetch_handle_states.numel():
            return

        capacity = self._next_power_of_two(max(required_capacity, 1))
        # This shadow is updated by the background worker after CUDA events
        # complete, which can happen outside the model runner's inference-mode
        # scope. Keep its storage as a normal tensor even when the cache is
        # first initialized from an inference-mode prefill call.
        with torch.inference_mode(False):
            states = torch.zeros(capacity, dtype=torch.uint8, device="cpu")
            states[: self._prefetch_handle_states.numel()].copy_(
                self._prefetch_handle_states
            )
        self._prefetch_handle_states = states

    def _set_prefetch_handle_state(
        self, cluster_ids: Collection[_ClusterId], state: int
    ) -> None:
        cluster_ids = tuple(dict.fromkeys(cluster_ids))
        if not cluster_ids:
            return
        if state not in (
            _PREFETCH_ABSENT,
            _PREFETCH_PENDING,
            _PREFETCH_RESIDENT,
        ):
            raise ValueError("Invalid resident prefetch state")

        self.reserve_prefetch_handle_states(max(cluster_ids) + 1)
        indices = torch.tensor(cluster_ids, dtype=torch.int64, device="cpu")
        self._prefetch_handle_states.index_fill_(0, indices, state)

    def prefetch_handle_states(self, required_capacity: int) -> torch.Tensor:
        """Return the CPU planner shadow after reaping completed H2D copies."""
        self._reap_completed_copy_batches()
        self.reserve_prefetch_handle_states(required_capacity)
        return self._prefetch_handle_states[:required_capacity]

    def _find_handle_bucket(self, cluster_id: _ClusterId) -> int | None:
        if self._handle_table_capacity == 0:
            return None

        mask = self._handle_table_capacity - 1
        first_tombstone: int | None = None
        first_bucket = _resident_handle_hash(cluster_id) & mask
        for probe in range(64):
            bucket = (first_bucket + probe) & mask
            stored_handle = self._bucket_handles[bucket]
            if stored_handle == cluster_id:
                return bucket
            if stored_handle == -2 and first_tombstone is None:
                first_tombstone = bucket
            if stored_handle == -1:
                return bucket if first_tombstone is None else first_tombstone
        return first_tombstone

    def _allocate_handle_table(
        self,
        capacity: int,
        max_pages_per_cluster: int,
    ) -> None:
        self._handle_table_capacity = capacity
        self._handle_table_max_pages = max_pages_per_cluster
        self._handle_table_handles = torch.full(
            (capacity,), -1, dtype=torch.int64, device=self.device
        )
        self._handle_table_versions = torch.zeros(
            capacity, dtype=torch.int32, device=self.device
        )
        self._handle_table_page_counts = torch.zeros(
            capacity, dtype=torch.int32, device=self.device
        )
        self._handle_table_page_slots = torch.full(
            (capacity, max_pages_per_cluster),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._handle_table_hit_gate_ready = torch.zeros(
            capacity, dtype=torch.bool, device=self.device
        )
        self._handle_table_last_access_epochs = torch.zeros(
            capacity, dtype=torch.int64, device=self.device
        )
        self._handle_to_bucket.clear()
        self._bucket_handles = [-1] * capacity
        self._handle_table_needs_rebuild = False
        self._bump_structure_revision()

    def _ensure_handle_table(
        self,
        max_pages_per_cluster: int,
        stream: torch.cuda.Stream | None = None,
    ) -> bool:
        required_capacity = self._next_power_of_two(max(2, self._logical_capacity * 2))
        requires_rebuild = (
            self._handle_table_needs_rebuild
            or required_capacity > self._handle_table_capacity
            or max_pages_per_cluster > self._handle_table_max_pages
        )
        if not requires_rebuild:
            return False

        previous_max_pages = self._handle_table_max_pages
        # A pending publication targets the current table allocation. Complete
        # it before replacing that allocation, then republish every live entry.
        self.synchronize_pending_copies()
        update_stream = (
            torch.cuda.current_stream(self.device) if stream is None else stream
        )
        if self._handle_table_capacity and self._group_states:
            self._refresh_group_lru_from_gpu(self._group_states.keys(), update_stream)
        with torch.cuda.stream(update_stream):
            self._allocate_handle_table(
                max(required_capacity, self._handle_table_capacity),
                max(max_pages_per_cluster, self._handle_table_max_pages),
            )
        entries = tuple(
            (
                cluster_id,
                self._cluster_to_slots[cluster_id],
                self._cluster_to_group[cluster_id],
            )
            for cluster_id in self._cluster_to_slots
        )
        published = self._write_handle_entries(entries, update_stream)

        # Readers immediately switch to the new tensor references after this
        # method returns. Complete this rare rebuild before releasing the
        # mutation guard so they cannot observe an uninitialized table.
        update_stream.synchronize()
        if published != len(entries):
            raise RuntimeError(
                "Resident handle table rebuild did not publish every live cluster"
            )

        stats = self.performance_stats
        if stats is not None and stats.enabled:
            stats.add_counter("resident_handle_table_rebuilds")
            if self._handle_table_max_pages > previous_max_pages:
                stats.add_counter("resident_handle_table_width_growths")
            stats.observe_peak(
                "resident_handle_table_max_pages",
                self._handle_table_max_pages,
            )
        return True

    def _publish_handle_entries(
        self,
        entries: Collection[tuple[_ClusterId, tuple[int, ...], RetroSpecClusterGroup]],
        stream: torch.cuda.Stream,
    ) -> int:
        entries = tuple(entries)
        if not entries:
            return 0

        max_pages_per_cluster = max(len(slots) for _, slots, _ in entries)
        if self._ensure_handle_table(max_pages_per_cluster, stream):
            # A rebuild republishes every live cluster, including this delta.
            return len(entries)

        return self._write_handle_entries(entries, stream)

    def _write_handle_entries(
        self,
        entries: Collection[tuple[_ClusterId, tuple[int, ...], RetroSpecClusterGroup]],
        stream: torch.cuda.Stream,
    ) -> int:
        entries = tuple(entries)
        if not entries or self._handle_table_capacity == 0:
            return 0

        bucket_ids: list[int] = []
        cluster_ids: list[int] = []
        page_counts: list[int] = []
        page_slots: list[list[int]] = []
        hit_gate_ready: list[bool] = []
        new_entry_indices: list[int] = []

        for cluster_id, slots, group in entries:
            if len(slots) > self._handle_table_max_pages:
                raise RuntimeError(
                    f"Resident cluster {cluster_id} has {len(slots)} pages, "
                    f"but the handle table width is "
                    f"{self._handle_table_max_pages}"
                )

            bucket = self._handle_to_bucket.get(cluster_id)
            if bucket is None:
                bucket = self._find_handle_bucket(cluster_id)
                if bucket is None:
                    self._handle_table_needs_rebuild = True
                    continue
                self._handle_to_bucket[cluster_id] = bucket
                self._bucket_handles[bucket] = cluster_id
                new_entry_indices.append(len(bucket_ids))

            padded_slots = list(slots)
            padded_slots.extend(
                [-1] * (self._handle_table_max_pages - len(padded_slots))
            )
            bucket_ids.append(bucket)
            cluster_ids.append(cluster_id)
            page_counts.append(len(slots))
            page_slots.append(padded_slots)
            hit_gate_ready.append(self._is_group_hit_gate_ready(group))

        if not cluster_ids:
            return 0

        with torch.cuda.stream(stream):
            bucket_ids_gpu = torch.tensor(
                bucket_ids, dtype=torch.int32, device=self.device
            )
            update_resident_handles(
                bucket_ids=bucket_ids_gpu,
                cluster_handles=torch.tensor(
                    cluster_ids, dtype=torch.int64, device=self.device
                ),
                page_counts=torch.tensor(
                    page_counts, dtype=torch.int32, device=self.device
                ),
                page_slots=torch.tensor(
                    page_slots, dtype=torch.int32, device=self.device
                ),
                hit_gate_ready=torch.tensor(
                    hit_gate_ready, dtype=torch.bool, device=self.device
                ),
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
            )
            if new_entry_indices:
                new_bucket_ids = bucket_ids_gpu.index_select(
                    0,
                    torch.tensor(
                        new_entry_indices,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )
                self._handle_table_last_access_epochs.index_fill_(
                    0, new_bucket_ids.to(torch.int64), self._next_access_epoch
                )
            if self._binding_publisher is not None:
                self._binding_publisher(
                    tuple(cluster_ids),
                    tuple(bucket_ids),
                    stream,
                )

        return len(cluster_ids)

    def _erase_handle_entries(
        self,
        cluster_ids: Collection[_ClusterId],
        stream: torch.cuda.Stream,
    ) -> None:
        if self._handle_table_capacity == 0:
            return

        bucket_ids: list[int] = []
        erased_cluster_ids: list[int] = []
        for cluster_id in cluster_ids:
            bucket = self._handle_to_bucket.pop(cluster_id, None)
            if bucket is None:
                continue
            self._bucket_handles[bucket] = -2
            erased_cluster_ids.append(cluster_id)
            bucket_ids.append(bucket)

        if not bucket_ids:
            return

        num_entries = len(bucket_ids)
        with torch.cuda.stream(stream):
            bucket_ids_gpu = torch.tensor(
                bucket_ids, dtype=torch.int32, device=self.device
            )
            update_resident_handles(
                bucket_ids=bucket_ids_gpu,
                cluster_handles=torch.full(
                    (num_entries,), -2, dtype=torch.int64, device=self.device
                ),
                page_counts=torch.zeros(
                    num_entries, dtype=torch.int32, device=self.device
                ),
                page_slots=torch.full(
                    (num_entries, self._handle_table_max_pages),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                hit_gate_ready=torch.zeros(
                    num_entries, dtype=torch.bool, device=self.device
                ),
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
            )
            self._handle_table_last_access_epochs.index_fill_(
                0, bucket_ids_gpu.to(torch.int64), 0
            )
            if self._binding_publisher is not None:
                self._binding_publisher(
                    tuple(erased_cluster_ids),
                    (-1,) * len(erased_cluster_ids),
                    stream,
                )

    def republish_handle_bindings(
        self,
        cluster_ids: Collection[_ClusterId],
        stream: torch.cuda.Stream,
    ) -> None:
        """Replay live buckets after a matching index segment is published."""
        if self._binding_publisher is None:
            return

        published_cluster_ids: list[int] = []
        published_buckets: list[int] = []
        seen: set[int] = set()
        for cluster_id in cluster_ids:
            if cluster_id < 0 or cluster_id in seen:
                continue
            seen.add(cluster_id)
            bucket = self._handle_to_bucket.get(cluster_id)
            if bucket is None:
                continue
            published_cluster_ids.append(cluster_id)
            published_buckets.append(bucket)

        if published_cluster_ids:
            self._binding_publisher(
                tuple(published_cluster_ids),
                tuple(published_buckets),
                stream,
            )

    def _snapshot_group_hit_gate_ready(
        self, groups: Collection[RetroSpecClusterGroup]
    ) -> dict[RetroSpecClusterGroup, bool]:
        return {group: self._is_group_hit_gate_ready(group) for group in set(groups)}

    def _publish_handle_delta(
        self,
        new_cluster_ids: Collection[_ClusterId],
        previous_gate_ready: Mapping[RetroSpecClusterGroup, bool],
        stream: torch.cuda.Stream,
    ) -> int:
        entries: list[tuple[_ClusterId, tuple[int, ...], RetroSpecClusterGroup]] = []
        seen: set[_ClusterId] = set()

        for cluster_id in new_cluster_ids:
            slots = self._cluster_to_slots.get(cluster_id)
            group = self._cluster_to_group.get(cluster_id)
            if slots is None or group is None or cluster_id in seen:
                continue
            entries.append((cluster_id, slots, group))
            seen.add(cluster_id)

        ordered_groups = sorted(
            previous_gate_ready,
            key=lambda group: (group.request_id, group.kv_head_index),
        )
        for group in ordered_groups:
            if previous_gate_ready[group] == self._is_group_hit_gate_ready(group):
                continue
            group_state = self._group_states.get(group)
            if group_state is None:
                continue
            for cluster_id in group_state.lru:
                if cluster_id in seen:
                    continue
                entries.append((cluster_id, self._cluster_to_slots[cluster_id], group))
                seen.add(cluster_id)

        published = self._publish_handle_entries(entries, stream)
        stats = self.performance_stats
        if (
            stats is not None
            and stats.enabled
            and stats.cuda_timing_level == "detailed"
        ):
            stats.add_counter("resident_handle_entries_published", published)
        return published

    def _reap_completed_copy_batches(self) -> None:
        """Release sources and pending markers for completed H2D batches."""
        while (
            self._pending_copy_batches
            and self._pending_copy_batches[0].ready_event.query()
        ):
            batch = self._pending_copy_batches.popleft()

            completed_cluster_ids: list[_ClusterId] = []
            for cluster_id in batch.cluster_ids:
                pending_event = self._pending_cluster_events.get(cluster_id)
                if pending_event is not batch.ready_event:
                    continue
                del self._pending_cluster_events[cluster_id]
                if cluster_id in self._cluster_to_slots:
                    completed_cluster_ids.append(cluster_id)
            self._set_prefetch_handle_state(completed_cluster_ids, _PREFETCH_RESIDENT)

    def _record_copy_batch(
        self,
        cluster_ids: tuple[_ClusterId, ...],
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> None:
        ready_event = torch.cuda.Event()
        ready_event.record(self._copy_stream)

        batch = _PendingCopyBatch(
            ready_event=ready_event,
            cluster_ids=cluster_ids,
            source_key_pages=source_key_pages,
            source_value_pages=source_value_pages,
        )
        self._pending_copy_batches.append(batch)

        for cluster_id in cluster_ids:
            self._pending_cluster_events[cluster_id] = ready_event

    def pending_copy_event(self) -> torch.cuda.Event | None:
        """Return the latest outstanding resident-copy event.

        Waiting for the latest event is sufficient because every admission copy
        is submitted to the same CUDA copy stream. Stream ordering guarantees
        that all earlier admission copies complete before the latest event.

        Completed batches are reaped first so callers do not insert unnecessary
        stream waits.
        """
        self._reap_completed_copy_batches()

        if not self._pending_copy_batches:
            return None

        return self._pending_copy_batches[-1].ready_event

    def _pending_event_for_clusters(
        self,
        cluster_ids: Collection[_ClusterId],
    ) -> torch.cuda.Event | None:
        """Return the last pending event that contains a selected cluster."""
        selected = set(cluster_ids)
        ready_event: torch.cuda.Event | None = None

        for batch in self._pending_copy_batches:
            if selected.intersection(batch.cluster_ids):
                ready_event = batch.ready_event

        return ready_event

    def wait_for_pending_copies(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Make a CUDA consumer stream wait for all submitted cache copies.

        This only inserts a device-side dependency. It does not block the CPU.
        """
        self._reap_completed_copy_batches()
        if not self._pending_copy_batches:
            return

        consumer_stream = (
            torch.cuda.current_stream(self.device) if stream is None else stream
        )
        consumer_stream.wait_event(self._pending_copy_batches[-1].ready_event)

    def synchronize_pending_copies(self) -> None:
        """Synchronize pending copies before CPU backing pages are reused."""
        self._reap_completed_copy_batches()
        if not self._pending_copy_batches:
            self._pending_cluster_events.clear()
            return

        # Every batch uses one copy stream, so completion of the final event also
        # implies completion of all earlier batches.
        pending_cluster_ids = tuple(self._pending_cluster_events)
        self._pending_copy_batches[-1].ready_event.synchronize()
        self._pending_copy_batches.clear()
        self._pending_cluster_events.clear()
        self._set_prefetch_handle_state(pending_cluster_ids, _PREFETCH_RESIDENT)

    def _grow_storage(self, required_capacity: int) -> None:
        if required_capacity <= self._physical_capacity:
            return

        # Old resident tensors may still be destinations of H2D copies. Make the
        # current stream wait before copying them into the enlarged allocation.
        self.wait_for_pending_copies()

        new_key_pages = torch.empty(
            required_capacity,
            self.page_size,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        new_value_pages = torch.empty_like(new_key_pages)

        if self._physical_capacity:
            new_key_pages[: self._physical_capacity].copy_(self.key_pages)
            new_value_pages[: self._physical_capacity].copy_(self.value_pages)

            # Resizing can run on the prefetch worker's per-thread default
            # stream. Subsequent admissions use _copy_stream, so preserve the
            # old resident contents before that stream writes the new arena.
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.device))

        self._free_slots.update(
            range(
                self._physical_capacity,
                required_capacity,
            )
        )

        self.key_pages = new_key_pages
        self.value_pages = new_value_pages
        self._physical_capacity = required_capacity

    def _register_cluster(
        self,
        cluster_id: _ClusterId,
        group: RetroSpecClusterGroup,
        logical_pages: _LogicalPages,
        slots: tuple[int, ...],
    ) -> None:
        """Register one newly resident cluster in its group and shared arena."""
        self._register_clusters(((cluster_id, group, logical_pages, slots),))

    def _register_clusters(
        self,
        entries: Collection[
            tuple[
                _ClusterId,
                RetroSpecClusterGroup,
                _LogicalPages,
                tuple[int, ...],
            ]
        ],
    ) -> None:
        """Atomically validate and register one resident admission batch."""
        entries = tuple(entries)
        if not entries:
            return

        cluster_ids: set[_ClusterId] = set()
        logical_page_ids: set[int] = set()
        slot_ids: set[int] = set()
        for cluster_id, _, logical_pages, slots in entries:
            if cluster_id in cluster_ids:
                raise RuntimeError(
                    f"Resident cluster {cluster_id} occurs more than once in a batch"
                )
            if cluster_id in self._cluster_to_slots:
                raise RuntimeError(
                    f"Resident cluster {cluster_id} is already registered"
                )
            if len(logical_pages) != len(slots):
                raise RuntimeError(
                    "Resident cluster pages and GPU slots must have equal lengths"
                )

            cluster_ids.add(cluster_id)
            for logical_page_id in logical_pages:
                if logical_page_id in logical_page_ids:
                    raise RuntimeError(
                        f"Logical page {logical_page_id} occurs more than once "
                        "in a resident batch"
                    )
                owner = self._page_to_cluster.get(logical_page_id)
                if owner is not None:
                    raise RuntimeError(
                        f"Logical page {logical_page_id} is already owned by {owner}"
                    )
                logical_page_ids.add(logical_page_id)

            for slot_id in slots:
                if slot_id < 0 or slot_id >= self._physical_capacity:
                    raise RuntimeError(f"Resident slot {slot_id} is out of range")
                if slot_id in slot_ids:
                    raise RuntimeError(
                        f"Resident slot {slot_id} occurs more than once in a batch"
                    )
                slot_ids.add(slot_id)

        for cluster_id, group, logical_pages, slots in entries:
            self._cluster_to_pages[cluster_id] = logical_pages
            self._cluster_to_slots[cluster_id] = slots
            self._cluster_to_group[cluster_id] = group

            group_state = self._group_states.setdefault(group, _ResidentGroupState())
            group_state.lru[cluster_id] = None
            group_state.num_pages += len(slots)
            for logical_page_id in logical_pages:
                self._page_to_cluster[logical_page_id] = cluster_id

        self._set_prefetch_handle_state(cluster_ids, _PREFETCH_PENDING)
        self._bump_structure_revision()

    def _touch_cluster(self, cluster_id: _ClusterId) -> None:
        """Mark a resident cluster as recent within its owning group."""
        group = self._cluster_to_group.get(cluster_id)
        if group is None:
            raise RuntimeError(
                f"Resident cluster {cluster_id} does not have an owning group"
            )

        group_state = self._group_states.get(group)
        if group_state is None or cluster_id not in group_state.lru:
            raise RuntimeError(f"Resident group state is missing cluster {cluster_id}")

        if next(reversed(group_state.lru)) != cluster_id:
            group_state.lru.move_to_end(cluster_id, last=True)
            self._bump_structure_revision()

    def _is_group_hit_gate_ready(self, group: RetroSpecClusterGroup) -> bool:
        """Return whether one request/head LRU reached its soft target."""
        target_pages = self._group_targets.get(group, 0)
        if target_pages <= 0:
            return False

        group_state = self._group_states.get(group)
        return group_state is not None and group_state.num_pages >= target_pages

    def _evict_cluster(
        self,
        cluster_id: _ClusterId,
        update_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._evict_clusters((cluster_id,), update_stream)

    def _evict_clusters(
        self,
        cluster_ids: Collection[_ClusterId],
        update_stream: torch.cuda.Stream | None = None,
    ) -> set[RetroSpecClusterGroup]:
        """Evict a validated cluster batch with one GPU table mutation."""
        entries: list[
            tuple[
                _ClusterId,
                RetroSpecClusterGroup,
                _LogicalPages,
                tuple[int, ...],
            ]
        ] = []
        seen: set[_ClusterId] = set()
        for cluster_id in cluster_ids:
            if cluster_id in seen:
                continue
            seen.add(cluster_id)

            slots = self._cluster_to_slots.get(cluster_id)
            if slots is None:
                continue
            if cluster_id in self._pending_cluster_events:
                raise RuntimeError(
                    f"Cannot evict pending resident cluster {cluster_id}"
                )

            logical_pages = self._cluster_to_pages[cluster_id]
            group = self._cluster_to_group[cluster_id]
            group_state = self._group_states.get(group)
            if group_state is None or cluster_id not in group_state.lru:
                raise RuntimeError(
                    f"Resident group state is missing cluster {cluster_id}"
                )
            entries.append((cluster_id, group, logical_pages, slots))

        if not entries:
            return set()

        if update_stream is None:
            update_stream = torch.cuda.current_stream(self.device)
        evicted_cluster_ids = tuple(entry[0] for entry in entries)
        self._set_prefetch_handle_state(evicted_cluster_ids, _PREFETCH_ABSENT)
        self._erase_handle_entries(evicted_cluster_ids, update_stream)

        affected_groups: set[RetroSpecClusterGroup] = set()
        for cluster_id, group, logical_pages, slots in entries:
            del self._cluster_to_pages[cluster_id]
            del self._cluster_to_slots[cluster_id]
            del self._cluster_to_group[cluster_id]

            group_state = self._group_states[group]
            del group_state.lru[cluster_id]
            group_state.num_pages -= len(slots)
            if group_state.num_pages < 0:
                raise RuntimeError("Resident group page count became negative")
            affected_groups.add(group)

            for logical_page_id in logical_pages:
                self._page_to_cluster.pop(logical_page_id, None)
            self._free_slots.update(slots)

            if not group_state.lru:
                if group_state.num_pages != 0:
                    raise RuntimeError("An empty resident group still owns GPU pages")
                del self._group_states[group]

        self._bump_structure_revision()
        return affected_groups

    @staticmethod
    def _oldest_unprotected_cluster(
        group_state: _ResidentGroupState,
        protected_clusters: set[_ClusterId],
    ) -> _ClusterId | None:
        for cluster_id in group_state.lru:
            if cluster_id not in protected_clusters:
                return cluster_id

        return None

    def _capture_group_lru_from_gpu(
        self,
        groups: Collection[RetroSpecClusterGroup],
        stream: torch.cuda.Stream,
    ) -> RetroSpecResidentLruCapture:
        """Capture the epoch allocation guarded by a structural revision."""
        return RetroSpecResidentLruCapture(
            structure_revision=self._structure_revision,
            groups=tuple(groups),
            epoch_table=self._handle_table_last_access_epochs,
            stream=stream,
        )

    @staticmethod
    def resolve_lru_capture(
        capture: RetroSpecResidentLruCapture,
    ) -> RetroSpecResolvedResidentLru:
        """Gather and resolve epochs without holding resident locks."""
        with (
            torch.cuda.device(capture.epoch_table.device),
            torch.cuda.stream(capture.stream),
        ):
            epochs_cpu = capture.epoch_table.to(device="cpu", dtype=torch.int64)

        return RetroSpecResolvedResidentLru(
            structure_revision=capture.structure_revision,
            groups=capture.groups,
            epochs=tuple(epochs_cpu.tolist()),
        )

    def _apply_resolved_group_lru(
        self,
        resolved: RetroSpecResolvedResidentLru,
    ) -> bool:
        """Apply a resolved snapshot if no structural mutation intervened."""
        if resolved.structure_revision != self._structure_revision:
            return False

        def cluster_epoch(cluster_id: _ClusterId) -> int:
            if cluster_id in self._pending_cluster_events:
                return 0
            bucket = self._handle_to_bucket.get(cluster_id)
            if bucket is None or bucket >= len(resolved.epochs):
                return 0
            return resolved.epochs[bucket]

        changed = False
        for group in resolved.groups:
            group_state = self._group_states.get(group)
            if group_state is None:
                continue
            order = tuple(group_state.lru)
            previous_positions = {
                cluster_id: position for position, cluster_id in enumerate(order)
            }
            ranked = tuple(
                sorted(
                    order,
                    key=lambda cluster_id: (
                        cluster_epoch(cluster_id),
                        previous_positions[cluster_id],
                    ),
                )
            )
            if ranked != order:
                self._group_states[group].lru = OrderedDict.fromkeys(ranked)
                changed = True

        if changed:
            self._bump_structure_revision()
        return True

    def _refresh_group_lru_from_gpu(
        self,
        groups: Collection[RetroSpecClusterGroup],
        stream: torch.cuda.Stream,
    ) -> None:
        """Synchronously refresh group-local LRU from GPU-recorded epochs.

        This is intentionally called only when an admission or resize must
        evict pages. Draft hits therefore remain entirely on the GPU hot path.
        """
        capture = self._capture_group_lru_from_gpu(groups, stream)
        resolved = self.resolve_lru_capture(capture)
        if not self._apply_resolved_group_lru(resolved):
            raise RuntimeError("Resident LRU changed during synchronous refresh")

    def _select_victim_cluster(
        self,
        protected_clusters: set[_ClusterId],
        incoming_group_pages: Mapping[RetroSpecClusterGroup, int],
    ) -> _ClusterId | None:
        """Choose one group-local LRU victim without a layer-global LRU."""
        candidates: list[tuple[int, int, int, str, int, _ClusterId]] = []

        for group, group_state in self._group_states.items():
            victim = self._oldest_unprotected_cluster(
                group_state,
                protected_clusters,
            )
            if victim is None:
                continue

            target_pages = self._group_targets.get(group, 0)
            projected_pages = group_state.num_pages + incoming_group_pages.get(group, 0)
            excess_pages = projected_pages - target_pages

            # Sorting is ascending, so negate descending priorities:
            #   1. groups above their soft target;
            #   2. larger excess;
            #   3. larger current resident footprint;
            #   4. deterministic request/head order.
            candidates.append(
                (
                    -int(excess_pages > 0),
                    -excess_pages,
                    -group_state.num_pages,
                    group.request_id,
                    group.kv_head_index,
                    victim,
                )
            )

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][-1]

    def _select_victim_clusters(
        self,
        protected_clusters: set[_ClusterId],
        incoming_group_pages: Mapping[RetroSpecClusterGroup, int],
        required_page_count: int,
    ) -> tuple[_ClusterId, ...]:
        """Plan an exact multi-cluster eviction without mutating live state."""
        pages_to_release = max(
            self.num_resident_pages + required_page_count - self._logical_capacity,
            0,
        )
        if pages_to_release == 0:
            return ()

        shadow_page_counts = {
            group: group_state.num_pages
            for group, group_state in self._group_states.items()
        }
        shadow_lrus = {
            group: deque(
                cluster_id
                for cluster_id in group_state.lru
                if cluster_id not in protected_clusters
            )
            for group, group_state in self._group_states.items()
        }

        victims: list[_ClusterId] = []
        released_pages = 0
        while released_pages < pages_to_release:
            candidates: list[
                tuple[
                    tuple[int, int, int, str, int, _ClusterId],
                    RetroSpecClusterGroup,
                    _ClusterId,
                ]
            ] = []
            for group, group_lru in shadow_lrus.items():
                if not group_lru:
                    continue

                victim = group_lru[0]
                page_count = shadow_page_counts[group]
                target_pages = self._group_targets.get(group, 0)
                projected_pages = page_count + incoming_group_pages.get(group, 0)
                excess_pages = projected_pages - target_pages
                priority = (
                    -int(excess_pages > 0),
                    -excess_pages,
                    -page_count,
                    group.request_id,
                    group.kv_head_index,
                    victim,
                )
                candidates.append((priority, group, victim))

            if not candidates:
                break

            _, selected_group, victim = min(
                candidates, key=lambda candidate: candidate[0]
            )
            shadow_lrus[selected_group].popleft()
            victim_pages = len(self._cluster_to_slots[victim])
            shadow_page_counts[selected_group] -= victim_pages
            released_pages += victim_pages
            victims.append(victim)

        return tuple(victims)

    def _evict_oldest_unprotected(
        self,
        protected_clusters: set[_ClusterId],
        incoming_group_pages: Mapping[RetroSpecClusterGroup, int],
        update_stream: torch.cuda.Stream | None = None,
        affected_groups: set[RetroSpecClusterGroup] | None = None,
    ) -> bool:
        victim = self._select_victim_cluster(
            protected_clusters,
            incoming_group_pages,
        )
        if victim is None:
            return False

        victim_group = self._cluster_to_group[victim]
        self._evict_cluster(victim, update_stream=update_stream)
        if affected_groups is not None:
            affected_groups.add(victim_group)
        return True

    def resize(
        self,
        capacity: int,
        group_targets: Mapping[RetroSpecClusterGroup, int] | None = None,
    ) -> None:
        """Change layer capacity and update group soft targets."""
        if capacity < 0:
            raise ValueError("Resident cache capacity must be non-negative")

        new_group_targets = (
            dict(self._group_targets) if group_targets is None else dict(group_targets)
        )
        for group, target_pages in new_group_targets.items():
            if target_pages < 0:
                raise ValueError(
                    f"Resident target for group {group!r} must be non-negative"
                )
        if sum(new_group_targets.values()) > capacity:
            raise ValueError("Resident group targets cannot exceed layer capacity")
        if (
            capacity == self._logical_capacity
            and new_group_targets == self._group_targets
        ):
            return

        affected_groups = (
            set(self._group_states) | set(self._group_targets) | set(new_group_targets)
        )
        previous_gate_ready = self._snapshot_group_hit_gate_ready(affected_groups)
        if self._pending_cluster_events:
            self.synchronize_pending_copies()

        current_stream = torch.cuda.current_stream(self.device)
        self._grow_storage(capacity)
        self._logical_capacity = capacity
        self._group_targets = new_group_targets
        self._bump_structure_revision()

        if self.num_resident_pages > capacity:
            self._refresh_group_lru_from_gpu(
                self._group_states.keys(),
                current_stream,
            )
        while self.num_resident_pages > capacity:
            if not self._evict_oldest_unprotected(
                protected_clusters=set(),
                incoming_group_pages={},
                update_stream=current_stream,
            ):
                raise RuntimeError("Resident cache cannot satisfy the reduced capacity")

        self._publish_handle_delta((), previous_gate_ready, current_stream)

    @staticmethod
    def _parse_clusters(
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        allocated_cluster_ids: Collection[int] | None,
        allocated_page_ids: Collection[int] | None,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[_ClusterId | None],
        list[_LogicalPages],
        list[tuple[int, ...]],
    ]:
        if cluster_ids.ndim < 1:
            raise ValueError("Cluster IDs must have at least one dimension")
        if cluster_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cluster IDs must use an integral dtype")
        if page_ids.ndim != cluster_ids.ndim + 1:
            raise ValueError("Cluster pages must add one page dimension to cluster IDs")
        if page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Cluster ID and page-table shapes do not match")
        if page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cluster page IDs must use an integral dtype")
        if cluster_ids_cpu is None:
            cluster_ids_cpu = cluster_ids.detach().to(
                device="cpu",
                dtype=torch.int64,
            )
        if page_ids_cpu is None:
            page_ids_cpu = page_ids.detach().to(
                device="cpu",
                dtype=torch.int64,
            )

        if cluster_ids_cpu.device.type != "cpu" or page_ids_cpu.device.type != "cpu":
            raise ValueError("Parsed cluster metadata must reside on CPU")
        if cluster_ids_cpu.dtype != torch.int64 or page_ids_cpu.dtype != torch.int64:
            raise ValueError("Parsed cluster metadata must use int64")
        if cluster_ids_cpu.shape != cluster_ids.shape:
            raise ValueError("CPU cluster IDs do not match the selection shape")
        if page_ids_cpu.shape != page_ids.shape:
            raise ValueError("CPU page IDs do not match the selection shape")

        if torch.any(cluster_ids_cpu < -1).item():
            raise ValueError("Cluster IDs must be at least -1")
        if torch.any(page_ids_cpu < -1).item():
            raise ValueError("Cluster page IDs must be at least -1")

        flat_cluster_ids = cluster_ids_cpu.reshape(-1)
        flat_page_ids = page_ids_cpu.reshape(
            flat_cluster_ids.numel(),
            page_ids_cpu.shape[-1],
        )

        parsed_cluster_ids: list[_ClusterId | None] = []
        parsed_cluster_pages: list[_LogicalPages] = []
        valid_positions: list[tuple[int, ...]] = []

        requested_cluster_pages: dict[_ClusterId, _LogicalPages] = {}
        requested_page_owners: dict[int, _ClusterId] = {}

        for raw_cluster_id, row in zip(
            flat_cluster_ids.tolist(),
            flat_page_ids.tolist(),
        ):
            positions = tuple(
                index for index, page_id in enumerate(row) if page_id >= 0
            )
            logical_pages = tuple(row[index] for index in positions)

            if len(logical_pages) != len(set(logical_pages)):
                raise ValueError(
                    "A cluster cannot reference the same logical page twice"
                )

            if raw_cluster_id < 0:
                if logical_pages:
                    raise ValueError(
                        "A padded cluster ID cannot reference logical pages"
                    )

                parsed_cluster_ids.append(None)
                parsed_cluster_pages.append(())
                valid_positions.append(())
                continue

            cluster_id = int(raw_cluster_id)

            if (
                allocated_cluster_ids is not None
                and cluster_id not in allocated_cluster_ids
            ):
                raise RuntimeError(
                    f"Cluster selection references unallocated cluster {cluster_id}"
                )
            if not logical_pages:
                raise ValueError(
                    "A valid cluster ID must reference at least one logical page"
                )

            previous_pages = requested_cluster_pages.get(cluster_id)
            if previous_pages is not None and previous_pages != logical_pages:
                raise ValueError(
                    "One cluster ID cannot reference different logical pages"
                )
            requested_cluster_pages[cluster_id] = logical_pages

            for logical_page_id in logical_pages:
                if (
                    allocated_page_ids is not None
                    and logical_page_id not in allocated_page_ids
                ):
                    raise RuntimeError(
                        "Cluster page table references an unallocated "
                        f"logical page {logical_page_id}"
                    )

                previous_owner = requested_page_owners.get(logical_page_id)
                if previous_owner is not None and previous_owner != cluster_id:
                    raise ValueError(
                        "A logical page cannot belong to multiple clusters"
                    )
                requested_page_owners[logical_page_id] = cluster_id

            parsed_cluster_ids.append(cluster_id)
            parsed_cluster_pages.append(logical_pages)
            valid_positions.append(positions)

        return (
            cluster_ids_cpu,
            page_ids_cpu,
            parsed_cluster_ids,
            parsed_cluster_pages,
            valid_positions,
        )

    @staticmethod
    def _priority_ordered_clusters(
        cluster_ids: list[_ClusterId | None],
        leading_shape: torch.Size,
    ) -> list[_ClusterId]:
        """Return unique cluster IDs in retrieval-priority order.

        The final dimension is retrieval rank. Earlier dimensions identify
        independent request/KV-head groups.
        """
        if len(leading_shape) <= 1:
            ordered_clusters = cluster_ids
        else:
            num_ranked_clusters = leading_shape[-1]
            num_groups = prod(leading_shape[:-1])

            ordered_clusters = [
                cluster_ids[group_index * num_ranked_clusters + rank]
                for rank in range(num_ranked_clusters)
                for group_index in range(num_groups)
            ]

        return list(
            dict.fromkeys(
                cluster_id for cluster_id in ordered_clusters if cluster_id is not None
            )
        )

    @staticmethod
    def _validate_cluster_groups(
        parsed_cluster_ids: list[_ClusterId | None],
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
    ) -> None:
        """Require group metadata for every valid requested cluster."""
        checked_clusters: set[_ClusterId] = set()

        for cluster_id in parsed_cluster_ids:
            if cluster_id is None or cluster_id in checked_clusters:
                continue
            if cluster_id not in cluster_groups:
                raise RuntimeError(
                    f"Missing resident group metadata for cluster {cluster_id}"
                )
            checked_clusters.add(cluster_id)

    def lookup(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        touch: bool = True,
        include_pending: bool = True,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Resolve selected cluster blocks against resident GPU slots.

        When include_pending is false, clusters whose H2D copies are incomplete
        are reported as misses. Draft attention can then continue with centroid
        estimation instead of waiting for verification prefetches.
        """
        if not include_pending:
            self._reap_completed_copy_batches()

        (
            cluster_ids_cpu,
            page_ids_cpu,
            parsed_cluster_ids,
            parsed_cluster_pages,
            valid_positions,
        ) = self._parse_clusters(
            cluster_ids,
            page_ids,
            allocated_cluster_ids,
            allocated_page_ids,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )
        self._validate_cluster_groups(parsed_cluster_ids, cluster_groups)

        cache_page_ids_cpu = torch.full_like(page_ids_cpu, -1)
        flat_cache_page_ids = cache_page_ids_cpu.reshape(
            len(parsed_cluster_ids),
            page_ids_cpu.shape[-1],
        )

        hit_cluster_mask_cpu = torch.zeros(
            cluster_ids_cpu.shape,
            dtype=torch.bool,
        )
        miss_cluster_mask_cpu = torch.zeros_like(hit_cluster_mask_cpu)
        hit_gate_ready_mask_cpu = torch.zeros_like(hit_cluster_mask_cpu)

        flat_hit_mask = hit_cluster_mask_cpu.reshape(-1)
        flat_miss_mask = miss_cluster_mask_cpu.reshape(-1)
        flat_hit_gate_ready_mask = hit_gate_ready_mask_cpu.reshape(-1)
        hit_clusters: set[_ClusterId] = set()

        for cluster_index, (
            cluster_id,
            logical_pages,
            positions,
        ) in enumerate(
            zip(
                parsed_cluster_ids,
                parsed_cluster_pages,
                valid_positions,
            )
        ):
            if cluster_id is None:
                continue

            group = cluster_groups[cluster_id]
            flat_hit_gate_ready_mask[cluster_index] = self._is_group_hit_gate_ready(
                group
            )

            resident_slots = self._cluster_to_slots.get(cluster_id)
            pending = cluster_id in self._pending_cluster_events

            if resident_slots is None or (pending and not include_pending):
                flat_miss_mask[cluster_index] = True
                continue

            resident_group = self._cluster_to_group.get(cluster_id)
            if resident_group != cluster_groups[cluster_id]:
                raise RuntimeError(
                    "Resident cluster group does not match requested ownership"
                )

            resident_pages = self._cluster_to_pages[cluster_id]
            if resident_pages != logical_pages:
                raise RuntimeError(
                    "Resident cluster pages do not match the requested descriptor"
                )

            flat_hit_mask[cluster_index] = True
            hit_clusters.add(cluster_id)

            for position, slot_id in zip(positions, resident_slots):
                flat_cache_page_ids[cluster_index, position] = slot_id

        if touch:
            requested_clusters = self._priority_ordered_clusters(
                parsed_cluster_ids,
                cluster_ids_cpu.shape,
            )

            # Touch low-priority entries first so rank-zero clusters finish as MRU.
            for cluster_id in reversed(requested_clusters):
                if cluster_id in hit_clusters:
                    self._touch_cluster(cluster_id)

        ready_event = self._pending_event_for_clusters(hit_clusters)

        return RetroSpecResidentPageAccess(
            cache_page_ids=cache_page_ids_cpu.to(
                device=page_ids.device,
                non_blocking=False,
            ),
            hit_cluster_mask=hit_cluster_mask_cpu.to(
                device=cluster_ids.device,
                non_blocking=False,
            ),
            miss_cluster_mask=miss_cluster_mask_cpu.to(
                device=cluster_ids.device,
                non_blocking=False,
            ),
            hit_gate_ready_mask=hit_gate_ready_mask_cpu.to(
                device=cluster_ids.device,
                non_blocking=False,
            ),
            logical_page_ids_cpu=page_ids_cpu,
            miss_cluster_mask_cpu=miss_cluster_mask_cpu,
            ready_event=ready_event,
        )

    def lookup_gpu(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        active_mask: torch.Tensor | None,
        cache_page_ids: torch.Tensor,
        hit_cluster_mask: torch.Tensor,
        miss_cluster_mask: torch.Tensor,
        hit_gate_ready_mask: torch.Tensor,
        access_kinds: torch.Tensor,
        plan_row_indices: torch.Tensor | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Resolve handles without synchronizing or parsing on the CPU."""
        if cluster_ids.device != self.device or page_ids.device != self.device:
            raise ValueError("GPU resident lookup tensors must use the cache device")
        if page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Cluster IDs and logical pages do not match")
        output_batch = (
            cluster_ids.shape[0]
            if plan_row_indices is None
            else plan_row_indices.shape[0]
        )
        output_cluster_shape = (output_batch, *cluster_ids.shape[1:])
        output_page_shape = (*output_cluster_shape, page_ids.shape[-1])
        if cache_page_ids.shape != output_page_shape:
            raise ValueError("Resident page output has the wrong indexed shape")
        for output in (
            hit_cluster_mask,
            miss_cluster_mask,
            hit_gate_ready_mask,
            access_kinds,
        ):
            if output.shape != output_cluster_shape:
                raise ValueError("Resident lookup output has the wrong indexed shape")

        self._gpu_access_lock.acquire()
        try:
            self._ensure_handle_table(page_ids.shape[-1])
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            lookup_resident_handles(
                cluster_handles=cluster_ids,
                logical_page_ids=page_ids,
                active_mask=active_mask,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                output_page_slots=cache_page_ids,
                output_hit_mask=hit_cluster_mask,
                output_miss_mask=miss_cluster_mask,
                output_hit_gate_ready=hit_gate_ready_mask,
                output_access_kinds=access_kinds,
                plan_row_indices=plan_row_indices,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecResidentPageAccess(
            cache_page_ids=cache_page_ids,
            hit_cluster_mask=hit_cluster_mask,
            miss_cluster_mask=miss_cluster_mask,
            hit_gate_ready_mask=hit_gate_ready_mask,
            logical_page_ids_cpu=None,
            miss_cluster_mask_cpu=None,
            ready_event=None,
            access_kinds=access_kinds,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )

    def lookup_ranked_compact_draft_gpu(
        self,
        *,
        ranked_values: torch.Tensor,
        candidate_counts: torch.Tensor,
        arena_resident_table_buckets: torch.Tensor,
        arena_cluster_page_starts: torch.Tensor,
        arena_cluster_page_counts: torch.Tensor,
        arena_page_ids: torch.Tensor,
        arena_page_token_counts: torch.Tensor,
        arena_cluster_offsets: torch.Tensor,
        arena_page_offsets: torch.Tensor,
        request_slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
        retrieval_ratio: float,
        estimation_ratio: float,
        expanded_retrieval_width: int,
        max_pages_per_cluster: int,
        fallback_token_counts: torch.Tensor,
        sparse_cluster_indices: torch.Tensor,
        cluster_handles: torch.Tensor,
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
        sparse_attention: torch.Tensor,
        expanded_attention: torch.Tensor,
        emit_misses: bool = True,
        statistics_buffer: torch.Tensor | None = None,
        statistics_indices: tuple[int, ...] | None = None,
    ) -> RetroSpecCompactResidentPageAccess:
        """Resolve ranked DRAFT rows without logical-page intermediates."""
        if ranked_values.device != self.device:
            raise ValueError("Ranked draft lookup must use the cache device")

        self._gpu_access_lock.acquire()
        try:
            self._ensure_handle_table(max_pages_per_cluster)
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            resolve_compact_draft_pages(
                ranked_values=ranked_values,
                candidate_counts=candidate_counts,
                arena_resident_table_buckets=arena_resident_table_buckets,
                arena_cluster_page_starts=arena_cluster_page_starts,
                arena_cluster_page_counts=arena_cluster_page_counts,
                arena_page_ids=arena_page_ids,
                arena_page_token_counts=arena_page_token_counts,
                arena_cluster_offsets=arena_cluster_offsets,
                arena_page_offsets=arena_page_offsets,
                request_slot_ids=request_slot_ids,
                active_mask=active_mask,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                retrieval_ratio=retrieval_ratio,
                estimation_ratio=estimation_ratio,
                expanded_retrieval_width=expanded_retrieval_width,
                max_pages_per_cluster=max_pages_per_cluster,
                fallback_token_counts=fallback_token_counts,
                sparse_cluster_indices=sparse_cluster_indices,
                cluster_handles=cluster_handles,
                output_page_slots=cache_page_ids,
                output_page_token_counts=page_token_counts,
                output_page_counts=page_counts,
                output_clustered_token_counts=clustered_token_counts,
                output_attention=attention_mass,
                output_hit_attention_by_head=hit_attention_by_head,
                output_selected_counts=selected_cluster_counts,
                output_hit_counts=hit_cluster_counts,
                output_miss_counts=miss_cluster_counts,
                output_gate_ready=hit_gate_ready,
                output_miss_handles=miss_cluster_ids,
                output_miss_positions=miss_positions,
                output_miss_count=miss_count,
                sparse_attention=sparse_attention,
                expanded_attention=expanded_attention,
                emit_misses=emit_misses,
                statistics_buffer=statistics_buffer,
                statistics_indices=statistics_indices,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecCompactResidentPageAccess(
            cache_page_ids=cache_page_ids,
            page_token_counts=page_token_counts,
            page_counts=page_counts,
            clustered_token_counts=clustered_token_counts,
            attention_mass=attention_mass,
            selected_cluster_counts=selected_cluster_counts,
            hit_cluster_counts=hit_cluster_counts,
            miss_cluster_counts=miss_cluster_counts,
            hit_gate_ready=hit_gate_ready,
            access_kinds=None,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )

    def lookup_ranked_draft_gpu(
        self,
        *,
        ranked_values: torch.Tensor,
        ranked_indices: torch.Tensor,
        candidate_counts: torch.Tensor,
        arena_cluster_ids: torch.Tensor,
        arena_resident_table_buckets: torch.Tensor,
        arena_cluster_token_counts: torch.Tensor,
        arena_cluster_page_counts: torch.Tensor,
        arena_cluster_offsets: torch.Tensor,
        arena_generations: torch.Tensor,
        request_slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
        retrieval_ratio: float,
        estimation_ratio: float,
        expanded_retrieval_width: int,
        max_pages_per_cluster: int,
        plan_valid_rows: torch.Tensor,
        output_request_slot_ids: torch.Tensor,
        output_request_slot_generations: torch.Tensor,
        cluster_handles: torch.Tensor,
        resident_bucket_ids: torch.Tensor,
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
        sparse_attention: torch.Tensor,
        expanded_attention: torch.Tensor,
        capture_request_descriptors: bool,
        emit_misses: bool = True,
        statistics_buffer: torch.Tensor | None = None,
        statistics_indices: tuple[int, ...] | None = None,
    ) -> RetroSpecRankedDraftResidentAccess:
        """Resolve ranked DRAFT clusters without compact page intermediates."""
        if ranked_values.device != self.device:
            raise ValueError("Ranked DRAFT lookup must use the cache device")

        stats = self.performance_stats
        lock_started_at = (
            perf_counter() if stats is not None and stats.enabled else None
        )
        self._gpu_access_lock.acquire()
        if lock_started_at is not None:
            stats.record_cpu_time(
                "draft_ranked_resident_lock_wait_wall",
                perf_counter() - lock_started_at,
            )
        try:
            self._ensure_handle_table(max_pages_per_cluster)
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            resolve_ranked_draft_buckets(
                ranked_values=ranked_values,
                ranked_indices=ranked_indices,
                candidate_counts=candidate_counts,
                arena_cluster_ids=arena_cluster_ids,
                arena_resident_table_buckets=arena_resident_table_buckets,
                arena_cluster_token_counts=arena_cluster_token_counts,
                arena_cluster_page_counts=arena_cluster_page_counts,
                arena_cluster_offsets=arena_cluster_offsets,
                arena_generations=arena_generations,
                request_slot_ids=request_slot_ids,
                active_mask=active_mask,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                retrieval_ratio=retrieval_ratio,
                estimation_ratio=estimation_ratio,
                expanded_retrieval_width=expanded_retrieval_width,
                max_pages_per_cluster=max_pages_per_cluster,
                output_valid_rows=plan_valid_rows,
                output_request_slot_ids=output_request_slot_ids,
                output_request_slot_generations=(output_request_slot_generations),
                output_cluster_handles=cluster_handles,
                output_resident_buckets=resident_bucket_ids,
                output_clustered_token_counts=clustered_token_counts,
                output_attention=attention_mass,
                output_hit_attention_by_head=hit_attention_by_head,
                output_selected_counts=selected_cluster_counts,
                output_hit_counts=hit_cluster_counts,
                output_miss_counts=miss_cluster_counts,
                output_gate_ready=hit_gate_ready,
                output_miss_handles=miss_cluster_ids,
                output_miss_positions=miss_positions,
                output_miss_count=miss_count,
                sparse_attention=sparse_attention,
                expanded_attention=expanded_attention,
                capture_request_descriptors=capture_request_descriptors,
                emit_misses=emit_misses,
                statistics_buffer=statistics_buffer,
                statistics_indices=statistics_indices,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecRankedDraftResidentAccess(
            cluster_handles=cluster_handles,
            resident_bucket_ids=resident_bucket_ids,
            clustered_token_counts=clustered_token_counts,
            attention_mass=attention_mass,
            selected_cluster_counts=selected_cluster_counts,
            hit_cluster_counts=hit_cluster_counts,
            miss_cluster_counts=miss_cluster_counts,
            hit_gate_ready=hit_gate_ready,
            resident_table_page_counts=self._handle_table_page_counts,
            resident_table_page_slots=self._handle_table_page_slots,
            resident_key_pages=self.key_pages,
            resident_value_pages=self.value_pages,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )

    def lookup_compact_verification_gpu(
        self,
        selected_cluster_indices: torch.Tensor,
        plan_valid_rows: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
        arena_cluster_ids: torch.Tensor,
        arena_cluster_page_starts: torch.Tensor,
        arena_cluster_page_counts: torch.Tensor,
        arena_page_ids: torch.Tensor,
        arena_page_token_counts: torch.Tensor,
        arena_cluster_offsets: torch.Tensor,
        arena_page_offsets: torch.Tensor,
        arena_generations: torch.Tensor,
        resident_page_ids: torch.Tensor,
        staging_page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
        page_counts: torch.Tensor,
        selected_cluster_counts: torch.Tensor,
        hit_cluster_counts: torch.Tensor,
        miss_cluster_counts: torch.Tensor,
        unique_cluster_ids: torch.Tensor,
        unique_logical_page_ids: torch.Tensor,
        unique_page_counts: torch.Tensor,
        miss_hash_buckets: torch.Tensor,
        miss_unique_indices: torch.Tensor,
        miss_output_page_offsets: torch.Tensor,
        miss_count: torch.Tensor,
        unique_miss_count: torch.Tensor,
        miss_table_handles: torch.Tensor,
        miss_table_unique_indices: torch.Tensor,
        invalid_descriptor_count: torch.Tensor,
    ) -> RetroSpecCompactVerificationPageAccess:
        """Resolve verification selections directly from the resident arena."""
        if selected_cluster_indices.device != self.device:
            raise ValueError("Compact verification lookup uses the wrong device")

        self._gpu_access_lock.acquire()
        try:
            self._ensure_handle_table(unique_logical_page_ids.shape[1])
            access_epoch = self._next_access_epoch
            self._next_access_epoch += 1
            resolve_compact_verification_pages(
                selected_cluster_indices=selected_cluster_indices,
                plan_valid_rows=plan_valid_rows,
                request_slot_ids=request_slot_ids,
                request_slot_generations=request_slot_generations,
                arena_cluster_ids=arena_cluster_ids,
                arena_cluster_page_starts=arena_cluster_page_starts,
                arena_cluster_page_counts=arena_cluster_page_counts,
                arena_page_ids=arena_page_ids,
                arena_page_token_counts=arena_page_token_counts,
                arena_cluster_offsets=arena_cluster_offsets,
                arena_page_offsets=arena_page_offsets,
                arena_generations=arena_generations,
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_last_access_epochs=self._handle_table_last_access_epochs,
                access_epoch=access_epoch,
                output_resident_page_ids=resident_page_ids,
                output_staging_page_ids=staging_page_ids,
                output_page_token_counts=page_token_counts,
                output_page_counts=page_counts,
                output_selected_counts=selected_cluster_counts,
                output_hit_counts=hit_cluster_counts,
                output_miss_counts=miss_cluster_counts,
                output_miss_hash_buckets=miss_hash_buckets,
                output_miss_unique_indices=miss_unique_indices,
                output_miss_page_offsets=miss_output_page_offsets,
                output_miss_count=miss_count,
                output_unique_handles=unique_cluster_ids,
                output_unique_logical_page_ids=unique_logical_page_ids,
                output_unique_page_counts=unique_page_counts,
                output_unique_miss_count=unique_miss_count,
                miss_table_handles=miss_table_handles,
                miss_table_unique_indices=miss_table_unique_indices,
                output_invalid_descriptor_count=invalid_descriptor_count,
            )
        except BaseException:
            self._gpu_access_lock.release()
            raise

        return RetroSpecCompactVerificationPageAccess(
            resident_page_ids=resident_page_ids,
            staging_page_ids=staging_page_ids,
            page_token_counts=page_token_counts,
            page_counts=page_counts,
            selected_cluster_counts=selected_cluster_counts,
            hit_cluster_counts=hit_cluster_counts,
            miss_cluster_counts=miss_cluster_counts,
            unique_cluster_ids=unique_cluster_ids,
            unique_logical_page_ids=unique_logical_page_ids,
            unique_page_counts=unique_page_counts,
            miss_unique_indices=miss_unique_indices,
            miss_output_page_offsets=miss_output_page_offsets,
            miss_count=miss_count,
            unique_miss_count=unique_miss_count,
            invalid_descriptor_count=invalid_descriptor_count,
            read_lease=RetroSpecResidentReadLease(self._gpu_access_lock),
        )

    def _validate_source_pages(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
    ) -> None:
        if key_pages.shape != value_pages.shape:
            raise ValueError("Source key and value page shapes must match")
        if key_pages.ndim != 3:
            raise ValueError(
                "Source pages must have shape [pages, page_size, head_size]"
            )
        if key_pages.shape[1:] != (
            self.page_size,
            self.head_size,
        ):
            raise ValueError("Source page shape does not match resident cache")
        if key_pages.dtype != self.dtype:
            raise ValueError("Source key dtype does not match resident cache")
        if value_pages.dtype != self.dtype:
            raise ValueError("Source value dtype does not match resident cache")
        if key_pages.device != value_pages.device:
            raise ValueError("Source key and value pages must use one device")

    def _validate_backing_pages(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
    ) -> None:
        self._validate_source_pages(key_pages, value_pages)
        if key_pages.device.type != "cpu":
            raise ValueError("Resident cache admission requires CPU backing pages")

    def _copy_cluster_to_slots(
        self,
        source_page_ids: tuple[int, ...],
        slots: tuple[int, ...],
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> None:
        """Enqueue one complete cluster on the dedicated copy stream."""
        self._copy_pages_to_slots(
            source_page_ids,
            slots,
            source_key_pages,
            source_value_pages,
        )

    def _copy_pages_to_slots(
        self,
        source_page_ids: Sequence[int],
        slots: Sequence[int],
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> None:
        """Enqueue one coalesced K/V page mapping on the copy stream."""
        if len(source_page_ids) != len(slots):
            raise ValueError("Source page and destination slot counts must match")
        if not source_page_ids:
            return

        block_mapping = torch.tensor(
            tuple(zip(source_page_ids, slots, strict=True)),
            dtype=torch.int64,
            device="cpu",
        )
        block_size = self.page_size * self.head_size * source_key_pages.element_size()
        with torch.cuda.stream(self._copy_stream):
            span_count = ops.copy_kv_blocks_coalesced(
                source_key_pages,
                source_value_pages,
                self.key_pages,
                self.value_pages,
                block_size,
                block_mapping,
            )

        stats = self.performance_stats
        if (
            stats is not None
            and stats.enabled
            and stats.cuda_timing_level == "detailed"
        ):
            stats.add_counter("resident_copy_pages", len(source_page_ids))
            stats.add_counter("resident_copy_spans", span_count)

    def _prepare_admission_from_sources(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        source_page_ids: torch.Tensor,
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
    ) -> RetroSpecPreparedResidentAdmission:
        """Parse immutable metadata without reading live resident state."""
        self._validate_source_pages(source_key_pages, source_value_pages)

        if source_page_ids.shape != page_ids.shape:
            raise ValueError(
                "Source page IDs must have the same shape as logical page IDs"
            )
        if source_page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Source page IDs must use an integral dtype")

        (
            cluster_ids_cpu,
            page_ids_cpu,
            parsed_cluster_ids,
            parsed_cluster_pages,
            valid_positions,
        ) = self._parse_clusters(
            cluster_ids,
            page_ids,
            None,
            None,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )
        self._validate_cluster_groups(parsed_cluster_ids, cluster_groups)

        source_page_ids_cpu = source_page_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        if torch.any(source_page_ids_cpu < -1).item():
            raise ValueError("Source page IDs must be at least -1")

        flat_source_page_ids = source_page_ids_cpu.reshape(
            len(parsed_cluster_ids),
            source_page_ids_cpu.shape[-1],
        )
        cluster_page_map: dict[_ClusterId, _LogicalPages] = {}
        cluster_source_ids: dict[_ClusterId, tuple[int, ...]] = {}

        for cluster_id, logical_pages, positions, source_row in zip(
            parsed_cluster_ids,
            parsed_cluster_pages,
            valid_positions,
            flat_source_page_ids,
            strict=True,
        ):
            if cluster_id is None:
                continue

            cluster_page_map[cluster_id] = logical_pages
            current_source_ids = tuple(
                int(source_row[position]) for position in positions
            )
            previous_source_ids = cluster_source_ids.get(cluster_id)

            # A duplicated cluster may occur in more than one batch position.
            # Prefer an occurrence with a complete source mapping.
            if previous_source_ids is None or any(
                source_page_id < 0 for source_page_id in previous_source_ids
            ):
                cluster_source_ids[cluster_id] = current_source_ids

        requested_clusters = tuple(
            self._priority_ordered_clusters(
                parsed_cluster_ids,
                cluster_ids_cpu.shape,
            )
        )
        referenced_page_ids = frozenset(
            page_id
            for logical_pages in cluster_page_map.values()
            for page_id in logical_pages
        )

        return RetroSpecPreparedResidentAdmission(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=MappingProxyType(dict(cluster_groups)),
            source_page_ids=source_page_ids_cpu,
            source_key_pages=source_key_pages,
            source_value_pages=source_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
            parsed_cluster_ids=tuple(parsed_cluster_ids),
            parsed_cluster_pages=tuple(parsed_cluster_pages),
            valid_positions=tuple(valid_positions),
            requested_clusters=requested_clusters,
            cluster_page_map=MappingProxyType(cluster_page_map),
            cluster_source_ids=MappingProxyType(cluster_source_ids),
            referenced_cluster_ids=frozenset(cluster_page_map),
            referenced_page_ids=referenced_page_ids,
        )

    @staticmethod
    def _validate_prepared_admission_allocations(
        prepared: RetroSpecPreparedResidentAdmission,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
    ) -> None:
        """Reject a descriptor whose request storage was released or reused."""
        missing_cluster_ids = prepared.referenced_cluster_ids.difference(
            allocated_cluster_ids
        )
        if missing_cluster_ids:
            cluster_id = min(missing_cluster_ids)
            raise RuntimeError(
                f"Prepared admission references unallocated cluster {cluster_id}"
            )

        missing_page_ids = prepared.referenced_page_ids.difference(allocated_page_ids)
        if missing_page_ids:
            page_id = min(missing_page_ids)
            raise RuntimeError(
                f"Prepared admission references unallocated logical page {page_id}"
            )

    def prepare_staged_admission(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
    ) -> RetroSpecPreparedResidentAdmission:
        """Prepare staged admission metadata without taking mutation_guard()."""
        self._validate_source_pages(staging_key_pages, staging_value_pages)
        if (
            staging_key_pages.device.type != "cpu"
            and staging_key_pages.device != self.device
        ):
            raise ValueError("Staging pages must use CPU or the resident CUDA device")

        return self._prepare_admission_from_sources(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            source_page_ids=staging_page_ids,
            source_key_pages=staging_key_pages,
            source_value_pages=staging_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )

    def _plan_prepared_admission_targets(
        self,
        prepared: RetroSpecPreparedResidentAdmission,
    ) -> tuple[
        tuple[_ClusterId, ...],
        set[_ClusterId],
        tuple[_ClusterId, ...],
        int,
        dict[RetroSpecClusterGroup, int],
    ]:
        """Plan the priority prefix against the current resident state."""
        target_clusters: list[_ClusterId] = []
        target_page_count = 0
        for cluster_id in prepared.requested_clusters:
            cluster_page_count = len(prepared.cluster_page_map[cluster_id])
            if cluster_page_count > self._logical_capacity:
                continue
            if target_page_count + cluster_page_count > self._logical_capacity:
                break
            target_clusters.append(cluster_id)
            target_page_count += cluster_page_count

        missing_targets = tuple(
            cluster_id
            for cluster_id in target_clusters
            if cluster_id not in self._cluster_to_slots
        )
        required_page_count = sum(
            len(prepared.cluster_page_map[cluster_id]) for cluster_id in missing_targets
        )
        incoming_group_pages: dict[RetroSpecClusterGroup, int] = {}
        for cluster_id in missing_targets:
            group = prepared.cluster_groups[cluster_id]
            incoming_group_pages[group] = incoming_group_pages.get(group, 0) + len(
                prepared.cluster_page_map[cluster_id]
            )

        target_cluster_tuple = tuple(target_clusters)
        return (
            target_cluster_tuple,
            set(target_cluster_tuple),
            missing_targets,
            required_page_count,
            incoming_group_pages,
        )

    def capture_prepared_admission_lru(
        self,
        prepared: RetroSpecPreparedResidentAdmission,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        stream: torch.cuda.Stream,
    ) -> RetroSpecResidentLruCapture | None:
        """Capture stable eviction descriptors without waiting on the GPU."""
        self._validate_prepared_admission_allocations(
            prepared,
            allocated_cluster_ids,
            allocated_page_ids,
        )
        _, _, _, required_page_count, _ = self._plan_prepared_admission_targets(
            prepared
        )
        if self.num_resident_pages + required_page_count <= self._logical_capacity:
            return None

        self._reap_completed_copy_batches()
        stats = self.performance_stats
        if stats is not None and stats.enabled:
            stats.add_counter("resident_lru_snapshot_requested")
        return self._capture_group_lru_from_gpu(self._group_states.keys(), stream)

    def _admit_from_sources(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        source_page_ids: torch.Tensor,
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
        prepared: RetroSpecPreparedResidentAdmission | None = None,
        resolved_lru: RetroSpecResolvedResidentLru | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from CPU or GPU source pages."""
        if prepared is None:
            prepared = self._prepare_admission_from_sources(
                cluster_ids=cluster_ids,
                page_ids=page_ids,
                cluster_groups=cluster_groups,
                source_page_ids=source_page_ids,
                source_key_pages=source_key_pages,
                source_value_pages=source_value_pages,
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
            )

        self._validate_prepared_admission_allocations(
            prepared,
            allocated_cluster_ids,
            allocated_page_ids,
        )

        cluster_ids = prepared.cluster_ids
        page_ids = prepared.page_ids
        cluster_groups = prepared.cluster_groups
        source_key_pages = prepared.source_key_pages
        source_value_pages = prepared.source_value_pages
        cluster_ids_cpu = prepared.cluster_ids_cpu
        page_ids_cpu = prepared.page_ids_cpu
        requested_clusters = prepared.requested_clusters
        cluster_page_map = prepared.cluster_page_map
        cluster_source_ids = prepared.cluster_source_ids

        for cluster_id in requested_clusters:
            logical_pages = cluster_page_map[cluster_id]
            resident_pages = self._cluster_to_pages.get(cluster_id)

            if resident_pages is not None:
                if resident_pages != logical_pages:
                    raise RuntimeError(
                        "Cluster ID conflicts with an existing resident descriptor"
                    )
                if self._cluster_to_group.get(cluster_id) != cluster_groups[cluster_id]:
                    raise RuntimeError(
                        "Resident cluster group does not match requested ownership"
                    )

            for logical_page_id in logical_pages:
                resident_owner = self._page_to_cluster.get(logical_page_id)
                if resident_owner is not None and resident_owner != cluster_id:
                    raise RuntimeError(
                        "Logical page ownership conflicts with an existing "
                        "resident cluster"
                    )

        (
            _,
            target_cluster_set,
            missing_targets,
            required_page_count,
            incoming_group_pages,
        ) = self._plan_prepared_admission_targets(prepared)

        # Validate every source before changing resident ownership.
        for cluster_id in missing_targets:
            logical_pages = cluster_page_map[cluster_id]
            source_ids = cluster_source_ids.get(cluster_id)

            if source_ids is None or len(source_ids) != len(logical_pages):
                raise RuntimeError("Missing source-page mapping for resident admission")
            if any(source_page_id < 0 for source_page_id in source_ids):
                raise RuntimeError(
                    "A missing cluster does not have complete source pages"
                )
            if any(
                source_page_id >= source_key_pages.shape[0]
                for source_page_id in source_ids
            ):
                raise RuntimeError("Source page ID exceeds source storage capacity")

        if mutation_stream is None:
            mutation_stream = torch.cuda.current_stream(self.device)
        if reuse_ready_event is not None:
            mutation_stream.wait_event(reuse_ready_event)

        previous_gate_ready = self._snapshot_group_hit_gate_ready(incoming_group_pages)
        victims: tuple[_ClusterId, ...] = ()
        if self.num_resident_pages + required_page_count > self._logical_capacity:
            self._reap_completed_copy_batches()
            snapshot_applied = False
            if resolved_lru is not None:
                snapshot_applied = self._apply_resolved_group_lru(resolved_lru)
                stats = self.performance_stats
                if stats is not None and stats.enabled:
                    stats.add_counter(
                        "resident_lru_snapshot_applied"
                        if snapshot_applied
                        else "resident_lru_snapshot_retried"
                    )
            if not snapshot_applied:
                self._refresh_group_lru_from_gpu(
                    self._group_states.keys(), mutation_stream
                )
            protected_clusters = target_cluster_set | set(self._pending_cluster_events)
            victims = self._select_victim_clusters(
                protected_clusters,
                incoming_group_pages,
                required_page_count,
            )
            victim_page_count = sum(
                len(self._cluster_to_slots[cluster_id]) for cluster_id in victims
            )

            if (
                self.num_resident_pages - victim_page_count + required_page_count
                > self._logical_capacity
                and self._pending_cluster_events
            ):
                self.synchronize_pending_copies()
                self._refresh_group_lru_from_gpu(
                    self._group_states.keys(), mutation_stream
                )
                victims = self._select_victim_clusters(
                    target_cluster_set,
                    incoming_group_pages,
                    required_page_count,
                )
                victim_page_count = sum(
                    len(self._cluster_to_slots[cluster_id]) for cluster_id in victims
                )

            if (
                self.num_resident_pages - victim_page_count + required_page_count
                > self._logical_capacity
            ):
                raise RuntimeError(
                    "Resident cache cannot free enough slots for priority admission"
                )
            for cluster_id in victims:
                group = self._cluster_to_group[cluster_id]
                previous_gate_ready.setdefault(
                    group, self._is_group_hit_gate_ready(group)
                )
            self._evict_clusters(victims, update_stream=mutation_stream)

        copy_scheduled = False
        copied_cluster_ids: tuple[_ClusterId, ...] = ()
        reserved_slots: tuple[int, ...] = ()

        try:
            if missing_targets:
                slots_released_event = torch.cuda.Event()
                slots_released_event.record(mutation_stream)
                self._copy_stream.wait_event(slots_released_event)

            registration_entries: list[
                tuple[
                    _ClusterId,
                    RetroSpecClusterGroup,
                    _LogicalPages,
                    tuple[int, ...],
                ]
            ] = []
            source_page_ids: list[int] = []
            destination_slots: list[int] = []

            reserved_slots = tuple(sorted(self._free_slots)[:required_page_count])
            if len(reserved_slots) != required_page_count:
                raise RuntimeError("Resident cache does not have enough free GPU slots")
            self._free_slots.difference_update(reserved_slots)

            slot_offset = 0
            for cluster_id in missing_targets:
                logical_pages = cluster_page_map[cluster_id]
                cluster_page_count = len(logical_pages)
                slots = reserved_slots[slot_offset : slot_offset + cluster_page_count]
                slot_offset += cluster_page_count
                source_ids = cluster_source_ids[cluster_id]
                source_page_ids.extend(source_ids)
                destination_slots.extend(slots)
                registration_entries.append(
                    (
                        cluster_id,
                        cluster_groups[cluster_id],
                        logical_pages,
                        slots,
                    )
                )

            if missing_targets:
                copy_scheduled = True
                self._copy_pages_to_slots(
                    source_page_ids,
                    destination_slots,
                    source_key_pages,
                    source_value_pages,
                )
                self._register_clusters(registration_entries)
                copied_cluster_ids = tuple(missing_targets)
        except BaseException:
            if not copied_cluster_ids:
                self._free_slots.update(reserved_slots)
            raise
        finally:
            if copy_scheduled:
                try:
                    self._publish_handle_delta(
                        copied_cluster_ids,
                        previous_gate_ready,
                        self._copy_stream,
                    )
                finally:
                    self._record_copy_batch(
                        cluster_ids=copied_cluster_ids,
                        source_key_pages=source_key_pages,
                        source_value_pages=source_value_pages,
                    )
            elif victims:
                self._publish_handle_delta(
                    (),
                    previous_gate_ready,
                    mutation_stream,
                )

        for cluster_id in reversed(requested_clusters):
            if cluster_id in self._cluster_to_slots:
                self._touch_cluster(cluster_id)

        if not lookup_after_admit:
            ready_event = self._pending_event_for_clusters(copied_cluster_ids)
            empty_page_ids = torch.empty(0, dtype=torch.int64, device=self.device)
            empty_mask = torch.empty(0, dtype=torch.bool, device=self.device)
            return RetroSpecResidentPageAccess(
                cache_page_ids=empty_page_ids,
                hit_cluster_mask=empty_mask,
                miss_cluster_mask=empty_mask,
                hit_gate_ready_mask=empty_mask,
                logical_page_ids_cpu=None,
                miss_cluster_mask_cpu=None,
                ready_event=ready_event,
            )

        return self.lookup(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            touch=False,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )

    def admit(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        backing_key_pages: torch.Tensor,
        backing_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from stable CPU backing pages."""
        self._validate_backing_pages(
            backing_key_pages,
            backing_value_pages,
        )

        return self._admit_from_sources(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=page_ids if page_ids_cpu is None else page_ids_cpu,
            source_key_pages=backing_key_pages,
            source_value_pages=backing_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
            reuse_ready_event=reuse_ready_event,
            mutation_stream=mutation_stream,
            lookup_after_admit=lookup_after_admit,
        )

    def admit_staged(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from bounded CPU/GPU staging pages."""
        self._validate_source_pages(
            staging_key_pages,
            staging_value_pages,
        )
        if (
            staging_key_pages.device.type != "cpu"
            and staging_key_pages.device != self.device
        ):
            raise ValueError("Staging pages must use CPU or the resident CUDA device")

        return self._admit_from_sources(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=staging_page_ids,
            source_key_pages=staging_key_pages,
            source_value_pages=staging_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
            reuse_ready_event=reuse_ready_event,
            mutation_stream=mutation_stream,
            lookup_after_admit=lookup_after_admit,
        )

    def admit_prepared_staged(
        self,
        prepared: RetroSpecPreparedResidentAdmission,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
        resolved_lru: RetroSpecResolvedResidentLru | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Commit a previously prepared admission under mutation_guard()."""
        return self._admit_from_sources(
            cluster_ids=prepared.cluster_ids,
            page_ids=prepared.page_ids,
            cluster_groups=prepared.cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=prepared.source_page_ids,
            source_key_pages=prepared.source_key_pages,
            source_value_pages=prepared.source_value_pages,
            cluster_ids_cpu=prepared.cluster_ids_cpu,
            page_ids_cpu=prepared.page_ids_cpu,
            reuse_ready_event=reuse_ready_event,
            mutation_stream=mutation_stream,
            lookup_after_admit=lookup_after_admit,
            prepared=prepared,
            resolved_lru=resolved_lru,
        )

    def invalidate(
        self,
        cluster_ids: torch.Tensor,
    ) -> None:
        """Evict released cluster IDs before their backing pages are reused."""
        self.synchronize_pending_copies()
        cluster_ids_cpu = cluster_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        released_cluster_ids = tuple(
            sorted(set(cluster_ids_cpu[cluster_ids_cpu >= 0].tolist()))
        )
        resident_cluster_ids = tuple(
            cluster_id
            for cluster_id in released_cluster_ids
            if cluster_id in self._cluster_to_slots
        )
        affected_groups = {
            self._cluster_to_group[cluster_id] for cluster_id in resident_cluster_ids
        }
        previous_gate_ready = self._snapshot_group_hit_gate_ready(affected_groups)
        current_stream = torch.cuda.current_stream(self.device)
        self._evict_clusters(resident_cluster_ids, update_stream=current_stream)
        self._publish_handle_delta((), previous_gate_ready, current_stream)
