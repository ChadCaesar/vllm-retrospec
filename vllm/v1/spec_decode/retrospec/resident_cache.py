# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict, deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from math import prod

import torch

from .cluster_identity import RetroSpecClusterGroup

_ClusterId = int
_LogicalPages = tuple[int, ...]


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
    logical_page_ids_cpu: torch.Tensor
    miss_cluster_mask_cpu: torch.Tensor
    ready_event: torch.cuda.Event | None


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

    def _reap_completed_copy_batches(self) -> None:
        """Release sources and pending markers for completed H2D batches."""
        while (
            self._pending_copy_batches
            and self._pending_copy_batches[0].ready_event.query()
        ):
            batch = self._pending_copy_batches.popleft()

            for cluster_id in batch.cluster_ids:
                pending_event = self._pending_cluster_events.get(cluster_id)
                if pending_event is batch.ready_event:
                    del self._pending_cluster_events[cluster_id]

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
        self._pending_copy_batches[-1].ready_event.synchronize()
        self._pending_copy_batches.clear()
        self._pending_cluster_events.clear()

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
        if cluster_id in self._cluster_to_slots:
            raise RuntimeError(f"Resident cluster {cluster_id} is already registered")
        if len(logical_pages) != len(slots):
            raise RuntimeError(
                "Resident cluster pages and GPU slots must have equal lengths"
            )

        group_state = self._group_states.setdefault(group, _ResidentGroupState())

        self._cluster_to_pages[cluster_id] = logical_pages
        self._cluster_to_slots[cluster_id] = slots
        self._cluster_to_group[cluster_id] = group

        group_state.lru[cluster_id] = None
        group_state.num_pages += len(slots)

        for logical_page_id in logical_pages:
            self._page_to_cluster[logical_page_id] = cluster_id

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

        group_state.lru.move_to_end(cluster_id, last=True)

    def _is_group_hit_gate_ready(self, group: RetroSpecClusterGroup) -> bool:
        """Return whether one request/head LRU reached its soft target."""
        target_pages = self._group_targets.get(group, 0)
        if target_pages <= 0:
            return False

        group_state = self._group_states.get(group)
        return group_state is not None and group_state.num_pages >= target_pages

    def _evict_cluster(self, cluster_id: _ClusterId) -> None:
        slots = self._cluster_to_slots.pop(cluster_id, None)
        if slots is None:
            return

        logical_pages = self._cluster_to_pages.pop(cluster_id)
        group = self._cluster_to_group.pop(cluster_id)

        group_state = self._group_states.get(group)
        if group_state is None or cluster_id not in group_state.lru:
            raise RuntimeError(f"Resident group state is missing cluster {cluster_id}")

        group_state.lru.pop(cluster_id)
        group_state.num_pages -= len(slots)

        if group_state.num_pages < 0:
            raise RuntimeError("Resident group page count became negative")

        if not group_state.lru:
            if group_state.num_pages != 0:
                raise RuntimeError("An empty resident group still owns GPU pages")
            del self._group_states[group]

        for logical_page_id, slot_id in zip(logical_pages, slots):
            self._page_to_cluster.pop(logical_page_id, None)
            self._free_slots.add(slot_id)

    @staticmethod
    def _oldest_unprotected_cluster(
        group_state: _ResidentGroupState,
        protected_clusters: set[_ClusterId],
    ) -> _ClusterId | None:
        for cluster_id in group_state.lru:
            if cluster_id not in protected_clusters:
                return cluster_id

        return None

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

    def _evict_oldest_unprotected(
        self,
        protected_clusters: set[_ClusterId],
        incoming_group_pages: Mapping[RetroSpecClusterGroup, int],
    ) -> bool:
        victim = self._select_victim_cluster(
            protected_clusters,
            incoming_group_pages,
        )
        if victim is None:
            return False

        self._evict_cluster(victim)
        return True

    def resize(
        self,
        capacity: int,
        group_targets: Mapping[RetroSpecClusterGroup, int] | None = None,
    ) -> None:
        """Change layer capacity and update group soft targets.

        Physical storage grows when necessary but is not shrunk. Capacity
        reductions evict clusters from groups exceeding their soft targets first,
        using only each selected group's local LRU.
        """
        if capacity < 0:
            raise ValueError("Resident cache capacity must be non-negative")

        if group_targets is None:
            new_group_targets = dict(self._group_targets)
        else:
            new_group_targets = dict(group_targets)

        for group, target_pages in new_group_targets.items():
            if target_pages < 0:
                raise ValueError(
                    f"Resident target for group {group!r} must be non-negative"
                )

        if sum(new_group_targets.values()) > capacity:
            raise ValueError("Resident group targets cannot exceed layer capacity")

        self._grow_storage(capacity)
        self._logical_capacity = capacity
        self._group_targets = new_group_targets

        while self.num_resident_pages > capacity:
            if not self._evict_oldest_unprotected(
                protected_clusters=set(),
                incoming_group_pages={},
            ):
                raise RuntimeError("Resident cache cannot satisfy the reduced capacity")

    @staticmethod
    def _parse_clusters(
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
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

            if cluster_id not in allocated_cluster_ids:
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
                if logical_page_id not in allocated_page_ids:
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
        with torch.cuda.stream(self._copy_stream):
            for source_page_id, slot_id in zip(source_page_ids, slots):
                self.key_pages[slot_id].copy_(
                    source_key_pages[source_page_id],
                    non_blocking=True,
                )
                self.value_pages[slot_id].copy_(
                    source_value_pages[source_page_id],
                    non_blocking=True,
                )

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
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from CPU or GPU source pages."""
        self._validate_source_pages(source_key_pages, source_value_pages)

        if source_page_ids.shape != page_ids.shape:
            raise ValueError(
                "Source page IDs must have the same shape as logical page IDs"
            )
        if source_page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Source page IDs must use an integral dtype")

        (
            cluster_ids_cpu,
            _,
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

        requested_clusters = self._priority_ordered_clusters(
            parsed_cluster_ids,
            cluster_ids_cpu.shape,
        )

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

        target_clusters: list[_ClusterId] = []
        target_page_count = 0

        for cluster_id in requested_clusters:
            cluster_page_count = len(cluster_page_map[cluster_id])

            if cluster_page_count > self._logical_capacity:
                continue
            if target_page_count + cluster_page_count > self._logical_capacity:
                break

            target_clusters.append(cluster_id)
            target_page_count += cluster_page_count

        target_cluster_set = set(target_clusters)
        missing_targets = [
            cluster_id
            for cluster_id in target_clusters
            if cluster_id not in self._cluster_to_slots
        ]

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

        required_page_count = sum(
            len(cluster_page_map[cluster_id]) for cluster_id in missing_targets
        )

        incoming_group_pages: dict[RetroSpecClusterGroup, int] = {}
        for cluster_id in missing_targets:
            group = cluster_groups[cluster_id]
            incoming_group_pages[group] = incoming_group_pages.get(group, 0) + len(
                cluster_page_map[cluster_id]
            )

        while self.num_resident_pages + required_page_count > self._logical_capacity:
            if not self._evict_oldest_unprotected(
                protected_clusters=target_cluster_set,
                incoming_group_pages=incoming_group_pages,
            ):
                raise RuntimeError(
                    "Resident cache cannot free enough slots for priority admission"
                )

        copy_batch_started = False
        copy_scheduled = False
        copied_cluster_ids: list[_ClusterId] = []

        try:
            for cluster_id in missing_targets:
                logical_pages = cluster_page_map[cluster_id]
                cluster_page_count = len(logical_pages)
                slots = tuple(sorted(self._free_slots)[:cluster_page_count])

                if len(slots) != cluster_page_count:
                    raise RuntimeError(
                        "Resident cache does not have enough free GPU slots"
                    )

                for slot_id in slots:
                    self._free_slots.remove(slot_id)

                if not copy_batch_started:
                    self._reap_completed_copy_batches()

                    # Commit 24 requires this wait to occur after exact packing.
                    if reuse_ready_event is None:
                        current_stream = torch.cuda.current_stream(self.device)
                        self._copy_stream.wait_stream(current_stream)
                    else:
                        self._copy_stream.wait_event(reuse_ready_event)
                    copy_batch_started = True

                source_ids = cluster_source_ids[cluster_id]
                copy_scheduled = True

                try:
                    self._copy_cluster_to_slots(
                        source_ids,
                        slots,
                        source_key_pages,
                        source_value_pages,
                    )
                except Exception:
                    self._free_slots.update(slots)
                    raise

                self._register_cluster(
                    cluster_id=cluster_id,
                    group=cluster_groups[cluster_id],
                    logical_pages=logical_pages,
                    slots=slots,
                )
                copied_cluster_ids.append(cluster_id)
        finally:
            if copy_scheduled:
                self._record_copy_batch(
                    cluster_ids=tuple(copied_cluster_ids),
                    source_key_pages=source_key_pages,
                    source_value_pages=source_value_pages,
                )

        for cluster_id in reversed(requested_clusters):
            if cluster_id in self._cluster_to_slots:
                self._touch_cluster(cluster_id)

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
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from temporary GPU staging pages."""
        self._validate_source_pages(
            staging_key_pages,
            staging_value_pages,
        )
        if staging_key_pages.device != self.device:
            raise ValueError("Staging pages must use the resident cache CUDA device")

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
        released_cluster_ids = set(cluster_ids_cpu[cluster_ids_cpu >= 0].tolist())

        for cluster_id in released_cluster_ids:
            self._evict_cluster(cluster_id)
