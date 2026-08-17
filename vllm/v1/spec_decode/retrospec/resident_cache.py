# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict, deque
from collections.abc import Collection
from dataclasses import dataclass
from math import prod

import torch

_ClusterId = int
_LogicalPages = tuple[int, ...]


@dataclass(frozen=True)
class _PendingCopyBatch:
    """Keep asynchronous cache-copy sources alive until completion."""

    ready_event: torch.cuda.Event
    source_key_pages: torch.Tensor
    source_value_pages: torch.Tensor


@dataclass(frozen=True)
class RetroSpecResidentPageAccess:
    """Result of resolving logical cluster pages against the GPU cache."""

    cache_page_ids: torch.Tensor
    hit_cluster_mask: torch.Tensor
    miss_cluster_mask: torch.Tensor


class RetroSpecResidentClusterCache:
    """Bounded GPU page cache with cluster-atomic LRU ownership.

    Logical page IDs address the stable CPU backing store. Resident page IDs
    address slots in key_pages and value_pages.

    Every LRU entry owns all pages of one cluster. A cluster is therefore either
    completely resident or completely missing.
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

        # cluster_id is the stable identity used by lookup, LRU and admission.
        # Logical pages only describe where the cluster payload is stored.
        self._cluster_to_pages: dict[_ClusterId, _LogicalPages] = {}
        self._cluster_to_slots: dict[_ClusterId, tuple[int, ...]] = {}
        self._page_to_cluster: dict[int, _ClusterId] = {}
        self._lru: OrderedDict[_ClusterId, None] = OrderedDict()
        self._free_slots: set[int] = set()

        self._copy_stream = torch.cuda.Stream(device=device)
        self._pending_copy_batches: deque[_PendingCopyBatch] = deque()

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
    def num_pending_copy_batches(self) -> int:
        """Number of submitted copy batches not yet reaped by the host."""
        self._reap_completed_copy_batches()
        return len(self._pending_copy_batches)

    def _reap_completed_copy_batches(self) -> None:
        """Release source tensors belonging to completed copy batches."""
        while (
            self._pending_copy_batches
            and self._pending_copy_batches[0].ready_event.query()
        ):
            self._pending_copy_batches.popleft()

    def _record_copy_batch(
        self,
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> None:
        ready_event = torch.cuda.Event()
        ready_event.record(self._copy_stream)

        self._pending_copy_batches.append(
            _PendingCopyBatch(
                ready_event=ready_event,
                source_key_pages=source_key_pages,
                source_value_pages=source_value_pages,
            )
        )

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
            return

        # Every batch uses one copy stream, so completion of the final event also
        # implies completion of all earlier batches.
        self._pending_copy_batches[-1].ready_event.synchronize()
        self._pending_copy_batches.clear()

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

        self._free_slots.update(
            range(
                self._physical_capacity,
                required_capacity,
            )
        )

        self.key_pages = new_key_pages
        self.value_pages = new_value_pages
        self._physical_capacity = required_capacity

    def _evict_cluster(
        self,
        cluster_id: _ClusterId,
    ) -> None:
        slots = self._cluster_to_slots.pop(cluster_id, None)
        if slots is None:
            return

        logical_pages = self._cluster_to_pages.pop(cluster_id)
        self._lru.pop(cluster_id, None)

        for logical_page_id, slot_id in zip(logical_pages, slots):
            self._page_to_cluster.pop(logical_page_id, None)
            self._free_slots.add(slot_id)

    def _evict_oldest_unprotected(
        self,
        protected_clusters: set[_ClusterId],
    ) -> bool:
        victim = None

        for cluster_id in self._lru:
            if cluster_id not in protected_clusters:
                victim = cluster_id
                break

        if victim is None:
            return False

        self._evict_cluster(victim)
        return True

    def resize(self, capacity: int) -> None:
        """Change the active resident-page capacity.

        Physical storage grows when necessary but is not shrunk. Reducing the
        active capacity evicts complete LRU clusters until the resident page
        count satisfies the new limit.
        """
        if capacity < 0:
            raise ValueError("Resident cache capacity must be non-negative")

        self._grow_storage(capacity)
        self._logical_capacity = capacity

        while self.num_resident_pages > capacity:
            oldest_cluster = next(iter(self._lru))
            self._evict_cluster(oldest_cluster)

    @staticmethod
    def _parse_clusters(
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
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
        if cluster_ids.device != page_ids.device:
            raise ValueError("Cluster IDs and page IDs must use one device")

        cluster_ids_cpu = cluster_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        page_ids_cpu = page_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )

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

    def lookup(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        touch: bool = True,
    ) -> RetroSpecResidentPageAccess:
        """Resolve selected cluster blocks against resident GPU slots."""
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
        )

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

        flat_hit_mask = hit_cluster_mask_cpu.reshape(-1)
        flat_miss_mask = miss_cluster_mask_cpu.reshape(-1)
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

            resident_slots = self._cluster_to_slots.get(cluster_id)
            if resident_slots is None:
                flat_miss_mask[cluster_index] = True
                continue

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
                    self._lru.move_to_end(cluster_id, last=True)

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
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        source_page_ids: torch.Tensor,
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from CPU or GPU source pages."""
        self._validate_source_pages(
            source_key_pages,
            source_value_pages,
        )

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
        )

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

            if resident_pages is not None and resident_pages != logical_pages:
                raise RuntimeError(
                    "Cluster ID conflicts with an existing resident descriptor"
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

        # Validate all mappings before changing cache ownership.
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

        while self.num_resident_pages + required_page_count > self._logical_capacity:
            if not self._evict_oldest_unprotected(target_cluster_set):
                raise RuntimeError(
                    "Resident cache cannot free enough slots for priority admission"
                )

        copy_batch_started = False
        copy_scheduled = False

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
                    current_stream = torch.cuda.current_stream(self.device)
                    self._copy_stream.wait_stream(current_stream)
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

                self._cluster_to_pages[cluster_id] = logical_pages
                self._cluster_to_slots[cluster_id] = slots
                self._lru[cluster_id] = None

                for logical_page_id in logical_pages:
                    self._page_to_cluster[logical_page_id] = cluster_id
        finally:
            if copy_scheduled:
                self._record_copy_batch(
                    source_key_pages,
                    source_value_pages,
                )

        for cluster_id in reversed(requested_clusters):
            if cluster_id in self._cluster_to_slots:
                self._lru.move_to_end(cluster_id, last=True)

        return self.lookup(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            touch=False,
        )

    def admit(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        backing_key_pages: torch.Tensor,
        backing_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from stable CPU backing pages."""
        self._validate_backing_pages(
            backing_key_pages,
            backing_value_pages,
        )

        return self._admit_from_sources(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=page_ids,
            source_key_pages=backing_key_pages,
            source_value_pages=backing_value_pages,
        )

    def admit_staged(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
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
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=staging_page_ids,
            source_key_pages=staging_key_pages,
            source_value_pages=staging_value_pages,
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
