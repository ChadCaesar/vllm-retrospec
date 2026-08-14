# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict, deque
from collections.abc import Collection
from dataclasses import dataclass
from math import prod

import torch

_ClusterKey = tuple[int, ...]


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

        self._cluster_to_slots: dict[_ClusterKey, tuple[int, ...]] = {}
        self._page_to_cluster: dict[int, _ClusterKey] = {}
        self._lru: OrderedDict[_ClusterKey, None] = OrderedDict()
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
        cluster_key: _ClusterKey,
    ) -> None:
        slots = self._cluster_to_slots.pop(
            cluster_key,
            None,
        )
        if slots is None:
            return

        self._lru.pop(
            cluster_key,
            None,
        )

        for logical_page_id, slot_id in zip(
            cluster_key,
            slots,
        ):
            self._page_to_cluster.pop(
                logical_page_id,
                None,
            )
            self._free_slots.add(slot_id)

    def _evict_oldest_unprotected(
        self,
        protected_clusters: set[_ClusterKey],
    ) -> bool:
        victim = None

        for cluster_key in self._lru:
            if cluster_key not in protected_clusters:
                victim = cluster_key
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
        page_ids: torch.Tensor,
        allocated_page_ids: Collection[int],
    ) -> tuple[
        torch.Tensor,
        list[_ClusterKey],
        list[tuple[int, ...]],
    ]:
        if page_ids.ndim < 1:
            raise ValueError("Cluster page IDs must have at least one dimension")
        if page_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Cluster page IDs must use an integral dtype")

        page_ids_cpu = page_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )

        if torch.any(page_ids_cpu < -1).item():
            raise ValueError("Cluster page IDs must be at least -1")

        leading_shape = page_ids_cpu.shape[:-1]
        max_pages_per_cluster = page_ids_cpu.shape[-1]
        num_clusters = prod(leading_shape)

        flattened_page_ids = page_ids_cpu.reshape(
            num_clusters,
            max_pages_per_cluster,
        )

        cluster_keys: list[_ClusterKey] = []
        valid_positions: list[tuple[int, ...]] = []
        requested_page_owners: dict[int, _ClusterKey] = {}

        for row in flattened_page_ids.tolist():
            positions = tuple(
                index for index, page_id in enumerate(row) if page_id >= 0
            )
            cluster_key = tuple(row[index] for index in positions)

            if len(cluster_key) != len(set(cluster_key)):
                raise ValueError(
                    "A cluster cannot reference the same logical page twice"
                )

            for logical_page_id in cluster_key:
                if logical_page_id not in allocated_page_ids:
                    raise RuntimeError(
                        "Cluster page table references an unallocated "
                        f"logical page {logical_page_id}"
                    )

                previous_owner = requested_page_owners.get(logical_page_id)
                if previous_owner is not None and previous_owner != cluster_key:
                    raise ValueError(
                        "A logical page cannot belong to multiple clusters"
                    )
                requested_page_owners[logical_page_id] = cluster_key

            cluster_keys.append(cluster_key)
            valid_positions.append(positions)

        return (
            page_ids_cpu,
            cluster_keys,
            valid_positions,
        )

    @staticmethod
    def _priority_ordered_clusters(
        cluster_keys: list[_ClusterKey],
        leading_shape: torch.Size,
    ) -> list[_ClusterKey]:
        """Return unique clusters in retrieval-priority order.

        For a page table shaped [..., ranked_clusters, pages], the final leading
        dimension is the retrieval rank. Earlier leading dimensions identify
        independent request/KV-head groups.

        Rank-major ordering prevents the first request or KV head from consuming
        the entire shared resident cache before other groups receive their
        highest-ranked cluster.
        """
        if len(leading_shape) <= 1:
            ordered_clusters = cluster_keys
        else:
            num_ranked_clusters = leading_shape[-1]
            num_groups = prod(leading_shape[:-1])

            ordered_clusters = [
                cluster_keys[group_index * num_ranked_clusters + rank]
                for rank in range(num_ranked_clusters)
                for group_index in range(num_groups)
            ]

        return list(
            dict.fromkeys(
                cluster_key for cluster_key in ordered_clusters if cluster_key
            )
        )

    def lookup(
        self,
        page_ids: torch.Tensor,
        allocated_page_ids: Collection[int],
        touch: bool = True,
    ) -> RetroSpecResidentPageAccess:
        """Resolve selected logical clusters against resident GPU slots."""
        (
            page_ids_cpu,
            cluster_keys,
            valid_positions,
        ) = self._parse_clusters(
            page_ids,
            allocated_page_ids,
        )

        cache_page_ids_cpu = torch.full_like(
            page_ids_cpu,
            -1,
        )
        flat_cache_page_ids = cache_page_ids_cpu.reshape(
            len(cluster_keys),
            page_ids_cpu.shape[-1],
        )

        cluster_mask_shape = page_ids_cpu.shape[:-1]
        hit_cluster_mask_cpu = torch.zeros(
            cluster_mask_shape,
            dtype=torch.bool,
        )
        miss_cluster_mask_cpu = torch.zeros_like(hit_cluster_mask_cpu)

        flat_hit_mask = hit_cluster_mask_cpu.reshape(-1)
        flat_miss_mask = miss_cluster_mask_cpu.reshape(-1)
        hit_clusters: set[_ClusterKey] = set()

        for cluster_index, (
            cluster_key,
            positions,
        ) in enumerate(
            zip(
                cluster_keys,
                valid_positions,
            )
        ):
            if not cluster_key:
                continue

            resident_slots = self._cluster_to_slots.get(cluster_key)
            if resident_slots is None:
                flat_miss_mask[cluster_index] = True
                continue

            flat_hit_mask[cluster_index] = True
            hit_clusters.add(cluster_key)

            for position, slot_id in zip(
                positions,
                resident_slots,
            ):
                flat_cache_page_ids[
                    cluster_index,
                    position,
                ] = slot_id

        if touch:
            requested_clusters = self._priority_ordered_clusters(
                cluster_keys,
                page_ids_cpu.shape[:-1],
            )

            # OrderedDict stores oldest entries first and newest entries last.
            # Touch low-priority entries first so rank-zero clusters end up newest.
            for cluster_key in reversed(requested_clusters):
                if cluster_key in hit_clusters:
                    self._lru.move_to_end(
                        cluster_key,
                        last=True,
                    )

        return RetroSpecResidentPageAccess(
            cache_page_ids=cache_page_ids_cpu.to(
                device=page_ids.device,
                non_blocking=False,
            ),
            hit_cluster_mask=hit_cluster_mask_cpu.to(
                device=page_ids.device,
                non_blocking=False,
            ),
            miss_cluster_mask=miss_cluster_mask_cpu.to(
                device=page_ids.device,
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
        page_ids: torch.Tensor,
        allocated_page_ids: Collection[int],
        source_page_ids: torch.Tensor,
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority prefix using CPU or GPU source pages."""
        self._validate_source_pages(
            source_key_pages,
            source_value_pages,
        )

        if source_page_ids.shape != page_ids.shape:
            raise ValueError(
                "Source page IDs must have the same shape as logical page IDs"
            )
        if source_page_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Source page IDs must use an integral dtype")

        (
            page_ids_cpu,
            cluster_keys,
            valid_positions,
        ) = self._parse_clusters(
            page_ids,
            allocated_page_ids,
        )

        source_page_ids_cpu = source_page_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        if torch.any(source_page_ids_cpu < -1).item():
            raise ValueError("Source page IDs must be at least -1")

        flat_source_page_ids = source_page_ids_cpu.reshape(
            len(cluster_keys),
            source_page_ids_cpu.shape[-1],
        )

        cluster_source_ids: dict[_ClusterKey, tuple[int, ...]] = {}

        for cluster_key, positions, source_row in zip(
            cluster_keys,
            valid_positions,
            flat_source_page_ids,
        ):
            if not cluster_key:
                continue

            current_source_ids = tuple(
                int(source_row[position]) for position in positions
            )
            previous_source_ids = cluster_source_ids.get(cluster_key)

            # A duplicated logical cluster may appear in more than one request
            # position. Prefer an occurrence with complete source-page mappings.
            if previous_source_ids is None or any(
                source_page_id < 0 for source_page_id in previous_source_ids
            ):
                cluster_source_ids[cluster_key] = current_source_ids

        requested_clusters = self._priority_ordered_clusters(
            cluster_keys,
            page_ids_cpu.shape[:-1],
        )

        for cluster_key in requested_clusters:
            for logical_page_id in cluster_key:
                resident_owner = self._page_to_cluster.get(logical_page_id)
                if resident_owner is not None and resident_owner != cluster_key:
                    raise RuntimeError(
                        "Logical page ownership conflicts with an existing "
                        "resident cluster"
                    )

        target_clusters: list[_ClusterKey] = []
        target_page_count = 0

        for cluster_key in requested_clusters:
            cluster_page_count = len(cluster_key)

            if cluster_page_count > self._logical_capacity:
                continue

            if target_page_count + cluster_page_count > self._logical_capacity:
                break

            target_clusters.append(cluster_key)
            target_page_count += cluster_page_count

        target_cluster_set = set(target_clusters)
        missing_targets = [
            cluster_key
            for cluster_key in target_clusters
            if cluster_key not in self._cluster_to_slots
        ]

        # Validate every source before evicting existing data.
        for cluster_key in missing_targets:
            source_ids = cluster_source_ids.get(cluster_key)
            if source_ids is None or len(source_ids) != len(cluster_key):
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

        required_page_count = sum(len(cluster_key) for cluster_key in missing_targets)

        while self.num_resident_pages + required_page_count > self._logical_capacity:
            if not self._evict_oldest_unprotected(target_cluster_set):
                raise RuntimeError(
                    "Resident cache cannot free enough slots for priority admission"
                )

        copy_batch_started = False
        copy_scheduled = False

        try:
            for cluster_key in missing_targets:
                cluster_page_count = len(cluster_key)
                slots = tuple(sorted(self._free_slots)[:cluster_page_count])
                if len(slots) != cluster_page_count:
                    raise RuntimeError(
                        "Resident cache does not have enough free GPU slots"
                    )

                for slot_id in slots:
                    self._free_slots.remove(slot_id)

                if not copy_batch_started:
                    self._reap_completed_copy_batches()

                    # For staged admission this wait is recorded after the exact
                    # packing kernels. Cache slots cannot be overwritten before
                    # the current verification has consumed them.
                    current_stream = torch.cuda.current_stream(self.device)
                    self._copy_stream.wait_stream(current_stream)
                    copy_batch_started = True

                source_ids = cluster_source_ids[cluster_key]
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

                self._cluster_to_slots[cluster_key] = slots
                self._lru[cluster_key] = None

                for logical_page_id in cluster_key:
                    self._page_to_cluster[logical_page_id] = cluster_key
        finally:
            if copy_scheduled:
                self._record_copy_batch(
                    source_key_pages,
                    source_value_pages,
                )

        # Touch every selected resident cluster, but preserve retrieval priority:
        # rank-zero clusters become the newest entries.
        for cluster_key in reversed(requested_clusters):
            if cluster_key in self._cluster_to_slots:
                self._lru.move_to_end(
                    cluster_key,
                    last=True,
                )

        return self.lookup(
            page_ids,
            allocated_page_ids,
            touch=False,
        )

    def admit(
        self,
        page_ids: torch.Tensor,
        allocated_page_ids: Collection[int],
        backing_key_pages: torch.Tensor,
        backing_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority prefix from stable CPU backing pages."""
        self._validate_backing_pages(
            backing_key_pages,
            backing_value_pages,
        )

        return self._admit_from_sources(
            page_ids=page_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=page_ids,
            source_key_pages=backing_key_pages,
            source_value_pages=backing_value_pages,
        )

    def admit_staged(
        self,
        page_ids: torch.Tensor,
        allocated_page_ids: Collection[int],
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority prefix from temporary GPU staging pages.

        The caller must invoke this only after all current consumers of resident
        slots and staging pages have been submitted to the current CUDA stream.
        The cache copy stream then waits for that stream before updating slots.
        """
        self._validate_source_pages(
            staging_key_pages,
            staging_value_pages,
        )
        if staging_key_pages.device != self.device:
            raise ValueError("Staging pages must use the resident cache CUDA device")

        return self._admit_from_sources(
            page_ids=page_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=staging_page_ids,
            source_key_pages=staging_key_pages,
            source_value_pages=staging_value_pages,
        )

    def invalidate(
        self,
        logical_page_ids: torch.Tensor,
    ) -> None:
        """Evict clusters owning any released logical page.

        Backing-page release may allow the CPU allocator to immediately reuse the
        same memory, so pending H2D reads must finish before invalidation returns.
        """
        self.synchronize_pending_copies()

        page_ids_cpu = logical_page_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        released_page_ids = set(page_ids_cpu[page_ids_cpu >= 0].tolist())

        affected_clusters = {
            cluster_key
            for logical_page_id in released_page_ids
            if (cluster_key := self._page_to_cluster.get(logical_page_id)) is not None
        }

        for cluster_key in affected_clusters:
            self._evict_cluster(cluster_key)
