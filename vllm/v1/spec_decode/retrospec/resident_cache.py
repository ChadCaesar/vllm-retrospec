# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Collection
from dataclasses import dataclass
from math import prod

import torch

_ClusterKey = tuple[int, ...]


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

        self._cluster_to_slots: dict[
            _ClusterKey,
            tuple[int, ...],
        ] = {}
        self._page_to_cluster: dict[
            int,
            _ClusterKey,
        ] = {}
        self._lru: OrderedDict[
            _ClusterKey,
            None,
        ] = OrderedDict()

        self._free_slots: set[int] = set()

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

    def _grow_storage(self, required_capacity: int) -> None:
        if required_capacity <= self._physical_capacity:
            return

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

        touched_clusters: set[_ClusterKey] = set()

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

            for position, slot_id in zip(
                positions,
                resident_slots,
            ):
                flat_cache_page_ids[
                    cluster_index,
                    position,
                ] = slot_id

            if touch and cluster_key not in touched_clusters:
                self._lru.move_to_end(
                    cluster_key,
                    last=True,
                )
                touched_clusters.add(cluster_key)

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

    def _validate_backing_pages(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
    ) -> None:
        if key_pages.shape != value_pages.shape:
            raise ValueError("Backing key and value page shapes must match")
        if key_pages.ndim != 3:
            raise ValueError(
                "Backing pages must have shape [pages, page_size, head_size]"
            )
        if key_pages.shape[1:] != (
            self.page_size,
            self.head_size,
        ):
            raise ValueError("Backing page shape does not match resident cache")
        if key_pages.dtype != self.dtype:
            raise ValueError("Backing key dtype does not match resident cache")
        if value_pages.dtype != self.dtype:
            raise ValueError("Backing value dtype does not match resident cache")

    def _copy_cluster_to_slots(
        self,
        cluster_key: _ClusterKey,
        slots: tuple[int, ...],
        backing_key_pages: torch.Tensor,
        backing_value_pages: torch.Tensor,
    ) -> None:
        logical_page_ids = torch.tensor(
            cluster_key,
            dtype=torch.int64,
            device=backing_key_pages.device,
        )
        resident_slot_ids = torch.tensor(
            slots,
            dtype=torch.int64,
            device=self.device,
        )

        source_keys = backing_key_pages.index_select(
            0,
            logical_page_ids,
        )
        source_values = backing_value_pages.index_select(
            0,
            logical_page_ids,
        )

        # Commit 19 provides the synchronous correctness path. Commit 20
        # replaces this staging with stream/event-managed asynchronous copies.
        resident_keys = source_keys.to(
            device=self.device,
            non_blocking=False,
        )
        resident_values = source_values.to(
            device=self.device,
            non_blocking=False,
        )

        self.key_pages.index_copy_(
            0,
            resident_slot_ids,
            resident_keys,
        )
        self.value_pages.index_copy_(
            0,
            resident_slot_ids,
            resident_values,
        )

    def admit(
        self,
        page_ids: torch.Tensor,
        allocated_page_ids: Collection[int],
        backing_key_pages: torch.Tensor,
        backing_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        """Synchronously admit a priority-ordered cluster prefix.

        Clusters already resident are protected and moved to MRU. Missing
        clusters are admitted in input order while complete clusters fit in the
        active page capacity. Lower-priority clusters are left as misses.
        """
        self._validate_backing_pages(
            backing_key_pages,
            backing_value_pages,
        )
        (
            _,
            cluster_keys,
            _,
        ) = self._parse_clusters(
            page_ids,
            allocated_page_ids,
        )

        requested_clusters = list(
            dict.fromkeys(cluster_key for cluster_key in cluster_keys if cluster_key)
        )

        protected_clusters: set[_ClusterKey] = set()
        protected_page_count = 0

        for cluster_key in requested_clusters:
            for logical_page_id in cluster_key:
                resident_owner = self._page_to_cluster.get(logical_page_id)
                if resident_owner is not None and resident_owner != cluster_key:
                    raise RuntimeError(
                        "Logical page ownership conflicts with an existing "
                        "resident cluster"
                    )

            if cluster_key not in self._cluster_to_slots:
                continue

            self._lru.move_to_end(
                cluster_key,
                last=True,
            )
            protected_clusters.add(cluster_key)
            protected_page_count += len(cluster_key)

        for cluster_key in requested_clusters:
            if cluster_key in self._cluster_to_slots:
                continue

            cluster_page_count = len(cluster_key)
            if cluster_page_count > self._logical_capacity:
                continue

            if protected_page_count + cluster_page_count > self._logical_capacity:
                break

            while self.num_resident_pages + cluster_page_count > self._logical_capacity:
                if not self._evict_oldest_unprotected(protected_clusters):
                    break

            if self.num_resident_pages + cluster_page_count > self._logical_capacity:
                break

            slots = tuple(sorted(self._free_slots)[:cluster_page_count])
            if len(slots) != cluster_page_count:
                raise RuntimeError("Resident cache does not have enough free GPU slots")

            for slot_id in slots:
                self._free_slots.remove(slot_id)

            try:
                self._copy_cluster_to_slots(
                    cluster_key,
                    slots,
                    backing_key_pages,
                    backing_value_pages,
                )
            except Exception:
                self._free_slots.update(slots)
                raise

            self._cluster_to_slots[cluster_key] = slots
            self._lru[cluster_key] = None

            for logical_page_id in cluster_key:
                self._page_to_cluster[logical_page_id] = cluster_key

            protected_clusters.add(cluster_key)
            protected_page_count += cluster_page_count

        return self.lookup(
            page_ids,
            allocated_page_ids,
            touch=False,
        )

    def invalidate(
        self,
        logical_page_ids: torch.Tensor,
    ) -> None:
        """Evict clusters owning any released logical page."""
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
