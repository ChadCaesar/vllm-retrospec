# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from math import ceil
from typing import Literal

import torch

from .cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from .resident_cache import (
    RetroSpecResidentClusterCache,
    RetroSpecResidentPageAccess,
)

RetroSpecClusterStorageMode = Literal[
    "gpu_reference",
    "cpu_offload",
]
RetroSpecClusterResolveMode = Literal[
    "resident_only",
    "verification",
]


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
class _ClusterBlockDescriptor:
    """CPU metadata mapped from one layer-global stable cluster handle."""

    identity: RetroSpecClusterIdentity
    page_ids: tuple[int, ...]
    page_token_counts: tuple[int, ...]


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

    resident_ready_event is recorded on the resident-cache copy stream. The
    execution path may perform work independent of resident pages before
    waiting on this event.
    """

    resident_page_ids: torch.Tensor
    staging_page_ids: torch.Tensor

    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor

    staging_key_pages: torch.Tensor
    staging_value_pages: torch.Tensor

    hit_cluster_mask: torch.Tensor
    miss_cluster_mask: torch.Tensor
    resident_ready_event: torch.cuda.Event | None


class _LayerClusterPagePool:
    """Growable cluster backing-page pool for one attention layer."""

    _MIN_CAPACITY = 64

    def __init__(
        self,
        page_size: int,
        head_size: int,
        dtype: torch.dtype,
        storage_device: torch.device,
        metadata_device: torch.device,
        pin_memory: bool,
    ) -> None:
        if pin_memory and storage_device.type != "cpu":
            raise ValueError("Only CPU cluster storage can use pinned memory")

        self.page_size = page_size
        self.head_size = head_size
        self.dtype = dtype
        self.storage_device = storage_device
        self.metadata_device = metadata_device
        self.pin_memory = pin_memory

        self.key_pages = self._allocate_page_tensor(0)
        self.value_pages = self._allocate_page_tensor(0)

        # Allocation state remains on the CPU because page allocation and
        # request release are control-plane operations.
        self._free_page_ids: list[int] = []
        self._allocated_page_ids: set[int] = set()

    def _allocate_page_tensor(
        self,
        num_pages: int,
    ) -> torch.Tensor:
        shape = (
            num_pages,
            self.page_size,
            self.head_size,
        )

        if self.storage_device.type == "cpu":
            return torch.empty(
                shape,
                dtype=self.dtype,
                device=self.storage_device,
                pin_memory=self.pin_memory,
            )

        return torch.empty(
            shape,
            dtype=self.dtype,
            device=self.storage_device,
        )

    @property
    def capacity(self) -> int:
        return self.key_pages.shape[0]

    @property
    def num_allocated_pages(self) -> int:
        return len(self._allocated_page_ids)

    @property
    def allocated_page_ids(self) -> set[int]:
        """Return allocator-owned IDs for internal membership checks."""
        return self._allocated_page_ids

    def _grow(self, required_capacity: int) -> None:
        old_capacity = self.capacity
        new_capacity = max(
            self._MIN_CAPACITY,
            old_capacity,
        )

        while new_capacity < required_capacity:
            new_capacity *= 2

        new_key_pages = self._allocate_page_tensor(new_capacity)
        new_value_pages = self._allocate_page_tensor(new_capacity)

        if old_capacity:
            new_key_pages[:old_capacity].copy_(self.key_pages)
            new_value_pages[:old_capacity].copy_(self.value_pages)

        self.key_pages = new_key_pages
        self.value_pages = new_value_pages

        # Reverse insertion makes pop() return the lowest new page ID first.
        self._free_page_ids.extend(
            range(
                new_capacity - 1,
                old_capacity - 1,
                -1,
            )
        )

    def allocate(self, num_pages: int) -> torch.Tensor:
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages == 0:
            return torch.empty(
                0,
                dtype=torch.int64,
                device=self.storage_device,
            )

        missing_pages = num_pages - len(self._free_page_ids)
        if missing_pages > 0:
            self._grow(self.capacity + missing_pages)

        page_ids = [self._free_page_ids.pop() for _ in range(num_pages)]

        for page_id in page_ids:
            if page_id in self._allocated_page_ids:
                raise RuntimeError(
                    f"RetroSpec cluster page {page_id} is already allocated"
                )
            self._allocated_page_ids.add(page_id)

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

        for page_id in sorted(
            unique_page_ids,
            reverse=True,
        ):
            self._allocated_page_ids.remove(page_id)
            self._free_page_ids.append(page_id)

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

        self.key_pages.index_copy_(
            0,
            page_ids,
            key_pages,
        )
        self.value_pages.index_copy_(
            0,
            page_ids,
            value_pages,
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

        valid_page_ids = storage_page_ids[storage_page_ids >= 0]
        if valid_page_ids.numel() == 0:
            empty_keys = torch.zeros(
                output_shape,
                dtype=self.dtype,
                device=self.storage_device,
            )
            return empty_keys, empty_keys.clone()

        if valid_page_ids.max().item() >= self.capacity:
            raise RuntimeError("Cluster page table references a page outside the pool")

        # Invalid padded page IDs read page zero. Their vectors are removed by
        # page_token_counts before attention.
        safe_page_ids = storage_page_ids.clamp_min(0)
        key_pages = self.key_pages.index_select(
            0,
            safe_page_ids.reshape(-1),
        )
        value_pages = self.value_pages.index_select(
            0,
            safe_page_ids.reshape(-1),
        )

        return (
            key_pages.view(output_shape),
            value_pages.view(output_shape),
        )


class RetroSpecClusterPageStore:
    """Per-layer secondary KV store organized by token clusters.

    gpu_reference stores complete cluster pages on the model CUDA device.

    cpu_offload stores complete cluster pages in CPU memory. When supported,
    pinned memory enables asynchronous host-to-device admission into a bounded
    GPU resident cluster cache.
    """

    def __init__(
        self,
        page_size: int,
        storage_mode: RetroSpecClusterStorageMode = "gpu_reference",
        pin_memory: bool = False,
        cache_ratio: float = 0.0,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if storage_mode not in (
            "gpu_reference",
            "cpu_offload",
        ):
            raise ValueError(
                f"Unsupported RetroSpec cluster storage mode: {storage_mode}"
            )
        if not 0.0 <= cache_ratio <= 1.0:
            raise ValueError("cache_ratio must be between zero and one")

        self.page_size = page_size
        self.storage_mode = storage_mode
        self.pin_memory = pin_memory if storage_mode == "cpu_offload" else False
        self.cache_ratio = cache_ratio

        self._layer_pools: dict[str, _LayerClusterPagePool] = {}
        self._resident_caches: dict[str, RetroSpecResidentClusterCache] = {}

        self._next_cluster_ids: dict[str, int] = {}
        self._allocated_cluster_ids: dict[str, set[int]] = {}
        self._cluster_block_descriptors: dict[
            str, dict[int, _ClusterBlockDescriptor]
        ] = {}

    @property
    def is_cpu_backed(self) -> bool:
        return self.storage_mode == "cpu_offload"

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
            pin_memory=self.pin_memory,
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

        if allocated.intersection(new_descriptors):
            raise RuntimeError("RetroSpec cluster ID allocator produced a duplicate ID")

        allocated.update(new_descriptors)
        descriptors.update(new_descriptors)
        self._next_cluster_ids[layer_name] = cluster_id_end

        return cluster_ids_cpu

    def _free_cluster_ids(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
    ) -> None:
        allocated = self._allocated_cluster_ids.get(layer_name)
        descriptors = self._cluster_block_descriptors.get(layer_name)

        if allocated is None or descriptors is None:
            raise RuntimeError(f"No RetroSpec clusters exist for layer {layer_name!r}")

        cluster_ids_cpu = cluster_ids.detach().to(device="cpu", dtype=torch.int64)
        released = set(cluster_ids_cpu[cluster_ids_cpu >= 0].tolist())

        for cluster_id in released:
            if cluster_id not in allocated:
                raise RuntimeError(f"RetroSpec cluster {cluster_id} is not allocated")

        allocated.difference_update(released)

        for cluster_id in released:
            del descriptors[cluster_id]

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
    ) -> torch.Tensor:
        if page_ids.ndim != cluster_ids.ndim + 1:
            raise ValueError("Cluster pages must add one page dimension to cluster IDs")
        if page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Cluster ID and page-table shapes do not match")
        if page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cluster page IDs must use an integral dtype")
        if cluster_ids.device != page_ids.device:
            raise ValueError("Cluster IDs and page IDs must use one device")

        cluster_ids_cpu = self._validate_cluster_ids(layer_name, cluster_ids)
        page_ids_cpu = page_ids.detach().to(device="cpu", dtype=torch.int64)

        if torch.any(page_ids_cpu < -1).item():
            raise ValueError("Cluster page IDs must be at least -1")

        descriptors = self._cluster_block_descriptors[layer_name]
        flat_cluster_ids = cluster_ids_cpu.reshape(-1).tolist()
        flat_page_ids = page_ids_cpu.reshape(
            len(flat_cluster_ids), page_ids_cpu.shape[-1]
        ).tolist()

        for cluster_id, page_row in zip(flat_cluster_ids, flat_page_ids):
            logical_pages = tuple(page_id for page_id in page_row if page_id >= 0)

            if cluster_id < 0:
                if logical_pages:
                    raise ValueError(
                        "A padded cluster ID cannot reference logical pages"
                    )
                continue

            expected_pages = descriptors[cluster_id].page_ids
            if logical_pages != expected_pages:
                raise RuntimeError(
                    "Cluster page descriptor does not match its stable cluster ID"
                )

        return cluster_ids_cpu

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
        descriptors = self._cluster_block_descriptors[layer_name]

        max_pages = max(
            (
                len(descriptors[cluster_id].page_ids)
                for cluster_id in cluster_ids_cpu.reshape(-1).tolist()
                if cluster_id >= 0
            ),
            default=0,
        )
        output_shape = (*cluster_ids_cpu.shape, max_pages)

        page_ids_cpu = torch.full(
            output_shape,
            -1,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        page_token_counts_cpu = torch.zeros(
            output_shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
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

        target_device = cluster_ids.device if device is None else device

        return RetroSpecClusterBlockMetadata(
            page_ids=page_ids_cpu.to(device=target_device, non_blocking=False),
            page_token_counts=page_token_counts_cpu.to(
                device=target_device, non_blocking=False
            ),
        )

    def num_allocated_clusters(
        self,
        layer_name: str,
    ) -> int:
        allocated = self._allocated_cluster_ids.get(layer_name)
        return 0 if allocated is None else len(allocated)

    def _get_storage_device(
        self,
        vectors: torch.Tensor,
    ) -> torch.device:
        if self.is_cpu_backed:
            return torch.device("cpu")
        return vectors.device

    def _get_or_create_pool(
        self,
        layer_name: str,
        vectors: torch.Tensor,
    ) -> _LayerClusterPagePool:
        if vectors.ndim != 3:
            raise ValueError(
                "Cluster vectors must have shape [num_kv_heads, num_tokens, head_size]"
            )

        head_size = vectors.shape[2]
        storage_device = self._get_storage_device(vectors)
        pool = self._layer_pools.get(layer_name)

        if pool is None:
            pool = _LayerClusterPagePool(
                page_size=self.page_size,
                head_size=head_size,
                dtype=vectors.dtype,
                storage_device=storage_device,
                metadata_device=vectors.device,
                pin_memory=self.pin_memory,
            )
            self._layer_pools[layer_name] = pool
            return pool

        if pool.head_size != head_size:
            raise ValueError("Cluster vectors do not match the layer head size")
        if pool.dtype != vectors.dtype:
            raise ValueError("Cluster vectors do not match the layer KV dtype")
        if pool.storage_device != storage_device:
            raise ValueError("Cluster vectors do not match the layer storage device")
        if pool.metadata_device != vectors.device:
            raise ValueError("Cluster metadata device changed for an existing layer")

        return pool

    def _resident_target_capacity(
        self,
        pool: _LayerClusterPagePool,
    ) -> int:
        if not self.is_cpu_backed:
            return 0
        if pool.num_allocated_pages == 0:
            return 0

        return min(
            pool.num_allocated_pages,
            ceil(pool.num_allocated_pages * self.cache_ratio),
        )

    def _resize_resident_cache(
        self,
        layer_name: str,
        pool: _LayerClusterPagePool,
    ) -> None:
        resident_cache = self._resident_caches.get(layer_name)
        if resident_cache is None:
            return

        resident_cache.resize(self._resident_target_capacity(pool))

    def _get_or_create_resident_cache(
        self,
        layer_name: str,
    ) -> tuple[
        _LayerClusterPagePool,
        RetroSpecResidentClusterCache,
    ]:
        if not self.is_cpu_backed:
            raise RuntimeError(
                "Resident GPU cache is only used by CPU-backed cluster storage"
            )

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

        resident_cache.resize(self._resident_target_capacity(pool))

        return pool, resident_cache

    @staticmethod
    def _move_to_storage(
        tensor: torch.Tensor,
        storage_device: torch.device,
    ) -> torch.Tensor:
        if tensor.device == storage_device:
            return tensor

        # Commit 18 deliberately performs a synchronous control-plane copy.
        # Async D2H staging and stream/event management are added separately.
        return tensor.to(
            device=storage_device,
            non_blocking=False,
        )

    @staticmethod
    def _allocate_packed_pages(
        pool: _LayerClusterPagePool,
        num_pages: int,
    ) -> torch.Tensor:
        shape = (
            num_pages,
            pool.page_size,
            pool.head_size,
        )

        if pool.storage_device.type == "cpu":
            return torch.zeros(
                shape,
                dtype=pool.dtype,
                device=pool.storage_device,
                pin_memory=pool.pin_memory,
            )

        return torch.zeros(
            shape,
            dtype=pool.dtype,
            device=pool.storage_device,
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
    ) -> RetroSpecClusterBlockTable:
        """Reorder token KV into stable per-head, per-cluster backing pages."""
        if cluster_start < 0:
            raise ValueError("cluster_start must be non-negative")
        if token_values.shape != token_keys.shape:
            raise ValueError("token_keys and token_values must have equal shapes")
        if assignments.shape != token_keys.shape[:2]:
            raise ValueError("assignments must have shape [num_kv_heads, num_tokens]")
        if cluster_token_counts.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if cluster_token_counts.shape[0] != token_keys.shape[0]:
            raise ValueError(
                "cluster_token_counts KV-head count does not match token KV"
            )
        if assignments.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("assignments must use an integral dtype")
        if cluster_token_counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cluster_token_counts must use an integral dtype")
        if token_values.device != token_keys.device:
            raise ValueError("Token keys and values must be on one device")
        if assignments.device != token_keys.device:
            raise ValueError("Assignments and token KV must be on one device")
        if cluster_token_counts.device != token_keys.device:
            raise ValueError("Cluster counts and token KV must be on one device")
        if torch.any(cluster_token_counts < 0).item():
            raise ValueError("cluster_token_counts must be non-negative")

        pool = self._get_or_create_pool(layer_name, token_keys)
        storage_keys = self._move_to_storage(token_keys, pool.storage_device)
        storage_values = self._move_to_storage(token_values, pool.storage_device)
        storage_assignments = self._move_to_storage(assignments, pool.storage_device)
        storage_cluster_counts = self._move_to_storage(
            cluster_token_counts, pool.storage_device
        )

        num_kv_heads, _, head_size = token_keys.shape
        num_clusters = cluster_token_counts.shape[1]

        cluster_page_counts = torch.div(
            storage_cluster_counts.to(torch.int64) + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        total_pages = int(cluster_page_counts.sum().item())
        max_pages_per_cluster = (
            int(cluster_page_counts.max().item()) if cluster_page_counts.numel() else 0
        )

        allocated_storage_page_ids = pool.allocate(total_pages)
        allocated_metadata_page_ids = allocated_storage_page_ids.detach().to(
            device="cpu", dtype=torch.int64
        )

        page_ids = torch.full(
            (
                num_kv_heads,
                num_clusters,
                max_pages_per_cluster,
            ),
            -1,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        page_token_counts = torch.zeros(
            page_ids.shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        cursor = 0
        try:
            for head_index in range(num_kv_heads):
                for cluster_index in range(num_clusters):
                    token_count = int(
                        storage_cluster_counts[
                            head_index,
                            cluster_index,
                        ].item()
                    )
                    if token_count == 0:
                        continue

                    num_pages = (token_count + self.page_size - 1) // self.page_size
                    storage_page_ids = allocated_storage_page_ids[
                        cursor : cursor + num_pages
                    ]
                    metadata_page_ids = allocated_metadata_page_ids[
                        cursor : cursor + num_pages
                    ]
                    cursor += num_pages

                    member_indices = torch.nonzero(
                        storage_assignments[head_index] == cluster_index,
                        as_tuple=False,
                    ).flatten()

                    if member_indices.numel() != token_count:
                        raise RuntimeError(
                            "Cluster assignment count does not match "
                            "cluster_token_counts"
                        )

                    packed_keys = self._allocate_packed_pages(pool, num_pages)
                    packed_values = self._allocate_packed_pages(pool, num_pages)

                    packed_keys.view(-1, head_size)[:token_count].copy_(
                        storage_keys[head_index].index_select(0, member_indices)
                    )
                    packed_values.view(-1, head_size)[:token_count].copy_(
                        storage_values[head_index].index_select(0, member_indices)
                    )

                    pool.write(
                        storage_page_ids,
                        packed_keys,
                        packed_values,
                    )

                    page_ids[
                        head_index,
                        cluster_index,
                        :num_pages,
                    ] = metadata_page_ids

                    page_token_counts[
                        head_index,
                        cluster_index,
                        :num_pages,
                    ] = self.page_size
                    page_token_counts[
                        head_index,
                        cluster_index,
                        num_pages - 1,
                    ] = token_count - (num_pages - 1) * self.page_size
        except Exception:
            pool.free(allocated_storage_page_ids)
            raise

        if cursor != total_pages:
            pool.free(allocated_storage_page_ids)
            raise RuntimeError(
                "Cluster page construction did not consume all allocated pages"
            )

        try:
            self._resize_resident_cache(layer_name, pool)
        except Exception:
            pool.free(allocated_storage_page_ids)
            raise

        try:
            cluster_ids = self._allocate_cluster_ids(
                layer_name=layer_name,
                request_id=request_id,
                cluster_start=cluster_start,
                cluster_token_counts=cluster_token_counts,
                page_ids=page_ids,
                page_token_counts=page_token_counts,
            )
        except Exception:
            pool.free(allocated_storage_page_ids)
            self._resize_resident_cache(layer_name, pool)
            raise

        return RetroSpecClusterBlockTable(cluster_ids=cluster_ids)

    def free(
        self,
        layer_name: str,
        block_table: RetroSpecClusterBlockTable,
    ) -> None:
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

    def get_page_storage(
        self,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the physical backing-page tensors for one layer."""
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        return pool.key_pages, pool.value_pages

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

    @staticmethod
    def _stage_missing_pages(
        pool: _LayerClusterPagePool,
        logical_page_ids: torch.Tensor,
        miss_cluster_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copy non-resident selected pages into a temporary GPU page pool."""
        if logical_page_ids.ndim < 1:
            raise ValueError("Logical page IDs must have at least one dimension")
        if miss_cluster_mask.shape != logical_page_ids.shape[:-1]:
            raise ValueError("Miss-cluster mask does not match logical page IDs")
        if pool.metadata_device.type != "cuda":
            raise RuntimeError("Temporary cluster-page staging requires CUDA metadata")

        logical_page_ids_cpu = logical_page_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        miss_cluster_mask_cpu = miss_cluster_mask.detach().to(
            device="cpu",
            dtype=torch.bool,
        )

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

        missing_logical_page_ids = logical_page_ids_cpu[missing_page_mask_cpu].tolist()

        for staging_page_id, logical_page_id in enumerate(missing_logical_page_ids):
            staging_key_pages[staging_page_id].copy_(
                pool.key_pages[logical_page_id],
                non_blocking=pool.pin_memory,
            )
            staging_value_pages[staging_page_id].copy_(
                pool.value_pages[logical_page_id],
                non_blocking=pool.pin_memory,
            )

        return (
            staging_page_ids,
            staging_key_pages,
            staging_value_pages,
        )

    def resolve_cluster_blocks(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        mode: RetroSpecClusterResolveMode = "verification",
    ) -> RetroSpecResolvedClusterPages:
        """Resolve selected cluster blocks to physical GPU page sources."""
        if mode not in ("resident_only", "verification"):
            raise ValueError(f"Unsupported RetroSpec cluster resolve mode: {mode}")
        if logical_page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Logical page-table shape does not match cluster IDs")
        if logical_page_ids.device != cluster_ids.device:
            raise ValueError("Cluster IDs and logical pages must use one device")

        cluster_ids_cpu = self._validate_cluster_blocks(
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

        if not self.is_cpu_backed:
            staging_page_ids = torch.full_like(logical_page_ids, -1)
            hit_cluster_mask = cluster_ids >= 0

            return RetroSpecResolvedClusterPages(
                resident_page_ids=logical_page_ids,
                staging_page_ids=staging_page_ids,
                resident_key_pages=pool.key_pages,
                resident_value_pages=pool.value_pages,
                staging_key_pages=pool.key_pages[:0],
                staging_value_pages=pool.value_pages[:0],
                hit_cluster_mask=hit_cluster_mask,
                miss_cluster_mask=torch.zeros_like(hit_cluster_mask),
                resident_ready_event=None,
            )

        cluster_groups = self._get_cluster_groups(
            layer_name,
            cluster_ids_cpu,
        )
        pool, resident_cache = self._get_or_create_resident_cache(layer_name)

        resident_access = resident_cache.lookup(
            cluster_ids=cluster_ids,
            page_ids=logical_page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=pool.allocated_page_ids,
            touch=True,
        )

        if mode == "verification":
            (
                staging_page_ids,
                staging_key_pages,
                staging_value_pages,
            ) = self._stage_missing_pages(
                pool,
                logical_page_ids,
                resident_access.miss_cluster_mask,
            )
        else:
            staging_page_ids = torch.full_like(logical_page_ids, -1)
            staging_key_pages = resident_cache.key_pages[:0]
            staging_value_pages = resident_cache.value_pages[:0]

        return RetroSpecResolvedClusterPages(
            resident_page_ids=resident_access.cache_page_ids,
            staging_page_ids=staging_page_ids,
            resident_key_pages=resident_cache.key_pages,
            resident_value_pages=resident_cache.value_pages,
            staging_key_pages=staging_key_pages,
            staging_value_pages=staging_value_pages,
            hit_cluster_mask=resident_access.hit_cluster_mask,
            miss_cluster_mask=resident_access.miss_cluster_mask,
            resident_ready_event=resident_cache.pending_copy_event(),
        )

    def lookup_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        touch: bool = True,
    ) -> RetroSpecResidentPageAccess:
        cluster_ids_cpu = self._validate_cluster_blocks(
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
        )

    def admit_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        cluster_ids_cpu = self._validate_cluster_blocks(
            layer_name,
            cluster_ids,
            page_ids,
        )
        pool, resident_cache = self._get_or_create_resident_cache(layer_name)

        return resident_cache.admit(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=self._get_cluster_groups(
                layer_name,
                cluster_ids_cpu,
            ),
            allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
            allocated_page_ids=pool.allocated_page_ids,
            backing_key_pages=pool.key_pages,
            backing_value_pages=pool.value_pages,
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
        cluster_ids_cpu = self._validate_cluster_blocks(
            layer_name,
            cluster_ids,
            logical_page_ids,
        )
        pool, resident_cache = self._get_or_create_resident_cache(layer_name)

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
