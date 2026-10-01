# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from functools import partial
from math import ceil

import torch

from vllm.v1.spec_decode.retrospec.legacy.cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecResidentTableBinding,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_cache import (
    RetroSpecResidentClusterCache,
)
from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    RetroSpecClusterBlockMetadata,
    _ClusterBlockDescriptor,
    _LayerClusterPagePool,
    _LayerPrefetchDescriptorArena,
)


class _RetroSpecClusterPageStoreIdentityMixin:
    def _publish_resident_table_bindings(
        self,
        layer_name: str,
        cluster_ids: tuple[int, ...],
        table_buckets: tuple[int, ...],
        stream: torch.cuda.Stream,
    ) -> None:
        if self._gpu_index_residency is None:
            return
        if len(cluster_ids) != len(table_buckets):
            raise ValueError("Resident handle and bucket counts must match")

        with self._resident_state_lock:
            descriptors = self._cluster_block_descriptors.get(layer_name, {})
            bindings: list[RetroSpecResidentTableBinding] = []
            for cluster_id, table_bucket in zip(
                cluster_ids, table_buckets, strict=True
            ):
                descriptor = descriptors.get(cluster_id)
                if descriptor is None:
                    continue
                identity = descriptor.identity
                bindings.append(
                    RetroSpecResidentTableBinding(
                        request_id=identity.group.request_id,
                        kv_head_index=identity.group.kv_head_index,
                        local_cluster_index=identity.local_cluster_id,
                        cluster_handle=cluster_id,
                        table_bucket=table_bucket,
                    )
                )

        self._gpu_index_residency.publish_resident_table_bindings(
            layer_name=layer_name,
            bindings=bindings,
            stream=stream,
        )

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

        descriptor_arena = self._prefetch_descriptor_arenas.setdefault(
            layer_name, _LayerPrefetchDescriptorArena()
        )
        if new_descriptors:
            required_pages = max(
                len(descriptor.page_ids) for descriptor in new_descriptors.values()
            )
            descriptor_arena.reserve(cluster_id_end, required_pages)
        resident_cache = self._resident_caches.get(layer_name)
        if resident_cache is not None:
            resident_cache.reserve_prefetch_handle_states(descriptor_arena.capacity)
        descriptor_arena.publish(new_descriptors)

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
        descriptor_arena = self._prefetch_descriptor_arenas.get(layer_name)
        if descriptor_arena is not None:
            descriptor_arena.invalidate(released)

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
        if resident_cache.requires_resize(capacity, group_targets):
            with resident_cache.mutation_guard():
                resident_cache.resize(capacity, group_targets=group_targets)

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
                performance_stats=self.performance_stats,
                binding_publisher=partial(
                    self._publish_resident_table_bindings,
                    layer_name,
                ),
            )
            self._resident_caches[layer_name] = resident_cache

        capacity = self._resident_target_capacity(pool)
        group_targets = self._resident_group_targets(layer_name, capacity)
        if resident_cache.requires_resize(capacity, group_targets):
            with resident_cache.mutation_guard():
                resident_cache.resize(capacity, group_targets=group_targets)
        descriptor_arena = self._prefetch_descriptor_arenas.get(layer_name)
        if descriptor_arena is not None:
            resident_cache.reserve_prefetch_handle_states(descriptor_arena.capacity)
        return pool, resident_cache

    def republish_resident_table_bindings(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        cluster_ids_cpu = cluster_ids.detach().to(device="cpu", dtype=torch.int64)
        valid_cluster_ids = tuple(
            int(cluster_id)
            for cluster_id in cluster_ids_cpu.reshape(-1).tolist()
            if cluster_id >= 0
        )
        if not valid_cluster_ids:
            return

        with self._resident_state_lock:
            resident_cache = self._resident_caches.get(layer_name)
            if resident_cache is None:
                return
            with resident_cache.mutation_guard():
                resident_cache.republish_handle_bindings(valid_cluster_ids, stream)

    def _get_resident_cache_for_lookup(
        self,
        layer_name: str,
    ) -> RetroSpecResidentClusterCache:
        """Return a stable per-layer resident cache for lookup."""
        resident_cache = self._resident_caches.get(layer_name)
        if resident_cache is not None:
            return resident_cache

        # Cache objects are published once and are not removed or replaced.
        # Only the cold creation path requires the global lifecycle lock.
        with self._resident_state_lock:
            _, resident_cache = self._get_or_create_resident_cache(layer_name)
        return resident_cache
