# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable cluster-handle lifecycle and infrequent metadata queries."""

import torch

from .cluster_identity import RetroSpecClusterGroup, RetroSpecClusterIdentity
from .cluster_store_support import (
    _ClusterBlockDescriptor,
    _LayerPrefetchDescriptorArena,
)


class _ClusterStoreMetadataMethods:
    """Methods bound directly to RetroSpecClusterPageStore."""

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

    def num_allocated_clusters(
        self,
        layer_name: str,
    ) -> int:
        allocated = self._allocated_cluster_ids.get(layer_name)
        return 0 if allocated is None else len(allocated)
