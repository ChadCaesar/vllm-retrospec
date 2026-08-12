# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetroSpecClusterPageTable:
    """Physical pages occupied by a group of per-head clusters.

    page_ids and page_token_counts have shape:

        [num_kv_heads, num_clusters, max_pages_per_cluster]

    A page ID of -1 represents padding. page_token_counts records how many
    vectors in each page are valid.
    """

    page_ids: torch.Tensor
    page_token_counts: torch.Tensor


class _LayerClusterPagePool:
    """Growable private page pool for one attention layer."""

    _MIN_CAPACITY = 64

    def __init__(
        self,
        page_size: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
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

        # IDs are kept on the CPU because allocation and request release are
        # control-plane operations rather than attention hot-path operations.
        self._free_page_ids: list[int] = []
        self._allocated_page_ids: set[int] = set()

    @property
    def capacity(self) -> int:
        return self.key_pages.shape[0]

    @property
    def num_allocated_pages(self) -> int:
        return len(self._allocated_page_ids)

    def _grow(self, required_capacity: int) -> None:
        old_capacity = self.capacity
        new_capacity = max(self._MIN_CAPACITY, old_capacity)

        while new_capacity < required_capacity:
            new_capacity *= 2

        new_key_pages = torch.empty(
            new_capacity,
            self.page_size,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        new_value_pages = torch.empty_like(new_key_pages)

        if old_capacity:
            new_key_pages[:old_capacity].copy_(self.key_pages)
            new_value_pages[:old_capacity].copy_(self.value_pages)

        self.key_pages = new_key_pages
        self.value_pages = new_value_pages

        # Reverse insertion makes pop() return the lowest new page ID first.
        self._free_page_ids.extend(range(new_capacity - 1, old_capacity - 1, -1))

    def allocate(self, num_pages: int) -> torch.Tensor:
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages == 0:
            return torch.empty(
                0,
                dtype=torch.int64,
                device=self.device,
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
            device=self.device,
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
        if key_pages.dtype != self.dtype or value_pages.dtype != self.dtype:
            raise ValueError("Cluster page dtype does not match the layer page pool")
        if key_pages.device != self.device or value_pages.device != self.device:
            raise ValueError("Cluster pages must be on the page-pool device")

        self.key_pages.index_copy_(0, page_ids, key_pages)
        self.value_pages.index_copy_(0, page_ids, value_pages)

    def read(
        self,
        page_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if page_ids.device != self.device:
            raise ValueError("Cluster page IDs must be on the page-pool device")

        output_shape = (
            *page_ids.shape,
            self.page_size,
            self.head_size,
        )
        if page_ids.numel() == 0:
            empty = torch.empty(
                output_shape,
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty.clone()

        valid_page_ids = page_ids[page_ids >= 0]
        if valid_page_ids.numel() and valid_page_ids.max().item() >= self.capacity:
            raise RuntimeError("Cluster page table references a page outside the pool")

        # Invalid padded page IDs read page zero. The caller masks those
        # vectors using page_token_counts before attention.
        safe_page_ids = page_ids.clamp_min(0)
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
    """Private per-layer secondary KV store organized by token clusters."""

    def __init__(self, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        self.page_size = page_size
        self._layer_pools: dict[str, _LayerClusterPagePool] = {}

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
        pool = self._layer_pools.get(layer_name)

        if pool is None:
            pool = _LayerClusterPagePool(
                page_size=self.page_size,
                head_size=head_size,
                dtype=vectors.dtype,
                device=vectors.device,
            )
            self._layer_pools[layer_name] = pool
            return pool

        if pool.head_size != head_size:
            raise ValueError("Cluster vectors do not match the layer head size")
        if pool.dtype != vectors.dtype:
            raise ValueError("Cluster vectors do not match the layer KV dtype")
        if pool.device != vectors.device:
            raise ValueError("Cluster vectors do not match the layer device")

        return pool

    def store_clusters(
        self,
        layer_name: str,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
    ) -> RetroSpecClusterPageTable:
        """Reorder token KV into per-head, per-cluster physical pages."""
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

        pool = self._get_or_create_pool(layer_name, token_keys)

        num_kv_heads, _, head_size = token_keys.shape
        num_clusters = cluster_token_counts.shape[1]

        cluster_page_counts = torch.div(
            cluster_token_counts.to(torch.int64) + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        total_pages = int(cluster_page_counts.sum().item())
        max_pages_per_cluster = (
            int(cluster_page_counts.max().item()) if cluster_page_counts.numel() else 0
        )

        allocated_page_ids = pool.allocate(total_pages)
        page_ids = torch.full(
            (
                num_kv_heads,
                num_clusters,
                max_pages_per_cluster,
            ),
            -1,
            dtype=torch.int64,
            device=token_keys.device,
        )
        page_token_counts = torch.zeros(
            page_ids.shape,
            dtype=torch.int32,
            device=token_keys.device,
        )

        cursor = 0
        try:
            for head_index in range(num_kv_heads):
                for cluster_index in range(num_clusters):
                    token_count = int(
                        cluster_token_counts[
                            head_index,
                            cluster_index,
                        ].item()
                    )
                    if token_count == 0:
                        continue

                    num_pages = (token_count + self.page_size - 1) // self.page_size
                    cluster_page_ids = allocated_page_ids[cursor : cursor + num_pages]
                    cursor += num_pages

                    member_indices = torch.nonzero(
                        assignments[head_index] == cluster_index,
                        as_tuple=False,
                    ).flatten()

                    if member_indices.numel() != token_count:
                        raise RuntimeError(
                            "Cluster assignment count does not match "
                            "cluster_token_counts"
                        )

                    packed_keys = torch.zeros(
                        num_pages,
                        self.page_size,
                        head_size,
                        dtype=token_keys.dtype,
                        device=token_keys.device,
                    )
                    packed_values = torch.zeros_like(packed_keys)

                    packed_keys.view(-1, head_size)[:token_count].copy_(
                        token_keys[head_index].index_select(
                            0,
                            member_indices,
                        )
                    )
                    packed_values.view(-1, head_size)[:token_count].copy_(
                        token_values[head_index].index_select(
                            0,
                            member_indices,
                        )
                    )

                    pool.write(
                        cluster_page_ids,
                        packed_keys,
                        packed_values,
                    )

                    page_ids[
                        head_index,
                        cluster_index,
                        :num_pages,
                    ] = cluster_page_ids

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
            pool.free(allocated_page_ids)
            raise

        if cursor != total_pages:
            pool.free(allocated_page_ids)
            raise RuntimeError(
                "Cluster page construction did not consume all allocated pages"
            )

        return RetroSpecClusterPageTable(
            page_ids=page_ids,
            page_token_counts=page_token_counts,
        )

    def free(
        self,
        layer_name: str,
        page_table: RetroSpecClusterPageTable,
    ) -> None:
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        pool.free(page_table.page_ids)

    def gather_pages(
        self,
        layer_name: str,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather selected pages without compacting their internal padding."""
        if page_ids.shape != page_token_counts.shape:
            raise ValueError("page_ids and page_token_counts must have equal shapes")

        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        key_pages, value_pages = pool.read(page_ids)

        token_offsets = torch.arange(
            self.page_size,
            dtype=torch.int32,
            device=page_ids.device,
        )
        token_mask = token_offsets.view(
            *((1,) * page_token_counts.ndim),
            self.page_size,
        ) < page_token_counts.unsqueeze(-1)

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

    def num_allocated_pages(self, layer_name: str) -> int:
        pool = self._layer_pools.get(layer_name)
        return 0 if pool is None else pool.num_allocated_pages
