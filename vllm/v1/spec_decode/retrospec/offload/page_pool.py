# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops

from .cluster_store_support import (
    RetroSpecCompactTokenRange,
    RetroSpecFullVerificationDescriptor,
    _CPUPageSlab,
    _FullVerificationSourceSnapshot,
)


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


# Keep the original qualified path for serialized classes.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = "vllm.v1.spec_decode.retrospec.cluster_store"
del _legacy_type
