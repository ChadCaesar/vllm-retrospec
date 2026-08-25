# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import ceil
from threading import Lock, RLock
from time import perf_counter
from typing import Literal

import torch

from vllm import _custom_ops as ops

from .cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from .performance import RetroSpecPerformanceStats
from .resident_cache import (
    RetroSpecResidentClusterCache,
    RetroSpecResidentPageAccess,
)

RetroSpecClusterResolveMode = Literal[
    "resident_only",
    "verification",
]


@dataclass
class _PinnedStagingSlot:
    """Reusable pinned CPU buffers for one in-flight cluster build."""

    source_device: torch.device

    token_key_storage: torch.Tensor | None = None
    token_value_storage: torch.Tensor | None = None
    assignment_storage: torch.Tensor | None = None
    cluster_count_storage: torch.Tensor | None = None
    token_offset_storage: torch.Tensor | None = None

    in_use: bool = False

    @staticmethod
    def _reserve(
        storage: torch.Tensor | None,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required_numel = source.numel()
        if (
            storage is None
            or storage.dtype != source.dtype
            or storage.numel() < required_numel
        ):
            storage = torch.empty(
                required_numel,
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )

        view = storage[:required_numel].view(source.shape)
        return storage, view

    def reserve_token_kv(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.in_use:
            raise RuntimeError("Pinned staging slot must be acquired before use")

        self.token_key_storage, staged_token_keys = self._reserve(
            self.token_key_storage,
            token_keys,
        )
        self.token_value_storage, staged_token_values = self._reserve(
            self.token_value_storage,
            token_values,
        )
        return staged_token_keys, staged_token_values

    def reserve_cluster_metadata(
        self,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.in_use:
            raise RuntimeError("Pinned staging slot must be acquired before use")

        self.assignment_storage, staged_assignments = self._reserve(
            self.assignment_storage,
            assignments,
        )
        self.cluster_count_storage, staged_cluster_token_counts = self._reserve(
            self.cluster_count_storage,
            cluster_token_counts,
        )
        self.token_offset_storage, staged_token_offsets = self._reserve(
            self.token_offset_storage,
            token_offsets_in_cluster,
        )
        return (
            staged_assignments,
            staged_cluster_token_counts,
            staged_token_offsets,
        )


@dataclass
class _PinnedSelectionSlot:
    """Reusable pinned cluster-ID buffer for one asynchronous prefetch."""

    cluster_id_storage: torch.Tensor | None = None
    in_use: bool = False

    def reserve(self, source: torch.Tensor) -> torch.Tensor:
        if not self.in_use:
            raise RuntimeError("Pinned selection slot must be acquired before use")

        required_numel = source.numel()
        if (
            self.cluster_id_storage is None
            or self.cluster_id_storage.numel() < required_numel
        ):
            self.cluster_id_storage = torch.empty(
                required_numel,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )

        return self.cluster_id_storage[:required_numel].view(source.shape)


@dataclass(frozen=True)
class _StagedResidentPrefetch:
    """Cluster IDs staged for background descriptor parsing and admission."""

    layer_name: str
    cluster_ids_cpu: torch.Tensor
    metadata_ready_event: torch.cuda.Event
    reuse_ready_event: torch.cuda.Event
    slot: _PinnedSelectionSlot = field(repr=False, compare=False)


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
class RetroSpecStagedTokenKV:
    """Token KV staged before GPU clustering starts.

    For pinned CPU offload, token KV is copied on the per-device offload
    stream while segmented k-means runs on the model execution stream.
    """

    token_keys: torch.Tensor
    token_values: torch.Tensor

    source_device: torch.device
    ready_event: torch.cuda.Event | None
    staging_slot: _PinnedStagingSlot | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def wait(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()


@dataclass(frozen=True)
class RetroSpecStagedClusterInput:
    """Complete CPU inputs required to construct cluster pages.

    ready_event is recorded after both token KV and clustering metadata have
    been copied to CPU. Waiting for it therefore makes every tensor in this
    structure safe for CPU page construction.
    """

    token_keys: torch.Tensor
    token_values: torch.Tensor
    assignments: torch.Tensor
    cluster_token_counts: torch.Tensor
    token_offsets_in_cluster: torch.Tensor

    metadata_device: torch.device
    ready_event: torch.cuda.Event | None
    staging_slot: _PinnedStagingSlot | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def wait(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()


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

    hit_gate_ready_mask marks clusters whose request/head resident LRU has
    reached its soft page target. Cold groups remain protected from hit-based
    draft transitions until this mask becomes true.

    resident_ready_event is recorded on the resident-cache copy stream.
    staging_ready_event is recorded after a full-verification layer transfer.
    The execution stream must wait for the corresponding event before reading
    a page source.
    """

    resident_page_ids: torch.Tensor
    staging_page_ids: torch.Tensor

    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor

    staging_key_pages: torch.Tensor
    staging_value_pages: torch.Tensor

    hit_cluster_mask: torch.Tensor
    miss_cluster_mask: torch.Tensor
    hit_gate_ready_mask: torch.Tensor
    resident_ready_event: torch.cuda.Event | None
    staging_ready_event: torch.cuda.Event | None = None


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


class _FullVerificationTransferBuffer:
    """One reusable full-layer H2D page arena for a CUDA device."""

    _MIN_CAPACITY = 64

    def __init__(
        self,
        page_size: int,
        device: torch.device,
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if device.type != "cuda":
            raise ValueError("Full-verification transfer buffer requires CUDA")

        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.page_size = page_size
        self.device = device
        self.performance_stats = performance_stats

        self._dtype: torch.dtype | None = None
        self._head_size: int | None = None
        self._capacity = 0

        self.key_pages: torch.Tensor | None = None
        self.value_pages: torch.Tensor | None = None

        self._transfer_stream = torch.cuda.Stream(device=device)

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (max(value, 1) - 1).bit_length()

    @property
    def capacity(self) -> int:
        return self._capacity

    def _release_old_storage(self) -> None:
        if self.key_pages is None or self.value_pages is None:
            return

        # The transfer stream already waits for the execution stream before
        # this method is called. Recording the old tensors on the transfer
        # stream prevents the CUDA allocator from recycling them too early.
        self.key_pages.record_stream(self._transfer_stream)
        self.value_pages.record_stream(self._transfer_stream)

    def _ensure_capacity(
        self,
        required_pages: int,
        dtype: torch.dtype,
        head_size: int,
    ) -> None:
        if required_pages < 0:
            raise ValueError("required_pages must be non-negative")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        layout_changed = self._dtype != dtype or self._head_size != head_size
        if not layout_changed and required_pages <= self._capacity:
            return

        self._release_old_storage()

        self._dtype = dtype
        self._head_size = head_size
        self._capacity = max(
            self._MIN_CAPACITY,
            self._next_power_of_two(required_pages),
        )

        shape = (
            self._capacity,
            self.page_size,
            head_size,
        )
        self.key_pages = torch.empty(shape, dtype=dtype, device=self.device)
        self.value_pages = torch.empty_like(self.key_pages)

    def stage(
        self,
        pool: _LayerClusterPagePool,
        logical_page_ids: torch.Tensor,
        logical_page_ids_cpu: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.cuda.Event,
    ]:
        """Copy every valid logical page into the shared CUDA arena."""
        if pool.storage_device.type != "cpu":
            raise ValueError("Full-verification staging requires CPU backing pages")
        if pool.metadata_device != self.device:
            raise ValueError(
                "Full-verification metadata and transfer buffer devices differ"
            )
        if logical_page_ids.device != self.device:
            raise ValueError("Logical page IDs must be on the transfer device")
        if logical_page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Logical page IDs must use an integral dtype")
        if logical_page_ids_cpu.device.type != "cpu":
            raise ValueError("CPU logical page IDs must reside on CPU")
        if logical_page_ids_cpu.dtype != torch.int64:
            raise ValueError("CPU logical page IDs must use int64")
        if logical_page_ids_cpu.shape != logical_page_ids.shape:
            raise ValueError("CPU and GPU logical page layouts do not match")
        valid_page_mask_cpu = logical_page_ids_cpu >= 0
        source_page_ids = logical_page_ids_cpu[valid_page_mask_cpu].contiguous()
        num_pages = source_page_ids.numel()

        valid_page_mask = logical_page_ids >= 0
        staging_page_ids = torch.full_like(
            logical_page_ids,
            -1,
            dtype=torch.int64,
        )

        if num_pages:
            staging_page_ids.masked_scatter_(
                valid_page_mask,
                torch.arange(
                    num_pages,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )

        current_stream = torch.cuda.current_stream(self.device)

        # The previous layer may still be reading this arena. Waiting on the
        # execution stream makes it safe to overwrite for the next layer.
        self._transfer_stream.wait_stream(current_stream)
        ready_event = torch.cuda.Event()
        transfer_timer = (
            None
            if self.performance_stats is None
            else self.performance_stats.start_cuda_timer(
                "full_verify_h2d",
                self._transfer_stream,
            )
        )

        with torch.cuda.stream(self._transfer_stream):
            self._ensure_capacity(
                required_pages=num_pages,
                dtype=pool.dtype,
                head_size=pool.head_size,
            )

            assert self.key_pages is not None
            assert self.value_pages is not None

            staging_key_pages = self.key_pages[:num_pages]
            staging_value_pages = self.value_pages[:num_pages]

            if num_pages:
                destination_page_ids = torch.arange(
                    num_pages,
                    dtype=torch.int64,
                    device="cpu",
                )
                block_mapping = torch.stack(
                    (source_page_ids, destination_page_ids),
                    dim=1,
                ).contiguous()

                block_size_in_bytes = (
                    pool.key_pages.element_size() * pool.key_pages.stride(0)
                )
                ops.swap_blocks(
                    pool.key_pages,
                    staging_key_pages,
                    block_size_in_bytes,
                    block_mapping,
                )
                ops.swap_blocks(
                    pool.value_pages,
                    staging_value_pages,
                    block_size_in_bytes,
                    block_mapping,
                )

            if self.performance_stats is not None:
                transfer_bytes = (
                    num_pages
                    * self.page_size
                    * pool.head_size
                    * pool.key_pages.element_size()
                    * 2
                )
                self.performance_stats.add_counter(
                    "full_verify_h2d_pages",
                    num_pages,
                )
                self.performance_stats.add_counter(
                    "full_verify_h2d_bytes",
                    transfer_bytes,
                )
                self.performance_stats.stop_cuda_timer(
                    transfer_timer,
                    self._transfer_stream,
                )

            ready_event.record(self._transfer_stream)

        return (
            staging_page_ids,
            staging_key_pages,
            staging_value_pages,
            ready_event,
        )


class RetroSpecClusterPageStore:
    """CPU cluster-page backing store with a bounded GPU resident cache."""

    _RESIDENT_PREFETCH_RING_SIZE = 2

    def __init__(
        self,
        page_size: int,
        pin_memory: bool = False,
        cache_ratio: float = 0.0,
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if not 0.0 <= cache_ratio <= 1.0:
            raise ValueError("cache_ratio must be between zero and one")

        self.page_size = page_size
        self.pin_memory = pin_memory
        self.cache_ratio = cache_ratio
        self.performance_stats = performance_stats

        self._layer_pools: dict[str, _LayerClusterPagePool] = {}
        self._resident_caches: dict[str, RetroSpecResidentClusterCache] = {}

        self._next_cluster_ids: dict[str, int] = {}
        self._allocated_cluster_ids: dict[str, set[int]] = {}
        self._cluster_block_descriptors: dict[
            str, dict[int, _ClusterBlockDescriptor]
        ] = {}

        # layer_name -> group -> number of owned CPU backing pages
        self._group_backing_page_counts: dict[
            str, dict[RetroSpecClusterGroup, int]
        ] = {}

        # One serialized D2H stream per CUDA device. Different model layers
        # share the stream so their staged CPU buffers are completed in enqueue
        # order without synchronizing the model execution stream.
        self._offload_streams: dict[torch.device, torch.cuda.Stream] = {}

        # Pinned D2H staging slots are shared by all model layers on one CUDA
        # device. The segmented index bounds the number simultaneously in use.
        self._pinned_staging_slots: dict[
            torch.device,
            list[_PinnedStagingSlot],
        ] = {}
        self._pinned_staging_lock = Lock()

        # Full verification transfers one complete layer at a time. Layers on
        # the same CUDA device reuse one growable page arena.
        self._full_verification_buffers: dict[
            torch.device, _FullVerificationTransferBuffer
        ] = {}

        self._resident_prefetch_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._resident_prefetch_slots: dict[
            tuple[torch.device, str], list[_PinnedSelectionSlot]
        ] = {}
        self._resident_prefetch_futures: dict[str, deque[Future[None]]] = {}
        self._resident_prefetch_lock = Lock()
        self._resident_prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="retrospec-resident-prefetch",
        )
        self._resident_state_lock = RLock()
        self._closed = False

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
        resident_cache.resize(
            capacity,
            group_targets=group_targets,
        )

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
            )
            self._resident_caches[layer_name] = resident_cache

        capacity = self._resident_target_capacity(pool)
        resident_cache.resize(
            capacity,
            group_targets=self._resident_group_targets(
                layer_name,
                capacity,
            ),
        )
        return pool, resident_cache

    @staticmethod
    def _validate_token_kv_input(
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> None:
        if token_keys.ndim != 3:
            raise ValueError(
                "token_keys must have shape [num_kv_heads, num_tokens, head_size]"
            )
        if token_values.shape != token_keys.shape:
            raise ValueError("token_keys and token_values must have equal shapes")
        if token_values.dtype != token_keys.dtype:
            raise ValueError("token_keys and token_values must have equal dtypes")
        if token_values.device != token_keys.device:
            raise ValueError("Token keys and values must be on one device")

    @classmethod
    def _validate_cluster_input_metadata(
        cls,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> None:
        cls._validate_token_kv_input(token_keys, token_values)

        token_shape = token_keys.shape[:2]
        if assignments.shape != token_shape:
            raise ValueError("assignments must have shape [num_kv_heads, num_tokens]")
        if token_offsets_in_cluster.shape != token_shape:
            raise ValueError(
                "token_offsets_in_cluster must have shape [num_kv_heads, num_tokens]"
            )
        if cluster_token_counts.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if cluster_token_counts.shape[0] != token_keys.shape[0]:
            raise ValueError(
                "cluster_token_counts KV-head count does not match token KV"
            )
        metadata = (
            ("assignments", assignments),
            ("cluster_token_counts", cluster_token_counts),
            ("token_offsets_in_cluster", token_offsets_in_cluster),
        )
        for name, tensor in metadata:
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must use an integral dtype")
            if tensor.device != token_keys.device:
                raise ValueError(f"{name} and token KV must be on one device")

    @staticmethod
    def _validate_staged_cluster_metadata(
        staged_token_kv: RetroSpecStagedTokenKV,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> None:
        token_shape = staged_token_kv.token_keys.shape[:2]

        if assignments.shape != token_shape:
            raise ValueError("assignments must have shape [num_kv_heads, num_tokens]")
        if token_offsets_in_cluster.shape != token_shape:
            raise ValueError(
                "token_offsets_in_cluster must have shape [num_kv_heads, num_tokens]"
            )
        if cluster_token_counts.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if cluster_token_counts.shape[0] != token_shape[0]:
            raise ValueError(
                "cluster_token_counts KV-head count does not match token KV"
            )
        metadata = (
            ("Assignments", assignments),
            ("Cluster counts", cluster_token_counts),
            ("Cluster token offsets", token_offsets_in_cluster),
        )
        for name, tensor in metadata:
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must use an integral dtype")
            if tensor.device != staged_token_kv.source_device:
                raise ValueError(f"{name} must remain on the token KV source device")

    @staticmethod
    def _canonical_cuda_device(device: torch.device) -> torch.device:
        if device.type != "cuda":
            raise ValueError("RetroSpec staging requires a CUDA source device")

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        return torch.device("cuda", device_index)

    def _get_offload_stream(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream:
        canonical_device = self._canonical_cuda_device(device)
        stream = self._offload_streams.get(canonical_device)

        if stream is None:
            stream = torch.cuda.Stream(device=canonical_device)
            self._offload_streams[canonical_device] = stream

        return stream

    def _acquire_pinned_staging_slot(
        self,
        source_device: torch.device,
    ) -> _PinnedStagingSlot:
        canonical_device = self._canonical_cuda_device(source_device)

        with self._pinned_staging_lock:
            slots = self._pinned_staging_slots.setdefault(canonical_device, [])

            for slot in slots:
                if not slot.in_use:
                    slot.in_use = True
                    return slot

            slot = _PinnedStagingSlot(
                source_device=canonical_device,
                in_use=True,
            )
            slots.append(slot)
            return slot

    def _release_pinned_staging_slot(
        self,
        slot: _PinnedStagingSlot | None,
    ) -> None:
        if slot is None:
            return

        with self._pinned_staging_lock:
            if not slot.in_use:
                raise RuntimeError("Pinned staging slot has already been released")
            slot.in_use = False

    def discard_staged_token_kv(
        self,
        staged: RetroSpecStagedTokenKV,
    ) -> None:
        """Wait for an abandoned token-KV transfer and release its slot."""
        try:
            staged.wait()
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    def discard_staged_clusters(
        self,
        staged: RetroSpecStagedClusterInput,
    ) -> None:
        """Wait for abandoned cluster inputs and release their slot."""
        try:
            staged.wait()
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    def stage_token_kv(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> RetroSpecStagedTokenKV:
        """Start staging token KV before segmented clustering."""
        self._validate_token_kv_input(token_keys, token_values)
        source_device = token_keys.device

        if source_device.type == "cpu":
            return RetroSpecStagedTokenKV(
                token_keys=token_keys,
                token_values=token_values,
                source_device=source_device,
                ready_event=None,
            )

        if source_device.type != "cuda":
            raise ValueError("CPU-backed cluster staging requires CPU or CUDA inputs")

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "token_kv_d2h_bytes",
                token_keys.nbytes + token_values.nbytes,
            )

        if not self.pin_memory:
            return RetroSpecStagedTokenKV(
                token_keys=token_keys.to(device="cpu", non_blocking=False),
                token_values=token_values.to(device="cpu", non_blocking=False),
                source_device=source_device,
                ready_event=None,
            )

        staging_slot = self._acquire_pinned_staging_slot(source_device)
        offload_stream: torch.cuda.Stream | None = None

        try:
            staged_token_keys, staged_token_values = staging_slot.reserve_token_kv(
                token_keys,
                token_values,
            )

            offload_stream = self._get_offload_stream(source_device)
            current_stream = torch.cuda.current_stream(source_device)
            offload_stream.wait_stream(current_stream)

            ready_event = torch.cuda.Event()

            with torch.cuda.stream(offload_stream):
                staged_token_keys.copy_(token_keys, non_blocking=True)
                staged_token_values.copy_(token_values, non_blocking=True)
                ready_event.record(offload_stream)

            token_keys.record_stream(offload_stream)
            token_values.record_stream(offload_stream)
        except BaseException:
            if offload_stream is not None:
                offload_stream.synchronize()
            self._release_pinned_staging_slot(staging_slot)
            raise

        return RetroSpecStagedTokenKV(
            token_keys=staged_token_keys,
            token_values=staged_token_values,
            source_device=source_device,
            ready_event=ready_event,
            staging_slot=staging_slot,
        )

    def finish_stage_clusters(
        self,
        staged_token_kv: RetroSpecStagedTokenKV,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> RetroSpecStagedClusterInput:
        """Stage clustering metadata after GPU clustering finishes."""
        self._validate_staged_cluster_metadata(
            staged_token_kv,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )
        source_device = staged_token_kv.source_device

        if source_device.type == "cpu":
            return RetroSpecStagedClusterInput(
                token_keys=staged_token_kv.token_keys,
                token_values=staged_token_kv.token_values,
                assignments=assignments,
                cluster_token_counts=cluster_token_counts,
                token_offsets_in_cluster=token_offsets_in_cluster,
                metadata_device=source_device,
                ready_event=None,
            )

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "cluster_metadata_d2h_bytes",
                assignments.nbytes
                + cluster_token_counts.nbytes
                + token_offsets_in_cluster.nbytes,
            )

        if not self.pin_memory:
            return RetroSpecStagedClusterInput(
                token_keys=staged_token_kv.token_keys,
                token_values=staged_token_kv.token_values,
                assignments=assignments.to(device="cpu", non_blocking=False),
                cluster_token_counts=cluster_token_counts.to(
                    device="cpu",
                    non_blocking=False,
                ),
                token_offsets_in_cluster=token_offsets_in_cluster.to(
                    device="cpu",
                    non_blocking=False,
                ),
                metadata_device=source_device,
                ready_event=None,
            )

        staging_slot = staged_token_kv.staging_slot
        if staging_slot is None:
            raise RuntimeError("Pinned token KV does not own a staging slot")

        offload_stream = self._get_offload_stream(source_device)

        try:
            (
                staged_assignments,
                staged_cluster_token_counts,
                staged_token_offsets,
            ) = staging_slot.reserve_cluster_metadata(
                assignments,
                cluster_token_counts,
                token_offsets_in_cluster,
            )

            current_stream = torch.cuda.current_stream(source_device)

            # Token-KV copies were enqueued earlier on the same offload stream.
            # This wait delays only metadata D2H until clustering has completed.
            offload_stream.wait_stream(current_stream)

            ready_event = torch.cuda.Event()

            with torch.cuda.stream(offload_stream):
                staged_assignments.copy_(assignments, non_blocking=True)
                staged_cluster_token_counts.copy_(
                    cluster_token_counts,
                    non_blocking=True,
                )
                staged_token_offsets.copy_(
                    token_offsets_in_cluster,
                    non_blocking=True,
                )
                ready_event.record(offload_stream)

            assignments.record_stream(offload_stream)
            cluster_token_counts.record_stream(offload_stream)
            token_offsets_in_cluster.record_stream(offload_stream)
        except BaseException:
            # Ownership remains with staged_token_kv until this method returns.
            # Synchronize partially queued metadata copies before it is discarded.
            offload_stream.synchronize()
            raise

        return RetroSpecStagedClusterInput(
            token_keys=staged_token_kv.token_keys,
            token_values=staged_token_kv.token_values,
            assignments=staged_assignments,
            cluster_token_counts=staged_cluster_token_counts,
            token_offsets_in_cluster=staged_token_offsets,
            metadata_device=source_device,
            ready_event=ready_event,
            staging_slot=staging_slot,
        )

    def stage_clusters(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> RetroSpecStagedClusterInput:
        """Stage complete inputs when overlap is not controlled by the caller."""
        self._validate_cluster_input_metadata(
            token_keys,
            token_values,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )
        staged_token_kv = self.stage_token_kv(token_keys, token_values)

        try:
            return self.finish_stage_clusters(
                staged_token_kv,
                assignments,
                cluster_token_counts,
                token_offsets_in_cluster,
            )
        except BaseException:
            self.discard_staged_token_kv(staged_token_kv)
            raise

    def store_staged_clusters(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        staged: RetroSpecStagedClusterInput,
    ) -> RetroSpecClusterBlockTable:
        """Wait for staged D2H copies and construct CPU cluster pages."""
        wait_started_at = perf_counter()
        try:
            staged.wait()
            if self.performance_stats is not None:
                self.performance_stats.record_cpu_time(
                    "cluster_build_wait",
                    perf_counter() - wait_started_at,
                )

            build_started_at = perf_counter()
            result = self.store_clusters(
                layer_name=layer_name,
                request_id=request_id,
                cluster_start=cluster_start,
                token_keys=staged.token_keys,
                token_values=staged.token_values,
                assignments=staged.assignments,
                cluster_token_counts=staged.cluster_token_counts,
                token_offsets_in_cluster=staged.token_offsets_in_cluster,
                metadata_device=staged.metadata_device,
            )
            if self.performance_stats is not None:
                self.performance_stats.record_cpu_time(
                    "cluster_page_build",
                    perf_counter() - build_started_at,
                )
                self.performance_stats.add_counter("cluster_builds")
            return result
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    @staticmethod
    def _move_to_storage(
        tensor: torch.Tensor,
        storage_device: torch.device,
    ) -> torch.Tensor:
        if tensor.device == storage_device:
            return tensor

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

    @staticmethod
    def _validate_cluster_assignment_counts(
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assignments_int64 = assignments.to(torch.int64)
        cluster_token_counts_int64 = cluster_token_counts.to(torch.int64)
        token_offsets_int64 = token_offsets_in_cluster.to(torch.int64)
        num_clusters = cluster_token_counts.shape[1]

        if assignments_int64.numel():
            invalid_assignments = (assignments_int64 < 0) | (
                assignments_int64 >= num_clusters
            )
            if torch.any(invalid_assignments).item():
                raise RuntimeError(
                    "Cluster assignment count does not match cluster_token_counts"
                )

        actual_cluster_counts = torch.zeros_like(cluster_token_counts_int64)
        actual_cluster_counts.scatter_add_(
            dim=1,
            index=assignments_int64,
            src=torch.ones_like(assignments_int64),
        )

        if not torch.equal(actual_cluster_counts, cluster_token_counts_int64):
            raise RuntimeError(
                "Cluster assignment count does not match cluster_token_counts"
            )

        assigned_cluster_counts = torch.gather(
            cluster_token_counts_int64,
            dim=1,
            index=assignments_int64,
        )
        invalid_offsets = (token_offsets_int64 < 0) | (
            token_offsets_int64 >= assigned_cluster_counts
        )
        if torch.any(invalid_offsets).item():
            raise RuntimeError("Cluster token offsets exceed cluster boundaries")

        cluster_starts = (
            torch.cumsum(cluster_token_counts_int64, dim=1) - cluster_token_counts_int64
        )
        compact_positions = (
            torch.gather(cluster_starts, dim=1, index=assignments_int64)
            + token_offsets_int64
        )
        occupied_positions = torch.zeros_like(assignments_int64)
        occupied_positions.scatter_add_(
            dim=1,
            index=compact_positions,
            src=torch.ones_like(compact_positions),
        )
        if not torch.all(occupied_positions == 1).item():
            raise RuntimeError("Cluster token offsets must be unique within clusters")

        return assignments_int64, cluster_token_counts_int64, token_offsets_int64

    def _pack_cluster_pages(
        self,
        pool: _LayerClusterPagePool,
        storage_keys: torch.Tensor,
        storage_values: torch.Tensor,
        storage_assignments: torch.Tensor,
        storage_token_offsets: torch.Tensor,
        cluster_page_counts: torch.Tensor,
        total_pages: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_kv_heads, num_tokens, head_size = storage_keys.shape

        packed_keys = self._allocate_packed_pages(pool, total_pages)
        packed_values = self._allocate_packed_pages(pool, total_pages)

        if num_tokens == 0:
            return packed_keys, packed_values

        flat_cluster_page_counts = cluster_page_counts.reshape(-1)
        flat_cluster_page_offsets = (
            torch.cumsum(
                flat_cluster_page_counts,
                dim=0,
            )
            - flat_cluster_page_counts
        )
        cluster_page_offsets = flat_cluster_page_offsets.view_as(cluster_page_counts)

        token_page_offsets = torch.gather(
            cluster_page_offsets,
            dim=1,
            index=storage_assignments,
        )
        packed_token_positions = (
            token_page_offsets * self.page_size + storage_token_offsets
        ).reshape(-1)

        packed_keys.view(-1, head_size).index_copy_(
            0,
            packed_token_positions,
            storage_keys.reshape(num_kv_heads * num_tokens, head_size),
        )
        packed_values.view(-1, head_size).index_copy_(
            0,
            packed_token_positions,
            storage_values.reshape(num_kv_heads * num_tokens, head_size),
        )

        return packed_keys, packed_values

    def _build_cluster_page_metadata(
        self,
        cluster_token_counts: torch.Tensor,
        cluster_page_counts: torch.Tensor,
        allocated_metadata_page_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cluster_token_counts_cpu = cluster_token_counts.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        cluster_page_counts_cpu = cluster_page_counts.detach().to(
            device="cpu",
            dtype=torch.int64,
        )

        num_kv_heads, num_clusters = cluster_token_counts_cpu.shape
        max_pages_per_cluster = (
            int(cluster_page_counts_cpu.max().item())
            if cluster_page_counts_cpu.numel()
            else 0
        )

        metadata_shape = (
            num_kv_heads,
            num_clusters,
            max_pages_per_cluster,
        )
        page_ids = torch.full(
            metadata_shape,
            -1,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        page_token_counts = torch.zeros(
            metadata_shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        if max_pages_per_cluster == 0:
            return page_ids, page_token_counts

        page_offsets = torch.arange(
            max_pages_per_cluster,
            dtype=torch.int64,
            device="cpu",
        ).view(1, 1, max_pages_per_cluster)
        valid_page_mask = page_offsets < cluster_page_counts_cpu.unsqueeze(-1)
        expected_page_count = int(valid_page_mask.sum().item())

        if expected_page_count != allocated_metadata_page_ids.numel():
            raise RuntimeError(
                "Cluster page metadata does not cover all allocated pages"
            )

        # Row-major (head, cluster, page) order matches the flattened page
        # offsets used for packing and the order returned by pool.allocate().
        page_ids.masked_scatter_(
            valid_page_mask,
            allocated_metadata_page_ids,
        )
        page_token_counts.masked_fill_(
            valid_page_mask,
            self.page_size,
        )

        valid_cluster_mask = cluster_token_counts_cpu > 0
        last_page_mask = valid_page_mask & (
            page_offsets == cluster_page_counts_cpu.unsqueeze(-1) - 1
        )
        last_page_token_counts = (
            torch.remainder(
                cluster_token_counts_cpu[valid_cluster_mask] - 1,
                self.page_size,
            )
            + 1
        ).to(torch.int32)
        page_token_counts.masked_scatter_(
            last_page_mask,
            last_page_token_counts,
        )

        return page_ids, page_token_counts

    def store_clusters(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
        metadata_device: torch.device | None = None,
    ) -> RetroSpecClusterBlockTable:
        """Pack token KV into per-head, per-cluster backing pages."""
        if cluster_start < 0:
            raise ValueError("cluster_start must be non-negative")

        self._validate_cluster_input_metadata(
            token_keys,
            token_values,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )

        if torch.any(cluster_token_counts < 0).item():
            raise ValueError("cluster_token_counts must be non-negative")

        pool = self._get_or_create_pool(
            layer_name,
            token_keys,
            metadata_device=metadata_device,
        )
        storage_keys = self._move_to_storage(token_keys, pool.storage_device)
        storage_values = self._move_to_storage(token_values, pool.storage_device)
        storage_assignments = self._move_to_storage(
            assignments,
            pool.storage_device,
        )
        storage_cluster_counts = self._move_to_storage(
            cluster_token_counts, pool.storage_device
        )
        storage_token_offsets = self._move_to_storage(
            token_offsets_in_cluster,
            pool.storage_device,
        )

        (
            storage_assignments,
            storage_cluster_counts,
            storage_token_offsets,
        ) = self._validate_cluster_assignment_counts(
            storage_assignments,
            storage_cluster_counts,
            storage_token_offsets,
        )

        cluster_page_counts = torch.div(
            storage_cluster_counts + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        total_pages = int(cluster_page_counts.sum().item())
        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "cluster_pages_built",
                total_pages,
            )

        allocated_storage_page_ids = pool.allocate(total_pages)
        allocated_metadata_page_ids = allocated_storage_page_ids.detach().to(
            device="cpu", dtype=torch.int64
        )

        try:
            packed_keys, packed_values = self._pack_cluster_pages(
                pool=pool,
                storage_keys=storage_keys,
                storage_values=storage_values,
                storage_assignments=storage_assignments,
                storage_token_offsets=storage_token_offsets,
                cluster_page_counts=cluster_page_counts,
                total_pages=total_pages,
            )
            page_ids, page_token_counts = self._build_cluster_page_metadata(
                cluster_token_counts=storage_cluster_counts,
                cluster_page_counts=cluster_page_counts,
                allocated_metadata_page_ids=allocated_metadata_page_ids,
            )
            pool.write(
                allocated_storage_page_ids,
                packed_keys,
                packed_values,
            )
        except Exception:
            pool.free(allocated_storage_page_ids)
            raise

        cluster_ids: torch.Tensor | None = None
        with self._resident_state_lock:
            try:
                cluster_ids = self._allocate_cluster_ids(
                    layer_name=layer_name,
                    request_id=request_id,
                    cluster_start=cluster_start,
                    cluster_token_counts=cluster_token_counts,
                    page_ids=page_ids,
                    page_token_counts=page_token_counts,
                )
                self._resize_resident_cache(layer_name, pool)
            except Exception:
                if cluster_ids is not None:
                    self._free_cluster_ids(layer_name, cluster_ids)
                pool.free(allocated_storage_page_ids)
                raise

        assert cluster_ids is not None
        return RetroSpecClusterBlockTable(cluster_ids=cluster_ids)

    def free(
        self,
        layer_name: str,
        block_table: RetroSpecClusterBlockTable,
    ) -> None:
        self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
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

    def _get_full_verification_buffer(
        self,
        pool: _LayerClusterPagePool,
    ) -> _FullVerificationTransferBuffer:
        device = pool.metadata_device

        if device.type != "cuda":
            raise RuntimeError("Full-verification transfer requires CUDA metadata")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        buffer = self._full_verification_buffers.get(device)
        if buffer is None:
            buffer = _FullVerificationTransferBuffer(
                page_size=self.page_size,
                device=device,
                performance_stats=self.performance_stats,
            )
            self._full_verification_buffers[device] = buffer

        return buffer

    @staticmethod
    def _stage_missing_pages(
        pool: _LayerClusterPagePool,
        logical_page_ids: torch.Tensor,
        miss_cluster_mask: torch.Tensor,
        logical_page_ids_cpu: torch.Tensor,
        miss_cluster_mask_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copy non-resident selected pages into a temporary GPU page pool."""
        if logical_page_ids.ndim < 1:
            raise ValueError("Logical page IDs must have at least one dimension")
        if miss_cluster_mask.shape != logical_page_ids.shape[:-1]:
            raise ValueError("Miss-cluster mask does not match logical page IDs")
        if pool.metadata_device.type != "cuda":
            raise RuntimeError("Temporary cluster-page staging requires CUDA metadata")

        if logical_page_ids_cpu.device.type != "cpu":
            raise ValueError("CPU logical page IDs must reside on CPU")
        if miss_cluster_mask_cpu.device.type != "cpu":
            raise ValueError("CPU miss mask must reside on CPU")
        if logical_page_ids_cpu.shape != logical_page_ids.shape:
            raise ValueError("CPU logical page IDs do not match the GPU layout")
        if miss_cluster_mask_cpu.shape != miss_cluster_mask.shape:
            raise ValueError("CPU miss mask does not match the GPU layout")

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

    def _get_resident_prefetch_stream(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream:
        device = self._canonical_cuda_device(device)
        stream = self._resident_prefetch_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._resident_prefetch_streams[device] = stream
        return stream

    def _acquire_resident_prefetch_slot(
        self,
        layer_name: str,
        device: torch.device,
    ) -> _PinnedSelectionSlot | None:
        device = self._canonical_cuda_device(device)
        key = (device, layer_name)

        with self._resident_prefetch_lock:
            slots = self._resident_prefetch_slots.setdefault(
                key,
                [
                    _PinnedSelectionSlot()
                    for _ in range(self._RESIDENT_PREFETCH_RING_SIZE)
                ],
            )
            for slot in slots:
                if not slot.in_use:
                    slot.in_use = True
                    return slot

        return None

    def _release_resident_prefetch_slot(
        self,
        slot: _PinnedSelectionSlot,
    ) -> None:
        with self._resident_prefetch_lock:
            if not slot.in_use:
                raise RuntimeError("Resident prefetch slot was already released")
            slot.in_use = False

    @torch.inference_mode()
    def _finish_resident_prefetch(
        self,
        staged: _StagedResidentPrefetch,
    ) -> None:
        try:
            staged.metadata_ready_event.synchronize()

            if self.performance_stats is not None:
                num_clusters = int((staged.cluster_ids_cpu >= 0).sum().item())
                self.performance_stats.add_counter(
                    "prefetch_candidate_clusters",
                    num_clusters,
                )

            with self._resident_state_lock:
                metadata = self._materialize_cluster_block_metadata_cpu(
                    staged.layer_name,
                    staged.cluster_ids_cpu,
                )
                pool, resident_cache = self._get_or_create_resident_cache(
                    staged.layer_name
                )
                cluster_groups = self._get_cluster_groups(
                    staged.layer_name,
                    staged.cluster_ids_cpu,
                )

                with torch.cuda.device(pool.metadata_device):
                    resident_cache.admit(
                        cluster_ids=staged.cluster_ids_cpu,
                        page_ids=metadata.page_ids,
                        cluster_groups=cluster_groups,
                        allocated_cluster_ids=self._get_allocated_cluster_ids(
                            staged.layer_name
                        ),
                        allocated_page_ids=pool.allocated_page_ids,
                        backing_key_pages=pool.key_pages,
                        backing_value_pages=pool.value_pages,
                        cluster_ids_cpu=staged.cluster_ids_cpu,
                        page_ids_cpu=metadata.page_ids,
                        reuse_ready_event=staged.reuse_ready_event,
                    )
        finally:
            self._release_resident_prefetch_slot(staged.slot)

    def _reap_resident_prefetches(
        self,
        layer_names: Sequence[str] | None = None,
        wait: bool = False,
    ) -> None:
        if layer_names is None:
            layer_names = tuple(self._resident_prefetch_futures)

        ready: list[Future[None]] = []
        with self._resident_prefetch_lock:
            for layer_name in layer_names:
                futures = self._resident_prefetch_futures.get(layer_name)
                if futures is None:
                    continue

                retained: deque[Future[None]] = deque()
                while futures:
                    future = futures.popleft()
                    if wait or future.done():
                        ready.append(future)
                    else:
                        retained.append(future)

                if retained:
                    self._resident_prefetch_futures[layer_name] = retained
                else:
                    self._resident_prefetch_futures.pop(layer_name, None)

        for future in ready:
            future.result()

    def prefetch_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
    ) -> bool:
        """Queue descriptor parsing and resident admission without host waits."""
        if self._closed:
            raise RuntimeError("RetroSpec cluster page store is closed")
        self._reap_resident_prefetches((layer_name,), wait=False)

        if not self.pin_memory or cluster_ids.numel() == 0:
            return False
        if cluster_ids.device.type != "cuda":
            raise ValueError("Asynchronous resident prefetch requires CUDA cluster IDs")
        if cluster_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cluster IDs must use an integral dtype")

        slot = self._acquire_resident_prefetch_slot(
            layer_name,
            cluster_ids.device,
        )
        if slot is None:
            if self.performance_stats is not None:
                self.performance_stats.add_counter("prefetch_dropped")
            return False

        stream = self._get_resident_prefetch_stream(cluster_ids.device)
        current_stream = torch.cuda.current_stream(cluster_ids.device)

        try:
            cluster_ids_cpu = slot.reserve(cluster_ids)
            stream.wait_stream(current_stream)

            with torch.cuda.stream(stream):
                cluster_ids_cpu.copy_(cluster_ids, non_blocking=True)
                metadata_ready_event = torch.cuda.Event()
                metadata_ready_event.record(stream)

            reuse_ready_event = torch.cuda.Event()
            reuse_ready_event.record(current_stream)
            cluster_ids.record_stream(stream)

            staged = _StagedResidentPrefetch(
                layer_name=layer_name,
                cluster_ids_cpu=cluster_ids_cpu,
                metadata_ready_event=metadata_ready_event,
                reuse_ready_event=reuse_ready_event,
                slot=slot,
            )
            future = self._resident_prefetch_executor.submit(
                self._finish_resident_prefetch,
                staged,
            )

            with self._resident_prefetch_lock:
                self._resident_prefetch_futures.setdefault(
                    layer_name,
                    deque(),
                ).append(future)

            if self.performance_stats is not None:
                self.performance_stats.add_counter("prefetch_submitted")
            return True
        except BaseException:
            stream.synchronize()
            self._release_resident_prefetch_slot(slot)
            raise

    def wait_for_resident_prefetches(
        self,
        layer_names: Sequence[str] | None = None,
    ) -> None:
        self._reap_resident_prefetches(layer_names, wait=True)

    def synchronize_resident_prefetches(
        self,
        layer_names: Sequence[str],
    ) -> None:
        """Wait for background admission and resident H2D copies to finish."""
        layer_names = tuple(dict.fromkeys(layer_names))
        self.wait_for_resident_prefetches(layer_names)

        with self._resident_state_lock:
            resident_caches = tuple(
                resident_cache
                for layer_name in layer_names
                if (resident_cache := self._resident_caches.get(layer_name)) is not None
            )

        for resident_cache in resident_caches:
            resident_cache.synchronize_pending_copies()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            self.wait_for_resident_prefetches()
        finally:
            self._resident_prefetch_executor.shutdown(wait=True)

    def resolve_cluster_blocks(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        mode: RetroSpecClusterResolveMode = "verification",
    ) -> RetroSpecResolvedClusterPages:
        if mode == "verification":
            self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
            return self._resolve_cluster_blocks_locked(
                layer_name,
                cluster_ids,
                logical_page_ids,
                mode,
            )

    def _resolve_cluster_blocks_locked(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        mode: RetroSpecClusterResolveMode = "verification",
    ) -> RetroSpecResolvedClusterPages:
        """Resolve selected cluster blocks to physical GPU page sources.

        resident_only exposes only completed resident pages. Verification also
        exposes pages whose asynchronous H2D copies are pending and returns the
        event that the execution stream must wait for.
        """
        if mode not in ("resident_only", "verification"):
            raise ValueError(f"Unsupported RetroSpec cluster resolve mode: {mode}")
        if logical_page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Logical page-table shape does not match cluster IDs")
        if logical_page_ids.device != cluster_ids.device:
            raise ValueError("Cluster IDs and logical pages must use one device")

        cluster_ids_cpu, logical_page_ids_cpu = self._validate_cluster_blocks(
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

        cluster_groups = self._get_cluster_groups(
            layer_name,
            cluster_ids_cpu,
        )
        pool, resident_cache = self._get_or_create_resident_cache(layer_name)

        verification = mode == "verification"
        resident_access = resident_cache.lookup(
            cluster_ids=cluster_ids,
            page_ids=logical_page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=pool.allocated_page_ids,
            touch=True,
            include_pending=verification,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=logical_page_ids_cpu,
        )

        if self.performance_stats is not None:
            valid_clusters_cpu = cluster_ids_cpu >= 0
            miss_clusters_cpu = (
                valid_clusters_cpu & resident_access.miss_cluster_mask_cpu
            )
            num_misses = int(miss_clusters_cpu.sum().item())
            num_valid = int(valid_clusters_cpu.sum().item())
            self.performance_stats.add_counter(
                "resident_cluster_hits",
                num_valid - num_misses,
            )
            self.performance_stats.add_counter(
                "resident_cluster_misses",
                num_misses,
            )

        if verification:
            (
                staging_page_ids,
                staging_key_pages,
                staging_value_pages,
            ) = self._stage_missing_pages(
                pool,
                logical_page_ids,
                resident_access.miss_cluster_mask,
                resident_access.logical_page_ids_cpu,
                resident_access.miss_cluster_mask_cpu,
            )
            resident_ready_event = resident_access.ready_event
        else:
            staging_page_ids = torch.full_like(logical_page_ids, -1)
            staging_key_pages = resident_cache.key_pages[:0]
            staging_value_pages = resident_cache.value_pages[:0]
            resident_ready_event = None

        return RetroSpecResolvedClusterPages(
            resident_page_ids=resident_access.cache_page_ids,
            staging_page_ids=staging_page_ids,
            resident_key_pages=resident_cache.key_pages,
            resident_value_pages=resident_cache.value_pages,
            staging_key_pages=staging_key_pages,
            staging_value_pages=staging_value_pages,
            hit_cluster_mask=resident_access.hit_cluster_mask,
            miss_cluster_mask=resident_access.miss_cluster_mask,
            hit_gate_ready_mask=resident_access.hit_gate_ready_mask,
            resident_ready_event=resident_ready_event,
        )

    def resolve_full_verification_blocks(
        self,
        layer_name: str,
        logical_page_ids: torch.Tensor,
        logical_page_ids_cpu: torch.Tensor,
    ) -> RetroSpecResolvedClusterPages:
        """Stage every clustered KV page required by full verification."""
        self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
            pool = self._layer_pools.get(layer_name)
            if pool is None:
                raise RuntimeError(
                    f"No RetroSpec page pool exists for layer {layer_name!r}"
                )

            transfer_buffer = self._get_full_verification_buffer(pool)
            (
                staging_page_ids,
                staging_key_pages,
                staging_value_pages,
                staging_ready_event,
            ) = transfer_buffer.stage(
                pool=pool,
                logical_page_ids=logical_page_ids,
                logical_page_ids_cpu=logical_page_ids_cpu,
            )

        resident_page_ids = torch.full_like(logical_page_ids, -1)
        valid_cluster_mask = (logical_page_ids >= 0).any(dim=-1)

        return RetroSpecResolvedClusterPages(
            resident_page_ids=resident_page_ids,
            staging_page_ids=staging_page_ids,
            resident_key_pages=staging_key_pages[:0],
            resident_value_pages=staging_value_pages[:0],
            staging_key_pages=staging_key_pages,
            staging_value_pages=staging_value_pages,
            hit_cluster_mask=torch.zeros_like(valid_cluster_mask),
            miss_cluster_mask=valid_cluster_mask,
            hit_gate_ready_mask=torch.zeros_like(valid_cluster_mask),
            resident_ready_event=None,
            staging_ready_event=staging_ready_event,
        )

    def lookup_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        touch: bool = True,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
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
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
            )

    def admit_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
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
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
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
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
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
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
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
