# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from functools import partial
from math import ceil
from threading import Lock, RLock
from time import perf_counter

import torch

from ..performance import RetroSpecPerformanceStats
from .cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from .cluster_prefetch import RetroSpecClusterPrefetchMixin
from .cluster_store_support import (
    RetroSpecClusterBlockMetadata,
    RetroSpecClusterBlockTable,
    RetroSpecClusterResolveMode,
    RetroSpecCompactResolvedClusterPages,
    RetroSpecCompactTokenRange,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationStaging,
    RetroSpecFullVerificationTicket,
    RetroSpecRankedDraftResolvedClusters,
    RetroSpecResidentPrefetchInput,
    RetroSpecResidentPrefetchSource,
    RetroSpecResolvedClusterPages,
    RetroSpecStagedClusterInput,
    RetroSpecStagedTokenKV,
    RetroSpecVerificationMissAdmission,
    RetroSpecVerificationResolveRequest,
    _ClusterBlockDescriptor,
    _DeferredResidentPrefetchWave,
    _LayerPrefetchDescriptorArena,
    _PinnedSelectionSlot,
    _PinnedStagingSlot,
    _PinnedVerificationMissSlot,
    _ResidentPrefetchPriorityExecutor,
    _ResidentPrefetchWaveFuture,
    _VerificationResolveGPUArena,
)
from .cluster_verification import RetroSpecClusterVerificationMixin
from .index_residency import (
    RetroSpecGPUIndexResidencyManager,
    RetroSpecResidentTableBinding,
)
from .page_pool import (
    _LayerClusterPagePool,
)
from .pinned_memory import RetroSpecPinnedMemoryManager
from .resident_cache import (
    RetroSpecResidentClusterCache,
    RetroSpecResidentPageAccess,
)
from .verification_transfer import (
    _FullVerificationTransferBuffer,
)


class RetroSpecClusterPageStore(
    RetroSpecClusterPrefetchMixin, RetroSpecClusterVerificationMixin
):
    """CPU cluster-page backing store with a bounded GPU resident cache."""

    _RESIDENT_PREFETCH_RING_SIZE = 2
    _VERIFICATION_MISS_RING_SIZE = 2
    _VERIFICATION_ADMISSION_PRIORITY = 2

    def __init__(
        self,
        page_size: int,
        pin_memory: bool = False,
        cache_ratio: float = 0.0,
        cpu_page_initial_slab_bytes: int | None = None,
        cpu_page_slab_bytes: int = 1 << 20,
        max_pinned_memory_bytes: int = 64 << 20,
        max_pending_cluster_builds: int = 2,
        cpu_page_build_workers: int = 4,
        full_verify_gather_workers: int = 4,
        performance_stats: RetroSpecPerformanceStats | None = None,
        pinned_memory: RetroSpecPinnedMemoryManager | None = None,
        gpu_index_residency: RetroSpecGPUIndexResidencyManager | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if not 0.0 <= cache_ratio <= 1.0:
            raise ValueError("cache_ratio must be between zero and one")
        if cpu_page_slab_bytes <= 0:
            raise ValueError("cpu_page_slab_bytes must be positive")
        if cpu_page_initial_slab_bytes is None:
            cpu_page_initial_slab_bytes = min(8 << 20, cpu_page_slab_bytes)
        if cpu_page_initial_slab_bytes <= 0:
            raise ValueError("cpu_page_initial_slab_bytes must be positive")
        if cpu_page_initial_slab_bytes > cpu_page_slab_bytes:
            raise ValueError(
                "cpu_page_initial_slab_bytes must not exceed cpu_page_slab_bytes"
            )
        if max_pinned_memory_bytes <= 0:
            raise ValueError("max_pinned_memory_bytes must be positive")
        if max_pending_cluster_builds <= 0:
            raise ValueError("max_pending_cluster_builds must be positive")
        if cpu_page_build_workers <= 0:
            raise ValueError("cpu_page_build_workers must be positive")
        if full_verify_gather_workers <= 0:
            raise ValueError("full_verify_gather_workers must be positive")

        if pinned_memory is None:
            pinned_memory = RetroSpecPinnedMemoryManager(
                enabled=pin_memory,
                max_bytes=max_pinned_memory_bytes,
            )
        elif pin_memory and not pinned_memory.enabled:
            raise ValueError("pin_memory conflicts with the shared pinned manager")

        self.page_size = page_size
        self._pinned_memory = pinned_memory
        self.pin_memory = pinned_memory.enabled
        self.cache_ratio = cache_ratio
        self.cpu_page_initial_slab_bytes = cpu_page_initial_slab_bytes
        self.cpu_page_slab_bytes = cpu_page_slab_bytes
        self.max_pinned_memory_bytes = pinned_memory.max_bytes
        self.max_pending_cluster_builds = max_pending_cluster_builds
        self.cpu_page_build_workers = cpu_page_build_workers
        self.full_verify_gather_workers = full_verify_gather_workers
        self.performance_stats = performance_stats
        self._gpu_index_residency = gpu_index_residency
        self._draft_resolve_counter_buffer: torch.Tensor | None = None
        self._draft_resolve_counter_indices: tuple[int, ...] | None = None
        if performance_stats is not None and performance_stats.enabled:
            (
                self._draft_resolve_counter_buffer,
                self._draft_resolve_counter_indices,
            ) = performance_stats.get_gpu_counter_buffer(
                (
                    "resident_cluster_hits",
                    "resident_cluster_misses",
                    "draft_compact_resident_pages",
                    "draft_compact_selected_clusters",
                    "resident_bound_direct_hits",
                    "resident_hash_fallback_lookups",
                    "resident_hash_fallback_hits",
                    "resident_hash_fallback_misses",
                    "resident_hash_probe_steps",
                    "resident_hash_max_probe",
                    "resident_binding_invalidations",
                )
            )

        self._layer_pools: dict[str, _LayerClusterPagePool] = {}
        self._resident_caches: dict[str, RetroSpecResidentClusterCache] = {}

        self._next_cluster_ids: dict[str, int] = {}
        self._allocated_cluster_ids: dict[str, set[int]] = {}
        self._cluster_block_descriptors: dict[
            str, dict[int, _ClusterBlockDescriptor]
        ] = {}
        self._prefetch_descriptor_arenas: dict[str, _LayerPrefetchDescriptorArena] = {}

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
            torch.device, list[_PinnedSelectionSlot]
        ] = {}
        self._resident_access_record_capacity = 0
        self._resident_prefetch_wave_max_records = 1
        self._resident_prefetch_futures: deque[_ResidentPrefetchWaveFuture] = deque()
        self._resident_prefetch_deferred: dict[
            torch.device, _DeferredResidentPrefetchWave
        ] = {}
        self._resident_prefetch_last_metadata_events: dict[
            torch.device, torch.cuda.Event
        ] = {}
        self._resident_prefetch_lock = Lock()
        self._resident_prefetch_executor = _ResidentPrefetchPriorityExecutor()
        self._resident_prefetch_next_sequence = 1
        self._resident_prefetch_latest: dict[str, tuple[int, int]] = {}
        self._verification_miss_slots: dict[
            torch.device, list[_PinnedVerificationMissSlot]
        ] = {}
        self._verification_gpu_arenas: dict[
            torch.device, list[_VerificationResolveGPUArena]
        ] = {}
        self._verification_resolve_cursors: dict[torch.device, int] = {}
        self._verification_metadata_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._verification_resolve_lock = Lock()
        self._verification_resolve_ring_size = self._VERIFICATION_MISS_RING_SIZE
        self._verification_admission_futures: deque[Future[None]] = deque()
        self._resident_state_lock = RLock()
        self._resident_admission_frozen = False
        self._closed = False

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

            if len(slots) >= self.max_pending_cluster_builds:
                raise RuntimeError(
                    "No RetroSpec cluster-build staging slot is available"
                )

            slot = _PinnedStagingSlot(
                source_device=canonical_device,
                max_bytes=(
                    self.max_pinned_memory_bytes // 2 // self.max_pending_cluster_builds
                ),
                pinned_memory=self._pinned_memory,
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
                timer = (
                    None
                    if self.performance_stats is None
                    else self.performance_stats.start_cuda_timer(
                        "prefill_token_kv_d2h", offload_stream
                    )
                )
                try:
                    staged_token_keys.copy_(token_keys, non_blocking=True)
                    staged_token_values.copy_(token_values, non_blocking=True)
                finally:
                    if self.performance_stats is not None:
                        self.performance_stats.stop_cuda_timer(timer, offload_stream)
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
                timer = (
                    None
                    if self.performance_stats is None
                    else self.performance_stats.start_cuda_timer(
                        "prefill_cluster_metadata_d2h", offload_stream
                    )
                )
                try:
                    staged_assignments.copy_(assignments, non_blocking=True)
                    staged_cluster_token_counts.copy_(
                        cluster_token_counts, non_blocking=True
                    )
                    staged_token_offsets.copy_(
                        token_offsets_in_cluster, non_blocking=True
                    )
                finally:
                    if self.performance_stats is not None:
                        self.performance_stats.stop_cuda_timer(timer, offload_stream)
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
        stats = self.performance_stats
        measure = stats is not None and stats.enabled
        phase_started = perf_counter() if measure else 0.0
        if cluster_start < 0:
            raise ValueError("cluster_start must be non-negative")

        self._validate_cluster_input_metadata(
            token_keys,
            token_values,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )

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

        if torch.any(storage_cluster_counts < 0).item():
            raise ValueError("cluster_token_counts must be non-negative")

        cluster_page_counts = torch.div(
            storage_cluster_counts.to(dtype=torch.int64) + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        total_pages = int(cluster_page_counts.sum().item())
        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "cluster_pages_built",
                total_pages,
            )
        if measure:
            stats.record_cpu_time(
                "cluster_page_prepare_wall", perf_counter() - phase_started
            )

        phase_started = perf_counter() if measure else 0.0
        allocated_page_ids = pool.allocate(total_pages)
        if measure:
            stats.record_cpu_time(
                "cluster_page_allocate_wall", perf_counter() - phase_started
            )

        try:
            phase_started = perf_counter() if measure else 0.0
            page_ids, page_token_counts, full_descriptor = pool.build_cluster_pages(
                allocated_page_ids=allocated_page_ids,
                token_keys=storage_keys,
                token_values=storage_values,
                assignments=storage_assignments,
                cluster_token_counts=storage_cluster_counts,
                token_offsets_in_cluster=storage_token_offsets,
                num_workers=self.cpu_page_build_workers,
            )
            if measure:
                stats.record_cpu_time(
                    "cluster_page_native_wall", perf_counter() - phase_started
                )
        except Exception:
            pool.free(allocated_page_ids)
            raise

        cluster_ids: torch.Tensor | None = None
        with self._resident_state_lock:
            try:
                phase_started = perf_counter() if measure else 0.0
                cluster_ids = self._allocate_cluster_ids(
                    layer_name=layer_name,
                    request_id=request_id,
                    cluster_start=cluster_start,
                    cluster_token_counts=storage_cluster_counts,
                    page_ids=page_ids,
                    page_token_counts=page_token_counts,
                )
                if measure:
                    stats.record_cpu_time(
                        "cluster_page_register_wall", perf_counter() - phase_started
                    )
                phase_started = perf_counter() if measure else 0.0
                self._resize_resident_cache(layer_name, pool)
                if measure:
                    stats.record_cpu_time(
                        "cluster_page_resize_wall", perf_counter() - phase_started
                    )
            except Exception:
                if cluster_ids is not None:
                    self._free_cluster_ids(layer_name, cluster_ids)
                pool.free(allocated_page_ids)
                raise

        assert cluster_ids is not None
        return RetroSpecClusterBlockTable(
            cluster_ids=cluster_ids,
            page_metadata=RetroSpecClusterBlockMetadata(
                page_ids=page_ids,
                page_token_counts=page_token_counts,
            ),
            full_verification_descriptor=full_descriptor,
        )

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

            resident_cache = self._resident_caches.get(layer_name)
            if resident_cache is not None:
                with resident_cache.mutation_guard():
                    resident_cache.invalidate(block_table.cluster_ids)

            pool.free(block_table.page_metadata.page_ids)
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

    def read_page_storage(
        self,
        layer_name: str,
        page_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read logical page handles from the layer's pageable CPU slabs."""
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        return pool.read(page_ids)

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
            selected_cluster_ids, selected_page_ids = (
                self._select_resident_staging_prefix(
                    pool, cluster_ids_cpu, page_ids_cpu
                )
            )
            (
                source_page_ids,
                source_key_pages,
                source_value_pages,
                transfer_buffer,
                transfer_slot,
            ) = self._stage_resident_pages(pool, selected_page_ids)

            try:
                with resident_cache.mutation_guard():
                    access = resident_cache.admit_staged(
                        cluster_ids=selected_cluster_ids,
                        page_ids=selected_page_ids,
                        cluster_groups=self._get_cluster_groups(
                            layer_name,
                            selected_cluster_ids,
                        ),
                        allocated_cluster_ids=self._get_allocated_cluster_ids(
                            layer_name
                        ),
                        allocated_page_ids=pool.allocated_page_ids,
                        staging_page_ids=source_page_ids,
                        staging_key_pages=source_key_pages,
                        staging_value_pages=source_value_pages,
                        cluster_ids_cpu=selected_cluster_ids,
                        page_ids_cpu=selected_page_ids,
                    )
            except BaseException:
                resident_cache.synchronize_pending_copies()
                if transfer_slot is not None:
                    transfer_buffer.release_cpu_slot(transfer_slot, None)
                raise
            if transfer_slot is not None:
                transfer_buffer.release_cpu_slot(transfer_slot, access.ready_event)
            return resident_cache.lookup(
                cluster_ids=cluster_ids,
                page_ids=page_ids,
                cluster_groups=self._get_cluster_groups(layer_name, cluster_ids_cpu),
                allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
                allocated_page_ids=pool.allocated_page_ids,
                touch=False,
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

            with resident_cache.mutation_guard():
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

    @torch.inference_mode()
    def admit_verification_misses(
        self,
        admission: RetroSpecVerificationMissAdmission | None,
    ) -> None:
        """Admit compact verification misses without re-reading GPU metadata."""
        if self._resident_admission_frozen:
            return
        if admission is None or admission.cluster_ids_cpu.numel() == 0:
            return

        with self._resident_state_lock:
            pool, resident_cache = self._get_or_create_resident_cache(
                admission.layer_name
            )
            cluster_groups = self._get_cluster_groups(
                admission.layer_name, admission.cluster_ids_cpu
            )
            with resident_cache.mutation_guard():
                resident_cache.admit_staged(
                    cluster_ids=admission.cluster_ids_cpu,
                    page_ids=admission.logical_page_ids_cpu,
                    cluster_groups=cluster_groups,
                    allocated_cluster_ids=self._get_allocated_cluster_ids(
                        admission.layer_name
                    ),
                    allocated_page_ids=pool.allocated_page_ids,
                    staging_page_ids=admission.staging_page_ids_cpu,
                    staging_key_pages=admission.staging_key_pages,
                    staging_value_pages=admission.staging_value_pages,
                    cluster_ids_cpu=admission.cluster_ids_cpu,
                    page_ids_cpu=admission.logical_page_ids_cpu,
                    reuse_ready_event=admission.staging_ready_event,
                    lookup_after_admit=False,
                )

    def submit_verification_miss_admission(
        self,
        admission: RetroSpecVerificationMissAdmission | None,
    ) -> None:
        """Move post-attention LRU admission off the model thread."""
        self._reap_verification_admissions(wait=False)
        if admission is None or admission.cluster_ids_cpu.numel() == 0:
            return
        future = self._resident_prefetch_executor.submit(
            self._VERIFICATION_ADMISSION_PRIORITY,
            self.admit_verification_misses,
            admission,
        )
        self._verification_admission_futures.append(future)
        if self.performance_stats is not None:
            self.performance_stats.add_counter("verification_async_admissions")

    def wait_for_verification_admissions(self) -> None:
        """Drain sparse admissions before a bandwidth-heavy phase starts."""
        self._reap_verification_admissions(wait=True)

    def _reap_verification_admissions(self, wait: bool) -> None:
        remaining: deque[Future[None]] = deque()
        while self._verification_admission_futures:
            future = self._verification_admission_futures.popleft()
            if wait or future.done():
                future.result()
            else:
                remaining.append(future)
        self._verification_admission_futures = remaining

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


# Keep existing class paths stable for serialization.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = "vllm.v1.spec_decode.retrospec.cluster_store"
del _legacy_type

__all__ = [
    "RetroSpecClusterResolveMode",
    "RetroSpecResidentPrefetchSource",
    "RetroSpecResidentPrefetchInput",
    "RetroSpecClusterBlockTable",
    "RetroSpecClusterBlockMetadata",
    "RetroSpecStagedTokenKV",
    "RetroSpecStagedClusterInput",
    "RetroSpecCompactTokenRange",
    "RetroSpecFullVerificationDescriptor",
    "RetroSpecFullVerificationStaging",
    "RetroSpecFullVerificationTicket",
    "RetroSpecResolvedClusterPages",
    "RetroSpecCompactResolvedClusterPages",
    "RetroSpecRankedDraftResolvedClusters",
    "RetroSpecCompactVerificationResolvedPages",
    "RetroSpecVerificationMissAdmission",
    "RetroSpecVerificationResolveRequest",
    "RetroSpecClusterPageStore",
]
