# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from threading import Lock, RLock

import torch

from vllm import _custom_ops as ops
from vllm.v1.spec_decode.retrospec.legacy.store.admission import (
    _RetroSpecClusterPageStoreAdmissionMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.store.build import (
    _RetroSpecClusterPageStoreBuildMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.store.identity import (
    _RetroSpecClusterPageStoreIdentityMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.store.prefetch_stage import (
    _RetroSpecClusterPageStorePrefetchStageMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.store.prefetch_submit import (
    _RetroSpecClusterPageStorePrefetchSubmitMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    _PREFETCH_SOURCE_PRIORITY,
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
    _CPUPageSlab,
    _DeferredResidentPrefetchWave,
    _FullVerificationGPUArena,
    _FullVerificationSourceSnapshot,
    _FullVerificationTransferBuffer,
    _LayerClusterPagePool,
    _LayerPrefetchDescriptorArena,
    _PinnedPageTransferSlot,
    _PinnedSelectionSlot,
    _PinnedStagingSlot,
    _PinnedVerificationMissSlot,
    _PreparedResidentPrefetchRecord,
    _ResidentPrefetchPriorityExecutor,
    _ResidentPrefetchWaveFuture,
    _ResidentPrefetchWaveProgress,
    _StagedResidentPrefetchRecord,
    _StagedResidentPrefetchWave,
    _SubmittedVerificationResolve,
    _VerificationResolveGPUArena,
)
from vllm.v1.spec_decode.retrospec.legacy.store.verification_resolve import (
    _RetroSpecClusterPageStoreVerificationResolveMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.store.verification_stage import (
    _RetroSpecClusterPageStoreVerificationStageMixin,
)

from ..performance import RetroSpecPerformanceStats
from .cluster_identity import (
    RetroSpecClusterGroup,
)
from .index_residency import (
    RetroSpecGPUIndexResidencyManager,
)
from .pinned_memory import RetroSpecPinnedMemoryManager
from .resident_cache import (
    RetroSpecResidentClusterCache,
)

__all__ = [
    "ops",
    "RetroSpecClusterResolveMode",
    "RetroSpecResidentPrefetchSource",
    "_PREFETCH_SOURCE_PRIORITY",
    "_PinnedStagingSlot",
    "RetroSpecResidentPrefetchInput",
    "_PinnedSelectionSlot",
    "_PinnedVerificationMissSlot",
    "_VerificationResolveGPUArena",
    "_StagedResidentPrefetchRecord",
    "_ResidentPrefetchWaveProgress",
    "_StagedResidentPrefetchWave",
    "_DeferredResidentPrefetchWave",
    "_ResidentPrefetchPriorityExecutor",
    "_PreparedResidentPrefetchRecord",
    "_ResidentPrefetchWaveFuture",
    "RetroSpecClusterBlockTable",
    "RetroSpecClusterBlockMetadata",
    "RetroSpecStagedTokenKV",
    "RetroSpecStagedClusterInput",
    "_ClusterBlockDescriptor",
    "_LayerPrefetchDescriptorArena",
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
    "_SubmittedVerificationResolve",
    "_CPUPageSlab",
    "_FullVerificationSourceSnapshot",
    "_PinnedPageTransferSlot",
    "_LayerClusterPagePool",
    "_FullVerificationGPUArena",
    "_FullVerificationTransferBuffer",
    "RetroSpecClusterPageStore",
]


class RetroSpecClusterPageStore(
    _RetroSpecClusterPageStoreIdentityMixin,
    _RetroSpecClusterPageStoreBuildMixin,
    _RetroSpecClusterPageStorePrefetchStageMixin,
    _RetroSpecClusterPageStorePrefetchSubmitMixin,
    _RetroSpecClusterPageStoreVerificationStageMixin,
    _RetroSpecClusterPageStoreVerificationResolveMixin,
    _RetroSpecClusterPageStoreAdmissionMixin,
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
