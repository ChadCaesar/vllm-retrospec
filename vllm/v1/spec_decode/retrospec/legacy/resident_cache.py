# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from threading import Lock

import torch

from vllm.v1.spec_decode.retrospec.legacy.residency.cache.admission import (
    _RetroSpecResidentClusterCacheAdmissionMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.cache.handles import (
    _RetroSpecResidentClusterCacheHandlesMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.cache.lifecycle import (
    _RetroSpecResidentClusterCacheLifecycleMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.cache.lookup import (
    _RetroSpecResidentClusterCacheLookupMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.cache.types import (
    _PREFETCH_ABSENT,
    _PREFETCH_PENDING,
    _PREFETCH_RESIDENT,
    RetroSpecCompactResidentPageAccess,
    RetroSpecCompactVerificationPageAccess,
    RetroSpecPreparedResidentAdmission,
    RetroSpecRankedDraftResidentAccess,
    RetroSpecResidentBindingPublisher,
    RetroSpecResidentLruCapture,
    RetroSpecResidentPageAccess,
    RetroSpecResidentReadLease,
    RetroSpecResolvedResidentLru,
    _ClusterId,
    _LogicalPages,
    _PendingCopyBatch,
    _resident_handle_hash,
    _ResidentGroupState,
)

from ..performance import RetroSpecPerformanceStats
from .cluster_identity import RetroSpecClusterGroup

__all__ = [
    "_ClusterId",
    "_LogicalPages",
    "RetroSpecResidentBindingPublisher",
    "_PREFETCH_ABSENT",
    "_PREFETCH_PENDING",
    "_PREFETCH_RESIDENT",
    "_resident_handle_hash",
    "_PendingCopyBatch",
    "_ResidentGroupState",
    "RetroSpecResidentPageAccess",
    "RetroSpecPreparedResidentAdmission",
    "RetroSpecResidentLruCapture",
    "RetroSpecResolvedResidentLru",
    "RetroSpecCompactResidentPageAccess",
    "RetroSpecRankedDraftResidentAccess",
    "RetroSpecCompactVerificationPageAccess",
    "RetroSpecResidentReadLease",
    "RetroSpecResidentClusterCache",
]


class RetroSpecResidentClusterCache(
    _RetroSpecResidentClusterCacheHandlesMixin,
    _RetroSpecResidentClusterCacheLifecycleMixin,
    _RetroSpecResidentClusterCacheLookupMixin,
    _RetroSpecResidentClusterCacheAdmissionMixin,
):
    """Shared GPU page arena with group-scoped replacement state.

    Logical page IDs address the stable CPU backing store. Resident page IDs
    address slots in the shared key_pages and value_pages tensors.

    Every cluster belongs to one request/KV-head group. Each group owns an
    independent LRU and a soft resident-page target. Physical GPU slots remain
    shared by the whole layer, so unused group capacity may be borrowed.
    Admission and eviction always operate on complete clusters.
    """

    def __init__(
        self,
        page_size: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        performance_stats: RetroSpecPerformanceStats | None = None,
        binding_publisher: RetroSpecResidentBindingPublisher | None = None,
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
        self.performance_stats = performance_stats
        self._binding_publisher = binding_publisher

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

        # Physical page and slot ownership remains layer-wide.
        self._cluster_to_pages: dict[_ClusterId, _LogicalPages] = {}
        self._cluster_to_slots: dict[_ClusterId, tuple[int, ...]] = {}
        self._page_to_cluster: dict[int, _ClusterId] = {}
        self._free_slots: set[int] = set()

        # Semantic ownership and recency are isolated by request/KV-head group.
        self._cluster_to_group: dict[_ClusterId, RetroSpecClusterGroup] = {}
        self._group_states: dict[RetroSpecClusterGroup, _ResidentGroupState] = {}

        # Soft resident-page targets. Physical slots remain shared and groups may
        # temporarily borrow unused capacity from one another.
        self._group_targets: dict[RetroSpecClusterGroup, int] = {}

        self._copy_stream = torch.cuda.Stream(device=device)
        self._pending_copy_batches: deque[_PendingCopyBatch] = deque()
        self._pending_cluster_events: dict[_ClusterId, torch.cuda.Event] = {}

        # GPU draft lookups and resident mutations share this short host-side
        # guard. It is held only while kernels are submitted, never while CUDA
        # work completes or CPU descriptors are parsed.
        self._gpu_access_lock = Lock()

        # Open-addressed GPU handle table. Stable cluster IDs are the keys;
        # versions protect readers from concurrent publication on the H2D
        # stream. CPU shadows are used only by the background mutation path.
        self._handle_table_capacity = 0
        self._handle_table_max_pages = 0
        self._handle_table_handles = torch.empty(0, dtype=torch.int64, device=device)
        self._handle_table_versions = torch.empty(0, dtype=torch.int32, device=device)
        self._handle_table_page_counts = torch.empty(
            0, dtype=torch.int32, device=device
        )
        self._handle_table_page_slots = torch.empty(
            (0, 0), dtype=torch.int32, device=device
        )
        self._handle_table_hit_gate_ready = torch.empty(
            0, dtype=torch.bool, device=device
        )
        self._handle_table_last_access_epochs = torch.empty(
            0, dtype=torch.int64, device=device
        )
        self._next_access_epoch = 1
        self._handle_to_bucket: dict[_ClusterId, int] = {}
        self._bucket_handles: list[int] = []
        self._handle_table_needs_rebuild = False
        self._prefetch_handle_states = torch.zeros(0, dtype=torch.uint8, device="cpu")
        self._structure_revision = 0

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
    def num_resident_groups(self) -> int:
        return len(self._group_states)

    @property
    def num_pending_copy_batches(self) -> int:
        """Number of submitted copy batches not yet reaped by the host."""
        self._reap_completed_copy_batches()
        return len(self._pending_copy_batches)
