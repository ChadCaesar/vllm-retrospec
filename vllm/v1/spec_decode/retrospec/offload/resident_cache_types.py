# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Lock

import torch

from .cluster_identity import RetroSpecClusterGroup

_ClusterId = int
_LogicalPages = tuple[int, ...]
RetroSpecResidentBindingPublisher = Callable[
    [tuple[int, ...], tuple[int, ...], torch.cuda.Stream], None
]

_PREFETCH_ABSENT = 0
_PREFETCH_PENDING = 1
_PREFETCH_RESIDENT = 2


@dataclass(frozen=True)
class _PendingCopyBatch:
    """Keep asynchronous cache-copy sources alive until completion."""

    ready_event: torch.cuda.Event
    cluster_ids: tuple[_ClusterId, ...]
    source_key_pages: torch.Tensor
    source_value_pages: torch.Tensor


@dataclass
class _ResidentGroupState:
    """Resident replacement state for one request and KV head."""

    lru: OrderedDict[_ClusterId, None] = field(default_factory=OrderedDict)
    num_pages: int = 0


@dataclass(frozen=True)
class RetroSpecResidentPageAccess:
    """Result of resolving logical cluster pages against the GPU cache."""

    cache_page_ids: torch.Tensor
    hit_cluster_mask: torch.Tensor
    miss_cluster_mask: torch.Tensor
    hit_gate_ready_mask: torch.Tensor
    logical_page_ids_cpu: torch.Tensor | None
    miss_cluster_mask_cpu: torch.Tensor | None
    ready_event: torch.cuda.Event | None
    access_kinds: torch.Tensor | None = None
    read_lease: "RetroSpecResidentReadLease | None" = None


@dataclass(frozen=True)
class RetroSpecPreparedResidentAdmission:
    """Immutable CPU descriptors prepared before taking the mutation guard."""

    cluster_ids: torch.Tensor
    page_ids: torch.Tensor
    cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup]
    source_page_ids: torch.Tensor
    source_key_pages: torch.Tensor
    source_value_pages: torch.Tensor
    cluster_ids_cpu: torch.Tensor
    page_ids_cpu: torch.Tensor
    parsed_cluster_ids: tuple[_ClusterId | None, ...]
    parsed_cluster_pages: tuple[_LogicalPages, ...]
    valid_positions: tuple[tuple[int, ...], ...]
    requested_clusters: tuple[_ClusterId, ...]
    cluster_page_map: Mapping[_ClusterId, _LogicalPages]
    cluster_source_ids: Mapping[_ClusterId, tuple[int, ...]]
    referenced_cluster_ids: frozenset[_ClusterId]
    referenced_page_ids: frozenset[int]


@dataclass(frozen=True)
class RetroSpecResidentLruCapture:
    """Immutable resident descriptors captured while structure is stable."""

    structure_revision: int
    groups: tuple[RetroSpecClusterGroup, ...]
    epoch_table: torch.Tensor = field(repr=False, compare=False)
    stream: torch.cuda.Stream = field(repr=False, compare=False)


@dataclass(frozen=True)
class RetroSpecResolvedResidentLru:
    """CPU-resolved LRU epoch snapshot awaiting revision validation."""

    structure_revision: int
    groups: tuple[RetroSpecClusterGroup, ...]
    epochs: tuple[int, ...]


@dataclass(frozen=True)
class RetroSpecCompactResidentPageAccess:
    """Row-local compact resident pages and fused DRAFT statistics."""

    cache_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    clustered_token_counts: torch.Tensor
    attention_mass: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    hit_gate_ready: torch.Tensor
    access_kinds: torch.Tensor | None
    read_lease: "RetroSpecResidentReadLease"


@dataclass(frozen=True)
class RetroSpecRankedDraftResidentAccess:
    """Stable resident-table view for one ranked DRAFT selection."""

    cluster_handles: torch.Tensor
    resident_bucket_ids: torch.Tensor
    clustered_token_counts: torch.Tensor
    attention_mass: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    hit_gate_ready: torch.Tensor
    resident_table_page_counts: torch.Tensor
    resident_table_page_slots: torch.Tensor
    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor
    read_lease: "RetroSpecResidentReadLease"


@dataclass(frozen=True)
class RetroSpecCompactVerificationPageAccess:
    """Query-row compact resident pages and GPU-unique verification misses."""

    resident_page_ids: torch.Tensor
    staging_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    unique_cluster_ids: torch.Tensor
    unique_logical_page_ids: torch.Tensor
    unique_page_counts: torch.Tensor
    miss_unique_indices: torch.Tensor
    miss_output_page_offsets: torch.Tensor
    miss_count: torch.Tensor
    unique_miss_count: torch.Tensor
    invalid_descriptor_count: torch.Tensor
    read_lease: "RetroSpecResidentReadLease"


class RetroSpecResidentReadLease:
    """Keep resident slots stable until their attention kernels are submitted."""

    def __init__(self, lock: Lock) -> None:
        self._lock = lock
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._lock.release()


# Keep serialized class paths compatible with the original public module.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = __name__.replace(
            ".offload.resident_cache_types", ".resident_cache"
        )
del _legacy_type
