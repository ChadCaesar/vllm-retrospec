# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import PriorityQueue
from threading import Event as ThreadEvent
from threading import Lock, Thread
from typing import TYPE_CHECKING, Literal

import torch

from .cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from .cluster_staging_types import (
    _PREFETCH_SOURCE_PRIORITY as _PREFETCH_SOURCE_PRIORITY,
)
from .cluster_staging_types import (
    RetroSpecResidentPrefetchInput,
    RetroSpecResidentPrefetchSource,
    _PinnedSelectionSlot,
    _PinnedStagingSlot,
)
from .cluster_staging_types import (
    _CPUPageSlab as _CPUPageSlab,
)
from .cluster_staging_types import (
    _FullVerificationSourceSnapshot as _FullVerificationSourceSnapshot,
)
from .cluster_staging_types import (
    _PinnedPageTransferSlot as _PinnedPageTransferSlot,
)
from .cluster_staging_types import (
    _PinnedVerificationMissSlot as _PinnedVerificationMissSlot,
)
from .cluster_staging_types import (
    _VerificationResolveGPUArena as _VerificationResolveGPUArena,
)
from .cluster_verification_types import (
    RetroSpecCompactResolvedClusterPages as RetroSpecCompactResolvedClusterPages,
)
from .cluster_verification_types import (
    RetroSpecCompactTokenRange as RetroSpecCompactTokenRange,
)
from .cluster_verification_types import (
    RetroSpecCompactVerificationResolvedPages,  # noqa: F401 (compatibility export)
    RetroSpecFullVerificationDescriptor,
)
from .cluster_verification_types import (
    RetroSpecFullVerificationStaging as RetroSpecFullVerificationStaging,
)
from .cluster_verification_types import (
    RetroSpecFullVerificationTicket as RetroSpecFullVerificationTicket,
)
from .cluster_verification_types import (
    RetroSpecRankedDraftResolvedClusters as RetroSpecRankedDraftResolvedClusters,
)
from .cluster_verification_types import (
    RetroSpecResolvedClusterPages as RetroSpecResolvedClusterPages,
)
from .cluster_verification_types import (
    RetroSpecVerificationMissAdmission as RetroSpecVerificationMissAdmission,
)
from .cluster_verification_types import (
    RetroSpecVerificationResolveRequest as RetroSpecVerificationResolveRequest,
)
from .cluster_verification_types import (
    _SubmittedVerificationResolve as _SubmittedVerificationResolve,
)
from .resident_cache import RetroSpecResidentClusterCache

if TYPE_CHECKING:
    from .page_pool import _LayerClusterPagePool

RetroSpecClusterResolveMode = Literal[
    "resident_only",
    "resident_pending",
    "verification",
]


@dataclass(frozen=True)
class _StagedResidentPrefetchRecord:
    """One layer's resident miss commands staged in pinned CPU memory."""

    layer_name: str
    miss_cluster_ids_cpu: torch.Tensor
    miss_positions_cpu: torch.Tensor
    miss_count_cpu: torch.Tensor
    num_groups: int
    num_ranks: int
    source: RetroSpecResidentPrefetchSource
    sequence: int


@dataclass
class _ResidentPrefetchWaveProgress:
    """Per-layer completion state for one resident-prefetch wave."""

    layer_events: dict[str, ThreadEvent]
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @classmethod
    def create(cls, layer_names: Sequence[str]) -> "_ResidentPrefetchWaveProgress":
        return cls(
            layer_events={layer_name: ThreadEvent() for layer_name in layer_names}
        )

    def complete_layer(self, layer_name: str) -> None:
        event = self.layer_events.get(layer_name)
        if event is None:
            raise RuntimeError(f"Unknown resident-prefetch layer: {layer_name}")
        event.set()

    def fail(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error
        for event in self.layer_events.values():
            event.set()

    def wait_for(self, layer_names: Sequence[str]) -> None:
        events: list[ThreadEvent] = []
        for layer_name in dict.fromkeys(layer_names):
            event = self.layer_events.get(layer_name)
            if event is None:
                raise RuntimeError(f"Unknown resident-prefetch layer: {layer_name}")
            events.append(event)

        for event in events:
            event.wait()

        with self._lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(
                "Resident-prefetch background processing failed"
            ) from failure


@dataclass(frozen=True)
class _StagedResidentPrefetchWave:
    """One draft step's cross-layer resident access records."""

    records: tuple[_StagedResidentPrefetchRecord, ...]
    device: torch.device
    metadata_ready_event: torch.cuda.Event
    execution_stream: torch.cuda.Stream
    progress: _ResidentPrefetchWaveProgress = field(repr=False, compare=False)
    slot: _PinnedSelectionSlot = field(repr=False, compare=False)


@dataclass(frozen=True)
class _DeferredResidentPrefetchWave:
    """Latest per-layer commands retained while the pinned ring is full."""

    records: tuple[RetroSpecResidentPrefetchInput, ...]
    source_ready_events: tuple[torch.cuda.Event, ...]


class _ResidentPrefetchPriorityExecutor:
    """One fixed worker that lets fresh DRAFT work overtake queued hints."""

    def __init__(self) -> None:
        self._queue = PriorityQueue()
        self._lock = Lock()
        self._next_sequence = 0
        self._closed = False
        self._thread = Thread(
            target=self._run,
            name="retrospec-resident-prefetch",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self, priority: int, function: Callable[..., None], *args: object
    ) -> Future[None]:
        future: Future[None] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("Resident prefetch executor is closed")
            sequence = self._next_sequence
            self._next_sequence += 1
            self._queue.put((-priority, sequence, function, args, future))
        return future

    def _run(self) -> None:
        while True:
            _, _, function, args, future = self._queue.get()
            try:
                if function is None:
                    return
                assert future is not None
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    function(*args)
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(None)
            finally:
                self._queue.task_done()

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sequence = self._next_sequence
            self._next_sequence += 1
            self._queue.put((1 << 30, sequence, None, (), None))
        if wait:
            self._thread.join()


@dataclass(frozen=True)
class _PreparedResidentPrefetchRecord:
    """CPU descriptors prepared for one layer's resident admission."""

    layer_name: str
    pool: "_LayerClusterPagePool"
    resident_cache: RetroSpecResidentClusterCache
    cluster_ids_cpu: torch.Tensor
    page_ids_cpu: torch.Tensor
    source_page_ids_cpu: torch.Tensor
    unique_page_ids_cpu: torch.Tensor
    cluster_groups: dict[int, RetroSpecClusterGroup]


@dataclass(frozen=True)
class _ResidentPrefetchWaveFuture:
    """One background wave and the CUDA device whose ring slot it owns."""

    device: torch.device
    layer_names: frozenset[str]
    progress: _ResidentPrefetchWaveProgress = field(repr=False, compare=False)
    future: Future[None] = field(repr=False, compare=False)


@dataclass(frozen=True)
class RetroSpecClusterBlockTable:
    """Ownership handle for CPU-managed cluster blocks.

    cluster_ids has shape:

        [num_kv_heads, num_clusters]

    page_metadata is the immutable CPU page layout produced while the cluster
    pages are built. Retaining it avoids rebuilding the same descriptor during
    request publication and release. Arbitrary active selections continue to
    resolve through RetroSpecClusterPageStore.
    """

    cluster_ids: torch.Tensor
    page_metadata: "RetroSpecClusterBlockMetadata" = field(
        repr=False,
        compare=False,
    )
    full_verification_descriptor: "RetroSpecFullVerificationDescriptor"


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


@dataclass
class _LayerPrefetchDescriptorArena:
    """Pageable CPU descriptors indexed directly by stable cluster handle."""

    page_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.int64)
    )
    page_counts: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    group_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )
    groups: list[RetroSpecClusterGroup] = field(default_factory=list)
    group_to_id: dict[RetroSpecClusterGroup, int] = field(default_factory=dict)

    @property
    def capacity(self) -> int:
        return self.page_counts.numel()

    @property
    def max_pages(self) -> int:
        return self.page_ids.shape[1]

    def _group_id(self, group: RetroSpecClusterGroup) -> int:
        group_id = self.group_to_id.get(group)
        if group_id is not None:
            return group_id
        group_id = len(self.groups)
        self.groups.append(group)
        self.group_to_id[group] = group_id
        return group_id

    def reserve(self, required_capacity: int, required_pages: int) -> None:
        if required_capacity <= self.capacity and required_pages <= self.max_pages:
            return

        capacity = 1 << (max(required_capacity, 1) - 1).bit_length()
        capacity = max(capacity, self.capacity)
        max_pages = max(required_pages, self.max_pages)
        # Publish/free can run on a CPU worker after an inference-mode prefill.
        # Persistent descriptor storage must therefore remain mutable outside
        # the inference-mode scope in which it was first allocated.
        with torch.inference_mode(False):
            page_ids = torch.full((capacity, max_pages), -1, dtype=torch.int64)
            page_counts = torch.zeros(capacity, dtype=torch.int32)
            group_ids = torch.full((capacity,), -1, dtype=torch.int64)
            if self.capacity:
                page_ids[: self.capacity, : self.max_pages].copy_(self.page_ids)
                page_counts[: self.capacity].copy_(self.page_counts)
                group_ids[: self.capacity].copy_(self.group_ids)
        self.page_ids = page_ids
        self.page_counts = page_counts
        self.group_ids = group_ids

    def publish(self, descriptors: dict[int, _ClusterBlockDescriptor]) -> None:
        if not descriptors:
            return

        ordered = tuple(sorted(descriptors.items()))
        required_pages = max(len(descriptor.page_ids) for _, descriptor in ordered)
        self.reserve(ordered[-1][0] + 1, required_pages)
        cluster_ids = torch.tensor(
            [cluster_id for cluster_id, _ in ordered], dtype=torch.int64
        )
        pages = torch.tensor(
            [
                descriptor.page_ids
                + (-1,) * (self.max_pages - len(descriptor.page_ids))
                for _, descriptor in ordered
            ],
            dtype=torch.int64,
        )
        counts = torch.tensor(
            [len(descriptor.page_ids) for _, descriptor in ordered],
            dtype=torch.int32,
        )
        groups = torch.tensor(
            [self._group_id(descriptor.identity.group) for _, descriptor in ordered],
            dtype=torch.int64,
        )
        self.page_ids.index_copy_(0, cluster_ids, pages)
        self.page_counts.index_copy_(0, cluster_ids, counts)
        self.group_ids.index_copy_(0, cluster_ids, groups)

    def invalidate(self, cluster_ids: set[int]) -> None:
        if not cluster_ids:
            return
        indices = torch.tensor(sorted(cluster_ids), dtype=torch.int64)
        self.page_ids.index_fill_(0, indices, -1)
        self.page_counts.index_fill_(0, indices, 0)
        self.group_ids.index_fill_(0, indices, -1)

    def resolve_groups(
        self, cluster_ids: torch.Tensor, group_ids: torch.Tensor
    ) -> dict[int, RetroSpecClusterGroup]:
        return {
            cluster_id: self.groups[group_id]
            for cluster_id, group_id in zip(
                cluster_ids.tolist(), group_ids.tolist(), strict=True
            )
        }


# Keep the original qualified path for serialized classes.
for _legacy_type in tuple(globals().values()):
    if isinstance(_legacy_type, type) and _legacy_type.__module__ == __name__:
        _legacy_type.__module__ = "vllm.v1.spec_decode.retrospec.cluster_store"
del _legacy_type
