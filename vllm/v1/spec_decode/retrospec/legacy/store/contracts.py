# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import CancelledError, Future
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Event as ThreadEvent
from typing import TYPE_CHECKING

import torch

from vllm.v1.spec_decode.retrospec.legacy.cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecResidentLayerArena,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_cache import (
    RetroSpecCompactVerificationPageAccess,
    RetroSpecResidentClusterCache,
    RetroSpecResidentReadLease,
)
from vllm.v1.spec_decode.retrospec.legacy.store.prefetch_types import (
    _PinnedStagingSlot,
    _PinnedVerificationMissSlot,
    _VerificationResolveGPUArena,
)

if TYPE_CHECKING:
    from vllm.v1.spec_decode.retrospec.legacy.store.pool import _LayerClusterPagePool


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
        pages = torch.full((len(ordered), self.max_pages), -1, dtype=torch.int64)
        counts = torch.empty(len(ordered), dtype=torch.int32)
        groups = torch.empty(len(ordered), dtype=torch.int64)
        for row, (_, descriptor) in enumerate(ordered):
            page_count = len(descriptor.page_ids)
            pages[row, :page_count] = torch.tensor(
                descriptor.page_ids, dtype=torch.int64
            )
            counts[row] = page_count
            groups[row] = self._group_id(descriptor.identity.group)
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


@dataclass(frozen=True)
class RetroSpecCompactTokenRange:
    """One valid-token range in a layer's pageable CPU slab storage."""

    slab_id: int
    token_offset: int
    token_count: int


class RetroSpecFullVerificationDescriptor:
    """Persistent compact full-prefix layout for every KV head of a request."""

    def __init__(
        self,
        head_ranges: tuple[tuple[RetroSpecCompactTokenRange, ...], ...],
        head_token_counts: tuple[int, ...],
    ) -> None:
        if len(head_ranges) != len(head_token_counts):
            raise ValueError("Full-verification descriptor head counts differ")

        rows: list[tuple[int, int, int, int, int]] = []
        for head_index, (ranges, expected_count) in enumerate(
            zip(head_ranges, head_token_counts)
        ):
            head_token_offset = 0
            for token_range in ranges:
                if token_range.slab_id < 0:
                    raise ValueError("Compact range slab ID must be non-negative")
                if token_range.token_offset < 0:
                    raise ValueError("Compact range token offset must be non-negative")
                if token_range.token_count <= 0:
                    raise ValueError("Compact range token count must be positive")

                rows.append(
                    (
                        head_index,
                        token_range.slab_id,
                        token_range.token_offset,
                        token_range.token_count,
                        head_token_offset,
                    )
                )
                head_token_offset += token_range.token_count

            if head_token_offset != expected_count:
                raise ValueError(
                    "Full-verification descriptor token count is inconsistent"
                )

        range_table = (
            torch.tensor(rows, dtype=torch.int64, device="cpu")
            if rows
            else torch.empty((0, 5), dtype=torch.int64, device="cpu")
        )
        counts = torch.tensor(head_token_counts, dtype=torch.int32, device="cpu")
        self._set_tensor_representation(range_table, counts)
        self._head_ranges_cache = head_ranges
        self._head_token_counts_cache = head_token_counts

    def _set_tensor_representation(
        self,
        range_table: torch.Tensor,
        head_token_counts: torch.Tensor,
    ) -> None:
        if range_table.device.type != "cpu" or range_table.dtype != torch.int64:
            raise ValueError("Full-verification range table must be CPU int64")
        if range_table.ndim != 2 or range_table.shape[1] != 5:
            raise ValueError(
                "Full-verification range table must have shape [ranges, 5]"
            )
        if (
            head_token_counts.device.type != "cpu"
            or head_token_counts.dtype != torch.int32
        ):
            raise ValueError("Full-verification head counts must be CPU int32")
        if head_token_counts.ndim != 1:
            raise ValueError("Full-verification head counts must be one-dimensional")

        range_table = range_table.contiguous()
        head_token_counts = head_token_counts.contiguous()
        num_heads = head_token_counts.shape[0]

        if range_table.numel():
            head_ids = range_table[:, 0]
            if torch.any((head_ids < 0) | (head_ids >= num_heads)).item():
                raise ValueError("Full-verification range contains an invalid head")
            if torch.any(range_table[:, 1:4] < 0).item():
                raise ValueError("Full-verification range contains negative fields")
            if torch.any(range_table[:, 3] == 0).item():
                raise ValueError("Full-verification range must contain tokens")
            if torch.any(range_table[:, 4] < 0).item():
                raise ValueError("Full-verification output offset is negative")

            actual_counts = torch.zeros(num_heads, dtype=torch.int64, device="cpu")
            actual_counts.scatter_add_(0, head_ids, range_table[:, 3])
            if not torch.equal(actual_counts, head_token_counts.to(dtype=torch.int64)):
                raise ValueError(
                    "Full-verification range counts do not match head counts"
                )
        elif torch.any(head_token_counts != 0).item():
            raise ValueError("Empty range table has non-zero head counts")

        self.range_table = range_table
        self.head_token_counts_tensor = head_token_counts

    @classmethod
    def from_tensors(
        cls,
        range_table: torch.Tensor,
        head_token_counts: torch.Tensor,
    ) -> "RetroSpecFullVerificationDescriptor":
        descriptor = cls.__new__(cls)
        descriptor._set_tensor_representation(range_table, head_token_counts)
        descriptor._head_ranges_cache = None
        descriptor._head_token_counts_cache = None
        return descriptor

    @classmethod
    def empty(cls, num_kv_heads: int) -> "RetroSpecFullVerificationDescriptor":
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        return cls.from_tensors(
            torch.empty((0, 5), dtype=torch.int64, device="cpu"),
            torch.zeros(num_kv_heads, dtype=torch.int32, device="cpu"),
        )

    @property
    def num_kv_heads(self) -> int:
        return self.head_token_counts_tensor.shape[0]

    @property
    def num_tokens(self) -> int:
        return int(self.head_token_counts_tensor.sum().item())

    @property
    def head_token_counts(self) -> tuple[int, ...]:
        if self._head_token_counts_cache is None:
            self._head_token_counts_cache = tuple(
                self.head_token_counts_tensor.tolist()
            )
        return self._head_token_counts_cache

    @property
    def head_ranges(
        self,
    ) -> tuple[tuple[RetroSpecCompactTokenRange, ...], ...]:
        if self._head_ranges_cache is None:
            ranges: list[list[RetroSpecCompactTokenRange]] = [
                [] for _ in range(self.num_kv_heads)
            ]
            for (
                head_index,
                slab_id,
                token_offset,
                token_count,
                _,
            ) in self.range_table.tolist():
                ranges[head_index].append(
                    RetroSpecCompactTokenRange(
                        slab_id=slab_id,
                        token_offset=token_offset,
                        token_count=token_count,
                    )
                )
            self._head_ranges_cache = tuple(tuple(row) for row in ranges)
        return self._head_ranges_cache

    def append(
        self, other: "RetroSpecFullVerificationDescriptor"
    ) -> "RetroSpecFullVerificationDescriptor":
        if self.num_kv_heads != other.num_kv_heads:
            raise ValueError("Full-verification descriptors changed KV-head count")

        appended_ranges = other.range_table.clone()
        if appended_ranges.numel():
            appended_heads = appended_ranges[:, 0]
            appended_ranges[:, 4].add_(
                self.head_token_counts_tensor.index_select(0, appended_heads).to(
                    dtype=torch.int64
                )
            )

        return RetroSpecFullVerificationDescriptor.from_tensors(
            torch.cat((self.range_table, appended_ranges), dim=0),
            self.head_token_counts_tensor + other.head_token_counts_tensor,
        )


@dataclass(frozen=True)
class RetroSpecFullVerificationStaging:
    """Token-contiguous GPU staging used only by full verification."""

    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    token_offsets: torch.Tensor
    token_counts: torch.Tensor
    max_tokens_per_head: int
    ready_event: torch.cuda.Event | None


@dataclass(frozen=True)
class RetroSpecFullVerificationTicket:
    """One asynchronously prepared full-verification layer."""

    future: Future[RetroSpecFullVerificationStaging]
    cancel_event: ThreadEvent

    def result(self) -> RetroSpecFullVerificationStaging:
        return self.future.result()

    def ready(self) -> bool:
        if not self.future.done() or self.future.cancelled():
            return False

        try:
            staging = self.future.result()
        except BaseException:
            return False

        return staging.ready_event is None or staging.ready_event.query()

    def cancel(self, wait: bool = False) -> bool:
        self.cancel_event.set()
        cancelled = self.future.cancel()
        if wait and not cancelled:
            with suppress(CancelledError):
                self.future.result()
        return cancelled


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
    access_kinds: torch.Tensor | None = None
    read_lease: RetroSpecResidentReadLease | None = None
    miss_admission: "RetroSpecVerificationMissAdmission | None" = None


@dataclass(frozen=True)
class RetroSpecCompactResolvedClusterPages:
    """Compact row-local resident page source produced for one DRAFT layer."""

    resident_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    clustered_token_counts: torch.Tensor
    attention_mass: torch.Tensor
    selected_cluster_counts: torch.Tensor
    hit_cluster_counts: torch.Tensor
    miss_cluster_counts: torch.Tensor
    hit_gate_ready: torch.Tensor
    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor
    read_lease: RetroSpecResidentReadLease


@dataclass(frozen=True)
class RetroSpecRankedDraftResolvedClusters:
    """Resident-table and GPU-index views consumed by DRAFT attention."""

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
    read_lease: RetroSpecResidentReadLease


@dataclass(frozen=True)
class RetroSpecCompactVerificationResolvedPages:
    """Query-row compact resident and staging pages for verification."""

    resident_page_ids: torch.Tensor
    staging_page_ids: torch.Tensor
    page_token_counts: torch.Tensor
    page_counts: torch.Tensor
    resident_key_pages: torch.Tensor
    resident_value_pages: torch.Tensor
    staging_key_pages: torch.Tensor
    staging_value_pages: torch.Tensor
    staging_ready_event: torch.cuda.Event | None
    read_lease: RetroSpecResidentReadLease
    miss_admission: "RetroSpecVerificationMissAdmission | None" = None


@dataclass(frozen=True)
class RetroSpecVerificationMissAdmission:
    """Compact CPU metadata for admitting verification misses after attention."""

    layer_name: str
    cluster_ids_cpu: torch.Tensor
    logical_page_ids_cpu: torch.Tensor
    staging_page_ids_cpu: torch.Tensor
    staging_key_pages: torch.Tensor
    staging_value_pages: torch.Tensor
    staging_ready_event: torch.cuda.Event | None


@dataclass(frozen=True)
class RetroSpecVerificationResolveRequest:
    """One layer's packed exact selection in a verification transaction."""

    layer_name: str
    selected_cluster_indices: torch.Tensor
    plan_valid_rows: torch.Tensor
    request_slot_ids: torch.Tensor
    request_slot_generations: torch.Tensor
    arena: RetroSpecResidentLayerArena
    max_pages_per_cluster: int


@dataclass
class _SubmittedVerificationResolve:
    """GPU lookup whose compact miss metadata has not been consumed yet."""

    layer_name: str
    pool: "_LayerClusterPagePool"
    resident_cache: RetroSpecResidentClusterCache
    slot: _PinnedVerificationMissSlot
    resolve_arena: _VerificationResolveGPUArena
    access: RetroSpecCompactVerificationPageAccess
    cluster_capacity: int
    max_pages_per_cluster: int
    lookup_ready_event: torch.cuda.Event
