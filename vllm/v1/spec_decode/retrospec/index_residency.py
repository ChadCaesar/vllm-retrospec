# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetroSpecClusterSummary:
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor


@dataclass(frozen=True)
class RetroSpecStagedClusterSummary:
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    resident_summary: RetroSpecClusterSummary
    ready_event: torch.cuda.Event | None


@dataclass(frozen=True)
class RetroSpecResidentSegment:
    """Transient payload published into a layer-level GPU arena."""

    layer_name: str
    request_id: str

    indexed_start: int
    indexed_end: int
    cluster_start: int

    cluster_ids_cpu: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor

    cluster_page_ids_cpu: torch.Tensor
    cluster_page_token_counts_cpu: torch.Tensor
    cluster_page_counts_cpu: torch.Tensor


@dataclass
class RetroSpecResidentLayerArena:
    """Fixed-capacity resident cluster index for one attention layer."""

    cluster_ids: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor

    cluster_page_offsets: torch.Tensor
    page_ids: torch.Tensor
    page_token_counts: torch.Tensor

    num_clusters: torch.Tensor
    indexed_starts: torch.Tensor
    indexed_ends: torch.Tensor


@dataclass(frozen=True)
class RetroSpecResidentBatchView:
    """Map one active batch onto a persistent layer arena."""

    arena: RetroSpecResidentLayerArena | None
    request_slot_ids: torch.Tensor
    max_num_clusters: int
    max_pages_per_cluster: int


@dataclass(frozen=True)
class RetroSpecResidentIndex:
    indexed_token_mask: torch.Tensor

    cluster_ids: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_mask: torch.Tensor

    cluster_page_ids: torch.Tensor
    cluster_page_token_counts: torch.Tensor


@dataclass(frozen=True)
class _ResidentRequestState:
    """CPU control metadata for one request in one layer arena."""

    slot: int
    indexed_start: int
    indexed_end: int
    num_clusters: int
    page_ends: tuple[int, ...]
    max_pages_per_cluster: int


@dataclass(frozen=True)
class _ResidentIndexEntry:
    request_ids: tuple[str, ...]
    max_num_tokens: int
    index: RetroSpecResidentIndex


class RetroSpecGPUIndexResidencyManager:
    """Own request-slot layer arenas and temporary compatibility views."""

    def __init__(
        self,
        pin_memory: bool,
        max_resident_requests: int,
        max_clusters_per_request: int,
        max_pages_per_head_per_request: int,
    ) -> None:
        if max_resident_requests <= 0:
            raise ValueError("max_resident_requests must be positive")
        if max_clusters_per_request <= 0:
            raise ValueError("max_clusters_per_request must be positive")
        if max_pages_per_head_per_request <= 0:
            raise ValueError("max_pages_per_head_per_request must be positive")

        self.pin_memory = pin_memory
        self.max_resident_requests = max_resident_requests
        self.max_clusters_per_request = max_clusters_per_request
        self.max_pages_per_head_per_request = max_pages_per_head_per_request

        self._active_request_ids: tuple[str, ...] | None = None
        self._entries: dict[str, _ResidentIndexEntry] = {}
        self._active_views: dict[str, RetroSpecResidentBatchView] = {}

        self._request_slots: dict[str, int] = {}
        self._free_request_slots = list(reversed(range(max_resident_requests)))

        self._layer_arenas: dict[str, RetroSpecResidentLayerArena] = {}
        self._resident_states: dict[
            str,
            dict[str, _ResidentRequestState],
        ] = {}
        self._offload_streams: dict[torch.device, torch.cuda.Stream] = {}

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return () if self._active_request_ids is None else self._active_request_ids

    @property
    def resident_request_ids(self) -> tuple[str, ...]:
        request_ids = {
            request_id
            for layer_states in self._resident_states.values()
            for request_id in layer_states
        }
        return tuple(sorted(request_ids))

    @property
    def num_resident_requests(self) -> int:
        return len(self.resident_request_ids)

    @property
    def num_resident_layers(self) -> int:
        return sum(bool(states) for states in self._resident_states.values())

    @property
    def num_packed_layers(self) -> int:
        return len(self._entries)

    def activate(self, request_ids: Sequence[str]) -> None:
        if self._active_request_ids is not None:
            raise RuntimeError("A RetroSpec GPU index residency set is already active")

        request_ids = tuple(request_ids)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("RetroSpec resident request IDs must be unique")
        if len(request_ids) > self.max_resident_requests:
            raise RuntimeError(
                "RetroSpec GPU index residency exceeds max_num_seqs: "
                f"{len(request_ids)} > {self.max_resident_requests}"
            )

        self._entries.clear()
        self._active_views.clear()
        self._active_request_ids = request_ids

    def deactivate(self) -> None:
        if self._active_request_ids is None:
            raise RuntimeError("No RetroSpec GPU index residency set is active")

        self._entries.clear()
        self._active_views.clear()
        self._active_request_ids = None

    def _validate_active_requests(self, request_ids: tuple[str, ...]) -> None:
        if self._active_request_ids is None:
            raise RuntimeError(
                "RetroSpec resident indices may be accessed only inside an "
                "active proposal or full-verification context"
            )
        if request_ids != self._active_request_ids:
            raise RuntimeError(
                "RetroSpec request order does not match the active GPU residency set"
            )

    def get(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        max_num_tokens: int,
    ) -> RetroSpecResidentIndex | None:
        request_ids = tuple(request_ids)
        self._validate_active_requests(request_ids)

        entry = self._entries.get(layer_name)
        if (
            entry is None
            or entry.request_ids != request_ids
            or entry.max_num_tokens != max_num_tokens
        ):
            return None

        return entry.index

    def put(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        max_num_tokens: int,
        index: RetroSpecResidentIndex,
    ) -> None:
        request_ids = tuple(request_ids)
        self._validate_active_requests(request_ids)

        self._entries[layer_name] = _ResidentIndexEntry(
            request_ids=request_ids,
            max_num_tokens=max_num_tokens,
            index=index,
        )

    def _get_or_allocate_request_slot(self, request_id: str) -> int:
        slot = self._request_slots.get(request_id)
        if slot is not None:
            return slot

        if not self._free_request_slots:
            raise RuntimeError(
                "RetroSpec persistent GPU index residency exceeds max_num_seqs"
            )

        slot = self._free_request_slots.pop()
        self._request_slots[request_id] = slot
        return slot

    def _get_or_create_arena(
        self,
        layer_name: str,
        segment: RetroSpecResidentSegment,
    ) -> RetroSpecResidentLayerArena:
        arena = self._layer_arenas.get(layer_name)
        if arena is not None:
            if arena.cluster_keys.device != segment.cluster_keys.device:
                raise RuntimeError("Resident layer arena changed device")
            if arena.cluster_keys.dtype != segment.cluster_keys.dtype:
                raise RuntimeError("Resident layer arena changed dtype")
            if arena.cluster_keys.shape[1] != segment.cluster_keys.shape[0]:
                raise RuntimeError("Resident layer arena changed KV-head count")
            if arena.cluster_keys.shape[3] != segment.cluster_keys.shape[2]:
                raise RuntimeError("Resident layer arena changed head size")
            return arena

        num_kv_heads, _, head_size = segment.cluster_keys.shape
        device = segment.cluster_keys.device
        dtype = segment.cluster_keys.dtype

        summary_shape = (
            self.max_resident_requests,
            num_kv_heads,
            self.max_clusters_per_request,
            head_size,
        )
        cluster_shape = summary_shape[:-1]
        page_shape = (
            self.max_resident_requests,
            num_kv_heads,
            self.max_pages_per_head_per_request,
        )

        arena = RetroSpecResidentLayerArena(
            cluster_ids=torch.empty(cluster_shape, dtype=torch.int64, device=device),
            cluster_keys=torch.empty(summary_shape, dtype=dtype, device=device),
            cluster_values=torch.empty(summary_shape, dtype=dtype, device=device),
            cluster_token_counts=torch.empty(
                cluster_shape, dtype=torch.int32, device=device
            ),
            cluster_page_offsets=torch.empty(
                self.max_resident_requests,
                num_kv_heads,
                self.max_clusters_per_request + 1,
                dtype=torch.int64,
                device=device,
            ),
            page_ids=torch.empty(page_shape, dtype=torch.int64, device=device),
            page_token_counts=torch.empty(page_shape, dtype=torch.int32, device=device),
            num_clusters=torch.zeros(
                self.max_resident_requests, dtype=torch.int32, device=device
            ),
            indexed_starts=torch.zeros(
                self.max_resident_requests, dtype=torch.int64, device=device
            ),
            indexed_ends=torch.zeros(
                self.max_resident_requests, dtype=torch.int64, device=device
            ),
        )
        self._layer_arenas[layer_name] = arena
        return arena

    def build_resident_segment(
        self,
        layer_name: str,
        request_id: str,
        indexed_start: int,
        indexed_end: int,
        cluster_start: int,
        staged_summary: RetroSpecStagedClusterSummary,
        cluster_ids: torch.Tensor,
        cluster_page_ids: torch.Tensor,
        cluster_page_token_counts: torch.Tensor,
    ) -> RetroSpecResidentSegment:
        if indexed_start < 0 or indexed_end <= indexed_start:
            raise ValueError("Resident segment token range is invalid")
        if cluster_start < 0:
            raise ValueError("Resident segment cluster offset must be non-negative")

        summary = staged_summary.resident_summary
        device = summary.cluster_keys.device

        if summary.cluster_keys.shape != summary.cluster_values.shape:
            raise ValueError("Resident cluster key/value shapes must match")
        if summary.cluster_keys.ndim != 3:
            raise ValueError(
                "Resident cluster summaries must have shape "
                "[num_kv_heads, num_clusters, head_size]"
            )
        if summary.cluster_token_counts.shape != summary.cluster_keys.shape[:2]:
            raise ValueError("Resident cluster counts do not match summaries")
        if cluster_ids.shape != summary.cluster_token_counts.shape:
            raise ValueError("Resident cluster IDs do not match cluster counts")
        if cluster_page_ids.shape != cluster_page_token_counts.shape:
            raise ValueError("Resident cluster page metadata shapes must match")
        if cluster_page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Resident cluster pages do not match cluster IDs")

        cluster_page_ids_cpu = cluster_page_ids.to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        cluster_page_token_counts_cpu = cluster_page_token_counts.to(
            device="cpu", dtype=torch.int32
        ).contiguous()
        cluster_ids_cpu = cluster_ids.to(device="cpu", dtype=torch.int64)
        cluster_token_counts_cpu = staged_summary.cluster_token_counts.to(
            device="cpu", dtype=torch.int32
        )
        valid_clusters = cluster_ids_cpu >= 0
        positive_clusters = cluster_token_counts_cpu > 0
        valid_pages = cluster_page_ids_cpu >= 0
        positive_pages = cluster_page_token_counts_cpu > 0

        if not torch.equal(valid_clusters, positive_clusters):
            raise ValueError(
                "Resident cluster IDs and token counts describe different clusters"
            )
        if not torch.equal(valid_pages, positive_pages):
            raise ValueError(
                "Resident cluster page IDs and token counts describe different pages"
            )

        cluster_page_counts_cpu = (cluster_page_ids_cpu >= 0).sum(
            dim=-1, dtype=torch.int32
        )
        if not torch.equal(cluster_page_counts_cpu > 0, valid_clusters):
            raise ValueError("Resident clusters and page descriptors do not match")
        if not torch.equal(
            cluster_page_token_counts_cpu.sum(dim=-1),
            cluster_token_counts_cpu,
        ):
            raise ValueError(
                "Resident cluster token counts do not match page descriptors"
            )

        resident_token_counts = summary.cluster_token_counts.to(
            device=device, dtype=torch.int32
        ).contiguous()

        return RetroSpecResidentSegment(
            layer_name=layer_name,
            request_id=request_id,
            indexed_start=indexed_start,
            indexed_end=indexed_end,
            cluster_start=cluster_start,
            cluster_ids_cpu=cluster_ids_cpu.contiguous(),
            cluster_keys=summary.cluster_keys.contiguous(),
            cluster_values=summary.cluster_values.contiguous(),
            cluster_token_counts=resident_token_counts,
            cluster_page_ids_cpu=cluster_page_ids_cpu,
            cluster_page_token_counts_cpu=cluster_page_token_counts_cpu,
            cluster_page_counts_cpu=cluster_page_counts_cpu.contiguous(),
        )

    def _write_resident_segment(
        self,
        arena: RetroSpecResidentLayerArena,
        segment: RetroSpecResidentSegment,
        slot: int,
        previous_state: _ResidentRequestState | None,
    ) -> _ResidentRequestState:
        num_kv_heads, num_segment_clusters = segment.cluster_ids_cpu.shape
        cluster_start = segment.cluster_start
        cluster_end = cluster_start + num_segment_clusters

        if cluster_end > self.max_clusters_per_request:
            raise RuntimeError("Resident cluster arena capacity was exceeded")

        if previous_state is None:
            if cluster_start != 0:
                raise RuntimeError(
                    "The first resident segment must start at cluster zero"
                )
            page_ends = (0,) * num_kv_heads
            indexed_start = segment.indexed_start
            previous_indexed_end = segment.indexed_start
            previous_num_clusters = 0
            max_pages_per_cluster = 0
        else:
            if len(previous_state.page_ends) != num_kv_heads:
                raise RuntimeError("Resident segment changed KV-head count")
            page_ends = previous_state.page_ends
            indexed_start = previous_state.indexed_start
            previous_indexed_end = previous_state.indexed_end
            previous_num_clusters = previous_state.num_clusters
            max_pages_per_cluster = previous_state.max_pages_per_cluster

        if previous_indexed_end != segment.indexed_start:
            raise RuntimeError(
                "Resident segment does not follow the indexed token prefix"
            )
        if previous_num_clusters != cluster_start:
            raise RuntimeError("Resident segment does not follow the cluster prefix")

        next_page_ends: list[int] = []
        for head_index, page_start in enumerate(page_ends):
            page_counts = segment.cluster_page_counts_cpu[head_index]
            page_end = page_start + int(page_counts.sum().item())
            if page_end > self.max_pages_per_head_per_request:
                raise RuntimeError(
                    "Resident page-descriptor arena capacity was exceeded"
                )
            next_page_ends.append(page_end)

        arena.cluster_ids[slot, :, cluster_start:cluster_end].copy_(
            segment.cluster_ids_cpu
        )
        arena.cluster_keys[slot, :, cluster_start:cluster_end].copy_(
            segment.cluster_keys
        )
        arena.cluster_values[slot, :, cluster_start:cluster_end].copy_(
            segment.cluster_values
        )
        arena.cluster_token_counts[slot, :, cluster_start:cluster_end].copy_(
            segment.cluster_token_counts
        )

        for head_index, (page_start, page_end) in enumerate(
            zip(page_ends, next_page_ends)
        ):
            page_counts = segment.cluster_page_counts_cpu[head_index]
            page_offsets_cpu = torch.empty(num_segment_clusters + 1, dtype=torch.int64)
            page_offsets_cpu[0] = page_start
            torch.cumsum(page_counts.to(torch.int64), dim=0, out=page_offsets_cpu[1:])
            page_offsets_cpu[1:].add_(page_start)

            arena.cluster_page_offsets[
                slot, head_index, cluster_start : cluster_end + 1
            ].copy_(page_offsets_cpu, non_blocking=self.pin_memory)

            valid_pages = segment.cluster_page_ids_cpu[head_index] >= 0
            flat_page_ids = segment.cluster_page_ids_cpu[head_index].masked_select(
                valid_pages
            )
            flat_page_token_counts = segment.cluster_page_token_counts_cpu[
                head_index
            ].masked_select(valid_pages)

            arena.page_ids[slot, head_index, page_start:page_end].copy_(flat_page_ids)
            arena.page_token_counts[slot, head_index, page_start:page_end].copy_(
                flat_page_token_counts
            )

        segment_max_pages = int(segment.cluster_page_counts_cpu.max().item())
        return _ResidentRequestState(
            slot=slot,
            indexed_start=indexed_start,
            indexed_end=segment.indexed_end,
            num_clusters=cluster_end,
            page_ends=tuple(next_page_ends),
            max_pages_per_cluster=max(
                max_pages_per_cluster,
                segment_max_pages,
            ),
        )

    def publish_resident_segments(
        self,
        segments: Sequence[RetroSpecResidentSegment],
    ) -> None:
        if not segments:
            return

        incoming_request_ids = {segment.request_id for segment in segments}
        resident_request_ids = set(self.resident_request_ids)
        new_request_ids = resident_request_ids | incoming_request_ids
        if len(new_request_ids) > self.max_resident_requests:
            raise RuntimeError(
                "RetroSpec persistent GPU index residency exceeds max_num_seqs: "
                f"{len(new_request_ids)} > {self.max_resident_requests}"
            )

        projected_states = {
            layer_name: dict(layer_states)
            for layer_name, layer_states in self._resident_states.items()
        }
        changed_layers: set[str] = set()
        allocated_request_ids: list[str] = []

        try:
            for segment in segments:
                if segment.request_id not in self._request_slots:
                    allocated_request_ids.append(segment.request_id)
                slot = self._get_or_allocate_request_slot(segment.request_id)
                arena = self._get_or_create_arena(segment.layer_name, segment)
                layer_states = projected_states.setdefault(segment.layer_name, {})
                previous_state = layer_states.get(segment.request_id)
                layer_states[segment.request_id] = self._write_resident_segment(
                    arena=arena,
                    segment=segment,
                    slot=slot,
                    previous_state=previous_state,
                )
                changed_layers.add(segment.layer_name)
        except BaseException:
            for request_id in reversed(allocated_request_ids):
                slot = self._request_slots.pop(request_id, None)
                if slot is not None:
                    self._free_request_slots.append(slot)
            raise

        self._resident_states = projected_states
        for layer_name in changed_layers:
            arena = self._layer_arenas[layer_name]
            for state in self._resident_states[layer_name].values():
                arena.num_clusters[state.slot] = state.num_clusters
                arena.indexed_starts[state.slot] = state.indexed_start
                arena.indexed_ends[state.slot] = state.indexed_end
            self._entries.pop(layer_name, None)
            self._active_views.pop(layer_name, None)

    def get_active_view(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        device: torch.device,
    ) -> RetroSpecResidentBatchView:
        request_ids = tuple(request_ids)
        cache_active = self._active_request_ids is not None
        if cache_active:
            self._validate_active_requests(request_ids)

        cached = self._active_views.get(layer_name) if cache_active else None
        if cached is not None:
            return cached

        arena = self._layer_arenas.get(layer_name)
        layer_states = self._resident_states.get(layer_name, {})
        request_slots = [
            -1 if (state := layer_states.get(request_id)) is None else state.slot
            for request_id in request_ids
        ]
        request_slot_ids = torch.tensor(request_slots, dtype=torch.int64, device=device)
        max_num_clusters = max(
            (
                state.num_clusters
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )
        max_pages_per_cluster = max(
            (
                state.max_pages_per_cluster
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )

        view = RetroSpecResidentBatchView(
            arena=arena,
            request_slot_ids=request_slot_ids,
            max_num_clusters=max(max_num_clusters, 1),
            max_pages_per_cluster=max_pages_per_cluster,
        )
        if cache_active:
            self._active_views[layer_name] = view
        return view

    def get_num_clusters(self, layer_name: str, request_id: str) -> int:
        state = self._resident_states.get(layer_name, {}).get(request_id)
        return 0 if state is None else state.num_clusters

    def get_indexed_end(self, layer_name: str, request_id: str) -> int | None:
        state = self._resident_states.get(layer_name, {}).get(request_id)
        return None if state is None else state.indexed_end

    def materialize_packed(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        max_num_tokens: int,
        key_cache: torch.Tensor,
    ) -> RetroSpecResidentIndex:
        request_ids = tuple(request_ids)
        cache_active = self._active_request_ids is not None
        cached = None
        if cache_active:
            self._validate_active_requests(request_ids)
            cached = self.get(layer_name, request_ids, max_num_tokens)
        if cached is not None:
            return cached

        batch_size = len(request_ids)
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]
        view = self.get_active_view(layer_name, request_ids, key_cache.device)
        arena = view.arena
        layer_states = self._resident_states.get(layer_name, {})

        indexed_token_mask = torch.zeros(
            batch_size,
            max_num_tokens,
            dtype=torch.bool,
            device=key_cache.device,
        )
        cluster_ids = torch.full(
            (batch_size, num_kv_heads, view.max_num_clusters),
            -1,
            dtype=torch.int64,
            device=key_cache.device,
        )
        cluster_keys = torch.zeros(
            batch_size,
            num_kv_heads,
            view.max_num_clusters,
            head_size,
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
        cluster_values = torch.zeros_like(cluster_keys)
        cluster_token_counts = torch.zeros(
            cluster_ids.shape, dtype=torch.int32, device=key_cache.device
        )
        cluster_mask = torch.zeros(
            cluster_ids.shape, dtype=torch.bool, device=key_cache.device
        )

        page_shape = (
            batch_size,
            num_kv_heads,
            view.max_num_clusters,
            view.max_pages_per_cluster,
        )
        cluster_page_ids = torch.full(
            page_shape, -1, dtype=torch.int64, device=key_cache.device
        )
        cluster_page_token_counts = torch.zeros(
            page_shape, dtype=torch.int32, device=key_cache.device
        )

        if arena is not None:
            if arena.cluster_keys.device != key_cache.device:
                raise RuntimeError(
                    "Resident RetroSpec arena and attention KV use different devices"
                )
            if arena.cluster_keys.dtype != key_cache.dtype:
                raise RuntimeError(
                    "Resident RetroSpec arena and attention KV use different dtypes"
                )
            if arena.cluster_keys.shape[1] != num_kv_heads:
                raise RuntimeError("Resident RetroSpec arena changed KV-head count")
            if arena.cluster_keys.shape[3] != head_size:
                raise RuntimeError("Resident RetroSpec arena changed head size")

            page_offsets = torch.arange(
                view.max_pages_per_cluster,
                dtype=torch.int64,
                device=key_cache.device,
            )

            for row, request_id in enumerate(request_ids):
                state = layer_states.get(request_id)
                if state is None:
                    continue

                slot = state.slot
                num_clusters = state.num_clusters
                indexed_end = min(state.indexed_end, max_num_tokens)
                if indexed_end > state.indexed_start:
                    indexed_token_mask[row, state.indexed_start : indexed_end] = True

                cluster_ids[row, :, :num_clusters].copy_(
                    arena.cluster_ids[slot, :, :num_clusters]
                )
                cluster_keys[row, :, :num_clusters].copy_(
                    arena.cluster_keys[slot, :, :num_clusters]
                )
                cluster_values[row, :, :num_clusters].copy_(
                    arena.cluster_values[slot, :, :num_clusters]
                )
                cluster_token_counts[row, :, :num_clusters].copy_(
                    arena.cluster_token_counts[slot, :, :num_clusters]
                )

                request_cluster_ids = cluster_ids[row, :, :num_clusters]
                request_cluster_counts = cluster_token_counts[row, :, :num_clusters]
                cluster_mask[row, :, :num_clusters].copy_(
                    (request_cluster_ids >= 0) & (request_cluster_counts > 0)
                )

                if view.max_pages_per_cluster == 0:
                    continue

                page_starts = arena.cluster_page_offsets[slot, :, :num_clusters]
                page_ends = arena.cluster_page_offsets[slot, :, 1 : num_clusters + 1]
                page_positions = page_starts.unsqueeze(-1) + page_offsets
                valid_pages = page_positions < page_ends.unsqueeze(-1)
                safe_positions = page_positions.clamp(
                    min=0,
                    max=self.max_pages_per_head_per_request - 1,
                )

                expanded_page_ids = (
                    arena.page_ids[slot].unsqueeze(1).expand(-1, num_clusters, -1)
                )
                expanded_page_counts = (
                    arena.page_token_counts[slot]
                    .unsqueeze(1)
                    .expand(-1, num_clusters, -1)
                )
                selected_page_ids = expanded_page_ids.gather(2, safe_positions)
                selected_page_counts = expanded_page_counts.gather(2, safe_positions)

                cluster_page_ids[row, :, :num_clusters].copy_(
                    selected_page_ids.masked_fill(~valid_pages, -1)
                )
                cluster_page_token_counts[row, :, :num_clusters].copy_(
                    selected_page_counts.masked_fill(~valid_pages, 0)
                )

        packed = RetroSpecResidentIndex(
            indexed_token_mask=indexed_token_mask,
            cluster_ids=cluster_ids,
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            cluster_token_counts=cluster_token_counts,
            cluster_mask=cluster_mask,
            cluster_page_ids=cluster_page_ids,
            cluster_page_token_counts=cluster_page_token_counts,
        )
        if cache_active:
            self.put(layer_name, request_ids, max_num_tokens, packed)
        return packed

    def invalidate_packed_layer(self, layer_name: str) -> None:
        self._entries.pop(layer_name, None)
        self._active_views.pop(layer_name, None)

    def discard_request_layer(self, layer_name: str, request_id: str) -> None:
        layer_states = self._resident_states.get(layer_name)
        if layer_states is None:
            return

        state = layer_states.pop(request_id, None)
        if state is None:
            return

        arena = self._layer_arenas[layer_name]
        arena.num_clusters[state.slot] = 0
        arena.indexed_starts[state.slot] = 0
        arena.indexed_ends[state.slot] = 0
        self._entries.pop(layer_name, None)
        self._active_views.pop(layer_name, None)

    def invalidate_requests(self, request_ids: Sequence[str]) -> None:
        removed = set(request_ids)
        if not removed:
            return

        for layer_name, layer_states in self._resident_states.items():
            arena = self._layer_arenas[layer_name]
            for request_id in removed:
                state = layer_states.pop(request_id, None)
                if state is None:
                    continue
                arena.num_clusters[state.slot] = 0
                arena.indexed_starts[state.slot] = 0
                arena.indexed_ends[state.slot] = 0

            self._entries.pop(layer_name, None)

        self._active_views.clear()

        for request_id in removed:
            slot = self._request_slots.pop(request_id, None)
            if slot is not None:
                self._free_request_slots.append(slot)

    def _get_offload_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        stream = self._offload_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._offload_streams[device] = stream

        return stream

    def stage_cluster_summary(
        self,
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        cluster_token_counts: torch.Tensor,
    ) -> RetroSpecStagedClusterSummary:
        if cluster_keys.shape != cluster_values.shape:
            raise ValueError("Cluster key/value summary shapes must match")
        if cluster_keys.ndim != 3:
            raise ValueError(
                "Cluster summaries must have shape "
                "[num_kv_heads, num_clusters, head_size]"
            )
        if cluster_token_counts.shape != cluster_keys.shape[:2]:
            raise ValueError("Cluster token counts do not match cluster summaries")

        resident_summary = RetroSpecClusterSummary(
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            cluster_token_counts=cluster_token_counts,
        )

        host_keys = torch.empty_like(
            cluster_keys, device="cpu", pin_memory=self.pin_memory
        )
        host_values = torch.empty_like(
            cluster_values, device="cpu", pin_memory=self.pin_memory
        )
        host_counts = torch.empty_like(
            cluster_token_counts, device="cpu", pin_memory=self.pin_memory
        )

        if cluster_keys.device.type != "cuda":
            host_keys.copy_(cluster_keys)
            host_values.copy_(cluster_values)
            host_counts.copy_(cluster_token_counts)
            return RetroSpecStagedClusterSummary(
                cluster_keys=host_keys,
                cluster_values=host_values,
                cluster_token_counts=host_counts,
                resident_summary=resident_summary,
                ready_event=None,
            )

        if not self.pin_memory:
            host_keys.copy_(cluster_keys, non_blocking=False)
            host_values.copy_(cluster_values, non_blocking=False)
            host_counts.copy_(cluster_token_counts, non_blocking=False)
            return RetroSpecStagedClusterSummary(
                cluster_keys=host_keys,
                cluster_values=host_values,
                cluster_token_counts=host_counts,
                resident_summary=resident_summary,
                ready_event=None,
            )

        device = cluster_keys.device
        transfer_stream = self._get_offload_stream(device)
        current_stream = torch.cuda.current_stream(device)
        transfer_stream.wait_stream(current_stream)

        with torch.cuda.stream(transfer_stream):
            host_keys.copy_(cluster_keys, non_blocking=True)
            host_values.copy_(cluster_values, non_blocking=True)
            host_counts.copy_(cluster_token_counts, non_blocking=True)
            ready_event = torch.cuda.Event()
            ready_event.record(transfer_stream)

        return RetroSpecStagedClusterSummary(
            cluster_keys=host_keys,
            cluster_values=host_values,
            cluster_token_counts=host_counts,
            resident_summary=resident_summary,
            ready_event=ready_event,
        )

    @staticmethod
    def finish_cluster_summary(
        staged: RetroSpecStagedClusterSummary,
    ) -> RetroSpecClusterSummary:
        if staged.ready_event is not None:
            staged.ready_event.synchronize()

        return RetroSpecClusterSummary(
            cluster_keys=staged.cluster_keys,
            cluster_values=staged.cluster_values,
            cluster_token_counts=staged.cluster_token_counts,
        )

    @staticmethod
    def discard_cluster_summary(staged: RetroSpecStagedClusterSummary) -> None:
        if staged.ready_event is not None:
            staged.ready_event.synchronize()
