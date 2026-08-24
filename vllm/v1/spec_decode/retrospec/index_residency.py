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
    layer_name: str
    request_id: str

    indexed_start: int
    indexed_end: int
    cluster_start: int

    cluster_ids: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_mask: torch.Tensor

    cluster_page_ids: torch.Tensor
    cluster_page_token_counts: torch.Tensor


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
class _ResidentIndexEntry:
    request_ids: tuple[str, ...]
    max_num_tokens: int
    index: RetroSpecResidentIndex


class RetroSpecGPUIndexResidencyManager:
    """Own persistent request indices and temporary packed batch views.

    Authoritative cluster summaries remain on CPU. Once a cluster segment has
    been published, the summary and page metadata needed
    for coarse scoring stay on the execution device until the request is
    removed or the corresponding layer is rolled back. Proposal and full
    verification contexts may additionally cache one packed batch view.
    """

    def __init__(
        self,
        pin_memory: bool,
        max_resident_requests: int,
    ) -> None:
        if max_resident_requests <= 0:
            raise ValueError("max_resident_requests must be positive")

        self.pin_memory = pin_memory
        self.max_resident_requests = max_resident_requests

        self._active_request_ids: tuple[str, ...] | None = None
        self._entries: dict[str, _ResidentIndexEntry] = {}
        self._resident_segments: dict[
            str,
            dict[str, tuple[RetroSpecResidentSegment, ...]],
        ] = {}
        self._offload_streams: dict[torch.device, torch.cuda.Stream] = {}

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return () if self._active_request_ids is None else self._active_request_ids

    @property
    def resident_request_ids(self) -> tuple[str, ...]:
        request_ids = {
            request_id
            for layer_segments in self._resident_segments.values()
            for request_id in layer_segments
        }
        return tuple(sorted(request_ids))

    @property
    def num_resident_requests(self) -> int:
        return len(self.resident_request_ids)

    @property
    def num_resident_layers(self) -> int:
        return len(self._resident_segments)

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
        self._active_request_ids = request_ids

    def deactivate(self) -> None:
        if self._active_request_ids is None:
            raise RuntimeError("No RetroSpec GPU index residency set is active")

        self._entries.clear()
        self._active_request_ids = None

    def _validate_active_requests(self, request_ids: tuple[str, ...]) -> None:
        if self._active_request_ids is None:
            raise RuntimeError(
                "CPU-backed RetroSpec indices may be packed only inside an "
                "active proposal or full-verification context"
            )
        if request_ids != self._active_request_ids:
            raise RuntimeError(
                "Packed RetroSpec request order does not match the active "
                "GPU index residency set"
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

    def get_resident_segments(
        self,
        layer_name: str,
        request_id: str,
    ) -> tuple[RetroSpecResidentSegment, ...]:
        return self._resident_segments.get(layer_name, {}).get(request_id, ())

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

        resident_cluster_ids = cluster_ids.to(
            device=device, dtype=torch.int64, non_blocking=False
        ).contiguous()
        resident_page_ids = cluster_page_ids.to(
            device=device, dtype=torch.int64, non_blocking=False
        ).contiguous()
        resident_page_token_counts = cluster_page_token_counts.to(
            device=device, dtype=torch.int32, non_blocking=False
        ).contiguous()
        resident_token_counts = summary.cluster_token_counts.to(
            device=device, dtype=torch.int32
        ).contiguous()

        return RetroSpecResidentSegment(
            layer_name=layer_name,
            request_id=request_id,
            indexed_start=indexed_start,
            indexed_end=indexed_end,
            cluster_start=cluster_start,
            cluster_ids=resident_cluster_ids,
            cluster_keys=summary.cluster_keys.contiguous(),
            cluster_values=summary.cluster_values.contiguous(),
            cluster_token_counts=resident_token_counts,
            cluster_mask=(
                (resident_cluster_ids >= 0) & (resident_token_counts > 0)
            ).contiguous(),
            cluster_page_ids=resident_page_ids,
            cluster_page_token_counts=resident_page_token_counts,
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

        new_resident_segments = {
            layer_name: dict(layer_segments)
            for layer_name, layer_segments in self._resident_segments.items()
        }
        changed_layers: set[str] = set()

        for segment in segments:
            layer_segments = new_resident_segments.setdefault(segment.layer_name, {})
            current_segments = list(layer_segments.get(segment.request_id, ()))

            if current_segments:
                previous = current_segments[-1]
                previous_cluster_end = (
                    previous.cluster_start + previous.cluster_keys.shape[1]
                )
                if previous.indexed_end != segment.indexed_start:
                    raise RuntimeError(
                        "Resident segment does not follow the indexed token prefix"
                    )
                if previous_cluster_end != segment.cluster_start:
                    raise RuntimeError(
                        "Resident segment does not follow the cluster prefix"
                    )
            elif segment.cluster_start != 0:
                raise RuntimeError(
                    "The first resident segment must start at cluster zero"
                )

            current_segments.append(segment)
            layer_segments[segment.request_id] = tuple(current_segments)
            changed_layers.add(segment.layer_name)

        self._resident_segments = new_resident_segments
        for layer_name in changed_layers:
            self._entries.pop(layer_name, None)

    def invalidate_packed_layer(self, layer_name: str) -> None:
        self._entries.pop(layer_name, None)

    def discard_request_layer(self, layer_name: str, request_id: str) -> None:
        layer_segments = self._resident_segments.get(layer_name)
        if layer_segments is not None:
            layer_segments.pop(request_id, None)
            if not layer_segments:
                del self._resident_segments[layer_name]

        self._entries.pop(layer_name, None)

    def invalidate_requests(self, request_ids: Sequence[str]) -> None:
        removed = set(request_ids)
        if not removed:
            return

        for layer_name in tuple(self._resident_segments):
            layer_segments = self._resident_segments[layer_name]
            for request_id in removed:
                layer_segments.pop(request_id, None)
            if not layer_segments:
                del self._resident_segments[layer_name]

        invalid_layers = [
            layer_name
            for layer_name, entry in self._entries.items()
            if removed.intersection(entry.request_ids)
        ]
        for layer_name in invalid_layers:
            del self._entries[layer_name]

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
