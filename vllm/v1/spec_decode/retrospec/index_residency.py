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
    ready_event: torch.cuda.Event | None
    source_tensors: tuple[torch.Tensor, ...]


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
    """Own the packed index for one active RetroSpec request set.

    CPU-offload mode keeps authoritative cluster summaries on CPU and packs
    only the requests participating in the current proposal or full
    verification onto GPU. GPU-reference mode retains the previous cache
    behavior because its summaries are GPU-resident by definition.
    """

    def __init__(
        self,
        cpu_offload: bool,
        pin_memory: bool,
        max_resident_requests: int,
    ) -> None:
        if max_resident_requests <= 0:
            raise ValueError("max_resident_requests must be positive")

        self.cpu_offload = cpu_offload
        self.pin_memory = pin_memory if cpu_offload else False
        self.max_resident_requests = max_resident_requests

        self._active_request_ids: tuple[str, ...] | None = None
        self._entries: dict[str, _ResidentIndexEntry] = {}
        self._offload_streams: dict[torch.device, torch.cuda.Stream] = {}

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return () if self._active_request_ids is None else self._active_request_ids

    @property
    def num_resident_layers(self) -> int:
        return len(self._entries)

    def activate(self, request_ids: Sequence[str]) -> None:
        if not self.cpu_offload:
            return
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
        if not self.cpu_offload:
            return
        if self._active_request_ids is None:
            raise RuntimeError("No RetroSpec GPU index residency set is active")

        self._entries.clear()
        self._active_request_ids = None

    def _validate_active_requests(self, request_ids: tuple[str, ...]) -> None:
        if not self.cpu_offload:
            return
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

    def invalidate_layer(self, layer_name: str) -> None:
        self._entries.pop(layer_name, None)

    def invalidate_requests(self, request_ids: Sequence[str]) -> None:
        removed = set(request_ids)
        if not removed:
            return

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

        if not self.cpu_offload:
            return RetroSpecStagedClusterSummary(
                cluster_keys=cluster_keys,
                cluster_values=cluster_values,
                cluster_token_counts=cluster_token_counts,
                ready_event=None,
                source_tensors=(),
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
                ready_event=None,
                source_tensors=(),
            )

        if not self.pin_memory:
            host_keys.copy_(cluster_keys, non_blocking=False)
            host_values.copy_(cluster_values, non_blocking=False)
            host_counts.copy_(cluster_token_counts, non_blocking=False)
            return RetroSpecStagedClusterSummary(
                cluster_keys=host_keys,
                cluster_values=host_values,
                cluster_token_counts=host_counts,
                ready_event=None,
                source_tensors=(),
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
            ready_event=ready_event,
            source_tensors=(cluster_keys, cluster_values, cluster_token_counts),
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
