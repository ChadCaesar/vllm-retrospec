# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Lock, RLock

import torch

from vllm.v1.spec_decode.retrospec.legacy.residency.index.arena import (
    _IndexResidencyArenaMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.index.segments import (
    _IndexResidencySegmentsMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.index.staging import (
    _IndexResidencyStagingMixin,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.index.types import (
    RetroSpecClusterSummary,
    RetroSpecResidentBatchView,
    RetroSpecResidentLayerArena,
    RetroSpecResidentSegment,
    RetroSpecResidentTableBinding,
    RetroSpecStagedClusterSummary,
    _FreeSpanAllocator,
    _PackedLayerState,
    _PinnedSummarySlot,
    _ResidentRequestState,
    _ResidentSpanTransaction,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.index.views import (
    _IndexResidencyViewsMixin,
)

from ..performance import RetroSpecPerformanceStats
from .pinned_memory import RetroSpecPinnedMemoryManager

__all__ = [
    "RetroSpecClusterSummary",
    "_PinnedSummarySlot",
    "RetroSpecStagedClusterSummary",
    "RetroSpecResidentSegment",
    "RetroSpecResidentTableBinding",
    "RetroSpecResidentLayerArena",
    "RetroSpecResidentBatchView",
    "_ResidentRequestState",
    "_FreeSpanAllocator",
    "_PackedLayerState",
    "_ResidentSpanTransaction",
    "RetroSpecGPUIndexResidencyManager",
]


class RetroSpecGPUIndexResidencyManager(
    _IndexResidencyArenaMixin,
    _IndexResidencySegmentsMixin,
    _IndexResidencyViewsMixin,
    _IndexResidencyStagingMixin,
):
    """Own request-slot layer arenas and active batch descriptors."""

    def __init__(
        self,
        pin_memory: bool = False,
        max_resident_requests: int = 1,
        max_gpu_index_memory_bytes: int = 4 << 30,
        pinned_memory: RetroSpecPinnedMemoryManager | None = None,
        max_summary_slots: int = 2,
        performance_stats: RetroSpecPerformanceStats | None = None,
    ) -> None:
        if max_summary_slots <= 0:
            raise ValueError("max_summary_slots must be positive")
        if max_resident_requests <= 0:
            raise ValueError("max_resident_requests must be positive")
        if max_gpu_index_memory_bytes <= 0:
            raise ValueError("max_gpu_index_memory_bytes must be positive")

        if pinned_memory is None:
            pinned_memory = RetroSpecPinnedMemoryManager(
                enabled=pin_memory,
                max_bytes=64 << 20,
            )
        elif pin_memory and not pinned_memory.enabled:
            raise ValueError("pin_memory conflicts with the shared pinned manager")

        self._pinned_memory = pinned_memory
        self.pin_memory = pinned_memory.enabled
        self.max_summary_slots = max_summary_slots
        self.max_resident_requests = max_resident_requests
        self.max_gpu_index_memory_bytes = max_gpu_index_memory_bytes
        self.performance_stats = performance_stats
        self._allocated_gpu_index_bytes = 0

        self._active_request_ids: tuple[str, ...] | None = None
        self._active_views: dict[str, RetroSpecResidentBatchView] = {}

        self._request_slots: dict[str, int] = {}
        self._slot_generations = [0] * max_resident_requests
        self._free_request_slots = list(reversed(range(max_resident_requests)))

        self._layer_arenas: dict[str, _PackedLayerState] = {}
        self._resident_states: dict[
            str,
            dict[str, _ResidentRequestState],
        ] = {}
        self._offload_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._summary_slots: dict[torch.device, list[_PinnedSummarySlot]] = {}
        self._summary_slot_lock = Lock()
        self._arena_lock = RLock()
        self._arena_ready_events: dict[str, torch.cuda.Event] = {}
        self._binding_update_events: dict[str, torch.cuda.Event] = {}
