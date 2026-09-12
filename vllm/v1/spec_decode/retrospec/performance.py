# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from time import monotonic, perf_counter
from typing import Literal

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

RetroSpecCudaTimingLevel = Literal["coarse", "detailed"]

_CUDA_EVENT_POOL_SIZE = 128
_COARSE_CUDA_TIMER_NAMES = frozenset(
    {
        "draft_model",
        "sparse_verify_model",
        "expanded_verify_model",
        "full_verify_transaction",
    }
)


@dataclass(frozen=True)
class _CudaTimer:
    name: str
    pair_index: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    sample_weight: int


@dataclass(frozen=True)
class _PendingCudaSample:
    name: str
    pair_index: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    sample_weight: int


class RetroSpecPerformanceStats:
    """Low-overhead, opt-in RetroSpec performance statistics.

    CPU counters may be updated from background workers. Request-stage counters
    remain on the model device and are copied to the CPU only once per logging
    interval. CUDA timings use events and do not synchronize when recorded.
    """

    _GPU_COUNTER_NAMES = (
        "proposal_requests",
        "draft_round_requests",
        "draft_tokens",
        "verified_tokens",
        "proposed_tokens",
        "resident_cluster_hits",
        "resident_cluster_misses",
        "draft_compact_resident_pages",
        "draft_compact_selected_clusters",
        "verification_lookup_clusters",
        "verification_resident_hits",
        "verification_resident_misses",
    )
    _GPU_HISTOGRAM_NAMES = (
        "draft_to_sparse_tokens",
        "sparse_to_expanded_prefix",
        "expanded_to_full_prefix",
    )

    def __init__(
        self,
        device: torch.device,
        log_interval_seconds: float,
        histogram_max_value: int = 0,
        cuda_timing_level: RetroSpecCudaTimingLevel = "coarse",
        cuda_sample_interval: int = 8,
    ) -> None:
        if log_interval_seconds < 0:
            raise ValueError("RetroSpec stats interval must be non-negative")
        if histogram_max_value < 0:
            raise ValueError("RetroSpec histogram maximum must be non-negative")
        if cuda_timing_level not in ("coarse", "detailed"):
            raise ValueError("RetroSpec CUDA timing level must be coarse or detailed")
        if cuda_sample_interval <= 0:
            raise ValueError("RetroSpec CUDA sample interval must be positive")

        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.log_interval_seconds = log_interval_seconds
        self.enabled = log_interval_seconds > 0
        self.histogram_max_value = histogram_max_value
        self._histogram_num_bins = histogram_max_value + 1
        self.cuda_timing_level = cuda_timing_level
        self.cuda_sample_interval = cuda_sample_interval

        self._gpu_counter_indices = {
            name: index for index, name in enumerate(self._GPU_COUNTER_NAMES)
        }
        num_counters = len(self._GPU_COUNTER_NAMES)
        num_histogram_values = len(self._GPU_HISTOGRAM_NAMES) * self._histogram_num_bins
        self._gpu_observations = (
            torch.zeros(
                num_counters + num_histogram_values,
                dtype=torch.int64,
                device=device,
            )
            if self.enabled
            else torch.empty(0, dtype=torch.int64)
        )
        self._gpu_counters = self._gpu_observations[:num_counters]
        histogram_values = self._gpu_observations[num_counters:]
        self._gpu_histograms = {
            name: histogram_values[
                index * self._histogram_num_bins : (index + 1)
                * self._histogram_num_bins
            ]
            for index, name in enumerate(self._GPU_HISTOGRAM_NAMES)
        }

        self._lock = Lock()
        self._cpu_counters: Counter[str] = Counter()
        self._peaks: dict[str, int] = {}
        self._cpu_times: dict[str, tuple[float, int]] = {}
        self._cuda_times: dict[str, tuple[float, int]] = {}
        self._cuda_timer_call_counts: Counter[str] = Counter()
        self._pending_cuda_samples: deque[_PendingCudaSample] = deque()
        self._cuda_event_pairs = (
            tuple(
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(_CUDA_EVENT_POOL_SIZE)
            )
            if self.enabled and device.type == "cuda"
            else ()
        )
        self._free_cuda_event_pairs = deque(range(len(self._cuda_event_pairs)))
        self._last_log_time = monotonic()

    def add_counter(self, name: str, value: int = 1) -> None:
        if not self.enabled or value == 0:
            return
        with self._lock:
            self._cpu_counters[name] += int(value)

    def add_gpu_counter(self, name: str, value: int | torch.Tensor) -> None:
        if not self.enabled:
            return

        counter_index = self._gpu_counter_indices.get(name)
        if counter_index is None:
            raise KeyError(f"Unknown RetroSpec GPU counter: {name}")

        counter = self._gpu_counters[counter_index]
        if isinstance(value, torch.Tensor):
            if value.device != self.device:
                raise ValueError(
                    f"RetroSpec GPU counter {name!r} must use device {self.device}"
                )
            counter.add_(value.sum(dtype=torch.int64))
        else:
            counter.add_(int(value))

    def add_gpu_histogram(
        self,
        name: str,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return

        histogram = self._gpu_histograms.get(name)
        if histogram is None:
            raise KeyError(f"Unknown RetroSpec GPU histogram: {name}")
        if values.device != self.device:
            raise ValueError(
                f"RetroSpec GPU histogram {name!r} must use device {self.device}"
            )
        values = values.reshape(-1)
        if mask is None:
            weights = torch.ones_like(values, dtype=torch.int64)
        else:
            if mask.device != self.device:
                raise ValueError(
                    f"RetroSpec GPU histogram mask {name!r} must use device "
                    f"{self.device}"
                )
            if mask.numel() != values.numel():
                raise ValueError("RetroSpec histogram values and mask must match")
            weights = mask.reshape(-1).to(dtype=torch.int64)
        bins = values.to(dtype=torch.int64).clamp_(0, self.histogram_max_value)
        histogram.scatter_add_(0, bins, weights)

    def observe_peak(self, name: str, value: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._peaks[name] = max(self._peaks.get(name, 0), int(value))

    def record_cpu_time(self, name: str, elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        self._record_time(self._cpu_times, name, elapsed_seconds * 1000.0)

    def _cuda_timer_sample_weight(self, name: str) -> int:
        if name in _COARSE_CUDA_TIMER_NAMES:
            return 1
        if self.cuda_timing_level != "detailed":
            return 0

        with self._lock:
            self._cuda_timer_call_counts[name] += 1
            call_count = self._cuda_timer_call_counts[name]

        if call_count % self.cuda_sample_interval != 0:
            return 0
        return self.cuda_sample_interval

    def _take_cuda_event_pair(
        self,
    ) -> tuple[int, torch.cuda.Event, torch.cuda.Event] | None:
        with self._lock:
            if self._free_cuda_event_pairs:
                pair_index = self._free_cuda_event_pairs.popleft()
                start_event, end_event = self._cuda_event_pairs[pair_index]
                return pair_index, start_event, end_event

        self._drain_cuda_samples(wait_for_completion=False)

        with self._lock:
            if self._free_cuda_event_pairs:
                pair_index = self._free_cuda_event_pairs.popleft()
                start_event, end_event = self._cuda_event_pairs[pair_index]
                return pair_index, start_event, end_event

            self._cpu_counters["cuda_timer_pool_exhausted"] += 1
        return None

    @contextmanager
    def cpu_timer(self, name: str) -> Iterator[None]:
        started_at = perf_counter() if self.enabled else None
        try:
            yield
        finally:
            if started_at is not None:
                self.record_cpu_time(name, perf_counter() - started_at)

    @contextmanager
    def cuda_timer(
        self,
        name: str,
        stream: torch.cuda.Stream | None = None,
    ) -> Iterator[None]:
        timer = self.start_cuda_timer(name, stream)
        if timer is not None:
            torch.cuda.nvtx.range_push(f"retrospec::{name}")

        try:
            yield
        finally:
            if timer is not None:
                try:
                    self.stop_cuda_timer(timer, stream)
                finally:
                    torch.cuda.nvtx.range_pop()

    def start_cuda_timer(
        self,
        name: str,
        stream: torch.cuda.Stream | None = None,
    ) -> _CudaTimer | None:
        if not self.enabled or self.device.type != "cuda":
            return None

        sample_weight = self._cuda_timer_sample_weight(name)
        if sample_weight == 0:
            return None

        event_pair = self._take_cuda_event_pair()
        if event_pair is None:
            return None

        pair_index, start_event, end_event = event_pair
        start_event.record(stream)
        return _CudaTimer(
            name=name,
            pair_index=pair_index,
            start_event=start_event,
            end_event=end_event,
            sample_weight=sample_weight,
        )

    def stop_cuda_timer(
        self,
        timer: _CudaTimer | None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if timer is None:
            return

        timer.end_event.record(stream)
        with self._lock:
            self._pending_cuda_samples.append(
                _PendingCudaSample(
                    name=timer.name,
                    pair_index=timer.pair_index,
                    start_event=timer.start_event,
                    end_event=timer.end_event,
                    sample_weight=timer.sample_weight,
                )
            )

    def _record_time(
        self,
        target: dict[str, tuple[float, int]],
        name: str,
        elapsed_ms: float,
    ) -> None:
        with self._lock:
            total_ms, count = target.get(name, (0.0, 0))
            target[name] = (total_ms + elapsed_ms, count + 1)

    def _drain_cuda_samples(self, wait_for_completion: bool = False) -> None:
        with self._lock:
            pending = tuple(self._pending_cuda_samples)
            self._pending_cuda_samples.clear()

        retained: list[_PendingCudaSample] = []
        completed: list[tuple[str, float, int]] = []
        released_pair_indices: list[int] = []
        for sample in pending:
            if wait_for_completion:
                sample.end_event.synchronize()
            elif not sample.end_event.query():
                retained.append(sample)
                continue
            completed.append(
                (
                    sample.name,
                    sample.start_event.elapsed_time(sample.end_event),
                    sample.sample_weight,
                )
            )
            released_pair_indices.append(sample.pair_index)

        with self._lock:
            self._pending_cuda_samples.extend(retained)
            self._free_cuda_event_pairs.extend(released_pair_indices)
            for name, elapsed_ms, sample_weight in completed:
                total_ms, count = self._cuda_times.get(name, (0.0, 0))
                self._cuda_times[name] = (
                    total_ms + elapsed_ms * sample_weight,
                    count + sample_weight,
                )

    @staticmethod
    def _format_counters(counters: dict[str, int]) -> str:
        if not counters:
            return "none"
        return ", ".join(f"{name}={value}" for name, value in sorted(counters.items()))

    @staticmethod
    def _format_times(times: dict[str, tuple[float, int]]) -> str:
        if not times:
            return "none"

        parts = []
        for name, (total_ms, count) in sorted(times.items()):
            average_ms = total_ms / count if count else 0.0
            parts.append(f"{name}={average_ms:.3f}ms/{count}")
        return ", ".join(parts)

    @staticmethod
    def _format_histograms(histograms: dict[str, list[int]]) -> str:
        if not histograms:
            return "none"

        parts = []
        for name, bins in sorted(histograms.items()):
            populated = ",".join(
                f"{value}:{count}" for value, count in enumerate(bins) if count
            )
            parts.append(f"{name}=[{populated}]")
        return ", ".join(parts)

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def maybe_log(self) -> None:
        if not self.enabled:
            return
        self._drain_cuda_samples(wait_for_completion=False)
        self._log(force=False, wait_for_cuda=False, reason="interval")

    def flush(self, reason: str) -> None:
        if not self.enabled:
            return
        self._log(force=True, wait_for_cuda=True, reason=reason)

    def _log(self, force: bool, wait_for_cuda: bool, reason: str) -> None:
        if not self.enabled:
            return

        now = monotonic()
        elapsed_seconds = now - self._last_log_time
        if not force and elapsed_seconds < self.log_interval_seconds:
            return

        # This is the only periodic device-to-host synchronization introduced
        # by RetroSpec performance observation.
        gpu_values = self._gpu_observations.detach().cpu().tolist()
        self._gpu_observations.zero_()
        num_counters = len(self._GPU_COUNTER_NAMES)
        gpu_counter_values = gpu_values[:num_counters]
        gpu_histogram_values = gpu_values[num_counters:]
        histograms = {
            name: gpu_histogram_values[
                index * self._histogram_num_bins : (index + 1)
                * self._histogram_num_bins
            ]
            for index, name in enumerate(self._GPU_HISTOGRAM_NAMES)
        }

        # The counter synchronization completes main-stream CUDA timers. Timers
        # from transfer streams remain queued until their events finish.
        self._drain_cuda_samples(wait_for_completion=wait_for_cuda)

        with self._lock:
            counters = dict(self._cpu_counters)
            counters.update(
                {
                    name: int(value)
                    for name, value in zip(self._GPU_COUNTER_NAMES, gpu_counter_values)
                }
            )
            peaks = dict(self._peaks)
            cpu_times = dict(self._cpu_times)
            cuda_times = dict(self._cuda_times)

            self._cpu_counters.clear()
            self._peaks.clear()
            self._cpu_times.clear()
            self._cuda_times.clear()

        self._last_log_time = now

        proposal_requests = counters.get("proposal_requests", 0)
        draft_tokens = counters.get("draft_tokens", 0)
        sparse_tokens = counters.get("sparse_verify_tokens", 0)
        expanded_tokens = counters.get("expanded_verify_tokens", 0)
        full_requests = counters.get("full_verify_requests", 0)
        resident_hits = counters.get("resident_cluster_hits", 0)
        resident_misses = counters.get("resident_cluster_misses", 0)
        verification_hits = counters.get("verification_resident_hits", 0)
        verification_misses = counters.get("verification_resident_misses", 0)
        prefetch_waves = counters.get("prefetch_waves_submitted", 0)
        prefetch_coalesced = counters.get("prefetch_waves_coalesced", 0)
        prefetch_backpressured = counters.get("prefetch_backpressure_waits", 0)
        prefetch_wave_opportunities = prefetch_waves + prefetch_coalesced
        terminal_proposed = counters.get("terminal_proposal_tokens", 0)
        terminal_wasted = counters.get("terminal_wasted_proposal_tokens", 0)

        def cudagraph_replay_rate(stage_name: str) -> float:
            replay = counters.get(f"{stage_name}_cudagraph_replay", 0)
            fallback = counters.get(f"{stage_name}_cudagraph_fallback", 0)
            eager = counters.get(f"{stage_name}_cudagraph_eager", 0)
            return self._ratio(replay, replay + fallback + eager)

        logger.info(
            "RetroSpec performance over %.2fs (reason=%s): counters={%s}; "
            "peaks={%s}; histograms={%s}; "
            "derived={draft_tokens/request=%.2f, expanded/sparse=%.3f, "
            "full/request=%.3f, resident_hit_rate=%.3f, "
            "verification_hit_rate=%.3f, "
            "prefetch_coalesce_rate=%.3f, "
            "prefetch_backpressure_rate=%.3f, "
            "prefetch_records/wave=%.2f, "
            "terminal_waste_rate=%.3f, "
            "draft_graph_replay=%.3f, "
            "sparse_verify_graph_replay=%.3f, "
            "expanded_verify_graph_replay=%.3f}; "
            "cpu_avg={%s}; cuda_avg={%s}",
            elapsed_seconds,
            reason,
            self._format_counters(counters),
            self._format_counters(peaks),
            self._format_histograms(histograms),
            self._ratio(draft_tokens, proposal_requests),
            self._ratio(expanded_tokens, sparse_tokens),
            self._ratio(full_requests, proposal_requests),
            self._ratio(resident_hits, resident_hits + resident_misses),
            self._ratio(
                verification_hits,
                verification_hits + verification_misses,
            ),
            self._ratio(
                prefetch_coalesced,
                prefetch_wave_opportunities,
            ),
            self._ratio(prefetch_backpressured, prefetch_waves),
            self._ratio(counters.get("prefetch_wave_records", 0), prefetch_waves),
            self._ratio(terminal_wasted, terminal_proposed),
            cudagraph_replay_rate("draft"),
            cudagraph_replay_rate("sparse_verify"),
            cudagraph_replay_rate("expanded_verify"),
            self._format_times(cpu_times),
            self._format_times(cuda_times),
        )
