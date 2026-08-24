# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class _CudaTimer:
    name: str
    start_event: torch.cuda.Event


@dataclass(frozen=True)
class _PendingCudaSample:
    name: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


class RetroSpecPerformanceStats:
    """Low-overhead, opt-in RetroSpec performance statistics.

    CPU counters may be updated from background workers. Request-stage counters
    remain on the model device and are copied to the CPU only once per logging
    interval. CUDA timings use events and do not synchronize when recorded.
    """

    _GPU_COUNTER_NAMES = (
        "draft_round_requests",
        "draft_tokens",
        "verified_tokens",
        "proposed_tokens",
    )

    def __init__(
        self,
        device: torch.device,
        log_interval_seconds: float,
    ) -> None:
        if log_interval_seconds < 0:
            raise ValueError("RetroSpec stats interval must be non-negative")

        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.log_interval_seconds = log_interval_seconds
        self.enabled = log_interval_seconds > 0

        self._gpu_counter_indices = {
            name: index for index, name in enumerate(self._GPU_COUNTER_NAMES)
        }
        self._gpu_counters = (
            torch.zeros(len(self._GPU_COUNTER_NAMES), dtype=torch.int64, device=device)
            if self.enabled
            else torch.empty(0, dtype=torch.int64)
        )

        self._lock = Lock()
        self._cpu_counters: Counter[str] = Counter()
        self._peaks: dict[str, int] = {}
        self._cpu_times: dict[str, tuple[float, int]] = {}
        self._cuda_times: dict[str, tuple[float, int]] = {}
        self._pending_cuda_samples: deque[_PendingCudaSample] = deque()
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

    def observe_peak(self, name: str, value: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._peaks[name] = max(self._peaks.get(name, 0), int(value))

    def record_cpu_time(self, name: str, elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        self._record_time(self._cpu_times, name, elapsed_seconds * 1000.0)

    def start_cuda_timer(
        self,
        name: str,
        stream: torch.cuda.Stream | None = None,
    ) -> _CudaTimer | None:
        if not self.enabled or self.device.type != "cuda":
            return None

        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        return _CudaTimer(name=name, start_event=start_event)

    def stop_cuda_timer(
        self,
        timer: _CudaTimer | None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if timer is None:
            return

        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(stream)
        with self._lock:
            self._pending_cuda_samples.append(
                _PendingCudaSample(
                    name=timer.name,
                    start_event=timer.start_event,
                    end_event=end_event,
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

    def _drain_cuda_samples(self) -> None:
        with self._lock:
            pending = tuple(self._pending_cuda_samples)
            self._pending_cuda_samples.clear()

        retained: list[_PendingCudaSample] = []
        completed: list[tuple[str, float]] = []
        for sample in pending:
            if not sample.end_event.query():
                retained.append(sample)
                continue
            completed.append(
                (
                    sample.name,
                    sample.start_event.elapsed_time(sample.end_event),
                )
            )

        with self._lock:
            self._pending_cuda_samples.extend(retained)
            for name, elapsed_ms in completed:
                total_ms, count = self._cuda_times.get(name, (0.0, 0))
                self._cuda_times[name] = (total_ms + elapsed_ms, count + 1)

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
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def maybe_log(self) -> None:
        if not self.enabled:
            return

        now = monotonic()
        elapsed_seconds = now - self._last_log_time
        if elapsed_seconds < self.log_interval_seconds:
            return

        # This is the only periodic device-to-host synchronization introduced
        # by RetroSpec performance observation.
        gpu_values = self._gpu_counters.detach().cpu().tolist()
        self._gpu_counters.zero_()

        # The counter synchronization completes main-stream CUDA timers. Timers
        # from transfer streams remain queued until their events finish.
        self._drain_cuda_samples()

        with self._lock:
            counters = dict(self._cpu_counters)
            counters.update(
                {
                    name: int(value)
                    for name, value in zip(self._GPU_COUNTER_NAMES, gpu_values)
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
        prefetch_submitted = counters.get("prefetch_submitted", 0)
        prefetch_dropped = counters.get("prefetch_dropped", 0)

        logger.info(
            "RetroSpec performance over %.2fs: counters={%s}; peaks={%s}; "
            "derived={draft_tokens/request=%.2f, expanded/sparse=%.3f, "
            "full/request=%.3f, resident_hit_rate=%.3f, "
            "prefetch_drop_rate=%.3f}; cpu_avg={%s}; cuda_avg={%s}",
            elapsed_seconds,
            self._format_counters(counters),
            self._format_counters(peaks),
            self._ratio(draft_tokens, proposal_requests),
            self._ratio(expanded_tokens, sparse_tokens),
            self._ratio(full_requests, proposal_requests),
            self._ratio(resident_hits, resident_hits + resident_misses),
            self._ratio(
                prefetch_dropped,
                prefetch_submitted + prefetch_dropped,
            ),
            self._format_times(cpu_times),
            self._format_times(cuda_times),
        )
