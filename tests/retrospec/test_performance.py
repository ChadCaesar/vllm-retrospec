# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator

import pytest
import torch

from vllm.v1.spec_decode.retrospec import performance
from vllm.v1.spec_decode.retrospec.performance import RetroSpecPerformanceStats


def fake_clock(values: list[float]) -> Iterator[float]:
    yield from values


def test_disabled_stats_allocate_no_device_workspace(monkeypatch: pytest.MonkeyPatch):
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(performance.logger, "info", lambda *args: messages.append(args))

    stats = RetroSpecPerformanceStats(
        device=torch.device("cpu"),
        log_interval_seconds=0.0,
    )
    stats.add_counter("proposal_calls")
    stats.add_gpu_counter("unknown_counter", torch.ones(1, dtype=torch.int64))
    stats.observe_peak("queue_depth", 2)
    stats.record_cpu_time("proposal_wall", 0.1)
    stats.maybe_log()

    assert not stats.enabled
    assert stats._gpu_counters.numel() == 0
    assert not messages


def test_negative_interval_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        RetroSpecPerformanceStats(
            device=torch.device("cpu"),
            log_interval_seconds=-1.0,
        )


def test_stats_wait_for_logging_interval(monkeypatch: pytest.MonkeyPatch):
    clock = fake_clock([100.0, 104.0])
    monkeypatch.setattr(performance, "monotonic", lambda: next(clock))
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(performance.logger, "info", lambda *args: messages.append(args))

    stats = RetroSpecPerformanceStats(
        device=torch.device("cpu"),
        log_interval_seconds=5.0,
    )
    stats.add_counter("proposal_calls")
    stats.maybe_log()

    assert not messages
    assert stats._cpu_counters["proposal_calls"] == 1


def test_stats_log_counts_ratios_and_timings(monkeypatch: pytest.MonkeyPatch):
    clock = fake_clock([100.0, 106.0])
    monkeypatch.setattr(performance, "monotonic", lambda: next(clock))
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(performance.logger, "info", lambda *args: messages.append(args))

    stats = RetroSpecPerformanceStats(
        device=torch.device("cpu"),
        log_interval_seconds=5.0,
    )
    stats.add_gpu_counter("proposal_requests", torch.tensor([1, 1]))
    stats.add_counter("sparse_verify_tokens", 8)
    stats.add_counter("expanded_verify_tokens", 2)
    stats.add_counter("full_verify_requests", 1)
    stats.add_counter("resident_cluster_hits", 9)
    stats.add_counter("resident_cluster_misses", 1)
    stats.add_counter("prefetch_submitted", 3)
    stats.add_counter("prefetch_dropped", 1)
    stats.add_gpu_counter("draft_round_requests", torch.tensor([1, 1]))
    stats.add_gpu_counter("draft_tokens", torch.tensor([3, 5]))
    stats.add_gpu_counter("verified_tokens", torch.tensor([2, 4]))
    stats.add_gpu_counter("proposed_tokens", torch.tensor([2, 3]))
    stats.observe_peak("cluster_build_queue_depth", 2)
    stats.record_cpu_time("proposal_wall", 0.012)

    stats.maybe_log()

    assert len(messages) == 1
    message = messages[0][0] % messages[0][1:]
    assert "proposal_requests=2" in message
    assert "draft_tokens=8" in message
    assert "verified_tokens=6" in message
    assert "cluster_build_queue_depth=2" in message
    assert "draft_tokens/request=4.00" in message
    assert "expanded/sparse=0.250" in message
    assert "full/request=0.500" in message
    assert "resident_hit_rate=0.900" in message
    assert "prefetch_drop_rate=0.250" in message
    assert "proposal_wall=12.000ms/1" in message

    assert not stats._cpu_counters
    assert not stats._peaks
    assert not stats._cpu_times
    assert torch.count_nonzero(stats._gpu_counters).item() == 0


def test_enabled_stats_reject_unknown_gpu_counter():
    stats = RetroSpecPerformanceStats(
        device=torch.device("cpu"),
        log_interval_seconds=1.0,
    )

    with pytest.raises(KeyError, match="unknown"):
        stats.add_gpu_counter("unknown", 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_timer_is_drained_when_complete(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda", torch.cuda.current_device())
    clock = fake_clock([100.0, 102.0])
    monkeypatch.setattr(performance, "monotonic", lambda: next(clock))
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(performance.logger, "info", lambda *args: messages.append(args))
    stats = RetroSpecPerformanceStats(device=device, log_interval_seconds=1.0)

    timer = stats.start_cuda_timer("kernel")
    torch.ones(32, device=device).mul_(2)
    stats.stop_cuda_timer(timer)
    torch.cuda.synchronize(device)
    stats.maybe_log()

    assert len(messages) == 1
    message = messages[0][0] % messages[0][1:]
    assert "kernel=" in message
    assert "ms/1" in message
