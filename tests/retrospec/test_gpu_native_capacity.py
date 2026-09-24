# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.worker import gpu_worker


@pytest.mark.parametrize(
    ("speculative_config", "expected_bytes"),
    [
        (None, 8 << 30),
        (
            SimpleNamespace(method="retrospec", retrospec_max_gpu_index_memory=2.0),
            6 << 30,
        ),
    ],
)
def test_gpu_native_index_budget_is_reserved_before_kv_allocation(
    monkeypatch: pytest.MonkeyPatch,
    speculative_config: SimpleNamespace | None,
    expected_bytes: int,
) -> None:
    @contextmanager
    def profile_memory(*args: object, **kwargs: object):
        yield SimpleNamespace(
            non_torch_increase=0,
            torch_peak_increase=0,
            non_kv_cache_memory=2 << 30,
            after_profile=SimpleNamespace(free_memory=10 << 30),
        )

    monkeypatch.setattr(gpu_worker, "memory_profiling", profile_memory)
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.9,
        ),
        model_runner=SimpleNamespace(model_memory_usage=1 << 30, profile_run=Mock()),
        init_snapshot=SimpleNamespace(free_memory=12 << 30),
        requested_memory=10 << 30,
        vllm_config=SimpleNamespace(speculative_config=speculative_config),
    )
    assert gpu_worker.Worker.determine_available_memory(worker) == expected_bytes
    worker.model_runner.profile_run.assert_called_once()


def test_gpu_native_index_budget_rejects_zero_kv_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def profile_memory(*args: object, **kwargs: object):
        yield SimpleNamespace(
            non_torch_increase=0,
            torch_peak_increase=0,
            non_kv_cache_memory=2 << 30,
            after_profile=SimpleNamespace(free_memory=10 << 30),
        )

    monkeypatch.setattr(gpu_worker, "memory_profiling", profile_memory)
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=None),
        model_runner=SimpleNamespace(model_memory_usage=1 << 30, profile_run=Mock()),
        init_snapshot=SimpleNamespace(free_memory=12 << 30),
        requested_memory=10 << 30,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="retrospec", retrospec_max_gpu_index_memory=8.0
            )
        ),
    )
    with pytest.raises(ValueError, match="leaves no room for the KV cache"):
        gpu_worker.Worker.determine_available_memory(worker)
