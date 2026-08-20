# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.v1.core.utils import create_scheduler
from vllm.config import DeviceConfig, SpeculativeConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVCacheRetirement

pytestmark = pytest.mark.cpu_test


def test_retrospec_reserves_speculative_lookahead_tokens():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=64,
    )

    scheduler = create_scheduler(
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )

    assert scheduler.num_spec_tokens == 64
    assert scheduler.num_lookahead_tokens == 64


def test_retrospec_scheduler_change_does_not_affect_ngram():
    scheduler = create_scheduler(
        num_speculative_tokens=4,
        device_config=DeviceConfig(device="cpu"),
    )

    assert scheduler.num_spec_tokens == 4
    assert scheduler.num_lookahead_tokens == 0


def test_scheduler_applies_worker_kv_cache_retirement():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = Mock()
    scheduler.requests = {
        "request": SimpleNamespace(is_finished=lambda: False),
    }
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"request": 1}
    retirement = KVCacheRetirement(
        request_id="request",
        kv_cache_group_id=2,
        start_block=1,
        end_block=5,
    )

    scheduler._apply_kv_cache_retirements(scheduler_output, [retirement])

    scheduler.kv_cache_manager.retire_blocks.assert_called_once_with(
        request_id="request",
        kv_cache_group_id=2,
        start_block=1,
        end_block=5,
    )


def test_scheduler_rejects_retirement_for_unscheduled_request():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = Mock()
    scheduler.requests = {}
    scheduler_output = SchedulerOutput.make_empty()
    retirement = KVCacheRetirement(
        request_id="request",
        kv_cache_group_id=0,
        start_block=1,
        end_block=2,
    )

    with pytest.raises(RuntimeError, match="unscheduled request"):
        scheduler._apply_kv_cache_retirements(scheduler_output, [retirement])
