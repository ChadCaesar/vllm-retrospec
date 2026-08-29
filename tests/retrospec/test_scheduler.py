# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import DeviceConfig, SpeculativeConfig
from vllm.v1.core.sched.output import (
    RetroSpecLayerMajorPrefillDescriptor,
    SchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import (
    KVCacheRetirement,
    RetroSpecLayerMajorPrefillCompletion,
)
from vllm.v1.spec_decode.retrospec.prefill import (
    RetroSpecLayerMajorPrefillProtocol,
)

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


def make_layer_major_scheduler_output(
    *,
    scheduled_start: int = 4,
    scheduled_end: int = 8,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {
        "request": scheduled_end - scheduled_start,
    }
    output.total_num_scheduled_tokens = scheduled_end - scheduled_start
    output.retrospec_layer_major_prefill = RetroSpecLayerMajorPrefillDescriptor(
        request_id="request",
        prompt_num_tokens=16,
        scheduled_start=scheduled_start,
        scheduled_end=scheduled_end,
    )
    return output


def test_layer_major_prefill_descriptor_validates_ranges():
    with pytest.raises(ValueError, match="greater than scheduled_start"):
        RetroSpecLayerMajorPrefillDescriptor(
            request_id="request",
            prompt_num_tokens=16,
            scheduled_start=4,
            scheduled_end=4,
        )

    with pytest.raises(ValueError, match="cannot exceed prompt_num_tokens"):
        RetroSpecLayerMajorPrefillDescriptor(
            request_id="request",
            prompt_num_tokens=16,
            scheduled_start=4,
            scheduled_end=17,
        )


def test_layer_major_prefill_protocol_returns_completed_range():
    scheduler_output = make_layer_major_scheduler_output()

    completion = RetroSpecLayerMajorPrefillProtocol.complete_scheduled_range(
        scheduler_output
    )

    assert completion == RetroSpecLayerMajorPrefillCompletion(
        request_id="request",
        num_completed_prompt_tokens=8,
    )


def test_layer_major_prefill_protocol_rejects_mixed_batch():
    scheduler_output = make_layer_major_scheduler_output()
    scheduler_output.num_scheduled_tokens["decode"] = 1
    scheduler_output.total_num_scheduled_tokens += 1

    with pytest.raises(RuntimeError, match="exclusive batch"):
        RetroSpecLayerMajorPrefillProtocol.validate_scheduler_output(scheduler_output)


def test_layer_major_prefill_protocol_rejects_speculative_tokens():
    scheduler_output = make_layer_major_scheduler_output()
    scheduler_output.scheduled_spec_decode_tokens = {"request": [1]}

    with pytest.raises(RuntimeError, match="cannot contain speculative tokens"):
        RetroSpecLayerMajorPrefillProtocol.validate_scheduler_output(scheduler_output)


def test_retrospec_layer_major_prefill_is_scheduled_exclusively():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_max_draft_tokens=4,
    )
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        max_model_len=32,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=8,
        req_ids=["prefill-0", "prefill-1"],
    )
    for request in requests:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {"prefill-0": 4}
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    assert scheduler_output.retrospec_layer_major_prefill == (
        RetroSpecLayerMajorPrefillDescriptor(
            request_id="prefill-0",
            prompt_num_tokens=8,
            scheduled_start=0,
            scheduled_end=4,
        )
    )

    # Once the first request enters decode, the second prefill must remain
    # waiting instead of silently joining a regular mixed batch.
    requests[0].num_computed_tokens = requests[0].num_prompt_tokens
    requests[0].append_output_token_ids(1)
    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {"prefill-0": 1}
    assert scheduler_output.retrospec_layer_major_prefill is None
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1


def test_scheduler_applies_actual_layer_major_prefill_completion():
    scheduler = Scheduler.__new__(Scheduler)
    request = SimpleNamespace(
        num_computed_tokens=8,
        num_tokens=16,
        num_output_placeholders=0,
        is_prefill_chunk=False,
        is_finished=lambda: False,
    )
    scheduler.requests = {"request": request}
    scheduler_output = make_layer_major_scheduler_output()
    completion = RetroSpecLayerMajorPrefillCompletion(
        request_id="request",
        num_completed_prompt_tokens=6,
    )

    scheduler._update_retrospec_layer_major_prefill_completion(
        scheduler_output, completion
    )

    assert request.num_computed_tokens == 6
    assert request.is_prefill_chunk


def test_scheduler_rejects_completion_outside_reserved_range():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.requests = {
        "request": SimpleNamespace(
            is_finished=lambda: False,
        )
    }
    scheduler_output = make_layer_major_scheduler_output()
    completion = RetroSpecLayerMajorPrefillCompletion(
        request_id="request",
        num_completed_prompt_tokens=9,
    )

    with pytest.raises(RuntimeError, match="outside its reserved range"):
        scheduler._update_retrospec_layer_major_prefill_completion(
            scheduler_output, completion
        )
