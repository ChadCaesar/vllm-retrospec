# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import DeviceConfig, SpeculativeConfig
from vllm.v1.core.kv_cache_utils import KV_CACHE_NULL_BLOCK_ID
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
    prompt_num_tokens: int = 16,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"request": prompt_num_tokens}
    output.total_num_scheduled_tokens = prompt_num_tokens
    output.retrospec_layer_major_prefill = RetroSpecLayerMajorPrefillDescriptor(
        request_id="request",
        prompt_num_tokens=prompt_num_tokens,
        scheduled_start=0,
        scheduled_end=prompt_num_tokens,
        resident_start_block=1,
        num_logical_blocks=2,
    )
    return output


def test_layer_major_prefill_descriptor_validates_ranges():
    with pytest.raises(ValueError, match="start at token zero"):
        RetroSpecLayerMajorPrefillDescriptor(
            request_id="request",
            prompt_num_tokens=16,
            scheduled_start=4,
            scheduled_end=16,
            resident_start_block=1,
            num_logical_blocks=2,
        )

    with pytest.raises(ValueError, match="complete prompt"):
        RetroSpecLayerMajorPrefillDescriptor(
            request_id="request",
            prompt_num_tokens=16,
            scheduled_start=0,
            scheduled_end=8,
            resident_start_block=1,
            num_logical_blocks=2,
        )


def test_layer_major_prefill_protocol_returns_completed_range():
    scheduler_output = make_layer_major_scheduler_output()

    completion = RetroSpecLayerMajorPrefillProtocol.complete_scheduled_range(
        scheduler_output
    )

    assert completion == RetroSpecLayerMajorPrefillCompletion(
        request_id="request",
        num_completed_prompt_tokens=16,
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

    assert scheduler_output.num_scheduled_tokens == {"prefill-0": 8}
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    assert scheduler_output.retrospec_layer_major_prefill == (
        RetroSpecLayerMajorPrefillDescriptor(
            request_id="prefill-0",
            prompt_num_tokens=8,
            scheduled_start=0,
            scheduled_end=8,
            resident_start_block=1,
            num_logical_blocks=1,
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


def test_layer_major_prefill_allocates_only_sink_and_resident_suffix():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_max_draft_tokens=4,
    )
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        max_model_len=128,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["long-prefill"],
    )[0]
    scheduler.add_request(request)
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    initial_free_blocks = manager.block_pool.get_num_free_blocks()

    scheduler_output = scheduler.schedule()

    descriptor = scheduler_output.retrospec_layer_major_prefill
    assert descriptor is not None
    assert scheduler_output.num_scheduled_tokens == {"long-prefill": 64}
    assert descriptor.resident_start_block == 2
    assert descriptor.num_logical_blocks == 5

    block_ids = scheduler.kv_cache_manager.get_block_ids("long-prefill")[0]
    assert len(block_ids) == 5
    assert block_ids[0] != KV_CACHE_NULL_BLOCK_ID
    assert block_ids[1] == KV_CACHE_NULL_BLOCK_ID
    assert all(block_id != KV_CACHE_NULL_BLOCK_ID for block_id in block_ids[2:])
    assert manager.block_pool.get_num_free_blocks() == initial_free_blocks - 4

    scheduler.kv_cache_manager.free(request)

    assert manager.block_pool.get_num_free_blocks() == initial_free_blocks


def test_layer_major_prefill_keeps_native_blocks_without_complete_cluster():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_blocks_per_cluster=4,
        retrospec_max_draft_tokens=4,
    )
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        max_model_len=128,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["empty-cluster-prefill"],
    )[0]
    scheduler.add_request(request)
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    initial_free_blocks = manager.block_pool.get_num_free_blocks()

    scheduler_output = scheduler.schedule()

    descriptor = scheduler_output.retrospec_layer_major_prefill
    assert descriptor is not None
    assert descriptor.resident_start_block == 1
    assert descriptor.retired_start_block == descriptor.retired_end_block

    block_ids = scheduler.kv_cache_manager.get_block_ids("empty-cluster-prefill")[0]
    assert len(block_ids) == 5
    assert all(block_id != KV_CACHE_NULL_BLOCK_ID for block_id in block_ids)
    assert manager.block_pool.get_num_free_blocks() == initial_free_blocks - 5

    scheduler.kv_cache_manager.free(request)

    assert manager.block_pool.get_num_free_blocks() == initial_free_blocks


def test_layer_major_prefill_keeps_cluster_remainder_blocks_native():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_blocks_per_cluster=4,
        retrospec_max_draft_tokens=4,
    )
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        max_model_len=256,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=128,
        req_ids=["cluster-remainder-prefill"],
    )[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    descriptor = scheduler_output.retrospec_layer_major_prefill
    assert descriptor is not None
    assert descriptor.resident_start_block == 5
    assert descriptor.num_logical_blocks == 9

    block_ids = scheduler.kv_cache_manager.get_block_ids("cluster-remainder-prefill")[0]
    assert block_ids[0] != KV_CACHE_NULL_BLOCK_ID
    assert all(block_id == KV_CACHE_NULL_BLOCK_ID for block_id in block_ids[1:5])
    assert all(block_id != KV_CACHE_NULL_BLOCK_ID for block_id in block_ids[5:])


def test_scheduler_applies_actual_layer_major_prefill_completion():
    scheduler = Scheduler.__new__(Scheduler)
    request = SimpleNamespace(
        num_computed_tokens=16,
        num_tokens=16,
        num_output_placeholders=0,
        is_prefill_chunk=False,
        is_finished=lambda: False,
    )
    scheduler.requests = {"request": request}
    scheduler_output = make_layer_major_scheduler_output()
    completion = RetroSpecLayerMajorPrefillCompletion(
        request_id="request",
        num_completed_prompt_tokens=16,
    )

    scheduler._update_retrospec_layer_major_prefill_completion(
        scheduler_output, completion
    )

    assert request.num_computed_tokens == 16
    assert not request.is_prefill_chunk


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
        num_completed_prompt_tokens=15,
    )

    with pytest.raises(RuntimeError, match="atomic prompt transaction"):
        scheduler._update_retrospec_layer_major_prefill_completion(
            scheduler_output, completion
        )
