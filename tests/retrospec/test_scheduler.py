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
    DraftTokenIds,
    KVCacheRetirement,
    ModelRunnerOutput,
    RetroSpecLayerMajorPrefillCompletion,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.retrospec.prefill import (
    RetroSpecLayerMajorPrefillProtocol,
)

pytestmark = pytest.mark.cpu_test


def make_retrospec_pp_scheduler(
    *,
    num_requests: int = 4,
) -> tuple[Scheduler, list[Request]]:
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_max_draft_tokens=4,
        retrospec_index_segment_size=32,
    )
    scheduler = create_scheduler(
        max_num_seqs=num_requests,
        max_num_batched_tokens=num_requests * 8,
        max_model_len=64,
        pipeline_parallel_size=2,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    requests = create_requests(
        num_requests=num_requests,
        num_tokens=8,
        req_ids=[f"request-{index}" for index in range(num_requests)],
    )
    for request in requests:
        scheduler.add_request(request)
    return scheduler, requests


def make_model_runner_output(
    scheduler_output: SchedulerOutput,
    *,
    draft_token_ids: list[list[int]] | None = None,
) -> ModelRunnerOutput:
    request_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={
            request_id: index for index, request_id in enumerate(request_ids)
        },
        sampled_token_ids=[[1] for _ in request_ids],
        retrospec_draft_token_ids=(
            DraftTokenIds(request_ids, draft_token_ids)
            if draft_token_ids is not None
            else None
        ),
    )


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


def test_retrospec_scheduler_reports_remaining_generation_budgets():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.is_retrospec = True
    scheduler.requests = {
        "first": SimpleNamespace(
            sampling_params=object(),
            max_tokens=12,
            num_output_tokens=5,
        ),
        "second": SimpleNamespace(
            sampling_params=object(),
            max_tokens=3,
            num_output_tokens=3,
        ),
        "pooling": SimpleNamespace(
            sampling_params=None,
            max_tokens=1,
            num_output_tokens=0,
        ),
    }

    budgets = scheduler._get_retrospec_generation_token_budgets(
        ["first", "second", "pooling"]
    )

    assert budgets == {"first": 7, "second": 0}


def test_retrospec_scheduler_trims_drafts_to_authoritative_budget():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.is_retrospec = True
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = False
    request = SimpleNamespace(
        is_finished=lambda: False,
        is_prefill_chunk=False,
        max_tokens=5,
        num_output_tokens=3,
        spec_token_ids=[],
    )
    scheduler.requests = {"request": request}

    scheduler.update_draft_token_ids(DraftTokenIds(["request"], [[10, 11, 12, 13]]))

    assert request.spec_token_ids == [10, 11]


def test_retrospec_pp_schedules_request_disjoint_batches():
    scheduler, _ = make_retrospec_pp_scheduler()

    first = scheduler.schedule()
    second = scheduler.schedule()

    first_request_ids = set(first.num_scheduled_tokens)
    second_request_ids = set(second.num_scheduled_tokens)
    assert len(first_request_ids) == 2
    assert len(second_request_ids) == 2
    assert first_request_ids.isdisjoint(second_request_ids)
    assert first.retrospec_pp_batch_id == 0
    assert second.retrospec_pp_batch_id == 1
    assert scheduler._retrospec_in_flight_req_ids == (
        first_request_ids | second_request_ids
    )


def test_retrospec_pp_releases_dependency_before_rescheduling_request():
    scheduler, _ = make_retrospec_pp_scheduler()
    first = scheduler.schedule()
    second = scheduler.schedule()
    first_request_ids = set(first.num_scheduled_tokens)
    draft_token_ids = [[11, 12] for _ in first_request_ids]

    scheduler.update_from_output(
        first,
        make_model_runner_output(first, draft_token_ids=draft_token_ids),
    )
    third = scheduler.schedule()

    assert set(third.num_scheduled_tokens) == first_request_ids
    assert set(third.num_scheduled_tokens).isdisjoint(second.num_scheduled_tokens)
    assert third.retrospec_pp_batch_id == 2
    for request_id in first_request_ids:
        assert third.scheduled_spec_decode_tokens[request_id] == [11, 12]


def test_retrospec_pp_rejects_out_of_order_completion():
    scheduler, _ = make_retrospec_pp_scheduler()
    scheduler.schedule()
    second = scheduler.schedule()

    with pytest.raises(RuntimeError, match="completed out of order"):
        scheduler._complete_retrospec_pp_batch(second)


@pytest.mark.parametrize("batch_index", [0, 1])
def test_retrospec_pp_defers_abort_until_owning_batch_completes(batch_index: int):
    scheduler, _ = make_retrospec_pp_scheduler()
    first = scheduler.schedule()
    second = scheduler.schedule()
    owning_batch = (first, second)[batch_index]
    aborted_request_id = next(iter(owning_batch.num_scheduled_tokens))

    scheduler.finish_requests(aborted_request_id, RequestStatus.FINISHED_ABORTED)

    assert aborted_request_id in scheduler.requests
    assert aborted_request_id in scheduler._retrospec_deferred_finish_status
    first_output = make_model_runner_output(first)
    first_engine_outputs = scheduler.update_from_output(first, first_output)
    if batch_index == 0:
        assert aborted_request_id not in scheduler.requests
        assert all(
            output.request_id != aborted_request_id
            for client_outputs in first_engine_outputs.values()
            for output in client_outputs.outputs
        )
    else:
        assert aborted_request_id in scheduler.requests

    second_output = make_model_runner_output(second)
    engine_outputs = scheduler.update_from_output(second, second_output)

    assert aborted_request_id not in scheduler.requests
    assert aborted_request_id not in scheduler._retrospec_deferred_finish_status
    assert all(
        output.request_id != aborted_request_id
        for client_outputs in engine_outputs.values()
        for output in client_outputs.outputs
    )


def test_retrospec_pp_protects_in_flight_requests_from_preemption():
    scheduler, _ = make_retrospec_pp_scheduler()
    scheduler.schedule()
    scheduler.schedule()

    assert scheduler._select_preemption_victim() is None
    assert not scheduler.reset_prefix_cache(reset_running_requests=True)


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
        retrospec_index_segment_size=4,
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
    assert scheduler_output.retrospec_generation_token_budgets == {
        "prefill-0": requests[0].max_tokens
    }
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
    assert scheduler_output.retrospec_generation_token_budgets == {
        "prefill-0": requests[0].max_tokens - 1
    }
    assert scheduler_output.retrospec_layer_major_prefill is None
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1


def test_retrospec_pp_layer_major_prefill_blocks_pipeline_fill():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_max_draft_tokens=4,
        retrospec_index_segment_size=4,
    )
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=32,
        pipeline_parallel_size=2,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=8,
        req_ids=["long-prefill"],
    )[0]
    scheduler.add_request(request)

    first = scheduler.schedule()
    blocked = scheduler.schedule()

    assert first.retrospec_layer_major_prefill is not None
    assert first.retrospec_pp_batch_id == 0
    assert blocked.total_num_scheduled_tokens == 0
    assert blocked.retrospec_pp_batch_id is None


def test_short_prompt_uses_native_chunked_prefill():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_max_draft_tokens=4,
        retrospec_index_segment_size=8,
    )
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        max_model_len=32,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    request = create_requests(num_requests=1, num_tokens=8)[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {request.request_id: 8}
    assert scheduler_output.retrospec_layer_major_prefill is None


def test_layer_major_prefill_allocates_only_sink_and_resident_suffix():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=4,
        retrospec_max_draft_tokens=4,
        retrospec_index_segment_size=4,
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
        retrospec_index_segment_size=4,
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
        retrospec_index_segment_size=4,
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
