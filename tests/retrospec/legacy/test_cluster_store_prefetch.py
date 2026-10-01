# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future
from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.cluster_store import (
    get_block_metadata,
    get_runtime_blocks,
    make_cluster_data,
    store_cluster_data,
)
from vllm import _custom_ops as ops
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.spec_decode.retrospec import cluster_store as cluster_store_module
from vllm.v1.spec_decode.retrospec.cluster_identity import (
    RetroSpecClusterGroup,
    RetroSpecClusterIdentity,
)
from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterPageStore,
    RetroSpecResidentPrefetchInput,
)
from vllm.v1.spec_decode.retrospec.performance import RetroSpecPerformanceStats


def test_prefetch_descriptor_arena_preserves_and_invalidates_stable_handles():
    arena = cluster_store_module._LayerPrefetchDescriptorArena()
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 1)
    arena.publish(
        {
            1: cluster_store_module._ClusterBlockDescriptor(
                RetroSpecClusterIdentity(first_group, 3), (10, 11), (2, 1)
            )
        }
    )
    arena.publish(
        {
            4: cluster_store_module._ClusterBlockDescriptor(
                RetroSpecClusterIdentity(second_group, 7), (20,), (2,)
            )
        }
    )

    assert arena.capacity >= 5
    assert arena.page_ids[1, :2].tolist() == [10, 11]
    assert arena.page_counts.tolist()[1] == 2
    assert arena.resolve_groups(
        torch.tensor([1, 4]), arena.group_ids.index_select(0, torch.tensor([1, 4]))
    ) == {1: first_group, 4: second_group}

    arena.invalidate({1})
    assert arena.page_counts[1].item() == 0
    assert arena.group_ids[1].item() == -1
    assert torch.all(arena.page_ids[1] == -1)

    outputs = ops.retrospec_plan_prefetch_admissions(
        (torch.tensor([1], dtype=torch.int64),),
        (torch.tensor([0], dtype=torch.int64),),
        (torch.tensor([1], dtype=torch.int32),),
        (1,),
        (1,),
        (arena.page_ids,),
        (arena.page_counts,),
        (arena.group_ids,),
        (torch.zeros(arena.capacity, dtype=torch.uint8),),
        (2,),
    )
    assert outputs[0][0].numel() == 0
    assert outputs[5].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]


def test_resident_prefetch_source_priority_supersedes_hint_and_latest_draft():
    store = RetroSpecClusterPageStore(page_size=2)

    def make_record(source):
        return RetroSpecResidentPrefetchInput(
            layer_name="layer",
            miss_cluster_ids=torch.empty(0),
            miss_positions=torch.empty(0),
            miss_count=torch.zeros(1),
            num_groups=1,
            num_ranks=1,
            source=source,
        )

    hint = store._stamp_resident_prefetch_records((make_record("prefill_hint"),))[0]
    first_draft = store._stamp_resident_prefetch_records((make_record("draft"),))[0]
    assert first_draft.sequence > hint.sequence
    assert not store._stamp_resident_prefetch_records((make_record("prefill_hint"),))

    latest_draft = store._stamp_resident_prefetch_records((make_record("draft"),))[0]
    assert latest_draft.sequence > first_draft.sequence
    store._discard_resident_prefetch_records((latest_draft,))
    assert store._stamp_resident_prefetch_records((make_record("prefill_hint"),))
    store.close()


def test_resident_prefetch_priority_executor_runs_queued_draft_before_hint():
    executor = cluster_store_module._ResidentPrefetchPriorityExecutor()
    first_started = threading.Event()
    release_first = threading.Event()
    order = []

    def first():
        first_started.set()
        assert release_first.wait(timeout=1.0)
        order.append("running_hint")

    running = executor.submit(0, first)
    assert first_started.wait(timeout=1.0)
    queued_hint = executor.submit(0, order.append, "queued_hint")
    queued_draft = executor.submit(1, order.append, "queued_draft")
    release_first.set()
    running.result(timeout=1.0)
    queued_draft.result(timeout=1.0)
    queued_hint.result(timeout=1.0)
    executor.shutdown()

    assert order == ["running_hint", "queued_draft", "queued_hint"]


def test_resident_prefetch_wave_progress_waits_only_for_requested_layer():
    progress = cluster_store_module._ResidentPrefetchWaveProgress.create(
        ("first", "second")
    )
    wait_completed = threading.Event()

    def wait_for_first() -> None:
        progress.wait_for(("first",))
        wait_completed.set()

    waiter = threading.Thread(target=wait_for_first)
    waiter.start()
    progress.complete_layer("second")
    assert not wait_completed.wait(timeout=0.05)

    progress.complete_layer("first")
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()
    assert wait_completed.is_set()


def test_resident_prefetch_wave_progress_propagates_failure():
    progress = cluster_store_module._ResidentPrefetchWaveProgress.create(("layer",))
    failure = ValueError("worker failed")

    progress.fail(failure)

    with pytest.raises(RuntimeError, match="background processing") as exc_info:
        progress.wait_for(("layer",))
    assert exc_info.value.__cause__ is failure


def test_resident_prefetch_bounded_wait_is_device_local():
    store = RetroSpecClusterPageStore(page_size=2)
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")
    future_0: Future[None] = Future()
    future_1: Future[None] = Future()
    future_0.set_result(None)
    future_1.set_result(None)
    store._resident_prefetch_futures.extend(
        (
            cluster_store_module._ResidentPrefetchWaveFuture(
                device=device_0,
                layer_names=frozenset(("layer-0",)),
                progress=cluster_store_module._ResidentPrefetchWaveProgress.create(
                    ("layer-0",)
                ),
                future=future_0,
            ),
            cluster_store_module._ResidentPrefetchWaveFuture(
                device=device_1,
                layer_names=frozenset(("layer-1",)),
                progress=cluster_store_module._ResidentPrefetchWaveProgress.create(
                    ("layer-1",)
                ),
                future=future_1,
            ),
        )
    )

    assert store._wait_for_one_resident_prefetch(device_1)
    assert tuple(wave.device for wave in store._resident_prefetch_futures) == (
        device_0,
    )
    assert store._wait_for_one_resident_prefetch(device_0)
    assert not store._resident_prefetch_futures
    store.close()


@pytest.mark.parametrize("submitted", (False, True))
def test_resident_prefetch_worker_auto_submit_is_nonblocking(submitted: bool):
    stats = RetroSpecPerformanceStats(
        device=torch.device("cpu"), log_interval_seconds=60.0
    )
    store = RetroSpecClusterPageStore(
        page_size=2,
        performance_stats=stats,
    )
    try_submit = Mock(return_value=submitted)
    store._try_submit_deferred_resident_prefetch = try_submit
    device = torch.device("cuda:1")

    try:
        store._auto_submit_deferred_resident_prefetch(device)

        try_submit.assert_called_once_with(device, wait_for_slot=False)
        expected = 1 if submitted else 0
        assert stats._cpu_counters["prefetch_worker_auto_submits"] == expected
    finally:
        store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA pinned memory is required for asynchronous resident prefetch",
)
def test_resident_prefetch_wave_batches_layers_and_waits_per_layer():
    device = torch.device("cuda", torch.cuda.current_device())
    stats = RetroSpecPerformanceStats(device=device, log_interval_seconds=60.0)
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=0.5,
        performance_stats=stats,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    records: list[RetroSpecResidentPrefetchInput] = []
    for layer_name in ("first", "second"):
        table = store_cluster_data(
            store,
            layer_name,
            keys.to(device),
            values.to(device),
            assignments.to(device),
            cluster_token_counts.to(device),
        )
        cluster_ids = table.cluster_ids.to(device)
        metadata = store.get_cluster_block_metadata(
            layer_name, table.cluster_ids, device=device
        )
        with torch.inference_mode():
            store.resolve_cluster_blocks(
                layer_name,
                cluster_ids,
                metadata.page_ids,
                mode="resident_only",
            )
        records.append(
            RetroSpecResidentPrefetchInput(
                layer_name=layer_name,
                miss_cluster_ids=cluster_ids.reshape(-1),
                miss_positions=torch.arange(
                    cluster_ids.numel(), dtype=torch.int64, device=device
                ),
                miss_count=torch.tensor(
                    [cluster_ids.numel()], dtype=torch.int32, device=device
                ),
                num_groups=cluster_ids.shape[0],
                num_ranks=cluster_ids.shape[1],
            )
        )
    stats._cpu_counters.clear()

    second_layer_started = threading.Event()
    second_layer_completed = threading.Event()
    release_second_layer = threading.Event()
    original_process = store._process_prepared_resident_prefetch

    def block_second_layer(prepared, execution_stream):
        if prepared.layer_name == "second":
            second_layer_started.set()
            assert release_second_layer.wait(timeout=10.0)
        original_process(prepared, execution_stream)
        if prepared.layer_name == "second":
            second_layer_completed.set()

    store._process_prepared_resident_prefetch = block_second_layer
    store.configure_resident_prefetch_wave(2)
    assert store.prefetch_resident_cluster_wave(records)
    try:
        store.wait_for_resident_prefetches(("first",))
        assert second_layer_started.wait(timeout=10.0)

        assert store._resident_prefetch_futures
        assert store.num_resident_clusters("first") == 2
        assert not second_layer_completed.is_set()
        assert stats._cpu_counters["prefetch_submitted"] == 2
        assert stats._cpu_counters["prefetch_waves_submitted"] == 1
        assert stats._cpu_counters["prefetch_wave_records"] == 2
        assert stats._cpu_counters["prefetch_waited_waves"] == 1
        assert stats._cpu_counters["prefetch_layer_waits"] == 1

        release_second_layer.set()
        store.wait_for_resident_prefetches(("second",))
        assert second_layer_completed.is_set()
        assert store.num_resident_clusters("second") == 2
        assert stats._cpu_counters["prefetch_layer_waits"] == 2
        store.wait_for_resident_prefetches()
        assert not store._resident_prefetch_futures
        assert stats._cpu_counters["prefetch_worker_completed"] == 1
        assert stats._cpu_counters["prefetch_reaped_tasks"] == 1
    finally:
        release_second_layer.set()
    store.close()
    store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA pinned memory is required for asynchronous resident prefetch",
)
def test_resident_prefetch_retains_metadata_slot_during_descriptor_preparation():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    cluster_ids = table.cluster_ids.to(device).reshape(-1)
    prepare_started = threading.Event()
    release_prepare = threading.Event()
    original_prepare = store._prepare_resident_prefetch_wave

    def blocking_prepare(staged_records):
        prepare_started.set()
        assert release_prepare.wait(timeout=10.0)
        return original_prepare(staged_records)

    store._prepare_resident_prefetch_wave = blocking_prepare
    store.configure_resident_prefetch_wave(1)
    record = RetroSpecResidentPrefetchInput(
        layer_name="layer",
        miss_cluster_ids=cluster_ids,
        miss_positions=torch.arange(
            cluster_ids.numel(), dtype=torch.int64, device=device
        ),
        miss_count=torch.tensor(
            [cluster_ids.numel()], dtype=torch.int32, device=device
        ),
        num_groups=table.cluster_ids.shape[0],
        num_ranks=table.cluster_ids.shape[1],
    )

    assert store.prefetch_resident_cluster_wave((record,))
    try:
        assert prepare_started.wait(timeout=10.0)
        assert any(
            slot.in_use
            for slots in store._resident_prefetch_slots.values()
            for slot in slots
        )
    finally:
        release_prepare.set()

    store.wait_for_resident_prefetches()
    assert all(
        not slot.in_use
        for slots in store._resident_prefetch_slots.values()
        for slot in slots
    )
    store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA pinned memory is required for resident-prefetch auto drain",
)
def test_resident_prefetch_worker_auto_drains_deferred_wave_without_flush():
    device = torch.device("cuda", torch.cuda.current_device())
    stats = RetroSpecPerformanceStats(device=device, log_interval_seconds=60.0)
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=0.5,
        performance_stats=stats,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    cluster_ids = table.cluster_ids.to(device).reshape(-1)
    metadata = get_block_metadata(store, table, device=device)
    with torch.inference_mode():
        store.resolve_cluster_blocks(
            "layer",
            table.cluster_ids.to(device),
            metadata.page_ids,
            mode="resident_only",
        )

    worker_started = threading.Event()
    release_worker = threading.Event()
    auto_submit_finished = threading.Event()
    auto_submit_devices: list[torch.device] = []
    original_finish = store._finish_resident_prefetch_wave
    original_auto_submit = store._auto_submit_deferred_resident_prefetch

    def blocking_finish(staged):
        worker_started.set()
        assert release_worker.wait(timeout=10.0)
        original_finish(staged)

    def observed_auto_submit(target_device):
        previous_submits = stats._cpu_counters["prefetch_worker_auto_submits"]
        original_auto_submit(target_device)
        current_submits = stats._cpu_counters["prefetch_worker_auto_submits"]
        if current_submits > previous_submits:
            auto_submit_devices.append(target_device)
            auto_submit_finished.set()

    store._finish_resident_prefetch_wave = blocking_finish
    store._auto_submit_deferred_resident_prefetch = observed_auto_submit

    def make_record(offset: int) -> RetroSpecResidentPrefetchInput:
        return RetroSpecResidentPrefetchInput(
            layer_name="layer",
            miss_cluster_ids=torch.roll(cluster_ids, offset),
            miss_positions=torch.arange(
                cluster_ids.numel(), dtype=torch.int64, device=device
            ),
            miss_count=torch.tensor(
                [cluster_ids.numel()], dtype=torch.int32, device=device
            ),
            num_groups=table.cluster_ids.shape[0],
            num_ranks=table.cluster_ids.shape[1],
        )

    assert store.prefetch_resident_cluster_wave((make_record(0),))
    assert worker_started.wait(timeout=10.0)
    assert store.prefetch_resident_cluster_wave((make_record(1),))
    assert store.prefetch_resident_cluster_wave((make_record(2),))
    assert store._resident_prefetch_deferred

    try:
        release_worker.set()
        assert auto_submit_finished.wait(timeout=10.0)
        assert auto_submit_devices == [device]
        assert not store._resident_prefetch_deferred
        assert stats._cpu_counters["prefetch_worker_auto_submits"] == 1

        store.wait_for_resident_prefetches()
        assert not store._resident_prefetch_futures
        assert stats._cpu_counters["prefetch_waves_submitted"] == 3
        assert stats._cpu_counters["prefetch_waves_deferred"] == 1
        assert stats._cpu_counters["prefetch_worker_completed"] == 3
    finally:
        release_worker.set()
        store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA pinned memory is required for resident-prefetch backpressure",
)
def test_resident_prefetch_coalesces_latest_wave_and_applies_bounded_backpressure():
    device = torch.device("cuda", torch.cuda.current_device())
    stats = RetroSpecPerformanceStats(device=device, log_interval_seconds=60.0)
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=0.5,
        performance_stats=stats,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    cluster_ids = table.cluster_ids.to(device).reshape(-1)
    metadata = get_block_metadata(store, table, device=device)
    with torch.inference_mode():
        store.resolve_cluster_blocks(
            "layer",
            table.cluster_ids.to(device),
            metadata.page_ids,
            mode="resident_only",
        )

    worker_started = threading.Event()
    release_worker = threading.Event()
    backpressure_started = threading.Event()
    original_finish = store._finish_resident_prefetch_wave
    original_wait = store._wait_for_one_resident_prefetch

    def blocking_finish(staged):
        worker_started.set()
        assert release_worker.wait(timeout=10.0)
        original_finish(staged)

    def observed_wait(target_device):
        backpressure_started.set()
        return original_wait(target_device)

    store._finish_resident_prefetch_wave = blocking_finish
    store._wait_for_one_resident_prefetch = observed_wait

    def make_record(offset: int) -> RetroSpecResidentPrefetchInput:
        return RetroSpecResidentPrefetchInput(
            layer_name="layer",
            miss_cluster_ids=torch.roll(cluster_ids, offset),
            miss_positions=torch.arange(
                cluster_ids.numel(), dtype=torch.int64, device=device
            ),
            miss_count=torch.tensor(
                [cluster_ids.numel()], dtype=torch.int32, device=device
            ),
            num_groups=table.cluster_ids.shape[0],
            num_ranks=table.cluster_ids.shape[1],
        )

    assert store.prefetch_resident_cluster_wave((make_record(0),))
    assert worker_started.wait(timeout=10.0)
    assert store.prefetch_resident_cluster_wave((make_record(1),))
    assert store.prefetch_resident_cluster_wave((make_record(2),))
    assert store.prefetch_resident_cluster_wave((make_record(3),))
    assert len(store._resident_prefetch_deferred) == 1

    flush_error: list[BaseException] = []

    def flush() -> None:
        try:
            store.flush_resident_prefetch_commands()
        except BaseException as error:
            flush_error.append(error)

    flush_thread = threading.Thread(target=flush)
    flush_thread.start()
    assert backpressure_started.wait(timeout=10.0)
    release_worker.set()
    flush_thread.join(timeout=10.0)
    assert not flush_thread.is_alive()
    assert not flush_error

    store.wait_for_resident_prefetches()
    assert not store._resident_prefetch_deferred
    assert stats._cpu_counters["prefetch_waves_submitted"] == 3
    assert stats._cpu_counters["prefetch_waves_deferred"] == 2
    assert stats._cpu_counters["prefetch_waves_coalesced"] == 1
    assert stats._cpu_counters["prefetch_records_superseded"] == 1
    assert stats._cpu_counters["prefetch_backpressure_waits"] == 1
    assert "prefetch_dropped" not in stats._cpu_counters
    store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_deferred_prefetch_keeps_per_layer_producer_events():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(page_size=2)
    store.configure_resident_prefetch_wave(2)

    def make_record(layer_name: str, handle: int) -> RetroSpecResidentPrefetchInput:
        return RetroSpecResidentPrefetchInput(
            layer_name=layer_name,
            miss_cluster_ids=torch.tensor([handle], dtype=torch.int64, device=device),
            miss_positions=torch.zeros(1, dtype=torch.int64, device=device),
            miss_count=torch.ones(1, dtype=torch.int32, device=device),
            num_groups=1,
            num_ranks=1,
        )

    first = make_record("first", 1)
    second = make_record("second", 2)
    replacement = make_record("first", 3)
    store._defer_resident_prefetch_wave(device, (first,))
    first_event = store._resident_prefetch_deferred[device].source_ready_events[0]
    store._defer_resident_prefetch_wave(device, (second,))
    second_event = store._resident_prefetch_deferred[device].source_ready_events[1]

    deferred = store._resident_prefetch_deferred[device]
    assert deferred.records[0] is first
    assert deferred.records[1] is second
    assert deferred.source_ready_events == (first_event, second_event)

    store._defer_resident_prefetch_wave(device, (replacement,))
    deferred = store._resident_prefetch_deferred[device]
    assert deferred.records[0] is replacement
    assert deferred.records[1] is second
    assert deferred.source_ready_events[1] is second_event
    assert deferred.source_ready_events[0] is not first_event

    store._resident_prefetch_deferred.clear()
    store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to resolve cluster pages",
)
def test_cpu_backing_store_stages_before_updating_resident_cache():
    device = torch.device("cuda", torch.cuda.current_device())
    stats = RetroSpecPerformanceStats(
        device=device,
        log_interval_seconds=60.0,
        cuda_timing_level="detailed",
        cuda_sample_interval=1,
    )
    store = RetroSpecClusterPageStore(
        page_size=2,
        cache_ratio=0.5,
        performance_stats=stats,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.cuda(),
        values.cuda(),
        assignments.cuda(),
        cluster_token_counts.cuda(),
    )
    cluster_ids, metadata = get_runtime_blocks(store, table, torch.device("cuda"))

    resolved = store.resolve_cluster_blocks("layer", cluster_ids, metadata.page_ids)
    torch.cuda.current_stream().synchronize()
    stats._drain_cuda_samples()

    valid_pages = metadata.page_ids >= 0
    resident_pages = resolved.resident_page_ids >= 0
    staging_pages = resolved.staging_page_ids >= 0

    assert torch.equal(resident_pages | staging_pages, valid_pages)
    assert not torch.any(resident_pages & staging_pages)
    assert resident_pages.sum().item() == 0
    assert staging_pages.sum().item() == 6
    assert resolved.hit_cluster_mask.tolist() == [
        [False, False],
        [False, False],
    ]
    assert resolved.miss_cluster_mask.tolist() == [
        [True, True],
        [True, True],
    ]
    assert stats._cpu_counters["verification_miss_pages"] == 6
    assert stats._cpu_counters["verification_miss_h2d_bytes"] == (
        resolved.staging_key_pages.nbytes + resolved.staging_value_pages.nbytes
    )
    assert stats._cpu_times["verification_miss_cpu_gather"][1] == 1
    assert stats._cuda_times["verification_miss_h2d"][1] == 1
    assert store.num_resident_pages("layer") == 0
    assert store.num_resident_clusters("layer") == 0

    staging_slots = resolved.staging_page_ids[staging_pages].to(torch.int64)
    staging_logical_ids = metadata.page_ids[staging_pages].cpu().to(torch.int64)
    backing_keys, backing_values = store.read_page_storage("layer", staging_logical_ids)

    torch.testing.assert_close(
        resolved.staging_key_pages.index_select(0, staging_slots).cpu(),
        backing_keys,
    )
    torch.testing.assert_close(
        resolved.staging_value_pages.index_select(0, staging_slots).cpu(),
        backing_values,
    )

    access = store.admit_staged_clusters(
        layer_name="layer",
        cluster_ids=cluster_ids,
        logical_page_ids=metadata.page_ids,
        staging_page_ids=resolved.staging_page_ids,
        staging_key_pages=resolved.staging_key_pages,
        staging_value_pages=resolved.staging_value_pages,
    )

    assert access.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert access.miss_cluster_mask.tolist() == [
        [False, True],
        [False, True],
    ]
    assert store.num_resident_pages("layer") == 3
    assert store.num_resident_clusters("layer") == 2

    resident_keys, resident_values = store.get_resident_page_storage("layer")
    resident_pages = access.cache_page_ids >= 0
    resident_slots = access.cache_page_ids[resident_pages].to(torch.int64)
    resident_logical_ids = metadata.page_ids[resident_pages].cpu().to(torch.int64)
    resident_backing_keys, resident_backing_values = store.read_page_storage(
        "layer", resident_logical_ids
    )
    torch.testing.assert_close(
        resident_keys.index_select(0, resident_slots).cpu(),
        resident_backing_keys,
    )
    torch.testing.assert_close(
        resident_values.index_select(0, resident_slots).cpu(),
        resident_backing_values,
    )
