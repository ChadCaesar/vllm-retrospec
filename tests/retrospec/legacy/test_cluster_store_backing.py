# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
)
from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterPageStore,
)
from vllm.v1.spec_decode.retrospec.performance import RetroSpecPerformanceStats


def test_cpu_backing_store_preserves_page_layout_and_reuses_pages():
    store = RetroSpecClusterPageStore(
        page_size=2,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    first_metadata = get_block_metadata(store, first)
    first_page_ids = first_metadata.page_ids.clone()
    valid_page_ids = first_metadata.page_ids[first_metadata.page_ids >= 0]
    key_pages, value_pages = store.read_page_storage("layer", valid_page_ids)

    assert store.get_storage_device("layer") == torch.device("cpu")
    assert first.cluster_ids.device.type == "cpu"
    assert first_metadata.page_ids.device.type == "cpu"
    assert first_metadata.page_token_counts.device.type == "cpu"
    assert key_pages.device.type == "cpu"
    assert value_pages.device.type == "cpu"

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        first_metadata.page_ids.unsqueeze(0),
        first_metadata.page_token_counts.unsqueeze(0),
    )
    assert gathered_keys.device.type == "cpu"
    assert gathered_values.device.type == "cpu"
    assert token_mask.device.type == "cpu"
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    store.free("layer", first)
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    second_metadata = get_block_metadata(store, second)

    torch.testing.assert_close(second_metadata.page_ids, first_page_ids)
    assert not torch.equal(second.cluster_ids, first.cluster_ids)


def test_cpu_backing_store_appends_pageable_slabs_without_migration():
    store = RetroSpecClusterPageStore(
        page_size=2,
        cpu_page_initial_slab_bytes=32,
        cpu_page_slab_bytes=64,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    first_metadata = get_block_metadata(store, first)
    pool = store._layer_pools["layer"]

    assert pool.initial_pages_per_slab == 2
    assert pool.max_pages_per_slab == 4
    assert [slab.key_pages.shape[0] for slab in pool._slabs] == [2, 4]
    assert first_metadata.page_ids[first_metadata.page_ids >= 0].tolist() == [
        0,
        1,
        1 << 32,
        (1 << 32) + 1,
        (1 << 32) + 2,
        (1 << 32) + 3,
    ]
    slab_pointers = [slab.key_pages.data_ptr() for slab in pool._slabs]
    assert all(not slab.key_pages.is_pinned() for slab in pool._slabs)
    assert all(not slab.value_pages.is_pinned() for slab in pool._slabs)

    second = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    assert [slab.key_pages.shape[0] for slab in pool._slabs] == [2, 4, 4, 4]
    assert [slab.key_pages.data_ptr() for slab in pool._slabs[:2]] == slab_pointers

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        first_metadata.page_ids.unsqueeze(0),
        first_metadata.page_token_counts.unsqueeze(0),
    )
    for head_index in range(keys.shape[0]):
        gathered_head_keys = gathered_keys[0, head_index, token_mask[0, head_index]]
        gathered_head_values = gathered_values[0, head_index, token_mask[0, head_index]]
        torch.testing.assert_close(
            gathered_head_keys[:, 0].sort().values, keys[head_index, :, 0]
        )
        torch.testing.assert_close(gathered_head_values, gathered_head_keys + 100.0)

    store.free("layer", first)
    reused = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    reused_metadata = get_block_metadata(store, reused)
    torch.testing.assert_close(reused_metadata.page_ids, first_metadata.page_ids)
    store.free("layer", second)
    store.free("layer", reused)
    assert pool.num_slabs == 0
    assert pool.capacity == 0


def test_cpu_backing_store_growth_preserves_existing_logical_pages():
    store = RetroSpecClusterPageStore(
        page_size=2,
    )
    first_keys = torch.arange(60, dtype=torch.float32).view(1, 60, 1)
    first_values = first_keys + 1000.0
    first_assignments = torch.zeros(1, 60, dtype=torch.int64)
    first_counts = torch.tensor([[60]], dtype=torch.int32)
    first = store_cluster_data(
        store,
        "layer",
        first_keys,
        first_values,
        first_assignments,
        first_counts,
    )
    first_metadata = get_block_metadata(store, first)

    second_keys = torch.arange(100, dtype=torch.float32).view(1, 100, 1)
    second_values = second_keys + 2000.0
    second_assignments = torch.zeros(1, 100, dtype=torch.int64)
    second_counts = torch.tensor([[100]], dtype=torch.int32)
    second = store_cluster_data(
        store,
        "layer",
        second_keys,
        second_values,
        second_assignments,
        second_counts,
    )

    pool = store._layer_pools["layer"]
    assert pool.num_slabs == 1
    assert pool.capacity >= 80
    assert store.num_allocated_pages("layer") == 80

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        first_metadata.page_ids.unsqueeze(0),
        first_metadata.page_token_counts.unsqueeze(0),
    )
    torch.testing.assert_close(
        gathered_keys[0, 0, token_mask[0, 0]],
        first_keys[0],
    )
    torch.testing.assert_close(
        gathered_values[0, 0, token_mask[0, 0]],
        first_values[0],
    )

    store.free("layer", first)
    store.free("layer", second)
    assert store.num_allocated_pages("layer") == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA with pinned host memory is required",
)
def test_cpu_backing_store_keeps_long_lived_metadata_on_pageable_cpu():
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
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
    metadata = get_block_metadata(store, table)
    assert table.cluster_ids.device.type == "cpu"
    assert not table.cluster_ids.is_pinned()
    assert metadata.page_ids.device.type == "cpu"
    assert not metadata.page_ids.is_pinned()
    assert metadata.page_token_counts.device.type == "cpu"
    assert not metadata.page_token_counts.is_pinned()
    pool = store._layer_pools["layer"]
    assert all(not slab.key_pages.is_pinned() for slab in pool._slabs)
    assert all(not slab.value_pages.is_pinned() for slab in pool._slabs)

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )
    assert gathered_keys.device.type == "cpu"
    assert gathered_values.device.type == "cpu"
    assert token_mask.device.type == "cpu"
    assert gathered_keys[0, 1, token_mask[0, 1], 0].tolist() == [
        11.0,
        13.0,
        10.0,
        12.0,
        14.0,
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the resident cluster cache",
)
def test_cpu_backing_store_admits_and_invalidates_resident_clusters():
    store = RetroSpecClusterPageStore(
        page_size=2,
        cache_ratio=0.5,
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

    access = store.admit_resident_clusters("layer", cluster_ids, metadata.page_ids)
    resident_cache = store._resident_caches["layer"]

    assert store.resident_capacity("layer") == 3
    assert store.num_resident_pages("layer") == 3
    assert store.num_resident_clusters("layer") == 2
    assert store.num_resident_groups("layer") == 2
    assert resident_cache._group_targets == {
        RetroSpecClusterGroup("request", 0): 2,
        RetroSpecClusterGroup("request", 1): 1,
    }
    assert (
        resident_cache._group_states[RetroSpecClusterGroup("request", 0)].num_pages == 2
    )
    assert (
        resident_cache._group_states[RetroSpecClusterGroup("request", 1)].num_pages == 1
    )
    assert access.hit_cluster_mask.tolist() == [[True, False], [True, False]]
    assert access.miss_cluster_mask.tolist() == [[False, True], [False, True]]
    assert len(resident_cache._pending_copy_batches) == 1

    cache_keys, cache_values = store.get_resident_page_storage("layer")
    valid = access.cache_page_ids >= 0
    slots = access.cache_page_ids[valid].to(torch.int64)
    logical_ids = metadata.page_ids[valid].cpu().to(torch.int64)
    backing_keys, backing_values = store.read_page_storage("layer", logical_ids)
    torch.testing.assert_close(
        cache_keys.index_select(0, slots).cpu(),
        backing_keys,
    )
    torch.testing.assert_close(
        cache_values.index_select(0, slots).cpu(),
        backing_values,
    )

    store.free("layer", table)
    assert resident_cache.num_pending_copy_batches == 0
    assert store.resident_capacity("layer") == 0
    assert store.num_resident_pages("layer") == 0
    assert store.num_resident_clusters("layer") == 0
    assert store.num_resident_groups("layer") == 0
    assert resident_cache._group_targets == {}


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA pinned memory is required for asynchronous resident prefetch",
)
def test_cpu_backing_store_prefetches_resident_clusters_in_background(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    stats = RetroSpecPerformanceStats(
        device=device,
        log_interval_seconds=60.0,
    )
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
    cluster_ids = table.cluster_ids.to(device)
    metadata = get_block_metadata(store, table, device=device)

    # vLLM creates the resident arena on its inference-mode execution thread.
    # The background worker must explicitly enter inference mode before it can
    # update those tensors in place.
    with torch.inference_mode():
        store.resolve_cluster_blocks(
            "layer", cluster_ids, metadata.page_ids, mode="resident_only"
        )
    resident_cache = store._resident_caches["layer"]
    assert resident_cache.performance_stats is stats
    calls = {"prepare": 0, "capture": 0, "commit": 0}
    original_prepare = resident_cache.prepare_staged_admission
    original_capture = resident_cache.capture_prepared_admission_lru
    original_commit = resident_cache.admit_prepared_staged

    def prepare_outside_mutation_guard(*args, **kwargs):
        assert not resident_cache.mutation_guard().locked()
        calls["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def capture_inside_mutation_guard(*args, **kwargs):
        assert resident_cache.mutation_guard().locked()
        calls["capture"] += 1
        return original_capture(*args, **kwargs)

    def commit_inside_mutation_guard(*args, **kwargs):
        assert resident_cache.mutation_guard().locked()
        calls["commit"] += 1
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(
        resident_cache,
        "prepare_staged_admission",
        prepare_outside_mutation_guard,
    )
    monkeypatch.setattr(
        resident_cache,
        "capture_prepared_admission_lru",
        capture_inside_mutation_guard,
    )
    monkeypatch.setattr(
        resident_cache,
        "admit_prepared_staged",
        commit_inside_mutation_guard,
    )
    stats._cpu_counters.clear()

    access_kinds = torch.full_like(cluster_ids, 2, dtype=torch.uint8)
    assert store.prefetch_resident_clusters("layer", cluster_ids, access_kinds)
    store.wait_for_resident_prefetches(("layer",))
    store.wait_for_resident_prefetches()
    assert stats._cpu_counters["prefetch_submitted"] == 1
    assert stats._cpu_counters["prefetch_command_capacity"] == cluster_ids.numel()
    assert stats._cpu_counters["prefetch_worker_completed"] == 1
    assert stats._cpu_counters["prefetch_reaped_tasks"] == 1
    assert stats._cpu_counters["prefetch_waited_tasks"] >= 1
    assert stats._cpu_counters["prefetch_miss_commands"] == 4
    assert stats._cpu_counters["prefetch_duplicate_misses"] == 0
    assert stats._cpu_times["prefetch_metadata_wait"][1] == 1
    assert stats._cpu_times["prefetch_page_gather_wall"][1] == 1
    assert stats._cpu_times["prefetch_resident_prepare_wall"][1] == 1
    assert stats._cpu_times["prefetch_resident_lru_capture_wall"][1] == 1
    assert stats._cpu_times["prefetch_resident_lru_resolve_wall"][1] == 1
    assert stats._cpu_times["prefetch_resident_commit_wall"][1] == 1
    assert stats._cpu_times["prefetch_resident_admission_wall"][1] == 1
    assert stats._cpu_times["prefetch_worker_wall"][1] == 1
    assert stats._cpu_times["prefetch_wait_wall"][1] >= 1
    assert calls == {"prepare": 1, "capture": 1, "commit": 1}

    access = store.lookup_resident_clusters(
        "layer", cluster_ids, metadata.page_ids, touch=False
    )
    assert access.hit_cluster_mask.tolist() == [[True, False], [True, False]]
    assert access.miss_cluster_mask.tolist() == [[False, True], [False, True]]
    assert store.num_resident_pages("layer") == 3
    assert store.num_resident_clusters("layer") == 2
    assert not store._resident_prefetch_futures
    assert all(
        not slot.in_use
        for slots in store._resident_prefetch_slots.values()
        for slot in slots
    )

    resident_keys, resident_values = store.get_resident_page_storage("layer")
    torch.cuda.current_stream(device).synchronize()
    resident_pages = access.cache_page_ids >= 0
    resident_slots = access.cache_page_ids[resident_pages].to(torch.int64)
    logical_ids = metadata.page_ids[resident_pages].cpu().to(torch.int64)
    backing_keys, backing_values = store.read_page_storage("layer", logical_ids)
    torch.testing.assert_close(
        resident_keys.index_select(0, resident_slots).cpu(),
        backing_keys,
    )
    torch.testing.assert_close(
        resident_values.index_select(0, resident_slots).cpu(),
        backing_values,
    )

    resident_cluster_ids = cluster_ids[access.hit_cluster_mask]
    stats._cpu_counters.clear()
    original_stage = store._stage_resident_pages
    store._stage_resident_pages = Mock(wraps=original_stage)
    assert store.prefetch_resident_clusters(
        "layer",
        resident_cluster_ids,
        torch.full_like(resident_cluster_ids, 2, dtype=torch.uint8),
    )
    store.wait_for_resident_prefetches()
    store._stage_resident_pages.assert_not_called()
    assert stats._cpu_counters["prefetch_skipped_resident_clusters"] == 2
    assert stats._cpu_counters.get("prefetch_skipped_pending_clusters", 0) == 0
    store.close()


def test_resident_prefetch_native_batch_plans_rank_priority_and_pages():
    outputs = ops.retrospec_plan_prefetch_admissions(
        (
            torch.tensor([0, 1, 0, -1], dtype=torch.int64),
            torch.tensor([0, 1, -1], dtype=torch.int64),
        ),
        (
            torch.tensor([4, 1, 4, -1], dtype=torch.int64),
            torch.tensor([3, 0, -1], dtype=torch.int64),
        ),
        (
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        ),
        (2, 2),
        (3, 2),
        (
            torch.tensor([[10, -1], [11, 12]], dtype=torch.int64),
            torch.tensor([[20, -1], [21, 22]], dtype=torch.int64),
        ),
        (
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int32),
        ),
        (
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
        ),
        (
            torch.zeros(2, dtype=torch.uint8),
            torch.zeros(2, dtype=torch.uint8),
        ),
        (3, 3),
    )

    cluster_ids, page_ids, source_page_ids, unique_page_ids, group_ids, stats = outputs
    assert tuple(record.tolist() for record in cluster_ids) == ([1, 0], [1, 0])
    assert tuple(record.tolist() for record in page_ids) == (
        [[11, 12], [10, -1]],
        [[21, 22], [20, -1]],
    )
    assert tuple(record.tolist() for record in source_page_ids) == (
        [[0, 1], [2, -1]],
        [[0, 1], [2, -1]],
    )
    assert tuple(record.tolist() for record in unique_page_ids) == (
        [11, 12, 10],
        [21, 22, 20],
    )
    assert tuple(record.tolist() for record in group_ids) == ([1, 0], [1, 0])
    assert stats.tolist() == [
        [3, 2, 0, 0, 0, 2, 3, 0],
        [2, 2, 0, 0, 0, 2, 3, 0],
    ]


def test_resident_prefetch_native_batch_ignores_unused_suffix():
    outputs = ops.retrospec_plan_prefetch_admissions(
        (torch.tensor([-1, -1], dtype=torch.int64),),
        (torch.tensor([-1, -1], dtype=torch.int64),),
        (torch.tensor([0], dtype=torch.int32),),
        (1,),
        (2,),
        (torch.tensor([[-1]], dtype=torch.int64),),
        (torch.zeros(1, dtype=torch.int32),),
        (torch.full((1,), -1, dtype=torch.int64),),
        (torch.zeros(1, dtype=torch.uint8),),
        (1,),
    )

    assert all(records[0].numel() == 0 for records in outputs[:5])
    assert outputs[5].tolist() == [[0, 0, 0, 0, 0, 0, 0, 0]]


def test_resident_prefetch_native_batch_rejects_invalid_prefix():
    with pytest.raises(RuntimeError, match="count exceeds"):
        ops.retrospec_plan_prefetch_admissions(
            (torch.tensor([10], dtype=torch.int64),),
            (torch.tensor([0], dtype=torch.int64),),
            (torch.tensor([2], dtype=torch.int32),),
            (1,),
            (1,),
            (torch.tensor([[0]], dtype=torch.int64),),
            (torch.ones(1, dtype=torch.int32),),
            (torch.zeros(1, dtype=torch.int64),),
            (torch.zeros(1, dtype=torch.uint8),),
            (1,),
        )


@pytest.mark.parametrize(
    ("cluster_id", "position", "error"),
    (
        (-1, 0, "invalid handle"),
        (0, -1, "out of range"),
        (0, 1, "out of range"),
    ),
)
def test_resident_prefetch_native_batch_rejects_invalid_commands(
    cluster_id: int, position: int, error: str
):
    with pytest.raises(RuntimeError, match=error):
        ops.retrospec_plan_prefetch_admissions(
            (torch.tensor([cluster_id], dtype=torch.int64),),
            (torch.tensor([position], dtype=torch.int64),),
            (torch.tensor([1], dtype=torch.int32),),
            (1,),
            (1,),
            (torch.tensor([[0]], dtype=torch.int64),),
            (torch.ones(1, dtype=torch.int32),),
            (torch.zeros(1, dtype=torch.int64),),
            (torch.zeros(1, dtype=torch.uint8),),
            (1,),
        )


def test_resident_prefetch_native_batch_filters_state_and_honors_budget():
    outputs = ops.retrospec_plan_prefetch_admissions(
        (torch.tensor([0, 1, 2, 3], dtype=torch.int64),),
        (torch.arange(4, dtype=torch.int64),),
        (torch.tensor([4], dtype=torch.int32),),
        (1,),
        (4,),
        (torch.tensor([[10, -1], [11, -1], [12, -1], [13, 14]], dtype=torch.int64),),
        (torch.tensor([1, 1, 1, 2], dtype=torch.int32),),
        (torch.arange(4, dtype=torch.int64),),
        (torch.tensor([1, 2, 0, 0], dtype=torch.uint8),),
        (1,),
    )

    assert outputs[0][0].tolist() == [2]
    assert outputs[3][0].tolist() == [12]
    assert outputs[5].tolist() == [[4, 4, 0, 1, 1, 1, 1, 1]]


def test_native_cluster_page_gather_reads_encoded_slab_ranges():
    key_slabs = (
        torch.arange(4, dtype=torch.float32).view(2, 2, 1),
        torch.arange(6, dtype=torch.float32).view(3, 2, 1) + 10,
    )
    value_slabs = tuple(slab + 100 for slab in key_slabs)
    page_ids = torch.tensor(
        [
            cluster_store_module._LayerClusterPagePool.encode_page_id(0, 1),
            cluster_store_module._LayerClusterPagePool.encode_page_id(1, 0),
            cluster_store_module._LayerClusterPagePool.encode_page_id(1, 2),
        ],
        dtype=torch.int64,
    )
    key_output = torch.empty((3, 2, 1), dtype=torch.float32)
    value_output = torch.empty_like(key_output)

    ops.retrospec_gather_cluster_pages(
        key_slabs, value_slabs, page_ids, 2, key_output, value_output, 2
    )

    torch.testing.assert_close(
        key_output,
        torch.stack((key_slabs[0][1], key_slabs[1][0], key_slabs[1][2])),
    )
    torch.testing.assert_close(value_output, key_output + 100)
