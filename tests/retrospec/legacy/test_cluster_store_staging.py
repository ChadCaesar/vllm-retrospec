# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.cluster_store import (
    get_block_metadata,
    make_cluster_data,
    make_token_offsets,
    store_cluster_data,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterPageStore,
)
from vllm.v1.spec_decode.retrospec.performance import RetroSpecPerformanceStats


def test_cpu_backing_store_stages_and_commits_cpu_inputs():
    store = RetroSpecClusterPageStore(
        page_size=2,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    staged = store.stage_clusters(
        token_keys=keys,
        token_values=values,
        assignments=assignments,
        cluster_token_counts=cluster_token_counts,
        token_offsets_in_cluster=make_token_offsets(
            assignments,
            cluster_token_counts,
        ),
    )

    assert staged.ready_event is None
    assert staged.metadata_device == torch.device("cpu")
    assert "layer" not in store._layer_pools

    table = store.store_staged_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        staged=staged,
    )
    metadata = get_block_metadata(store, table)
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert store.num_allocated_pages("layer") == 6
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    torch.testing.assert_close(
        gathered_values[token_mask.unsqueeze(-1)],
        gathered_keys[token_mask.unsqueeze(-1)] + 100.0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_resident_admission_limits_prefetch_to_fixed_h2d_slot():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=1.0,
        max_pinned_memory_bytes=64,
    )
    keys = torch.arange(4, dtype=torch.float32, device=device).view(1, 4, 1)
    table = store_cluster_data(
        store,
        "layer",
        keys,
        keys + 100.0,
        torch.tensor([[0, 0, 1, 1]], dtype=torch.int64, device=device),
        torch.tensor([[2, 2]], dtype=torch.int32, device=device),
    )
    cluster_ids = table.cluster_ids.to(device)
    metadata = store.get_cluster_block_metadata("layer", cluster_ids, device=device)

    access = store.admit_resident_clusters("layer", cluster_ids, metadata.page_ids)

    assert access.hit_cluster_mask.tolist() == [[True, False]]
    assert access.miss_cluster_mask.tolist() == [[False, True]]
    transfer_buffer = store._full_verification_buffers[device]
    assert transfer_buffer._cpu_slot_capacity == 1
    assert store.num_resident_pages("layer") == 1


def test_cpu_backing_store_supports_two_phase_staging():
    stats = RetroSpecPerformanceStats(
        device=torch.device("cpu"),
        log_interval_seconds=60.0,
    )
    store = RetroSpecClusterPageStore(
        page_size=2,
        performance_stats=stats,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    staged_token_kv = store.stage_token_kv(keys, values)

    assert staged_token_kv.source_device == torch.device("cpu")
    assert staged_token_kv.ready_event is None
    assert staged_token_kv.token_keys is keys
    assert staged_token_kv.token_values is values

    staged = store.finish_stage_clusters(
        staged_token_kv,
        assignments,
        cluster_token_counts,
        make_token_offsets(assignments, cluster_token_counts),
    )

    assert staged.ready_event is None
    assert staged.metadata_device == torch.device("cpu")
    assert staged.token_keys is keys
    assert staged.token_values is values
    assert staged.assignments is assignments
    assert staged.cluster_token_counts is cluster_token_counts

    table = store.store_staged_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        staged=staged,
    )

    assert store.num_allocated_pages("layer") == 6
    assert table.cluster_ids.tolist() == [[0, 1], [2, 3]]
    assert stats._cpu_counters["cluster_builds"] == 1
    assert stats._cpu_counters["cluster_pages_built"] == 6
    assert stats._cpu_times["cluster_build_wait"][1] == 1
    assert stats._cpu_times["cluster_page_build"][1] == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_asynchronously_stages_cuda_inputs():
    device = torch.device("cuda", torch.cuda.current_device())
    stats = RetroSpecPerformanceStats(
        device=device,
        log_interval_seconds=60.0,
        cuda_timing_level="detailed",
        cuda_sample_interval=1,
    )
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        performance_stats=stats,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    keys = keys.to(device)
    values = values.to(device)
    assignments = assignments.to(device)
    cluster_token_counts = cluster_token_counts.to(device)
    token_offsets = make_token_offsets(assignments, cluster_token_counts)

    staged_token_kv = store.stage_token_kv(keys, values)

    assert staged_token_kv.ready_event is not None
    assert staged_token_kv.source_device == device
    assert staged_token_kv.token_keys.device.type == "cpu"
    assert staged_token_kv.token_values.device.type == "cpu"
    assert staged_token_kv.token_keys.is_pinned()
    assert staged_token_kv.token_values.is_pinned()
    assert stats._cpu_counters["token_kv_d2h_bytes"] == keys.nbytes + values.nbytes

    # Enqueue work after token-KV staging. finish_stage_clusters() must make
    # metadata D2H wait for this work without serializing the earlier KV copy.
    assignments = assignments.clone()
    cluster_token_counts = cluster_token_counts.clone()
    staged = store.finish_stage_clusters(
        staged_token_kv,
        assignments,
        cluster_token_counts,
        token_offsets,
    )

    assert staged.ready_event is not None
    assert staged.metadata_device == device
    assert stats._cpu_counters["cluster_metadata_d2h_bytes"] == (
        assignments.nbytes + cluster_token_counts.nbytes + token_offsets.nbytes
    )
    assert all(
        tensor.device.type == "cpu" and tensor.is_pinned()
        for tensor in (
            staged.token_keys,
            staged.token_values,
            staged.assignments,
            staged.cluster_token_counts,
            staged.token_offsets_in_cluster,
        )
    )
    assert "layer" not in store._layer_pools

    table = store.store_staged_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        staged=staged,
    )
    stats._drain_cuda_samples(wait_for_completion=True)
    assert stats._cuda_times["prefill_token_kv_d2h"][1] == 1
    assert stats._cuda_times["prefill_cluster_metadata_d2h"][1] == 1
    pool = store._layer_pools["layer"]
    metadata = store.get_cluster_block_metadata(
        "layer",
        table.cluster_ids,
        device=torch.device("cpu"),
    )
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert pool.storage_device == torch.device("cpu")
    assert pool.metadata_device == device
    assert all(not slab.key_pages.is_pinned() for slab in pool._slabs)
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    torch.testing.assert_close(
        gathered_values[token_mask.unsqueeze(-1)],
        gathered_keys[token_mask.unsqueeze(-1)] + 100.0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_reuses_pinned_staging_slot_after_build():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    keys = keys.to(device)
    values = values.to(device)
    assignments = assignments.to(device)
    cluster_token_counts = cluster_token_counts.to(device)
    token_offsets = make_token_offsets(assignments, cluster_token_counts)

    first = store.stage_clusters(
        keys,
        values,
        assignments,
        cluster_token_counts,
        token_offsets,
    )
    first_slot = first.staging_slot
    assert first_slot is not None
    first_pointers = (
        first.token_keys.data_ptr(),
        first.token_values.data_ptr(),
        first.assignments.data_ptr(),
        first.cluster_token_counts.data_ptr(),
        first.token_offsets_in_cluster.data_ptr(),
    )

    store.store_staged_clusters("first-layer", "request", 0, first)

    assert not first_slot.in_use

    second = store.stage_clusters(
        keys,
        values,
        assignments,
        cluster_token_counts,
        token_offsets,
    )
    second_pointers = (
        second.token_keys.data_ptr(),
        second.token_values.data_ptr(),
        second.assignments.data_ptr(),
        second.cluster_token_counts.data_ptr(),
        second.token_offsets_in_cluster.data_ptr(),
    )

    assert second.staging_slot is first_slot
    assert second_pointers == first_pointers
    assert len(store._pinned_staging_slots[device]) == 1

    store.store_staged_clusters("second-layer", "request", 0, second)
    assert not first_slot.in_use


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_does_not_reuse_busy_pinned_staging_slot():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
    )
    keys, values, _, _ = make_cluster_data()
    keys = keys.to(device)
    values = values.to(device)

    first = store.stage_token_kv(keys, values)
    second = store.stage_token_kv(keys, values)

    assert first.staging_slot is not None
    assert second.staging_slot is not None
    assert second.staging_slot is not first.staging_slot
    assert len(store._pinned_staging_slots[device]) == 2
    with pytest.raises(RuntimeError, match="staging slot is available"):
        store.stage_token_kv(keys, values)

    store.discard_staged_token_kv(first)
    store.discard_staged_token_kv(second)

    assert not first.staging_slot.in_use
    assert not second.staging_slot.in_use


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_enforces_pinned_build_slot_budget():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        max_pinned_memory_bytes=128,
        max_pending_cluster_builds=2,
    )
    keys, values, _, _ = make_cluster_data()

    with pytest.raises(RuntimeError, match="pinned-memory slot budget"):
        store.stage_token_kv(keys.to(device), values.to(device))

    slots = store._pinned_staging_slots[device]
    assert len(slots) == 1
    assert not slots[0].in_use


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_grows_reused_pinned_staging_slot():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
    )
    small_keys = torch.arange(4, dtype=torch.float32, device=device).view(1, 4, 1)
    small_values = small_keys + 10
    small = store.stage_token_kv(small_keys, small_values)
    slot = small.staging_slot
    assert slot is not None
    assert slot.token_key_storage is not None
    old_capacity = slot.token_key_storage.numel()
    store.discard_staged_token_kv(small)

    large_keys = torch.arange(12, dtype=torch.float32, device=device).view(1, 12, 1)
    large_values = large_keys + 20
    large = store.stage_token_kv(large_keys, large_values)
    large.wait()

    assert large.staging_slot is slot
    assert slot.token_key_storage is not None
    assert slot.token_key_storage.numel() == large_keys.numel()
    assert slot.token_key_storage.numel() > old_capacity
    torch.testing.assert_close(large.token_keys, large_keys.cpu())
    torch.testing.assert_close(large.token_values, large_values.cpu())

    store.discard_staged_token_kv(large)
    assert not slot.in_use


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_cpu_backing_store_releases_pinned_slot_when_build_fails(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    staged = store.stage_clusters(
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
        make_token_offsets(assignments, cluster_token_counts).to(device),
    )
    slot = staged.staging_slot
    assert slot is not None

    monkeypatch.setattr(
        store,
        "store_clusters",
        Mock(side_effect=RuntimeError("cluster build failed")),
    )

    with pytest.raises(RuntimeError, match="cluster build failed"):
        store.store_staged_clusters("layer", "request", 0, staged)

    assert not slot.in_use


def test_cluster_store_rejects_invalid_storage_metadata():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    metadata = get_block_metadata(store, table)

    invalid_page_ids = metadata.page_ids.unsqueeze(0).clone()
    invalid_page_ids[0, 0, 0, 0] = -2
    with pytest.raises(ValueError, match="at least -1"):
        store.gather_pages(
            "layer",
            invalid_page_ids,
            metadata.page_token_counts.unsqueeze(0),
        )

    negative_counts = cluster_token_counts.clone()
    negative_counts[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        store_cluster_data(
            store,
            "other",
            keys,
            values,
            assignments,
            negative_counts,
        )

    with pytest.raises(ValueError, match="cluster_start"):
        store_cluster_data(
            store,
            "other",
            keys,
            values,
            assignments,
            cluster_token_counts,
            cluster_start=-1,
        )

    invalid_offsets = make_token_offsets(assignments, cluster_token_counts)
    invalid_offsets[0, 0] = cluster_token_counts[0, 0]
    with pytest.raises(RuntimeError, match="offsets exceed"):
        store.store_clusters(
            layer_name="invalid-offset-layer",
            request_id="request",
            cluster_start=0,
            token_keys=keys,
            token_values=values,
            assignments=assignments,
            cluster_token_counts=cluster_token_counts,
            token_offsets_in_cluster=invalid_offsets,
        )

    duplicate_offsets = make_token_offsets(assignments, cluster_token_counts)
    duplicate_offsets[0, 1] = duplicate_offsets[0, 0]
    with pytest.raises(RuntimeError, match="offsets must be unique"):
        store.store_clusters(
            layer_name="duplicate-offset-layer",
            request_id="request",
            cluster_start=0,
            token_keys=keys,
            token_values=values,
            assignments=assignments,
            cluster_token_counts=cluster_token_counts,
            token_offsets_in_cluster=duplicate_offsets,
        )


def test_cluster_store_uses_cpu_descriptor_instead_of_gpu_page_contents():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = get_block_metadata(store, table)

    mismatched_page_ids = metadata.page_ids.clone()
    mismatched_page_ids[0, 0] = metadata.page_ids[0, 1]

    cluster_ids_cpu, page_ids_cpu = store._validate_cluster_blocks(
        "layer",
        table.cluster_ids,
        mismatched_page_ids,
    )

    assert torch.equal(cluster_ids_cpu, table.cluster_ids.cpu())
    assert torch.equal(page_ids_cpu, metadata.page_ids.cpu())


def test_cluster_store_pads_cpu_descriptor_to_packed_page_width():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = get_block_metadata(store, table)
    selected_cluster_ids = table.cluster_ids[0:1, 1:2]
    packed_page_ids = metadata.page_ids[0:1, 1:2]

    _, page_ids_cpu = store._validate_cluster_blocks(
        "layer", selected_cluster_ids, packed_page_ids
    )

    assert page_ids_cpu.shape == packed_page_ids.shape
    assert page_ids_cpu[0, 0, 0] >= 0
    assert page_ids_cpu[0, 0, 1] == -1


def test_cluster_store_rejects_packed_page_width_smaller_than_descriptor():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = get_block_metadata(store, table)

    with pytest.raises(RuntimeError, match="smaller"):
        store._validate_cluster_blocks(
            "layer", table.cluster_ids[0:1, 0:1], metadata.page_ids[0:1, 0:1, :1]
        )


def test_cluster_store_close_is_idempotent_and_rejects_new_prefetches():
    store = RetroSpecClusterPageStore(page_size=2)

    store.close()
    store.close()

    with pytest.raises(RuntimeError, match="closed"):
        store.prefetch_resident_clusters(
            "layer",
            torch.tensor([0]),
            torch.tensor([2], dtype=torch.uint8),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA pinned memory is required for the resident access ring",
)
def test_resident_access_ring_grows_outside_an_in_use_slot():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(page_size=2, pin_memory=True)
    store.configure_resident_prefetch_wave(3)
    store.reserve_resident_access_ring(device, 5)

    slot = store._acquire_resident_prefetch_slot(device)
    assert slot is not None
    assert slot.cluster_id_storage is not None
    assert slot.cluster_id_storage.numel() >= 15

    store.reserve_resident_access_ring(device, 17)
    free_slot = next(
        candidate
        for candidate in store._resident_prefetch_slots[device]
        if candidate is not slot
    )
    assert free_slot.cluster_id_storage is not None
    assert free_slot.cluster_id_storage.numel() >= 51

    store._release_resident_prefetch_slot(slot)
    assert slot.cluster_id_storage.numel() >= 51
    store.close()


def test_cluster_store_rejects_released_cluster_id_after_page_reuse():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    first_metadata = get_block_metadata(store, first)
    stale_cluster_ids = first.cluster_ids.clone()
    reused_page_ids = first_metadata.page_ids.clone()
    store.free("layer", first)

    second = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    second_metadata = get_block_metadata(store, second)
    torch.testing.assert_close(second_metadata.page_ids, reused_page_ids)

    with pytest.raises(RuntimeError, match="not allocated"):
        store.resolve_cluster_blocks(
            "layer", stale_cluster_ids, second_metadata.page_ids
        )


def test_cluster_store_materializes_selected_cpu_block_metadata():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    selected_cluster_ids = torch.tensor(
        [[table.cluster_ids[0, 0], -1], [table.cluster_ids[1, 1], -1]],
        dtype=torch.int64,
    )

    metadata = store.get_cluster_block_metadata("layer", selected_cluster_ids)

    assert metadata.page_ids.device.type == "cpu"
    assert metadata.page_ids.shape == (2, 2, 2)
    assert metadata.page_token_counts.tolist() == [
        [[2, 1], [0, 0]],
        [[2, 1], [0, 0]],
    ]
    assert store.max_pages_per_cluster("layer", selected_cluster_ids) == 2


@pytest.mark.parametrize("page_size", [0, -1])
def test_cluster_store_rejects_non_positive_page_size(page_size):
    with pytest.raises(ValueError, match="positive"):
        RetroSpecClusterPageStore(page_size)


def test_cluster_store_rejects_non_positive_page_build_workers():
    with pytest.raises(ValueError, match="cpu_page_build_workers"):
        RetroSpecClusterPageStore(page_size=2, cpu_page_build_workers=0)


def test_cluster_store_rejects_non_positive_full_verify_gather_workers():
    with pytest.raises(ValueError, match="full_verify_gather_workers"):
        RetroSpecClusterPageStore(page_size=2, full_verify_gather_workers=0)
