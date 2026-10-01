# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import CancelledError
from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.cluster_store import (
    get_block_metadata,
    get_runtime_blocks,
    make_cluster_data,
    make_resident_arena,
    materialize_resolved_pages,
    store_cluster_data,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.spec_decode.retrospec import cluster_store as cluster_store_module
from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterPageStore,
    RetroSpecVerificationResolveRequest,
)
from vllm.v1.spec_decode.retrospec.performance import RetroSpecPerformanceStats


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for GPU-native verification resolution",
)
def test_gpu_verification_resolution_deduplicates_miss_pages_before_h2d():
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
    cluster_ids, metadata = get_runtime_blocks(store, table, device)
    arena = make_resident_arena(table, metadata, cluster_token_counts, device)

    reference_store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=0.5,
    )
    reference_table = store_cluster_data(
        reference_store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    reference_cluster_ids, reference_metadata = get_runtime_blocks(
        reference_store, reference_table, device
    )
    reference = reference_store.resolve_cluster_blocks(
        "layer",
        torch.cat((reference_cluster_ids, reference_cluster_ids), dim=-1).unsqueeze(0),
        torch.cat(
            (reference_metadata.page_ids, reference_metadata.page_ids), dim=-2
        ).unsqueeze(0),
    )

    selected_cluster_indices = (
        torch.arange(cluster_ids.shape[-1], dtype=torch.int32, device=device)
        .repeat(2)[None, None, :]
        .expand(1, cluster_ids.shape[0], -1)
    )
    resolved = store.resolve_verification_cluster_blocks(
        layer_name="layer",
        selected_cluster_indices=selected_cluster_indices,
        plan_valid_rows=torch.ones(1, dtype=torch.bool, device=device),
        request_slot_ids=torch.zeros(1, dtype=torch.int64, device=device),
        request_slot_generations=torch.ones(1, dtype=torch.int64, device=device),
        arena=arena,
        max_pages_per_cluster=metadata.page_ids.shape[-1],
    )
    assert not (resolved.resident_page_ids >= 0).any()
    assert resolved.page_counts.tolist() == [[6, 6]]
    assert resolved.miss_admission is not None
    assert resolved.miss_admission.cluster_ids_cpu.numel() == cluster_ids.numel()
    assert resolved.staging_key_pages.shape[0] == (metadata.page_ids >= 0).sum()
    assert torch.equal(
        resolved.staging_page_ids[..., :3],
        resolved.staging_page_ids[..., 3:6],
    )

    reference.staging_ready_event.synchronize()
    resolved.staging_ready_event.synchronize()
    reference_keys, reference_values, reference_mask = materialize_resolved_pages(
        reference
    )
    resolved_keys, resolved_values, resolved_mask = materialize_resolved_pages(resolved)
    for head_index in range(cluster_ids.shape[0]):
        reference_head_mask = reference_mask[0, head_index].flatten()
        expected_keys = reference_keys[0, head_index].flatten(0, 1)[reference_head_mask]
        expected_values = reference_values[0, head_index].flatten(0, 1)[
            reference_head_mask
        ]
        page_count = int(resolved.page_counts[0, head_index].item())
        torch.testing.assert_close(
            resolved_keys[0, head_index, :page_count], expected_keys
        )
        torch.testing.assert_close(
            resolved_values[0, head_index, :page_count], expected_values
        )
        assert resolved_mask[0, head_index, :page_count].all()
        assert not resolved_mask[0, head_index, page_count:].any()
    if reference.read_lease is not None:
        reference.read_lease.release()
    if resolved.read_lease is not None:
        resolved.read_lease.release()
    store.admit_verification_misses(resolved.miss_admission)
    store.synchronize_resident_prefetches(("layer",))
    assert store.num_resident_pages("layer") == 3
    assert stats._cpu_counters["verification_unique_miss_clusters"] == 4
    assert stats._cpu_counters["verification_duplicate_miss_clusters"] == 4
    assert stats._cpu_counters["verification_miss_pages"] == 6
    expected_metadata_bytes = 3 * torch.tensor([], dtype=torch.int32).element_size()
    expected_metadata_bytes += cluster_ids.numel() * (
        torch.tensor([], dtype=torch.int64).element_size()
        + torch.tensor([], dtype=torch.int32).element_size()
        + metadata.page_ids.shape[-1]
        * torch.tensor([], dtype=torch.int64).element_size()
    )
    assert (
        stats._cpu_counters["verification_miss_metadata_d2h_bytes"]
        == expected_metadata_bytes
    )
    reference_store.close()
    store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for batched verification resolution",
)
@torch.inference_mode()
def test_gpu_verification_resolution_batches_cross_layer_metadata_waits():
    device = torch.device("cuda", torch.cuda.current_device())
    stats = RetroSpecPerformanceStats(device=device, log_interval_seconds=60.0)
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=0.5,
        performance_stats=stats,
    )
    store.configure_resident_prefetch_wave(2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    requests = []
    for layer_name in ("layer.0", "layer.1"):
        table = store_cluster_data(
            store,
            layer_name,
            keys.to(device),
            values.to(device),
            assignments.to(device),
            cluster_token_counts.to(device),
        )
        cluster_ids = table.cluster_ids.to(device=device)
        metadata = store.get_cluster_block_metadata(
            layer_name=layer_name,
            cluster_ids=cluster_ids,
            device=device,
        )
        arena = make_resident_arena(table, metadata, cluster_token_counts, device)
        requests.append(
            RetroSpecVerificationResolveRequest(
                layer_name=layer_name,
                selected_cluster_indices=torch.arange(
                    cluster_ids.shape[-1], dtype=torch.int32, device=device
                )[None, None, :].expand(1, cluster_ids.shape[0], -1),
                plan_valid_rows=torch.ones(1, dtype=torch.bool, device=device),
                request_slot_ids=torch.zeros(1, dtype=torch.int64, device=device),
                request_slot_generations=torch.ones(
                    1, dtype=torch.int64, device=device
                ),
                arena=arena,
                max_pages_per_cluster=metadata.page_ids.shape[-1],
            )
        )

    resolved = store.resolve_verification_cluster_batch(requests)
    assert tuple(resolved) == ("layer.0", "layer.1")
    assert stats._cpu_times["verification_batch_count_wait"][1] == 1
    assert stats._cpu_times["verification_batch_metadata_wait"][1] == 1
    assert stats._cpu_counters["verification_prepared_layers"] == 2
    assert stats._cpu_counters["verification_miss_layers"] == 2
    for pages in resolved.values():
        assert pages.miss_admission is not None
        assert pages.read_lease is not None
        pages.read_lease.release()
        store.submit_verification_miss_admission(pages.miss_admission)
    store.wait_for_verification_admissions()
    assert stats._cpu_counters["verification_async_admissions"] == 2
    store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for GPU-native verification resolution",
)
def test_gpu_verification_resolution_all_hit_skips_staging_and_admission():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        cache_ratio=1.0,
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
    cluster_ids, metadata = get_runtime_blocks(store, table, device)
    arena = make_resident_arena(table, metadata, cluster_token_counts, device)
    selected_cluster_ids = cluster_ids.unsqueeze(0)
    selected_page_ids = metadata.page_ids.unsqueeze(0)
    store.admit_resident_clusters("layer", selected_cluster_ids, selected_page_ids)

    resolved = store.resolve_verification_cluster_blocks(
        layer_name="layer",
        selected_cluster_indices=torch.arange(
            cluster_ids.shape[-1], dtype=torch.int32, device=device
        )[None, None, :].expand(1, cluster_ids.shape[0], -1),
        plan_valid_rows=torch.ones(1, dtype=torch.bool, device=device),
        request_slot_ids=torch.zeros(1, dtype=torch.int64, device=device),
        request_slot_generations=torch.ones(1, dtype=torch.int64, device=device),
        arena=arena,
        max_pages_per_cluster=metadata.page_ids.shape[-1],
    )
    assert (resolved.resident_page_ids >= 0).sum().item() == 6
    assert resolved.page_counts.tolist() == [[3, 3]]
    assert resolved.staging_key_pages.shape[0] == 0
    assert resolved.staging_value_pages.shape[0] == 0
    assert resolved.miss_admission is None
    assert torch.all(resolved.staging_page_ids == -1)
    if resolved.read_lease is not None:
        resolved.read_lease.release()
    store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for indexed verification resolution",
)
def test_gpu_verification_resolution_accepts_packed_query_rows():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(page_size=2, pin_memory=True, cache_ratio=1.0)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    cluster_ids, metadata = get_runtime_blocks(store, table, device)
    arena = make_resident_arena(table, metadata, cluster_token_counts, device)
    store.admit_resident_clusters(
        "layer", cluster_ids.unsqueeze(0), metadata.page_ids.unsqueeze(0)
    )

    local_indices = torch.arange(
        cluster_ids.shape[-1], dtype=torch.int32, device=device
    )
    table_cluster_indices = torch.stack(
        (
            local_indices[None, :].expand(cluster_ids.shape[0], -1),
            local_indices.flip(-1)[None, :].expand(cluster_ids.shape[0], -1),
            torch.full_like(cluster_ids, -1, dtype=torch.int32),
        )
    )
    plan_rows = torch.tensor([1, 0, 1], dtype=torch.int64, device=device)
    packed = table_cluster_indices.index_select(0, plan_rows)
    indexed = store.resolve_verification_cluster_blocks(
        layer_name="layer",
        selected_cluster_indices=packed,
        plan_valid_rows=torch.ones(3, dtype=torch.bool, device=device),
        request_slot_ids=torch.zeros(3, dtype=torch.int64, device=device),
        request_slot_generations=torch.ones(3, dtype=torch.int64, device=device),
        arena=arena,
        max_pages_per_cluster=metadata.page_ids.shape[-1],
    )
    indexed.read_lease.release()

    gathered = store.resolve_verification_cluster_blocks(
        layer_name="layer",
        selected_cluster_indices=packed.clone(),
        plan_valid_rows=torch.ones(3, dtype=torch.bool, device=device),
        request_slot_ids=torch.zeros(3, dtype=torch.int64, device=device),
        request_slot_generations=torch.ones(3, dtype=torch.int64, device=device),
        arena=arena,
        max_pages_per_cluster=metadata.page_ids.shape[-1],
    )

    torch.testing.assert_close(indexed.resident_page_ids, gathered.resident_page_ids)
    torch.testing.assert_close(indexed.page_token_counts, gathered.page_token_counts)
    torch.testing.assert_close(indexed.page_counts, gathered.page_counts)
    assert indexed.miss_admission is None
    assert gathered.miss_admission is None
    gathered.read_lease.release()
    store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to resolve cluster pages",
)
def test_cpu_backing_store_resident_only_resolution_does_not_admit_misses():
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
    store.get_resident_page_storage("layer")
    torch.cuda.current_stream().synchronize()

    resident_pages_before = store.num_resident_pages("layer")
    resident_clusters_before = store.num_resident_clusters("layer")
    resolved = store.resolve_cluster_blocks(
        "layer",
        cluster_ids,
        metadata.page_ids,
        mode="resident_only",
    )

    assert torch.equal(
        resolved.resident_page_ids,
        access.cache_page_ids,
    )
    assert torch.all(resolved.staging_page_ids == -1)
    assert resolved.staging_key_pages.shape[0] == 0
    assert resolved.staging_value_pages.shape[0] == 0
    assert resolved.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert resolved.miss_cluster_mask.tolist() == [
        [False, True],
        [False, True],
    ]
    assert store.num_resident_pages("layer") == resident_pages_before
    assert store.num_resident_clusters("layer") == resident_clusters_before


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to resolve pending cluster pages",
)
def test_cpu_backing_store_exposes_pending_pages_only_when_requested():
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

    store.admit_resident_clusters("layer", cluster_ids, metadata.page_ids)
    resident_cache = store._resident_caches["layer"]
    resident_cache._reap_completed_copy_batches = Mock()

    draft = store.resolve_cluster_blocks(
        "layer",
        cluster_ids,
        metadata.page_ids,
        mode="resident_only",
    )
    first_draft = store.resolve_cluster_blocks(
        "layer",
        cluster_ids,
        metadata.page_ids,
        mode="resident_pending",
    )
    verification = store.resolve_cluster_blocks(
        "layer",
        cluster_ids,
        metadata.page_ids,
        mode="verification",
    )

    assert not draft.hit_cluster_mask.any()
    assert draft.miss_cluster_mask.all()
    assert torch.all(draft.resident_page_ids == -1)
    assert draft.resident_ready_event is None

    assert first_draft.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert first_draft.miss_cluster_mask.tolist() == [
        [False, True],
        [False, True],
    ]
    assert first_draft.resident_ready_event is not None
    assert torch.all(first_draft.staging_page_ids == -1)
    assert first_draft.staging_key_pages.shape[0] == 0

    assert verification.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert verification.miss_cluster_mask.tolist() == [
        [False, True],
        [False, True],
    ]
    assert verification.resident_ready_event is not None

    resident_cache.synchronize_pending_copies()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for full-verification staging",
)
def test_cpu_backing_store_reuses_full_verification_buffer_across_layers():
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

    first_table = store_cluster_data(
        store,
        "first-layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    second_table = store_cluster_data(
        store,
        "second-layer",
        (keys + 1000).to(device),
        (values + 1000).to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )

    first_metadata_cpu = store.get_cluster_block_metadata(
        "first-layer", first_table.cluster_ids, device=torch.device("cpu")
    )
    first_descriptor = store.build_full_verification_descriptor(
        "first-layer",
        first_metadata_cpu.page_ids,
        first_metadata_cpu.page_token_counts,
    )
    first_staging = store.resolve_full_verification_tokens(
        "first-layer", (first_descriptor,)
    )
    assert first_staging.ready_event is not None
    torch.cuda.current_stream(device).wait_event(first_staging.ready_event)
    first_key_snapshot = first_staging.key_tokens.clone()
    first_value_snapshot = first_staging.value_tokens.clone()

    first_key_ptr = first_staging.key_tokens.data_ptr()
    first_value_ptr = first_staging.value_tokens.data_ptr()
    resident_pages_before = store.num_resident_pages("first-layer")

    second_metadata_cpu = store.get_cluster_block_metadata(
        "second-layer", second_table.cluster_ids, device=torch.device("cpu")
    )
    second_descriptor = store.build_full_verification_descriptor(
        "second-layer",
        second_metadata_cpu.page_ids,
        second_metadata_cpu.page_token_counts,
    )
    second_staging = store.resolve_full_verification_tokens(
        "second-layer", (second_descriptor,)
    )
    assert second_staging.ready_event is not None
    torch.cuda.current_stream(device).wait_event(second_staging.ready_event)
    torch.testing.assert_close(first_key_snapshot.cpu(), first_staging.key_tokens.cpu())
    torch.testing.assert_close(
        first_value_snapshot.cpu(), first_staging.value_tokens.cpu()
    )
    torch.testing.assert_close(
        second_staging.key_tokens.cpu(), first_key_snapshot.cpu() + 1000
    )
    torch.testing.assert_close(
        second_staging.value_tokens.cpu(), first_value_snapshot.cpu() + 1000
    )

    second_key_ptr = second_staging.key_tokens.data_ptr()
    second_value_ptr = second_staging.value_tokens.data_ptr()
    assert second_key_ptr != first_key_ptr
    assert second_value_ptr != first_value_ptr
    assert len(store._full_verification_buffers) == 1
    transfer_buffer = store._full_verification_buffers[device]
    assert transfer_buffer._gpu_arenas[0].key_tokens.data_ptr() == first_key_ptr
    assert transfer_buffer._gpu_arenas[0].value_tokens.data_ptr() == first_value_ptr
    assert transfer_buffer._gpu_arenas[1].key_tokens.data_ptr() == second_key_ptr
    assert transfer_buffer._gpu_arenas[1].value_tokens.data_ptr() == second_value_ptr
    assert transfer_buffer._gpu_arena_cursor == 0
    assert store.num_resident_pages("first-layer") == resident_pages_before
    assert store.num_resident_pages("second-layer") == 0
    assert stats._cpu_counters["full_verify_h2d_tokens"] == 20
    assert stats._cpu_counters["full_verify_h2d_bytes"] == 160
    torch.cuda.synchronize(device)
    stats._drain_cuda_samples()
    assert stats._cuda_times["full_verify_h2d"][1] == 2
    assert stats._cpu_times["full_verify_cpu_gather"][1] == 2


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for asynchronous full-verification staging",
)
def test_full_verification_submission_gathers_on_background_worker(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(page_size=2, cache_ratio=0.5)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    metadata = store.get_cluster_block_metadata(
        "layer", table.cluster_ids, device=torch.device("cpu")
    )
    descriptor = store.build_full_verification_descriptor(
        "layer", metadata.page_ids, metadata.page_token_counts
    )

    gather_started = threading.Event()
    allow_gather = threading.Event()
    original_gather = cluster_store_module.ops.retrospec_gather_compact_kv

    def delayed_gather(*args):
        gather_started.set()
        if not allow_gather.wait(timeout=5):
            raise TimeoutError("Timed out waiting to release native gather")
        original_gather(*args)

    monkeypatch.setattr(
        cluster_store_module.ops,
        "retrospec_gather_compact_kv",
        delayed_gather,
    )

    try:
        ticket = store.submit_full_verification_tokens("layer", (descriptor,))
        assert gather_started.wait(timeout=5)
        assert not ticket.future.done()

        allow_gather.set()
        staging = ticket.result()
        assert staging.ready_event is not None
        torch.cuda.current_stream(device).wait_event(staging.ready_event)
        expected_keys = torch.cat((keys[0], keys[1, [1, 3, 0, 2, 4]]))
        expected_values = torch.cat((values[0], values[1, [1, 3, 0, 2, 4]]))
        torch.testing.assert_close(staging.key_tokens.cpu(), expected_keys)
        torch.testing.assert_close(staging.value_tokens.cpu(), expected_values)
    finally:
        allow_gather.set()
        store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for cancellable full-verification staging",
)
def test_full_verification_submission_cancels_during_cpu_gather(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(page_size=2, cache_ratio=0.5)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    metadata = store.get_cluster_block_metadata(
        "layer", table.cluster_ids, device=torch.device("cpu")
    )
    descriptor = store.build_full_verification_descriptor(
        "layer", metadata.page_ids, metadata.page_token_counts
    )

    gather_started = threading.Event()
    allow_gather = threading.Event()
    original_gather = cluster_store_module.ops.retrospec_gather_compact_kv

    def delayed_gather(*args):
        gather_started.set()
        if not allow_gather.wait(timeout=5):
            raise TimeoutError("Timed out waiting to release native gather")
        original_gather(*args)

    monkeypatch.setattr(
        cluster_store_module.ops,
        "retrospec_gather_compact_kv",
        delayed_gather,
    )

    try:
        ticket = store.submit_full_verification_tokens("layer", (descriptor,))
        assert gather_started.wait(timeout=5)
        assert not ticket.cancel()
        assert ticket.cancel_event.is_set()
        allow_gather.set()
        with pytest.raises(CancelledError):
            ticket.result()
    finally:
        allow_gather.set()
        store.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA and pinned host memory are required",
)
def test_full_verification_buffer_grows_for_a_larger_layer():
    device = torch.device("cuda", torch.cuda.current_device())
    store = RetroSpecClusterPageStore(
        page_size=2,
        pin_memory=True,
        max_pinned_memory_bytes=64,
    )

    small_keys = torch.arange(4, dtype=torch.float16, device=device).view(1, 4, 1)
    small_table = store_cluster_data(
        store,
        "small-layer",
        small_keys,
        small_keys + 100,
        torch.zeros(1, 4, dtype=torch.int64, device=device),
        torch.tensor([[4]], dtype=torch.int32, device=device),
    )
    small_metadata_cpu = store.get_cluster_block_metadata(
        "small-layer", small_table.cluster_ids, device=torch.device("cpu")
    )
    small_descriptor = store.build_full_verification_descriptor(
        "small-layer",
        small_metadata_cpu.page_ids,
        small_metadata_cpu.page_token_counts,
    )
    small_staging = store.resolve_full_verification_tokens(
        "small-layer", (small_descriptor,)
    )
    small_key_ptr = small_staging.key_tokens.data_ptr()

    large_keys = torch.arange(130, dtype=torch.float16, device=device).view(1, 130, 1)
    large_table = store_cluster_data(
        store,
        "large-layer",
        large_keys,
        large_keys + 100,
        torch.zeros(1, 130, dtype=torch.int64, device=device),
        torch.tensor([[130]], dtype=torch.int32, device=device),
    )
    large_metadata_cpu = store.get_cluster_block_metadata(
        "large-layer", large_table.cluster_ids, device=torch.device("cpu")
    )
    large_descriptor = store.build_full_verification_descriptor(
        "large-layer",
        large_metadata_cpu.page_ids,
        large_metadata_cpu.page_token_counts,
    )
    large_staging = store.resolve_full_verification_tokens(
        "large-layer", (large_descriptor,)
    )
    assert large_staging.ready_event is not None
    torch.cuda.current_stream(device).wait_event(large_staging.ready_event)

    torch.testing.assert_close(
        large_staging.key_tokens.cpu(), large_keys[:, :, 0].reshape(-1, 1).cpu()
    )
    torch.testing.assert_close(
        large_staging.value_tokens.cpu(),
        (large_keys[:, :, 0] + 100).reshape(-1, 1).cpu(),
    )
    transfer_buffer = store._full_verification_buffers[device]
    assert transfer_buffer.capacity == 256
    assert large_staging.key_tokens.data_ptr() != small_key_ptr
    assert len(transfer_buffer._cpu_slots) == 2
    assert all(slot.key_pages.is_pinned() for slot in transfer_buffer._cpu_slots)
    pinned_bytes = sum(
        slot.key_pages.nbytes + slot.value_pages.nbytes
        for slot in transfer_buffer._cpu_slots
    )
    assert pinned_bytes <= store.max_pinned_memory_bytes // 2


def test_cluster_store_rejects_invalid_resolve_mode():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = get_block_metadata(store, table)

    with pytest.raises(ValueError, match="Unsupported RetroSpec cluster resolve mode"):
        store.resolve_cluster_blocks(
            "layer",
            table.cluster_ids,
            metadata.page_ids,
            mode="invalid",  # type: ignore[arg-type]
        )
