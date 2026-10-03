# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from tests.retrospec.offload.cluster_store_helpers import (
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
    RetroSpecCompactTokenRange,
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationTicket,
)
from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecResidentTableBinding,
)


def test_resident_cache_lookup_avoids_global_lock_on_hot_path():
    store = RetroSpecClusterPageStore(page_size=2)
    resident_cache = Mock()
    lifecycle_lock = MagicMock()
    store._resident_caches["layer"] = resident_cache
    store._resident_state_lock = lifecycle_lock

    assert store._get_resident_cache_for_lookup("layer") is resident_cache
    lifecycle_lock.__enter__.assert_not_called()


def test_resident_binding_publication_reuses_cluster_identity_descriptors():
    residency = Mock()
    store = RetroSpecClusterPageStore(
        page_size=2,
        gpu_index_residency=residency,
    )
    group = RetroSpecClusterGroup("request", 3)
    store._cluster_block_descriptors["layer"] = {
        7: cluster_store_module._ClusterBlockDescriptor(
            identity=RetroSpecClusterIdentity(group=group, local_cluster_id=11),
            page_ids=(2,),
            page_token_counts=(2,),
        )
    }
    stream = Mock()

    store._publish_resident_table_bindings(
        "layer",
        (7, 8),
        (5, 6),
        stream,
    )

    residency.publish_resident_table_bindings.assert_called_once_with(
        layer_name="layer",
        bindings=[
            RetroSpecResidentTableBinding(
                request_id="request",
                kv_head_index=3,
                local_cluster_index=11,
                cluster_handle=7,
                table_bucket=5,
            )
        ],
        stream=stream,
    )


def test_resident_cache_lookup_serializes_cold_creation():
    store = RetroSpecClusterPageStore(page_size=2)
    resident_cache = Mock()
    lifecycle_lock = MagicMock()
    store._resident_state_lock = lifecycle_lock
    store._get_or_create_resident_cache = Mock(return_value=(Mock(), resident_cache))

    assert store._get_resident_cache_for_lookup("layer") is resident_cache
    lifecycle_lock.__enter__.assert_called_once_with()
    lifecycle_lock.__exit__.assert_called_once()
    store._get_or_create_resident_cache.assert_called_once_with("layer")


def test_full_verification_ticket_reports_ready_and_signals_cancellation():
    future = Future()
    cancel_event = threading.Event()
    ticket = RetroSpecFullVerificationTicket(future, cancel_event)

    assert not ticket.ready()
    assert ticket.cancel()
    assert cancel_event.is_set()
    assert not ticket.ready()

    completed_future = Future()
    completed_future.set_result(SimpleNamespace(ready_event=None))
    completed_ticket = RetroSpecFullVerificationTicket(
        completed_future, threading.Event()
    )
    assert completed_ticket.ready()


def test_resident_replay_freezes_prefetch_and_verification_admission(monkeypatch):
    store = RetroSpecClusterPageStore(page_size=2)
    synchronize = Mock()
    monkeypatch.setattr(store, "synchronize_resident_prefetches", synchronize)

    store.begin_resident_replay(("layer.0", "layer.0", "layer.1"))
    synchronize.assert_called_once_with(("layer.0", "layer.0", "layer.1"))
    assert store._resident_admission_frozen
    assert not store.prefetch_resident_cluster_wave((Mock(),))
    store.admit_verification_misses(Mock())

    with pytest.raises(RuntimeError, match="already active"):
        store.begin_resident_replay(("layer.0",))

    store.end_resident_replay()
    assert not store._resident_admission_frozen
    with pytest.raises(RuntimeError, match="not active"):
        store.end_resident_replay()
    store.close()


def test_full_verification_staging_ring_waits_for_released_slot():
    buffer = cluster_store_module._FullVerificationTransferBuffer.__new__(
        cluster_store_module._FullVerificationTransferBuffer
    )
    buffer._closed = False
    buffer._cpu_slot_lock = threading.Lock()
    buffer._cpu_slot_available = threading.Condition(buffer._cpu_slot_lock)
    buffer._cpu_slot_cursor = 0
    first_slot = SimpleNamespace(in_use=True, reuse_ready_event=None)
    second_slot = SimpleNamespace(in_use=True, reuse_ready_event=None)
    buffer._cpu_slots = [first_slot, second_slot]

    acquire_started = threading.Event()
    acquired_slots = []

    def acquire_slot():
        acquire_started.set()
        acquired_slots.append(buffer._acquire_cpu_slot())

    waiter = threading.Thread(target=acquire_slot)
    waiter.start()
    assert acquire_started.wait(timeout=1)
    waiter.join(timeout=0.1)
    assert waiter.is_alive()

    buffer.release_cpu_slot(first_slot, None)
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert acquired_slots == [first_slot]
    assert first_slot.in_use
    buffer.release_cpu_slot(first_slot, None)


def test_full_verification_descriptor_compacts_partial_pages():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    table = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    metadata = store.get_cluster_block_metadata(
        "layer", table.cluster_ids, device=torch.device("cpu")
    )
    descriptor = store.build_full_verification_descriptor(
        "layer", metadata.page_ids, metadata.page_token_counts
    )

    torch.testing.assert_close(
        table.full_verification_descriptor.range_table,
        descriptor.range_table,
    )
    torch.testing.assert_close(
        table.full_verification_descriptor.head_token_counts_tensor,
        descriptor.head_token_counts_tensor,
    )
    assert descriptor.head_token_counts == (5, 5)
    assert descriptor.num_tokens == 10
    assert (
        sum(
            token_range.token_count
            for head_ranges in descriptor.head_ranges
            for token_range in head_ranges
        )
        == 10
    )
    assert (metadata.page_ids >= 0).sum().item() * store.page_size == 12


def test_full_verification_descriptor_packs_head_local_range_offsets():
    descriptor = RetroSpecFullVerificationDescriptor(
        head_ranges=(
            (
                RetroSpecCompactTokenRange(0, 2, 3),
                RetroSpecCompactTokenRange(1, 0, 2),
            ),
            (RetroSpecCompactTokenRange(2, 4, 4),),
        ),
        head_token_counts=(5, 4),
    )

    torch.testing.assert_close(
        descriptor.range_table,
        torch.tensor(
            [
                [0, 0, 2, 3, 0],
                [0, 1, 0, 2, 3],
                [1, 2, 4, 4, 0],
            ],
            dtype=torch.int64,
        ),
    )


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_native_compact_gather_clips_multi_request_ranges_to_chunk(num_workers):
    key_slab = torch.arange(32, dtype=torch.float32).view(8, 2, 2)
    value_slab = key_slab + 100
    range_tables = (
        torch.tensor(
            [
                [0, 0, 0, 2, 0],
                [0, 0, 4, 1, 2],
                [1, 0, 8, 2, 0],
            ],
            dtype=torch.int64,
        ),
        torch.tensor(
            [
                [0, 0, 10, 1, 0],
                [1, 0, 12, 3, 0],
            ],
            dtype=torch.int64,
        ),
    )
    token_offsets = torch.tensor([[0, 3], [5, 6]], dtype=torch.int64)
    expected_keys = torch.cat(
        (
            key_slab.view(-1, 2)[0:2],
            key_slab.view(-1, 2)[4:5],
            key_slab.view(-1, 2)[8:10],
            key_slab.view(-1, 2)[10:11],
            key_slab.view(-1, 2)[12:15],
        )
    )
    key_output = torch.empty(5, 2)
    value_output = torch.empty_like(key_output)

    ops.retrospec_gather_compact_kv(
        (key_slab,),
        (value_slab,),
        range_tables,
        token_offsets,
        2,
        key_output,
        value_output,
        num_workers,
    )

    torch.testing.assert_close(key_output, expected_keys[2:7])
    torch.testing.assert_close(value_output, expected_keys[2:7] + 100)


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_native_compact_gather_parallelizes_across_slabs(num_workers):
    key_slabs = (
        torch.arange(16, dtype=torch.float32).view(2, 4, 2),
        torch.arange(100, 116, dtype=torch.float32).view(2, 4, 2),
    )
    value_slabs = tuple(key_slab + 1000 for key_slab in key_slabs)
    range_tables = (
        torch.tensor(
            [
                [0, 0, 1, 3, 0],
                [0, 1, 0, 2, 3],
                [1, 1, 2, 4, 0],
            ],
            dtype=torch.int64,
        ),
    )
    token_offsets = torch.tensor([[0, 5]], dtype=torch.int64)
    expected_keys = torch.cat(
        (
            key_slabs[0].view(-1, 2)[1:4],
            key_slabs[1].view(-1, 2)[0:2],
            key_slabs[1].view(-1, 2)[2:6],
        )
    )
    key_output = torch.empty(6, 2)
    value_output = torch.empty_like(key_output)

    ops.retrospec_gather_compact_kv(
        key_slabs,
        value_slabs,
        range_tables,
        token_offsets,
        2,
        key_output,
        value_output,
        num_workers,
    )

    torch.testing.assert_close(key_output, expected_keys[2:8])
    torch.testing.assert_close(value_output, expected_keys[2:8] + 1000)


def test_native_compact_gather_rejects_non_positive_worker_count():
    key_slab = torch.arange(4, dtype=torch.float32).view(1, 2, 2)
    value_slab = key_slab + 100
    range_table = torch.tensor([[0, 0, 0, 2, 0]], dtype=torch.int64)
    token_offsets = torch.tensor([[0]], dtype=torch.int64)
    key_output = torch.empty(2, 2)
    value_output = torch.empty_like(key_output)

    with pytest.raises(RuntimeError, match="worker count must be positive"):
        ops.retrospec_gather_compact_kv(
            (key_slab,),
            (value_slab,),
            (range_table,),
            token_offsets,
            0,
            key_output,
            value_output,
            0,
        )


def test_cluster_store_packs_per_head_clusters_across_pages():
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

    assert metadata.page_ids.shape == (2, 2, 2)
    assert table.cluster_ids.shape == (2, 2)
    assert table.cluster_ids.tolist() == [[0, 1], [2, 3]]
    torch.testing.assert_close(table.page_metadata.page_ids, metadata.page_ids)
    torch.testing.assert_close(
        table.page_metadata.page_token_counts,
        metadata.page_token_counts,
    )
    assert metadata.page_token_counts.tolist() == [
        [[2, 1], [2, 0]],
        [[2, 0], [2, 1]],
    ]
    assert store.num_allocated_pages("layer") == 6
    assert store.num_allocated_clusters("layer") == 4

    valid_page_ids = metadata.page_ids[metadata.page_ids >= 0]
    key_pages, value_pages = store.read_page_storage("layer", valid_page_ids)
    assert key_pages.shape[1:] == (2, 1)
    assert value_pages.shape == key_pages.shape

    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert token_mask[0, 0].tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert token_mask[0, 1].tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert gathered_keys[0, 1, token_mask[0, 1], 0].tolist() == [
        11.0,
        13.0,
        10.0,
        12.0,
        14.0,
    ]
    assert gathered_values[token_mask.unsqueeze(-1)].tolist() == pytest.approx(
        (gathered_keys[token_mask.unsqueeze(-1)] + 100.0).tolist()
    )
    assert not gathered_keys[~token_mask.unsqueeze(-1)].any()
    assert not gathered_values[~token_mask.unsqueeze(-1)].any()


def test_cluster_store_reuses_block_metadata_when_freeing(monkeypatch):
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
    metadata_lookup = Mock(
        side_effect=AssertionError("free rebuilt persistent block metadata")
    )
    monkeypatch.setattr(store, "get_cluster_block_metadata", metadata_lookup)

    store.free("layer", table)

    metadata_lookup.assert_not_called()
    assert store.num_allocated_pages("layer") == 0
    assert store.num_allocated_clusters("layer") == 0


def test_cluster_store_uses_gpu_generated_offsets_without_sorting_tokens():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    token_offsets = torch.tensor(
        [
            [2, 0, 1, 1, 0],
            [2, 1, 0, 0, 1],
        ],
        dtype=torch.int32,
    )

    table = store.store_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        token_keys=keys,
        token_values=values,
        assignments=assignments,
        cluster_token_counts=cluster_token_counts,
        token_offsets_in_cluster=token_offsets,
    )
    metadata = get_block_metadata(store, table)
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert gathered_keys[0, 0, token_mask[0, 0], 0].tolist() == [1, 2, 0, 4, 3]
    assert gathered_keys[0, 1, token_mask[0, 1], 0].tolist() == [13, 11, 12, 14, 10]
    torch.testing.assert_close(
        gathered_values[token_mask.unsqueeze(-1)],
        gathered_keys[token_mask.unsqueeze(-1)] + 100,
    )


@pytest.mark.parametrize(
    ("page_size", "num_kv_heads", "num_tokens", "num_clusters", "head_size"),
    [
        (1, 1, 7, 3, 2),
        (2, 2, 11, 5, 3),
        (4, 3, 13, 7, 1),
    ],
)
@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is required for GPU cluster packing",
            ),
        ),
    ],
)
def test_cluster_store_native_packing_matches_cluster_membership(
    page_size,
    num_kv_heads,
    num_tokens,
    num_clusters,
    head_size,
    device_type,
):
    device = torch.device(device_type)
    generator = torch.Generator().manual_seed(
        page_size * 1000 + num_kv_heads * 100 + num_clusters
    )
    assignments = torch.randint(
        num_clusters,
        (num_kv_heads, num_tokens),
        generator=generator,
        dtype=torch.int64,
    )
    cluster_token_counts = torch.zeros(
        num_kv_heads,
        num_clusters,
        dtype=torch.int32,
    )
    cluster_token_counts.scatter_add_(
        1,
        assignments,
        torch.ones_like(assignments, dtype=torch.int32),
    )

    keys = torch.arange(
        num_kv_heads * num_tokens * head_size,
        dtype=torch.float32,
    ).view(num_kv_heads, num_tokens, head_size)
    values = keys + 1000
    store = RetroSpecClusterPageStore(page_size=page_size)
    table = store_cluster_data(
        store,
        "layer",
        keys.to(device),
        values.to(device),
        assignments.to(device),
        cluster_token_counts.to(device),
    )
    metadata = get_block_metadata(store, table)
    for head_index in range(num_kv_heads):
        for cluster_index in range(num_clusters):
            token_count = int(cluster_token_counts[head_index, cluster_index])
            expected_keys = keys[head_index][assignments[head_index] == cluster_index]
            expected_values = values[head_index][
                assignments[head_index] == cluster_index
            ]

            valid_pages = metadata.page_ids[head_index, cluster_index] >= 0
            cluster_page_ids = metadata.page_ids[
                head_index,
                cluster_index,
                valid_pages,
            ].to(dtype=torch.int64)
            cluster_page_token_counts = metadata.page_token_counts[
                head_index,
                cluster_index,
                valid_pages,
            ]

            if token_count == 0:
                assert table.cluster_ids[head_index, cluster_index] == -1
                assert cluster_page_ids.numel() == 0
                continue

            cluster_key_pages, cluster_value_pages = store.read_page_storage(
                "layer", cluster_page_ids
            )
            token_mask = torch.arange(
                page_size,
                device=cluster_page_token_counts.device,
            ).view(1, page_size) < cluster_page_token_counts.view(-1, 1)

            torch.testing.assert_close(cluster_key_pages[token_mask], expected_keys)
            torch.testing.assert_close(
                cluster_value_pages[token_mask],
                expected_values,
            )
            assert not cluster_key_pages[~token_mask].any()
            assert not cluster_value_pages[~token_mask].any()


def test_cluster_store_native_builder_writes_final_slabs_once(monkeypatch):
    store = RetroSpecClusterPageStore(page_size=2, cpu_page_build_workers=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    pool = store._get_or_create_pool("layer", keys)
    native_builder = Mock(wraps=ops.retrospec_build_cluster_pages)
    monkeypatch.setattr(
        cluster_store_module.ops,
        "retrospec_build_cluster_pages",
        native_builder,
    )
    pool.write = Mock(side_effect=AssertionError("legacy page copy was used"))

    table = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )

    native_builder.assert_called_once()
    assert native_builder.call_args.args[-1] == 2
    pool.write.assert_not_called()
    assert table.full_verification_descriptor.head_token_counts == (5, 5)
    torch.testing.assert_close(
        table.full_verification_descriptor.head_token_counts_tensor,
        torch.tensor([5, 5], dtype=torch.int32),
    )


@pytest.mark.parametrize("invalid_assignment", [-1, 2])
def test_cluster_store_rejects_assignment_outside_cluster_range(
    invalid_assignment,
):
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    assignments[0, 0] = invalid_assignment

    with pytest.raises(RuntimeError, match="assignment count"):
        store_cluster_data(
            store,
            "layer",
            keys,
            values,
            assignments,
            cluster_token_counts,
        )

    assert store.num_allocated_pages("layer") == 0


def test_cluster_store_tracks_request_head_local_cluster_identities():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="first",
        cluster_start=4,
    )
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="second",
        cluster_start=4,
    )
    other_layer = store_cluster_data(
        store,
        "other-layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="first",
        cluster_start=4,
    )

    first_identities = store.get_cluster_identities("layer", first.cluster_ids)
    second_identities = store.get_cluster_identities("layer", second.cluster_ids)
    other_identities = store.get_cluster_identities(
        "other-layer", other_layer.cluster_ids
    )

    assert first.cluster_ids.tolist() == [[0, 1], [2, 3]]
    assert second.cluster_ids.tolist() == [[4, 5], [6, 7]]
    assert other_layer.cluster_ids.tolist() == [[0, 1], [2, 3]]
    assert first_identities == {
        0: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 0), 4),
        1: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 0), 5),
        2: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 1), 4),
        3: RetroSpecClusterIdentity(RetroSpecClusterGroup("first", 1), 5),
    }
    assert second_identities == {
        4: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 0), 4),
        5: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 0), 5),
        6: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 1), 4),
        7: RetroSpecClusterIdentity(RetroSpecClusterGroup("second", 1), 5),
    }
    assert other_identities == first_identities

    selected = torch.tensor([[3, 0, 3, -1]], dtype=torch.int64)
    selected_identities = store.get_cluster_identities("layer", selected)
    assert list(selected_identities) == [3, 0]

    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("first", 0): 3,
        RetroSpecClusterGroup("first", 1): 3,
        RetroSpecClusterGroup("second", 0): 3,
        RetroSpecClusterGroup("second", 1): 3,
    }
    assert store._group_backing_page_counts["other-layer"] == {
        RetroSpecClusterGroup("first", 0): 3,
        RetroSpecClusterGroup("first", 1): 3,
    }


def test_cluster_store_distributes_soft_targets_by_backing_page_ownership():
    store = RetroSpecClusterPageStore(
        page_size=2,
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="first",
    )
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="second",
    )

    capacity = store._resident_target_capacity(store._layer_pools["layer"])
    assert capacity == 6
    assert store._resident_group_targets("layer", capacity) == {
        RetroSpecClusterGroup("first", 0): 2,
        RetroSpecClusterGroup("first", 1): 2,
        RetroSpecClusterGroup("second", 0): 1,
        RetroSpecClusterGroup("second", 1): 1,
    }
    assert store.resident_group_target_pages(
        "layer", ["first", "missing", "second"], num_kv_heads=2
    ) == ((2, 2), (0, 0), (1, 1))

    store.free("layer", first)
    capacity = store._resident_target_capacity(store._layer_pools["layer"])
    assert capacity == 3
    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("second", 0): 3,
        RetroSpecClusterGroup("second", 1): 3,
    }
    assert store._resident_group_targets("layer", capacity) == {
        RetroSpecClusterGroup("second", 0): 2,
        RetroSpecClusterGroup("second", 1): 1,
    }

    store.free("layer", second)
    assert store._group_backing_page_counts["layer"] == {}
    assert store._resident_group_targets("layer", 0) == {}


def test_cluster_store_synchronizes_background_prefetch_and_resident_copies():
    store = RetroSpecClusterPageStore(page_size=2)
    first_cache = Mock()
    second_cache = Mock()
    store._resident_caches = {"first": first_cache, "second": second_cache}
    store.wait_for_resident_prefetches = Mock()

    store.synchronize_resident_prefetches(["second", "first", "second", "missing"])

    store.wait_for_resident_prefetches.assert_called_once_with(
        ("second", "first", "missing")
    )
    first_cache.synchronize_pending_copies.assert_called_once_with()
    second_cache.synchronize_pending_copies.assert_called_once_with()


def test_cluster_store_accumulates_group_pages_across_request_segments():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="request",
        cluster_start=0,
    )
    second = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
        request_id="request",
        cluster_start=2,
    )

    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("request", 0): 6,
        RetroSpecClusterGroup("request", 1): 6,
    }

    store.free("layer", first)
    assert store._group_backing_page_counts["layer"] == {
        RetroSpecClusterGroup("request", 0): 3,
        RetroSpecClusterGroup("request", 1): 3,
    }

    store.free("layer", second)
    assert store._group_backing_page_counts["layer"] == {}


def test_cluster_store_rejects_drift_in_group_page_accounting():
    store = RetroSpecClusterPageStore(
        page_size=2,
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    group = RetroSpecClusterGroup("request", 0)
    store._group_backing_page_counts["layer"][group] += 1

    with pytest.raises(RuntimeError, match="does not match the layer page pool"):
        store._resident_group_targets("layer", capacity=3)


def test_cluster_identity_rejects_negative_indices():
    with pytest.raises(ValueError, match="kv_head_index"):
        RetroSpecClusterGroup(request_id="request", kv_head_index=-1)

    with pytest.raises(ValueError, match="local_cluster_id"):
        RetroSpecClusterIdentity(
            group=RetroSpecClusterGroup("request", 0),
            local_cluster_id=-1,
        )


def test_cluster_store_frees_and_reuses_pages():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()

    first = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    first_metadata = get_block_metadata(store, first)
    first_page_ids = set(first_metadata.page_ids[first_metadata.page_ids >= 0].tolist())
    first_cluster_ids = set(first.cluster_ids[first.cluster_ids >= 0].tolist())
    store.free("layer", first)

    assert store.num_allocated_pages("layer") == 0
    assert store.num_allocated_clusters("layer") == 0

    second = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    second_metadata = get_block_metadata(store, second)
    second_page_ids = set(
        second_metadata.page_ids[second_metadata.page_ids >= 0].tolist()
    )
    second_cluster_ids = set(second.cluster_ids[second.cluster_ids >= 0].tolist())

    assert second_page_ids == first_page_ids
    assert second_cluster_ids.isdisjoint(first_cluster_ids)
    assert store.num_allocated_pages("layer") == 6


def test_cluster_store_releases_pages_when_packing_fails():
    store = RetroSpecClusterPageStore(page_size=2)
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    invalid_counts = cluster_token_counts.clone()
    invalid_counts[0, 0] = 2

    with pytest.raises(RuntimeError, match="assignment count"):
        store_cluster_data(
            store,
            "layer",
            keys,
            values,
            assignments,
            invalid_counts,
        )

    assert store.num_allocated_pages("layer") == 0


def test_cluster_store_releases_pages_when_resident_resize_fails():
    store = RetroSpecClusterPageStore(
        page_size=2,
        cache_ratio=0.5,
    )
    keys, values, assignments, cluster_token_counts = make_cluster_data()
    first = store_cluster_data(
        store, "layer", keys, values, assignments, cluster_token_counts
    )
    allocated_before = store.num_allocated_pages("layer")
    clusters_before = store.num_allocated_clusters("layer")
    group_page_counts_before = dict(store._group_backing_page_counts["layer"])
    store._resident_caches["layer"] = Mock(
        requires_resize=Mock(return_value=True),
        mutation_guard=Mock(return_value=nullcontext()),
        resize=Mock(side_effect=RuntimeError("resident resize failed")),
    )

    with pytest.raises(RuntimeError, match="resident resize failed"):
        store_cluster_data(
            store, "layer", keys, values, assignments, cluster_token_counts
        )

    assert store.num_allocated_pages("layer") == allocated_before
    assert store.num_allocated_clusters("layer") == clusters_before
    assert store._group_backing_page_counts["layer"] == group_page_counts_before
    del store._resident_caches["layer"]
    store.free("layer", first)


def test_cluster_store_handles_empty_clusters():
    store = RetroSpecClusterPageStore(page_size=2)
    keys = torch.empty(1, 0, 3)
    values = torch.empty_like(keys)
    assignments = torch.empty(1, 0, dtype=torch.int64)
    cluster_token_counts = torch.zeros(1, 2, dtype=torch.int32)

    table = store_cluster_data(
        store,
        "layer",
        keys,
        values,
        assignments,
        cluster_token_counts,
    )
    metadata = get_block_metadata(store, table)
    gathered_keys, gathered_values, token_mask = store.gather_pages(
        "layer",
        metadata.page_ids.unsqueeze(0),
        metadata.page_token_counts.unsqueeze(0),
    )

    assert metadata.page_ids.shape == (1, 2, 0)
    assert table.cluster_ids.tolist() == [[-1, -1]]
    assert gathered_keys.shape == (1, 1, 0, 3)
    assert gathered_values.shape == gathered_keys.shape
    assert token_mask.shape == (1, 1, 0)
    assert store.num_allocated_pages("layer") == 0
    assert store.num_allocated_clusters("layer") == 0
    assert store.get_cluster_identities("layer", table.cluster_ids) == {}


def test_cluster_store_rejects_storage_access_before_allocation():
    store = RetroSpecClusterPageStore(page_size=2)

    with pytest.raises(RuntimeError, match="No RetroSpec page pool"):
        store.read_page_storage("missing", torch.empty(0, dtype=torch.int64))


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
