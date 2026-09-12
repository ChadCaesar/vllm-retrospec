# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import CancelledError, Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

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
    RetroSpecResidentPrefetchInput,
)
from vllm.v1.spec_decode.retrospec.index_residency import RetroSpecResidentLayerArena
from vllm.v1.spec_decode.retrospec.performance import RetroSpecPerformanceStats


def make_cluster_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    keys = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[10.0], [11.0], [12.0], [13.0], [14.0]],
        ]
    )
    values = keys + 100.0
    assignments = torch.tensor(
        [
            [0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1],
        ],
        dtype=torch.int64,
    )
    cluster_token_counts = torch.tensor(
        [[3, 2], [2, 3]],
        dtype=torch.int32,
    )
    return keys, values, assignments, cluster_token_counts


def make_token_offsets(
    assignments: torch.Tensor,
    cluster_token_counts: torch.Tensor,
) -> torch.Tensor:
    num_heads, num_tokens = assignments.shape
    offsets = torch.empty_like(assignments, dtype=torch.int32)

    for head_idx in range(num_heads):
        next_offsets = torch.zeros(
            cluster_token_counts.shape[1],
            dtype=torch.int32,
            device=assignments.device,
        )
        for token_idx in range(num_tokens):
            cluster_idx = int(assignments[head_idx, token_idx].item())
            if not 0 <= cluster_idx < cluster_token_counts.shape[1]:
                offsets[head_idx, token_idx] = 0
                continue
            offsets[head_idx, token_idx] = next_offsets[cluster_idx]
            next_offsets[cluster_idx] += 1

    return offsets


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


def store_cluster_data(
    store: RetroSpecClusterPageStore,
    layer_name: str,
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
    assignments: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    request_id: str = "request",
    cluster_start: int = 0,
):
    return store.store_clusters(
        layer_name=layer_name,
        request_id=request_id,
        cluster_start=cluster_start,
        token_keys=token_keys,
        token_values=token_values,
        assignments=assignments,
        cluster_token_counts=cluster_token_counts,
        token_offsets_in_cluster=make_token_offsets(
            assignments,
            cluster_token_counts,
        ),
    )


def get_block_metadata(store, table, device=None):
    return store.get_cluster_block_metadata(
        layer_name="layer",
        cluster_ids=table.cluster_ids,
        device=device,
    )


def get_runtime_blocks(store, table, device):
    cluster_ids = table.cluster_ids.to(device=device)
    metadata = store.get_cluster_block_metadata(
        layer_name="layer",
        cluster_ids=cluster_ids,
        device=device,
    )
    return cluster_ids, metadata


def make_resident_arena(
    table, metadata, cluster_token_counts, device
) -> RetroSpecResidentLayerArena:
    num_kv_heads, num_clusters, max_pages = metadata.page_ids.shape
    cluster_shape = (num_kv_heads, num_clusters, 1)
    return RetroSpecResidentLayerArena(
        cluster_ids=table.cluster_ids.to(device),
        cluster_keys=torch.zeros(cluster_shape, device=device),
        cluster_values=torch.zeros(cluster_shape, device=device),
        cluster_token_counts=cluster_token_counts.to(device),
        cluster_page_starts=torch.arange(
            num_clusters, dtype=torch.int64, device=device
        )[None, :].expand(num_kv_heads, -1)
        * max_pages,
        cluster_page_counts=(metadata.page_ids >= 0).sum(dim=-1, dtype=torch.int32),
        resident_table_buckets=torch.full(
            (num_kv_heads, num_clusters),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        page_ids=metadata.page_ids.flatten(1),
        page_token_counts=metadata.page_token_counts.flatten(1),
        cluster_offsets=torch.zeros(1, dtype=torch.int64, device=device),
        num_clusters=torch.full((1,), num_clusters, dtype=torch.int32, device=device),
        page_offsets=torch.zeros(1, dtype=torch.int64, device=device),
        num_pages=torch.full(
            (1,), num_clusters * max_pages, dtype=torch.int32, device=device
        ),
        generations=torch.ones(1, dtype=torch.int64, device=device),
        indexed_starts=torch.zeros(1, dtype=torch.int64, device=device),
        indexed_ends=torch.ones(1, dtype=torch.int64, device=device),
    )


def materialize_resolved_pages(resolved):
    page_shape = resolved.resident_key_pages.shape[1:]
    output_shape = (*resolved.resident_page_ids.shape, *page_shape)
    keys = torch.zeros(
        output_shape,
        dtype=resolved.resident_key_pages.dtype,
        device=resolved.resident_key_pages.device,
    )
    values = torch.zeros_like(keys)

    resident_mask = resolved.resident_page_ids >= 0
    if resident_mask.any():
        resident_slots = resolved.resident_page_ids[resident_mask].to(torch.int64)
        keys[resident_mask] = resolved.resident_key_pages.index_select(
            0, resident_slots
        )
        values[resident_mask] = resolved.resident_value_pages.index_select(
            0, resident_slots
        )

    staging_mask = resolved.staging_page_ids >= 0
    if staging_mask.any():
        staging_slots = resolved.staging_page_ids[staging_mask].to(torch.int64)
        keys[staging_mask] = resolved.staging_key_pages.index_select(0, staging_slots)
        values[staging_mask] = resolved.staging_value_pages.index_select(
            0, staging_slots
        )
    return keys, values, resident_mask | staging_mask


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
        resize=Mock(side_effect=RuntimeError("resident resize failed"))
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_pin_memory_available(),
    reason="CUDA pinned memory is required for asynchronous resident prefetch",
)
def test_cpu_backing_store_prefetches_resident_clusters_in_background():
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
    assert stats._cpu_times["prefetch_worker_wall"][1] == 1
    assert stats._cpu_times["prefetch_wait_wall"][1] >= 1

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


def test_resident_prefetch_native_batch_preserves_rank_priority_and_miss():
    ordered_ids, raw_counts = ops.retrospec_order_prefetch_misses(
        (
            torch.tensor([10, 11, 10, -1], dtype=torch.int64),
            torch.tensor([20, 21, -1], dtype=torch.int64),
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
    )

    assert tuple(record.tolist() for record in ordered_ids) == ([11, 10], [21, 20])
    assert raw_counts.tolist() == [3, 2]


def test_resident_prefetch_native_batch_ignores_unused_suffix():
    ordered_ids, raw_counts = ops.retrospec_order_prefetch_misses(
        (torch.tensor([-1, -1], dtype=torch.int64),),
        (torch.tensor([-1, -1], dtype=torch.int64),),
        (torch.tensor([0], dtype=torch.int32),),
        (1,),
        (2,),
    )

    assert ordered_ids[0].numel() == 0
    assert raw_counts.tolist() == [0]


def test_resident_prefetch_native_batch_rejects_invalid_prefix():
    with pytest.raises(RuntimeError, match="count exceeds"):
        ops.retrospec_order_prefetch_misses(
            (torch.tensor([10], dtype=torch.int64),),
            (torch.tensor([0], dtype=torch.int64),),
            (torch.tensor([2], dtype=torch.int32),),
            (1,),
            (1,),
        )


@pytest.mark.parametrize(
    ("cluster_id", "position", "error"),
    (
        (-1, 0, "invalid cluster handle"),
        (10, -1, "outside its layout"),
        (10, 1, "outside its layout"),
    ),
)
def test_resident_prefetch_native_batch_rejects_invalid_commands(
    cluster_id: int, position: int, error: str
):
    with pytest.raises(RuntimeError, match=error):
        ops.retrospec_order_prefetch_misses(
            (torch.tensor([cluster_id], dtype=torch.int64),),
            (torch.tensor([position], dtype=torch.int64),),
            (torch.tensor([1], dtype=torch.int32),),
            (1,),
            (1,),
        )


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
def test_resident_prefetch_releases_metadata_slot_before_descriptor_preparation():
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
    original_prepare = store._prepare_resident_prefetch_record

    def blocking_prepare(staged, ordered_cluster_ids):
        prepare_started.set()
        assert release_prepare.wait(timeout=10.0)
        return original_prepare(staged, ordered_cluster_ids)

    store._prepare_resident_prefetch_record = blocking_prepare
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
        assert all(
            not slot.in_use
            for slots in store._resident_prefetch_slots.values()
            for slot in slots
        )
    finally:
        release_prepare.set()

    store.wait_for_resident_prefetches()
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
