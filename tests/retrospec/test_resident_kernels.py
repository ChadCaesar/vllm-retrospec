# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.resident_kernels import (
    compact_resident_misses,
    lookup_resident_handles,
    scatter_staging_page_ids,
    update_resident_handles,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for resident handle kernels",
)


def _make_table(
    capacity: int = 8,
    max_pages: int = 2,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    return (
        torch.full((capacity,), -1, dtype=torch.int64, device=device),
        torch.zeros(capacity, dtype=torch.int32, device=device),
        torch.zeros(capacity, dtype=torch.int32, device=device),
        torch.full((capacity, max_pages), -1, dtype=torch.int32, device=device),
        torch.zeros(capacity, dtype=torch.bool, device=device),
        torch.zeros(capacity, dtype=torch.int64, device=device),
    )


def _lookup(
    cluster_handles: torch.Tensor,
    logical_page_ids: torch.Tensor,
    active_mask: torch.Tensor | None,
    table: tuple[torch.Tensor, ...],
    plan_row_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    output_batch = (
        cluster_handles.shape[0]
        if plan_row_indices is None
        else plan_row_indices.shape[0]
    )
    cluster_shape = (output_batch, *cluster_handles.shape[1:])
    page_shape = (*cluster_shape, logical_page_ids.shape[-1])
    output_page_slots = torch.empty(
        page_shape, dtype=logical_page_ids.dtype, device=logical_page_ids.device
    )
    output_hit_mask = torch.empty(
        cluster_shape, dtype=torch.bool, device=cluster_handles.device
    )
    output_miss_mask = torch.empty_like(output_hit_mask)
    output_gate_ready = torch.empty_like(output_hit_mask)
    output_access_kinds = torch.empty(
        cluster_shape, dtype=torch.uint8, device=cluster_handles.device
    )

    lookup_resident_handles(
        cluster_handles=cluster_handles,
        logical_page_ids=logical_page_ids,
        active_mask=active_mask,
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
        table_last_access_epochs=table[5],
        access_epoch=7,
        output_page_slots=output_page_slots,
        output_hit_mask=output_hit_mask,
        output_miss_mask=output_miss_mask,
        output_hit_gate_ready=output_gate_ready,
        output_access_kinds=output_access_kinds,
        plan_row_indices=plan_row_indices,
    )
    return (
        output_page_slots,
        output_hit_mask,
        output_miss_mask,
        output_gate_ready,
        output_access_kinds,
    )


def test_resident_handle_lookup_returns_slots_and_gpu_access_records():
    device = torch.device("cuda")
    table = _make_table()
    update_resident_handles(
        bucket_ids=torch.tensor([3, 4], dtype=torch.int32, device=device),
        cluster_handles=torch.tensor([3, 11], dtype=torch.int64, device=device),
        page_counts=torch.tensor([2, 1], dtype=torch.int32, device=device),
        page_slots=torch.tensor([[5, 6], [7, -1]], dtype=torch.int32, device=device),
        hit_gate_ready=torch.tensor([True, False], device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
    )

    handles = torch.tensor([[[3, 11]], [[3, -1]]], dtype=torch.int64, device=device)
    logical_pages = torch.tensor(
        [[[[20, 21], [22, -1]]], [[[20, 21], [-1, -1]]]],
        dtype=torch.int64,
        device=device,
    )
    outputs = _lookup(
        handles,
        logical_pages,
        torch.tensor([True, False], device=device),
        table,
    )

    assert outputs[0].cpu().tolist() == [[[[5, 6], [7, -1]]], [[[-1, -1], [-1, -1]]]]
    assert outputs[1].cpu().tolist() == [[[True, True]], [[False, False]]]
    assert outputs[2].cpu().tolist() == [[[False, False]], [[False, False]]]
    assert outputs[3].cpu().tolist() == [[[True, False]], [[False, False]]]
    assert outputs[4].cpu().tolist() == [[[1, 1]], [[0, 0]]]
    assert table[5].cpu().tolist()[3:5] == [7, 7]


def test_resident_handle_lookup_reports_tombstone_and_unknown_handle_as_miss():
    device = torch.device("cuda")
    table = _make_table()
    update_resident_handles(
        bucket_ids=torch.tensor([3], dtype=torch.int32, device=device),
        cluster_handles=torch.tensor([-2], dtype=torch.int64, device=device),
        page_counts=torch.tensor([0], dtype=torch.int32, device=device),
        page_slots=torch.tensor([[-1, -1]], dtype=torch.int32, device=device),
        hit_gate_ready=torch.tensor([False], device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
    )

    handles = torch.tensor([[[3, 9]]], dtype=torch.int64, device=device)
    logical_pages = torch.tensor(
        [[[[20, -1], [30, -1]]]], dtype=torch.int64, device=device
    )
    outputs = _lookup(
        handles,
        logical_pages,
        torch.tensor([True], device=device),
        table,
    )

    assert outputs[0].cpu().tolist() == [[[[-1, -1], [-1, -1]]]]
    assert outputs[1].cpu().tolist() == [[[False, False]]]
    assert outputs[2].cpu().tolist() == [[[True, True]]]
    assert outputs[4].cpu().tolist() == [[[2, 2]]]


def test_resident_handle_lookup_can_activate_all_valid_verification_rows():
    device = torch.device("cuda")
    table = _make_table()
    update_resident_handles(
        bucket_ids=torch.tensor([3], dtype=torch.int32, device=device),
        cluster_handles=torch.tensor([3], dtype=torch.int64, device=device),
        page_counts=torch.tensor([1], dtype=torch.int32, device=device),
        page_slots=torch.tensor([[5, -1]], dtype=torch.int32, device=device),
        hit_gate_ready=torch.tensor([True], device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
    )

    handles = torch.tensor([[[3, 9]], [[3, -1]]], dtype=torch.int64, device=device)
    pages = torch.tensor(
        [[[[20, -1], [30, -1]]], [[[20, -1], [-1, -1]]]],
        dtype=torch.int64,
        device=device,
    )
    outputs = _lookup(handles, pages, None, table)

    assert outputs[1].cpu().tolist() == [[[True, False]], [[True, False]]]
    assert outputs[2].cpu().tolist() == [[[False, True]], [[False, False]]]
    assert outputs[4].cpu().tolist() == [[[1, 2]], [[1, 0]]]


def test_resident_handle_lookup_indexes_persistent_plan_rows():
    device = torch.device("cuda")
    table = _make_table()
    update_resident_handles(
        bucket_ids=torch.tensor([3, 4], dtype=torch.int32, device=device),
        cluster_handles=torch.tensor([3, 4], dtype=torch.int64, device=device),
        page_counts=torch.tensor([1, 2], dtype=torch.int32, device=device),
        page_slots=torch.tensor([[5, -1], [6, 7]], dtype=torch.int32, device=device),
        hit_gate_ready=torch.tensor([True, False], device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
    )
    handles = torch.tensor(
        [[[3, 9]], [[4, -1]], [[9, 3]], [[4, 3]]],
        dtype=torch.int64,
        device=device,
    )
    pages = torch.arange(16, dtype=torch.int64, device=device).view(4, 1, 2, 2)
    plan_rows = torch.tensor([3, 0, 1], dtype=torch.int64, device=device)

    indexed = _lookup(handles, pages, None, table, plan_rows)
    gathered = _lookup(
        handles.index_select(0, plan_rows),
        pages.index_select(0, plan_rows),
        None,
        table,
    )

    for actual, expected in zip(indexed, gathered):
        torch.testing.assert_close(actual, expected)


def test_compact_resident_misses_preserves_handles_and_flat_positions():
    device = torch.device("cuda")
    handles = torch.tensor(
        [[[10, 11, -1], [12, 10, 13]]], dtype=torch.int64, device=device
    )
    misses = torch.tensor([[[False, True, True], [True, True, False]]], device=device)
    output_handles = torch.empty(handles.numel(), dtype=torch.int64, device=device)
    output_positions = torch.empty_like(output_handles)
    output_count = torch.empty(1, dtype=torch.int32, device=device)

    compact_resident_misses(
        handles,
        misses,
        output_handles,
        output_positions,
        output_count,
    )

    count = int(output_count.item())
    records = sorted(
        zip(
            output_positions[:count].cpu().tolist(),
            output_handles[:count].cpu().tolist(),
        )
    )
    assert records == [(1, 11), (3, 12), (4, 10)]


def test_compact_resident_misses_indexes_source_plan_rows():
    device = torch.device("cuda")
    handles = torch.tensor(
        [[[10, 11]], [[20, 21]], [[30, 31]]],
        dtype=torch.int64,
        device=device,
    )
    misses = torch.tensor(
        [[[True, False]], [[False, True]], [[True, True]]], device=device
    )
    plan_rows = torch.tensor([2, 0, 1], dtype=torch.int64, device=device)
    output_handles = torch.empty(misses.numel(), dtype=torch.int64, device=device)
    output_positions = torch.empty_like(output_handles)
    output_count = torch.empty(1, dtype=torch.int32, device=device)

    compact_resident_misses(
        handles,
        misses,
        output_handles,
        output_positions,
        output_count,
        plan_rows,
    )

    count = int(output_count.item())
    records = sorted(
        zip(
            output_positions[:count].cpu().tolist(),
            output_handles[:count].cpu().tolist(),
        )
    )
    assert records == [(0, 30), (3, 11), (4, 20), (5, 21)]


def test_scatter_staging_page_ids_expands_compact_occurrences():
    device = torch.device("cuda")
    output = torch.empty((2, 1, 3, 3), dtype=torch.int64, device=device)
    positions = torch.tensor([1, 3, 4], dtype=torch.int64, device=device)
    starts = torch.tensor([0, 2, 0], dtype=torch.int64, device=device)
    counts = torch.tensor([2, 1, 2], dtype=torch.int32, device=device)

    scatter_staging_page_ids(positions, starts, counts, 3, output)

    assert output.cpu().tolist() == [
        [[[-1, -1, -1], [0, 1, -1], [-1, -1, -1]]],
        [[[2, -1, -1], [0, 1, -1], [-1, -1, -1]]],
    ]
