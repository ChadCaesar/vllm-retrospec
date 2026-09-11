# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.resident_kernels import (
    compact_resident_misses,
    lookup_resident_handles,
    resolve_compact_verification_pages,
    resolve_ranked_compact_draft_pages,
    scatter_compact_staging_page_ids,
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


def test_ranked_compact_draft_resolution_emits_journal_pages_and_misses():
    device = torch.device("cuda")
    table = _make_table(capacity=8, max_pages=2)
    update_resident_handles(
        bucket_ids=torch.tensor([2], dtype=torch.int32, device=device),
        cluster_handles=torch.tensor([10], dtype=torch.int64, device=device),
        page_counts=torch.tensor([2], dtype=torch.int32, device=device),
        page_slots=torch.tensor([[5, 6]], dtype=torch.int32, device=device),
        hit_gate_ready=torch.tensor([True], device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
    )

    ranked_values = torch.tensor([[[0.6, 0.3, 0.1]], [[0.9, 0.1, 0.0]]], device=device)
    ranked_indices = torch.tensor(
        [[[0, 1, 2]], [[2, 1, 0]]], dtype=torch.int64, device=device
    )
    candidate_counts = torch.tensor([[3], [0]], dtype=torch.int32, device=device)
    arena_cluster_ids = torch.tensor(
        [[10, 11, 12, -1]], dtype=torch.int64, device=device
    )
    arena_cluster_page_starts = torch.tensor(
        [[0, 2, 3, 0]], dtype=torch.int32, device=device
    )
    arena_cluster_page_counts = torch.tensor(
        [[2, 1, 1, 0]], dtype=torch.int32, device=device
    )
    arena_page_ids = torch.tensor(
        [[100, 101, 102, 103]], dtype=torch.int64, device=device
    )
    arena_page_token_counts = torch.tensor(
        [[2, 1, 2, 2]], dtype=torch.int32, device=device
    )

    sparse_indices = torch.empty((2, 1, 2), dtype=torch.int32, device=device)
    expanded_indices = torch.empty((2, 1, 3), dtype=torch.int32, device=device)
    cluster_handles = torch.empty((2, 1, 2), dtype=torch.int64, device=device)
    resident_page_ids = torch.empty((2, 1, 4), dtype=torch.int64, device=device)
    page_token_counts = torch.empty_like(resident_page_ids, dtype=torch.int32)
    row_shape = (2, 1)
    page_counts = torch.empty(row_shape, dtype=torch.int32, device=device)
    clustered_token_counts = torch.empty_like(page_counts)
    hit_attention_by_head = torch.empty(row_shape, device=device)
    selected_counts = torch.empty_like(page_counts)
    hit_counts = torch.empty_like(page_counts)
    miss_counts = torch.empty_like(page_counts)
    gate_ready = torch.empty(row_shape, dtype=torch.bool, device=device)
    miss_handles = torch.empty(4, dtype=torch.int64, device=device)
    miss_positions = torch.empty_like(miss_handles)
    miss_count = torch.empty(1, dtype=torch.int32, device=device)
    fallback_counts = torch.tensor(
        [[[3, 2]], [[7, 8]]], dtype=torch.int32, device=device
    )
    draft_attention = torch.empty(2, device=device)
    sparse_attention = torch.empty(2, device=device)
    expanded_attention = torch.empty(2, device=device)

    resolve_ranked_compact_draft_pages(
        ranked_values=ranked_values,
        ranked_indices=ranked_indices,
        candidate_counts=candidate_counts,
        arena_cluster_ids=arena_cluster_ids,
        arena_cluster_page_starts=arena_cluster_page_starts,
        arena_cluster_page_counts=arena_cluster_page_counts,
        arena_page_ids=arena_page_ids,
        arena_page_token_counts=arena_page_token_counts,
        arena_cluster_offsets=torch.tensor([0], dtype=torch.int64, device=device),
        arena_page_offsets=torch.tensor([0], dtype=torch.int64, device=device),
        request_slot_ids=torch.tensor([0, -1], dtype=torch.int64, device=device),
        active_mask=torch.tensor([True, True], device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
        table_last_access_epochs=table[5],
        access_epoch=9,
        retrieval_ratio=0.5,
        estimation_ratio=0.34,
        max_pages_per_cluster=2,
        fallback_token_counts=fallback_counts,
        sparse_cluster_indices=sparse_indices,
        expanded_cluster_indices=expanded_indices,
        output_cluster_handles=cluster_handles,
        output_page_slots=resident_page_ids,
        output_page_token_counts=page_token_counts,
        output_page_counts=page_counts,
        output_clustered_token_counts=clustered_token_counts,
        output_attention=draft_attention,
        output_hit_attention_by_head=hit_attention_by_head,
        output_selected_counts=selected_counts,
        output_hit_counts=hit_counts,
        output_miss_counts=miss_counts,
        output_gate_ready=gate_ready,
        output_miss_handles=miss_handles,
        output_miss_positions=miss_positions,
        output_miss_count=miss_count,
        sparse_attention=sparse_attention,
        expanded_attention=expanded_attention,
    )

    assert sparse_indices.cpu().tolist() == [[[0, 1]], [[-1, -1]]]
    assert expanded_indices.cpu().tolist() == [[[0, 1, 2]], [[-1, -1, -1]]]
    assert cluster_handles.cpu().tolist() == [[[10, 11]], [[-1, -1]]]
    assert resident_page_ids.cpu().tolist() == [[[5, 6, -1, -1]], [[-1] * 4]]
    assert page_token_counts.cpu().tolist() == [[[2, 1, 0, 0]], [[0] * 4]]
    assert page_counts.cpu().tolist() == [[2], [0]]
    assert clustered_token_counts.cpu().tolist() == [[3], [0]]
    assert selected_counts.cpu().tolist() == [[2], [0]]
    assert hit_counts.cpu().tolist() == [[1], [0]]
    assert miss_counts.cpu().tolist() == [[1], [0]]
    assert gate_ready.cpu().tolist() == [[True], [False]]
    assert fallback_counts.cpu().tolist() == [[[0, 2]], [[0, 0]]]
    assert miss_count.item() == 1
    assert miss_handles[0].item() == 11
    assert miss_positions[0].item() == 1
    torch.testing.assert_close(draft_attention.cpu(), torch.tensor([0.6, 1.0]))
    torch.testing.assert_close(sparse_attention.cpu(), torch.tensor([0.9, 1.0]))
    torch.testing.assert_close(expanded_attention.cpu(), torch.tensor([1.0, 1.0]))
    assert table[5][2].item() == 9


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


def _compact_verification_outputs(
    num_queries: int,
    num_kv_heads: int,
    num_clusters: int,
    max_pages: int,
    miss_page_stride: int | None = None,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    page_shape = (num_queries, num_kv_heads, num_clusters * max_pages)
    row_shape = (num_queries, num_kv_heads)
    miss_capacity = num_queries * num_kv_heads * num_clusters
    miss_page_stride = miss_page_stride or max_pages
    miss_page_storage = torch.empty(
        (miss_capacity, miss_page_stride), dtype=torch.int64, device=device
    )
    return (
        torch.empty(page_shape, dtype=torch.int64, device=device),
        torch.empty(page_shape, dtype=torch.int64, device=device),
        torch.empty(page_shape, dtype=torch.int32, device=device),
        torch.empty(row_shape, dtype=torch.int32, device=device),
        torch.empty(row_shape, dtype=torch.int32, device=device),
        torch.empty(row_shape, dtype=torch.int32, device=device),
        torch.empty(row_shape, dtype=torch.int32, device=device),
        torch.empty(miss_capacity, dtype=torch.int64, device=device),
        miss_page_storage[:, :max_pages],
        torch.empty(miss_capacity, dtype=torch.int32, device=device),
        torch.empty(miss_capacity, dtype=torch.int64, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
    )


def _resolve_compact_verification(
    selected_cluster_indices: torch.Tensor,
    plan_row_indices: torch.Tensor,
    request_slot_generations: torch.Tensor,
    table: tuple[torch.Tensor, ...],
    miss_page_stride: int | None = None,
) -> tuple[torch.Tensor, ...]:
    device = selected_cluster_indices.device
    outputs = _compact_verification_outputs(
        plan_row_indices.numel(),
        selected_cluster_indices.shape[1],
        selected_cluster_indices.shape[2],
        2,
        miss_page_stride,
    )
    resolve_compact_verification_pages(
        selected_cluster_indices=selected_cluster_indices,
        plan_row_indices=plan_row_indices,
        request_slot_ids=torch.tensor([0, 1], dtype=torch.int64, device=device),
        request_slot_generations=request_slot_generations,
        arena_cluster_ids=torch.tensor(
            [[10, 11, 20, 21]], dtype=torch.int64, device=device
        ),
        arena_cluster_page_starts=torch.tensor(
            [[0, 2, 0, 1]], dtype=torch.int64, device=device
        ),
        arena_cluster_page_counts=torch.tensor(
            [[2, 1, 1, 2]], dtype=torch.int32, device=device
        ),
        arena_page_ids=torch.tensor(
            [[100, 101, 102, 200, 201, 202]], dtype=torch.int64, device=device
        ),
        arena_page_token_counts=torch.tensor(
            [[2, 1, 2, 2, 2, 1]], dtype=torch.int32, device=device
        ),
        arena_cluster_offsets=torch.tensor([0, 2], dtype=torch.int64, device=device),
        arena_page_offsets=torch.tensor([0, 3], dtype=torch.int64, device=device),
        arena_generations=torch.tensor([5, 7], dtype=torch.int64, device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_last_access_epochs=table[5],
        access_epoch=19,
        output_resident_page_ids=outputs[0],
        output_staging_page_ids=outputs[1],
        output_page_token_counts=outputs[2],
        output_page_counts=outputs[3],
        output_selected_counts=outputs[4],
        output_hit_counts=outputs[5],
        output_miss_counts=outputs[6],
        output_miss_handles=outputs[7],
        output_miss_logical_page_ids=outputs[8],
        output_miss_page_counts=outputs[9],
        output_miss_page_offsets=outputs[10],
        output_miss_count=outputs[11],
        output_invalid_descriptor_count=outputs[12],
    )
    torch.cuda.synchronize()
    return outputs


def test_compact_verification_resolver_preserves_ranked_pages_and_emits_misses():
    device = torch.device("cuda")
    table = _make_table(max_pages=2)
    update_resident_handles(
        bucket_ids=torch.tensor([5, 3], dtype=torch.int32, device=device),
        cluster_handles=torch.tensor([21, 11], dtype=torch.int64, device=device),
        page_counts=torch.tensor([2, 1], dtype=torch.int32, device=device),
        page_slots=torch.tensor([[7, 8], [9, -1]], dtype=torch.int32, device=device),
        hit_gate_ready=torch.ones(2, dtype=torch.bool, device=device),
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
    )
    selected = torch.tensor(
        [[[1, -1]], [[0, 1]], [[0, 1]], [[1, 0]]],
        dtype=torch.int32,
        device=device,
    )
    outputs = _resolve_compact_verification(
        selected,
        torch.tensor([3, 0, 1], dtype=torch.int64, device=device),
        torch.tensor([5, 7], dtype=torch.int64, device=device),
        table,
    )

    assert outputs[0].cpu().tolist() == [
        [[7, 8, -1, -1]],
        [[9, -1, -1, -1]],
        [[-1, 7, 8, -1]],
    ]
    assert outputs[2].cpu().tolist() == [
        [[2, 1, 2, 0]],
        [[2, 0, 0, 0]],
        [[2, 2, 1, 0]],
    ]
    assert outputs[3].cpu().tolist() == [[3], [1], [3]]
    assert outputs[4].cpu().tolist() == [[2], [1], [2]]
    assert outputs[5].cpu().tolist() == [[1], [1], [1]]
    assert outputs[6].cpu().tolist() == [[1], [0], [1]]
    assert outputs[12].item() == 0

    miss_count = int(outputs[11].item())
    records = sorted(
        zip(
            outputs[10][:miss_count].cpu().tolist(),
            outputs[7][:miss_count].cpu().tolist(),
            outputs[9][:miss_count].cpu().tolist(),
            outputs[8][:miss_count].cpu().tolist(),
        )
    )
    assert records == [(2, 20, 1, [200, -1]), (8, 20, 1, [200, -1])]
    assert table[5].cpu().tolist()[3:6] == [19, 0, 19]


def test_compact_verification_resolver_honors_miss_page_row_stride():
    device = torch.device("cuda")
    selected = torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.int32, device=device)
    outputs = _resolve_compact_verification(
        selected,
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        torch.tensor([5, 7], dtype=torch.int64, device=device),
        _make_table(max_pages=2),
        miss_page_stride=4,
    )

    miss_count = int(outputs[11].item())
    records = sorted(
        zip(
            outputs[7][:miss_count].cpu().tolist(),
            outputs[9][:miss_count].cpu().tolist(),
            outputs[8][:miss_count].cpu().tolist(),
        )
    )
    assert records == [
        (10, 2, [100, 101]),
        (11, 1, [102, -1]),
        (20, 1, [200, -1]),
        (21, 2, [201, 202]),
    ]


def test_compact_verification_resolver_reports_stale_request_generation():
    device = torch.device("cuda")
    selected = torch.full((2, 1, 1), -1, dtype=torch.int32, device=device)
    selected[1, 0, 0] = 0
    outputs = _resolve_compact_verification(
        selected,
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        torch.tensor([5, 6], dtype=torch.int64, device=device),
        _make_table(max_pages=2),
    )

    assert outputs[3].cpu().tolist() == [[0], [0]]
    assert outputs[11].item() == 0
    assert outputs[12].item() == 1


def test_compact_verification_staging_scatter_uses_reserved_page_offsets():
    device = torch.device("cuda")
    output = torch.full((2, 1, 4), -1, dtype=torch.int64, device=device)
    scatter_compact_staging_page_ids(
        miss_output_page_offsets=torch.tensor([2, 4], device=device),
        staging_starts=torch.tensor([0, 3], device=device),
        page_counts=torch.tensor([2, 1], dtype=torch.int32, device=device),
        num_misses=2,
        max_pages=2,
        output_page_ids=output,
    )

    assert output.cpu().tolist() == [[[-1, -1, 0, 1]], [[3, -1, -1, -1]]]
