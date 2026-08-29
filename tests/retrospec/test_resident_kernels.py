# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.resident_kernels import (
    lookup_resident_handles,
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
    )


def _lookup(
    cluster_handles: torch.Tensor,
    logical_page_ids: torch.Tensor,
    active_mask: torch.Tensor,
    table: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    output_page_slots = torch.empty_like(logical_page_ids)
    output_hit_mask = torch.empty_like(cluster_handles, dtype=torch.bool)
    output_miss_mask = torch.empty_like(cluster_handles, dtype=torch.bool)
    output_gate_ready = torch.empty_like(cluster_handles, dtype=torch.bool)
    output_access_kinds = torch.empty_like(cluster_handles, dtype=torch.uint8)

    lookup_resident_handles(
        cluster_handles=cluster_handles,
        logical_page_ids=logical_page_ids,
        active_mask=active_mask,
        table_handles=table[0],
        table_versions=table[1],
        table_page_counts=table[2],
        table_page_slots=table[3],
        table_hit_gate_ready=table[4],
        output_page_slots=output_page_slots,
        output_hit_mask=output_hit_mask,
        output_miss_mask=output_miss_mask,
        output_hit_gate_ready=output_gate_ready,
        output_access_kinds=output_access_kinds,
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
