# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecResidentBatchView,
)
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)
from vllm.v1.spec_decode.retrospec.selection_kernels import (
    gather_resident_estimation,
    gather_resident_exact_pages,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gather_resident_estimation_matches_request_slot_reference():
    device = torch.device("cuda")
    num_slots, batch_size = 3, 3
    num_kv_heads, num_clusters, head_size = 2, 4, 7
    max_selected = 3

    slot_cluster_keys = torch.arange(
        num_slots * num_kv_heads * num_clusters * head_size,
        dtype=torch.float32,
        device=device,
    ).view(num_slots, num_kv_heads, num_clusters, head_size)
    slot_cluster_values = slot_cluster_keys + 1000
    slot_cluster_token_counts = torch.arange(
        1,
        num_slots * num_kv_heads * num_clusters + 1,
        dtype=torch.int32,
        device=device,
    ).view(num_slots, num_kv_heads, num_clusters)
    slot_cluster_token_counts[0, 1, 2] = 0
    cluster_offsets = torch.tensor([2, 9, 16], dtype=torch.int64, device=device)
    arena_capacity = 22
    cluster_keys = torch.zeros(
        num_kv_heads, arena_capacity, head_size, dtype=torch.float32, device=device
    )
    cluster_values = torch.zeros_like(cluster_keys)
    cluster_token_counts = torch.zeros(
        num_kv_heads, arena_capacity, dtype=torch.int32, device=device
    )
    for slot, cluster_offset in enumerate(cluster_offsets.tolist()):
        cluster_slice = slice(cluster_offset, cluster_offset + num_clusters)
        cluster_keys[:, cluster_slice].copy_(slot_cluster_keys[slot])
        cluster_values[:, cluster_slice].copy_(slot_cluster_values[slot])
        cluster_token_counts[:, cluster_slice].copy_(slot_cluster_token_counts[slot])

    request_slot_ids = torch.tensor([2, 0, -1], dtype=torch.int64, device=device)
    selected_indices = torch.tensor(
        [
            [[3, 1, 0], [0, 2, 1]],
            [[1, 3, 0], [2, 0, 3]],
            [[0, 1, 2], [3, 2, 1]],
        ],
        dtype=torch.int64,
        device=device,
    )
    selected_mask = torch.tensor(
        [
            [[True, True, False], [True, True, True]],
            [[True, False, True], [True, True, False]],
            [[True, True, True], [True, True, True]],
        ],
        dtype=torch.bool,
        device=device,
    )

    padded_keys = torch.empty(
        batch_size,
        num_kv_heads,
        max_selected + 1,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    padded_values = torch.empty_like(padded_keys)
    padded_counts = torch.empty(
        batch_size,
        num_kv_heads,
        max_selected + 1,
        dtype=torch.int32,
        device=device,
    )
    output_keys = padded_keys[:, :, :max_selected]
    output_values = padded_values[:, :, :max_selected]
    output_counts = padded_counts[:, :, :max_selected]

    gather_resident_estimation(
        cluster_keys,
        cluster_values,
        cluster_token_counts,
        cluster_offsets,
        request_slot_ids,
        selected_indices,
        selected_mask,
        output_keys,
        output_values,
        output_counts,
    )

    expected_keys = torch.zeros_like(output_keys)
    expected_values = torch.zeros_like(output_values)
    expected_counts = torch.zeros_like(output_counts)
    for batch_idx, request_slot in enumerate(request_slot_ids.tolist()):
        if request_slot < 0:
            continue
        for head_idx in range(num_kv_heads):
            for selected_idx in range(max_selected):
                cluster_idx = int(selected_indices[batch_idx, head_idx, selected_idx])
                valid = bool(selected_mask[batch_idx, head_idx, selected_idx])
                valid &= (
                    int(slot_cluster_token_counts[request_slot, head_idx, cluster_idx])
                    > 0
                )
                if not valid:
                    continue
                expected_keys[batch_idx, head_idx, selected_idx].copy_(
                    slot_cluster_keys[request_slot, head_idx, cluster_idx]
                )
                expected_values[batch_idx, head_idx, selected_idx].copy_(
                    slot_cluster_values[request_slot, head_idx, cluster_idx]
                )
                expected_counts[batch_idx, head_idx, selected_idx] = (
                    slot_cluster_token_counts[request_slot, head_idx, cluster_idx]
                )

    torch.cuda.synchronize()
    torch.testing.assert_close(output_keys, expected_keys)
    torch.testing.assert_close(output_values, expected_values)
    torch.testing.assert_close(output_counts, expected_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gather_resident_exact_pages_matches_request_slot_reference():
    device = torch.device("cuda")
    num_slots, batch_size = 2, 3
    num_kv_heads, num_clusters = 2, 3
    max_selected, max_pages = 2, 2
    num_pages = num_clusters * max_pages

    slot_cluster_ids = torch.arange(
        num_slots * num_kv_heads * num_clusters,
        dtype=torch.int64,
        device=device,
    ).view(num_slots, num_kv_heads, num_clusters)
    slot_cluster_ids[1, 0, 2] = -1
    cluster_offsets = torch.tensor([2, 9], dtype=torch.int64, device=device)
    cluster_capacity = 14
    cluster_ids = torch.full(
        (num_kv_heads, cluster_capacity), -1, dtype=torch.int64, device=device
    )
    cluster_page_starts = (
        torch.arange(0, num_pages, max_pages, dtype=torch.int64, device=device)
        .view(1, 1, num_clusters)
        .expand(num_slots, num_kv_heads, -1)
        .contiguous()
    )
    slot_cluster_page_counts = torch.full_like(cluster_page_starts, max_pages)
    packed_cluster_page_starts = torch.zeros_like(cluster_ids)
    cluster_page_counts = torch.zeros_like(cluster_ids, dtype=torch.int32)
    slot_page_ids = torch.arange(
        num_slots * num_kv_heads * num_pages,
        dtype=torch.int64,
        device=device,
    ).view(num_slots, num_kv_heads, num_pages)
    slot_page_token_counts = (
        torch.tensor(
            [2, 1] * num_clusters,
            dtype=torch.int32,
            device=device,
        )
        .view(1, 1, num_pages)
        .expand(num_slots, num_kv_heads, -1)
        .contiguous()
    )
    page_offsets = torch.tensor([3, 14], dtype=torch.int64, device=device)
    page_capacity = 22
    page_ids = torch.full(
        (num_kv_heads, page_capacity), -1, dtype=torch.int64, device=device
    )
    page_token_counts = torch.zeros(
        num_kv_heads, page_capacity, dtype=torch.int32, device=device
    )
    for slot, (cluster_offset, page_offset) in enumerate(
        zip(cluster_offsets.tolist(), page_offsets.tolist())
    ):
        cluster_slice = slice(cluster_offset, cluster_offset + num_clusters)
        page_slice = slice(page_offset, page_offset + num_pages)
        cluster_ids[:, cluster_slice].copy_(slot_cluster_ids[slot])
        packed_cluster_page_starts[:, cluster_slice].copy_(cluster_page_starts[slot])
        cluster_page_counts[:, cluster_slice].copy_(slot_cluster_page_counts[slot])
        page_ids[:, page_slice].copy_(slot_page_ids[slot])
        page_token_counts[:, page_slice].copy_(slot_page_token_counts[slot])

    request_slot_ids = torch.tensor([1, 0, -1], dtype=torch.int64, device=device)
    selected_indices = torch.tensor(
        [[[2, 0], [1, 2]], [[1, 0], [2, 1]], [[0, 1], [1, 2]]],
        dtype=torch.int64,
        device=device,
    )
    selected_mask = torch.tensor(
        [
            [[True, True], [True, False]],
            [[True, False], [True, True]],
            [[True, True], [True, True]],
        ],
        dtype=torch.bool,
        device=device,
    )

    padded_cluster_ids = torch.empty(
        batch_size,
        num_kv_heads,
        max_selected + 1,
        dtype=torch.int64,
        device=device,
    )
    padded_page_ids = torch.empty(
        batch_size,
        num_kv_heads,
        max_selected + 1,
        max_pages,
        dtype=torch.int64,
        device=device,
    )
    padded_page_counts = torch.empty_like(padded_page_ids, dtype=torch.int32)
    output_cluster_ids = padded_cluster_ids[:, :, :max_selected]
    output_page_ids = padded_page_ids[:, :, :max_selected]
    output_page_counts = padded_page_counts[:, :, :max_selected]

    gather_resident_exact_pages(
        cluster_ids,
        packed_cluster_page_starts,
        cluster_page_counts,
        page_ids,
        page_token_counts,
        cluster_offsets,
        page_offsets,
        request_slot_ids,
        selected_indices,
        selected_mask,
        output_cluster_ids,
        output_page_ids,
        output_page_counts,
    )

    expected_cluster_ids = torch.full_like(output_cluster_ids, -1)
    expected_page_ids = torch.full_like(output_page_ids, -1)
    expected_page_counts = torch.zeros_like(output_page_counts)
    for batch_idx, request_slot in enumerate(request_slot_ids.tolist()):
        if request_slot < 0:
            continue
        for head_idx in range(num_kv_heads):
            for selected_idx in range(max_selected):
                cluster_idx = int(selected_indices[batch_idx, head_idx, selected_idx])
                valid = bool(selected_mask[batch_idx, head_idx, selected_idx])
                cluster_id = int(slot_cluster_ids[request_slot, head_idx, cluster_idx])
                if not valid or cluster_id < 0:
                    continue
                expected_cluster_ids[batch_idx, head_idx, selected_idx] = cluster_id
                page_start = cluster_idx * max_pages
                page_end = page_start + max_pages
                num_cluster_pages = max_pages
                expected_page_ids[
                    batch_idx, head_idx, selected_idx, :num_cluster_pages
                ].copy_(slot_page_ids[request_slot, head_idx, page_start:page_end])
                expected_page_counts[
                    batch_idx, head_idx, selected_idx, :num_cluster_pages
                ].copy_(
                    slot_page_token_counts[request_slot, head_idx, page_start:page_end]
                )

    torch.cuda.synchronize()
    torch.testing.assert_close(output_cluster_ids, expected_cluster_ids)
    torch.testing.assert_close(output_page_ids, expected_page_ids)
    torch.testing.assert_close(output_page_counts, expected_page_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selection_output_workspace_recycles_step_slots_and_reuses_proposals():
    device = torch.device("cuda", torch.cuda.current_device())
    index = RetroSpecSegmentedTokenIndex(
        block_size=2,
        num_speculative_tokens=2,
        retrieval_ratio=0.5,
        estimation_ratio=0.25,
        prefill_segment_size_tokens=4,
        generation_update_interval=2,
        blocks_per_cluster=1,
        num_kmeans_iterations=2,
        max_model_len=64,
    )
    view = RetroSpecResidentBatchView(
        arena=None,
        request_slot_ids=torch.tensor([-1], dtype=torch.int64, device=device),
        max_num_clusters=4,
        max_pages_per_cluster=2,
        max_num_pages=0,
    )

    index.begin_proposal(["request"])
    try:
        first = index._get_selection_output_workspace(
            "layer", view, 1, 1, 8, torch.float16, device
        )
        second = index._get_selection_output_workspace(
            "layer", view, 1, 1, 8, torch.float16, device
        )
        recycled = index._get_selection_output_workspace(
            "layer", view, 1, 1, 8, torch.float16, device
        )
    finally:
        index.end_proposal()

    assert first is not second
    assert recycled is first
    assert first.draft_estimation_keys.data_ptr() != (
        second.draft_estimation_keys.data_ptr()
    )

    index.begin_proposal(["request"])
    try:
        reused = index._get_selection_output_workspace(
            "layer", view, 1, 1, 8, torch.float16, device
        )
    finally:
        index.end_proposal()

    assert reused is first
    assert reused.draft_estimation_keys.data_ptr() == (
        first.draft_estimation_keys.data_ptr()
    )
