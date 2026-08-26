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

    cluster_keys = torch.arange(
        num_slots * num_kv_heads * num_clusters * head_size,
        dtype=torch.float32,
        device=device,
    ).view(num_slots, num_kv_heads, num_clusters, head_size)
    cluster_values = cluster_keys + 1000
    cluster_token_counts = torch.arange(
        1,
        num_slots * num_kv_heads * num_clusters + 1,
        dtype=torch.int32,
        device=device,
    ).view(num_slots, num_kv_heads, num_clusters)
    cluster_token_counts[0, 1, 2] = 0

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
                    int(cluster_token_counts[request_slot, head_idx, cluster_idx]) > 0
                )
                if not valid:
                    continue
                expected_keys[batch_idx, head_idx, selected_idx].copy_(
                    cluster_keys[request_slot, head_idx, cluster_idx]
                )
                expected_values[batch_idx, head_idx, selected_idx].copy_(
                    cluster_values[request_slot, head_idx, cluster_idx]
                )
                expected_counts[batch_idx, head_idx, selected_idx] = (
                    cluster_token_counts[request_slot, head_idx, cluster_idx]
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

    cluster_ids = torch.arange(
        num_slots * num_kv_heads * num_clusters,
        dtype=torch.int64,
        device=device,
    ).view(num_slots, num_kv_heads, num_clusters)
    cluster_ids[1, 0, 2] = -1
    cluster_page_offsets = (
        torch.arange(
            0,
            num_pages + 1,
            max_pages,
            dtype=torch.int64,
            device=device,
        )
        .view(1, 1, num_clusters + 1)
        .expand(num_slots, num_kv_heads, -1)
        .contiguous()
    )
    page_ids = torch.arange(
        num_slots * num_kv_heads * num_pages,
        dtype=torch.int64,
        device=device,
    ).view(num_slots, num_kv_heads, num_pages)
    page_token_counts = (
        torch.tensor(
            [2, 1] * num_clusters,
            dtype=torch.int32,
            device=device,
        )
        .view(1, 1, num_pages)
        .expand(num_slots, num_kv_heads, -1)
        .contiguous()
    )

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
        cluster_page_offsets,
        page_ids,
        page_token_counts,
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
                cluster_id = int(cluster_ids[request_slot, head_idx, cluster_idx])
                if not valid or cluster_id < 0:
                    continue
                expected_cluster_ids[batch_idx, head_idx, selected_idx] = cluster_id
                page_start = int(
                    cluster_page_offsets[request_slot, head_idx, cluster_idx]
                )
                page_end = int(
                    cluster_page_offsets[request_slot, head_idx, cluster_idx + 1]
                )
                num_cluster_pages = page_end - page_start
                expected_page_ids[
                    batch_idx, head_idx, selected_idx, :num_cluster_pages
                ].copy_(page_ids[request_slot, head_idx, page_start:page_end])
                expected_page_counts[
                    batch_idx, head_idx, selected_idx, :num_cluster_pages
                ].copy_(page_token_counts[request_slot, head_idx, page_start:page_end])

    torch.cuda.synchronize()
    torch.testing.assert_close(output_cluster_ids, expected_cluster_ids)
    torch.testing.assert_close(output_page_ids, expected_page_ids)
    torch.testing.assert_close(output_page_counts, expected_page_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selection_output_workspace_uses_step_slots_and_reuses_proposals():
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
        with pytest.raises(RuntimeError, match="selection capacity"):
            index._get_selection_output_workspace(
                "layer", view, 1, 1, 8, torch.float16, device
            )
    finally:
        index.end_proposal()

    assert first is not second
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
