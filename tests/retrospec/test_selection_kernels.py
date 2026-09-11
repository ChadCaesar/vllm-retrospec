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
    add_indexed_values,
    emit_ranked_estimation_plan,
    emit_ranked_selection_plan,
    gather_resident_estimation,
    gather_resident_exact_pages,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_add_indexed_values_reads_persistent_rows_without_gather():
    device = torch.device("cuda")
    destination = torch.tensor([1.0, 2.0, 3.0], device=device)
    source = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], device=device)
    rows = torch.tensor([4, 0, 3], dtype=torch.int64, device=device)

    add_indexed_values(destination, source, rows)

    torch.testing.assert_close(
        destination, torch.tensor([5.0, 2.25, 5.0], device=device)
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
def test_emit_ranked_selection_plan_matches_request_head_reference():
    device = torch.device("cuda")
    num_slots, batch_size = 3, 4
    num_kv_heads, num_clusters, head_size = 2, 5, 3
    max_pages = 2
    sparse_width, estimation_width, expanded_width = 2, 2, 4
    retrieval_ratio, estimation_ratio = 0.4, 0.4

    cluster_capacity = num_slots * num_clusters
    cluster_offsets = torch.arange(
        0,
        cluster_capacity,
        num_clusters,
        dtype=torch.int64,
        device=device,
    )
    cluster_ids = torch.empty(
        num_kv_heads, cluster_capacity, dtype=torch.int64, device=device
    )
    cluster_keys = torch.empty(
        num_kv_heads,
        cluster_capacity,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    cluster_values = torch.empty_like(cluster_keys)
    cluster_token_counts = torch.empty(
        num_kv_heads, cluster_capacity, dtype=torch.int32, device=device
    )
    cluster_page_starts = torch.empty_like(cluster_ids)
    cluster_page_counts = torch.empty_like(cluster_ids, dtype=torch.int32)

    page_capacity = cluster_capacity * max_pages
    page_offsets = torch.arange(
        0,
        page_capacity,
        num_clusters * max_pages,
        dtype=torch.int64,
        device=device,
    )
    page_ids = torch.empty(
        num_kv_heads, page_capacity, dtype=torch.int64, device=device
    )
    page_token_counts = torch.empty_like(page_ids, dtype=torch.int32)

    for slot in range(num_slots):
        cluster_start = int(cluster_offsets[slot])
        page_start = int(page_offsets[slot])
        for head in range(num_kv_heads):
            for cluster in range(num_clusters):
                absolute_cluster = cluster_start + cluster
                cluster_ids[head, absolute_cluster] = slot * 100 + head * 10 + cluster
                cluster_keys[head, absolute_cluster] = torch.tensor(
                    [slot, head, cluster], dtype=torch.float32, device=device
                )
                cluster_values[head, absolute_cluster] = (
                    cluster_keys[head, absolute_cluster] + 1000
                )
                cluster_token_counts[head, absolute_cluster] = cluster + 1
                cluster_page_starts[head, absolute_cluster] = cluster * max_pages
                num_pages = 1 + (cluster % max_pages)
                cluster_page_counts[head, absolute_cluster] = num_pages
                for page in range(max_pages):
                    absolute_page = page_start + cluster * max_pages + page
                    page_ids[head, absolute_page] = (
                        slot * 1000 + head * 100 + cluster * 10 + page
                    )
                    page_token_counts[head, absolute_page] = page + 1

    ranked_indices = torch.tensor(
        [
            [[4, 1, 3, 0], [2, 4, 1, 0]],
            [[1, 3, 0, 4], [4, 0, 2, 3]],
            [[0, 1, 2, 3], [3, 2, 1, 0]],
            [[2, 0, 4, 1], [1, 3, 0, 2]],
        ],
        dtype=torch.int64,
        device=device,
    )
    ranked_values = torch.tensor(
        [
            [[0.40, 0.30, 0.20, 0.10], [0.70, 0.20, 0.08, 0.02]],
            [[0.60, 0.25, 0.10, 0.05], [0.45, 0.30, 0.15, 0.10]],
            [[0.40, 0.30, 0.20, 0.10], [0.40, 0.30, 0.20, 0.10]],
            [[0.40, 0.30, 0.20, 0.10], [0.40, 0.30, 0.20, 0.10]],
        ],
        dtype=torch.float32,
        device=device,
    )
    ranked_indices = torch.cat(
        (ranked_indices, torch.zeros_like(ranked_indices[:, :, :2])), dim=2
    )[:, :, :4]
    ranked_values = torch.cat(
        (ranked_values, torch.zeros_like(ranked_values[:, :, :2])), dim=2
    )[:, :, :4]
    assert not ranked_indices.is_contiguous()
    assert not ranked_values.is_contiguous()
    candidate_counts = torch.tensor(
        [[5, 0], [2, 4], [5, 5], [3, 3]],
        dtype=torch.int32,
        device=device,
    )
    request_slot_ids = torch.tensor([2, 0, -1, 1], device=device)
    active_mask = torch.tensor([True, True, True, False], device=device)

    sparse_cluster_indices = torch.full(
        (batch_size, num_kv_heads, sparse_width),
        777,
        dtype=torch.int32,
        device=device,
    )
    draft_cluster_ids = torch.full(
        (batch_size, num_kv_heads, sparse_width),
        777,
        dtype=torch.int64,
        device=device,
    )
    draft_page_ids = torch.full(
        (batch_size, num_kv_heads, sparse_width, max_pages),
        777,
        dtype=torch.int64,
        device=device,
    )
    draft_page_counts = torch.full_like(draft_page_ids, 777, dtype=torch.int32)
    expanded_cluster_indices = torch.full(
        (batch_size, num_kv_heads, expanded_width),
        777,
        dtype=torch.int32,
        device=device,
    )
    sparse_estimation_indices = torch.full(
        (batch_size, num_kv_heads, estimation_width),
        777,
        dtype=torch.int32,
        device=device,
    )
    expanded_estimation_indices = torch.full_like(sparse_estimation_indices, 777)
    draft_width = estimation_width + sparse_width
    draft_keys = torch.full(
        (batch_size, num_kv_heads, draft_width, head_size),
        777,
        dtype=torch.float32,
        device=device,
    )
    draft_values = torch.full_like(draft_keys, 777)
    draft_counts = torch.full(
        (batch_size, num_kv_heads, draft_width),
        777,
        dtype=torch.int32,
        device=device,
    )
    sparse_attn = torch.full((batch_size,), 777.0, device=device)
    expanded_attn = torch.full((batch_size,), 777.0, device=device)

    emit_ranked_selection_plan(
        ranked_values=ranked_values,
        ranked_indices=ranked_indices,
        candidate_counts=candidate_counts,
        cluster_keys=cluster_keys,
        cluster_values=cluster_values,
        cluster_token_counts=cluster_token_counts,
        cluster_ids=cluster_ids,
        cluster_page_starts=cluster_page_starts,
        cluster_page_counts=cluster_page_counts,
        page_ids=page_ids,
        page_token_counts=page_token_counts,
        cluster_offsets=cluster_offsets,
        page_offsets=page_offsets,
        request_slot_ids=request_slot_ids,
        active_mask=active_mask,
        retrieval_ratio=retrieval_ratio,
        estimation_ratio=estimation_ratio,
        sparse_exact_cluster_indices=sparse_cluster_indices,
        sparse_estimation_cluster_indices=sparse_estimation_indices,
        expanded_exact_cluster_indices=expanded_cluster_indices,
        expanded_estimation_cluster_indices=expanded_estimation_indices,
        draft_exact_cluster_ids=draft_cluster_ids,
        draft_exact_page_ids=draft_page_ids,
        draft_exact_page_token_counts=draft_page_counts,
        draft_estimation_keys=draft_keys,
        draft_estimation_values=draft_values,
        draft_estimation_token_counts=draft_counts,
        sparse_attn=sparse_attn,
        expanded_attn=expanded_attn,
    )

    expected_sparse_indices = torch.full_like(sparse_cluster_indices, -1)
    expected_draft_ids = torch.full_like(draft_cluster_ids, -1)
    expected_draft_page_ids = torch.full_like(draft_page_ids, -1)
    expected_draft_page_counts = torch.zeros_like(draft_page_counts)
    expected_expanded_indices = torch.full_like(expanded_cluster_indices, -1)
    expected_sparse_estimation_indices = torch.full_like(sparse_estimation_indices, -1)
    expected_expanded_estimation_indices = torch.full_like(
        expanded_estimation_indices, -1
    )
    expected_draft_keys = torch.zeros_like(draft_keys)
    expected_draft_values = torch.zeros_like(draft_values)
    expected_draft_counts = torch.zeros_like(draft_counts)
    expected_sparse_attn = torch.ones_like(sparse_attn)
    expected_expanded_attn = torch.ones_like(expanded_attn)

    for batch in range(batch_size):
        slot = int(request_slot_ids[batch])
        if not bool(active_mask[batch]) or slot < 0:
            continue
        sparse_mass = 0.0
        expanded_mass = 0.0
        for head in range(num_kv_heads):
            count = int(candidate_counts[batch, head])
            retrieval_count = min(
                int(torch.ceil(torch.tensor(count * retrieval_ratio))), count
            )
            estimation_count = min(
                int(torch.ceil(torch.tensor(count * estimation_ratio))),
                count - retrieval_count,
            )
            total_count = retrieval_count + estimation_count
            expanded_count = min(2 * retrieval_count, total_count)
            sparse_mass += (
                float(ranked_values[batch, head, :retrieval_count].sum())
                if count
                else 1.0
            )
            expanded_mass += (
                float(ranked_values[batch, head, :expanded_count].sum())
                if count
                else 1.0
            )

            for output_idx in range(retrieval_count):
                cluster = int(ranked_indices[batch, head, output_idx])
                absolute_cluster = int(cluster_offsets[slot]) + cluster
                expected_sparse_indices[batch, head, output_idx] = cluster
                expected_draft_ids[batch, head, output_idx] = cluster_ids[
                    head, absolute_cluster
                ]
                fallback_idx = estimation_width + output_idx
                expected_draft_keys[batch, head, fallback_idx] = cluster_keys[
                    head, absolute_cluster
                ]
                expected_draft_values[batch, head, fallback_idx] = cluster_values[
                    head, absolute_cluster
                ]
                expected_draft_counts[batch, head, fallback_idx] = cluster_token_counts[
                    head, absolute_cluster
                ]
                num_pages = int(cluster_page_counts[head, absolute_cluster])
                page_start = int(page_offsets[slot]) + int(
                    cluster_page_starts[head, absolute_cluster]
                )
                expected_draft_page_ids[batch, head, output_idx, :num_pages] = page_ids[
                    head, page_start : page_start + num_pages
                ]
                expected_draft_page_counts[batch, head, output_idx, :num_pages] = (
                    page_token_counts[head, page_start : page_start + num_pages]
                )

            for output_idx in range(estimation_count):
                rank = retrieval_count + output_idx
                cluster = int(ranked_indices[batch, head, rank])
                absolute_cluster = int(cluster_offsets[slot]) + cluster
                expected_sparse_estimation_indices[batch, head, output_idx] = cluster
                expected_draft_keys[batch, head, output_idx] = cluster_keys[
                    head, absolute_cluster
                ]
                expected_draft_values[batch, head, output_idx] = cluster_values[
                    head, absolute_cluster
                ]
                expected_draft_counts[batch, head, output_idx] = cluster_token_counts[
                    head, absolute_cluster
                ]

            for output_idx in range(expanded_count):
                cluster = int(ranked_indices[batch, head, output_idx])
                expected_expanded_indices[batch, head, output_idx] = cluster

            for output_idx in range(total_count - expanded_count):
                rank = expanded_count + output_idx
                cluster = int(ranked_indices[batch, head, rank])
                expected_expanded_estimation_indices[batch, head, output_idx] = cluster

        expected_sparse_attn[batch] = sparse_mass / num_kv_heads
        expected_expanded_attn[batch] = expanded_mass / num_kv_heads

    torch.cuda.synchronize()
    torch.testing.assert_close(sparse_cluster_indices, expected_sparse_indices)
    torch.testing.assert_close(draft_cluster_ids, expected_draft_ids)
    torch.testing.assert_close(draft_page_ids, expected_draft_page_ids)
    torch.testing.assert_close(draft_page_counts, expected_draft_page_counts)
    torch.testing.assert_close(expanded_cluster_indices, expected_expanded_indices)
    torch.testing.assert_close(
        sparse_estimation_indices, expected_sparse_estimation_indices
    )
    torch.testing.assert_close(
        expanded_estimation_indices, expected_expanded_estimation_indices
    )
    torch.testing.assert_close(draft_keys, expected_draft_keys)
    torch.testing.assert_close(draft_values, expected_draft_values)
    torch.testing.assert_close(draft_counts, expected_draft_counts)
    torch.testing.assert_close(sparse_attn, expected_sparse_attn)
    torch.testing.assert_close(expanded_attn, expected_expanded_attn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_emit_ranked_estimation_plan_omits_exact_page_intermediates():
    device = torch.device("cuda")
    cluster_keys = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], device=device)
    cluster_values = cluster_keys * 10
    cluster_counts = torch.tensor([[4, 5, 6]], dtype=torch.int32, device=device)
    sparse_estimation = torch.empty((1, 1, 1), dtype=torch.int32, device=device)
    expanded_estimation = torch.empty_like(sparse_estimation)
    draft_keys = torch.empty((1, 1, 3, 2), device=device)
    draft_values = torch.empty_like(draft_keys)
    draft_counts = torch.empty((1, 1, 3), dtype=torch.int32, device=device)

    emit_ranked_estimation_plan(
        ranked_indices=torch.tensor([[[2, 0, 1]]], dtype=torch.int64, device=device),
        candidate_counts=torch.tensor([[3]], dtype=torch.int32, device=device),
        cluster_keys=cluster_keys,
        cluster_values=cluster_values,
        cluster_token_counts=cluster_counts,
        cluster_offsets=torch.tensor([0], dtype=torch.int64, device=device),
        request_slot_ids=torch.tensor([0], dtype=torch.int64, device=device),
        active_mask=torch.tensor([True], device=device),
        retrieval_ratio=0.34,
        estimation_ratio=0.34,
        sparse_exact_width=2,
        sparse_estimation_cluster_indices=sparse_estimation,
        expanded_estimation_cluster_indices=expanded_estimation,
        draft_estimation_keys=draft_keys,
        draft_estimation_values=draft_values,
        draft_estimation_token_counts=draft_counts,
    )

    assert sparse_estimation.item() == 1
    assert expanded_estimation.item() == -1
    torch.testing.assert_close(
        draft_keys.cpu(),
        torch.tensor([[[[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]]]]),
    )
    torch.testing.assert_close(draft_values.cpu(), draft_keys.cpu() * 10)
    assert draft_counts.cpu().tolist() == [[[5, 6, 4]]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selection_plan_table_uses_one_shared_draft_scratch():
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
        _, first, table = index._get_selection_plan_step(
            "layer", 0, view, 1, 1, 8, torch.float16, device
        )
        _, second, second_table = index._get_selection_plan_step(
            "layer", 1, view, 1, 1, 8, torch.float16, device
        )
        _, same_first, same_table = index._get_selection_plan_step(
            "layer", 0, view, 1, 1, 8, torch.float16, device
        )
    finally:
        index.end_proposal()

    assert second_table is table
    assert same_table is table
    assert not hasattr(table, "draft_estimation_keys")
    assert not hasattr(table, "expanded_estimation_keys")
    assert table.sparse_estimation_cluster_indices.shape == (2, 1, 1, 1)
    assert table.expanded_estimation_cluster_indices.shape == (2, 1, 1, 1)
    assert first.draft_estimation_keys.data_ptr() == (
        second.draft_estimation_keys.data_ptr()
    )
    assert same_first.draft_estimation_keys.data_ptr() == (
        first.draft_estimation_keys.data_ptr()
    )

    index.begin_proposal(["request"])
    try:
        _, reused, reused_table = index._get_selection_plan_step(
            "layer", 0, view, 1, 1, 8, torch.float16, device
        )
    finally:
        index.end_proposal()

    assert reused_table is table
    assert reused.draft_estimation_keys.data_ptr() == (
        first.draft_estimation_keys.data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_shared_draft_scratch_grows_and_returns_layer_sized_views():
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
    small_view = RetroSpecResidentBatchView(
        arena=None,
        request_slot_ids=torch.tensor([-1], dtype=torch.int64, device=device),
        max_num_clusters=4,
        max_pages_per_cluster=1,
        max_num_pages=0,
    )
    large_view = RetroSpecResidentBatchView(
        arena=None,
        request_slot_ids=torch.tensor([-1], dtype=torch.int64, device=device),
        max_num_clusters=8,
        max_pages_per_cluster=3,
        max_num_pages=0,
    )

    index.begin_proposal(["request"])
    try:
        _, small_before_growth, _ = index._get_selection_plan_step(
            "small", 0, small_view, 1, 1, 8, torch.float16, device
        )
        _, large, _ = index._get_selection_plan_step(
            "large", 0, large_view, 1, 1, 8, torch.float16, device
        )
        _, small_after_growth, _ = index._get_selection_plan_step(
            "small", 0, small_view, 1, 1, 8, torch.float16, device
        )
    finally:
        index.end_proposal()

    assert small_before_growth.draft_compact_page_ids.shape == (1, 1, 2)
    assert large.draft_compact_page_ids.shape == (1, 1, 12)
    assert small_after_growth.draft_compact_page_ids.shape == (1, 1, 2)
    assert small_after_growth.draft_estimation_keys.shape == (1, 1, 3, 8)
    assert large.draft_compact_page_ids.data_ptr() == (
        small_after_growth.draft_compact_page_ids.data_ptr()
    )
