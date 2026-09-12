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
    emit_primary_exact_token_plan,
    emit_ranked_draft_plan,
    gather_resident_estimation,
    gather_resident_exact_pages,
    pack_indexed_verification_plan,
)


def _reference_primary_exact_token_plan(
    seq_lens: torch.Tensor,
    indexed_starts: torch.Tensor,
    indexed_ends: torch.Tensor,
    indexed_requests: torch.Tensor,
    *,
    num_kv_heads: int,
    max_num_tokens: int,
    block_size: int,
    num_recent_blocks: int,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.zeros(
        (seq_lens.numel(), num_kv_heads, output_width), dtype=torch.int64
    )
    mask = torch.zeros_like(indices, dtype=torch.bool)
    logical_tokens = torch.arange(max_num_tokens, dtype=torch.int64)

    for request_idx, raw_seq_len in enumerate(seq_lens.tolist()):
        seq_len = min(max(int(raw_seq_len), 1), max_num_tokens)
        valid_blocks = (seq_len + block_size - 1) // block_size
        recent_start = max(valid_blocks - num_recent_blocks, 0) * block_size
        recent_start = min(recent_start, seq_len)
        sink_end = min(block_size, seq_len)

        if indexed_requests[request_idx]:
            indexed_start = min(max(int(indexed_starts[request_idx]), 0), seq_len)
            indexed_end = min(
                max(int(indexed_ends[request_idx]), indexed_start), seq_len
            )
            exact = (
                (logical_tokens < max(indexed_start, sink_end))
                | (logical_tokens >= min(indexed_end, recent_start))
            ) & (logical_tokens < seq_len)
        else:
            exact = logical_tokens < seq_len

        selected = logical_tokens[exact][:output_width]
        selected_width = selected.numel()
        indices[request_idx, :, :selected_width] = selected
        mask[request_idx, :, :selected_width] = True
        invalid_width = min(output_width, max_num_tokens)
        indices[request_idx, :, selected_width:invalid_width] = max_num_tokens - 1

    return indices, mask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_emit_primary_exact_token_plan_matches_dense_reference():
    device = torch.device("cuda")
    seq_lens = torch.tensor([3, 64, 73, 120, 96], dtype=torch.int32)
    indexed_starts = torch.tensor([0, 16, 16, 32, -3], dtype=torch.int64)
    indexed_ends = torch.tensor([0, 32, 48, 96, 200], dtype=torch.int64)
    indexed_requests = torch.tensor([False, True, True, True, True])
    num_kv_heads = 3
    max_num_tokens = 128
    output_width = 48
    block_size = 16
    num_recent_blocks = 3

    padded_indices = torch.empty(
        (seq_lens.numel(), num_kv_heads, output_width + 2),
        dtype=torch.int64,
        device=device,
    )
    padded_mask = torch.empty_like(padded_indices, dtype=torch.bool)
    output_indices = padded_indices[..., :output_width]
    output_mask = padded_mask[..., :output_width]

    emit_primary_exact_token_plan(
        seq_lens=seq_lens.to(device),
        indexed_starts=indexed_starts.to(device),
        indexed_ends=indexed_ends.to(device),
        indexed_requests=indexed_requests.to(device),
        num_kv_heads=num_kv_heads,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        num_recent_blocks=num_recent_blocks,
        output_indices=output_indices,
        output_mask=output_mask,
    )

    expected_indices, expected_mask = _reference_primary_exact_token_plan(
        seq_lens,
        indexed_starts,
        indexed_ends,
        indexed_requests,
        num_kv_heads=num_kv_heads,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        num_recent_blocks=num_recent_blocks,
        output_width=output_width,
    )
    torch.testing.assert_close(output_indices.cpu(), expected_indices)
    torch.testing.assert_close(output_mask.cpu(), expected_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_emit_primary_exact_token_plan_clears_slots_past_logical_width():
    device = torch.device("cuda")
    output_indices = torch.full((1, 2, 12), -7, dtype=torch.int64, device=device)
    output_mask = torch.ones_like(output_indices, dtype=torch.bool)

    emit_primary_exact_token_plan(
        seq_lens=torch.tensor([3], dtype=torch.int32, device=device),
        indexed_starts=torch.tensor([0], dtype=torch.int64, device=device),
        indexed_ends=torch.tensor([0], dtype=torch.int64, device=device),
        indexed_requests=torch.tensor([False], device=device),
        num_kv_heads=2,
        max_num_tokens=8,
        block_size=4,
        num_recent_blocks=2,
        output_indices=output_indices,
        output_mask=output_mask,
    )

    expected_indices = torch.tensor(
        [[[0, 1, 2, 7, 7, 7, 7, 7, 0, 0, 0, 0]] * 2],
        dtype=torch.int64,
    )
    expected_mask = torch.tensor(
        [[[True, True, True] + [False] * 9] * 2],
        dtype=torch.bool,
    )
    torch.testing.assert_close(output_indices.cpu(), expected_indices)
    torch.testing.assert_close(output_mask.cpu(), expected_mask)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_pack_indexed_verification_plan_matches_query_rows(device_type: str):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device(device_type)
    valid_rows = torch.tensor([[True, True], [True, False]], device=device)
    request_slots = torch.tensor([3, 7], dtype=torch.int64, device=device)
    request_generations = torch.tensor([11, 13], dtype=torch.int64, device=device)
    exact = torch.arange(4 * 2 * 3, dtype=torch.int32, device=device).view(4, 2, 3)
    estimation = torch.tensor(
        [
            [[0, 1], [2, -1]],
            [[3, 4], [-1, 5]],
            [[6, 7], [8, 9]],
            [[10, 11], [12, 13]],
        ],
        dtype=torch.int32,
        device=device,
    )
    attention = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
    requests = torch.tensor([1, 0, 1, 2], dtype=torch.int64, device=device)
    tokens = torch.tensor([0, 1, 1, 0], dtype=torch.int64, device=device)
    num_pairs = requests.numel()

    plan_rows = torch.empty(num_pairs, dtype=torch.int64, device=device)
    plan_valid = torch.empty(num_pairs, dtype=torch.bool, device=device)
    packed_slots = torch.empty(num_pairs, dtype=torch.int64, device=device)
    packed_generations = torch.empty(num_pairs, dtype=torch.int64, device=device)
    packed_exact = torch.empty(num_pairs, 2, 3, dtype=torch.int32, device=device)
    packed_estimation = torch.empty(num_pairs, 2, 2, dtype=torch.int32, device=device)
    packed_mask = torch.empty_like(packed_estimation, dtype=torch.bool)
    packed_attention = torch.empty(num_pairs, device=device)

    pack_indexed_verification_plan(
        request_indices=requests,
        token_indices=tokens,
        valid_rows=valid_rows,
        request_slot_ids=request_slots,
        request_slot_generations=request_generations,
        exact_cluster_indices=exact,
        estimation_cluster_indices=estimation,
        attention_mass=attention,
        output_plan_row_indices=plan_rows,
        output_plan_valid_rows=plan_valid,
        output_request_slot_ids=packed_slots,
        output_request_slot_generations=packed_generations,
        output_exact_cluster_indices=packed_exact,
        output_estimation_cluster_indices=packed_estimation,
        output_estimation_cluster_mask=packed_mask,
        output_attention_mass=packed_attention,
    )

    assert plan_rows.cpu().tolist() == [1, 2, 3, 0]
    assert plan_valid.cpu().tolist() == [True, True, False, False]
    assert packed_slots.cpu().tolist() == [7, 3, -1, -1]
    assert packed_generations.cpu().tolist() == [13, 11, -1, -1]
    expected_exact = exact.index_select(0, torch.tensor([1, 2], device=device))
    torch.testing.assert_close(packed_exact[:2], expected_exact)
    assert (packed_exact[2:] == -1).all()
    expected_estimation = estimation.index_select(
        0, torch.tensor([1, 2], device=device)
    )
    torch.testing.assert_close(packed_estimation[:2], expected_estimation)
    assert (packed_estimation[2:] == -1).all()
    torch.testing.assert_close(packed_mask[:2], expected_estimation >= 0)
    assert not packed_mask[2:].any()
    torch.testing.assert_close(
        packed_attention.cpu(), torch.tensor([0.2, 0.3, 1.0, 1.0])
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
def test_emit_ranked_draft_plan_packs_exact_handles_and_summaries():
    device = torch.device("cuda")
    cluster_keys = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], device=device)
    cluster_values = cluster_keys * 10
    cluster_ids = torch.tensor([[10, 11, 12]], dtype=torch.int64, device=device)
    cluster_counts = torch.tensor([[4, 5, 6]], dtype=torch.int32, device=device)
    sparse_exact = torch.empty((1, 1, 2), dtype=torch.int32, device=device)
    expanded_exact = torch.empty((1, 1, 3), dtype=torch.int32, device=device)
    sparse_estimation = torch.empty((1, 1, 1), dtype=torch.int32, device=device)
    expanded_estimation = torch.empty_like(sparse_estimation)
    draft_keys = torch.empty((1, 1, 3, 2), device=device)
    draft_values = torch.empty_like(draft_keys)
    draft_counts = torch.empty((1, 1, 3), dtype=torch.int32, device=device)
    draft_handles = torch.empty((1, 1, 2), dtype=torch.int64, device=device)

    emit_ranked_draft_plan(
        ranked_indices=torch.tensor([[[2, 0, 1]]], dtype=torch.int64, device=device),
        candidate_counts=torch.tensor([[3]], dtype=torch.int32, device=device),
        cluster_keys=cluster_keys,
        cluster_values=cluster_values,
        cluster_ids=cluster_ids,
        cluster_token_counts=cluster_counts,
        cluster_offsets=torch.tensor([0], dtype=torch.int64, device=device),
        request_slot_ids=torch.tensor([0], dtype=torch.int64, device=device),
        active_mask=torch.tensor([True], device=device),
        retrieval_ratio=0.34,
        estimation_ratio=0.34,
        sparse_exact_width=2,
        sparse_exact_cluster_indices=sparse_exact,
        expanded_exact_cluster_indices=expanded_exact,
        sparse_estimation_cluster_indices=sparse_estimation,
        expanded_estimation_cluster_indices=expanded_estimation,
        draft_exact_cluster_handles=draft_handles,
        draft_estimation_keys=draft_keys,
        draft_estimation_values=draft_values,
        draft_estimation_token_counts=draft_counts,
    )

    assert sparse_exact.cpu().tolist() == [[[2, 0]]]
    assert expanded_exact.cpu().tolist() == [[[2, 0, 1]]]
    assert draft_handles.cpu().tolist() == [[[12, 10]]]
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
    assert not hasattr(index._draft_selection_scratch, "primary_topk_order")
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
