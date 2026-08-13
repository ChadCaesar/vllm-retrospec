# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.execution import (
    RetroSpecExactExecutionBuffer,
)


def test_execution_buffer_rejects_invalid_page_size():
    with pytest.raises(ValueError, match="positive"):
        RetroSpecExactExecutionBuffer(page_size=0)


def _make_primary_cache(
    num_blocks: int,
    page_size: int,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_ids = torch.arange(num_blocks, device=device)[:, None, None, None]
    offsets = torch.arange(page_size, device=device)[None, :, None, None]
    head_ids = torch.arange(num_kv_heads, device=device)[None, None, :, None]
    dimensions = torch.arange(head_size, device=device)[None, None, None, :]
    keys = (block_ids * 1000 + offsets * 100 + head_ids * 10 + dimensions).to(
        torch.bfloat16
    )
    return keys, keys + 5


def _make_cluster_pages(
    num_pages: int,
    page_size: int,
    head_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_ids = torch.arange(num_pages, device=device)[:, None, None]
    offsets = torch.arange(page_size, device=device)[None, :, None]
    dimensions = torch.arange(head_size, device=device)[None, None, :]
    keys = (10000 + page_ids * 1000 + offsets * 100 + dimensions).to(torch.bfloat16)
    return keys, keys + 7


def _reference_packed_tokens(
    primary_cache: torch.Tensor,
    block_table: torch.Tensor,
    primary_token_indices: torch.Tensor,
    primary_token_mask: torch.Tensor,
    page_ids: torch.Tensor,
    page_token_counts: torch.Tensor,
    cluster_pages: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    batch_size, num_kv_heads = primary_token_indices.shape[:2]
    sequences = []

    for batch_idx in range(batch_size):
        for head_idx in range(num_kv_heads):
            tokens = []
            for logical_idx, selected in zip(
                primary_token_indices[batch_idx, head_idx].tolist(),
                primary_token_mask[batch_idx, head_idx].tolist(),
            ):
                if selected:
                    logical_block = logical_idx // page_size
                    block_offset = logical_idx % page_size
                    physical_block = int(block_table[batch_idx, logical_block])
                    tokens.append(primary_cache[physical_block, block_offset, head_idx])

            for page_id, token_count in zip(
                page_ids[batch_idx, head_idx].reshape(-1).tolist(),
                page_token_counts[batch_idx, head_idx].reshape(-1).tolist(),
            ):
                if page_id >= 0:
                    tokens.extend(cluster_pages[page_id, :token_count])

            sequences.extend(tokens)

    return torch.stack(sequences).view(-1, 1, primary_cache.shape[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_execution_buffer_packs_primary_and_cluster_tokens_on_cuda():
    device = torch.device("cuda")
    page_size = 2
    num_kv_heads = 2
    head_size = 64
    keys, values = _make_primary_cache(6, page_size, num_kv_heads, head_size, device)
    cluster_keys, cluster_values = _make_cluster_pages(6, page_size, head_size, device)
    block_table = torch.tensor([[2, 0, 1], [3, 4, 5]], dtype=torch.int32, device=device)
    primary_indices = torch.tensor(
        [
            [[0, 3, 4], [1, 2, 5]],
            [[0, 1, 3], [2, 4, 5]],
        ],
        dtype=torch.int64,
        device=device,
    )
    primary_mask = torch.tensor(
        [
            [[True, False, True], [True, True, False]],
            [[False, True, True], [True, False, True]],
        ],
        device=device,
    )
    page_ids = torch.tensor(
        [
            [[[0, 1], [-1, -1]], [[2, -1], [-1, -1]]],
            [[[3, 4], [-1, -1]], [[5, -1], [-1, -1]]],
        ],
        dtype=torch.int64,
        device=device,
    )
    page_counts = torch.tensor(
        [
            [[[2, 1], [0, 0]], [[1, 0], [0, 0]]],
            [[[2, 2], [0, 0]], [[1, 0], [0, 0]]],
        ],
        dtype=torch.int32,
        device=device,
    )

    buffer = RetroSpecExactExecutionBuffer(page_size)
    execution = buffer.pack(
        keys,
        values,
        block_table,
        primary_indices,
        primary_mask,
        page_ids,
        page_counts,
        cluster_keys,
        cluster_values,
    )
    torch.cuda.synchronize()

    expected_keys = _reference_packed_tokens(
        keys,
        block_table,
        primary_indices,
        primary_mask,
        page_ids,
        page_counts,
        cluster_keys,
        page_size,
    )
    expected_values = _reference_packed_tokens(
        values,
        block_table,
        primary_indices,
        primary_mask,
        page_ids,
        page_counts,
        cluster_values,
        page_size,
    )
    actual_tokens = int(execution.cu_seqlens_k[-1])

    assert execution.exact_seq_lens.tolist() == [5, 3, 6, 3]
    assert execution.cu_seqlens_q.tolist() == [0, 1, 2, 3, 4]
    assert execution.cu_seqlens_k.tolist() == [0, 5, 8, 14, 17]
    assert execution.max_exact_seq_len == 11
    torch.testing.assert_close(execution.keys[:actual_tokens], expected_keys)
    torch.testing.assert_close(execution.values[:actual_tokens], expected_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_execution_buffer_handles_empty_batch_and_reuses_storage_on_cuda():
    device = torch.device("cuda")
    page_size = 2
    head_size = 64
    keys, values = _make_primary_cache(2, page_size, 1, head_size, device)
    buffer = RetroSpecExactExecutionBuffer(page_size)

    first = buffer.pack(
        keys,
        values,
        torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        torch.tensor([[[0, 1]]], dtype=torch.int64, device=device),
        torch.tensor([[[True, True]]], device=device),
        torch.empty(1, 1, 0, 0, dtype=torch.int64, device=device),
        torch.empty(1, 1, 0, 0, dtype=torch.int32, device=device),
        None,
        None,
    )
    first_pointer = first.keys.data_ptr()

    second = buffer.pack(
        keys,
        values,
        torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        torch.tensor([[[1]]], dtype=torch.int64, device=device),
        torch.tensor([[[True]]], device=device),
        torch.empty(1, 1, 0, 0, dtype=torch.int64, device=device),
        torch.empty(1, 1, 0, 0, dtype=torch.int32, device=device),
        None,
        None,
    )
    empty = buffer.pack(
        keys,
        values,
        torch.empty(0, 0, dtype=torch.int32, device=device),
        torch.empty(0, 1, 2, dtype=torch.int64, device=device),
        torch.empty(0, 1, 2, dtype=torch.bool, device=device),
        torch.empty(0, 1, 0, 0, dtype=torch.int64, device=device),
        torch.empty(0, 1, 0, 0, dtype=torch.int32, device=device),
        None,
        None,
    )
    torch.cuda.synchronize()

    assert second.keys.data_ptr() == first_pointer
    torch.testing.assert_close(second.keys[0], keys[0, 1, 0].view(1, head_size))
    assert empty.keys.shape == (0, 1, head_size)
    assert empty.exact_seq_lens.numel() == 0
    assert empty.cu_seqlens_k.tolist() == [0]
