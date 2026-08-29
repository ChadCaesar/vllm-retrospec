# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.execution import (
    EXACT_ATTENTION_PARTITION_SIZE,
    RetroSpecExactAttentionWorkspace,
    RetroSpecExactKVSource,
    RetroSpecExactPageKVSource,
    RetroSpecExactPrimaryKVSource,
)


def test_exact_attention_workspace_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="page_size must be positive"):
        RetroSpecExactAttentionWorkspace(
            page_size=0, max_num_queries=1, partition_capacity=1
        )
    with pytest.raises(ValueError, match="max_num_queries must be positive"):
        RetroSpecExactAttentionWorkspace(
            page_size=16, max_num_queries=0, partition_capacity=1
        )
    with pytest.raises(ValueError, match="partition_capacity must be positive"):
        RetroSpecExactAttentionWorkspace(
            page_size=16, max_num_queries=1, partition_capacity=0
        )
    with pytest.raises(ValueError, match="power of two"):
        RetroSpecExactAttentionWorkspace(
            page_size=16, max_num_queries=1, partition_capacity=3
        )


def _make_source(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    token_mask: torch.Tensor,
    page_token_counts: torch.Tensor,
    resident_pages: RetroSpecExactPageKVSource | None = None,
    staging_pages: RetroSpecExactPageKVSource | None = None,
    ready_event: torch.cuda.Event | None = None,
) -> RetroSpecExactKVSource:
    return RetroSpecExactKVSource(
        primary=RetroSpecExactPrimaryKVSource(
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            token_indices=token_indices,
            token_mask=token_mask,
            ready_event=ready_event,
        ),
        page_token_counts=page_token_counts,
        resident_pages=resident_pages,
        staging_pages=staging_pages,
    )


def _materialize_head(
    source: RetroSpecExactKVSource,
    request_idx: int,
    kv_head_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    primary = source.primary
    keys = []
    values = []
    for logical_idx, selected in zip(
        primary.token_indices[request_idx, kv_head_idx].tolist(),
        primary.token_mask[request_idx, kv_head_idx].tolist(),
    ):
        if not selected:
            continue
        logical_block, block_offset = divmod(logical_idx, primary.key_cache.shape[1])
        physical_block = int(primary.block_table[request_idx, logical_block])
        keys.append(primary.key_cache[physical_block, block_offset, kv_head_idx])
        values.append(primary.value_cache[physical_block, block_offset, kv_head_idx])

    page_counts = source.page_token_counts[request_idx, kv_head_idx].reshape(-1)
    for page_slot, token_count in enumerate(page_counts.tolist()):
        page_source = None
        page_id = -1
        if source.resident_pages is not None:
            candidate = int(
                source.resident_pages.page_ids[request_idx, kv_head_idx].reshape(-1)[
                    page_slot
                ]
            )
            if candidate >= 0:
                page_source = source.resident_pages
                page_id = candidate
        if page_source is None and source.staging_pages is not None:
            candidate = int(
                source.staging_pages.page_ids[request_idx, kv_head_idx].reshape(-1)[
                    page_slot
                ]
            )
            if candidate >= 0:
                page_source = source.staging_pages
                page_id = candidate
        if page_source is not None:
            keys.extend(page_source.key_pages[page_id, :token_count])
            values.extend(page_source.value_pages[page_id, :token_count])

    head_size = primary.key_cache.shape[-1]
    if not keys:
        empty = primary.key_cache.new_empty(0, head_size)
        return empty, empty
    return torch.stack(keys), torch.stack(values)


def _reference_attention(
    source: RetroSpecExactKVSource,
    query: torch.Tensor,
    scale: float,
    request_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_queries, num_query_heads, _ = query.shape
    num_kv_heads = source.primary.key_cache.shape[2]
    queries_per_kv_head = num_query_heads // num_kv_heads
    output = torch.zeros_like(query)
    lse = torch.full(
        (num_query_heads, num_queries),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    for query_idx in range(num_queries):
        request_idx = (
            query_idx if request_indices is None else int(request_indices[query_idx])
        )
        for query_head_idx in range(num_query_heads):
            kv_head_idx = query_head_idx // queries_per_kv_head
            keys, values = _materialize_head(source, request_idx, kv_head_idx)
            if not keys.numel():
                continue
            logits = torch.mv(keys.float(), query[query_idx, query_head_idx].float())
            logits *= scale
            weights = torch.softmax(logits, dim=0)
            output[query_idx, query_head_idx] = torch.mv(
                values.float().t(), weights
            ).to(query.dtype)
            lse[query_head_idx, query_idx] = torch.logsumexp(logits, dim=0)
    return output, lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multi_source_exact_attention_matches_reference_on_cuda():
    device = torch.device("cuda")
    torch.manual_seed(0)
    page_size = 4
    batch_size = 2
    num_kv_heads = 2
    num_query_heads = 4
    head_size = 64
    key_cache = torch.randn(
        8, page_size, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor(
        [[2, 0, 1, 3], [4, 6, 5, 7]], dtype=torch.int32, device=device
    )
    token_indices = torch.tensor(
        [
            [[0, 3, 7, 10], [1, 4, 8, 11]],
            [[0, 2, 5, 9], [1, 3, 6, 10]],
        ],
        dtype=torch.int64,
        device=device,
    )
    token_mask = torch.tensor(
        [
            [[True, False, True, True], [True, True, False, True]],
            [[True, True, False, True], [False, True, True, True]],
        ],
        device=device,
    )
    page_counts = torch.tensor(
        [
            [[[4, 2]], [[3, 0]]],
            [[[1, 4]], [[2, 3]]],
        ],
        dtype=torch.int32,
        device=device,
    )
    resident_keys = torch.randn(
        4, page_size, head_size, dtype=torch.float16, device=device
    )
    resident_values = torch.randn_like(resident_keys)
    staging_keys = torch.randn_like(resident_keys)
    staging_values = torch.randn_like(resident_keys)
    resident_ids = torch.tensor(
        [[[[0, -1]], [[1, -1]]], [[[2, -1]], [[3, -1]]]],
        dtype=torch.int64,
        device=device,
    )
    staging_ids = torch.tensor(
        [[[[-1, 0]], [[-1, -1]]], [[[-1, 1]], [[-1, 2]]]],
        dtype=torch.int64,
        device=device,
    )
    source = _make_source(
        key_cache,
        value_cache,
        block_table,
        token_indices,
        token_mask,
        page_counts,
        resident_pages=RetroSpecExactPageKVSource(
            resident_keys, resident_values, resident_ids
        ),
        staging_pages=RetroSpecExactPageKVSource(
            staging_keys, staging_values, staging_ids
        ),
    )
    query = torch.randn(
        batch_size,
        num_query_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    workspace = RetroSpecExactAttentionWorkspace(page_size, batch_size, 1)

    output, lse = workspace.run(source, query, scale=0.125)
    expected_output, expected_lse = _reference_attention(source, query, 0.125)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_attention_maps_multiple_queries_to_their_requests():
    device = torch.device("cuda")
    page_size = 4
    head_size = 64
    key_cache = torch.randn(
        4, page_size, 1, head_size, dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    source = _make_source(
        key_cache,
        value_cache,
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device),
        torch.tensor([[[0, 1, 2]], [[0, 2, 5]]], device=device),
        torch.ones(2, 1, 3, dtype=torch.bool, device=device),
        torch.empty(2, 1, 0, 0, dtype=torch.int32, device=device),
    )
    query = torch.randn(3, 2, head_size, dtype=torch.bfloat16, device=device)
    request_indices = torch.tensor([0, 0, 1], dtype=torch.int64, device=device)
    workspace = RetroSpecExactAttentionWorkspace(page_size, 3, 1)

    output, lse = workspace.run(source, query, 0.125, request_indices)
    expected = _reference_attention(source, query, 0.125, request_indices)

    torch.testing.assert_close(output, expected[0], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, expected[1], atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_attention_reduces_multiple_partitions_and_reuses_workspace():
    device = torch.device("cuda")
    page_size = 16
    head_size = 64
    num_tokens = EXACT_ATTENTION_PARTITION_SIZE + 37
    num_blocks = (num_tokens + page_size - 1) // page_size
    key_cache = torch.randn(
        num_blocks, page_size, 1, head_size, dtype=torch.float16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    source = _make_source(
        key_cache,
        value_cache,
        torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1),
        torch.arange(num_tokens, dtype=torch.int64, device=device).view(1, 1, -1),
        torch.ones(1, 1, num_tokens, dtype=torch.bool, device=device),
        torch.empty(1, 1, 0, 0, dtype=torch.int32, device=device),
    )
    query = torch.randn(1, 2, head_size, dtype=torch.float16, device=device)
    workspace = RetroSpecExactAttentionWorkspace(page_size, 2, 4)

    output, lse = workspace.run(source, query, 0.125)
    partial_pointer = workspace._partial_output.data_ptr()
    second_output, _ = workspace.run(source, query, 0.125)
    expected = _reference_attention(source, query, 0.125)

    torch.testing.assert_close(output, expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected[1], atol=3e-3, rtol=3e-3)
    assert workspace._partial_output.shape[2] == 4
    assert workspace._partial_output.data_ptr() == partial_pointer
    assert second_output.data_ptr() == output.data_ptr()

    undersized_workspace = RetroSpecExactAttentionWorkspace(page_size, 2, 1)
    with pytest.raises(RuntimeError, match="planned workspace capacity"):
        undersized_workspace.run(source, query, 0.125)
    assert undersized_workspace._partial_output is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_attention_handles_empty_sources():
    device = torch.device("cuda")
    key_cache = torch.empty(1, 4, 1, 64, dtype=torch.float16, device=device)
    source = _make_source(
        key_cache,
        key_cache.clone(),
        torch.zeros(2, 1, dtype=torch.int32, device=device),
        torch.empty(2, 1, 0, dtype=torch.int64, device=device),
        torch.empty(2, 1, 0, dtype=torch.bool, device=device),
        torch.empty(2, 1, 0, 0, dtype=torch.int32, device=device),
    )
    query = torch.ones(2, 2, 64, dtype=torch.float16, device=device)

    output, lse = RetroSpecExactAttentionWorkspace(4, 2, 1).run(source, query, 1.0)

    assert not output.any()
    assert torch.isneginf(lse).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_attention_waits_for_page_ready_event():
    device = torch.device("cuda")
    page_size = 4
    head_size = 64
    key_cache = torch.empty(
        1, page_size, 1, head_size, dtype=torch.float16, device=device
    )
    source_keys = torch.randn(
        1, page_size, head_size, dtype=torch.float16, device=device
    )
    source_values = torch.randn_like(source_keys)
    staged_keys = torch.zeros_like(source_keys)
    staged_values = torch.zeros_like(source_values)
    copy_stream = torch.cuda.Stream(device=device)
    copy_stream.wait_stream(torch.cuda.current_stream(device))
    ready_event = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):
        torch.cuda._sleep(20_000_000)
        staged_keys.copy_(source_keys)
        staged_values.copy_(source_values)
        ready_event.record()

    page_ids = torch.zeros(1, 1, 1, 1, dtype=torch.int64, device=device)
    source = _make_source(
        key_cache,
        key_cache.clone(),
        torch.zeros(1, 1, dtype=torch.int32, device=device),
        torch.empty(1, 1, 0, dtype=torch.int64, device=device),
        torch.empty(1, 1, 0, dtype=torch.bool, device=device),
        torch.full((1, 1, 1, 1), page_size, dtype=torch.int32, device=device),
        staging_pages=RetroSpecExactPageKVSource(
            staged_keys, staged_values, page_ids, ready_event
        ),
    )
    query = torch.randn(1, 1, head_size, dtype=torch.float16, device=device)

    output, lse = RetroSpecExactAttentionWorkspace(page_size, 1, 1).run(
        source, query, 0.125
    )
    expected = _reference_attention(source, query, 0.125)
    torch.cuda.current_stream(device).synchronize()

    torch.testing.assert_close(output, expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected[1], atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_attention_rejects_query_over_workspace_capacity():
    device = torch.device("cuda")
    key_cache = torch.zeros(1, 4, 1, 64, dtype=torch.float16, device=device)
    source = _make_source(
        key_cache,
        key_cache.clone(),
        torch.zeros(2, 1, dtype=torch.int32, device=device),
        torch.zeros(2, 1, 1, dtype=torch.int64, device=device),
        torch.ones(2, 1, 1, dtype=torch.bool, device=device),
        torch.empty(2, 1, 0, 0, dtype=torch.int32, device=device),
    )

    with pytest.raises(ValueError, match="workspace capacity"):
        RetroSpecExactAttentionWorkspace(4, 1, 1).run(
            source,
            torch.zeros(2, 1, 64, dtype=torch.float16, device=device),
            1.0,
        )
