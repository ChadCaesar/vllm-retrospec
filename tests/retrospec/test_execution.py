# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterPageStore,
)
from vllm.v1.spec_decode.retrospec.execution import (
    RetroSpecExactExecutionBuffer,
    RetroSpecExactKVSource,
    RetroSpecExactPageKVSource,
    RetroSpecExactPrimaryKVSource,
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


def _make_exact_source(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    primary_token_indices: torch.Tensor,
    primary_token_mask: torch.Tensor,
    page_token_counts: torch.Tensor,
    page_sources: tuple[RetroSpecExactPageKVSource, ...] = (),
    primary_ready_event: torch.cuda.Event | None = None,
) -> RetroSpecExactKVSource:
    return RetroSpecExactKVSource(
        primary=RetroSpecExactPrimaryKVSource(
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            token_indices=primary_token_indices,
            token_mask=primary_token_mask,
            ready_event=primary_ready_event,
        ),
        page_token_counts=page_token_counts,
        page_sources=page_sources,
    )


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
        _make_exact_source(
            keys,
            values,
            block_table,
            primary_indices,
            primary_mask,
            page_counts,
            page_sources=(
                RetroSpecExactPageKVSource(
                    key_pages=cluster_keys,
                    value_pages=cluster_values,
                    page_ids=page_ids,
                ),
            ),
        )
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
def test_execution_buffer_preserves_order_across_resident_and_staging_pages():
    device = torch.device("cuda")
    page_size = 2
    head_size = 64
    keys, values = _make_primary_cache(1, page_size, 1, head_size, device)
    cluster_keys, cluster_values = _make_cluster_pages(4, page_size, head_size, device)
    logical_page_ids = torch.tensor(
        [[[[0, 1], [2, 3]]]], dtype=torch.int64, device=device
    )
    resident_page_ids = torch.tensor(
        [[[[0, -1], [1, -1]]]], dtype=torch.int64, device=device
    )
    staging_page_ids = torch.tensor(
        [[[[-1, 0], [-1, 1]]]], dtype=torch.int64, device=device
    )
    page_counts = torch.tensor([[[[2, 1], [2, 1]]]], dtype=torch.int32, device=device)
    resident_keys = cluster_keys.index_select(0, torch.tensor([0, 2], device=device))
    resident_values = cluster_values.index_select(
        0, torch.tensor([0, 2], device=device)
    )
    staging_keys = cluster_keys.index_select(0, torch.tensor([1, 3], device=device))
    staging_values = cluster_values.index_select(0, torch.tensor([1, 3], device=device))

    execution = RetroSpecExactExecutionBuffer(page_size).pack(
        _make_exact_source(
            keys,
            values,
            torch.tensor([[0]], dtype=torch.int32, device=device),
            torch.empty(1, 1, 0, dtype=torch.int64, device=device),
            torch.empty(1, 1, 0, dtype=torch.bool, device=device),
            page_counts,
            page_sources=(
                RetroSpecExactPageKVSource(
                    key_pages=resident_keys,
                    value_pages=resident_values,
                    page_ids=resident_page_ids,
                ),
                RetroSpecExactPageKVSource(
                    key_pages=staging_keys,
                    value_pages=staging_values,
                    page_ids=staging_page_ids,
                ),
            ),
        )
    )
    torch.cuda.synchronize()

    expected_keys = _reference_packed_tokens(
        keys,
        torch.tensor([[0]], dtype=torch.int32, device=device),
        torch.empty(1, 1, 0, dtype=torch.int64, device=device),
        torch.empty(1, 1, 0, dtype=torch.bool, device=device),
        logical_page_ids,
        page_counts,
        cluster_keys,
        page_size,
    )
    expected_values = _reference_packed_tokens(
        values,
        torch.tensor([[0]], dtype=torch.int32, device=device),
        torch.empty(1, 1, 0, dtype=torch.int64, device=device),
        torch.empty(1, 1, 0, dtype=torch.bool, device=device),
        logical_page_ids,
        page_counts,
        cluster_values,
        page_size,
    )

    assert execution.exact_seq_lens.tolist() == [6]
    torch.testing.assert_close(execution.keys[:6], expected_keys)
    torch.testing.assert_close(execution.values[:6], expected_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_execution_buffer_waits_for_page_source_copy_event():
    device = torch.device("cuda")
    page_size = 2
    head_size = 64
    primary_keys, primary_values = _make_primary_cache(
        1, page_size, 1, head_size, device
    )
    source_keys, source_values = _make_cluster_pages(1, page_size, head_size, device)
    resident_keys = torch.zeros_like(source_keys)
    resident_values = torch.zeros_like(source_values)

    current_stream = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device=device)
    copy_stream.wait_stream(current_stream)
    ready_event = torch.cuda.Event()

    with torch.cuda.stream(copy_stream):
        # Keep the producer behind the consumer long enough that omitting the
        # event dependency would deterministically pack the zero-filled pages.
        torch.cuda._sleep(20_000_000)
        resident_keys.copy_(source_keys)
        resident_values.copy_(source_values)
        ready_event.record()

    page_counts = torch.full((1, 1, 1, 1), page_size, dtype=torch.int32, device=device)
    page_source = RetroSpecExactPageKVSource(
        key_pages=resident_keys,
        value_pages=resident_values,
        page_ids=torch.zeros(1, 1, 1, 1, dtype=torch.int64, device=device),
        ready_event=ready_event,
    )
    execution = RetroSpecExactExecutionBuffer(page_size).pack(
        _make_exact_source(
            primary_keys,
            primary_values,
            torch.tensor([[0]], dtype=torch.int32, device=device),
            torch.empty(1, 1, 0, dtype=torch.int64, device=device),
            torch.empty(1, 1, 0, dtype=torch.bool, device=device),
            page_counts,
            page_sources=(page_source,),
        )
    )
    current_stream.synchronize()

    torch.testing.assert_close(
        execution.keys[:page_size], source_keys[0].view(page_size, 1, head_size)
    )
    torch.testing.assert_close(
        execution.values[:page_size], source_values[0].view(page_size, 1, head_size)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_execution_buffer_waits_for_primary_copy_event():
    device = torch.device("cuda")
    page_size = 2
    head_size = 64
    source_keys, source_values = _make_primary_cache(1, page_size, 1, head_size, device)
    staged_keys = torch.zeros_like(source_keys)
    staged_values = torch.zeros_like(source_values)

    current_stream = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device=device)
    copy_stream.wait_stream(current_stream)
    ready_event = torch.cuda.Event()

    with torch.cuda.stream(copy_stream):
        torch.cuda._sleep(20_000_000)
        staged_keys.copy_(source_keys)
        staged_values.copy_(source_values)
        ready_event.record()

    source = _make_exact_source(
        staged_keys,
        staged_values,
        torch.tensor([[0]], dtype=torch.int32, device=device),
        torch.tensor([[[0, 1]]], dtype=torch.int64, device=device),
        torch.tensor([[[True, True]]], device=device),
        torch.empty(1, 1, 0, 0, dtype=torch.int32, device=device),
        primary_ready_event=ready_event,
    )
    execution = RetroSpecExactExecutionBuffer(page_size).pack(source)
    current_stream.synchronize()

    torch.testing.assert_close(
        execution.keys[:page_size], source_keys[0].view(page_size, 1, head_size)
    )
    torch.testing.assert_close(
        execution.values[:page_size], source_values[0].view(page_size, 1, head_size)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_execution_buffer_requires_a_source_for_nonempty_page_metadata():
    device = torch.device("cuda")
    page_size = 2
    head_size = 64
    keys, values = _make_primary_cache(1, page_size, 1, head_size, device)
    source = _make_exact_source(
        keys,
        values,
        torch.tensor([[0]], dtype=torch.int32, device=device),
        torch.empty(1, 1, 0, dtype=torch.int64, device=device),
        torch.empty(1, 1, 0, dtype=torch.bool, device=device),
        torch.full((1, 1, 1, 1), page_size, dtype=torch.int32, device=device),
    )

    with pytest.raises(RuntimeError, match="at least one page source"):
        RetroSpecExactExecutionBuffer(page_size).pack(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_execution_buffer_packs_resolved_cpu_backing_pages():
    device = torch.device("cuda")
    page_size = 2
    head_size = 64
    token_keys = (
        torch.arange(6 * head_size, dtype=torch.float32, device=device)
        .to(torch.bfloat16)
        .view(1, 6, head_size)
    )
    token_values = token_keys + 10
    assignments = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.int64, device=device)
    cluster_counts = torch.tensor([[2, 2, 2]], dtype=torch.int32, device=device)
    token_offsets = torch.tensor(
        [[0, 1, 0, 1, 0, 1]],
        dtype=torch.int32,
        device=device,
    )
    store = RetroSpecClusterPageStore(
        page_size=page_size,
        cache_ratio=0.5,
    )
    table = store.store_clusters(
        layer_name="layer",
        request_id="request",
        cluster_start=0,
        token_keys=token_keys,
        token_values=token_values,
        assignments=assignments,
        cluster_token_counts=cluster_counts,
        token_offsets_in_cluster=token_offsets,
    )
    cluster_ids = table.cluster_ids.to(device=device).unsqueeze(0)
    metadata = store.get_cluster_block_metadata("layer", cluster_ids, device=device)
    logical_page_ids = metadata.page_ids
    page_counts = metadata.page_token_counts
    resolved = store.resolve_cluster_blocks("layer", cluster_ids, logical_page_ids)

    primary_keys, primary_values = _make_primary_cache(
        1, page_size, 1, head_size, device
    )
    execution = RetroSpecExactExecutionBuffer(page_size).pack(
        _make_exact_source(
            primary_keys,
            primary_values,
            torch.tensor([[0]], dtype=torch.int32, device=device),
            torch.empty(1, 1, 0, dtype=torch.int64, device=device),
            torch.empty(1, 1, 0, dtype=torch.bool, device=device),
            page_counts,
            page_sources=(
                RetroSpecExactPageKVSource(
                    key_pages=resolved.resident_key_pages,
                    value_pages=resolved.resident_value_pages,
                    page_ids=resolved.resident_page_ids,
                    ready_event=resolved.resident_ready_event,
                ),
                RetroSpecExactPageKVSource(
                    key_pages=resolved.staging_key_pages,
                    value_pages=resolved.staging_value_pages,
                    page_ids=resolved.staging_page_ids,
                ),
            ),
        )
    )
    torch.cuda.synchronize()

    assert not (resolved.resident_page_ids >= 0).any()
    assert (resolved.staging_page_ids >= 0).sum().item() == 3
    assert execution.exact_seq_lens.tolist() == [6]
    torch.testing.assert_close(execution.keys[:6], token_keys[0].view(6, 1, head_size))
    torch.testing.assert_close(
        execution.values[:6], token_values[0].view(6, 1, head_size)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_execution_buffer_handles_empty_batch_and_reuses_storage_on_cuda():
    device = torch.device("cuda")
    page_size = 2
    head_size = 64
    keys, values = _make_primary_cache(2, page_size, 1, head_size, device)
    buffer = RetroSpecExactExecutionBuffer(page_size)

    first = buffer.pack(
        _make_exact_source(
            keys,
            values,
            torch.tensor([[0, 1]], dtype=torch.int32, device=device),
            torch.tensor([[[0, 1]]], dtype=torch.int64, device=device),
            torch.tensor([[[True, True]]], device=device),
            torch.empty(1, 1, 0, 0, dtype=torch.int32, device=device),
        )
    )
    first_pointer = first.keys.data_ptr()

    second = buffer.pack(
        _make_exact_source(
            keys,
            values,
            torch.tensor([[0, 1]], dtype=torch.int32, device=device),
            torch.tensor([[[1]]], dtype=torch.int64, device=device),
            torch.tensor([[[True]]], device=device),
            torch.empty(1, 1, 0, 0, dtype=torch.int32, device=device),
        )
    )
    empty = buffer.pack(
        _make_exact_source(
            keys,
            values,
            torch.empty(0, 0, dtype=torch.int32, device=device),
            torch.empty(0, 1, 2, dtype=torch.int64, device=device),
            torch.empty(0, 1, 2, dtype=torch.bool, device=device),
            torch.empty(0, 1, 0, 0, dtype=torch.int32, device=device),
        )
    )
    torch.cuda.synchronize()

    assert second.keys.data_ptr() == first_pointer
    torch.testing.assert_close(second.keys[0], keys[0, 1, 0].view(1, head_size))
    assert empty.keys.shape == (0, 1, head_size)
    assert empty.exact_seq_lens.numel() == 0
    assert empty.cu_seqlens_k.tolist() == [0]
