# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.gpu_native import (
    RetroSpecGPUNativeIndex,
    _cluster_rank_dot_kernel,
    _NativeBatchLayer,
    _NativeLayerRecord,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("clusters", (1914, 5000))
def test_gpu_native_rank_dot_matches_float32_reference(
    dtype: torch.dtype, clusters: int
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this GPU")
    torch.manual_seed(41)
    batch, heads, groups, head_size = 2, 4, 7, 128
    query = torch.randn(batch, heads, groups, head_size, device="cuda", dtype=dtype)
    keys = torch.randn(batch, heads, clusters, head_size, device="cuda", dtype=dtype)
    logits = torch.empty(batch, heads, groups, clusters, device="cuda")
    _cluster_rank_dot_kernel[(triton.cdiv(clusters, 128), batch * heads)](
        query,
        keys,
        logits,
        query,
        *query.stride()[:3],
        *keys.stride()[:3],
        *logits.stride()[:3],
        heads,
        groups,
        clusters,
        head_size,
        128,
        head_size,
        False,
        num_warps=4,
    )
    reference = torch.einsum("bhgd,bhcd->bhgc", query.float(), keys.float())
    torch.testing.assert_close(logits, reference, atol=2e-5, rtol=2e-5)
    width = triton.cdiv(clusters, 4)
    assert torch.equal(
        logits.topk(width, dim=-1).indices.sort(dim=-1).values,
        reference.topk(width, dim=-1).indices.sort(dim=-1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("clusters", (33, 8193))
def test_gpu_native_rank_maps_verification_rows_to_request_slots(clusters: int) -> None:
    torch.manual_seed(63)
    device = torch.device("cuda")
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=8,
        retrieval_ratio=0.125,
        estimation_ratio=0.25,
        prefill_segment_size_tokens=32,
        generation_update_interval=16,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    keys = torch.randn(2, 2, clusters, 32, device=device, dtype=torch.bfloat16)
    counts = torch.randint(1, 9, (2, 2, clusters), device=device, dtype=torch.int32)
    counts[1, :, ::3] = 0
    layout = _NativeBatchLayer(
        keys=keys,
        values=keys,
        counts=counts,
        token_indices=torch.empty(2, 2, 0, device=device, dtype=torch.int32),
        cluster_offsets=torch.empty(2, 2, 0, device=device, dtype=torch.int32),
        indexed_ends=torch.zeros(2, device=device, dtype=torch.int32),
    )
    request_rows = torch.tensor([1, 0, 1], device=device, dtype=torch.int64)
    query = torch.randn(3, 4, 32, device=device, dtype=torch.bfloat16)
    active = torch.ones(3, device=device, dtype=torch.bool)
    ranked = index._compute_rank_draft(
        query, 32**-0.5, active, layout, request_rows=request_rows
    )
    mapped = _NativeBatchLayer(
        keys=keys.index_select(0, request_rows),
        values=keys,
        counts=counts.index_select(0, request_rows),
        token_indices=layout.token_indices,
        cluster_offsets=layout.cluster_offsets,
        indexed_ends=layout.indexed_ends,
    )
    reference = index._compute_rank_draft_torch(query, 32**-0.5, active, mapped)
    assert torch.equal(ranked.candidate_counts, reference.candidate_counts)
    assert torch.equal(ranked.ranked, reference.ranked)
    torch.testing.assert_close(
        ranked.sparse_mass, reference.sparse_mass, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        ranked.verification_mass, reference.verification_mass, atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_bonus_rank_plan_uses_request_and_step_slots() -> None:
    torch.manual_seed(70)
    device = torch.device("cuda")
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=8,
        retrieval_ratio=0.125,
        estimation_ratio=0.25,
        prefill_segment_size_tokens=32,
        generation_update_interval=16,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    keys = torch.randn(2, 2, 33, 32, device=device, dtype=torch.bfloat16)
    counts = torch.ones(2, 2, 33, device=device, dtype=torch.int32)
    counts[1, :, ::2] = 0
    layout = _NativeBatchLayer(
        keys=keys,
        values=keys,
        counts=counts,
        token_indices=torch.empty(2, 2, 0, device=device, dtype=torch.int32),
        cluster_offsets=torch.empty(2, 2, 0, device=device, dtype=torch.int32),
        indexed_ends=torch.zeros(2, device=device, dtype=torch.int32),
    )
    index._prepare_plan_workspace("layer", layout, 2, device)
    workspace = index._active_plan_workspaces["layer"]
    workspace.ranked.fill_(-1)
    request_rows = torch.tensor([1, 0], device=device, dtype=torch.int64)
    token_steps = torch.tensor([2, 5], device=device, dtype=torch.int64)
    query = torch.randn(2, 4, 32, device=device, dtype=torch.bfloat16)
    active = torch.ones(2, device=device, dtype=torch.bool)
    bonus = index._rank_parallel_bonus(
        "layer", query, 32**-0.5, request_rows, token_steps, active, layout
    )
    assert torch.equal(workspace.ranked[token_steps, request_rows], bonus.ranked)
    assert torch.equal(
        workspace.candidate_counts[token_steps, request_rows], bonus.candidate_counts
    )
    torch.testing.assert_close(
        workspace.verification_mass[token_steps, request_rows],
        bonus.verification_mass,
    )
    assert torch.all(workspace.ranked[2, 0] == -1)
    assert torch.all(workspace.ranked[5, 1] == -1)


def _reference_attention(
    index: RetroSpecGPUNativeIndex,
    layer_name: str,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    plan_step: int,
    expanded: bool,
    *,
    sparse_verify: bool = False,
    request_id: str = "request",
    plan_row: int = 0,
    block_table_row: torch.Tensor | None = None,
    seq_len: int | None = None,
) -> torch.Tensor:
    record = index._records[layer_name][request_id]
    plan = index._plans[layer_name][plan_step]
    num_kv_heads = key_cache.shape[2]
    group_size = query.shape[1] // num_kv_heads
    if block_table_row is None:
        block_table_row = torch.arange(key_cache.shape[0], device=query.device)
    if seq_len is None:
        seq_len = block_table_row.numel() * 16

    def token_kv(token: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
        physical = int(block_table_row[token // 16])
        return (
            key_cache[physical, token % 16, head],
            value_cache[physical, token % 16, head],
        )

    outputs = []
    for head in range(query.shape[1]):
        kv_head = head // group_size
        candidates = int(plan.candidate_counts[plan_row, kv_head])
        retrieval = int(torch.ceil(torch.tensor(candidates * index.retrieval_ratio)))
        estimation = min(
            int(torch.ceil(torch.tensor(candidates * index.estimation_ratio))),
            candidates - retrieval,
        )
        total = retrieval + estimation
        if expanded:
            retrieval = total
        elif sparse_verify:
            retrieval += int(
                torch.ceil(
                    torch.tensor(estimation * index.sparse_verify_exact_fraction)
                )
            )
        keys = []
        values = []
        weights = []
        for token in list(range(16)) + list(range(record.indexed_end, seq_len)):
            token_key, token_value = token_kv(token, kv_head)
            keys.append(token_key)
            values.append(token_value)
            weights.append(1)
        for rank in range(retrieval):
            cluster = int(plan.ranked[plan_row, kv_head, rank])
            start = int(record.cluster_offsets[kv_head, cluster])
            end = int(record.cluster_offsets[kv_head, cluster + 1])
            for offset in range(start, end):
                token = int(record.token_indices[kv_head, offset])
                token_key, token_value = token_kv(token, kv_head)
                keys.append(token_key)
                values.append(token_value)
                weights.append(1)
        for rank in range(retrieval, total):
            cluster = int(plan.ranked[plan_row, kv_head, rank])
            keys.append(record.keys[kv_head, cluster])
            values.append(record.values[kv_head, cluster])
            weights.append(int(record.counts[kv_head, cluster]))
        stacked_keys = torch.stack(keys).float()
        stacked_values = torch.stack(values).float()
        logits = stacked_keys @ query[0, head].float() * (query.shape[-1] ** -0.5)
        logits += torch.tensor(weights, device=query.device).float().log()
        outputs.append(torch.softmax(logits, 0) @ stacked_values)
    return torch.stack(outputs).to(query.dtype).unsqueeze(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("query_heads", (4, 14))
def test_gpu_native_clustered_draft_and_expanded_attention(query_heads: int) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.25,
        estimation_ratio=0.5,
        prefill_segment_size_tokens=32,
        generation_update_interval=16,
        blocks_per_cluster=1,
        num_kmeans_iterations=2,
    )
    key_cache = torch.randn(8, 16, 2, 32, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(8, device=device, dtype=torch.int32).unsqueeze(0)
    index.build_or_update(
        "layer",
        ("request",),
        (128,),
        (True,),
        (0,),
        key_cache,
        value_cache,
        block_table,
        prefill_complete=(True,),
    )
    index.flush_staged_updates()
    record = index._records["layer"]["request"]
    assert record.indexed_end == 96
    assert all(
        tensor.is_cuda
        for tensor in (
            record.keys,
            record.values,
            record.counts,
            record.token_indices,
            record.cluster_offsets,
        )
    )
    index.begin_proposal(("request",))
    try:
        query = torch.randn(1, query_heads, 32, device=device, dtype=torch.float16)
        output = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([128], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=32**-0.5,
            output=output,
            step=0,
        )
        torch.testing.assert_close(
            output,
            _reference_attention(
                index, "layer", query, key_cache, value_cache, 0, False
            ),
            atol=2e-2,
            rtol=2e-2,
        )
        first_ranked = index._plans["layer"][0].ranked.clone()
        next_query = torch.randn_like(query)
        next_output = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=next_query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([128], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=32**-0.5,
            output=next_output,
            step=1,
        )
        assert torch.equal(index._plans["layer"][0].ranked, first_ranked)
        torch.testing.assert_close(
            next_output,
            _reference_attention(
                index, "layer", next_query, key_cache, value_cache, 1, False
            ),
            atol=2e-2,
            rtol=2e-2,
        )
        sparse_output = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([128], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=32**-0.5,
            output=sparse_output,
            step=-1,
            sparse_verify=True,
            request_indices=torch.tensor([0], device=device, dtype=torch.int32),
            token_indices=torch.tensor([0], device=device, dtype=torch.int32),
        )
        torch.testing.assert_close(
            sparse_output,
            _reference_attention(
                index,
                "layer",
                query,
                key_cache,
                value_cache,
                0,
                False,
                sparse_verify=True,
            ),
            atol=2e-2,
            rtol=2e-2,
        )
        expanded_output = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([128], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=32**-0.5,
            output=expanded_output,
            step=-1,
            expanded=True,
            request_indices=torch.tensor([0], device=device, dtype=torch.int32),
            token_indices=torch.tensor([0], device=device, dtype=torch.int32),
        )
        torch.testing.assert_close(
            expanded_output,
            _reference_attention(
                index, "layer", query, key_cache, value_cache, 0, True
            ),
            atol=2e-2,
            rtol=2e-2,
        )
    finally:
        index.end_proposal()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_parallel_verification_routes_requests() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.25,
        estimation_ratio=0.5,
        prefill_segment_size_tokens=32,
        generation_update_interval=16,
        blocks_per_cluster=1,
        num_kmeans_iterations=2,
    )
    key_cache = torch.randn(16, 16, 2, 32, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(16, device=device, dtype=torch.int32).reshape(2, 8)
    index.build_or_update(
        "layer",
        ("first", "second"),
        (128, 128),
        (True, True),
        (0, 1),
        key_cache,
        value_cache,
        block_table,
        prefill_complete=(True, True),
    )
    index.flush_staged_updates()
    index.begin_proposal(("first", "second"))
    try:
        draft_query = torch.randn(2, 4, 32, device=device, dtype=torch.float16)
        index.forward(
            layer_name="layer",
            query=draft_query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([128, 128], device=device, dtype=torch.int32),
            active_mask=torch.ones(2, device=device, dtype=torch.bool),
            scale=32**-0.5,
            output=torch.empty_like(draft_query),
            step=0,
        )
        request_indices = torch.tensor([1, 0, 1], device=device, dtype=torch.int32)
        verification_query = torch.randn(3, 4, 32, device=device, dtype=torch.float16)
        output = torch.empty_like(verification_query)
        index.forward(
            layer_name="layer",
            query=verification_query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table.index_select(0, request_indices.long()),
            seq_lens=torch.tensor([128, 128, 128], device=device, dtype=torch.int32),
            active_mask=torch.ones(3, device=device, dtype=torch.bool),
            scale=32**-0.5,
            output=output,
            step=-1,
            expanded=False,
            sparse_verify=True,
            request_indices=request_indices,
            token_indices=torch.zeros(3, device=device, dtype=torch.int32),
        )
        for row, request_row in enumerate((1, 0, 1)):
            expected = _reference_attention(
                index,
                "layer",
                verification_query[row : row + 1],
                key_cache,
                value_cache,
                0,
                False,
                sparse_verify=True,
                request_id=("first", "second")[request_row],
                plan_row=request_row,
                block_table_row=block_table[request_row],
                seq_len=128,
            )
            torch.testing.assert_close(
                output[row : row + 1], expected, atol=2e-2, rtol=2e-2
            )
    finally:
        index.end_proposal()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_rollback_drops_uncommitted_clusters() -> None:
    device = torch.device("cuda")
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.25,
        estimation_ratio=0.5,
        prefill_segment_size_tokens=32,
        generation_update_interval=16,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    key_cache = torch.randn(8, 16, 1, 32, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(8, device=device, dtype=torch.int32).unsqueeze(0)
    index.build_or_update(
        "layer",
        ("request",),
        (128,),
        (True,),
        (0,),
        key_cache,
        value_cache,
        block_table,
        prefill_complete=(True,),
    )
    index.flush_staged_updates()
    assert index.has_cluster_pages("layer", ("request",))
    assert index.needs_update("request", 32, ("layer",), False)
    index.build_or_update(
        "layer",
        ("request",),
        (32,),
        (False,),
        (0,),
        key_cache,
        value_cache,
        block_table,
    )
    index.flush_staged_updates()
    assert not index.has_cluster_pages("layer", ("request",))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_workspace_reuses_addresses_and_grows_on_demand() -> None:
    device = torch.device("cuda")
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.25,
        estimation_ratio=0.5,
        prefill_segment_size_tokens=32,
        generation_update_interval=16,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    attention_query = torch.empty(1, 2, 32, device=device)
    small_attention = index._attention_workspace(attention_query, 4)
    large_attention = index._attention_workspace(attention_query, 8)
    assert large_attention.output.data_ptr() != small_attention.output.data_ptr()
    assert index._attention_workspace(attention_query, 4) is large_attention

    def record(clusters: int) -> _NativeLayerRecord:
        counts = torch.full((1, clusters), 16, device=device, dtype=torch.int32)
        return _NativeLayerRecord(
            indexed_end=16 + clusters * 16,
            keys=torch.ones(1, clusters, 32, device=device),
            values=torch.ones(1, clusters, 32, device=device),
            counts=counts,
            token_indices=torch.arange(
                16, 16 + clusters * 16, device=device, dtype=torch.int32
            ).unsqueeze(0),
            cluster_offsets=torch.arange(
                clusters + 1, device=device, dtype=torch.int32
            ).unsqueeze(0)
            * 16,
        )

    index._records["layer"] = {"first": record(1)}
    index.begin_proposal(("first",))
    first = index._batch_layer("layer", device, torch.float32)
    first_address = first.keys.data_ptr()
    plan_address = index._active_plan_workspaces["layer"].ranked.data_ptr()
    assert first.keys.shape[:3] == (1, 1, 1)
    index._active_plan_workspaces["layer"].candidate_counts.fill_(7)
    index.end_proposal()

    index.begin_proposal(("first",))
    reused = index._batch_layer("layer", device, torch.float32)
    assert reused.keys.data_ptr() == first_address
    reused_plan = index._active_plan_workspaces["layer"]
    assert reused_plan.ranked.data_ptr() == plan_address
    assert not reused_plan.candidate_counts.any()
    index.end_proposal()

    index._records["layer"]["second"] = record(5)
    index.begin_proposal(("first", "second"))
    grown = index._batch_layer("layer", device, torch.float32)
    assert grown.keys.shape[:3] == (2, 1, 8)
    assert grown.keys.data_ptr() != first_address
    assert index._active_plan_workspaces["layer"].ranked.data_ptr() != plan_address
    assert torch.equal(
        grown.counts[0, 0, 1:], torch.zeros(7, device=device, dtype=torch.int32)
    )
    query = torch.randn(2, 2, 32, device=device)
    active = torch.tensor([True, False], device=device)
    fused = index._compute_rank_draft(query, 32**-0.5, active, grown)
    reference = index._compute_rank_draft_torch(query, 32**-0.5, active, grown)
    assert torch.equal(fused.candidate_counts, reference.candidate_counts)
    assert torch.equal(fused.ranked[0], reference.ranked[0])
    torch.testing.assert_close(
        fused.sparse_mass, reference.sparse_mass, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        fused.verification_mass, reference.verification_mass, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        fused.expanded_mass, reference.expanded_mass, atol=1e-5, rtol=1e-5
    )
    assert torch.all(fused.sparse_mass <= fused.verification_mass + 1e-5)
    assert torch.all(fused.verification_mass <= fused.expanded_mass + 1e-5)
    index._rank_draft("layer", query, 32**-0.5, active, 0, grown)
    assert "layer" not in index._logit_workspaces
    grown_plan_address = index._active_plan_workspaces["layer"].ranked.data_ptr()
    index.end_proposal()

    index.begin_proposal(("first",))
    shrunk = index._batch_layer("layer", device, torch.float32)
    assert index._active_cluster_counts["layer"] == 1
    assert (
        index._active_plan_workspaces["layer"].ranked.data_ptr() == grown_plan_address
    )
    small_query = query[:1]
    small_active = torch.tensor([True], device=device)
    shrunk_plan = index._rank_draft(
        "layer", small_query, 32**-0.5, small_active, 0, shrunk
    )
    shrunk_reference = index._compute_rank_draft_torch(
        small_query, 32**-0.5, small_active, shrunk
    )
    assert "layer" not in index._logit_workspaces
    assert shrunk_plan.candidate_counts.item() == 1
    assert shrunk_plan.ranked[0, 0, 0].item() == 0
    assert torch.equal(shrunk_plan.ranked, shrunk_reference.ranked)
    torch.testing.assert_close(shrunk_plan.sparse_mass, shrunk_reference.sparse_mass)
    index.end_proposal()

    index.begin_proposal(("first", "second"))
    grown_bf16 = index._batch_layer("layer", device, torch.bfloat16)
    index._rank_draft("layer", query.bfloat16(), 32**-0.5, active, 0, grown_bf16)
    logit_address = index._logit_workspaces["layer"].data_ptr()
    index.end_proposal()

    index.begin_proposal(("first",))
    shrunk_bf16 = index._batch_layer("layer", device, torch.bfloat16)
    index._rank_draft(
        "layer", small_query.bfloat16(), 32**-0.5, small_active, 0, shrunk_bf16
    )
    assert index._logit_workspaces["layer"].data_ptr() == logit_address
    index.end_proposal()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_large_rank_fallback_does_not_allocate_unused_logits() -> None:
    device = torch.device("cuda")
    clusters = 8193
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.018,
        estimation_ratio=0.232,
        prefill_segment_size_tokens=8192,
        generation_update_interval=1024,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    keys = torch.ones(1, 1, clusters, 32, device=device, dtype=torch.bfloat16)
    layout = _NativeBatchLayer(
        keys=keys,
        values=keys,
        counts=torch.ones(1, 1, clusters, device=device, dtype=torch.int32),
        token_indices=torch.empty(1, 1, 0, device=device, dtype=torch.int32),
        cluster_offsets=torch.empty(1, 1, 0, device=device, dtype=torch.int32),
        indexed_ends=torch.zeros(1, device=device, dtype=torch.int32),
    )
    index._prepare_plan_workspace("layer", layout, 1, device)
    query = torch.ones(1, 1, 32, device=device, dtype=torch.bfloat16)
    active = torch.ones(1, device=device, dtype=torch.bool)
    plan = index._rank_draft("layer", query, 32**-0.5, active, 0, layout)
    assert plan.candidate_counts.item() == clusters
    assert "layer" not in index._logit_workspaces


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_large_cluster_rank_matches_reference() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    clusters = 5000
    heads = 4
    head_size = 128
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.018,
        estimation_ratio=0.232,
        prefill_segment_size_tokens=8192,
        generation_update_interval=1024,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    index._records["layer"] = {
        "request": _NativeLayerRecord(
            indexed_end=16 + clusters * 16,
            keys=torch.randn(heads, clusters, head_size, device=device),
            values=torch.randn(heads, clusters, head_size, device=device),
            counts=torch.full((heads, clusters), 16, device=device, dtype=torch.int32),
            token_indices=torch.arange(
                16, 16 + clusters * 16, device=device, dtype=torch.int32
            ).expand(heads, -1),
            cluster_offsets=(
                torch.arange(clusters + 1, device=device, dtype=torch.int32) * 16
            ).expand(heads, -1),
        )
    }
    index.begin_proposal(("request",))
    try:
        layout = index._batch_layer("layer", device, torch.float32)
        query = torch.randn(1, 28, head_size, device=device)
        active_mask = torch.ones(1, device=device, dtype=torch.bool)
        fused = index._rank_draft(
            "layer", query, head_size**-0.5, active_mask, 0, layout
        )
        reference = index._compute_rank_draft_torch(
            query, head_size**-0.5, active_mask, layout
        )
        assert torch.equal(fused.candidate_counts, reference.candidate_counts)
        expanded_width = 2 * int(
            torch.ceil(torch.tensor(clusters * index.retrieval_ratio))
        )
        assert torch.equal(
            fused.ranked[..., :expanded_width],
            reference.ranked[..., :expanded_width].long(),
        )
        assert torch.equal(
            fused.ranked.sort(dim=-1).values,
            reference.ranked.long().sort(dim=-1).values,
        )
        torch.testing.assert_close(
            fused.sparse_mass, reference.sparse_mass, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            fused.verification_mass,
            reference.verification_mass,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            fused.expanded_mass, reference.expanded_mass, atol=1e-5, rtol=1e-5
        )
    finally:
        index.end_proposal()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_partitioned_attention_matches_reference() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    clusters = 1536
    heads = 2
    head_size = 128
    pages = clusters + 2
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.003,
        estimation_ratio=0.005,
        prefill_segment_size_tokens=4096,
        generation_update_interval=1024,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    key_cache = torch.randn(
        pages, 16, heads, head_size, device=device, dtype=torch.float16
    )
    value_cache = torch.randn_like(key_cache)
    index._records["layer"] = {
        "request": _NativeLayerRecord(
            indexed_end=(clusters + 1) * 16,
            keys=torch.randn(
                heads, clusters, head_size, device=device, dtype=torch.float16
            ),
            values=torch.randn(
                heads, clusters, head_size, device=device, dtype=torch.float16
            ),
            counts=torch.full((heads, clusters), 16, device=device, dtype=torch.int32),
            token_indices=torch.arange(
                16, (clusters + 1) * 16, device=device, dtype=torch.int32
            ).expand(heads, -1),
            cluster_offsets=(
                torch.arange(clusters + 1, device=device, dtype=torch.int32) * 16
            ).expand(heads, -1),
        )
    }
    block_table = torch.arange(pages, device=device, dtype=torch.int32)[None, :]
    query = torch.randn(1, 14, head_size, device=device, dtype=torch.float16)
    index.begin_proposal(("request",))
    try:
        output = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([pages * 16], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=head_size**-0.5,
            output=output,
            step=0,
        )
        torch.testing.assert_close(
            output,
            _reference_attention(
                index,
                "layer",
                query,
                key_cache,
                value_cache,
                0,
                False,
                block_table_row=block_table[0],
            ),
            atol=2e-2,
            rtol=2e-2,
        )
        sparse = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([pages * 16], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=head_size**-0.5,
            output=sparse,
            step=-1,
            sparse_verify=True,
            request_indices=torch.tensor([0], device=device, dtype=torch.int32),
            token_indices=torch.tensor([0], device=device, dtype=torch.int32),
        )
        torch.testing.assert_close(
            sparse,
            _reference_attention(
                index,
                "layer",
                query,
                key_cache,
                value_cache,
                0,
                False,
                sparse_verify=True,
                block_table_row=block_table[0],
            ),
            atol=2e-2,
            rtol=2e-2,
        )
        expanded = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([pages * 16], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=head_size**-0.5,
            output=expanded,
            step=-1,
            expanded=True,
            request_indices=torch.tensor([0], device=device, dtype=torch.int32),
            token_indices=torch.tensor([0], device=device, dtype=torch.int32),
        )
        torch.testing.assert_close(
            expanded,
            _reference_attention(
                index,
                "layer",
                query,
                key_cache,
                value_cache,
                0,
                True,
                block_table_row=block_table[0],
            ),
            atol=2e-2,
            rtol=2e-2,
        )
    finally:
        index.end_proposal()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_native_eight_partitions_match_reference() -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    clusters = 4096
    pages = clusters + 2
    head_size = 32
    index = RetroSpecGPUNativeIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.001,
        estimation_ratio=0.004,
        prefill_segment_size_tokens=8192,
        generation_update_interval=1024,
        blocks_per_cluster=1,
        num_kmeans_iterations=1,
    )
    key_cache = torch.randn(pages, 16, 1, head_size, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    index._records["layer"] = {
        "request": _NativeLayerRecord(
            indexed_end=(clusters + 1) * 16,
            keys=torch.randn(
                1, clusters, head_size, device=device, dtype=torch.float16
            ),
            values=torch.randn(
                1, clusters, head_size, device=device, dtype=torch.float16
            ),
            counts=torch.full((1, clusters), 16, device=device, dtype=torch.int32),
            token_indices=torch.arange(
                16, (clusters + 1) * 16, device=device, dtype=torch.int32
            )[None, :],
            cluster_offsets=(
                torch.arange(clusters + 1, device=device, dtype=torch.int32) * 16
            )[None, :],
        )
    }
    block_table = torch.arange(pages, device=device, dtype=torch.int32)[None, :]
    query = torch.randn(1, 2, head_size, device=device, dtype=torch.float16)
    index.begin_proposal(("request",))
    try:
        output = torch.empty_like(query)
        index.forward(
            layer_name="layer",
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=torch.tensor([pages * 16], device=device, dtype=torch.int32),
            active_mask=torch.ones(1, device=device, dtype=torch.bool),
            scale=head_size**-0.5,
            output=output,
            step=0,
        )
        torch.testing.assert_close(
            output,
            _reference_attention(
                index,
                "layer",
                query,
                key_cache,
                value_cache,
                0,
                False,
                block_table_row=block_table[0],
            ),
            atol=2e-2,
            rtol=2e-2,
        )
    finally:
        index.end_proposal()
