# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.cluster_scoring import (
    reduce_grouped_cluster_scores,
)
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)


def reference_cluster_scores(
    logits: torch.Tensor,
    cluster_mask: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    valid_clusters = cluster_mask & (cluster_token_counts > 0)
    weighted_logits = logits.float() * scale + torch.log(
        cluster_token_counts.clamp_min(1).float()
    ).unsqueeze(2)
    weighted_logits.masked_fill_(~valid_clusters.unsqueeze(2), float("-inf"))

    has_clusters = valid_clusters.any(dim=2)
    safe_logits = torch.where(
        has_clusters[:, :, None, None],
        weighted_logits,
        torch.zeros_like(weighted_logits),
    )
    probabilities = torch.softmax(safe_logits, dim=3)
    probabilities.masked_fill_(~valid_clusters.unsqueeze(2), 0.0)
    return probabilities.mean(dim=2)


@pytest.mark.parametrize(
    ("num_clusters", "queries_per_kv", "count_dtype"),
    [
        (1, 1, torch.int32),
        (17, 4, torch.int64),
        (255, 2, torch.int32),
        (256, 4, torch.int64),
        (257, 8, torch.int32),
        (777, 4, torch.int32),
        (513, 32, torch.int32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cluster_scores_match_reference(
    num_clusters: int,
    queries_per_kv: int,
    count_dtype: torch.dtype,
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch_size = 3
    num_kv_heads = 2
    scale = 0.125

    # Slice padded tensors so every fused input exercises non-trivial strides.
    padded_logits = torch.randn(
        batch_size,
        num_kv_heads,
        queries_per_kv,
        num_clusters * 2,
        dtype=torch.float32,
        device=device,
    )
    logits = padded_logits[..., ::2]

    padded_mask = torch.rand(
        batch_size,
        num_kv_heads,
        num_clusters * 2,
        device=device,
    )
    cluster_mask = padded_mask[..., ::2] > 0.25
    cluster_mask[1, 1].zero_()

    padded_counts = torch.randint(
        0,
        65,
        (batch_size, num_kv_heads, num_clusters * 2),
        dtype=count_dtype,
        device=device,
    )
    cluster_token_counts = padded_counts[..., ::2]

    expected = reference_cluster_scores(
        logits,
        cluster_mask,
        cluster_token_counts,
        scale,
    )
    actual = reduce_grouped_cluster_scores(
        logits,
        cluster_mask,
        cluster_token_counts,
        scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[1, 1]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tensor_core_cluster_logits_match_float_reference(dtype: torch.dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    batch_size = 2
    num_kv_heads = 2
    queries_per_kv = 4
    num_clusters = 37
    head_size = 64

    query = torch.randn(
        batch_size,
        num_kv_heads * queries_per_kv,
        head_size,
        dtype=dtype,
        device=device,
    )
    cluster_keys = torch.randn(
        batch_size,
        num_kv_heads,
        num_clusters,
        head_size,
        dtype=dtype,
        device=device,
    )

    actual = RetroSpecSegmentedTokenIndex._compute_cluster_logits(
        query,
        cluster_keys,
    )
    expected = torch.einsum(
        "bhgd,bhcd->bhgc",
        query.float().view(
            batch_size,
            num_kv_heads,
            queries_per_kv,
            head_size,
        ),
        cluster_keys.float(),
    )
    torch.cuda.synchronize()

    tolerance = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.dtype == torch.float32
    assert actual.shape == (
        batch_size,
        num_kv_heads,
        queries_per_kv,
        num_clusters,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cluster_scores_handle_empty_batch():
    logits = torch.empty(0, 2, 4, 17, device="cuda")
    cluster_mask = torch.empty(0, 2, 17, dtype=torch.bool, device="cuda")
    cluster_token_counts = torch.empty(0, 2, 17, dtype=torch.int32, device="cuda")

    output = reduce_grouped_cluster_scores(
        logits,
        cluster_mask,
        cluster_token_counts,
        scale=0.125,
    )

    assert output.shape == (0, 2, 17)
    assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cluster_scores_reuse_outputs_and_write_ranking_scores():
    torch.manual_seed(23)
    device = torch.device("cuda")
    logits = torch.randn(2, 2, 4, 19, device=device)
    cluster_mask = torch.rand(2, 2, 19, device=device) > 0.25
    cluster_mask[..., 0] = True
    cluster_token_counts = torch.randint(
        1,
        33,
        (2, 2, 19),
        dtype=torch.int32,
        device=device,
    )
    cluster_token_counts[..., -1] = 0

    output = torch.empty(2, 2, 19, device=device)
    softmax_lse = torch.empty(2, 2, 4, device=device)
    ranking_output = torch.empty(2, 2, 19, device=device)
    data_ptrs = (
        output.data_ptr(),
        softmax_lse.data_ptr(),
        ranking_output.data_ptr(),
    )

    expected = reference_cluster_scores(
        logits,
        cluster_mask,
        cluster_token_counts,
        scale=0.125,
    )
    actual = reduce_grouped_cluster_scores(
        logits,
        cluster_mask,
        cluster_token_counts,
        scale=0.125,
        output=output,
        softmax_lse=softmax_lse,
        ranking_output=ranking_output,
    )
    torch.cuda.synchronize()

    valid_clusters = cluster_mask & (cluster_token_counts > 0)
    assert actual is output
    assert data_ptrs == (
        output.data_ptr(),
        softmax_lse.data_ptr(),
        ranking_output.data_ptr(),
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        ranking_output[valid_clusters],
        expected[valid_clusters],
        atol=2e-6,
        rtol=2e-5,
    )
    assert torch.isneginf(ranking_output[~valid_clusters]).all()
    assert torch.isfinite(softmax_lse).all()

    second_logits = torch.randn_like(logits)
    second_expected = reference_cluster_scores(
        second_logits,
        cluster_mask,
        cluster_token_counts,
        scale=0.125,
    )
    second_actual = reduce_grouped_cluster_scores(
        second_logits,
        cluster_mask,
        cluster_token_counts,
        scale=0.125,
        output=output,
        softmax_lse=softmax_lse,
        ranking_output=ranking_output,
    )
    torch.cuda.synchronize()

    assert second_actual is output
    assert data_ptrs == (
        output.data_ptr(),
        softmax_lse.data_ptr(),
        ranking_output.data_ptr(),
    )
    torch.testing.assert_close(second_actual, second_expected, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cluster_scores_reject_overlapping_outputs():
    logits = torch.randn(1, 2, 4, 17, device="cuda")
    cluster_mask = torch.ones(1, 2, 17, dtype=torch.bool, device="cuda")
    cluster_token_counts = torch.ones(1, 2, 17, dtype=torch.int32, device="cuda")
    shared_output = torch.empty(1, 2, 17, device="cuda")

    with pytest.raises(ValueError, match="must not overlap"):
        reduce_grouped_cluster_scores(
            logits,
            cluster_mask,
            cluster_token_counts,
            scale=0.125,
            output=shared_output,
            ranking_output=shared_output,
        )

    single_query_logits = torch.randn(1, 2, 1, 17, device="cuda")
    with pytest.raises(ValueError, match="logits and output must not overlap"):
        reduce_grouped_cluster_scores(
            single_query_logits,
            cluster_mask,
            cluster_token_counts,
            scale=0.125,
            output=single_query_logits[:, :, 0],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda logits, mask, counts: (logits.cpu(), mask, counts), "CUDA"),
        (
            lambda logits, mask, counts: (logits.half(), mask, counts),
            "float32",
        ),
        (
            lambda logits, mask, counts: (logits, mask.float(), counts),
            "boolean",
        ),
        (
            lambda logits, mask, counts: (logits, mask, counts.float()),
            "integral",
        ),
        (
            lambda logits, mask, counts: (logits, mask[..., :-1], counts),
            "mask shape",
        ),
        (
            lambda logits, mask, counts: (logits, mask, counts[..., :-1]),
            "counts shape",
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cluster_scores_validate_inputs(mutation, message: str):
    logits = torch.randn(1, 2, 4, 17, device="cuda")
    cluster_mask = torch.ones(1, 2, 17, dtype=torch.bool, device="cuda")
    cluster_token_counts = torch.ones(1, 2, 17, dtype=torch.int32, device="cuda")
    logits, cluster_mask, cluster_token_counts = mutation(
        logits,
        cluster_mask,
        cluster_token_counts,
    )

    with pytest.raises(ValueError, match=message):
        reduce_grouped_cluster_scores(
            logits,
            cluster_mask,
            cluster_token_counts,
            scale=0.125,
        )


def test_cluster_logits_float32_cpu_fallback_matches_reference():
    torch.manual_seed(19)
    query = torch.randn(2, 6, 8)
    cluster_keys = torch.randn(2, 2, 11, 8)

    actual = RetroSpecSegmentedTokenIndex._compute_cluster_logits(
        query,
        cluster_keys,
    )
    expected = torch.einsum(
        "bhgd,bhcd->bhgc",
        query.view(2, 2, 3, 8),
        cluster_keys,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tensor_core_cluster_logits_reuse_output():
    torch.manual_seed(29)
    query = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
    cluster_keys = torch.randn(
        2,
        2,
        31,
        64,
        dtype=torch.float16,
        device="cuda",
    )
    output = torch.empty(2, 2, 4, 31, device="cuda")

    actual = RetroSpecSegmentedTokenIndex._compute_cluster_logits(
        query,
        cluster_keys,
        output,
    )
    expected = torch.einsum(
        "bhgd,bhcd->bhgc",
        query.float().view(2, 2, 4, 64),
        cluster_keys.float(),
    )
    torch.cuda.synchronize()

    assert actual is output
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cluster_selection_workspace_is_reused_and_resized():
    index = RetroSpecSegmentedTokenIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.25,
        estimation_ratio=0.25,
        segment_size_tokens=256,
        blocks_per_cluster=2,
        num_kmeans_iterations=2,
    )
    query = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
    cluster_keys = torch.randn(
        2,
        2,
        23,
        64,
        dtype=torch.float16,
        device="cuda",
    )

    first = index._get_cluster_selection_workspace(query, cluster_keys)
    second = index._get_cluster_selection_workspace(query, cluster_keys)

    assert second is first
    assert first.logits.shape == (2, 2, 4, 23)
    assert first.scores.shape == (2, 2, 23)
    assert first.softmax_lse.shape == (2, 2, 4)
    assert first.topk_indices.shape == (2, 2, 12)

    larger_cluster_keys = torch.randn(
        2,
        2,
        29,
        64,
        dtype=torch.float16,
        device="cuda",
    )
    resized = index._get_cluster_selection_workspace(
        query,
        larger_cluster_keys,
    )

    assert resized is not first
    assert resized.logits.shape == (2, 2, 4, 29)
    assert resized.scores.shape == (2, 2, 29)
    assert resized.topk_indices.shape == (2, 2, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_workspace_cluster_selection_matches_allocating_path():
    torch.manual_seed(31)
    index = RetroSpecSegmentedTokenIndex(
        block_size=16,
        num_speculative_tokens=4,
        retrieval_ratio=0.3,
        estimation_ratio=0.4,
        segment_size_tokens=256,
        blocks_per_cluster=2,
        num_kmeans_iterations=2,
    )
    query = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
    cluster_keys = torch.randn(
        2,
        2,
        37,
        64,
        dtype=torch.float16,
        device="cuda",
    )
    cluster_mask = torch.rand(2, 2, 37, device="cuda") > 0.3
    cluster_mask[..., 0] = True
    cluster_token_counts = torch.randint(
        1,
        17,
        (2, 2, 37),
        dtype=torch.int32,
        device="cuda",
    )
    cluster_token_counts.masked_fill_(~cluster_mask, 0)

    workspace = index._get_cluster_selection_workspace(query, cluster_keys)
    workspace_scores = index._score_clusters(
        query,
        cluster_keys,
        cluster_mask,
        cluster_token_counts,
        scale=0.125,
        workspace=workspace,
    )
    workspace_zones = index._select_cluster_zones(
        workspace_scores,
        cluster_mask,
        workspace,
    )

    reference_scores = index._score_clusters(
        query,
        cluster_keys,
        cluster_mask,
        cluster_token_counts,
        scale=0.125,
    )
    reference_zones = index._select_cluster_zones(
        reference_scores,
        cluster_mask,
    )
    torch.cuda.synchronize()

    assert workspace_scores is workspace.scores
    torch.testing.assert_close(
        workspace_scores,
        reference_scores,
        atol=2e-6,
        rtol=2e-5,
    )
    for field_name in workspace_zones.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(workspace_zones, field_name),
            getattr(reference_zones, field_name),
        )

    with pytest.raises(ValueError, match="do not belong"):
        index._select_cluster_zones(
            workspace_scores.clone(),
            cluster_mask,
            workspace,
        )
