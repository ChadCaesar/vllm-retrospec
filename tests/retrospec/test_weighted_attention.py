# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.attention import RetroSpecSparseAttention
from vllm.v1.spec_decode.retrospec.weighted_attention import (
    merge_weighted_estimation,
)


def _merge_reference(
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
    estimation_output: torch.Tensor,
    estimation_lse: torch.Tensor,
) -> torch.Tensor:
    exact_lse = torch.where(
        torch.isfinite(exact_lse),
        exact_lse,
        torch.full_like(exact_lse, float("-inf")),
    )
    estimation_lse = torch.where(
        torch.isfinite(estimation_lse),
        estimation_lse,
        torch.full_like(estimation_lse, float("-inf")),
    )
    maximum = torch.maximum(exact_lse, estimation_lse)
    safe_maximum = torch.where(
        torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
    )
    exact_weight = torch.exp(exact_lse - safe_maximum)
    estimation_weight = torch.exp(estimation_lse - safe_maximum)
    normalizer = exact_weight + estimation_weight

    exact_weight = exact_weight.transpose(0, 1).unsqueeze(-1)
    estimation_weight = estimation_weight.transpose(0, 1).unsqueeze(-1)
    normalizer = normalizer.transpose(0, 1).unsqueeze(-1)

    numerator = (
        exact_output.float() * exact_weight
        + estimation_output.float() * estimation_weight
    )
    return torch.where(
        normalizer > 0,
        numerator / normalizer.clamp_min(1.0e-20),
        torch.zeros_like(numerator),
    ).to(exact_output.dtype)


def _run_reference(
    query: torch.Tensor,
    estimation_keys: torch.Tensor,
    estimation_values: torch.Tensor,
    estimation_token_counts: torch.Tensor,
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    impl = type("Impl", (), {"scale": scale})()
    estimation_output, estimation_lse = (
        RetroSpecSparseAttention._run_grouped_reference_attention(
            impl,
            query,
            estimation_keys,
            estimation_values,
            estimation_token_counts,
        )
    )
    return _merge_reference(
        exact_output,
        exact_lse,
        estimation_output,
        estimation_lse,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("query_dtype", "estimation_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_weighted_estimation_matches_reference_across_tiles(
    query_dtype, estimation_dtype
):
    device = torch.device("cuda")
    torch.manual_seed(11)
    batch_size = 2
    num_query_heads = 8
    num_kv_heads = 2
    num_vectors = 35
    head_size = 64
    scale = head_size**-0.5

    query = torch.randn(
        batch_size,
        num_query_heads,
        head_size,
        device=device,
        dtype=query_dtype,
    )
    estimation_keys = torch.randn(
        batch_size,
        num_kv_heads,
        num_vectors,
        head_size,
        device=device,
        dtype=estimation_dtype,
    )
    estimation_values = torch.randn_like(estimation_keys)
    estimation_token_counts = torch.randint(
        0,
        9,
        (batch_size, num_kv_heads, num_vectors),
        dtype=torch.int32,
        device=device,
    )
    estimation_token_counts[0, 1].zero_()

    exact_output = torch.randn_like(query)
    exact_lse_storage = torch.randn(
        batch_size,
        num_query_heads,
        dtype=torch.float32,
        device=device,
    )
    exact_lse = exact_lse_storage.transpose(0, 1)
    exact_lse[2:6, 0] = float("-inf")
    output = torch.empty_like(query)

    expected = _run_reference(
        query,
        estimation_keys,
        estimation_values,
        estimation_token_counts,
        exact_output,
        exact_lse,
        scale,
    )
    merge_weighted_estimation(
        output,
        query,
        estimation_keys,
        estimation_values,
        estimation_token_counts,
        exact_output,
        exact_lse,
        scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_weighted_estimation_handles_empty_exact_state():
    device = torch.device("cuda")
    query = torch.zeros(1, 4, 64, dtype=torch.bfloat16, device=device)
    keys = torch.zeros(1, 2, 2, 64, dtype=torch.bfloat16, device=device)
    values = torch.zeros_like(keys)
    values[0, 0, 0].fill_(2.0)
    values[0, 0, 1].fill_(4.0)
    values[0, 1, 0].fill_(10.0)
    values[0, 1, 1].fill_(20.0)
    counts = torch.tensor([[[1, 3], [3, 1]]], dtype=torch.int32, device=device)
    exact_output = torch.full_like(query, 99.0)
    exact_lse = torch.full((4, 1), float("-inf"), dtype=torch.float32, device=device)
    output = torch.empty_like(query)

    merge_weighted_estimation(
        output,
        query,
        keys,
        values,
        counts,
        exact_output,
        exact_lse,
        1.0,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output[0, :, 0].float(),
        torch.tensor([3.5, 3.5, 12.5, 12.5], device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_weighted_estimation_handles_no_valid_centroids():
    device = torch.device("cuda")
    torch.manual_seed(19)
    query = torch.randn(2, 4, 64, dtype=torch.bfloat16, device=device)
    keys = torch.randn(2, 2, 17, 64, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    counts = torch.zeros(2, 2, 17, dtype=torch.int32, device=device)
    exact_output = torch.randn_like(query)
    exact_lse = torch.randn(4, 2, dtype=torch.float32, device=device)
    exact_lse[2:, 1] = float("-inf")
    exact_output[1, 2:].zero_()
    output = torch.empty_like(query)

    merge_weighted_estimation(
        output,
        query,
        keys,
        values,
        counts,
        exact_output,
        exact_lse,
        1.0,
    )
    torch.cuda.synchronize()

    expected = exact_output.clone()
    expected[1, 2:].zero_()
    torch.testing.assert_close(output, expected)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_weighted_estimation_copies_exact_output_for_empty_layout():
    device = torch.device("cuda")
    query = torch.randn(1, 4, 64, dtype=torch.float16, device=device)
    exact_output = torch.randn_like(query)
    output = torch.empty_like(query)

    merge_weighted_estimation(
        output,
        query,
        torch.empty(1, 2, 0, 64, dtype=query.dtype, device=device),
        torch.empty(1, 2, 0, 64, dtype=query.dtype, device=device),
        torch.empty(1, 2, 0, dtype=torch.int32, device=device),
        exact_output,
        torch.randn(4, 1, dtype=torch.float32, device=device),
        1.0,
    )

    torch.testing.assert_close(output, exact_output)
