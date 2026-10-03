# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.clustering import (
    SegmentedKMeansResult,
    segmented_kmeans,
    segmented_kmeans_assignments,
)


def assert_fused_result_consistent(
    result: SegmentedKMeansResult,
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
) -> None:
    num_heads, num_tokens, head_size = token_keys.shape
    num_clusters = result.cluster_sizes.shape[1]

    assert result.assignments.shape == (num_heads, num_tokens)
    assert result.cluster_keys.shape == (num_heads, num_clusters, head_size)
    assert result.cluster_values.shape == (num_heads, num_clusters, head_size)
    assert result.token_offsets_in_cluster.shape == (num_heads, num_tokens)
    assert result.assignments.dtype == torch.int32
    assert result.cluster_sizes.dtype == torch.int32
    assert result.token_offsets_in_cluster.dtype == torch.int32

    for head_idx in range(num_heads):
        actual_counts = torch.bincount(
            result.assignments[head_idx].to(torch.int64),
            minlength=num_clusters,
        ).to(torch.int32)
        torch.testing.assert_close(actual_counts, result.cluster_sizes[head_idx])

        for cluster_idx in range(num_clusters):
            cluster_mask = result.assignments[head_idx] == cluster_idx
            cluster_size = int(result.cluster_sizes[head_idx, cluster_idx].item())
            cluster_offsets = result.token_offsets_in_cluster[head_idx, cluster_mask]
            expected_offsets = torch.arange(
                cluster_size,
                dtype=torch.int32,
                device=token_keys.device,
            )
            torch.testing.assert_close(
                torch.sort(cluster_offsets).values, expected_offsets
            )

            if cluster_size == 0:
                torch.testing.assert_close(
                    result.cluster_keys[head_idx, cluster_idx],
                    torch.zeros(head_size, device=token_keys.device),
                )
                torch.testing.assert_close(
                    result.cluster_values[head_idx, cluster_idx],
                    torch.zeros(head_size, device=token_keys.device),
                )
                continue

            expected_keys = token_keys[head_idx, cluster_mask].float().mean(dim=0)
            expected_values = token_values[head_idx, cluster_mask].float().mean(dim=0)
            torch.testing.assert_close(
                result.cluster_keys[head_idx, cluster_idx].float(),
                expected_keys,
                atol=2e-3,
                rtol=2e-3,
            )
            torch.testing.assert_close(
                result.cluster_values[head_idx, cluster_idx].float(),
                expected_values,
                atol=2e-3,
                rtol=2e-3,
            )


def test_segmented_kmeans_keeps_cluster_ids_local_to_each_segment():
    features = torch.zeros(8, 2)

    assignments, cluster_sizes = segmented_kmeans_assignments(
        features,
        segment_size=4,
        items_per_cluster=2,
        num_iterations=2,
    )

    assert assignments[:4].tolist() == [0, 0, 0, 0]
    assert assignments[4:].tolist() == [2, 2, 2, 2]
    assert cluster_sizes.tolist() == [4, 0, 4, 0]
    assert torch.equal(
        torch.bincount(assignments, minlength=4).to(torch.int32),
        cluster_sizes,
    )


def test_segmented_kmeans_returns_empty_tensors_for_empty_input():
    assignments, cluster_sizes = segmented_kmeans_assignments(
        torch.empty(0, 3),
        segment_size=4,
        items_per_cluster=2,
        num_iterations=1,
    )

    assert assignments.shape == (0,)
    assert cluster_sizes.shape == (0,)
    assert assignments.dtype == torch.int64
    assert cluster_sizes.dtype == torch.int32


def test_segmented_kmeans_clusters_groups_independently():
    features = torch.tensor(
        [
            [[1.0], [1.0], [-1.0], [-1.0]],
            [[1.0], [-1.0], [1.0], [-1.0]],
        ]
    )

    assignments, cluster_sizes = segmented_kmeans_assignments(
        features,
        segment_size=4,
        items_per_cluster=2,
        num_iterations=2,
    )

    assert assignments.shape == (2, 4)
    assert cluster_sizes.shape == (2, 2)
    assert assignments[0].tolist() == [0, 0, 1, 1]
    assert assignments[1].tolist() == [0, 1, 0, 1]
    assert cluster_sizes.tolist() == [[2, 2], [2, 2]]

    for group in range(2):
        assert torch.equal(
            torch.bincount(assignments[group], minlength=2).to(torch.int32),
            cluster_sizes[group],
        )


def test_segmented_kmeans_returns_grouped_empty_tensors():
    assignments, cluster_sizes = segmented_kmeans_assignments(
        torch.empty(3, 0, 2),
        segment_size=4,
        items_per_cluster=2,
        num_iterations=1,
    )

    assert assignments.shape == (3, 0)
    assert cluster_sizes.shape == (3, 0)


def test_fused_segmented_kmeans_returns_summaries_and_page_offsets_on_cpu():
    token_keys = torch.tensor(
        [
            [[1.0], [1.0], [-1.0], [-1.0]],
            [[1.0], [-1.0], [1.0], [-1.0]],
        ]
    )
    token_values = token_keys + 10.0

    result = segmented_kmeans(
        token_keys,
        token_values,
        segment_size=4,
        items_per_cluster=2,
        num_iterations=2,
    )

    assert result.assignments.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert result.token_offsets_in_cluster.tolist() == [[0, 1, 0, 1], [0, 0, 1, 1]]
    assert_fused_result_consistent(result, token_keys, token_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_segmented_kmeans_cuda_metadata_and_summaries_are_consistent(dtype):
    generator = torch.Generator(device="cuda").manual_seed(7)
    token_keys = torch.randn(
        2,
        16,
        64,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    token_values = torch.randn(
        2,
        16,
        64,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )

    result = segmented_kmeans(
        token_keys,
        token_values,
        segment_size=8,
        items_per_cluster=4,
        num_iterations=3,
    )

    assert result.assignments[:, :8].max().item() < 2
    assert result.assignments[:, 8:].min().item() >= 2
    assert_fused_result_consistent(result, token_keys, token_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_segmented_kmeans_cuda_supports_default_prefill_geometry():
    token_keys = torch.randn(
        1,
        8192,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    token_values = torch.randn_like(token_keys)

    result = segmented_kmeans(
        token_keys,
        token_values,
        segment_size=8192,
        items_per_cluster=16,
        num_iterations=2,
    )

    assert result.cluster_sizes.shape == (1, 512)
    assert result.cluster_sizes.sum().item() == 8192
    assert result.assignments.min().item() >= 0
    assert result.assignments.max().item() < 512
    assigned_counts = torch.gather(
        result.cluster_sizes,
        dim=1,
        index=result.assignments.to(torch.int64),
    )
    assert torch.all(result.token_offsets_in_cluster >= 0)
    assert torch.all(result.token_offsets_in_cluster < assigned_counts)
    assert torch.isfinite(result.cluster_keys).all()
    assert torch.isfinite(result.cluster_values).all()


def test_fused_segmented_kmeans_returns_grouped_empty_outputs():
    result = segmented_kmeans(
        torch.empty(3, 0, 2),
        torch.empty(3, 0, 2),
        segment_size=4,
        items_per_cluster=2,
        num_iterations=1,
    )

    assert result.assignments.shape == (3, 0)
    assert result.cluster_sizes.shape == (3, 0)
    assert result.cluster_keys.shape == (3, 0, 2)
    assert result.cluster_values.shape == (3, 0, 2)
    assert result.token_offsets_in_cluster.shape == (3, 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"segment_size": 0}, "segment_size"),
        ({"items_per_cluster": 0}, "items_per_cluster"),
        ({"num_iterations": 0}, "num_iterations"),
        ({"segment_size": 3}, "num_items"),
        ({"items_per_cluster": 3}, "segment_size"),
    ],
)
def test_segmented_kmeans_rejects_invalid_shapes_and_configuration(
    kwargs,
    message,
):
    values = {
        "segment_size": 4,
        "items_per_cluster": 2,
        "num_iterations": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        segmented_kmeans_assignments(torch.zeros(4, 2), **values)
