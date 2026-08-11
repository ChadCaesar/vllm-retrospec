# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.clustering import (
    segmented_kmeans_assignments,
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
