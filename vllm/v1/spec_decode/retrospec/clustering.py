# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F


def segmented_kmeans_assignments(
    features: torch.Tensor,
    segment_size: int,
    items_per_cluster: int,
    num_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster fixed-size segments independently.

    Args:
        features:
            Tensor with shape [num_items, feature_size]. One item represents
            one complete vLLM KV block.
        segment_size:
            Number of blocks in one independently clustered segment.
        items_per_cluster:
            Target average number of blocks assigned to one cluster.
        num_iterations:
            Number of k-means iterations.

    Returns:
        assignments:
            Global cluster ID for every input block, shape [num_items].
        cluster_sizes:
            Number of blocks assigned to each cluster.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [num_items, feature_size]")
    if segment_size <= 0:
        raise ValueError("segment_size must be greater than zero")
    if items_per_cluster <= 0:
        raise ValueError("items_per_cluster must be greater than zero")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be greater than zero")

    num_items, feature_size = features.shape
    if num_items == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=features.device),
            torch.empty(0, dtype=torch.int32, device=features.device),
        )
    if num_items % segment_size != 0:
        raise ValueError("num_items must be divisible by segment_size")
    if segment_size % items_per_cluster != 0:
        raise ValueError("segment_size must be divisible by items_per_cluster")

    num_segments = num_items // segment_size
    clusters_per_segment = segment_size // items_per_cluster

    data = features.float().view(num_segments, segment_size, feature_size)

    # RetroInfer centers every segment before clustering. This reduces the
    # influence of segment-level key offsets on the inner-product assignment.
    segment_means = data.mean(dim=1, keepdim=True)
    centered_data = data - segment_means

    initial_indices = (
        (
            torch.arange(
                clusters_per_segment, dtype=torch.float32, device=features.device
            )
            + 0.5
        )
        * segment_size
        / clusters_per_segment
    ).to(torch.int64)
    initial_indices.clamp_(max=segment_size - 1)

    centroids = centered_data.index_select(1, initial_indices).contiguous()
    assignments = torch.zeros(
        num_segments, segment_size, dtype=torch.int64, device=features.device
    )

    for iteration in range(num_iterations):
        scores = torch.einsum("snd,skd->snk", centered_data, centroids)
        assignments = scores.argmax(dim=-1)

        expanded_assignments = assignments.unsqueeze(-1).expand(-1, -1, feature_size)

        centroid_sums = torch.zeros(
            num_segments,
            clusters_per_segment,
            feature_size,
            dtype=torch.float32,
            device=features.device,
        )
        centroid_sums.scatter_add_(dim=1, index=expanded_assignments, src=centered_data)

        cluster_sizes = torch.zeros(
            num_segments,
            clusters_per_segment,
            dtype=torch.int32,
            device=features.device,
        )
        cluster_sizes.scatter_add_(
            dim=1,
            index=assignments,
            src=torch.ones_like(assignments, dtype=torch.int32),
        )

        nonempty = cluster_sizes > 0
        updated_centroids = centroid_sums / cluster_sizes.clamp_min(1).unsqueeze(-1)

        # Keep the previous centroid when a cluster is temporarily empty.
        centroids = torch.where(nonempty.unsqueeze(-1), updated_centroids, centroids)

        # RetroInfer normalizes centroids during intermediate iterations, then
        # keeps the final centroid as the actual arithmetic mean.
        if iteration + 1 < num_iterations:
            centroids = F.normalize(centroids, dim=-1)

    cluster_offsets = (
        torch.arange(
            num_segments,
            dtype=torch.int64,
            device=features.device,
        )
        * clusters_per_segment
    )
    assignments = assignments + cluster_offsets.unsqueeze(1)

    return (
        assignments.reshape(-1).contiguous(),
        cluster_sizes.reshape(-1).contiguous(),
    )
