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
            Either [num_items, feature_size] or
            [num_groups, num_items, feature_size].

            The grouped form is used by token-level RetroSpec, where one
            group represents one KV head.
        segment_size:
            Number of items in one independently clustered segment.
        items_per_cluster:
            Target average number of items assigned to one cluster.
        num_iterations:
            Number of k-means iterations.

    Returns:
        assignments:
            Cluster ID for every item. The shape is [num_items] for an
            ungrouped input or [num_groups, num_items] for a grouped input.
            Cluster IDs are local to each group.
        cluster_sizes:
            Number of items assigned to each cluster. The shape is
            [num_clusters] or [num_groups, num_clusters].
    """
    if features.ndim not in (2, 3):
        raise ValueError(
            "features must have shape [num_items, feature_size] or "
            "[num_groups, num_items, feature_size]"
        )
    if segment_size <= 0:
        raise ValueError("segment_size must be greater than zero")
    if items_per_cluster <= 0:
        raise ValueError("items_per_cluster must be greater than zero")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be greater than zero")

    grouped_input = features.ndim == 3
    if not grouped_input:
        features = features.unsqueeze(0)

    num_groups, num_items, feature_size = features.shape

    if num_items == 0:
        assignments = torch.empty(
            num_groups,
            0,
            dtype=torch.int64,
            device=features.device,
        )
        cluster_sizes = torch.empty(
            num_groups,
            0,
            dtype=torch.int32,
            device=features.device,
        )
        if grouped_input:
            return assignments, cluster_sizes
        return assignments.squeeze(0), cluster_sizes.squeeze(0)

    if num_items % segment_size != 0:
        raise ValueError("num_items must be divisible by segment_size")
    if segment_size % items_per_cluster != 0:
        raise ValueError("segment_size must be divisible by items_per_cluster")

    num_segments = num_items // segment_size
    clusters_per_segment = segment_size // items_per_cluster

    data = features.float().reshape(
        num_groups * num_segments,
        segment_size,
        feature_size,
    )

    # Each temporal segment is centered independently. The assignments still
    # refer to the original, uncentered KV vectors.
    segment_means = data.mean(dim=1, keepdim=True)
    centered_data = data - segment_means

    initial_indices = (
        (
            torch.arange(
                clusters_per_segment,
                dtype=torch.float32,
                device=features.device,
            )
            + 0.5
        )
        * segment_size
        / clusters_per_segment
    ).to(torch.int64)
    initial_indices.clamp_(max=segment_size - 1)

    centroids = centered_data.index_select(1, initial_indices).contiguous()
    assignments = torch.zeros(
        num_groups * num_segments,
        segment_size,
        dtype=torch.int64,
        device=features.device,
    )

    for iteration in range(num_iterations):
        scores = torch.einsum("gsd,gcd->gsc", centered_data, centroids)
        assignments = scores.argmax(dim=-1)

        expanded_assignments = assignments.unsqueeze(-1).expand(
            -1,
            -1,
            feature_size,
        )

        centroid_sums = torch.zeros(
            num_groups * num_segments,
            clusters_per_segment,
            feature_size,
            dtype=torch.float32,
            device=features.device,
        )
        centroid_sums.scatter_add_(
            dim=1,
            index=expanded_assignments,
            src=centered_data,
        )

        cluster_sizes = torch.zeros(
            num_groups * num_segments,
            clusters_per_segment,
            dtype=torch.int32,
            device=features.device,
        )
        cluster_sizes.scatter_add_(
            dim=1,
            index=assignments,
            src=torch.ones_like(assignments, dtype=torch.int32),
        )

        updated_centroids = centroid_sums / cluster_sizes.clamp_min(1).unsqueeze(-1)
        centroids = torch.where(
            (cluster_sizes > 0).unsqueeze(-1),
            updated_centroids,
            centroids,
        )

        if iteration + 1 < num_iterations:
            centroids = F.normalize(centroids, dim=-1)

    assignments = assignments.view(
        num_groups,
        num_segments,
        segment_size,
    )
    cluster_sizes = cluster_sizes.view(
        num_groups,
        num_segments,
        clusters_per_segment,
    )

    segment_offsets = (
        torch.arange(
            num_segments,
            dtype=torch.int64,
            device=features.device,
        )
        * clusters_per_segment
    )
    assignments += segment_offsets[None, :, None]

    assignments = assignments.reshape(num_groups, num_items).contiguous()
    cluster_sizes = cluster_sizes.reshape(num_groups, -1).contiguous()

    if grouped_input:
        return assignments, cluster_sizes

    return assignments.squeeze(0), cluster_sizes.squeeze(0)
