# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .clustering_kernels import segmented_kmeans_cuda


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


@dataclass(frozen=True)
class SegmentedKMeansResult:
    assignments: torch.Tensor
    cluster_sizes: torch.Tensor
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    token_offsets_in_cluster: torch.Tensor


def _cluster_means_reference(
    vectors: torch.Tensor,
    assignments: torch.Tensor,
    cluster_sizes: torch.Tensor,
) -> torch.Tensor:
    num_heads, _, head_size = vectors.shape
    num_clusters = cluster_sizes.shape[1]
    assignment_indices = assignments.to(torch.int64)

    expanded_assignments = assignment_indices.unsqueeze(-1).expand(-1, -1, head_size)
    cluster_sums = torch.zeros(
        num_heads,
        num_clusters,
        head_size,
        dtype=torch.float32,
        device=vectors.device,
    )
    cluster_sums.scatter_add_(
        dim=1,
        index=expanded_assignments,
        src=vectors.float(),
    )

    denominator = cluster_sizes.clamp_min(1).to(torch.float32).unsqueeze(-1)
    return (cluster_sums / denominator).to(vectors.dtype)


def _token_offsets_reference(
    assignments: torch.Tensor,
    cluster_sizes: torch.Tensor,
) -> torch.Tensor:
    num_heads, num_tokens = assignments.shape

    if num_tokens == 0:
        return torch.empty_like(assignments, dtype=torch.int32)

    assignment_indices = assignments.to(torch.int64)
    sorted_token_indices = torch.argsort(
        assignment_indices,
        dim=1,
        stable=True,
    )
    sorted_assignments = torch.gather(
        assignment_indices,
        dim=1,
        index=sorted_token_indices,
    )

    cluster_sizes_int64 = cluster_sizes.to(torch.int64)
    cluster_starts = torch.cumsum(cluster_sizes_int64, dim=1) - cluster_sizes_int64
    sorted_cluster_starts = torch.gather(
        cluster_starts,
        dim=1,
        index=sorted_assignments,
    )
    sorted_positions = torch.arange(
        num_tokens,
        dtype=torch.int64,
        device=assignments.device,
    ).view(1, num_tokens)
    sorted_offsets = sorted_positions - sorted_cluster_starts

    token_offsets = torch.empty(
        num_heads,
        num_tokens,
        dtype=torch.int64,
        device=assignments.device,
    )
    token_offsets.scatter_(
        dim=1,
        index=sorted_token_indices,
        src=sorted_offsets.expand(num_heads, -1),
    )
    return token_offsets.to(torch.int32)


def segmented_kmeans(
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
    segment_size: int,
    items_per_cluster: int,
    num_iterations: int,
) -> SegmentedKMeansResult:
    if token_keys.ndim != 3:
        raise ValueError(
            "token_keys must have shape [num_kv_heads, num_tokens, head_size]"
        )
    if token_values.shape != token_keys.shape:
        raise ValueError("token_keys and token_values must have equal shapes")
    if token_values.dtype != token_keys.dtype:
        raise ValueError("token_keys and token_values must have equal dtypes")
    if token_values.device != token_keys.device:
        raise ValueError("token_keys and token_values must be on one device")
    if segment_size <= 0:
        raise ValueError("segment_size must be greater than zero")
    if items_per_cluster <= 0:
        raise ValueError("items_per_cluster must be greater than zero")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be greater than zero")

    num_heads, num_tokens, head_size = token_keys.shape
    if num_tokens == 0:
        empty_assignments = torch.empty(
            num_heads,
            0,
            dtype=torch.int32,
            device=token_keys.device,
        )
        empty_summaries = token_keys.new_empty(num_heads, 0, head_size)
        return SegmentedKMeansResult(
            assignments=empty_assignments,
            cluster_sizes=empty_assignments.clone(),
            cluster_keys=empty_summaries,
            cluster_values=empty_summaries.clone(),
            token_offsets_in_cluster=empty_assignments.clone(),
        )
    if num_tokens % segment_size != 0:
        raise ValueError("num_tokens must be divisible by segment_size")
    if segment_size % items_per_cluster != 0:
        raise ValueError("segment_size must be divisible by items_per_cluster")

    use_triton = (
        token_keys.device.type == "cuda"
        and token_keys.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and head_size <= 256
    )

    if use_triton:
        (
            assignments,
            cluster_sizes,
            cluster_keys,
            cluster_values,
            token_offsets,
        ) = segmented_kmeans_cuda(
            token_keys=token_keys,
            token_values=token_values,
            segment_size=segment_size,
            items_per_cluster=items_per_cluster,
            num_iterations=num_iterations,
        )
        return SegmentedKMeansResult(
            assignments=assignments,
            cluster_sizes=cluster_sizes,
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            token_offsets_in_cluster=token_offsets,
        )

    assignments, cluster_sizes = segmented_kmeans_assignments(
        features=token_keys,
        segment_size=segment_size,
        items_per_cluster=items_per_cluster,
        num_iterations=num_iterations,
    )
    assignments = assignments.to(torch.int32)

    return SegmentedKMeansResult(
        assignments=assignments,
        cluster_sizes=cluster_sizes,
        cluster_keys=_cluster_means_reference(
            token_keys,
            assignments,
            cluster_sizes,
        ),
        cluster_values=_cluster_means_reference(
            token_values,
            assignments,
            cluster_sizes,
        ),
        token_offsets_in_cluster=_token_offsets_reference(
            assignments,
            cluster_sizes,
        ),
    )
