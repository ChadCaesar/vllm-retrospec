# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _assign_clusters_kernel(
    data,
    centroids,
    cluster_sums,
    cluster_counts,
    num_items,
    num_centroids,
    data_stride_g,
    data_stride_n,
    data_stride_d,
    centroid_stride_g,
    centroid_stride_c,
    centroid_stride_d,
    sum_stride_g,
    sum_stride_c,
    sum_stride_d,
    count_stride_g,
    count_stride_c,
    HEAD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    item_start = tl.program_id(0) * BLOCK_N
    group_idx = tl.program_id(1)

    item_offsets = item_start + tl.arange(0, BLOCK_N)
    cluster_offsets = tl.arange(0, BLOCK_C)
    dim_offsets = tl.arange(0, BLOCK_D)
    item_mask = item_offsets < num_items
    dim_mask = dim_offsets < HEAD_SIZE

    data_offsets = (
        group_idx * data_stride_g
        + item_offsets[:, None] * data_stride_n
        + dim_offsets[None, :] * data_stride_d
    )
    data_block = tl.load(
        data + data_offsets,
        mask=item_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    best_scores = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
    best_clusters = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for cluster_start in tl.range(0, num_centroids, BLOCK_C):
        current_clusters = cluster_start + cluster_offsets
        cluster_mask = current_clusters < num_centroids
        centroid_offsets = (
            group_idx * centroid_stride_g
            + current_clusters[None, :] * centroid_stride_c
            + dim_offsets[:, None] * centroid_stride_d
        )
        centroid_block = tl.load(
            centroids + centroid_offsets,
            mask=dim_mask[:, None] & cluster_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(data_block, centroid_block).to(tl.float32)
        scores = tl.where(cluster_mask[None, :], scores, float("-inf"))
        block_scores, block_clusters = tl.max(scores, axis=1, return_indices=True)
        block_clusters += cluster_start

        replace = block_scores > best_scores
        best_scores = tl.where(replace, block_scores, best_scores)
        best_clusters = tl.where(replace, block_clusters, best_clusters)

    sum_offsets = (
        group_idx * sum_stride_g
        + best_clusters[:, None] * sum_stride_c
        + dim_offsets[None, :] * sum_stride_d
    )
    tl.atomic_add(
        cluster_sums + sum_offsets,
        data_block.to(tl.float32),
        mask=item_mask[:, None] & dim_mask[None, :],
        sem="relaxed",
    )

    count_offsets = group_idx * count_stride_g + best_clusters * count_stride_c
    tl.atomic_add(
        cluster_counts + count_offsets,
        1,
        mask=item_mask,
        sem="relaxed",
    )


@triton.jit
def _update_centroids_kernel(
    centroids,
    cluster_sums,
    cluster_counts,
    num_centroids,
    centroid_stride_g,
    centroid_stride_c,
    centroid_stride_d,
    sum_stride_g,
    sum_stride_c,
    sum_stride_d,
    count_stride_g,
    count_stride_c,
    HEAD_SIZE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cluster_start = tl.program_id(0) * BLOCK_C
    group_idx = tl.program_id(1)

    cluster_offsets = cluster_start + tl.arange(0, BLOCK_C)
    dim_offsets = tl.arange(0, BLOCK_D)
    cluster_mask = cluster_offsets < num_centroids
    dim_mask = dim_offsets < HEAD_SIZE
    vector_mask = cluster_mask[:, None] & dim_mask[None, :]

    centroid_offsets = (
        group_idx * centroid_stride_g
        + cluster_offsets[:, None] * centroid_stride_c
        + dim_offsets[None, :] * centroid_stride_d
    )
    sum_offsets = (
        group_idx * sum_stride_g
        + cluster_offsets[:, None] * sum_stride_c
        + dim_offsets[None, :] * sum_stride_d
    )
    count_offsets = group_idx * count_stride_g + cluster_offsets * count_stride_c

    previous = tl.load(
        centroids + centroid_offsets,
        mask=vector_mask,
        other=0.0,
    ).to(tl.float32)
    sums = tl.load(
        cluster_sums + sum_offsets,
        mask=vector_mask,
        other=0.0,
    ).to(tl.float32)
    counts = tl.load(
        cluster_counts + count_offsets,
        mask=cluster_mask,
        other=0,
    ).to(tl.float32)

    updated = sums / tl.maximum(counts[:, None], 1.0)
    if NORMALIZE:
        squared_norm = tl.sum(updated * updated, axis=1)
        norm = tl.sqrt(tl.maximum(squared_norm, 1.0e-12))
        updated /= norm[:, None]

    updated = tl.where(counts[:, None] > 0, updated, previous)
    tl.store(centroids + centroid_offsets, updated, mask=vector_mask)


@triton.jit
def _final_assignment_kernel(
    centered_keys,
    token_keys,
    token_values,
    centroids,
    assignments,
    cluster_counts,
    token_offsets_in_cluster,
    key_sums,
    value_sums,
    num_items,
    num_centroids,
    num_segments,
    centered_stride_g,
    centered_stride_n,
    centered_stride_d,
    key_stride_g,
    key_stride_n,
    key_stride_d,
    value_stride_g,
    value_stride_n,
    value_stride_d,
    centroid_stride_g,
    centroid_stride_c,
    centroid_stride_d,
    assignment_stride_h,
    assignment_stride_n,
    count_stride_h,
    count_stride_c,
    offset_stride_h,
    offset_stride_n,
    key_sum_stride_h,
    key_sum_stride_c,
    key_sum_stride_d,
    value_sum_stride_h,
    value_sum_stride_c,
    value_sum_stride_d,
    HEAD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    item_start = tl.program_id(0) * BLOCK_N
    group_idx = tl.program_id(1)
    head_idx = group_idx // num_segments
    segment_idx = group_idx % num_segments

    item_offsets = item_start + tl.arange(0, BLOCK_N)
    cluster_offsets = tl.arange(0, BLOCK_C)
    dim_offsets = tl.arange(0, BLOCK_D)
    item_mask = item_offsets < num_items
    dim_mask = dim_offsets < HEAD_SIZE

    centered_offsets = (
        group_idx * centered_stride_g
        + item_offsets[:, None] * centered_stride_n
        + dim_offsets[None, :] * centered_stride_d
    )
    centered_block = tl.load(
        centered_keys + centered_offsets,
        mask=item_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    best_scores = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
    best_clusters = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for cluster_start in tl.range(0, num_centroids, BLOCK_C):
        current_clusters = cluster_start + cluster_offsets
        cluster_mask = current_clusters < num_centroids
        centroid_offsets = (
            group_idx * centroid_stride_g
            + current_clusters[None, :] * centroid_stride_c
            + dim_offsets[:, None] * centroid_stride_d
        )
        centroid_block = tl.load(
            centroids + centroid_offsets,
            mask=dim_mask[:, None] & cluster_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(centered_block, centroid_block).to(tl.float32)
        scores = tl.where(cluster_mask[None, :], scores, float("-inf"))
        block_scores, block_clusters = tl.max(scores, axis=1, return_indices=True)
        block_clusters += cluster_start

        replace = block_scores > best_scores
        best_scores = tl.where(replace, block_scores, best_scores)
        best_clusters = tl.where(replace, block_clusters, best_clusters)

    global_items = segment_idx * num_items + item_offsets
    global_clusters = segment_idx * num_centroids + best_clusters

    assignment_offsets = (
        head_idx * assignment_stride_h + global_items * assignment_stride_n
    )
    tl.store(assignments + assignment_offsets, global_clusters, mask=item_mask)

    count_offsets = head_idx * count_stride_h + global_clusters * count_stride_c
    cluster_token_offsets = tl.atomic_add(
        cluster_counts + count_offsets,
        1,
        mask=item_mask,
        sem="relaxed",
    )
    output_offset_offsets = head_idx * offset_stride_h + global_items * offset_stride_n
    tl.store(
        token_offsets_in_cluster + output_offset_offsets,
        cluster_token_offsets,
        mask=item_mask,
    )

    key_offsets = (
        group_idx * key_stride_g
        + item_offsets[:, None] * key_stride_n
        + dim_offsets[None, :] * key_stride_d
    )
    value_offsets = (
        group_idx * value_stride_g
        + item_offsets[:, None] * value_stride_n
        + dim_offsets[None, :] * value_stride_d
    )
    key_block = tl.load(
        token_keys + key_offsets,
        mask=item_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    value_block = tl.load(
        token_values + value_offsets,
        mask=item_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    key_sum_offsets = (
        head_idx * key_sum_stride_h
        + global_clusters[:, None] * key_sum_stride_c
        + dim_offsets[None, :] * key_sum_stride_d
    )
    value_sum_offsets = (
        head_idx * value_sum_stride_h
        + global_clusters[:, None] * value_sum_stride_c
        + dim_offsets[None, :] * value_sum_stride_d
    )
    vector_mask = item_mask[:, None] & dim_mask[None, :]
    tl.atomic_add(
        key_sums + key_sum_offsets,
        key_block.to(tl.float32),
        mask=vector_mask,
        sem="relaxed",
    )
    tl.atomic_add(
        value_sums + value_sum_offsets,
        value_block.to(tl.float32),
        mask=vector_mask,
        sem="relaxed",
    )


@triton.jit
def _finalize_cluster_summaries_kernel(
    key_sums,
    value_sums,
    cluster_counts,
    cluster_keys,
    cluster_values,
    num_clusters,
    key_sum_stride_h,
    key_sum_stride_c,
    key_sum_stride_d,
    value_sum_stride_h,
    value_sum_stride_c,
    value_sum_stride_d,
    count_stride_h,
    count_stride_c,
    key_stride_h,
    key_stride_c,
    key_stride_d,
    value_stride_h,
    value_stride_c,
    value_stride_d,
    HEAD_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cluster_start = tl.program_id(0) * BLOCK_C
    head_idx = tl.program_id(1)

    cluster_offsets = cluster_start + tl.arange(0, BLOCK_C)
    dim_offsets = tl.arange(0, BLOCK_D)
    cluster_mask = cluster_offsets < num_clusters
    dim_mask = dim_offsets < HEAD_SIZE
    vector_mask = cluster_mask[:, None] & dim_mask[None, :]

    key_sum_offsets = (
        head_idx * key_sum_stride_h
        + cluster_offsets[:, None] * key_sum_stride_c
        + dim_offsets[None, :] * key_sum_stride_d
    )
    value_sum_offsets = (
        head_idx * value_sum_stride_h
        + cluster_offsets[:, None] * value_sum_stride_c
        + dim_offsets[None, :] * value_sum_stride_d
    )
    count_offsets = head_idx * count_stride_h + cluster_offsets * count_stride_c

    keys = tl.load(
        key_sums + key_sum_offsets,
        mask=vector_mask,
        other=0.0,
    )
    values = tl.load(
        value_sums + value_sum_offsets,
        mask=vector_mask,
        other=0.0,
    )
    counts = tl.load(
        cluster_counts + count_offsets,
        mask=cluster_mask,
        other=0,
    ).to(tl.float32)

    denominator = tl.maximum(counts[:, None], 1.0)
    keys /= denominator
    values /= denominator
    keys = tl.where(counts[:, None] > 0, keys, 0.0)
    values = tl.where(counts[:, None] > 0, values, 0.0)

    output_key_offsets = (
        head_idx * key_stride_h
        + cluster_offsets[:, None] * key_stride_c
        + dim_offsets[None, :] * key_stride_d
    )
    output_value_offsets = (
        head_idx * value_stride_h
        + cluster_offsets[:, None] * value_stride_c
        + dim_offsets[None, :] * value_stride_d
    )
    tl.store(cluster_keys + output_key_offsets, keys, mask=vector_mask)
    tl.store(cluster_values + output_value_offsets, values, mask=vector_mask)


def segmented_kmeans_cuda(
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
    segment_size: int,
    items_per_cluster: int,
    num_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_heads, num_tokens, head_size = token_keys.shape
    num_segments = num_tokens // segment_size
    clusters_per_segment = segment_size // items_per_cluster
    num_clusters = num_segments * clusters_per_segment
    num_groups = num_heads * num_segments

    grouped_keys = token_keys.reshape(num_groups, segment_size, head_size)
    grouped_values = token_values.reshape(num_groups, segment_size, head_size)

    float_keys = grouped_keys.float()
    centered_keys = float_keys - float_keys.mean(dim=1, keepdim=True)
    initial_indices = (
        (
            torch.arange(
                clusters_per_segment,
                dtype=torch.float32,
                device=token_keys.device,
            )
            + 0.5
        )
        * segment_size
        / clusters_per_segment
    ).to(torch.int64)
    initial_indices.clamp_(max=segment_size - 1)

    centroids = centered_keys.index_select(1, initial_indices).contiguous()
    cluster_sums = torch.empty_like(centroids, dtype=torch.float32)
    training_counts = torch.empty(
        num_groups,
        clusters_per_segment,
        dtype=torch.int32,
        device=token_keys.device,
    )

    block_n = 64 if head_size > 128 else 128
    block_c = 64
    block_d = max(16, triton.next_power_of_2(head_size))
    assignment_grid = (triton.cdiv(segment_size, block_n), num_groups)
    update_grid = (triton.cdiv(clusters_per_segment, block_c), num_groups)

    for _ in range(num_iterations - 1):
        cluster_sums.zero_()
        training_counts.zero_()
        _assign_clusters_kernel[assignment_grid](
            centered_keys,
            centroids,
            cluster_sums,
            training_counts,
            segment_size,
            clusters_per_segment,
            centered_keys.stride(0),
            centered_keys.stride(1),
            centered_keys.stride(2),
            centroids.stride(0),
            centroids.stride(1),
            centroids.stride(2),
            cluster_sums.stride(0),
            cluster_sums.stride(1),
            cluster_sums.stride(2),
            training_counts.stride(0),
            training_counts.stride(1),
            HEAD_SIZE=head_size,
            BLOCK_N=block_n,
            BLOCK_C=block_c,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )
        _update_centroids_kernel[update_grid](
            centroids,
            cluster_sums,
            training_counts,
            clusters_per_segment,
            centroids.stride(0),
            centroids.stride(1),
            centroids.stride(2),
            cluster_sums.stride(0),
            cluster_sums.stride(1),
            cluster_sums.stride(2),
            training_counts.stride(0),
            training_counts.stride(1),
            HEAD_SIZE=head_size,
            NORMALIZE=True,
            BLOCK_C=block_c,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=1,
        )

    assignments = torch.empty(
        num_heads,
        num_tokens,
        dtype=torch.int32,
        device=token_keys.device,
    )
    cluster_counts = torch.zeros(
        num_heads,
        num_clusters,
        dtype=torch.int32,
        device=token_keys.device,
    )
    token_offsets_in_cluster = torch.empty_like(assignments)
    key_sums = torch.zeros(
        num_heads,
        num_clusters,
        head_size,
        dtype=torch.float32,
        device=token_keys.device,
    )
    value_sums = torch.zeros_like(key_sums)

    _final_assignment_kernel[assignment_grid](
        centered_keys,
        grouped_keys,
        grouped_values,
        centroids,
        assignments,
        cluster_counts,
        token_offsets_in_cluster,
        key_sums,
        value_sums,
        segment_size,
        clusters_per_segment,
        num_segments,
        centered_keys.stride(0),
        centered_keys.stride(1),
        centered_keys.stride(2),
        grouped_keys.stride(0),
        grouped_keys.stride(1),
        grouped_keys.stride(2),
        grouped_values.stride(0),
        grouped_values.stride(1),
        grouped_values.stride(2),
        centroids.stride(0),
        centroids.stride(1),
        centroids.stride(2),
        assignments.stride(0),
        assignments.stride(1),
        cluster_counts.stride(0),
        cluster_counts.stride(1),
        token_offsets_in_cluster.stride(0),
        token_offsets_in_cluster.stride(1),
        key_sums.stride(0),
        key_sums.stride(1),
        key_sums.stride(2),
        value_sums.stride(0),
        value_sums.stride(1),
        value_sums.stride(2),
        HEAD_SIZE=head_size,
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    cluster_keys = torch.empty(
        num_heads,
        num_clusters,
        head_size,
        dtype=token_keys.dtype,
        device=token_keys.device,
    )
    cluster_values = torch.empty_like(cluster_keys)
    finalize_grid = (triton.cdiv(num_clusters, block_c), num_heads)
    _finalize_cluster_summaries_kernel[finalize_grid](
        key_sums,
        value_sums,
        cluster_counts,
        cluster_keys,
        cluster_values,
        num_clusters,
        key_sums.stride(0),
        key_sums.stride(1),
        key_sums.stride(2),
        value_sums.stride(0),
        value_sums.stride(1),
        value_sums.stride(2),
        cluster_counts.stride(0),
        cluster_counts.stride(1),
        cluster_keys.stride(0),
        cluster_keys.stride(1),
        cluster_keys.stride(2),
        cluster_values.stride(0),
        cluster_values.stride(1),
        cluster_values.stride(2),
        HEAD_SIZE=head_size,
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )

    return (
        assignments,
        cluster_counts,
        cluster_keys,
        cluster_values,
        token_offsets_in_cluster,
    )
