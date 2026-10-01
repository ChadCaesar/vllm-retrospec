# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.v1.spec_decode.retrospec.cluster_store import (
    RetroSpecClusterPageStore,
)
from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecResidentLayerArena,
)


def make_cluster_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    keys = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[10.0], [11.0], [12.0], [13.0], [14.0]],
        ]
    )
    values = keys + 100.0
    assignments = torch.tensor(
        [
            [0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1],
        ],
        dtype=torch.int64,
    )
    cluster_token_counts = torch.tensor(
        [[3, 2], [2, 3]],
        dtype=torch.int32,
    )
    return keys, values, assignments, cluster_token_counts


def make_token_offsets(
    assignments: torch.Tensor,
    cluster_token_counts: torch.Tensor,
) -> torch.Tensor:
    num_heads, num_tokens = assignments.shape
    offsets = torch.empty_like(assignments, dtype=torch.int32)

    for head_idx in range(num_heads):
        next_offsets = torch.zeros(
            cluster_token_counts.shape[1],
            dtype=torch.int32,
            device=assignments.device,
        )
        for token_idx in range(num_tokens):
            cluster_idx = int(assignments[head_idx, token_idx].item())
            if not 0 <= cluster_idx < cluster_token_counts.shape[1]:
                offsets[head_idx, token_idx] = 0
                continue
            offsets[head_idx, token_idx] = next_offsets[cluster_idx]
            next_offsets[cluster_idx] += 1

    return offsets


def store_cluster_data(
    store: RetroSpecClusterPageStore,
    layer_name: str,
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
    assignments: torch.Tensor,
    cluster_token_counts: torch.Tensor,
    request_id: str = "request",
    cluster_start: int = 0,
):
    return store.store_clusters(
        layer_name=layer_name,
        request_id=request_id,
        cluster_start=cluster_start,
        token_keys=token_keys,
        token_values=token_values,
        assignments=assignments,
        cluster_token_counts=cluster_token_counts,
        token_offsets_in_cluster=make_token_offsets(
            assignments,
            cluster_token_counts,
        ),
    )


def get_block_metadata(store, table, device=None):
    return store.get_cluster_block_metadata(
        layer_name="layer",
        cluster_ids=table.cluster_ids,
        device=device,
    )


def get_runtime_blocks(store, table, device):
    cluster_ids = table.cluster_ids.to(device=device)
    metadata = store.get_cluster_block_metadata(
        layer_name="layer",
        cluster_ids=cluster_ids,
        device=device,
    )
    return cluster_ids, metadata


def make_resident_arena(
    table, metadata, cluster_token_counts, device
) -> RetroSpecResidentLayerArena:
    num_kv_heads, num_clusters, max_pages = metadata.page_ids.shape
    cluster_shape = (num_kv_heads, num_clusters, 1)
    return RetroSpecResidentLayerArena(
        cluster_ids=table.cluster_ids.to(device),
        cluster_keys=torch.zeros(cluster_shape, device=device),
        cluster_values=torch.zeros(cluster_shape, device=device),
        cluster_token_counts=cluster_token_counts.to(device),
        cluster_page_starts=torch.arange(
            num_clusters, dtype=torch.int64, device=device
        )[None, :].expand(num_kv_heads, -1)
        * max_pages,
        cluster_page_counts=(metadata.page_ids >= 0).sum(dim=-1, dtype=torch.int32),
        resident_table_buckets=torch.full(
            (num_kv_heads, num_clusters),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        page_ids=metadata.page_ids.flatten(1),
        page_token_counts=metadata.page_token_counts.flatten(1),
        cluster_offsets=torch.zeros(1, dtype=torch.int64, device=device),
        num_clusters=torch.full((1,), num_clusters, dtype=torch.int32, device=device),
        page_offsets=torch.zeros(1, dtype=torch.int64, device=device),
        num_pages=torch.full(
            (1,), num_clusters * max_pages, dtype=torch.int32, device=device
        ),
        generations=torch.ones(1, dtype=torch.int64, device=device),
        indexed_starts=torch.zeros(1, dtype=torch.int64, device=device),
        indexed_ends=torch.ones(1, dtype=torch.int64, device=device),
    )


def materialize_resolved_pages(resolved):
    page_shape = resolved.resident_key_pages.shape[1:]
    output_shape = (*resolved.resident_page_ids.shape, *page_shape)
    keys = torch.zeros(
        output_shape,
        dtype=resolved.resident_key_pages.dtype,
        device=resolved.resident_key_pages.device,
    )
    values = torch.zeros_like(keys)

    resident_mask = resolved.resident_page_ids >= 0
    if resident_mask.any():
        resident_slots = resolved.resident_page_ids[resident_mask].to(torch.int64)
        keys[resident_mask] = resolved.resident_key_pages.index_select(
            0, resident_slots
        )
        values[resident_mask] = resolved.resident_value_pages.index_select(
            0, resident_slots
        )

    staging_mask = resolved.staging_page_ids >= 0
    if staging_mask.any():
        staging_slots = resolved.staging_page_ids[staging_mask].to(torch.int64)
        keys[staging_mask] = resolved.staging_key_pages.index_select(0, staging_slots)
        values[staging_mask] = resolved.staging_value_pages.index_select(
            0, staging_slots
        )
    return keys, values, resident_mask | staging_mask
