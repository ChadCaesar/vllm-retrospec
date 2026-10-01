# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch

from vllm.v1.spec_decode.retrospec.index_residency import (
    RetroSpecResidentBatchView,
)
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)


def make_index(
    prefill_segment_size_tokens: int = 4,
    generation_update_interval: int = 2,
    blocks_per_cluster: int = 1,
    retrieval_ratio: float = 0.5,
    estimation_ratio: float = 0.5,
    cache_ratio: float = 0.0,
    pin_memory: bool = False,
    max_pending_cluster_builds: int = 2,
    max_resident_requests: int = 1,
    prefill_warmup_multiplier: int = 4,
    num_speculative_tokens: int = 1,
    replay_mode: str = "off",
) -> RetroSpecSegmentedTokenIndex:
    return RetroSpecSegmentedTokenIndex(
        block_size=2,
        num_speculative_tokens=num_speculative_tokens,
        retrieval_ratio=retrieval_ratio,
        estimation_ratio=estimation_ratio,
        prefill_segment_size_tokens=prefill_segment_size_tokens,
        generation_update_interval=generation_update_interval,
        blocks_per_cluster=blocks_per_cluster,
        num_kmeans_iterations=2,
        max_model_len=64,
        max_pending_cluster_builds=max_pending_cluster_builds,
        cache_ratio=cache_ratio,
        pin_memory=pin_memory,
        max_resident_requests=max_resident_requests,
        prefill_warmup_multiplier=prefill_warmup_multiplier,
        replay_mode=replay_mode,
    )


def make_cache(
    num_blocks: int = 8,
    num_kv_heads: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.empty(num_blocks, 2, num_kv_heads, 1)
    values = torch.empty_like(keys)
    for block_id in range(num_blocks):
        keys[block_id].fill_(float(block_id))
        values[block_id].fill_(float(block_id * 10))
    return keys, values


def make_empty_resident_view(
    batch_size: int,
    num_clusters: int,
    device: torch.device,
) -> RetroSpecResidentBatchView:
    return RetroSpecResidentBatchView(
        arena=None,
        request_slot_ids=torch.full(
            (batch_size,), -1, dtype=torch.int64, device=device
        ),
        max_num_clusters=num_clusters,
        max_pages_per_cluster=0,
        max_num_pages=0,
    )


@contextmanager
def active_residency(
    index: RetroSpecSegmentedTokenIndex,
    request_ids: list[str],
):
    index.begin_full_verification_residency(request_ids)
    try:
        yield
    finally:
        index.end_full_verification_residency()


def build_index(
    index: RetroSpecSegmentedTokenIndex,
    seq_len: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
    defer_cpu_store: bool = False,
    is_prefill: bool = True,
    prefill_complete: bool = False,
) -> None:
    index.build_or_update(
        layer_name="layer",
        request_ids=["request"],
        seq_lens=[seq_len],
        is_prefill=[is_prefill],
        rows=[0],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
        defer_cpu_store=defer_cpu_store,
        prefill_complete=[prefill_complete],
    )


def materialize_reference(
    index: RetroSpecSegmentedTokenIndex,
    selection,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return index.materialize_exact_reference(
        selection,
        keys,
        values,
        block_table,
    )
