# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import IntEnum
from math import ceil

import torch


class RetroSpecAttentionLevel(IntEnum):
    SPARSE = 0
    EXPANDED = 1


class RetroSpecIndexBase:
    """Common validation and recent-zone configuration for RetroSpec indices."""

    def __init__(
        self,
        block_size: int,
        num_speculative_tokens: int,
        retrieval_ratio: float,
        estimation_ratio: float,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        if not 0 < retrieval_ratio < 1:
            raise ValueError("retrieval_ratio must be in (0, 1)")
        if not 0 <= estimation_ratio < 1:
            raise ValueError("estimation_ratio must be in [0, 1)")
        if retrieval_ratio + estimation_ratio > 1:
            raise ValueError(
                "retrieval_ratio and estimation_ratio must sum to at most 1"
            )

        self.block_size = block_size
        self.retrieval_ratio = retrieval_ratio
        self.estimation_ratio = estimation_ratio
        self.num_recent_blocks = ceil(num_speculative_tokens / block_size) + 1

    @staticmethod
    def _validate_inputs(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        if query.ndim != 3:
            raise ValueError("query must have shape [batch, query_heads, head_size]")
        if key_cache.ndim != 4 or value_cache.ndim != 4:
            raise ValueError(
                "KV cache must have shape [num_blocks, block_size, kv_heads, head_size]"
            )
        if key_cache.shape != value_cache.shape:
            raise ValueError("key_cache and value_cache must have equal shapes")
        if block_table.ndim != 2:
            raise ValueError("block_table must be two-dimensional")
        if seq_lens.ndim != 1:
            raise ValueError("seq_lens must be one-dimensional")
        if active_mask.ndim != 1 or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a one-dimensional boolean tensor")

        batch_size = query.shape[0]
        if block_table.shape[0] != batch_size:
            raise ValueError("block_table batch size does not match query")
        if seq_lens.shape[0] != batch_size:
            raise ValueError("seq_lens batch size does not match query")
        if active_mask.shape[0] != batch_size:
            raise ValueError("active_mask batch size does not match query")

        device = query.device
        for name, tensor in (
            ("key_cache", key_cache),
            ("value_cache", value_cache),
            ("block_table", block_table),
            ("seq_lens", seq_lens),
            ("active_mask", active_mask),
        ):
            if tensor.device != device:
                raise ValueError(
                    f"{name} must be on device {device}, but is on {tensor.device}"
                )

    @staticmethod
    def _mask_rank_range(
        ranked_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
        start_counts: torch.Tensor,
        end_counts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_num_candidates = candidate_mask.shape
        rank_positions = torch.arange(
            max_num_candidates,
            dtype=torch.int64,
            device=candidate_mask.device,
        ).expand(batch_size, -1)
        selected_by_rank = (rank_positions >= start_counts.unsqueeze(1)) & (
            rank_positions < end_counts.unsqueeze(1)
        )

        selected_mask = torch.zeros_like(candidate_mask)
        selected_mask.scatter_(1, ranked_indices, selected_by_rank)
        selected_mask &= candidate_mask
        return selected_mask

    def _select_zone_masks(
        self,
        scores: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_counts = candidate_mask.sum(dim=1)
        retrieval_counts = torch.ceil(
            candidate_counts.float() * self.retrieval_ratio
        ).to(torch.int64)
        retrieval_counts = torch.minimum(retrieval_counts, candidate_counts)

        estimation_counts = torch.ceil(
            candidate_counts.float() * self.estimation_ratio
        ).to(torch.int64)
        estimation_counts = torch.minimum(
            estimation_counts,
            candidate_counts - retrieval_counts,
        )
        total_compute_counts = retrieval_counts + estimation_counts
        expanded_retrieval_counts = torch.minimum(
            retrieval_counts * 2,
            total_compute_counts,
        )
        ranking_scores = scores.masked_fill(~candidate_mask, float("-inf"))
        ranked_indices = torch.argsort(ranking_scores, dim=1, descending=True)
        zero_counts = torch.zeros_like(retrieval_counts)

        return (
            self._mask_rank_range(
                ranked_indices,
                candidate_mask,
                zero_counts,
                retrieval_counts,
            ),
            self._mask_rank_range(
                ranked_indices,
                candidate_mask,
                retrieval_counts,
                total_compute_counts,
            ),
            self._mask_rank_range(
                ranked_indices,
                candidate_mask,
                zero_counts,
                expanded_retrieval_counts,
            ),
            self._mask_rank_range(
                ranked_indices,
                candidate_mask,
                expanded_retrieval_counts,
                total_compute_counts,
            ),
        )
