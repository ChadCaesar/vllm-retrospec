# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import IntEnum
from math import ceil

import torch


class RetroSpecAttentionLevel(IntEnum):
    SPARSE = 0
    EXPANDED = 1


@dataclass(frozen=True)
class RetroSpecSelectionPlan:
    sparse_exact_indices: torch.Tensor
    sparse_exact_mask: torch.Tensor
    sparse_estimation_indices: torch.Tensor
    sparse_estimation_mask: torch.Tensor

    expanded_exact_indices: torch.Tensor
    expanded_exact_mask: torch.Tensor
    expanded_estimation_indices: torch.Tensor
    expanded_estimation_mask: torch.Tensor

    valid_token_counts: torch.Tensor
    sparse_attn: torch.Tensor
    expanded_attn: torch.Tensor


@dataclass(frozen=True)
class RetroSpecAttentionSelection:
    exact_block_table: torch.Tensor
    exact_seq_lens: torch.Tensor

    estimation_keys: torch.Tensor
    estimation_values: torch.Tensor
    estimation_token_counts: torch.Tensor

    attention_mass: torch.Tensor
    plan: RetroSpecSelectionPlan

    @property
    def hit_attn(self) -> torch.Tensor:
        return self.plan.sparse_attn


class RetroSpecBlockIndex:
    """Select KV blocks used by RetroSpec draft attention.

    The GPU-reference implementation treats one vLLM KV block as one
    retrieval unit. Prefix and recent blocks are always computed exactly.
    Other blocks are divided into retrieval, estimation, and ignored zones.
    """

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

        # Keep the block containing the current token and enough preceding
        # blocks to cover the complete speculative region.
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

    def _build_block_layout(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_num_blocks = block_table.shape[1]
        logical_block_ids = torch.arange(
            max_num_blocks, dtype=torch.int64, device=block_table.device
        )

        valid_block_counts = torch.div(
            seq_lens.to(torch.int64) + self.block_size - 1,
            self.block_size,
            rounding_mode="floor",
        )
        valid_block_counts.clamp_(min=1, max=max_num_blocks)

        valid_block_mask = logical_block_ids.unsqueeze(
            0
        ) < valid_block_counts.unsqueeze(1)

        remaining_tokens = (
            seq_lens.to(torch.int64).unsqueeze(1)
            - logical_block_ids.unsqueeze(0) * self.block_size
        )
        valid_token_counts = remaining_tokens.clamp(min=0, max=self.block_size)
        valid_token_counts *= valid_block_mask

        recent_start = (valid_block_counts - self.num_recent_blocks).clamp_min(0)

        forced_exact_mask = valid_block_mask & (
            (logical_block_ids.unsqueeze(0) == 0)
            | (logical_block_ids.unsqueeze(0) >= recent_start.unsqueeze(1))
        )

        return (
            logical_block_ids,
            valid_block_mask,
            valid_token_counts,
            forced_exact_mask,
        )

    def _compute_block_means(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        valid_block_mask: torch.Tensor,
        valid_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_num_blocks = block_table.shape
        _, cache_block_size, num_kv_heads, head_size = cache.shape

        if cache_block_size != self.block_size:
            raise ValueError(
                f"KV cache block size {cache_block_size} does not match "
                f"configured block size {self.block_size}"
            )

        flat_valid_mask = valid_block_mask.reshape(-1)
        valid_positions = torch.nonzero(flat_valid_mask, as_tuple=False).flatten()

        flat_block_table = block_table.reshape(-1)
        physical_block_ids = flat_block_table.index_select(0, valid_positions).to(
            torch.int64
        )

        selected_blocks = cache.index_select(0, physical_block_ids).float()

        flat_token_counts = valid_token_counts.reshape(-1)
        selected_token_counts = flat_token_counts.index_select(0, valid_positions)

        token_indices = torch.arange(self.block_size, device=cache.device)
        token_mask = token_indices.unsqueeze(0) < selected_token_counts.unsqueeze(1)

        selected_sums = (selected_blocks * token_mask[:, :, None, None]).sum(dim=1)

        selected_means = (
            selected_sums / selected_token_counts.clamp_min(1)[:, None, None]
        )

        flat_means = torch.zeros(
            batch_size * max_num_blocks,
            num_kv_heads,
            head_size,
            dtype=torch.float32,
            device=cache.device,
        )
        flat_means.index_copy_(0, valid_positions, selected_means)

        return flat_means.view(batch_size, max_num_blocks, num_kv_heads, head_size)

    @staticmethod
    def _score_blocks(
        query: torch.Tensor,
        key_centroids: torch.Tensor,
        candidate_mask: torch.Tensor,
        valid_token_counts: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        batch_size, num_query_heads, head_size = query.shape
        num_kv_heads = key_centroids.shape[2]

        if key_centroids.shape[0] != batch_size:
            raise ValueError("key centroid batch size does not match query")
        if key_centroids.shape[3] != head_size:
            raise ValueError("key centroid head size does not match query")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        num_queries_per_kv = num_query_heads // num_kv_heads
        grouped_query = query.float().view(
            batch_size, num_kv_heads, num_queries_per_kv, head_size
        )

        logits = torch.einsum("bhgd,bmhd->bhgm", grouped_query, key_centroids)
        logits *= scale

        # A centroid represents valid_token_counts equivalent keys.
        logits += torch.log(valid_token_counts.clamp_min(1).float())[:, None, None, :]

        expanded_candidate_mask = candidate_mask[:, None, None, :]
        logits.masked_fill_(~expanded_candidate_mask, float("-inf"))

        has_candidates = candidate_mask.any(dim=1)
        safe_logits = torch.where(
            has_candidates[:, None, None, None], logits, torch.zeros_like(logits)
        )

        head_probabilities = torch.softmax(
            safe_logits,
            dim=-1,
        )
        head_probabilities.masked_fill_(~expanded_candidate_mask, 0.0)

        return head_probabilities.mean(dim=(1, 2))

    @staticmethod
    def _mask_rank_range(
        ranked_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
        start_counts: torch.Tensor,
        end_counts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_num_blocks = candidate_mask.shape
        rank_positions = torch.arange(
            max_num_blocks, dtype=torch.int64, device=candidate_mask.device
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
        block_scores: torch.Tensor,
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
            estimation_counts, candidate_counts - retrieval_counts
        )

        total_compute_counts = retrieval_counts + estimation_counts

        # Expanded verification promotes up to another retrieval budget from
        # estimation into exact computation. The total covered area is unchanged.
        expanded_retrieval_counts = torch.minimum(
            retrieval_counts * 2, total_compute_counts
        )
        ranking_scores = block_scores.masked_fill(~candidate_mask, float("-inf"))
        ranked_indices = torch.argsort(ranking_scores, dim=1, descending=True)

        zero_counts = torch.zeros_like(retrieval_counts)

        sparse_retrieval_mask = self._mask_rank_range(
            ranked_indices, candidate_mask, zero_counts, retrieval_counts
        )
        sparse_estimation_mask = self._mask_rank_range(
            ranked_indices, candidate_mask, retrieval_counts, total_compute_counts
        )

        expanded_retrieval_mask = self._mask_rank_range(
            ranked_indices, candidate_mask, zero_counts, expanded_retrieval_counts
        )
        expanded_estimation_mask = self._mask_rank_range(
            ranked_indices,
            candidate_mask,
            expanded_retrieval_counts,
            total_compute_counts,
        )

        return (
            sparse_retrieval_mask,
            sparse_estimation_mask,
            expanded_retrieval_mask,
            expanded_estimation_mask,
        )

    @staticmethod
    def _pack_logical_indices(
        mask: torch.Tensor,
        logical_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_num_blocks = mask.shape
        expanded_logical_ids = logical_block_ids.expand(batch_size, -1)

        sentinel = torch.full_like(expanded_logical_ids, max_num_blocks)
        packed_indices = (
            torch.where(mask, expanded_logical_ids, sentinel).sort(dim=1).values
        )

        packed_counts = mask.sum(dim=1)
        packed_mask = logical_block_ids.unsqueeze(0) < packed_counts.unsqueeze(1)

        safe_indices = packed_indices.clamp(min=0, max=max_num_blocks - 1)
        return safe_indices, packed_mask

    @staticmethod
    def _build_estimation_selection(
        key_centroids: torch.Tensor,
        value_means: torch.Tensor,
        valid_token_counts: torch.Tensor,
        packed_indices: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_num_blocks = packed_indices.shape
        num_kv_heads = key_centroids.shape[2]
        head_size = key_centroids.shape[3]

        vector_indices = packed_indices[:, :, None, None].expand(
            batch_size, max_num_blocks, num_kv_heads, head_size
        )

        estimation_keys = key_centroids.gather(1, vector_indices)
        estimation_values = value_means.gather(1, vector_indices)
        estimation_token_counts = valid_token_counts.gather(1, packed_indices)

        estimation_keys.masked_fill_(~packed_mask[:, :, None, None], 0.0)
        estimation_values.masked_fill_(~packed_mask[:, :, None, None], 0.0)
        estimation_token_counts.masked_fill_(~packed_mask, 0)

        return (
            estimation_keys.contiguous(),
            estimation_values.contiguous(),
            estimation_token_counts.to(dtype=torch.int32).contiguous(),
        )

    def _build_plan(
        self,
        logical_block_ids: torch.Tensor,
        valid_token_counts: torch.Tensor,
        forced_exact_mask: torch.Tensor,
        sparse_retrieval_mask: torch.Tensor,
        sparse_estimation_mask: torch.Tensor,
        expanded_retrieval_mask: torch.Tensor,
        expanded_estimation_mask: torch.Tensor,
        sparse_attn: torch.Tensor,
        expanded_attn: torch.Tensor,
    ) -> RetroSpecSelectionPlan:
        sparse_exact_indices, sparse_exact_mask = self._pack_logical_indices(
            forced_exact_mask | sparse_retrieval_mask, logical_block_ids
        )
        (
            sparse_estimation_indices,
            sparse_estimation_packed_mask,
        ) = self._pack_logical_indices(sparse_estimation_mask, logical_block_ids)

        expanded_exact_indices, expanded_exact_mask = self._pack_logical_indices(
            forced_exact_mask | expanded_retrieval_mask,
            logical_block_ids,
        )
        (
            expanded_estimation_indices,
            expanded_estimation_packed_mask,
        ) = self._pack_logical_indices(expanded_estimation_mask, logical_block_ids)

        return RetroSpecSelectionPlan(
            sparse_exact_indices=sparse_exact_indices,
            sparse_exact_mask=sparse_exact_mask,
            sparse_estimation_indices=sparse_estimation_indices,
            sparse_estimation_mask=sparse_estimation_packed_mask,
            expanded_exact_indices=expanded_exact_indices,
            expanded_exact_mask=expanded_exact_mask,
            expanded_estimation_indices=expanded_estimation_indices,
            expanded_estimation_mask=expanded_estimation_packed_mask,
            valid_token_counts=valid_token_counts,
            sparse_attn=sparse_attn,
            expanded_attn=expanded_attn,
        )

    def _materialize_from_means(
        self,
        plan: RetroSpecSelectionPlan,
        level: RetroSpecAttentionLevel,
        block_table: torch.Tensor,
        key_centroids: torch.Tensor,
        value_means: torch.Tensor,
    ) -> RetroSpecAttentionSelection:
        if level == RetroSpecAttentionLevel.SPARSE:
            exact_indices = plan.sparse_exact_indices
            exact_mask = plan.sparse_exact_mask
            estimation_indices = plan.sparse_estimation_indices
            estimation_mask = plan.sparse_estimation_mask
            attention_mass = plan.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            exact_indices = plan.expanded_exact_indices
            exact_mask = plan.expanded_exact_mask
            estimation_indices = plan.expanded_estimation_indices
            estimation_mask = plan.expanded_estimation_mask
            attention_mass = plan.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        exact_block_table = block_table.gather(1, exact_indices)
        exact_block_table.masked_fill_(~exact_mask, 0)
        exact_block_table = exact_block_table.to(torch.int32).contiguous()

        exact_token_counts = plan.valid_token_counts.gather(1, exact_indices)
        exact_token_counts *= exact_mask
        exact_seq_lens = exact_token_counts.sum(dim=1).to(torch.int32).contiguous()

        (
            estimation_keys,
            estimation_values,
            estimation_token_counts,
        ) = self._build_estimation_selection(
            key_centroids,
            value_means,
            plan.valid_token_counts,
            estimation_indices,
            estimation_mask,
        )

        return RetroSpecAttentionSelection(
            exact_block_table=exact_block_table,
            exact_seq_lens=exact_seq_lens,
            estimation_keys=estimation_keys,
            estimation_values=estimation_values,
            estimation_token_counts=estimation_token_counts,
            attention_mass=attention_mass,
            plan=plan,
        )

    def materialize(
        self,
        plan: RetroSpecSelectionPlan,
        level: RetroSpecAttentionLevel,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecAttentionSelection:
        valid_block_mask = plan.valid_token_counts > 0

        key_centroids = self._compute_block_means(
            key_cache, block_table, valid_block_mask, plan.valid_token_counts
        )
        value_means = self._compute_block_means(
            value_cache, block_table, valid_block_mask, plan.valid_token_counts
        )

        return self._materialize_from_means(
            plan, level, block_table, key_centroids, value_means
        )

    def select(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        active_mask: torch.Tensor,
        scale: float,
    ) -> RetroSpecAttentionSelection:
        self._validate_inputs(
            query, key_cache, value_cache, block_table, seq_lens, active_mask
        )

        (
            logical_block_ids,
            valid_block_mask,
            valid_token_counts,
            forced_exact_mask,
        ) = self._build_block_layout(block_table, seq_lens)

        key_centroids = self._compute_block_means(
            key_cache, block_table, valid_block_mask, valid_token_counts
        )
        value_means = self._compute_block_means(
            value_cache, block_table, valid_block_mask, valid_token_counts
        )

        candidate_mask = valid_block_mask & ~forced_exact_mask
        block_scores = self._score_blocks(
            query, key_centroids, candidate_mask, valid_token_counts, scale
        )

        (
            sparse_retrieval_mask,
            sparse_estimation_mask,
            expanded_retrieval_mask,
            expanded_estimation_mask,
        ) = self._select_zone_masks(block_scores, candidate_mask)

        sparse_attn = (block_scores * sparse_retrieval_mask).sum(dim=1)
        expanded_attn = (block_scores * expanded_retrieval_mask).sum(dim=1)

        has_candidates = candidate_mask.any(dim=1)
        sparse_attn = torch.where(
            has_candidates, sparse_attn, torch.ones_like(sparse_attn)
        )
        expanded_attn = torch.where(
            has_candidates, expanded_attn, torch.ones_like(expanded_attn)
        )

        sparse_attn = torch.where(
            active_mask, sparse_attn, torch.ones_like(sparse_attn)
        )
        expanded_attn = torch.where(
            active_mask, expanded_attn, torch.ones_like(expanded_attn)
        )

        plan = self._build_plan(
            logical_block_ids,
            valid_token_counts,
            forced_exact_mask,
            sparse_retrieval_mask,
            sparse_estimation_mask,
            expanded_retrieval_mask,
            expanded_estimation_mask,
            sparse_attn,
            expanded_attn,
        )

        return self._materialize_from_means(
            plan,
            RetroSpecAttentionLevel.SPARSE,
            block_table,
            key_centroids,
            value_means,
        )
