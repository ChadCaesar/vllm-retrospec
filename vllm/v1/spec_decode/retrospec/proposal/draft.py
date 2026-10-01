# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.retrospec.attention import RetroSpecAttentionMode
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage

if TYPE_CHECKING:
    pass


class RetroSpecDraftMixin:
    def _run_draft_step(
        self,
        batch_size: int,
        draft_index: int,
        common_attn_metadata: CommonAttentionMetadata,
        active_mask: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        return self._run_model_step(
            batch_size=batch_size,
            step_index=draft_index,
            input_ids=self.input_ids,
            positions=self.positions[:batch_size],
            active_mask=active_mask,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            attention_mode=RetroSpecAttentionMode.DRAFT,
            compute_margin=(self.policy.draft_margin_threshold is not None),
        )

    def _begin_draft_round(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare one dynamic draft round without reading state on CPU."""
        if not 0 <= batch_size <= self.max_batch_size:
            raise ValueError("batch_size exceeds the proposal workspace capacity")

        round_start_counts = self._draft_round_start_counts[:batch_size]
        round_start_counts.copy_(self.state.pending_counts)

        draft_round_mask = self._draft_round_mask[:batch_size]
        torch.eq(
            self.state.stage,
            int(RetroSpecStage.DRAFT),
            out=draft_round_mask,
        )
        draft_round_mask.logical_and_(self.state.active_mask)
        draft_round_mask.logical_and_(
            self.state.pending_counts < self.policy.pending_limit
        )
        draft_round_mask.logical_and_(
            self.state.pending_counts < self._proposal_token_budgets.gpu[:batch_size]
        )
        draft_round_mask.logical_and_(
            self.positions[:batch_size] < self.max_model_len - 1
        )

        # Draft counts describe only tokens generated during this round.
        self.state.reset_draft_counts(self.state.active_mask)
        bonus_ready = self._bonus_ready_mask[:batch_size]
        bonus_ready.logical_and_(draft_round_mask)
        self.state.add_draft_counts(bonus_ready.to(torch.int32))
        bonus_ready.zero_()
        return draft_round_mask, round_start_counts

    def _prepare_next_draft_step(
        self,
        batch_size: int,
        draft_round_mask: torch.Tensor,
        round_start_counts: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """Return the next real draft column and its GPU request mask.

        ``num_speculative_tokens`` is the no-work sentinel. Only the selected
        column crosses to the host because Python must dispatch the matching
        target-model forward.
        """
        if draft_round_mask.shape != (batch_size,):
            raise ValueError("draft_round_mask must match the proposal batch")
        if round_start_counts.shape != (batch_size,):
            raise ValueError("round_start_counts must match the proposal batch")

        draft_stage_mask = self._draft_stage_mask[:batch_size]
        if batch_size == 0:
            return self.num_speculative_tokens, draft_stage_mask.zero_()

        next_token_indices = self._draft_next_token_indices[:batch_size]
        torch.add(
            round_start_counts,
            self.state.draft_counts,
            out=next_token_indices,
        )

        runnable_mask = self._draft_runnable_mask[:batch_size]
        runnable_mask.copy_(draft_round_mask)
        runnable_mask.logical_and_(self.state.active_mask)
        runnable_mask.logical_and_(self.state.stage == int(RetroSpecStage.DRAFT))
        runnable_mask.logical_and_(next_token_indices < self.policy.pending_limit)
        runnable_mask.logical_and_(
            next_token_indices < self._proposal_token_budgets.gpu[:batch_size]
        )
        runnable_mask.logical_and_(self.positions[:batch_size] < self.max_model_len - 1)

        next_token_indices.masked_fill_(
            ~runnable_mask,
            self.num_speculative_tokens,
        )
        token_index = int(next_token_indices.amin().item())

        if token_index >= self.num_speculative_tokens:
            draft_stage_mask.zero_()
            return token_index, draft_stage_mask

        torch.eq(next_token_indices, token_index, out=draft_stage_mask)
        draft_stage_mask.logical_and_(runnable_mask)
        return token_index, draft_stage_mask
