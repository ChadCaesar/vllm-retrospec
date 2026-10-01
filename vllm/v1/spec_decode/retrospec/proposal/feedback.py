# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


class RetroSpecFeedbackMixin:
    def remove_requests(self, request_ids: Collection[str]) -> None:
        request_ids = tuple(request_ids)
        self.index_update_state.remove_requests(request_ids)
        self.sparse_attention.remove_requests(request_ids)
        for request_id in request_ids:
            self._feedback_horizons.pop(request_id, None)
            self._feedback_recovery.pop(request_id, None)
            slot = self._feedback_request_slots.pop(request_id, None)
            if slot is not None:
                self._feedback_free_slots.append(slot)

    def record_previous_proposal_outcomes(
        self,
        previous_request_ids: Sequence[str],
        valid_sampled_token_counts: Sequence[int],
        finished_request_ids: Collection[str],
    ) -> None:
        if not self.performance_stats.enabled:
            return

        finished_request_ids = set(finished_request_ids)
        for request_id, valid_count in zip(
            previous_request_ids, valid_sampled_token_counts
        ):
            proposed_count = self._last_proposed_counts.get(request_id)
            if proposed_count is None:
                continue
            self._last_committed_proposal_counts[request_id] = min(
                proposed_count, max(int(valid_count) - 1, 0)
            )

        for request_id in finished_request_ids:
            proposed_count = self._last_proposed_counts.pop(request_id, None)
            committed_count = self._last_committed_proposal_counts.pop(request_id, None)
            if proposed_count is None or committed_count is None:
                continue
            self.performance_stats.add_counter(
                "terminal_proposal_tokens", proposed_count
            )
            self.performance_stats.add_counter(
                "terminal_committed_proposal_tokens", committed_count
            )
            self.performance_stats.add_counter(
                "terminal_wasted_proposal_tokens", proposed_count - committed_count
            )

        for request_id in finished_request_ids:
            self._seen_proposal_request_ids.discard(request_id)
        if finished_request_ids:
            self.performance_stats.flush("request_finished")

    def record_sampled_proposal_outcomes(
        self,
        request_ids: Sequence[str],
        valid_sampled_token_counts: Sequence[int],
    ) -> None:
        if not self.performance_stats.enabled:
            return
        for request_id, valid_count in zip(request_ids, valid_sampled_token_counts):
            proposed_count = self._last_proposed_counts.get(request_id)
            if proposed_count is None:
                continue
            self._last_committed_proposal_counts[request_id] = min(
                proposed_count, max(int(valid_count) - 1, 0)
            )

    def record_verified_proposal_outcomes(
        self,
        proposed_counts: Sequence[int],
        valid_sampled_token_counts: Sequence[int],
        request_ids: Sequence[str] | None = None,
    ) -> None:
        """Count outcomes using the proposal consumed by this target pass.

        A new proposal may already have replaced ``_last_proposed_counts`` by
        the time synchronous output bookkeeping runs, so that mapping cannot
        identify the proposal whose tokens were just verified.
        """
        if len(proposed_counts) != len(valid_sampled_token_counts):
            raise ValueError("Proposal and sampled-token counts must have equal length")
        if request_ids is not None and len(request_ids) != len(proposed_counts):
            raise ValueError("Request IDs and proposal counts must have equal length")

        rounds = proposed = accepted = fully_accepted = 0
        for row, (proposed_count, valid_count) in enumerate(
            zip(proposed_counts, valid_sampled_token_counts)
        ):
            if proposed_count <= 0:
                continue
            committed_count = min(proposed_count, max(int(valid_count) - 1, 0))
            if self._feedback_horizon_enabled and request_ids is not None:
                self._update_feedback_horizon(
                    request_ids[row], proposed_count, committed_count
                )
            rounds += 1
            proposed += proposed_count
            accepted += committed_count
            fully_accepted += committed_count == proposed_count

        if not self.performance_stats.enabled:
            return
        self.performance_stats.add_counter("proposal_verified_rounds", rounds)
        self.performance_stats.add_counter("proposal_verified_tokens", proposed)
        self.performance_stats.add_counter("proposal_accepted_tokens", accepted)
        self.performance_stats.add_counter(
            "proposal_rejected_tokens", proposed - accepted
        )
        self.performance_stats.add_counter("proposal_fully_accepted", fully_accepted)

    def _update_feedback_horizon(
        self, request_id: str, proposed_count: int, accepted_count: int
    ) -> None:
        if (
            proposed_count >= self._LOW_ACCEPTANCE_MIN_PROPOSAL
            and accepted_count * 5 < proposed_count * 2
        ):
            if request_id not in self._feedback_horizons:
                self.performance_stats.add_counter("feedback_horizon_reductions")
            self._feedback_horizons[request_id] = self._LOW_ACCEPTANCE_HORIZON
            self._feedback_recovery[request_id] = 0
        elif request_id in self._feedback_horizons and proposed_count >= 8:
            if accepted_count == proposed_count:
                recovered = self._feedback_recovery.get(request_id, 0) + 1
                if recovered >= self._LOW_ACCEPTANCE_RECOVERY_ROUNDS:
                    del self._feedback_horizons[request_id]
                    self._feedback_recovery.pop(request_id, None)
                    self.performance_stats.add_counter("feedback_horizon_restores")
                else:
                    self._feedback_recovery[request_id] = recovered
            else:
                self._feedback_recovery[request_id] = 0

    def _feedback_slots_for_requests(self, request_ids: Sequence[str]) -> torch.Tensor:
        slots = []
        for request_id in request_ids:
            slot = self._feedback_request_slots.get(request_id)
            if slot is None:
                if not self._feedback_free_slots:
                    raise RuntimeError("RetroSpec feedback request slots are exhausted")
                slot = self._feedback_free_slots.pop()
                self._feedback_request_slots[request_id] = slot
                self._feedback_device_horizons[slot] = self.num_speculative_tokens
                self._feedback_device_recovery[slot] = 0
            slots.append(slot)
        self._feedback_slot_ids.np[: len(slots)] = slots
        return self._feedback_slot_ids.copy_to_gpu(len(slots))

    def _update_device_feedback_horizon(
        self,
        request_ids: Sequence[str],
        previous_proposed_counts: Sequence[int] | None,
        valid_sampled_token_counts: torch.Tensor,
    ) -> None:
        if not self._feedback_device_enabled or previous_proposed_counts is None:
            return
        if len(previous_proposed_counts) != len(request_ids):
            raise ValueError("Previous proposal counts must match request IDs")
        batch_size = len(request_ids)
        slots = self._feedback_slots_for_requests(request_ids)
        self._feedback_prior_counts.np[:batch_size] = previous_proposed_counts
        proposed = self._feedback_prior_counts.copy_to_gpu(batch_size)
        accepted = torch.minimum(
            proposed, (valid_sampled_token_counts - 1).clamp(min=0)
        )
        horizons = self._feedback_device_horizons.index_select(0, slots)
        recovery = self._feedback_device_recovery.index_select(0, slots)
        reduced = (proposed >= self._LOW_ACCEPTANCE_MIN_PROPOSAL) & (
            accepted * 5 < proposed * 2
        )
        low_horizon = horizons == self._LOW_ACCEPTANCE_HORIZON
        eligible_recovery = low_horizon & (proposed >= 8)
        fully_accepted = eligible_recovery & (accepted == proposed)
        next_recovery = torch.where(
            reduced,
            0,
            torch.where(
                eligible_recovery,
                torch.where(fully_accepted, recovery + 1, 0),
                recovery,
            ),
        )
        restored = fully_accepted & (
            next_recovery >= self._LOW_ACCEPTANCE_RECOVERY_ROUNDS
        )
        next_horizons = torch.where(
            reduced,
            self._LOW_ACCEPTANCE_HORIZON,
            torch.where(restored, self.num_speculative_tokens, horizons),
        )
        next_recovery = torch.where(restored, 0, next_recovery)
        self._feedback_device_horizons.index_copy_(0, slots, next_horizons)
        self._feedback_device_recovery.index_copy_(0, slots, next_recovery)
        self.performance_stats.add_gpu_counter(
            "feedback_horizon_reductions", reduced & ~low_horizon
        )
        self.performance_stats.add_gpu_counter("feedback_horizon_restores", restored)
