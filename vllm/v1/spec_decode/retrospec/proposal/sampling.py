# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch

from vllm.v1.sample.metadata import SamplingMetadata


class RetroSpecSamplingMixin:
    def _compact_mask_indices(
        self,
        mask: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compact true positions into a fixed-capacity output buffer."""
        if mask.ndim != 1 or mask.dtype != torch.bool:
            raise ValueError("Compaction mask must be one-dimensional and boolean")
        workspace_device = self._verification_pair_mask.device
        if mask.device != workspace_device:
            raise ValueError("Compaction mask must be on the model device")
        if output.ndim != 1 or output.dtype != torch.int64:
            raise ValueError("Compaction output must be one-dimensional int64")
        if output.device != workspace_device:
            raise ValueError("Compaction output must be on the model device")

        capacity = mask.shape[0]
        if capacity > self.max_parallel_tokens or output.shape[0] < capacity:
            raise ValueError("Compaction exceeds the verification workspace capacity")
        if capacity == 0:
            return output[:0]

        flat_indices = self._verification_flat_indices[:capacity]
        prefix = self._verification_prefix[:capacity]
        destinations = self._verification_destinations[:capacity]
        valid_destinations = self._verification_valid_destinations[:capacity]
        torch.cumsum(mask, dim=0, dtype=torch.int64, out=prefix)

        # The model metadata still needs a host-visible valid-prefix length.
        num_selected = int(prefix[-1].item())

        # Valid rows occupy the compact prefix. Invalid rows are assigned the
        # remaining destinations, making destinations a conflict-free
        # permutation for scatter_.
        valid_destinations.copy_(prefix)
        valid_destinations.sub_(1)
        destinations.copy_(flat_indices)
        destinations.sub_(prefix)
        destinations.add_(num_selected)
        torch.where(mask, valid_destinations, destinations, out=destinations)

        compacted = output[:capacity]
        compacted.scatter_(0, destinations, flat_indices)
        return compacted[:num_selected]

    def _build_verification_pairs(
        self,
        batch_size: int,
        round_start_counts: torch.Tensor,
        draft_counts: torch.Tensor,
        verification_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact active request-token pairs into reusable GPU buffers."""
        if round_start_counts.shape != (batch_size,):
            raise ValueError("round_start_counts must match the verification batch")
        if draft_counts.shape != (batch_size,):
            raise ValueError("draft_counts must match the verification batch")
        if verification_active.shape != (batch_size,):
            raise ValueError("verification_active must match the verification batch")

        capacity = batch_size * self.num_speculative_tokens
        pair_mask = self._verification_pair_mask[:capacity]
        pair_mask_2d = pair_mask.view(batch_size, self.num_speculative_tokens)
        torch.lt(
            self.verification_token_offsets.unsqueeze(0),
            draft_counts.unsqueeze(1),
            out=pair_mask_2d,
        )
        pair_mask_2d.logical_and_(verification_active.unsqueeze(1))

        source_indices = self._compact_mask_indices(
            pair_mask,
            self._verification_compact_indices,
        )
        num_pairs = source_indices.shape[0]
        request_indices = self._verification_request_indices[:num_pairs]
        token_indices = self._verification_token_indices[:num_pairs]

        torch.div(
            source_indices,
            self.num_speculative_tokens,
            rounding_mode="floor",
            out=request_indices,
        )
        torch.remainder(
            source_indices,
            self.num_speculative_tokens,
            out=token_indices,
        )

        round_starts = self._verification_round_starts[:num_pairs]
        torch.index_select(
            round_start_counts,
            0,
            request_indices,
            out=round_starts,
        )
        token_indices.add_(round_starts)
        return request_indices, token_indices

    def _append_sparse_bonus_pairs(
        self,
        batch_size: int,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        round_start_counts: torch.Tensor,
        draft_counts: torch.Tensor,
        verification_active: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append one unverified next-token query per eligible request."""
        empty_requests = self._verification_bonus_requests[:0]
        num_pairs = request_indices.numel()
        if (
            not self._sparse_bonus_enabled
            or self.policy.max_draft_tokens <= 1
            or self.policy.draft_margin_threshold is not None
            or self.policy.hit_attn_threshold is not None
            or not self._can_use_raw_greedy_verification_sampling(sampling_metadata)
            or num_pairs >= self.max_parallel_tokens
        ):
            return request_indices, token_indices, empty_requests

        next_steps = round_start_counts + draft_counts
        next_positions = self.proposal_start_positions[:batch_size] + next_steps + 1
        candidates = self._verification_bonus_candidates[:batch_size]
        candidates.copy_(verification_active)
        candidates.logical_and_(next_steps + 1 < self.policy.pending_limit)
        candidates.logical_and_(
            next_steps + 1 < self._proposal_token_budgets.gpu[:batch_size]
        )
        candidates.logical_and_(next_positions < self.max_model_len - 1)
        candidates.logical_and_(
            next_positions < self.index_update_state.next_update_positions
        )
        bonus_requests = self._compact_mask_indices(
            candidates, self._verification_bonus_requests
        )
        bonus_requests = bonus_requests[: self.max_parallel_tokens - num_pairs]
        num_bonus = bonus_requests.numel()
        if num_bonus == 0:
            return request_indices, token_indices, empty_requests

        combined_requests = self._verification_request_indices[: num_pairs + num_bonus]
        combined_tokens = self._verification_token_indices[: num_pairs + num_bonus]
        combined_requests[num_pairs:].copy_(bonus_requests)
        combined_tokens[num_pairs:].copy_(next_steps.index_select(0, bonus_requests))
        return combined_requests, combined_tokens, bonus_requests

    def _find_first_boundary_indices(
        self,
        batch_size: int,
        request_indices: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return each request's first boundary in the flattened pair array."""
        if request_indices.shape != boundary_mask.shape:
            raise ValueError("request_indices and boundary_mask must have equal shapes")
        if boundary_mask.dtype != torch.bool:
            raise ValueError("boundary_mask must use boolean dtype")

        num_pairs = boundary_mask.shape[0]
        flat_indices = self._verification_flat_indices[:num_pairs]
        boundary_candidates = self._verification_boundary_candidates[:num_pairs]
        first_boundary_indices = self._verification_first_boundaries[:batch_size]
        boundary_candidates.copy_(flat_indices)
        boundary_candidates.masked_fill_(~boundary_mask, num_pairs)
        first_boundary_indices.fill_(num_pairs)
        first_boundary_indices.scatter_reduce_(
            0,
            request_indices,
            boundary_candidates,
            reduce="amin",
            include_self=True,
        )
        return first_boundary_indices

    def _get_verification_step_logits(
        self,
        logits: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        workspace = self._verification_step_logits
        expected_shape = (self.max_batch_size, logits.shape[-1])
        if (
            workspace is None
            or workspace.shape != expected_shape
            or workspace.dtype != logits.dtype
            or workspace.device != logits.device
        ):
            workspace = torch.empty(
                expected_shape,
                dtype=logits.dtype,
                device=logits.device,
            )
            self._verification_step_logits = workspace
        return workspace[:batch_size]

    @staticmethod
    def _can_use_raw_greedy_verification_sampling(
        sampling_metadata: SamplingMetadata,
    ) -> bool:
        return (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
            and not sampling_metadata.logitsprocs.non_argmax_invariant
        )

    @staticmethod
    def _build_parallel_sampling_metadata(
        sampling_metadata: SamplingMetadata,
        request_indices: torch.Tensor,
    ) -> SamplingMetadata:
        """Expand request-row sampling state to verification-pair rows."""
        if not sampling_metadata.all_greedy:
            raise ValueError("RetroSpec parallel sampling requires greedy requests")
        if sampling_metadata.logitsprocs.non_argmax_invariant:
            raise ValueError(
                "Row-indexed logits processors require position-wise sampling"
            )

        def select_rows(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.index_select(0, request_indices)

        needs_output_history = not sampling_metadata.no_penalties or bool(
            sampling_metadata.bad_words_token_ids
        )
        if needs_output_history:
            request_rows = request_indices.detach().cpu().tolist()
            output_token_ids = [
                sampling_metadata.output_token_ids[request_index]
                for request_index in request_rows
            ]
            bad_words_token_ids = {
                pair_index: bad_words
                for pair_index, request_index in enumerate(request_rows)
                if (
                    bad_words := sampling_metadata.bad_words_token_ids.get(
                        request_index
                    )
                )
            }
        else:
            output_token_ids = []
            bad_words_token_ids = {}

        return replace(
            sampling_metadata,
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            prompt_token_ids=select_rows(sampling_metadata.prompt_token_ids),
            frequency_penalties=sampling_metadata.frequency_penalties.index_select(
                0, request_indices
            ),
            presence_penalties=sampling_metadata.presence_penalties.index_select(
                0, request_indices
            ),
            repetition_penalties=sampling_metadata.repetition_penalties.index_select(
                0, request_indices
            ),
            output_token_ids=output_token_ids,
            allowed_token_ids_mask=select_rows(
                sampling_metadata.allowed_token_ids_mask
            ),
            bad_words_token_ids=bad_words_token_ids,
            spec_token_ids=None,
        )

    def _sample_parallel_logits_by_position(
        self,
        batch_size: int,
        logits: torch.Tensor,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        sampled_token_ids: torch.Tensor,
    ) -> int:
        """Sample with processors whose state is keyed by original batch row."""
        sampler_calls = 0
        for token_index in range(self.num_speculative_tokens):
            token_mask = self._verification_pair_mask[: token_indices.shape[0]]
            torch.eq(token_indices, token_index, out=token_mask)
            flat_indices = self._compact_mask_indices(
                token_mask,
                self._verification_compact_indices,
            )
            if flat_indices.numel() == 0:
                continue

            request_rows = request_indices.index_select(0, flat_indices)
            step_logits = self._get_verification_step_logits(logits, batch_size)
            step_logits.zero_()
            step_logits.index_copy_(
                0, request_rows, logits.index_select(0, flat_indices)
            )

            sampler_output = self.runner.sampler(
                logits=step_logits, sampling_metadata=sampling_metadata
            )
            step_token_ids = sampler_output.sampled_token_ids.view(-1).to(torch.int32)
            sampled_token_ids.index_copy_(
                0, flat_indices, step_token_ids.index_select(0, request_rows)
            )
            sampler_calls += 1
        return sampler_calls

    def _sample_parallel_logits(
        self,
        batch_size: int,
        logits: torch.Tensor,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if output.shape[0] < logits.shape[0]:
            raise ValueError("Sample output exceeds its verification workspace")
        if not sampling_metadata.all_greedy:
            raise ValueError("RetroSpec parallel sampling requires greedy requests")

        num_tokens = logits.shape[0]
        sampled_token_ids = output[:num_tokens]

        if self._can_use_raw_greedy_verification_sampling(sampling_metadata):
            argmax_token_ids = self._verification_argmax_token_ids[:num_tokens]
            torch.argmax(logits, dim=-1, out=argmax_token_ids)
            sampled_token_ids.copy_(argmax_token_ids)
            self.performance_stats.add_counter(
                "verification_sampling_argmax_tokens", num_tokens
            )
            self.performance_stats.add_counter("verification_sampling_launches")
            return sampled_token_ids

        if not sampling_metadata.logitsprocs.non_argmax_invariant:
            pair_metadata = self._build_parallel_sampling_metadata(
                sampling_metadata, request_indices
            )
            sampler_output = self.runner.sampler(
                logits=logits, sampling_metadata=pair_metadata
            )
            sampled_token_ids.copy_(sampler_output.sampled_token_ids.view(-1))
            self.performance_stats.add_counter(
                "verification_sampling_batched_tokens", num_tokens
            )
            self.performance_stats.add_counter("verification_sampling_launches")
            return sampled_token_ids

        sampled_token_ids.fill_(-1)
        sampler_calls = self._sample_parallel_logits_by_position(
            batch_size=batch_size,
            logits=logits,
            request_indices=request_indices,
            token_indices=token_indices,
            sampling_metadata=sampling_metadata,
            sampled_token_ids=sampled_token_ids,
        )
        self.performance_stats.add_counter(
            "verification_sampling_fallback_tokens", num_tokens
        )
        self.performance_stats.add_counter(
            "verification_sampling_launches", sampler_calls
        )
        return sampled_token_ids
