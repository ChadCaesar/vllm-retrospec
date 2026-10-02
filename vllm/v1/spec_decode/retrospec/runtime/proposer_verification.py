# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import replace

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.retrospec.decision import RetroSpecDecision, RetroSpecMetrics
from vllm.v1.spec_decode.retrospec.runtime.attention_types import RetroSpecAttentionMode
from vllm.v1.spec_decode.retrospec.runtime.proposer_types import (
    RetroSpecParallelVerificationOutput,
    RetroSpecVerificationResult,
)
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


class RetroSpecVerificationMixin:
    """RetroSpec Verification helpers."""

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

    def _run_parallel_verification(
        self,
        batch_size: int,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        attention_mode: RetroSpecAttentionMode,
    ) -> RetroSpecParallelVerificationOutput:
        assert self.model is not None
        if attention_mode not in (
            RetroSpecAttentionMode.SPARSE_VERIFY,
            RetroSpecAttentionMode.EXPANDED_VERIFY,
        ):
            raise ValueError(
                "Verification requires SPARSE_VERIFY or EXPANDED_VERIFY mode."
            )
        if request_indices.ndim != 1 or token_indices.ndim != 1:
            raise ValueError("Parallel verification indices must be one-dimensional")
        if request_indices.shape != token_indices.shape:
            raise ValueError("request_indices and token_indices must have equal shapes")
        if request_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("request_indices must use an integer dtype")
        if token_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_indices must use an integer dtype")
        if request_indices.device != self.device or token_indices.device != self.device:
            raise ValueError(
                "Parallel verification indices must be on the model device"
            )

        num_tokens = request_indices.shape[0]
        if num_tokens == 0:
            raise ValueError("Parallel verification requires at least one token")
        if num_tokens > self.max_parallel_tokens:
            raise ValueError("Parallel verification exceeds the configured capacity")

        stage_name = (
            "sparse_verify"
            if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
            else "expanded_verify"
        )
        model_timer = self.performance_stats.start_cuda_timer(f"{stage_name}_model")

        with self.performance_stats.cuda_timer(f"{stage_name}_prepare"):
            request_indices = request_indices.to(torch.int64)
            token_indices = token_indices.to(torch.int64)
            positions = (
                self.proposal_start_positions.index_select(0, request_indices)
                + token_indices
            )

            previous_token_indices = (token_indices - 1).clamp_min(0)
            previous_token_ids = self._draft_token_ids[
                request_indices, previous_token_indices
            ]
            initial_token_ids = self.proposal_input_ids.index_select(0, request_indices)
            input_ids = torch.where(
                token_indices == 0, initial_token_ids, previous_token_ids
            )

            block_table = common_attn_metadata.block_table_tensor.index_select(
                0, request_indices
            )
            block_numbers = positions // self.block_size
            block_ids = block_table.gather(1, block_numbers.view(-1, 1)).view(-1)

            slot_mapping = self._verification_slot_mapping[:num_tokens]
            slot_mapping.copy_(
                block_ids * self.block_size + positions % self.block_size
            )
            seq_lens = (positions + 1).to(dtype=common_attn_metadata.seq_lens.dtype)
            query_start_loc_cpu = torch.from_numpy(
                self.parallel_token_arange_np[: num_tokens + 1]
            ).clone()

            parallel_common_attn_metadata = common_attn_metadata.replace(
                query_start_loc=self.parallel_arange[: num_tokens + 1],
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                num_reqs=num_tokens,
                num_actual_tokens=num_tokens,
                max_query_len=1,
                max_seq_len=min(
                    common_attn_metadata.max_seq_len + self.num_speculative_tokens,
                    self.max_model_len,
                ),
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                dcp_local_seq_lens=None,
                dcp_local_seq_lens_cpu=None,
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                _num_computed_tokens_cache=None,
            )

            builder = self._get_attention_metadata_builder()
            attn_metadata = builder.build_for_drafting(
                parallel_common_attn_metadata, draft_index=0
            )
            per_layer_attn_metadata = {
                layer_name: attn_metadata for layer_name in self.attn_layer_names
            }
            (
                model_input_ids,
                model_positions,
                forward_slot_mapping,
                cudagraph_mode,
                batch_descriptor,
            ) = self._prepare_piecewise_model_inputs(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                slot_mapping_workspace=self._verification_slot_mapping,
                stage_name=stage_name,
            )
            per_layer_slot_mapping = {
                layer_name: forward_slot_mapping for layer_name in self.attn_layer_names
            }

            self.sparse_attention.begin_parallel_step(
                attention_mode, request_indices, token_indices
            )

        try:
            with (
                self.performance_stats.cuda_timer(f"{stage_name}_forward"),
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=batch_descriptor.num_tokens,
                    cudagraph_runtime_mode=cudagraph_mode,
                    batch_descriptor=(
                        batch_descriptor
                        if cudagraph_mode != CUDAGraphMode.NONE
                        else None
                    ),
                    slot_mapping=per_layer_slot_mapping,
                ),
            ):
                hidden_states = self._run_pipeline_stage_model(
                    model_input_ids,
                    model_positions,
                    batch_descriptor.num_tokens,
                    cudagraph_mode,
                    stage_name,
                )

            with self.performance_stats.cuda_timer(f"{stage_name}_end_step"):
                local_attention_stats = self.sparse_attention.end_step_statistics()
                attention_mass = self.pipeline_protocol.reduce_attention_mass(
                    local_attention_stats
                )
        except BaseException:
            self.sparse_attention.abort_step()
            raise

        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            compute_margin = self.policy.sparse_margin_threshold is not None
            sampled_output = self._sparse_sampled_token_ids
        else:
            compute_margin = self.policy.expanded_margin_threshold is not None
            sampled_output = self._expanded_sampled_token_ids

        stage = self._require_pipeline_stage()
        token_ids = None
        margin = None
        if stage.is_last:
            assert hidden_states is not None
            with self.performance_stats.cuda_timer(f"{stage_name}_logits"):
                logits = self.model.compute_logits(hidden_states[:num_tokens])
                if logits is None:
                    raise RuntimeError(
                        "The final RetroSpec PP stage did not produce logits"
                    )

                if compute_margin:
                    top2_logits = torch.topk(logits.float(), k=2, dim=-1).values
                    margin = top2_logits[:, 0] - top2_logits[:, 1]

            with self.performance_stats.cuda_timer(f"{stage_name}_sampling"):
                token_ids = self._sample_parallel_logits(
                    batch_size,
                    logits,
                    request_indices,
                    token_indices,
                    sampling_metadata,
                    sampled_output,
                )

        pipeline_output = self.pipeline_protocol.broadcast_model_output(
            num_tokens=num_tokens,
            token_ids=token_ids,
            margin=margin,
            compute_margin=compute_margin,
        )
        self.performance_stats.stop_cuda_timer(model_timer)
        return RetroSpecParallelVerificationOutput(
            request_indices=request_indices,
            token_indices=token_indices,
            token_ids=pipeline_output.token_ids,
            margin=pipeline_output.margin,
            attention_mass=attention_mass,
        )

    def _trace_draft_transition(
        self,
        request_ids: Sequence[str],
        proposal_round: int,
        transition_mask: torch.Tensor,
        decision: RetroSpecDecision,
        draft_margin: torch.Tensor | None,
        hit_attn: torch.Tensor,
        projected_pending_counts: torch.Tensor,
        generation_limit_reached: torch.Tensor,
        index_update_required: torch.Tensor,
    ) -> None:
        if not self.transition_tracer.enabled:
            return

        batch_size = len(request_ids)
        self.transition_tracer.record_masked(
            request_ids=request_ids,
            phase="draft_to_sparse",
            proposal_round=proposal_round,
            mask=transition_mask,
            integer_fields={
                "position": self.positions[:batch_size] + 1,
                "request_stage": self.state.stage,
                "next_stage": decision.next_stage,
                "draft_count": self.state.draft_counts,
                "pending_count": projected_pending_counts,
                "stop_draft": decision.stop_draft,
                "require_expanded": decision.require_expanded,
                "require_full": decision.require_full,
                "reasons": decision.reasons,
                "pending_limit_reached": (
                    projected_pending_counts >= self.policy.pending_limit
                ),
                "generation_limit_reached": generation_limit_reached,
                "index_update_required": index_update_required,
            },
            float_fields={
                "draft_margin": draft_margin,
                "hit_attention": hit_attn,
            },
        )

    def _trace_sparse_transitions(
        self,
        request_ids: Sequence[str],
        proposal_round: int,
        verification_active: torch.Tensor,
        round_start_counts: torch.Tensor,
        sparse: RetroSpecParallelVerificationOutput,
        expected_token_ids: torch.Tensor,
        sparse_token_changed: torch.Tensor,
        sparse_decision: RetroSpecDecision,
        candidate_positions: torch.Tensor,
        candidate_pending_counts: torch.Tensor,
        pair_draft_counts: torch.Tensor,
        pair_stages: torch.Tensor,
        generation_limit_reached: torch.Tensor,
        index_update_required: torch.Tensor,
        boundary_request_mask: torch.Tensor,
        safe_boundary_indices: torch.Tensor,
        verified_counts: torch.Tensor,
    ) -> None:
        if not self.transition_tracer.enabled:
            return

        boundary_request_indices = torch.nonzero(
            boundary_request_mask, as_tuple=False
        ).flatten()
        if boundary_request_indices.numel() > 0:
            boundary_pair_indices = safe_boundary_indices.index_select(
                0, boundary_request_indices
            )
            self.transition_tracer.record_compact(
                request_ids=request_ids,
                phase="sparse_boundary",
                proposal_round=proposal_round,
                request_indices=boundary_request_indices,
                integer_fields={
                    "position": candidate_positions.index_select(
                        0, boundary_pair_indices
                    ),
                    "request_stage": pair_stages.index_select(0, boundary_pair_indices),
                    "next_stage": sparse_decision.next_stage.index_select(
                        0, boundary_pair_indices
                    ),
                    "token_index": sparse.token_indices.index_select(
                        0, boundary_pair_indices
                    ),
                    "draft_token_id": expected_token_ids.index_select(
                        0, boundary_pair_indices
                    ),
                    "sparse_token_id": sparse.token_ids.index_select(
                        0, boundary_pair_indices
                    ),
                    "draft_count": pair_draft_counts.index_select(
                        0, boundary_pair_indices
                    ),
                    "pending_count": candidate_pending_counts.index_select(
                        0, boundary_pair_indices
                    ),
                    "verified_count": verified_counts.index_select(
                        0, boundary_request_indices
                    ),
                    "stop_draft": sparse_decision.stop_draft.index_select(
                        0, boundary_pair_indices
                    ),
                    "require_expanded": sparse_decision.require_expanded.index_select(
                        0, boundary_pair_indices
                    ),
                    "require_full": sparse_decision.require_full.index_select(
                        0, boundary_pair_indices
                    ),
                    "reasons": sparse_decision.reasons.index_select(
                        0, boundary_pair_indices
                    ),
                    "token_changed": sparse_token_changed.index_select(
                        0, boundary_pair_indices
                    ),
                    "pending_limit_reached": (
                        candidate_pending_counts.index_select(0, boundary_pair_indices)
                        >= self.policy.pending_limit
                    ),
                    "generation_limit_reached": (
                        generation_limit_reached.index_select(0, boundary_pair_indices)
                    ),
                    "index_update_required": index_update_required.index_select(
                        0, boundary_pair_indices
                    ),
                },
                float_fields={
                    "sparse_margin": (
                        None
                        if sparse.margin is None
                        else sparse.margin.index_select(0, boundary_pair_indices)
                    ),
                    "retrieval_attention": sparse.attention_mass.index_select(
                        0, boundary_pair_indices
                    ),
                },
            )

        no_boundary_mask = verification_active & ~boundary_request_mask
        batch_size = len(request_ids)
        no_boundary_pending_counts = round_start_counts + verified_counts
        sparse_stages = torch.full_like(
            self.state.stage, int(RetroSpecStage.SPARSE_VERIFY)
        )
        draft_stages = torch.full_like(self.state.stage, int(RetroSpecStage.DRAFT))
        false_values = torch.zeros_like(no_boundary_mask)
        no_reasons = torch.zeros_like(self.state.draft_counts, dtype=torch.int32)

        self.transition_tracer.record_masked(
            request_ids=request_ids,
            phase="sparse_complete",
            proposal_round=proposal_round,
            mask=no_boundary_mask,
            integer_fields={
                "position": (
                    self.proposal_start_positions[:batch_size]
                    + no_boundary_pending_counts
                ),
                "request_stage": sparse_stages,
                "next_stage": draft_stages,
                "draft_count": self.state.draft_counts,
                "pending_count": no_boundary_pending_counts,
                "verified_count": verified_counts,
                "stop_draft": false_values,
                "require_expanded": false_values,
                "require_full": false_values,
                "reasons": no_reasons,
                "token_changed": false_values,
                "pending_limit_reached": false_values,
                "generation_limit_reached": false_values,
                "index_update_required": false_values,
            },
        )

    def _trace_expanded_transition(
        self,
        request_ids: Sequence[str],
        proposal_round: int,
        expanded: RetroSpecParallelVerificationOutput,
        sparse_token_ids: torch.Tensor,
        expanded_token_changed: torch.Tensor,
        expanded_decision: RetroSpecDecision,
        expanded_positions: torch.Tensor,
        expanded_pending_counts: torch.Tensor,
        expanded_draft_counts: torch.Tensor,
        expanded_stages: torch.Tensor,
        expanded_generation_limit: torch.Tensor,
        expanded_index_update: torch.Tensor,
        verified_counts: torch.Tensor,
    ) -> None:
        if not self.transition_tracer.enabled:
            return

        self.transition_tracer.record_compact(
            request_ids=request_ids,
            phase="expanded_boundary",
            proposal_round=proposal_round,
            request_indices=expanded.request_indices,
            integer_fields={
                "position": expanded_positions,
                "request_stage": expanded_stages,
                "next_stage": expanded_decision.next_stage,
                "token_index": expanded.token_indices,
                "sparse_token_id": sparse_token_ids,
                "expanded_token_id": expanded.token_ids,
                "draft_count": expanded_draft_counts,
                "pending_count": expanded_pending_counts,
                "verified_count": verified_counts.index_select(
                    0, expanded.request_indices
                ),
                "stop_draft": expanded_decision.stop_draft,
                "require_expanded": expanded_decision.require_expanded,
                "require_full": expanded_decision.require_full,
                "reasons": expanded_decision.reasons,
                "token_changed": expanded_token_changed,
                "pending_limit_reached": (
                    expanded_pending_counts >= self.policy.pending_limit
                ),
                "generation_limit_reached": expanded_generation_limit,
                "index_update_required": expanded_index_update,
            },
            float_fields={
                "expanded_margin": expanded.margin,
                "expanded_attention": expanded.attention_mass,
            },
        )

    def _verify_draft_tokens(
        self,
        batch_size: int,
        request_ids: Sequence[str],
        proposal_round: int,
        round_start_counts: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> RetroSpecVerificationResult:
        with self.performance_stats.cpu_timer("draft_prefetch_backpressure"):
            self.sparse_attention.flush_sparse_verification_prefetch()

        draft_counts = self.state.draft_counts
        verification_active = self.state.active_mask & (draft_counts > 0)
        request_indices, token_indices = self._build_verification_pairs(
            batch_size,
            round_start_counts,
            draft_counts,
            verification_active,
        )
        self.performance_stats.add_counter(
            "sparse_verify_tokens",
            request_indices.numel(),
        )

        if request_indices.numel() == 0:
            return RetroSpecVerificationResult(
                verified_counts=self._verification_verified_counts[:batch_size].zero_(),
                require_full=self._verification_require_full[:batch_size].zero_(),
            )

        self.state.set_stage(verification_active, RetroSpecStage.SPARSE_VERIFY)
        with self.performance_stats.cpu_timer("full_verify_prime_submit"):
            self.sparse_attention.maybe_prime_full_verification(request_indices.numel())

        sparse = self._run_parallel_verification(
            batch_size,
            request_indices,
            token_indices,
            common_attn_metadata,
            sampling_metadata,
            RetroSpecAttentionMode.SPARSE_VERIFY,
        )
        sparse_boundary_timer = self.performance_stats.start_cuda_timer(
            "sparse_verify_boundary"
        )

        expected_token_ids = self._draft_token_ids[
            sparse.request_indices, sparse.token_indices
        ].clone()
        sparse_token_changed = sparse.token_ids != expected_token_ids
        candidate_pending_counts = (sparse.token_indices + 1).to(draft_counts.dtype)
        candidate_positions = (
            self.proposal_start_positions.index_select(0, sparse.request_indices)
            + candidate_pending_counts
        )
        pair_generation_token_budgets = self._proposal_token_budgets.gpu.index_select(
            0,
            sparse.request_indices,
        )
        generation_limit_reached = (candidate_positions >= self.max_model_len - 1) | (
            candidate_pending_counts >= pair_generation_token_budgets
        )
        next_update_positions = (
            self.index_update_state.next_update_positions.index_select(
                0, sparse.request_indices
            )
        )
        index_update_required = candidate_positions >= next_update_positions
        pair_draft_counts = draft_counts.index_select(0, sparse.request_indices)
        pair_stages = torch.full_like(
            pair_draft_counts, int(RetroSpecStage.SPARSE_VERIFY), dtype=torch.int8
        )

        sparse_decision = self.policy.evaluate(
            current_stage=RetroSpecStage.SPARSE_VERIFY,
            request_stages=pair_stages,
            metrics=RetroSpecMetrics(
                sparse_margin=sparse.margin,
                retrieval_attn=sparse.attention_mass,
            ),
            draft_counts=pair_draft_counts,
            pending_counts=candidate_pending_counts,
            sparse_token_changed=sparse_token_changed,
            generation_limit_reached=generation_limit_reached,
            index_update_required=index_update_required,
        )

        boundary_mask = (
            sparse_token_changed
            | sparse_decision.require_expanded
            | sparse_decision.require_full
        )
        first_boundary_indices = self._find_first_boundary_indices(
            batch_size,
            sparse.request_indices,
            boundary_mask,
        )

        num_pairs = sparse.request_indices.shape[0]
        flat_indices = self._verification_flat_indices[:num_pairs]
        request_boundary_indices = first_boundary_indices.index_select(
            0, sparse.request_indices
        )
        accepted_mask = flat_indices <= request_boundary_indices

        # Avoid compacting the accepted prefix. Accepted rows receive sparse
        # verification tokens, while the rejected suffix keeps its draft token.
        corrected_token_ids = torch.where(
            accepted_mask,
            sparse.token_ids,
            expected_token_ids,
        )
        self._draft_token_ids[sparse.request_indices, sparse.token_indices] = (
            corrected_token_ids
        )

        verified_counts = self._verification_verified_counts[:batch_size]
        verified_counts.zero_()
        verified_counts.scatter_add_(
            0,
            sparse.request_indices,
            accepted_mask.to(draft_counts.dtype),
        )

        # Requests without a boundary use num_pairs as a sentinel. Clamp it
        # before gathering, then mask the gathered value back out.
        boundary_request_mask = self._verification_boundary_mask[:batch_size]
        torch.lt(
            first_boundary_indices,
            num_pairs,
            out=boundary_request_mask,
        )

        safe_boundary_indices = self._verification_safe_boundaries[:batch_size]
        torch.clamp(
            first_boundary_indices,
            max=num_pairs - 1,
            out=safe_boundary_indices,
        )

        require_full = self._verification_require_full[:batch_size]
        torch.index_select(
            sparse_decision.require_full,
            0,
            safe_boundary_indices,
            out=require_full,
        )
        require_full.logical_and_(boundary_request_mask)

        run_expanded = self._verification_run_expanded[:batch_size]
        torch.index_select(
            sparse_decision.require_expanded,
            0,
            safe_boundary_indices,
            out=run_expanded,
        )
        run_expanded.logical_and_(boundary_request_mask)
        run_expanded.masked_fill_(require_full, False)
        self.performance_stats.add_gpu_histogram(
            "sparse_to_expanded_prefix", verified_counts, run_expanded
        )
        self._trace_sparse_transitions(
            request_ids=request_ids,
            proposal_round=proposal_round,
            verification_active=verification_active,
            round_start_counts=round_start_counts,
            sparse=sparse,
            expected_token_ids=expected_token_ids,
            sparse_token_changed=sparse_token_changed,
            sparse_decision=sparse_decision,
            candidate_positions=candidate_positions,
            candidate_pending_counts=candidate_pending_counts,
            pair_draft_counts=pair_draft_counts,
            pair_stages=pair_stages,
            generation_limit_reached=generation_limit_reached,
            index_update_required=index_update_required,
            boundary_request_mask=boundary_request_mask,
            safe_boundary_indices=safe_boundary_indices,
            verified_counts=verified_counts,
        )

        # Expanded verification changes the number of target-model rows, so
        # only this request-level subset still needs a host-visible length.
        expanded_request_indices = self._compact_mask_indices(
            run_expanded,
            self._verification_expanded_requests,
        )
        expanded_sparse_indices = self._verification_expanded_indices[
            : expanded_request_indices.shape[0]
        ]
        torch.index_select(
            safe_boundary_indices,
            0,
            expanded_request_indices,
            out=expanded_sparse_indices,
        )
        sparse_boundary_token_ids = self._verification_sparse_boundary_token_ids[
            : expanded_request_indices.shape[0]
        ]
        torch.index_select(
            sparse.token_ids,
            0,
            expanded_sparse_indices,
            out=sparse_boundary_token_ids,
        )
        self.performance_stats.stop_cuda_timer(sparse_boundary_timer)

        if expanded_sparse_indices.numel() > 0:
            expanded_token_indices = sparse.token_indices.index_select(
                0, expanded_sparse_indices
            )
            self.performance_stats.add_counter(
                "expanded_verify_tokens",
                expanded_request_indices.numel(),
            )
            self.state.set_stage(run_expanded, RetroSpecStage.EXPANDED_VERIFY)

            expanded = self._run_parallel_verification(
                batch_size,
                expanded_request_indices,
                expanded_token_indices,
                common_attn_metadata,
                sampling_metadata,
                RetroSpecAttentionMode.EXPANDED_VERIFY,
            )
            expanded_boundary_timer = self.performance_stats.start_cuda_timer(
                "expanded_verify_boundary"
            )
            expanded_token_changed = expanded.token_ids != sparse_boundary_token_ids
            expanded_pending_counts = (expanded.token_indices + 1).to(
                draft_counts.dtype
            )
            expanded_positions = (
                self.proposal_start_positions.index_select(0, expanded.request_indices)
                + expanded_pending_counts
            )
            expanded_generation_token_budgets = (
                self._proposal_token_budgets.gpu.index_select(
                    0,
                    expanded.request_indices,
                )
            )
            expanded_generation_limit = (
                expanded_positions >= self.max_model_len - 1
            ) | (expanded_pending_counts >= expanded_generation_token_budgets)
            expanded_index_update = expanded_positions >= (
                self.index_update_state.next_update_positions.index_select(
                    0, expanded.request_indices
                )
            )
            expanded_draft_counts = draft_counts.index_select(
                0, expanded.request_indices
            )
            expanded_stages = torch.full_like(
                expanded_draft_counts,
                int(RetroSpecStage.EXPANDED_VERIFY),
                dtype=torch.int8,
            )
            expanded_decision = self.policy.evaluate(
                current_stage=RetroSpecStage.EXPANDED_VERIFY,
                request_stages=expanded_stages,
                metrics=RetroSpecMetrics(
                    expanded_margin=expanded.margin,
                    expanded_attn=expanded.attention_mass,
                ),
                draft_counts=expanded_draft_counts,
                pending_counts=expanded_pending_counts,
                expanded_token_changed=expanded_token_changed,
                generation_limit_reached=expanded_generation_limit,
                index_update_required=expanded_index_update,
            )
            self._trace_expanded_transition(
                request_ids=request_ids,
                proposal_round=proposal_round,
                expanded=expanded,
                sparse_token_ids=sparse_boundary_token_ids,
                expanded_token_changed=expanded_token_changed,
                expanded_decision=expanded_decision,
                expanded_positions=expanded_positions,
                expanded_pending_counts=expanded_pending_counts,
                expanded_draft_counts=expanded_draft_counts,
                expanded_stages=expanded_stages,
                expanded_generation_limit=expanded_generation_limit,
                expanded_index_update=expanded_index_update,
                verified_counts=verified_counts,
            )
            self._draft_token_ids[expanded.request_indices, expanded.token_indices] = (
                expanded.token_ids
            )
            require_full.index_copy_(
                0,
                expanded.request_indices,
                expanded_decision.require_full,
            )
            self.performance_stats.add_gpu_histogram(
                "expanded_to_full_prefix",
                verified_counts,
                run_expanded & require_full,
            )
            self.performance_stats.stop_cuda_timer(expanded_boundary_timer)

        self.state.set_stage(verification_active, RetroSpecStage.DRAFT)
        self.state.set_stage(require_full, RetroSpecStage.FULL_VERIFY)

        return RetroSpecVerificationResult(verified_counts, require_full)
