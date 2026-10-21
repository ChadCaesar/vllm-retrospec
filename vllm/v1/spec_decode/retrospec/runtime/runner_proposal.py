# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch
import torch.distributed

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.retrospec import RetroSpecProposer


class RetroSpecPipelineProposalState(NamedTuple):
    """State retained by non-final PP ranks until sample_tokens()."""

    scheduler_output: "SchedulerOutput"
    spec_decode_metadata: SpecDecodeMetadata | None
    common_attn_metadata: CommonAttentionMetadata


RetroSpecPipelineProposalState.__module__ = "vllm.v1.worker.gpu_model_runner"


class RetroSpecProposalCoordinatorMixin:
    """RetroSpec runtime methods for RetroSpecProposalCoordinatorMixin."""

    def _append_retrospec_cpu_outputs(self, output_ids):
        drafter = getattr(self, "drafter", None)
        cpu = getattr(drafter, "cpu_coordinator", None)
        counts: dict[str, int] = {}
        if cpu is None:
            return counts
        for row, request_id in enumerate(self.input_batch.req_ids):
            committed = cpu.committed_tokens.pop(request_id, [])
            if not committed:
                continue
            start = self.input_batch.num_tokens_no_spec[row]
            end = start + len(committed)
            if end > self.max_model_len:
                raise RuntimeError("CPU target exceeded the model token budget")
            self.input_batch.token_ids_cpu[row, start:end] = committed
            self.input_batch.is_token_ids[row, start:end] = True
            self.input_batch.num_tokens_no_spec[row] = end
            self.requests[request_id].output_token_ids.extend(committed)
            output_ids[row].extend(committed)
            counts[request_id] = len(committed)
        return counts

    def _maybe_run_retrospec_cpu_proposal_only(self, scheduler_output, metadata):
        from vllm.v1.outputs import ModelRunnerOutput

        drafter = getattr(self, "drafter", None)
        cpu = getattr(drafter, "cpu_coordinator", None)
        if cpu is None or metadata is None or self.input_batch.num_reqs != 1:
            return None
        request_id = self.input_batch.req_ids[0]
        scheduled_drafts = scheduler_output.scheduled_spec_decode_tokens.get(
            request_id, []
        )
        carried = cpu.deferred_window
        if scheduled_drafts and (
            carried is None
            or carried[0] != request_id
            or carried[2] != scheduled_drafts
        ):
            return None
        if scheduler_output.total_num_scheduled_tokens != len(scheduled_drafts) + 1:
            return None
        request = self.requests[request_id]
        length = request.num_computed_tokens
        if cpu.request_id != request_id or cpu.authoritative_length != length:
            return None
        if length < request.num_prompt_tokens or not drafter._cpu_request_supported(
            [request_id], self.input_batch.sampling_metadata
        ):
            return None
        seed = request.get_token_id(length)
        if seed < 0:
            return None
        # The scheduler scheduled the last, already output token as an input.
        # Reuse that seed directly; it must not be emitted a second time.
        common = metadata.replace(
            seq_lens=metadata.seq_lens - (len(scheduled_drafts) + 1),
            max_seq_len=length,
            _seq_lens_cpu=torch.tensor([length], dtype=torch.int32),
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )
        budgets = scheduler_output.retrospec_generation_token_budgets
        self._draft_token_ids = drafter.propose(
            request_ids=[request_id],
            committed_positions=[length],
            next_token_ids=torch.tensor([seed], dtype=torch.int32, device=self.device),
            sampling_metadata=self.input_batch.sampling_metadata,
            common_attn_metadata=common,
            proposal_active_mask=torch.ones(1, dtype=torch.bool, device=self.device),
            remaining_generation_tokens=[budgets[request_id]],
            valid_sampled_tokens_count=torch.zeros(
                1, dtype=torch.int32, device=self.device
            ),
        )
        self._copy_draft_token_ids_to_cpu(scheduler_output)
        output_ids: list[list[int]] = [[]]
        counts = self._append_retrospec_cpu_outputs(output_ids)
        retirements = drafter.take_kv_cache_retirements((request_id,))
        self._apply_retrospec_kv_retirements(retirements)
        drafter.performance_stats.add_counter("cpu_skipped_full_verifications")
        self.input_batch.prev_sampled_token_ids = None
        return ModelRunnerOutput(
            req_ids=[request_id],
            req_id_to_index={request_id: 0},
            sampled_token_ids=output_ids,
            kv_cache_retirements=retirements,
            retrospec_cpu_committed_counts=counts,
            retrospec_computed_token_counts={
                request_id: length + counts.get(request_id, 0)
            },
        )

    def _save_retrospec_pipeline_proposal_state(
        self,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata | None,
    ) -> None:
        pp_group = get_pp_group()
        if (
            pp_group.world_size == 1
            or pp_group.is_last_rank
            or common_attn_metadata is None
        ):
            return
        if self.retrospec_pipeline_proposal_state is not None:
            raise RuntimeError("A RetroSpec PP proposal state is already pending")

        self.retrospec_pipeline_proposal_state = RetroSpecPipelineProposalState(
            scheduler_output=scheduler_output,
            spec_decode_metadata=spec_decode_metadata,
            common_attn_metadata=common_attn_metadata,
        )

    def _run_retrospec_pipeline_proposal(
        self,
        state: RetroSpecPipelineProposalState,
        sampled_token_ids: torch.Tensor | None,
    ) -> list[list[int]]:
        drafter = self.drafter
        if not isinstance(drafter, RetroSpecProposer):
            raise RuntimeError("RetroSpec pipeline proposal requires RetroSpecProposer")

        num_reqs = self.input_batch.num_reqs
        sampled_token_ids = (
            drafter.pipeline_protocol.broadcast_target_sampled_token_ids(
                num_reqs, sampled_token_ids
            )
        )

        partial_prefill_mask = self.discard_request_mask.np[:num_reqs]
        if bool(partial_prefill_mask.all()):
            return [[] for _ in range(num_reqs)]

        proposal_active_mask = ~self.discard_request_mask.gpu[:num_reqs]
        common_attn_metadata = state.common_attn_metadata
        next_token_ids, valid_sampled_tokens_count = (
            drafter.prepare_next_token_ids_padded(
                common_attn_metadata,
                sampled_token_ids,
                self.requests,
                self.input_batch,
                self.discard_request_mask.gpu,
            )
        )
        if get_pp_group().is_last_rank:
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

        num_rejected_tokens_gpu = None
        if state.spec_decode_metadata is not None:
            (
                common_attn_metadata,
                _,
                num_rejected_tokens_gpu,
            ) = drafter.prepare_inputs_padded(
                common_attn_metadata,
                state.spec_decode_metadata,
                valid_sampled_tokens_count,
            )

        request_ids = self.input_batch.req_ids
        generation_token_budgets = (
            state.scheduler_output.retrospec_generation_token_budgets
        )
        if generation_token_budgets is None:
            raise RuntimeError(
                "RetroSpec scheduler output is missing generation-token budgets"
            )
        remaining_generation_tokens = [
            generation_token_budgets.get(request_id, 0) for request_id in request_ids
        ]
        committed_positions = [
            self.requests[request_id].num_computed_tokens for request_id in request_ids
        ]
        return drafter.propose(
            request_ids=request_ids,
            committed_positions=committed_positions,
            next_token_ids=next_token_ids,
            sampling_metadata=self.input_batch.sampling_metadata,
            common_attn_metadata=common_attn_metadata,
            proposal_active_mask=proposal_active_mask,
            remaining_generation_tokens=remaining_generation_tokens,
            valid_sampled_tokens_count=valid_sampled_tokens_count,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            materialize_output=get_pp_group().is_last_rank,
        )
