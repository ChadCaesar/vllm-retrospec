# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.retrospec import RetroSpecProposer
from vllm.v1.worker.retrospec_runner_state import RetroSpecPipelineProposalState

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class RetroSpecRunnerPipelineMixin:
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
            previous_proposed_counts=(
                state.spec_decode_metadata.num_draft_tokens
                if state.spec_decode_metadata is not None
                else None
            ),
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            materialize_output=get_pp_group().is_last_rank,
        )
