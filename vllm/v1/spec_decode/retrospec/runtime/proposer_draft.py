# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.retrospec.attention import RetroSpecAttentionMode
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


class RetroSpecDraftMixin:
    """RetroSpec runtime methods for RetroSpecDraftMixin."""

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = gpu_input_batch.num_reqs
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = common_attn_metadata.seq_lens.cpu()

        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        next_token_ids = torch.empty(
            batch_size, dtype=torch.int32, device=sampled_token_ids.device
        )
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        block_size_tokens = triton.next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[(batch_size,)](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=block_size_tokens,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            num_reqs, dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            num_reqs, dtype=torch.int32, device=device
        )

        eagle_prepare_inputs_padded_kernel[(num_reqs,)](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        total_num_tokens = query_start_loc_cpu[-1].item()
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = common_attn_metadata.seq_lens.cpu()

        updated_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=query_lens_cpu.max().item(),
            max_seq_len=seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=(common_attn_metadata._num_computed_tokens_cpu),
        )

        return (
            updated_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def _prepare_proposal_token_budgets(
        self,
        remaining_generation_tokens: Sequence[int],
        valid_sampled_tokens_count: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(remaining_generation_tokens)
        if batch_size > self.max_batch_size:
            raise ValueError("proposal budget exceeds the batch workspace capacity")
        if any(budget < 0 for budget in remaining_generation_tokens):
            raise ValueError("remaining generation-token budgets must be non-negative")
        if valid_sampled_tokens_count.shape != (batch_size,):
            raise ValueError("valid_sampled_tokens_count must match the proposal batch")
        if valid_sampled_tokens_count.device != self.device:
            raise ValueError("valid_sampled_tokens_count must be on the model device")
        if valid_sampled_tokens_count.dtype != torch.int32:
            raise ValueError("valid_sampled_tokens_count must have dtype torch.int32")

        self._proposal_token_budgets.np[:batch_size] = remaining_generation_tokens
        proposal_token_budgets = self._proposal_token_budgets.copy_to_gpu(batch_size)
        proposal_token_budgets.sub_(valid_sampled_tokens_count)
        proposal_token_budgets.clamp_(min=0, max=self.num_speculative_tokens)
        return proposal_token_budgets

    def _record_cudagraph_fallback(self, stage_name: str, reason: str) -> None:
        self.performance_stats.add_counter(f"{stage_name}_cudagraph_fallback")
        self.performance_stats.add_counter(f"{stage_name}_cudagraph_fallback_{reason}")

    def _dispatch_piecewise_cudagraph(
        self, num_tokens: int, capacity: int, stage_name: str
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        eager_descriptor = BatchDescriptor(num_tokens)

        if self.speculative_config.enforce_eager:
            self.performance_stats.add_counter(f"{stage_name}_cudagraph_eager")
            return CUDAGraphMode.NONE, eager_descriptor

        if getattr(self.vllm_config, "lora_config", None) is not None:
            self._record_cudagraph_fallback(stage_name, "lora")
            return CUDAGraphMode.NONE, eager_descriptor

        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        if parallel_config is not None and parallel_config.data_parallel_size > 1:
            self._record_cudagraph_fallback(stage_name, "data_parallel")
            return CUDAGraphMode.NONE, eager_descriptor

        dispatcher = getattr(self.runner, "cudagraph_dispatcher", None)
        if dispatcher is None:
            self._record_cudagraph_fallback(stage_name, "missing_dispatcher")
            return CUDAGraphMode.NONE, eager_descriptor
        if self._cudagraph_registration_failure is not None:
            self._record_cudagraph_fallback(
                stage_name, self._cudagraph_registration_failure
            )
            return CUDAGraphMode.NONE, eager_descriptor

        cudagraph_mode, batch_descriptor = dispatcher.dispatch_piecewise_cudagraph(
            self._CUDAGRAPH_NAMESPACE, num_tokens
        )
        if cudagraph_mode != CUDAGraphMode.PIECEWISE:
            self._record_cudagraph_fallback(stage_name, "missing_key")
            return CUDAGraphMode.NONE, eager_descriptor
        if batch_descriptor.num_tokens > capacity:
            self._record_cudagraph_fallback(stage_name, "capacity")
            return CUDAGraphMode.NONE, eager_descriptor

        self.performance_stats.add_counter(f"{stage_name}_cudagraph_replay")
        self.performance_stats.add_counter(
            f"{stage_name}_cudagraph_padding_tokens",
            batch_descriptor.num_tokens - num_tokens,
        )
        return cudagraph_mode, batch_descriptor

    def _prepare_piecewise_model_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        slot_mapping_workspace: torch.Tensor,
        stage_name: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        CUDAGraphMode,
        BatchDescriptor,
    ]:
        num_tokens = input_ids.shape[0]
        if positions.shape != (num_tokens,):
            raise ValueError("Model positions must match the input token count")
        if slot_mapping.shape != (num_tokens,):
            raise ValueError("Slot mapping must match the input token count")

        cudagraph_mode, batch_descriptor = self._dispatch_piecewise_cudagraph(
            num_tokens, slot_mapping_workspace.shape[0], stage_name
        )
        if cudagraph_mode == CUDAGraphMode.NONE:
            return (
                input_ids,
                positions,
                slot_mapping,
                cudagraph_mode,
                batch_descriptor,
            )

        padded_tokens = batch_descriptor.num_tokens
        graph_input_ids = self._graph_input_ids[:padded_tokens]
        graph_positions = self._graph_positions[:padded_tokens]
        graph_slot_mapping = slot_mapping_workspace[:padded_tokens]

        graph_input_ids[:num_tokens].copy_(input_ids)
        graph_positions[:num_tokens].copy_(positions)
        graph_slot_mapping[:num_tokens].copy_(slot_mapping)
        if padded_tokens > num_tokens:
            graph_input_ids[num_tokens:].zero_()
            graph_positions[num_tokens:].zero_()
            graph_slot_mapping[num_tokens:].fill_(PADDING_SLOT_ID)

        return (
            graph_input_ids,
            graph_positions,
            graph_slot_mapping,
            cudagraph_mode,
            batch_descriptor,
        )

    def _run_model_step(
        self,
        batch_size: int,
        step_index: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        attention_mode: RetroSpecAttentionMode,
        compute_margin: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        assert self.model is not None
        model_timer = self.performance_stats.start_cuda_timer("draft_model")

        with self.performance_stats.cuda_timer("draft_prepare"):
            exceeds_max_model_len = positions >= self.max_model_len
            runnable_mask = active_mask & ~exceeds_max_model_len
            clamped_positions = torch.where(
                runnable_mask, positions, torch.zeros_like(positions)
            )

            block_numbers = clamped_positions // self.block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            ).view(-1)

            slot_mapping = self._slot_mapping[:batch_size]
            slot_mapping.copy_(
                block_ids * self.block_size + clamped_positions % self.block_size
            )
            slot_mapping.masked_fill_(~runnable_mask, PADDING_SLOT_ID)

            seq_lens = torch.where(
                runnable_mask,
                clamped_positions + 1,
                torch.ones_like(clamped_positions),
            ).to(dtype=common_attn_metadata.seq_lens.dtype)

            query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()

            step_common_attn_metadata = common_attn_metadata.replace(
                query_start_loc=self.arange[: batch_size + 1],
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                num_actual_tokens=batch_size,
                max_query_len=1,
                max_seq_len=min(
                    common_attn_metadata.max_seq_len + step_index + 1,
                    self.max_model_len,
                ),
                slot_mapping=slot_mapping,
            )

            builder = self._get_attention_metadata_builder()
            attn_metadata = builder.build_for_drafting(
                step_common_attn_metadata, step_index
            )

            per_layer_attn_metadata = {
                layer_name: attn_metadata for layer_name in self.attn_layer_names
            }

            self.sparse_attention.begin_step(attention_mode, step_index, runnable_mask)

            # Verification inputs can contain the -1 sentinel for rows that did
            # not produce a token at this draft position. The padded model call
            # still embeds every row, so replace inactive IDs with a valid token;
            # their output is ignored by the active mask.
            safe_input_ids = torch.where(
                runnable_mask,
                input_ids[:batch_size],
                torch.zeros_like(input_ids[:batch_size]),
            )
            (
                model_input_ids,
                model_positions,
                forward_slot_mapping,
                cudagraph_mode,
                batch_descriptor,
            ) = self._prepare_piecewise_model_inputs(
                input_ids=safe_input_ids,
                positions=clamped_positions,
                slot_mapping=slot_mapping,
                slot_mapping_workspace=self._slot_mapping,
                stage_name="draft",
            )
            per_layer_slot_mapping = {
                layer_name: forward_slot_mapping for layer_name in self.attn_layer_names
            }

        with (
            self.performance_stats.cuda_timer("draft_forward"),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=batch_descriptor.num_tokens,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=(
                    batch_descriptor if cudagraph_mode != CUDAGraphMode.NONE else None
                ),
                slot_mapping=per_layer_slot_mapping,
            ),
        ):
            hidden_states = self._run_pipeline_stage_model(
                model_input_ids,
                model_positions,
                batch_descriptor.num_tokens,
                cudagraph_mode,
                "draft",
            )

        with self.performance_stats.cuda_timer("draft_end_step"):
            local_attention_stats = self.sparse_attention.end_step_statistics()
            attention_mass = self.pipeline_protocol.reduce_attention_mass(
                local_attention_stats
            )

        stage = self._require_pipeline_stage()
        sampled_token_ids = None
        margin = None
        if stage.is_last:
            assert hidden_states is not None
            with self.performance_stats.cuda_timer("draft_logits"):
                logits = self.model.compute_logits(hidden_states[:batch_size])
                if logits is None:
                    raise RuntimeError(
                        "The final RetroSpec PP stage did not produce logits"
                    )

                if compute_margin:
                    top2_logits = torch.topk(logits.float(), k=2, dim=-1).values
                    margin = top2_logits[:, 0] - top2_logits[:, 1]

            with self.performance_stats.cuda_timer("draft_sampling"):
                sampler_output = self.runner.sampler(
                    logits=logits, sampling_metadata=sampling_metadata
                )
                sampled_token_ids = sampler_output.sampled_token_ids.view(-1).to(
                    torch.int32
                )

        pipeline_output = self.pipeline_protocol.broadcast_model_output(
            num_tokens=batch_size,
            token_ids=sampled_token_ids,
            margin=margin,
            compute_margin=compute_margin,
        )
        self.performance_stats.stop_cuda_timer(model_timer)
        return (
            pipeline_output.token_ids,
            pipeline_output.margin,
            attention_mass,
        )

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
