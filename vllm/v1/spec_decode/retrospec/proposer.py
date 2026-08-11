# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from .attention import RetroSpecAttentionMode, RetroSpecSparseAttention
from .decision import RetroSpecDecisionPolicy, RetroSpecMetrics
from .state import RetroSpecBatchState, RetroSpecStage

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass(frozen=True)
class RetroSpecVerificationResult:
    verified_counts: torch.Tensor
    require_full: torch.Tensor


class RetroSpecProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "GPUModelRunner",
    ) -> None:
        config = vllm_config.speculative_config
        assert config is not None
        assert config.method == "retrospec"
        assert config.num_speculative_tokens is not None

        if config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "RetroSpec currently requires padded drafter batches."
            )
        if config.retrospec_cache_mode != "gpu_reference":
            raise NotImplementedError(
                "RetroSpec CPU-offloaded KV cache is not implemented yet."
            )
        if vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError(
                "RetroSpec does not support async scheduling yet."
            )
        if runner.supports_mm_inputs:
            raise NotImplementedError(
                "RetroSpec does not support multimodal models yet."
            )
        if runner.uses_mrope or runner.uses_xdrope_dim > 0:
            raise NotImplementedError(
                "RetroSpec currently supports standard one-dimensional RoPE only."
            )

        self.vllm_config = vllm_config
        self.speculative_config = config
        self.device = device
        self.runner = runner
        self.model: nn.Module | None = None

        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs

        block_size = vllm_config.cache_config.block_size
        assert block_size is not None
        self.block_size = block_size

        self.policy = RetroSpecDecisionPolicy(config)
        self.state = RetroSpecBatchState(self.max_batch_size, device)
        self.sparse_attention = RetroSpecSparseAttention(vllm_config, device)

        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.attn_layer_names: list[str] = []

        self.input_ids = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self.proposal_input_ids = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self.proposal_start_positions = torch.zeros(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self.positions = torch.zeros(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self._slot_mapping = torch.full(
            (self.max_batch_size,), PADDING_SLOT_ID, dtype=torch.int64, device=device
        )
        self._draft_token_ids = torch.full(
            (self.max_batch_size, self.num_speculative_tokens),
            -1,
            dtype=torch.int32,
            device=device,
        )

        self.arange = torch.arange(
            self.max_batch_size + 1, dtype=torch.int32, device=device
        )
        self.token_arange_np = np.arange(self.max_batch_size + 1, dtype=np.int32)

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
            pin_memory=is_pin_memory_available(),
            with_numpy=True,
        )

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model

        attention_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        if not attention_layers:
            raise RuntimeError("No attention layers were registered for RetroSpec.")

        self.attn_layer_names = list(attention_layers)
        self.sparse_attention.install(attention_layers)

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        if self.attn_metadata_builder is not None:
            return self.attn_metadata_builder

        if not self.attn_layer_names:
            raise RuntimeError("No attention layers were registered for RetroSpec.")

        chosen_layer = self.attn_layer_names[0]
        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    self.attn_metadata_builder = attn_group.get_metadata_builder()
                    return self.attn_metadata_builder

        raise RuntimeError(
            "Failed to find the attention metadata builder for RetroSpec."
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        layer_to_group: dict[str, int] = {}
        for group_index, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                layer_to_group[layer_name] = group_index

        group_indices = {
            layer_to_group[layer_name] for layer_name in self.attn_layer_names
        }
        if len(group_indices) != 1:
            raise NotImplementedError(
                "RetroSpec currently requires all attention layers to use the "
                "same KV-cache group."
            )

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
            runnable_mask, clamped_positions + 1, torch.ones_like(clamped_positions)
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
                common_attn_metadata.max_seq_len + step_index + 1, self.max_model_len
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
        per_layer_slot_mapping = {
            layer_name: slot_mapping for layer_name in self.attn_layer_names
        }

        self.sparse_attention.begin_step(attention_mode, step_index, runnable_mask)

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=batch_size,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=per_layer_slot_mapping,
        ):
            hidden_states = self.model(
                input_ids=input_ids[:batch_size],
                positions=clamped_positions,
                inputs_embeds=None,
            )

        attention_mass = self.sparse_attention.end_step()

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "RetroSpec requires the target model to return hidden states."
            )

        logits = self.model.compute_logits(hidden_states[:batch_size])

        margin = None
        if compute_margin:
            top2_logits = torch.topk(logits.float(), k=2, dim=-1).values
            margin = top2_logits[:, 0] - top2_logits[:, 1]

        sampler_output = self.runner.sampler(
            logits=logits, sampling_metadata=sampling_metadata
        )
        sampled_token_ids = sampler_output.sampled_token_ids.view(-1).to(torch.int32)

        return sampled_token_ids, margin, attention_mass

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

    def _run_verification_step(
        self,
        batch_size: int,
        draft_index: int,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        attention_mode: RetroSpecAttentionMode,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        positions = self.proposal_start_positions[:batch_size] + draft_index

        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            compute_margin = self.policy.sparse_margin_threshold is not None
        elif attention_mode == RetroSpecAttentionMode.EXPANDED_VERIFY:
            compute_margin = self.policy.expanded_margin_threshold is not None
        else:
            raise ValueError(
                "Verification requires SPARSE_VERIFY or EXPANDED_VERIFY mode."
            )

        return self._run_model_step(
            batch_size=batch_size,
            step_index=draft_index,
            input_ids=input_ids,
            positions=positions,
            active_mask=active_mask,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            attention_mode=attention_mode,
            compute_margin=compute_margin,
        )

    def _verify_draft_tokens(
        self,
        batch_size: int,
        round_start_counts: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> RetroSpecVerificationResult:
        draft_counts = self.state.draft_counts
        verified_counts = torch.zeros_like(draft_counts)
        require_full = torch.zeros_like(self.state.active_mask)

        verification_active = self.state.active_mask & (draft_counts > 0)
        round_end_counts = round_start_counts + draft_counts

        for token_index in range(self.num_speculative_tokens):
            step_mask = (
                verification_active
                & (round_start_counts <= token_index)
                & (token_index < round_end_counts)
            )
            if not step_mask.any().item():
                continue

            if token_index == 0:
                verify_input_ids = self.proposal_input_ids[:batch_size]
            else:
                verify_input_ids = self._draft_token_ids[:batch_size, token_index - 1]

            self.state.set_stage(step_mask, RetroSpecStage.SPARSE_VERIFY)

            sparse_token_ids, sparse_margin, retrieval_attn = (
                self._run_verification_step(
                    batch_size,
                    token_index,
                    verify_input_ids,
                    step_mask,
                    common_attn_metadata,
                    sampling_metadata,
                    RetroSpecAttentionMode.SPARSE_VERIFY,
                )
            )

            expected_token_ids = self._draft_token_ids[:batch_size, token_index].clone()
            sparse_token_changed = step_mask & (sparse_token_ids != expected_token_ids)

            generation_limit_reached = step_mask & (
                self.proposal_start_positions[:batch_size] + token_index + 1
                >= self.max_model_len - 1
            )

            # Include the current token in the pending count evaluated by the
            # policy because this token is retained after sparse verification.
            candidate_pending_counts = (
                round_start_counts
                + verified_counts
                + step_mask.to(dtype=verified_counts.dtype)
            )

            sparse_decision = self.policy.evaluate(
                current_stage=RetroSpecStage.SPARSE_VERIFY,
                request_stages=self.state.stage,
                metrics=RetroSpecMetrics(
                    sparse_margin=sparse_margin,
                    retrieval_attn=retrieval_attn,
                ),
                draft_counts=draft_counts,
                pending_counts=candidate_pending_counts,
                active_mask=self.state.active_mask,
                sparse_token_changed=sparse_token_changed,
                generation_limit_reached=generation_limit_reached,
            )
            self.state.set_stages(sparse_decision.next_stage)
            require_full |= sparse_decision.require_full

            expanded_mask = step_mask & sparse_decision.require_expanded
            final_token_ids = sparse_token_ids
            expanded_failed = torch.zeros_like(step_mask)

            if expanded_mask.any().item():
                self.state.set_stage(expanded_mask, RetroSpecStage.EXPANDED_VERIFY)

                expanded_token_ids, expanded_margin, expanded_attn = (
                    self._run_verification_step(
                        batch_size,
                        token_index,
                        verify_input_ids,
                        expanded_mask,
                        common_attn_metadata,
                        sampling_metadata,
                        RetroSpecAttentionMode.EXPANDED_VERIFY,
                    )
                )

                expanded_token_changed = expanded_mask & (
                    expanded_token_ids != sparse_token_ids
                )

                expanded_decision = self.policy.evaluate(
                    current_stage=RetroSpecStage.EXPANDED_VERIFY,
                    request_stages=self.state.stage,
                    metrics=RetroSpecMetrics(
                        expanded_margin=expanded_margin,
                        expanded_attn=expanded_attn,
                    ),
                    draft_counts=draft_counts,
                    pending_counts=candidate_pending_counts,
                    active_mask=self.state.active_mask,
                    expanded_token_changed=expanded_token_changed,
                    generation_limit_reached=generation_limit_reached,
                )
                self.state.set_stages(expanded_decision.next_stage)
                require_full |= expanded_decision.require_full

                final_token_ids = torch.where(
                    expanded_mask,
                    expanded_token_ids,
                    sparse_token_ids,
                )
                expanded_failed = expanded_mask & expanded_decision.require_full

            output_column = self._draft_token_ids[:batch_size, token_index]
            output_column.copy_(torch.where(step_mask, final_token_ids, output_column))

            # Retain the current token even when verification corrected it.
            verified_counts.add_(step_mask.to(dtype=verified_counts.dtype))

            # A sparse correction ends this round. If expanded verification
            # succeeds, another round may still extend the pending prefix.
            stop_mask = (
                sparse_token_changed | expanded_failed | sparse_decision.require_full
            )
            verification_active &= ~stop_mask

        return RetroSpecVerificationResult(
            verified_counts=verified_counts,
            require_full=require_full,
        )

    @torch.inference_mode()
    def propose(
        self,
        next_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if not sampling_metadata.all_greedy:
            raise NotImplementedError(
                "RetroSpec currently supports greedy decoding only. "
                "Random sampling requires draft probabilities."
            )

        batch_size = common_attn_metadata.batch_size()
        self.state.begin_batch(batch_size)

        self._draft_token_ids[:batch_size].fill_(-1)
        self.input_ids[:batch_size].copy_(next_token_ids)
        self.proposal_input_ids[:batch_size].copy_(next_token_ids)

        seq_lens = common_attn_metadata.seq_lens
        if num_rejected_tokens_gpu is not None:
            seq_lens = seq_lens - num_rejected_tokens_gpu

        self.positions[:batch_size].copy_(seq_lens)
        self.proposal_start_positions[:batch_size].copy_(seq_lens)

        no_draft_space = self.positions[:batch_size] >= self.max_model_len - 1
        self.state.finish_requests(no_draft_space)

        with self.sparse_attention.proposal_context():
            while True:
                draft_round_mask = (
                    self.state.active_mask
                    & (self.state.stage == int(RetroSpecStage.DRAFT))
                    & (self.state.pending_counts < self.policy.pending_limit)
                    & (self.positions[:batch_size] < self.max_model_len - 1)
                )
                if not draft_round_mask.any().item():
                    break

                # Append this round after the existing pending prefix.
                round_start_counts = self.state.pending_counts.clone()

                # Draft counts are local to this round.
                self.state.reset_draft_counts(self.state.active_mask)

                for token_index in range(self.num_speculative_tokens):
                    next_token_indices = round_start_counts + self.state.draft_counts

                    draft_stage_mask = (
                        draft_round_mask
                        & self.state.active_mask
                        & (self.state.stage == int(RetroSpecStage.DRAFT))
                        & (next_token_indices == token_index)
                        & (next_token_indices < self.policy.pending_limit)
                        & (self.positions[:batch_size] < self.max_model_len - 1)
                    )
                    if not draft_stage_mask.any().item():
                        continue

                    sampled_token_ids, draft_margin, hit_attn = self._run_draft_step(
                        batch_size,
                        token_index,
                        common_attn_metadata,
                        draft_stage_mask,
                        sampling_metadata,
                    )

                    output_column = self._draft_token_ids[:batch_size, token_index]
                    output_column.copy_(
                        torch.where(
                            draft_stage_mask,
                            sampled_token_ids,
                            output_column,
                        )
                    )

                    emitted_counts = draft_stage_mask.to(torch.int32)
                    self.state.add_draft_counts(emitted_counts)

                    # Draft tokens are not pending until sparse verification.
                    # Only provide the projected count to the decision policy.
                    projected_pending_counts = (
                        round_start_counts + self.state.draft_counts
                    )

                    generation_limit_reached = draft_stage_mask & (
                        self.positions[:batch_size] + 1 >= self.max_model_len - 1
                    )

                    decision = self.policy.evaluate(
                        current_stage=RetroSpecStage.DRAFT,
                        request_stages=self.state.stage,
                        metrics=RetroSpecMetrics(
                            draft_margin=draft_margin,
                            hit_attn=hit_attn,
                        ),
                        draft_counts=self.state.draft_counts,
                        pending_counts=projected_pending_counts,
                        active_mask=self.state.active_mask,
                        generation_limit_reached=generation_limit_reached,
                    )
                    self.state.set_stages(decision.next_stage)

                    self.positions[:batch_size].add_(emitted_counts)

                    # Sampled IDs are meaningful only for rows that ran this
                    # step and remain in the draft stage.
                    continue_draft_mask = draft_stage_mask & (
                        self.state.stage == int(RetroSpecStage.DRAFT)
                    )
                    self.input_ids[:batch_size].copy_(
                        torch.where(
                            continue_draft_mask,
                            sampled_token_ids,
                            self.input_ids[:batch_size],
                        )
                    )

                verification = self._verify_draft_tokens(
                    batch_size,
                    round_start_counts,
                    common_attn_metadata,
                    sampling_metadata,
                )

                # Discard the unverified suffix of the current round.
                updated_pending_counts = torch.where(
                    draft_round_mask,
                    round_start_counts + verification.verified_counts,
                    self.state.pending_counts,
                )
                self.state.set_pending_counts(updated_pending_counts)

                # Verification can stop early. Roll positions back so a later
                # draft round overwrites KV slots belonging to discarded tokens.
                self.positions[:batch_size].copy_(
                    self.proposal_start_positions[:batch_size]
                    + self.state.pending_counts
                )

                can_defer_full = (
                    draft_round_mask
                    & self.state.active_mask
                    & (verification.verified_counts > 0)
                    & ~verification.require_full
                    & (self.state.pending_counts < self.policy.pending_limit)
                    & (self.positions[:batch_size] < self.max_model_len - 1)
                )

                require_full = (
                    draft_round_mask & (self.state.pending_counts > 0) & ~can_defer_full
                )

                self.state.set_stage(can_defer_full, RetroSpecStage.DRAFT)
                self.state.set_stage(require_full, RetroSpecStage.FULL_VERIFY)

                empty_round = draft_round_mask & (self.state.pending_counts == 0)
                if empty_round.any().item():
                    self.state.finish_requests(empty_round)

                # Continue drafting from the final token in the pending prefix.
                safe_last_indices = (self.state.pending_counts - 1).clamp(
                    min=0,
                    max=self.num_speculative_tokens - 1,
                )
                last_pending_tokens = (
                    self._draft_token_ids[:batch_size]
                    .gather(1, safe_last_indices.unsqueeze(1))
                    .squeeze(1)
                )

                next_round_input_ids = torch.where(
                    self.state.pending_counts > 0,
                    last_pending_tokens,
                    self.proposal_input_ids[:batch_size],
                )
                self.input_ids[:batch_size].copy_(
                    torch.where(
                        can_defer_full,
                        next_round_input_ids,
                        self.input_ids[:batch_size],
                    )
                )

        pending_counts_cpu = self.state.pending_counts.cpu().tolist()
        pending_token_ids = self._draft_token_ids[:batch_size].cpu().tolist()

        return [
            token_ids[:pending_count]
            for token_ids, pending_count in zip(pending_token_ids, pending_counts_cpu)
        ]
