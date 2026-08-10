# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

from .attention import RetroSpecSparseAttention
from .decision import RetroSpecDecisionPolicy, RetroSpecMetrics
from .state import RetroSpecBatchState, RetroSpecStage

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


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

    def _run_draft_step(
        self,
        batch_size: int,
        draft_index: int,
        common_attn_metadata: CommonAttentionMetadata,
        active_mask: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        assert self.model is not None

        positions = self.positions[:batch_size]
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
                common_attn_metadata.max_seq_len + draft_index + 1,
                self.max_model_len,
            ),
            slot_mapping=slot_mapping,
        )

        builder = self._get_attention_metadata_builder()
        attn_metadata = builder.build_for_drafting(
            step_common_attn_metadata, draft_index
        )

        per_layer_attn_metadata = {
            layer_name: attn_metadata for layer_name in self.attn_layer_names
        }
        per_layer_slot_mapping = {
            layer_name: slot_mapping for layer_name in self.attn_layer_names
        }

        self.sparse_attention.begin_step(runnable_mask)
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=batch_size,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=per_layer_slot_mapping,
        ):
            hidden_states = self.model(
                input_ids=self.input_ids[:batch_size],
                positions=clamped_positions,
                inputs_embeds=None,
            )

        hit_attn = self.sparse_attention.end_step()

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "RetroSpec requires the target model to return hidden states."
            )

        logits = self.model.compute_logits(hidden_states[:batch_size])

        draft_margin = None
        if self.policy.draft_margin_threshold is not None:
            top2_logits = torch.topk(logits.float(), k=2, dim=-1).values
            draft_margin = top2_logits[:, 0] - top2_logits[:, 1]

        sampler_output = self.runner.sampler(
            logits=logits, sampling_metadata=sampling_metadata
        )
        sampled_token_ids = sampler_output.sampled_token_ids.view(-1).to(torch.int32)

        return sampled_token_ids, draft_margin, hit_attn

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

        seq_lens = common_attn_metadata.seq_lens
        if num_rejected_tokens_gpu is not None:
            seq_lens = seq_lens - num_rejected_tokens_gpu

        self.positions[:batch_size].copy_(seq_lens)

        no_draft_space = self.positions[:batch_size] >= self.max_model_len - 1
        self.state.finish_requests(no_draft_space)

        with self.sparse_attention.draft_context():
            for draft_index in range(self.num_speculative_tokens):
                draft_stage_mask = (
                    self.state.active_mask
                    & (self.state.stage == int(RetroSpecStage.DRAFT))
                    & (self.positions[:batch_size] < self.max_model_len - 1)
                )

                if not draft_stage_mask.any().item():
                    break

                sampled_token_ids, draft_margin, hit_attn = self._run_draft_step(
                    batch_size,
                    draft_index,
                    common_attn_metadata,
                    draft_stage_mask,
                    sampling_metadata,
                )

                output_column = self._draft_token_ids[:batch_size, draft_index]
                output_column.copy_(
                    torch.where(
                        draft_stage_mask,
                        sampled_token_ids,
                        torch.full_like(sampled_token_ids, -1),
                    )
                )

                emitted_counts = draft_stage_mask.to(torch.int32)
                self.state.add_draft_counts(emitted_counts)
                self.state.add_pending_counts(emitted_counts)

                generation_limit_reached = draft_stage_mask & (
                    self.positions[:batch_size] + 1 >= self.max_model_len - 1
                )

                decision = self.policy.evaluate(
                    current_stage=RetroSpecStage.DRAFT,
                    request_stages=self.state.stage,
                    metrics=RetroSpecMetrics(
                        draft_margin=draft_margin, hit_attn=hit_attn
                    ),
                    draft_counts=self.state.draft_counts,
                    pending_counts=self.state.pending_counts,
                    active_mask=self.state.active_mask,
                    generation_limit_reached=generation_limit_reached,
                )
                self.state.set_stages(decision.next_stage)

                self.positions[:batch_size].add_(emitted_counts)

                next_draft_mask = self.state.active_mask & (
                    self.state.stage == int(RetroSpecStage.DRAFT)
                )
                self.input_ids[:batch_size].copy_(
                    torch.where(
                        next_draft_mask,
                        sampled_token_ids,
                        torch.zeros_like(sampled_token_ids),
                    )
                )

        draft_counts = self.state.draft_counts.cpu().tolist()
        draft_token_ids = self._draft_token_ids[:batch_size].cpu().tolist()

        return [
            token_ids[:draft_count]
            for token_ids, draft_count in zip(draft_token_ids, draft_counts)
        ]
