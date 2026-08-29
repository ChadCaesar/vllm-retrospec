# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from time import perf_counter
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
from vllm.v1.outputs import KVCacheRetirement
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
from .state import RetroSpecBatchState, RetroSpecIndexUpdateState, RetroSpecStage

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass(frozen=True)
class RetroSpecVerificationResult:
    verified_counts: torch.Tensor
    require_full: torch.Tensor


@dataclass(frozen=True)
class RetroSpecParallelVerificationOutput:
    request_indices: torch.Tensor
    token_indices: torch.Tensor
    token_ids: torch.Tensor
    margin: torch.Tensor | None
    attention_mass: torch.Tensor


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
        self.max_parallel_tokens = self.max_batch_size * self.num_speculative_tokens

        block_size = vllm_config.cache_config.block_size
        assert block_size is not None
        self.block_size = block_size

        self.policy = RetroSpecDecisionPolicy(config)
        self.state = RetroSpecBatchState(self.max_batch_size, device)
        self.sparse_attention = RetroSpecSparseAttention(vllm_config, device)
        self.performance_stats = self.sparse_attention.performance_stats
        self.index_update_state = RetroSpecIndexUpdateState(
            max_batch_size=self.max_batch_size,
            update_interval=config.retrospec_index_update_interval,
            device=device,
            pin_memory=is_pin_memory_available(),
        )
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.attn_layer_names: list[str] = []
        self.kv_cache_group_id: int | None = None

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
        self._verification_slot_mapping = torch.full(
            (self.max_parallel_tokens,),
            PADDING_SLOT_ID,
            dtype=torch.int64,
            device=device,
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
        self.parallel_arange = torch.arange(
            self.max_parallel_tokens + 1, dtype=torch.int32, device=device
        )
        self.parallel_token_arange_np = np.arange(
            self.max_parallel_tokens + 1, dtype=np.int32
        )
        self.verification_token_offsets = torch.arange(
            self.num_speculative_tokens, dtype=torch.int64, device=device
        )

        # Fixed-capacity verification control workspace. Model execution still
        # uses only the compact valid prefix, so inactive pair slots do not
        # replicate long-context block tables or enter the target model.
        self._verification_pair_mask = torch.zeros(
            self.max_parallel_tokens, dtype=torch.bool, device=device
        )
        self._verification_prefix = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_destinations = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_valid_destinations = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_flat_indices = torch.arange(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_compact_indices = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_request_indices = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_token_indices = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_round_starts = torch.empty(
            self.max_parallel_tokens, dtype=torch.int32, device=device
        )
        self._verification_boundary_candidates = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        self._verification_first_boundaries = torch.empty(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self._verification_boundary_requests = torch.empty(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self._verification_expanded_indices = torch.empty(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self._verification_verified_counts = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._verification_require_full = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._sparse_sampled_token_ids = torch.empty(
            self.max_parallel_tokens, dtype=torch.int32, device=device
        )
        self._expanded_sampled_token_ids = torch.empty(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._verification_step_logits: torch.Tensor | None = None

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
            pin_memory=is_pin_memory_available(),
            with_numpy=True,
        )

    def remove_requests(self, request_ids: Collection[str]) -> None:
        request_ids = tuple(request_ids)
        self.index_update_state.remove_requests(request_ids)
        self.sparse_attention.remove_requests(request_ids)

    @property
    def uses_full_verification_offload(self) -> bool:
        return self.sparse_attention.uses_full_verification_offload

    def full_verification_context(
        self,
        request_ids: Sequence[str],
        context_lens: Sequence[int],
        query_lens: Sequence[int],
    ):
        return self.sparse_attention.full_verification_context(
            request_ids,
            context_lens,
            query_lens,
        )

    def has_retired_kv_blocks(self, request_ids: Sequence[str]) -> bool:
        return self.sparse_attention.has_retired_kv_blocks(request_ids)

    def take_kv_cache_retirements(
        self,
        request_ids: Sequence[str],
    ) -> list[KVCacheRetirement]:
        ranges = self.sparse_attention.take_kv_cache_retirement_ranges(request_ids)
        if not ranges:
            return []
        if self.kv_cache_group_id is None:
            raise RuntimeError("RetroSpec KV cache group has not been initialized")

        return [
            KVCacheRetirement(
                request_id=request_id,
                kv_cache_group_id=self.kv_cache_group_id,
                start_block=start_block,
                end_block=end_block,
            )
            for request_id, start_block, end_block in ranges
        ]

    def needs_index_update(
        self,
        request_id: str,
        seq_len: int,
        is_prefill: bool,
        prefill_complete: bool,
    ) -> bool:
        return self.sparse_attention.needs_index_update(
            request_id,
            seq_len,
            is_prefill,
            prefill_complete,
        )

    def index_update_context(
        self,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        is_prefill: Sequence[bool],
        prefill_complete: Sequence[bool],
        build_rows: Sequence[int],
    ):
        return self.sparse_attention.index_update_context(
            request_ids,
            seq_lens,
            is_prefill,
            prefill_complete,
            build_rows,
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

        self.kv_cache_group_id = next(iter(group_indices))

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
        model_timer = self.performance_stats.start_cuda_timer("draft_model")

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

        # Verification inputs can contain the -1 sentinel for rows that did
        # not produce a token at this draft position. The padded model call
        # still embeds every row, so replace inactive IDs with a valid token;
        # their output is ignored by the active mask.
        safe_input_ids = torch.where(
            runnable_mask,
            input_ids[:batch_size],
            torch.zeros_like(input_ids[:batch_size]),
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=batch_size,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=per_layer_slot_mapping,
        ):
            hidden_states = self.model(
                input_ids=safe_input_ids,
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

        self.performance_stats.stop_cuda_timer(model_timer)
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

        sampled_token_ids = output[: logits.shape[0]]
        sampled_token_ids.fill_(-1)

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

        timer_name = (
            "sparse_verify_model"
            if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
            else "expanded_verify_model"
        )
        model_timer = self.performance_stats.start_cuda_timer(timer_name)

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
        slot_mapping.copy_(block_ids * self.block_size + positions % self.block_size)
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
        per_layer_slot_mapping = {
            layer_name: slot_mapping for layer_name in self.attn_layer_names
        }

        self.sparse_attention.begin_parallel_step(
            attention_mode, request_indices, token_indices
        )
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=per_layer_slot_mapping,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=None,
            )

        attention_mass = self.sparse_attention.end_step()
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "RetroSpec requires the target model to return hidden states."
            )

        logits = self.model.compute_logits(hidden_states[:num_tokens])
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            compute_margin = self.policy.sparse_margin_threshold is not None
        else:
            compute_margin = self.policy.expanded_margin_threshold is not None

        margin = None
        if compute_margin:
            top2_logits = torch.topk(logits.float(), k=2, dim=-1).values
            margin = top2_logits[:, 0] - top2_logits[:, 1]

        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            sampled_output = self._sparse_sampled_token_ids
        else:
            sampled_output = self._expanded_sampled_token_ids

        token_ids = self._sample_parallel_logits(
            batch_size,
            logits,
            request_indices,
            token_indices,
            sampling_metadata,
            sampled_output,
        )
        self.performance_stats.stop_cuda_timer(model_timer)
        return RetroSpecParallelVerificationOutput(
            request_indices=request_indices,
            token_indices=token_indices,
            token_ids=token_ids,
            margin=margin,
            attention_mass=attention_mass,
        )

    def _verify_draft_tokens(
        self,
        batch_size: int,
        round_start_counts: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> RetroSpecVerificationResult:
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
        sparse = self._run_parallel_verification(
            batch_size,
            request_indices,
            token_indices,
            common_attn_metadata,
            sampling_metadata,
            RetroSpecAttentionMode.SPARSE_VERIFY,
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
        generation_limit_reached = candidate_positions >= self.max_model_len - 1
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
        accepted_flat_indices = self._compact_mask_indices(
            accepted_mask,
            self._verification_compact_indices,
        )
        accepted_requests = sparse.request_indices.index_select(
            0, accepted_flat_indices
        )
        accepted_tokens = sparse.token_indices.index_select(0, accepted_flat_indices)
        accepted_token_ids = sparse.token_ids.index_select(0, accepted_flat_indices)
        self._draft_token_ids[accepted_requests, accepted_tokens] = accepted_token_ids

        verified_counts = self._verification_verified_counts[:batch_size]
        verified_counts.zero_()
        verified_counts.scatter_add_(
            0,
            sparse.request_indices,
            accepted_mask.to(draft_counts.dtype),
        )

        boundary_request_mask = first_boundary_indices < num_pairs
        boundary_requests = self._compact_mask_indices(
            boundary_request_mask,
            self._verification_boundary_requests,
        )
        boundary_flat_indices = first_boundary_indices.index_select(
            0, boundary_requests
        )

        require_full = self._verification_require_full[:batch_size]
        require_full.zero_()
        sparse_boundary_require_full = sparse_decision.require_full.index_select(
            0, boundary_flat_indices
        )
        require_full.index_copy_(
            0,
            boundary_requests,
            sparse_boundary_require_full,
        )

        sparse_boundary_require_expanded = (
            sparse_decision.require_expanded.index_select(0, boundary_flat_indices)
        )
        run_expanded = sparse_boundary_require_expanded & ~sparse_boundary_require_full
        expanded_boundary_offsets = self._compact_mask_indices(
            run_expanded,
            self._verification_compact_indices,
        )
        expanded_sparse_indices = self._verification_expanded_indices[
            : expanded_boundary_offsets.shape[0]
        ]
        torch.index_select(
            boundary_flat_indices,
            0,
            expanded_boundary_offsets,
            out=expanded_sparse_indices,
        )

        if expanded_sparse_indices.numel() > 0:
            expanded_request_indices = sparse.request_indices.index_select(
                0, expanded_sparse_indices
            )
            expanded_token_indices = sparse.token_indices.index_select(
                0, expanded_sparse_indices
            )
            self.performance_stats.add_counter(
                "expanded_verify_tokens",
                expanded_request_indices.numel(),
            )
            expanded_request_mask = torch.zeros_like(self.state.active_mask)
            expanded_request_mask.scatter_(0, expanded_request_indices, True)
            self.state.set_stage(expanded_request_mask, RetroSpecStage.EXPANDED_VERIFY)

            expanded = self._run_parallel_verification(
                batch_size,
                expanded_request_indices,
                expanded_token_indices,
                common_attn_metadata,
                sampling_metadata,
                RetroSpecAttentionMode.EXPANDED_VERIFY,
            )
            sparse_boundary_token_ids = sparse.token_ids.index_select(
                0, expanded_sparse_indices
            )
            expanded_token_changed = expanded.token_ids != sparse_boundary_token_ids
            expanded_pending_counts = (expanded.token_indices + 1).to(
                draft_counts.dtype
            )
            expanded_positions = (
                self.proposal_start_positions.index_select(0, expanded.request_indices)
                + expanded_pending_counts
            )
            expanded_generation_limit = expanded_positions >= self.max_model_len - 1
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
            self._draft_token_ids[expanded.request_indices, expanded.token_indices] = (
                expanded.token_ids
            )
            require_full.index_copy_(
                0,
                expanded.request_indices,
                expanded_decision.require_full,
            )

        self.state.set_stage(verification_active, RetroSpecStage.DRAFT)
        self.state.set_stage(require_full, RetroSpecStage.FULL_VERIFY)

        return RetroSpecVerificationResult(verified_counts, require_full)

    @torch.inference_mode()
    def propose(
        self,
        request_ids: list[str],
        committed_positions: list[int],
        next_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        proposal_active_mask: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if not sampling_metadata.all_greedy:
            raise NotImplementedError(
                "RetroSpec currently supports greedy decoding only. "
                "Random sampling requires draft probabilities."
            )

        proposal_started_at = perf_counter() if self.performance_stats.enabled else 0.0

        batch_size = common_attn_metadata.batch_size()
        if len(request_ids) != batch_size:
            raise ValueError("request_ids must match the proposal batch size")
        if len(committed_positions) != batch_size:
            raise ValueError("committed_positions must match the proposal batch size")

        self.state.begin_batch(batch_size, proposal_active_mask)
        self.index_update_state.begin_batch(request_ids, committed_positions)

        self.performance_stats.add_counter("proposal_calls")
        self.performance_stats.add_gpu_counter(
            "proposal_requests",
            proposal_active_mask,
        )

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

        with self.sparse_attention.proposal_context(request_ids):
            while True:
                draft_round_mask = (
                    self.state.active_mask
                    & (self.state.stage == int(RetroSpecStage.DRAFT))
                    & (self.state.pending_counts < self.policy.pending_limit)
                    & (self.positions[:batch_size] < self.max_model_len - 1)
                )
                if not draft_round_mask.any().item():
                    break

                self.performance_stats.add_gpu_counter(
                    "draft_round_requests",
                    draft_round_mask,
                )

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
                    self.performance_stats.add_gpu_counter(
                        "draft_tokens",
                        emitted_counts,
                    )

                    # Draft tokens are not pending until sparse verification.
                    # Only provide the projected count to the decision policy.
                    projected_pending_counts = (
                        round_start_counts + self.state.draft_counts
                    )

                    generation_limit_reached = draft_stage_mask & (
                        self.positions[:batch_size] + 1 >= self.max_model_len - 1
                    )

                    index_update_required = self.index_update_state.requires_update(
                        self.positions[:batch_size] + 1,
                        draft_stage_mask,
                    )

                    decision = self.policy.evaluate(
                        current_stage=RetroSpecStage.DRAFT,
                        request_stages=self.state.stage,
                        metrics=RetroSpecMetrics(
                            draft_margin=draft_margin, hit_attn=hit_attn
                        ),
                        draft_counts=self.state.draft_counts,
                        pending_counts=projected_pending_counts,
                        active_mask=self.state.active_mask,
                        generation_limit_reached=generation_limit_reached,
                        index_update_required=index_update_required,
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
                self.performance_stats.add_gpu_counter(
                    "verified_tokens",
                    verification.verified_counts,
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

        self.performance_stats.add_gpu_counter(
            "proposed_tokens",
            self.state.pending_counts,
        )
        pending_counts_cpu = self.state.pending_counts.cpu().tolist()
        pending_token_ids = self._draft_token_ids[:batch_size].cpu().tolist()

        result = [
            token_ids[:pending_count]
            for token_ids, pending_count in zip(pending_token_ids, pending_counts_cpu)
        ]
        if self.performance_stats.enabled:
            self.performance_stats.record_cpu_time(
                "proposal_wall",
                perf_counter() - proposal_started_at,
            )
        self.performance_stats.maybe_log()
        return result
