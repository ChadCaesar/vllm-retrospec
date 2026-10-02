# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Sequence
from contextlib import AbstractContextManager
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.sequence import IntermediateTensors
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVCacheRetirement
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
)
from vllm.v1.utils import CpuGpuBuffer

from .attention import RetroSpecSparseAttention
from .decision import RetroSpecDecisionPolicy, RetroSpecMetrics
from .pipeline import (
    RetroSpecPipelineControlState,
    RetroSpecPipelineProtocol,
    RetroSpecPipelineStage,
)
from .prefill import resolve_retrospec_layer_model
from .runtime.proposer_draft import RetroSpecDraftMixin
from .runtime.proposer_types import (
    RetroSpecParallelVerificationOutput as RetroSpecParallelVerificationOutput,
)
from .runtime.proposer_types import (
    RetroSpecVerificationResult as RetroSpecVerificationResult,
)
from .runtime.proposer_verification import RetroSpecVerificationMixin
from .state import RetroSpecBatchState, RetroSpecIndexUpdateState, RetroSpecStage
from .transition_trace import RetroSpecTransitionTracer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class RetroSpecProposer(RetroSpecDraftMixin, RetroSpecVerificationMixin):
    _CUDAGRAPH_NAMESPACE = "retrospec_proposal"

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
        self.pipeline_protocol = RetroSpecPipelineProtocol(
            vllm_config=vllm_config,
            device=device,
            max_batch_size=self.max_batch_size,
            max_parallel_tokens=self.max_parallel_tokens,
            max_sampled_tokens=self.num_speculative_tokens + 1,
        )
        self.pipeline_stage: RetroSpecPipelineStage | None = None

        block_size = vllm_config.cache_config.block_size
        assert block_size is not None
        self.block_size = block_size

        self.policy = RetroSpecDecisionPolicy(config)
        self.transition_tracer = RetroSpecTransitionTracer(
            config.retrospec_trace_transitions
        )
        self.state = RetroSpecBatchState(self.max_batch_size, device)
        self.sparse_attention = RetroSpecSparseAttention(vllm_config, device)
        self.performance_stats = self.sparse_attention.performance_stats
        self._cudagraph_registration_failure: str | None = "uninitialized"
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
        self._graph_input_ids = torch.zeros(
            self.max_parallel_tokens, dtype=torch.int32, device=device
        )
        self._graph_positions = torch.zeros(
            self.max_parallel_tokens, dtype=torch.int64, device=device
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

        # Fixed-capacity GPU proposal-control workspace. These tensors store
        # only request-level state and do not scale with context length.
        self._draft_round_start_counts = torch.empty(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._draft_next_token_indices = torch.empty(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._draft_round_mask = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._draft_runnable_mask = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._draft_stage_mask = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=device
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
        self._verification_safe_boundaries = torch.empty(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self._verification_sparse_boundary_token_ids = torch.empty(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._verification_boundary_mask = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._verification_run_expanded = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._verification_expanded_requests = torch.empty(
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
        self._verification_argmax_token_ids = torch.empty(
            self.max_parallel_tokens, dtype=torch.int64, device=device
        )
        # Allocated only when row-indexed logits processors force the legacy
        # per-position compatibility path.
        self._verification_step_logits: torch.Tensor | None = None

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
            pin_memory=is_pin_memory_available(),
            with_numpy=True,
        )
        self._proposal_token_budgets = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
            pin_memory=is_pin_memory_available(),
            with_numpy=True,
        )
        self._proposal_token_budgets.cpu.fill_(self.num_speculative_tokens)
        self._proposal_token_budgets.gpu.fill_(self.num_speculative_tokens)
        self._seen_proposal_request_ids: set[str] = set()
        self._last_proposed_counts: dict[str, int] = {}
        self._last_committed_proposal_counts: dict[str, int] = {}
        self._closed = False

    def remove_requests(self, request_ids: Collection[str]) -> None:
        request_ids = tuple(request_ids)
        self.index_update_state.remove_requests(request_ids)
        self.sparse_attention.remove_requests(request_ids)

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

    def stage_layer_major_prefill_layer(
        self,
        layer_name: str,
        request_id: str,
        seq_len: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.cuda.Event | None:
        return self.sparse_attention.stage_layer_major_prefill_layer(
            layer_name,
            request_id,
            seq_len,
            key_cache,
            value_cache,
            block_table,
        )

    def capture_layer_major_prefill_query(
        self, layer_name: str
    ) -> AbstractContextManager[None]:
        return self.sparse_attention.capture_layer_major_prefill_query(layer_name)

    def commit_layer_major_prefill(self, request_id: str) -> None:
        self.sparse_attention.commit_layer_major_prefill(
            request_id, self.attn_layer_names
        )

    def abort_layer_major_prefill(self) -> None:
        self.sparse_attention.abort_layer_major_prefill()

    def close(self) -> None:
        if self._closed:
            return
        self.sparse_attention.uninstall()
        self._closed = True

    def get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        return self._get_attention_metadata_builder()

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model

        attention_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        if not attention_layers:
            raise RuntimeError("No attention layers were registered for RetroSpec.")

        self.attn_layer_names = list(attention_layers)
        self.sparse_attention.install(attention_layers)

        layer_model = resolve_retrospec_layer_model(target_model)
        self.pipeline_stage = self.pipeline_protocol.describe_stage(
            layer_model, self.attn_layer_names
        )

    def _require_pipeline_stage(self) -> RetroSpecPipelineStage:
        if self.pipeline_stage is None:
            raise RuntimeError(
                "RetroSpec pipeline stage is unavailable before model loading"
            )
        return self.pipeline_stage

    def _get_pipeline_receive_destination(
        self,
        stage: RetroSpecPipelineStage,
        num_tokens: int,
        cudagraph_mode: CUDAGraphMode,
    ) -> IntermediateTensors | None:
        if stage.is_first or cudagraph_mode == CUDAGraphMode.NONE:
            return None

        if cudagraph_mode != CUDAGraphMode.PIECEWISE:
            raise RuntimeError(
                f"Unsupported RetroSpec proposal CUDA Graph mode: {cudagraph_mode}"
            )
        if self.runner.intermediate_tensors is None:
            raise RuntimeError(
                "RetroSpec PP PIECEWISE CUDA Graph replay requires the runner "
                "intermediate-tensor workspace"
            )

        return self.runner.sync_and_slice_intermediate_tensors(
            num_tokens, intermediate_tensors=None, sync_self=False
        )

    def _run_pipeline_stage_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
        cudagraph_mode: CUDAGraphMode,
        stage_name: str,
    ) -> torch.Tensor | None:
        if self.model is None:
            raise RuntimeError("RetroSpec target model is not loaded")

        stage = self._require_pipeline_stage()
        receive_destination = self._get_pipeline_receive_destination(
            stage, num_tokens, cudagraph_mode
        )
        if stage.is_first:
            intermediate_tensors = self.pipeline_protocol.receive_model_input(
                stage, num_tokens, receive_destination
            )
        else:
            with (
                self.performance_stats.cpu_timer(f"{stage_name}_pipeline_receive_wall"),
                self.performance_stats.cuda_timer(f"{stage_name}_pipeline_receive"),
            ):
                intermediate_tensors = self.pipeline_protocol.receive_model_input(
                    stage, num_tokens, receive_destination
                )
        model_output = self.model(
            input_ids=input_ids if stage.is_first else None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=None,
        )

        if not stage.is_last:
            if not isinstance(model_output, IntermediateTensors):
                raise RuntimeError(
                    "A non-final RetroSpec PP stage must return IntermediateTensors"
                )
            with (
                self.performance_stats.cpu_timer(f"{stage_name}_pipeline_send_wall"),
                self.performance_stats.cuda_timer(f"{stage_name}_pipeline_send"),
            ):
                self.pipeline_protocol.send_model_output(
                    stage, model_output, num_tokens
                )
            return None

        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if not isinstance(model_output, torch.Tensor):
            raise RuntimeError("The final RetroSpec PP stage must return hidden states")
        return model_output

    def _synchronize_pipeline_control(
        self,
        batch_size: int,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        stage = self._require_pipeline_stage()
        local_state = None
        if stage.is_last:
            local_state = RetroSpecPipelineControlState(
                token_ids=token_ids,
                stages=self.state.stage,
                draft_counts=self.state.draft_counts,
                pending_counts=self.state.pending_counts,
                active_mask=self.state.active_mask,
            )

        synchronized = self.pipeline_protocol.broadcast_control_state(
            batch_size, local_state
        )
        self.state.set_stages(synchronized.stages)
        self.state.draft_counts.copy_(synchronized.draft_counts)
        self.state.pending_counts.copy_(synchronized.pending_counts)
        self.state.active_mask.copy_(synchronized.active_mask)
        return synchronized.token_ids

    def initialize_cudagraph_keys(
        self,
        cudagraph_mode: CUDAGraphMode,
        capture_sizes: Sequence[int],
        max_capture_size: int,
    ) -> None:
        if self.speculative_config.enforce_eager:
            self._cudagraph_registration_failure = "eager"
            return

        dispatcher = getattr(self.runner, "cudagraph_dispatcher", None)
        if dispatcher is None:
            self._cudagraph_registration_failure = "missing_dispatcher"
            return
        if not cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE):
            self._cudagraph_registration_failure = "piecewise_disabled"
            return

        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        if parallel_config is not None and parallel_config.data_parallel_size > 1:
            self._cudagraph_registration_failure = "data_parallel"
            return

        runner_input_ids = getattr(getattr(self.runner, "input_ids", None), "gpu", None)
        runner_positions = getattr(getattr(self.runner, "positions", None), "gpu", None)
        if (
            not isinstance(runner_input_ids, torch.Tensor)
            or not isinstance(runner_positions, torch.Tensor)
            or runner_input_ids.numel() < self.max_parallel_tokens
            or runner_positions.numel() < self.max_parallel_tokens
        ):
            self._cudagraph_registration_failure = "missing_input_workspace"
            return

        self._graph_input_ids = runner_input_ids[: self.max_parallel_tokens]
        self._graph_positions = runner_positions[: self.max_parallel_tokens]

        limit = min(self.max_parallel_tokens, max_capture_size)
        sizes = {size for size in capture_sizes if 0 < size <= limit}
        sizes.update(
            capacity
            for capacity in (self.max_batch_size, self.max_parallel_tokens)
            if 0 < capacity <= limit
        )
        registered = dispatcher.register_piecewise_cudagraph_sizes(
            self._CUDAGRAPH_NAMESPACE, sizes
        )
        self._cudagraph_registration_failure = (
            None
            if registered
            and dispatcher.has_piecewise_cudagraph_namespace(self._CUDAGRAPH_NAMESPACE)
            else "missing_key"
        )

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

    @torch.inference_mode()
    def propose(
        self,
        request_ids: list[str],
        committed_positions: list[int],
        next_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        proposal_active_mask: torch.Tensor,
        remaining_generation_tokens: Sequence[int],
        valid_sampled_tokens_count: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        materialize_output: bool = True,
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

        proposal_token_budgets = self._prepare_proposal_token_budgets(
            remaining_generation_tokens,
            valid_sampled_tokens_count,
        )
        effective_proposal_active_mask = proposal_active_mask & (
            proposal_token_budgets > 0
        )

        self.state.begin_batch(batch_size, effective_proposal_active_mask)
        self.index_update_state.begin_batch(request_ids, committed_positions)

        self.performance_stats.add_counter("proposal_calls")
        self.performance_stats.add_gpu_counter(
            "proposal_requests",
            effective_proposal_active_mask,
        )

        self._draft_token_ids[:batch_size].fill_(-1)

        seq_lens = common_attn_metadata.seq_lens
        if num_rejected_tokens_gpu is not None:
            seq_lens = seq_lens - num_rejected_tokens_gpu

        self.positions[:batch_size].copy_(seq_lens)
        self.proposal_start_positions[:batch_size].copy_(seq_lens)

        no_draft_space = self.positions[:batch_size] >= self.max_model_len - 1
        self.state.finish_requests(no_draft_space)
        next_token_ids = self._synchronize_pipeline_control(batch_size, next_token_ids)
        self.input_ids[:batch_size].copy_(next_token_ids)
        self.proposal_input_ids[:batch_size].copy_(next_token_ids)

        proposal_round = 0
        with self.sparse_attention.proposal_context(request_ids, committed_positions):
            while True:
                proposal_round += 1
                draft_round_mask, round_start_counts = self._begin_draft_round(
                    batch_size
                )
                token_index, draft_stage_mask = self._prepare_next_draft_step(
                    batch_size,
                    draft_round_mask,
                    round_start_counts,
                )
                if token_index >= self.num_speculative_tokens:
                    break
                if self.sparse_attention.selection_provenance_enabled:
                    self.sparse_attention.set_proposal_round(proposal_round)

                self.performance_stats.add_gpu_counter(
                    "draft_round_requests",
                    draft_round_mask,
                )

                while token_index < self.num_speculative_tokens:
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
                        (self.positions[:batch_size] + 1 >= self.max_model_len - 1)
                        | (projected_pending_counts >= proposal_token_budgets)
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
                    draft_to_sparse = draft_stage_mask & (
                        decision.next_stage == int(RetroSpecStage.SPARSE_VERIFY)
                    )
                    self._trace_draft_transition(
                        request_ids=request_ids,
                        proposal_round=proposal_round,
                        transition_mask=draft_to_sparse,
                        decision=decision,
                        draft_margin=draft_margin,
                        hit_attn=hit_attn,
                        projected_pending_counts=projected_pending_counts,
                        generation_limit_reached=generation_limit_reached,
                        index_update_required=index_update_required,
                    )
                    self.performance_stats.add_gpu_histogram(
                        "draft_to_sparse_tokens",
                        self.state.draft_counts,
                        draft_to_sparse,
                    )
                    self.state.set_stages(decision.next_stage)
                    sampled_token_ids = self._synchronize_pipeline_control(
                        batch_size, sampled_token_ids
                    )

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

                    token_index, draft_stage_mask = self._prepare_next_draft_step(
                        batch_size,
                        draft_round_mask,
                        round_start_counts,
                    )

                verification = self._verify_draft_tokens(
                    batch_size,
                    request_ids,
                    proposal_round,
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
                    & (self.state.pending_counts < proposal_token_budgets)
                    & (self.positions[:batch_size] < self.max_model_len - 1)
                )

                require_full = (
                    draft_round_mask & (self.state.pending_counts > 0) & ~can_defer_full
                )

                self.state.set_stage(can_defer_full, RetroSpecStage.DRAFT)
                self.state.set_stage(require_full, RetroSpecStage.FULL_VERIFY)

                empty_round = draft_round_mask & (self.state.pending_counts == 0)
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
                next_round_input_ids = self._synchronize_pipeline_control(
                    batch_size, next_round_input_ids
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
        if not materialize_output:
            if self.performance_stats.enabled:
                self.performance_stats.record_cpu_time(
                    "proposal_wall",
                    perf_counter() - proposal_started_at,
                )
            self.performance_stats.maybe_log()
            return []

        pending_counts_cpu = self.state.pending_counts.cpu().tolist()
        pending_token_ids = self._draft_token_ids[:batch_size].cpu().tolist()

        result = [
            token_ids[:pending_count]
            for token_ids, pending_count in zip(pending_token_ids, pending_counts_cpu)
        ]
        if self.performance_stats.enabled:
            elapsed_seconds = perf_counter() - proposal_started_at
            first_request_count = 0
            for request_id, pending_count in zip(request_ids, pending_counts_cpu):
                if pending_count > 0:
                    self._last_proposed_counts[request_id] = pending_count
                    self._last_committed_proposal_counts.pop(request_id, None)
                    if request_id not in self._seen_proposal_request_ids:
                        self._seen_proposal_request_ids.add(request_id)
                        first_request_count += 1
            if first_request_count:
                self.performance_stats.add_counter(
                    "first_proposal_requests", first_request_count
                )
                self.performance_stats.record_cpu_time(
                    "first_proposal_batch_wall", elapsed_seconds
                )
            self.performance_stats.record_cpu_time("proposal_wall", elapsed_seconds)
        self.performance_stats.maybe_log()
        return result
