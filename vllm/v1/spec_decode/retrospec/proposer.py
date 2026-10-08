# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
)
from vllm.v1.utils import CpuGpuBuffer

from .attention import RetroSpecSparseAttention
from .decision import RetroSpecDecisionPolicy, RetroSpecMetrics
from .pipeline import (
    RetroSpecPipelineProtocol,
    RetroSpecPipelineStage,
)
from .prefill import resolve_retrospec_layer_model
from .state import RetroSpecBatchState, RetroSpecIndexUpdateState, RetroSpecStage
from .transition_trace import RetroSpecTransitionTracer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


from vllm.v1.spec_decode.retrospec.proposal.draft import RetroSpecDraftMixin
from vllm.v1.spec_decode.retrospec.proposal.feedback import RetroSpecFeedbackMixin
from vllm.v1.spec_decode.retrospec.proposal.model import RetroSpecModelMixin
from vllm.v1.spec_decode.retrospec.proposal.sampling import RetroSpecSamplingMixin
from vllm.v1.spec_decode.retrospec.proposal.types import (
    RetroSpecParallelVerificationOutput,
    RetroSpecVerificationResult,
)
from vllm.v1.spec_decode.retrospec.proposal.verification import (
    RetroSpecVerificationMixin,
)

__all__ = [
    "RetroSpecProposer",
    "RetroSpecParallelVerificationOutput",
    "RetroSpecVerificationResult",
]


class RetroSpecProposer(
    RetroSpecFeedbackMixin,
    RetroSpecModelMixin,
    RetroSpecDraftMixin,
    RetroSpecSamplingMixin,
    RetroSpecVerificationMixin,
):
    _CUDAGRAPH_NAMESPACE = "retrospec_proposal"
    _LOW_ACCEPTANCE_HORIZON = 16
    _LOW_ACCEPTANCE_MIN_PROPOSAL = 32
    _LOW_ACCEPTANCE_RECOVERY_ROUNDS = 3

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
        self._sparse_bonus_enabled = device.type == "cuda"
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
        self._verification_bonus_requests = torch.empty(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self._verification_bonus_candidates = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._verification_bonus_admitted = torch.zeros_like(
            self._verification_bonus_candidates
        )
        self._verification_bonus_token_ids = torch.full(
            (self.max_batch_size,), -1, dtype=torch.int32, device=device
        )
        self._bonus_ready_mask = torch.zeros_like(self._verification_bonus_candidates)
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
        # Single-rank output bookkeeping supplies host feedback. Multi-rank
        # workers derive the same feedback from target results on each GPU.
        parallel_config = vllm_config.parallel_config
        self._feedback_horizon_enabled = (
            device.type == "cuda"
            and getattr(parallel_config, "tensor_parallel_size", 1) == 1
            and getattr(parallel_config, "pipeline_parallel_size", 1) == 1
            and self.num_speculative_tokens > self._LOW_ACCEPTANCE_HORIZON
        )
        self._feedback_device_enabled = (
            device.type == "cuda"
            and not self._feedback_horizon_enabled
            and self.num_speculative_tokens > self._LOW_ACCEPTANCE_HORIZON
        )
        self._feedback_horizon_budgets = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
            pin_memory=is_pin_memory_available(),
            with_numpy=True,
        )
        self._feedback_horizons: dict[str, int] = {}
        self._feedback_recovery: dict[str, int] = {}
        self._feedback_device_horizons = torch.full(
            (self.max_batch_size,),
            self.num_speculative_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._feedback_device_recovery = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self._feedback_request_slots: dict[str, int] = {}
        self._feedback_free_slots = list(range(self.max_batch_size - 1, -1, -1))
        self._feedback_slot_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
            pin_memory=is_pin_memory_available(),
            with_numpy=True,
        )
        self._feedback_prior_counts = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
            pin_memory=is_pin_memory_available(),
            with_numpy=True,
        )
        self._seen_proposal_request_ids: set[str] = set()
        self._last_proposed_counts: dict[str, int] = {}
        self._last_committed_proposal_counts: dict[str, int] = {}
        self._closed = False

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
    ) -> None:
        self.sparse_attention.stage_layer_major_prefill_layer(
            layer_name,
            request_id,
            seq_len,
            key_cache,
            value_cache,
            block_table,
        )

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
        previous_proposed_counts: Sequence[int] | None = None,
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
            request_ids,
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
        self._bonus_ready_mask[:batch_size].zero_()

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
                if (
                    verification.bonus_mask is not None
                    and verification.bonus_token_ids is not None
                ):
                    bonus_ready = verification.bonus_mask & can_defer_full
                    bonus_steps = (
                        self.state.pending_counts.clamp(
                            max=self.num_speculative_tokens - 1
                        )
                        .long()
                        .unsqueeze(1)
                    )
                    existing_tokens = self._draft_token_ids[:batch_size].gather(
                        1, bonus_steps
                    )
                    bonus_tokens = torch.where(
                        bonus_ready,
                        verification.bonus_token_ids,
                        existing_tokens.squeeze(1),
                    )
                    self._draft_token_ids[:batch_size].scatter_(
                        1, bonus_steps, bonus_tokens.unsqueeze(1)
                    )
                    self.positions[:batch_size].add_(bonus_ready.to(torch.int64))
                    self._bonus_ready_mask[:batch_size].copy_(bonus_ready)
                    self.performance_stats.add_gpu_counter(
                        "sparse_bonus_admitted", bonus_ready
                    )
                    next_round_input_ids = torch.where(
                        bonus_ready, bonus_tokens, next_round_input_ids
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
        self._update_device_feedback_horizon(
            request_ids, previous_proposed_counts, valid_sampled_tokens_count
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
