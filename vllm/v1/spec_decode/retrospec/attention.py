# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import IntEnum

import torch

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

from .gpu_native import RetroSpecGPUNativeIndex
from .performance import RetroSpecPerformanceStats
from .pipeline import RetroSpecAttentionMassStats

LayerForward = Callable[..., torch.Tensor]


class RetroSpecAttentionMode(IntEnum):
    PASSTHROUGH = 0
    DRAFT = 1
    SPARSE_VERIFY = 2
    EXPANDED_VERIFY = 3
    FULL_VERIFY = 4


class _RetroSpecLayerForward:
    def __init__(
        self,
        controller: "RetroSpecSparseAttention",
        layer_name: str,
        original_forward: LayerForward,
    ) -> None:
        self.controller = controller
        self.layer_name = layer_name
        self.original_forward = original_forward

    def __call__(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.controller.forward(
            self.layer_name,
            self.original_forward,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class RetroSpecSparseAttention:
    """Override attention only while the RetroSpec drafter is running."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        config = vllm_config.speculative_config
        assert config is not None
        assert config.method == "retrospec"
        assert config.num_speculative_tokens is not None

        block_size = vllm_config.cache_config.block_size
        assert block_size is not None

        self.device = device
        if config.retrospec_replay_mode != "off":
            raise ValueError(
                "RetroSpec GPU-native mode does not support resident replay"
            )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        self.tensor_parallel_size = getattr(parallel_config, "tensor_parallel_size", 1)
        self.reduce_draft_attention_mass = (
            config.retrospec_hit_attn_threshold is not None
        )
        self.reduce_sparse_attention_mass = (
            config.retrospec_retrieval_attn_threshold is not None
        )
        self.reduce_expanded_attention_mass = (
            config.retrospec_expanded_attn_threshold is not None
        )
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.block_size = block_size
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_parallel_tokens = self.max_batch_size * self.num_speculative_tokens

        self.performance_stats = RetroSpecPerformanceStats(
            device=device,
            log_interval_seconds=getattr(
                config,
                "retrospec_stats_interval_seconds",
                0.0,
            ),
            histogram_max_value=config.num_speculative_tokens,
            cuda_timing_level=getattr(
                config,
                "retrospec_stats_cuda_timing_level",
                "coarse",
            ),
            cuda_sample_interval=getattr(
                config,
                "retrospec_stats_cuda_sample_interval",
                8,
            ),
        )
        self.index = RetroSpecGPUNativeIndex(
            block_size=block_size,
            num_speculative_tokens=config.num_speculative_tokens,
            retrieval_ratio=config.retrospec_retrieval_ratio,
            estimation_ratio=config.retrospec_estimation_ratio,
            sparse_verify_exact_fraction=getattr(
                config, "retrospec_sparse_verify_exact_fraction", 0.875
            ),
            prefill_segment_size_tokens=config.retrospec_index_segment_size,
            generation_update_interval=config.retrospec_index_update_interval,
            blocks_per_cluster=config.retrospec_blocks_per_cluster,
            num_kmeans_iterations=config.retrospec_kmeans_iterations,
            performance_stats=self.performance_stats,
        )

        self.proposal_request_ids: tuple[str, ...] = ()
        self.proposal_context_lens: tuple[int, ...] = ()
        self.proposal_round = 0

        self.index_update_active = False
        self.index_update_request_ids: tuple[str, ...] = ()
        self.index_update_seq_lens: tuple[int, ...] = ()
        self.index_update_is_prefill: tuple[bool, ...] = ()
        self.index_update_prefill_complete: tuple[bool, ...] = ()
        self.index_update_build_rows: tuple[int, ...] = ()
        self.mode = RetroSpecAttentionMode.PASSTHROUGH
        self.in_proposal = False
        self.step_active = False
        self.step_index = -1
        self.active_mask: torch.Tensor | None = None
        self.batch_size = 0
        self.parallel_request_indices: torch.Tensor | None = None
        self.parallel_token_indices: torch.Tensor | None = None
        self.parallel_bonus_start_index: int | None = None

        self.attention_mass_layer_count = 0
        self.attention_mass_sum = torch.zeros(
            self.max_parallel_tokens, dtype=torch.float32, device=device
        )

        self.original_forwards: dict[str, tuple[FlashAttentionImpl, LayerForward]] = {}
        self.forward_wrappers: dict[str, _RetroSpecLayerForward] = {}

    @property
    def uses_full_verification_offload(self) -> bool:
        return False

    @contextmanager
    def index_update_context(
        self,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        is_prefill: Sequence[bool],
        prefill_complete: Sequence[bool],
        build_rows: Sequence[int],
    ) -> Iterator[None]:
        if self.in_proposal:
            raise RuntimeError("Cannot update the index during a proposal")
        if self.index_update_active:
            raise RuntimeError("RetroSpec index update context cannot be nested")

        request_ids = tuple(request_ids)
        seq_lens = tuple(int(seq_len) for seq_len in seq_lens)
        is_prefill = tuple(bool(value) for value in is_prefill)
        prefill_complete = tuple(bool(value) for value in prefill_complete)
        build_rows = tuple(int(row) for row in build_rows)

        if len(seq_lens) != len(request_ids):
            raise ValueError("seq_lens must match request_ids")
        if len(is_prefill) != len(request_ids):
            raise ValueError("is_prefill must match request_ids")
        if len(prefill_complete) != len(request_ids):
            raise ValueError("prefill_complete must match request_ids")
        if any(
            complete and not prefill
            for complete, prefill in zip(prefill_complete, is_prefill)
        ):
            raise ValueError("prefill_complete requires is_prefill")
        if len(build_rows) != len(set(build_rows)):
            raise ValueError("build_rows must be unique")
        if any(row < 0 or row >= len(request_ids) for row in build_rows):
            raise IndexError("RetroSpec index build row is out of range")

        if self.index.has_staged_updates:
            raise RuntimeError("A previous RetroSpec update left staged index changes")
        self.index_update_active = True
        self.index_update_request_ids = request_ids
        self.index_update_seq_lens = seq_lens
        self.index_update_is_prefill = is_prefill
        self.index_update_prefill_complete = prefill_complete
        self.index_update_build_rows = build_rows

        try:
            yield
        except BaseException:
            self.index.discard_staged_updates()
            raise
        else:
            self.index.flush_staged_updates()
        finally:
            self.index_update_active = False
            self.index_update_request_ids = ()
            self.index_update_seq_lens = ()
            self.index_update_is_prefill = ()
            self.index_update_prefill_complete = ()
            self.index_update_build_rows = ()

    def needs_index_update(
        self,
        request_id: str,
        seq_len: int,
        is_prefill: bool,
        prefill_complete: bool,
    ) -> bool:
        return self.index.needs_update(
            request_id,
            seq_len,
            tuple(self.original_forwards),
            is_prefill,
            prefill_complete,
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
        del layer_name, request_id, seq_len, key_cache, value_cache, block_table
        raise NotImplementedError("GPU-native RetroSpec uses ordinary chunked prefill")

    @contextmanager
    def capture_layer_major_prefill_query(self, layer_name: str) -> Iterator[None]:
        del layer_name
        raise NotImplementedError("GPU-native RetroSpec uses ordinary chunked prefill")
        yield

    def commit_layer_major_prefill(
        self, request_id: str, layer_names: Sequence[str]
    ) -> None:
        del request_id, layer_names
        raise NotImplementedError("GPU-native RetroSpec uses ordinary chunked prefill")

    def abort_layer_major_prefill(self) -> None:
        self.index.discard_staged_updates()

    def has_retired_kv_blocks(self, request_ids: Sequence[str]) -> bool:
        del request_ids
        return False

    def take_kv_cache_retirement_ranges(
        self,
        request_ids: Sequence[str],
    ) -> list[tuple[str, int, int]]:
        del request_ids
        return []

    def remove_requests(self, request_ids: Sequence[str]) -> None:
        self.index.remove_requests(request_ids)

    @staticmethod
    def _validate_layer(
        layer_name: str,
        layer: Attention,
    ) -> FlashAttentionImpl:
        impl = layer.impl
        if not isinstance(impl, FlashAttentionImpl):
            raise NotImplementedError(
                f"RetroSpec sparse drafting requires FlashAttention, "
                f"but layer {layer_name!r} uses "
                f"{impl.__class__.__name__}."
            )

        if impl.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "RetroSpec currently supports decoder self-attention only."
            )
        if impl.dcp_world_size != 1:
            raise NotImplementedError(
                "RetroSpec sparse attention does not support DCP."
            )
        if impl.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "RetroSpec sparse attention does not support quantized KV caches."
            )
        if impl.kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "RetroSpec sparse attention does not support shared KV-cache layers."
            )
        if impl.alibi_slopes is not None:
            raise NotImplementedError(
                "RetroSpec sparse attention does not support ALiBi."
            )
        if impl.sliding_window != (-1, -1):
            raise NotImplementedError(
                "RetroSpec sparse attention does not support sliding-window attention."
            )
        if impl.logits_soft_cap != 0:
            raise NotImplementedError(
                "RetroSpec sparse attention does not support logits softcap."
            )
        if impl.sinks is not None:
            raise NotImplementedError(
                "RetroSpec sparse attention does not support attention sinks."
            )
        if impl.vllm_flash_attn_version not in (2, 3):
            raise NotImplementedError(
                "RetroSpec sparse attention requires FlashAttention 2 or 3."
            )

        return impl

    def install(
        self,
        layers: Mapping[str, Attention],
    ) -> None:
        if self.original_forwards:
            raise RuntimeError("RetroSpec sparse attention is already installed.")
        if not layers:
            raise RuntimeError("No attention layers were provided to RetroSpec.")

        validated_layers: dict[str, FlashAttentionImpl] = {}
        for layer_name, layer in layers.items():
            validated_layers[layer_name] = self._validate_layer(layer_name, layer)

        self.index.configure_sparse_prefetch_wave(len(validated_layers))
        for layer_name, impl in validated_layers.items():
            original_forward = impl.forward
            wrapper = _RetroSpecLayerForward(self, layer_name, original_forward)

            self.original_forwards[layer_name] = (impl, original_forward)
            self.forward_wrappers[layer_name] = wrapper
            impl.forward = wrapper  # type: ignore[method-assign]

    def uninstall(self) -> None:
        if self.in_proposal:
            raise RuntimeError(
                "Cannot uninstall RetroSpec attention during a proposal."
            )

        for impl, original_forward in self.original_forwards.values():
            impl.forward = original_forward  # type: ignore[method-assign]

        self.original_forwards.clear()
        self.forward_wrappers.clear()
        try:
            self.index.close()
        finally:
            self.performance_stats.flush("shutdown")

    @contextmanager
    def full_verification_context(
        self,
        request_ids: Sequence[str],
        context_lens: Sequence[int],
        query_lens: Sequence[int],
    ) -> Iterator[None]:
        del request_ids, context_lens, query_lens
        raise RuntimeError(
            "GPU-native full verification uses the original vLLM attention path"
        )
        yield  # Keep the context-manager contract for callers.

    @contextmanager
    def proposal_context(
        self,
        request_ids: Sequence[str],
        context_lens: Sequence[int] | None = None,
    ) -> Iterator[None]:
        if self.in_proposal:
            raise RuntimeError("RetroSpec proposal context cannot be nested.")
        if not self.original_forwards:
            raise RuntimeError(
                "RetroSpec attention must be installed before proposing."
            )
        request_ids = tuple(request_ids)
        if context_lens is None:
            normalized_context_lens = ()
        else:
            normalized_context_lens = tuple(int(length) for length in context_lens)
            if len(normalized_context_lens) != len(request_ids):
                raise ValueError("context_lens must match request_ids")
            if any(length < 0 for length in normalized_context_lens):
                raise ValueError("Proposal context lengths must be non-negative")
        self.index.begin_proposal(request_ids)

        try:
            self.proposal_request_ids = request_ids
            self.proposal_context_lens = normalized_context_lens
            self.proposal_round = 0

            self.in_proposal = True
            yield
        finally:
            self.in_proposal = False
            self.mode = RetroSpecAttentionMode.PASSTHROUGH
            self.step_active = False
            self.step_index = -1
            self.active_mask = None
            self.batch_size = 0
            self.parallel_request_indices = None
            self.parallel_token_indices = None
            self.attention_mass_layer_count = 0
            self.proposal_round = 0

            try:
                self.index.flush_sparse_verification_prefetch()
            finally:
                self.index.end_proposal()
                self.proposal_request_ids = ()
                self.proposal_context_lens = ()

    def set_proposal_round(self, proposal_round: int) -> None:
        if not self.in_proposal:
            raise RuntimeError("Proposal round may be set only inside proposal_context")
        if proposal_round <= 0:
            raise ValueError("proposal_round must be positive")
        if proposal_round < self.proposal_round:
            raise ValueError("proposal_round must be monotonic")
        self.proposal_round = proposal_round

    @property
    def selection_provenance_enabled(self) -> bool:
        return False

    def begin_step(
        self,
        mode: RetroSpecAttentionMode,
        step_index: int,
        active_mask: torch.Tensor,
    ) -> None:
        if not self.in_proposal:
            raise RuntimeError("begin_step must be called inside proposal_context.")
        if mode == RetroSpecAttentionMode.PASSTHROUGH:
            raise ValueError("PASSTHROUGH cannot be used as an active RetroSpec step.")
        if self.step_active:
            raise RuntimeError("The previous RetroSpec attention step is still active.")
        if not 0 <= step_index < self.num_speculative_tokens:
            raise ValueError("step_index is outside the speculative token range.")
        if active_mask.ndim != 1 or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a one-dimensional boolean tensor.")
        if active_mask.device != self.device:
            raise ValueError(
                f"active_mask must be on {self.device}, but is on {active_mask.device}."
            )
        if active_mask.shape[0] > self.max_batch_size:
            raise ValueError("active_mask exceeds the configured maximum batch size.")

        self.mode = mode
        self.step_index = step_index
        self.batch_size = active_mask.shape[0]
        self.active_mask = active_mask
        self.parallel_request_indices = None
        self.parallel_token_indices = None
        self.parallel_bonus_start_index = None

        self.attention_mass_sum[: self.batch_size].zero_()
        self.attention_mass_layer_count = 0
        self.step_active = True

    def begin_parallel_step(
        self,
        mode: RetroSpecAttentionMode,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        bonus_start_index: int | None = None,
    ) -> None:
        if not self.in_proposal:
            raise RuntimeError(
                "begin_parallel_step must be called inside proposal_context."
            )
        if mode not in (
            RetroSpecAttentionMode.SPARSE_VERIFY,
            RetroSpecAttentionMode.EXPANDED_VERIFY,
        ):
            raise ValueError("Parallel steps are supported only for verification.")
        if self.step_active:
            raise RuntimeError("The previous RetroSpec attention step is still active.")
        if request_indices.ndim != 1 or token_indices.ndim != 1:
            raise ValueError("Parallel plan indices must be one-dimensional.")
        if request_indices.shape != token_indices.shape:
            raise ValueError(
                "request_indices and token_indices must have equal shapes."
            )
        if request_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("request_indices must use an integer dtype.")
        if token_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_indices must use an integer dtype.")
        if request_indices.device != self.device or token_indices.device != self.device:
            raise ValueError("Parallel plan indices must be on the attention device.")

        num_tokens = request_indices.shape[0]
        if num_tokens == 0:
            raise ValueError("A parallel verification step cannot be empty.")
        if num_tokens > self.max_parallel_tokens:
            raise ValueError("Parallel verification exceeds the configured capacity.")
        if bonus_start_index is not None:
            if mode != RetroSpecAttentionMode.SPARSE_VERIFY:
                raise ValueError("Bonus rows require sparse verification.")
            if not 0 < bonus_start_index < num_tokens:
                raise ValueError("bonus_start_index must split ordinary and bonus rows")
        self.mode = mode
        self.step_index = -1
        self.batch_size = num_tokens
        self.active_mask = torch.ones(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.parallel_request_indices = request_indices
        self.parallel_token_indices = token_indices
        self.parallel_bonus_start_index = bonus_start_index

        self.attention_mass_sum[: self.batch_size].zero_()
        self.attention_mass_layer_count = 0
        self.step_active = True

    def _should_reduce_attention_mass(self) -> bool:
        if self.tensor_parallel_size == 1:
            return False
        if self.mode == RetroSpecAttentionMode.DRAFT:
            return self.reduce_draft_attention_mass
        if self.mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return self.reduce_sparse_attention_mass
        if self.mode == RetroSpecAttentionMode.EXPANDED_VERIFY:
            return self.reduce_expanded_attention_mass
        return False

    def _synchronize_attention_mass_sum(
        self,
        attention_mass_sum: torch.Tensor,
    ) -> torch.Tensor:
        if not self._should_reduce_attention_mass():
            return attention_mass_sum

        attention_mass_sum = tensor_model_parallel_all_reduce(attention_mass_sum)
        return attention_mass_sum / self.tensor_parallel_size

    def end_step_statistics(self) -> RetroSpecAttentionMassStats:
        if not self.step_active:
            raise RuntimeError("No RetroSpec attention step is active.")
        if self.attention_mass_layer_count == 0:
            raise RuntimeError("No attention layer ran during the RetroSpec step.")

        layer_count = self.attention_mass_layer_count
        attention_mass_sum = self.attention_mass_sum[: self.batch_size]
        attention_mass_sum = self._synchronize_attention_mass_sum(attention_mass_sum)

        if self.parallel_request_indices is not None:
            self.index.end_indexed_verification_transaction()

        self.mode = RetroSpecAttentionMode.PASSTHROUGH
        self.step_active = False
        self.step_index = -1
        self.active_mask = None
        self.batch_size = 0
        self.parallel_request_indices = None
        self.parallel_token_indices = None
        self.parallel_bonus_start_index = None
        self.attention_mass_layer_count = 0

        return RetroSpecAttentionMassStats(
            value_sum=attention_mass_sum,
            layer_count=layer_count,
        )

    def abort_step(self) -> None:
        """Release verification state after a failed model forward."""
        try:
            self.index.end_indexed_verification_transaction()
        finally:
            self.mode = RetroSpecAttentionMode.PASSTHROUGH
            self.step_active = False
            self.step_index = -1
            self.active_mask = None
            self.batch_size = 0
            self.parallel_request_indices = None
            self.parallel_token_indices = None
            self.parallel_bonus_start_index = None
            self.attention_mass_layer_count = 0

    def end_step(self) -> torch.Tensor:
        return self.end_step_statistics().mean()

    def flush_sparse_verification_prefetch(self) -> None:
        self.index.flush_sparse_verification_prefetch()

    def maybe_prime_full_verification(self, num_candidate_tokens: int) -> bool:
        """Native full verification has no pages to prefetch."""
        if not self.in_proposal:
            raise RuntimeError("Full-verification priming requires a proposal")
        del num_candidate_tokens
        return False

    def _maybe_update_index(
        self,
        layer_name: str,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> None:
        if not self.index_update_active or not self.index_update_build_rows:
            return

        key_cache, value_cache = kv_cache.unbind(0)
        self.index.build_or_update(
            layer_name=layer_name,
            request_ids=self.index_update_request_ids,
            seq_lens=self.index_update_seq_lens,
            is_prefill=self.index_update_is_prefill,
            rows=self.index_update_build_rows,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            prefill_complete=self.index_update_prefill_complete,
        )

    def forward(
        self,
        layer_name: str,
        original_forward: LayerForward,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == RetroSpecAttentionMode.PASSTHROUGH:
            result = original_forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
            self._maybe_update_index(layer_name, kv_cache, attn_metadata)
            return result

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "RetroSpec attention does not support fused output quantization"
            )
        if output is None:
            raise RuntimeError("RetroSpec FlashAttention requires an output buffer")
        if attn_metadata is None:
            raise RuntimeError("RetroSpec attention requires attention metadata")

        impl = getattr(layer, "impl", None)
        if not isinstance(impl, FlashAttentionImpl):
            raise RuntimeError(
                "RetroSpec attention wrapper received an incompatible layer"
            )

        if self.mode == RetroSpecAttentionMode.FULL_VERIFY:
            raise RuntimeError(
                "GPU-native full verification must use the original attention"
            )

        if not self.step_active or self.active_mask is None:
            raise RuntimeError("RetroSpec attention ran without an active step")

        return self._sparse_forward(
            layer_name,
            original_forward,
            impl,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )

    def _sparse_forward(
        self,
        layer_name: str,
        original_forward: LayerForward,
        impl: FlashAttentionImpl,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert self.active_mask is not None
        has_parallel_plan = self.parallel_request_indices is not None
        assert self.step_index >= 0 or has_parallel_plan

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens != self.batch_size:
            raise RuntimeError(
                "RetroSpec attention requires exactly one query token per request."
            )
        if attn_metadata.max_query_len != 1:
            raise RuntimeError("RetroSpec attention requires max_query_len=1.")

        if not self.index.has_cluster_pages(layer_name, self.proposal_request_ids):
            self.performance_stats.add_counter("proposal_native_fallback_layers")
            result = original_forward(
                layer, query, key, value, kv_cache, attn_metadata, output, None, None
            )
            self.attention_mass_sum[: self.batch_size].add_(1.0)
            self.attention_mass_layer_count += 1
            return result

        if self.mode not in (
            RetroSpecAttentionMode.DRAFT,
            RetroSpecAttentionMode.SPARSE_VERIFY,
            RetroSpecAttentionMode.EXPANDED_VERIFY,
        ):
            raise RuntimeError(f"Unexpected RetroSpec attention mode: {self.mode}")

        request_indices = self.parallel_request_indices
        token_indices = self.parallel_token_indices
        if self.mode != RetroSpecAttentionMode.DRAFT and request_indices is None:
            request_indices = torch.arange(
                num_actual_tokens, dtype=torch.int32, device=query.device
            )
            token_indices = torch.full_like(request_indices, self.step_index)

        key_cache, value_cache = kv_cache.unbind(0)
        attention_mass = self.index.forward(
            layer_name=layer_name,
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            active_mask=self.active_mask,
            scale=impl.scale,
            output=output[:num_actual_tokens],
            step=self.step_index,
            expanded=self.mode == RetroSpecAttentionMode.EXPANDED_VERIFY,
            sparse_verify=self.mode == RetroSpecAttentionMode.SPARSE_VERIFY,
            request_indices=request_indices,
            token_indices=token_indices,
            bonus_start_index=self.parallel_bonus_start_index,
        )
        self.performance_stats.add_counter("gpu_native_cluster_attention_layers")
        self.attention_mass_sum[: self.batch_size].add_(attention_mass)
        self.attention_mass_layer_count += 1
        return output
