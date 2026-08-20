# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from .cluster_store import RetroSpecResolvedClusterPages
from .execution import (
    RetroSpecExactExecution,
    RetroSpecExactExecutionBuffer,
    RetroSpecExactKVSource,
    RetroSpecExactPageKVSource,
    RetroSpecExactPrimaryKVSource,
)
from .index import (
    RetroSpecAttentionLevel,
    RetroSpecAttentionSelection,
    RetroSpecBlockIndex,
    RetroSpecSelectionPlan,
)
from .segmented_index import (
    RetroSpecFullVerificationPlan,
    RetroSpecSegmentedTokenIndex,
    RetroSpecTokenAttentionSelection,
    RetroSpecTokenSelectionPlan,
)
from .weighted_attention import merge_weighted_estimation

RetroSpecSelection = RetroSpecAttentionSelection | RetroSpecTokenAttentionSelection
RetroSpecPlan = RetroSpecSelectionPlan | RetroSpecTokenSelectionPlan


@dataclass(frozen=True)
class _RetroSpecFullVerificationBatch:
    request_ids: tuple[str, ...]
    context_lens: tuple[int, ...]
    query_lens: tuple[int, ...]


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
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.block_size = block_size
        self.num_speculative_tokens = config.num_speculative_tokens

        if config.retrospec_index_mode == "block_mean":
            self.index = RetroSpecBlockIndex(
                block_size=block_size,
                num_speculative_tokens=config.num_speculative_tokens,
                retrieval_ratio=config.retrospec_retrieval_ratio,
                estimation_ratio=config.retrospec_estimation_ratio,
            )
        else:
            self.index = RetroSpecSegmentedTokenIndex(
                block_size=block_size,
                num_speculative_tokens=config.num_speculative_tokens,
                retrieval_ratio=config.retrospec_retrieval_ratio,
                estimation_ratio=config.retrospec_estimation_ratio,
                segment_size_tokens=config.retrospec_index_segment_size,
                blocks_per_cluster=config.retrospec_blocks_per_cluster,
                num_kmeans_iterations=config.retrospec_kmeans_iterations,
                max_pending_cluster_builds=config.retrospec_max_pending_cluster_builds,
                cache_mode=config.retrospec_cache_mode,
                cache_ratio=config.retrospec_cache_ratio,
                pin_memory=(
                    config.retrospec_cache_mode == "cpu_offload"
                    and is_pin_memory_available()
                ),
            )

        self.exact_execution_buffer: RetroSpecExactExecutionBuffer | None = None

        if isinstance(self.index, RetroSpecSegmentedTokenIndex):
            self.exact_execution_buffer = RetroSpecExactExecutionBuffer(
                page_size=block_size
            )

        self.proposal_request_ids: tuple[str, ...] = ()

        self.full_verification_batch: _RetroSpecFullVerificationBatch | None = None

        # request_id -> exclusive logical block boundary already retired.
        # Block 0 remains the permanent exact sink.
        self._retired_block_ends: dict[str, int] = {}

        self.prefill_index_active = False
        self.prefill_request_ids: tuple[str, ...] = ()
        self.prefill_seq_lens: tuple[int, ...] = ()
        self.prefill_build_rows: tuple[int, ...] = ()

        self.mode = RetroSpecAttentionMode.PASSTHROUGH
        self.in_proposal = False
        self.step_active = False
        self.step_index = -1
        self.active_mask: torch.Tensor | None = None
        self.batch_size = 0

        self.attention_mass_layer_count = 0
        self.attention_mass_sum = torch.zeros(
            self.max_batch_size, dtype=torch.float32, device=device
        )

        self.original_forwards: dict[str, tuple[FlashAttentionImpl, LayerForward]] = {}
        self.forward_wrappers: dict[str, _RetroSpecLayerForward] = {}

        self.selection_plans: list[dict[str, RetroSpecPlan]] = [
            {} for _ in range(self.num_speculative_tokens)
        ]

    @property
    def uses_full_verification_offload(self) -> bool:
        return (
            isinstance(self.index, RetroSpecSegmentedTokenIndex)
            and self.index.cluster_store.is_cpu_backed
        )

    @contextmanager
    def prefill_index_context(
        self,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        build_rows: Sequence[int],
    ) -> Iterator[None]:
        if self.in_proposal:
            raise RuntimeError("Cannot build a prefill index during a proposal")
        if self.prefill_index_active:
            raise RuntimeError("RetroSpec prefill index context cannot be nested")

        segmented_index = (
            self.index if isinstance(self.index, RetroSpecSegmentedTokenIndex) else None
        )
        if segmented_index is not None and segmented_index.has_staged_updates:
            raise RuntimeError("A previous RetroSpec prefill left staged index updates")

        self.prefill_index_active = True
        self.prefill_request_ids = tuple(request_ids)
        self.prefill_seq_lens = tuple(seq_lens)
        self.prefill_build_rows = tuple(build_rows)

        try:
            yield
        except BaseException:
            if segmented_index is not None:
                segmented_index.discard_staged_updates()
            raise
        else:
            if segmented_index is not None:
                segmented_index.flush_staged_updates()
        finally:
            self.prefill_index_active = False
            self.prefill_request_ids = ()
            self.prefill_seq_lens = ()
            self.prefill_build_rows = ()

    def needs_index_update(
        self,
        request_id: str,
        seq_len: int,
    ) -> bool:
        if not isinstance(self.index, RetroSpecSegmentedTokenIndex):
            return False

        return self.index.needs_update(
            request_id,
            seq_len,
            tuple(self.original_forwards),
        )

    def has_retired_kv_blocks(self, request_ids: Sequence[str]) -> bool:
        if not self.uses_full_verification_offload:
            return False

        return any(
            self._retired_block_ends.get(request_id, 1) > 1
            for request_id in request_ids
        )

    def take_kv_cache_retirement_ranges(
        self,
        request_ids: Sequence[str],
    ) -> list[tuple[str, int, int]]:
        """Return newly replaceable native block ranges."""
        if not self.uses_full_verification_offload:
            return []
        if not isinstance(self.index, RetroSpecSegmentedTokenIndex):
            return []
        if self.index.has_staged_updates:
            raise RuntimeError("Cannot retire KV blocks before index commit")

        layer_names = tuple(self.original_forwards)
        retirements: list[tuple[str, int, int]] = []

        for request_id in request_ids:
            indexed_end = self.index.get_fully_stored_indexed_end(
                request_id,
                layer_names,
            )
            new_end_block = indexed_end // self.block_size
            old_end_block = self._retired_block_ends.get(request_id, 1)

            if new_end_block < old_end_block:
                raise RuntimeError(
                    f"Stored index for {request_id!r} moved behind retired KV"
                )
            if new_end_block == old_end_block:
                continue

            retirements.append((request_id, old_end_block, new_end_block))
            self._retired_block_ends[request_id] = new_end_block

        return retirements

    def remove_requests(self, request_ids: Sequence[str]) -> None:
        request_ids = tuple(request_ids)

        if isinstance(self.index, RetroSpecSegmentedTokenIndex):
            self.index.remove_requests(request_ids)

        for request_id in request_ids:
            self._retired_block_ends.pop(request_id, None)

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

    @contextmanager
    def full_verification_context(
        self,
        request_ids: Sequence[str],
        context_lens: Sequence[int],
        query_lens: Sequence[int],
    ) -> Iterator[None]:
        if self.in_proposal:
            raise RuntimeError("Full verification cannot run during a proposal")
        if self.step_active:
            raise RuntimeError("A RetroSpec proposal step is still active")
        if self.mode != RetroSpecAttentionMode.PASSTHROUGH:
            raise RuntimeError("A RetroSpec attention context is already active")
        if not self.original_forwards:
            raise RuntimeError(
                "RetroSpec attention must be installed before full verification"
            )
        if not self.uses_full_verification_offload:
            raise RuntimeError(
                "Full-verification offload requires a CPU-backed segmented index"
            )

        request_ids = tuple(request_ids)
        context_lens = tuple(int(length) for length in context_lens)
        query_lens = tuple(int(length) for length in query_lens)

        if not request_ids:
            raise ValueError("Full verification requires at least one request")
        if len(context_lens) != len(request_ids):
            raise ValueError("context_lens must match request_ids")
        if len(query_lens) != len(request_ids):
            raise ValueError("query_lens must match request_ids")
        if any(length < 0 for length in context_lens):
            raise ValueError("Full-verification context lengths must be non-negative")
        if any(length <= 0 for length in query_lens):
            raise ValueError("Full-verification query lengths must be positive")

        for request_id, context_len in zip(request_ids, context_lens):
            retired_end_block = self._retired_block_ends.get(request_id, 1)
            if retired_end_block <= 1:
                continue

            retired_end = retired_end_block * self.block_size
            if context_len < retired_end:
                raise RuntimeError(
                    f"Request {request_id!r} rolled back to {context_len}, "
                    f"behind retired KV boundary {retired_end}"
                )

        assert isinstance(self.index, RetroSpecSegmentedTokenIndex)
        self.index.prepare_full_verification(
            request_ids,
            context_lens,
            tuple(self.original_forwards),
        )

        self.full_verification_batch = _RetroSpecFullVerificationBatch(
            request_ids=request_ids,
            context_lens=context_lens,
            query_lens=query_lens,
        )
        self.mode = RetroSpecAttentionMode.FULL_VERIFY

        try:
            yield
        finally:
            self.mode = RetroSpecAttentionMode.PASSTHROUGH
            self.full_verification_batch = None

    @contextmanager
    def proposal_context(
        self,
        request_ids: Sequence[str],
    ) -> Iterator[None]:
        if self.in_proposal:
            raise RuntimeError("RetroSpec proposal context cannot be nested.")
        if not self.original_forwards:
            raise RuntimeError(
                "RetroSpec attention must be installed before proposing."
            )
        self.proposal_request_ids = tuple(request_ids)
        if isinstance(self.index, RetroSpecSegmentedTokenIndex):
            self.index.begin_proposal(request_ids)

        for plans in self.selection_plans:
            plans.clear()

        self.in_proposal = True
        try:
            yield
        finally:
            self.in_proposal = False
            self.mode = RetroSpecAttentionMode.PASSTHROUGH
            self.step_active = False
            self.step_index = -1
            self.active_mask = None
            self.batch_size = 0
            self.attention_mass_layer_count = 0

            if isinstance(self.index, RetroSpecSegmentedTokenIndex):
                self.index.end_proposal()
            self.proposal_request_ids = ()

            for plans in self.selection_plans:
                plans.clear()

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

        self.attention_mass_sum[: self.batch_size].zero_()
        self.attention_mass_layer_count = 0
        self.step_active = True

    def end_step(self) -> torch.Tensor:
        if not self.step_active:
            raise RuntimeError("No RetroSpec attention step is active.")
        if self.attention_mass_layer_count == 0:
            raise RuntimeError("No attention layer ran during the RetroSpec step.")

        attention_mass = (
            self.attention_mass_sum[: self.batch_size] / self.attention_mass_layer_count
        )

        self.mode = RetroSpecAttentionMode.PASSTHROUGH
        self.step_active = False
        self.step_index = -1
        self.active_mask = None
        self.batch_size = 0
        self.attention_mass_layer_count = 0

        return attention_mass

    def _maybe_update_prefill_index(
        self,
        layer_name: str,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> None:
        if (
            not self.prefill_index_active
            or not self.prefill_build_rows
            or not isinstance(self.index, RetroSpecSegmentedTokenIndex)
        ):
            return

        key_cache, value_cache = kv_cache.unbind(0)
        self.index.build_or_update(
            layer_name=layer_name,
            request_ids=self.prefill_request_ids,
            seq_lens=self.prefill_seq_lens,
            rows=self.prefill_build_rows,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            defer_cpu_store=True,
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
            self._maybe_update_prefill_index(layer_name, kv_cache, attn_metadata)
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
            result = self._full_verification_forward(
                layer_name,
                impl,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_update_prefill_index(layer_name, kv_cache, attn_metadata)
            return result

        if not self.step_active or self.active_mask is None:
            raise RuntimeError("RetroSpec attention ran without an active step")

        return self._sparse_forward(
            layer_name, impl, layer, query, kv_cache, attn_metadata, output
        )

    @staticmethod
    def _run_grouped_reference_attention(
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference grouped attention for weighted centroids and CPU fallback.

        Args:
            query:
                [batch, num_query_heads, head_size]
            keys/values:
                [batch, num_kv_heads, max_num_vectors, head_size]
            token_counts:
                [batch, num_kv_heads, max_num_vectors]. A value of one
                represents an exact token. A value larger than one represents
                an estimation centroid for that many tokens.
        """
        batch_size, num_query_heads, head_size = query.shape
        num_kv_heads = keys.shape[1]

        if values.shape != keys.shape:
            raise ValueError("Reference attention keys and values must match")
        if token_counts.shape != keys.shape[:3]:
            raise ValueError("Reference attention token counts do not match keys")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        num_queries_per_kv = num_query_heads // num_kv_heads
        grouped_query = query.float().view(
            batch_size,
            num_kv_heads,
            num_queries_per_kv,
            head_size,
        )

        logits = torch.einsum(
            "bhgd,bhmd->bhgm",
            grouped_query,
            keys.float(),
        )
        logits *= impl.scale

        token_counts_float = token_counts.float()
        valid_mask = token_counts > 0

        # Exact tokens have count one and therefore receive no correction.
        # Estimation centroids receive the same log(cluster_size) correction
        # as the existing RetroSpec estimation path.
        logits += torch.log(token_counts_float.clamp_min(1)).unsqueeze(2)
        logits.masked_fill_(
            ~valid_mask.unsqueeze(2),
            float("-inf"),
        )

        has_vectors = valid_mask.any(dim=2)
        safe_logits = torch.where(
            has_vectors[:, :, None, None],
            logits,
            torch.zeros_like(logits),
        )

        output_lse = torch.logsumexp(safe_logits, dim=-1)
        output_lse = torch.where(
            has_vectors[:, :, None],
            output_lse,
            torch.full_like(output_lse, float("-inf")),
        )

        safe_normalizer = torch.where(
            has_vectors[:, :, None],
            output_lse,
            torch.zeros_like(output_lse),
        )
        weights = torch.exp(logits - safe_normalizer.unsqueeze(-1))
        weights.masked_fill_(
            ~valid_mask.unsqueeze(2),
            0.0,
        )

        attention_output = torch.einsum(
            "bhgm,bhmd->bhgd",
            weights,
            values.float(),
        )
        attention_output = attention_output.reshape(
            batch_size,
            num_query_heads,
            head_size,
        ).to(query.dtype)

        output_lse = output_lse.reshape(
            batch_size,
            num_query_heads,
        )
        output_lse = output_lse.transpose(0, 1).contiguous()

        return attention_output, output_lse

    @staticmethod
    def _run_grouped_flash_exact_attention(
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        execution: RetroSpecExactExecution,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grouped FlashAttention over the packed execution buffer."""
        batch_size, num_query_heads, head_size = query.shape

        if execution.batch_size != batch_size:
            raise ValueError("Execution-buffer batch size does not match query")
        if execution.head_size != head_size:
            raise ValueError("Execution-buffer head size does not match query")

        num_kv_heads = execution.num_kv_heads
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        if batch_size == 0:
            empty_lse = torch.empty(
                num_query_heads,
                0,
                dtype=torch.float32,
                device=query.device,
            )
            return torch.empty_like(query), empty_lse

        if execution.max_exact_seq_len == 0:
            empty_lse = torch.full(
                (num_query_heads, batch_size),
                float("-inf"),
                dtype=torch.float32,
                device=query.device,
            )
            return torch.zeros_like(query), empty_lse

        num_queries_per_kv = num_query_heads // num_kv_heads
        num_grouped_sequences = batch_size * num_kv_heads

        grouped_query = (
            query.reshape(
                batch_size,
                num_kv_heads,
                num_queries_per_kv,
                head_size,
            )
            .reshape(
                num_grouped_sequences,
                num_queries_per_kv,
                head_size,
            )
            .contiguous()
        )

        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
        )

        grouped_output, grouped_lse = flash_attn_varlen_func(
            q=grouped_query,
            k=execution.keys,
            v=execution.values,
            out=None,
            cu_seqlens_q=execution.cu_seqlens_q,
            cu_seqlens_k=execution.cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=execution.max_exact_seq_len,
            softmax_scale=impl.scale,
            causal=False,
            alibi_slopes=None,
            window_size=[-1, -1],
            block_table=None,
            softcap=0.0,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            num_splits=0,
            s_aux=None,
        )

        exact_output = (
            grouped_output.reshape(
                batch_size,
                num_kv_heads,
                num_queries_per_kv,
                head_size,
            )
            .reshape(
                batch_size,
                num_query_heads,
                head_size,
            )
            .contiguous()
        )

        grouped_lse = grouped_lse.as_strided(
            (num_queries_per_kv, num_grouped_sequences),
            (num_grouped_sequences, 1),
        )
        grouped_lse.masked_fill_(
            execution.exact_seq_lens.unsqueeze(0) == 0,
            float("-inf"),
        )

        exact_lse = (
            grouped_lse.transpose(0, 1)
            .reshape(
                batch_size,
                num_kv_heads,
                num_queries_per_kv,
            )
            .reshape(
                batch_size,
                num_query_heads,
            )
            .transpose(0, 1)
            .contiguous()
        )

        return exact_output, exact_lse

    @staticmethod
    def _run_full_prefix_attention(
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        query_lens: tuple[int, ...],
        execution: RetroSpecExactExecution,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, num_query_heads, head_size = query.shape

        if execution.batch_size != len(query_lens):
            raise ValueError("Execution batch size does not match query_lens")
        if execution.head_size != head_size:
            raise ValueError("Execution head size does not match the query")
        if sum(query_lens) != num_tokens:
            raise ValueError("query_lens do not cover the full query tensor")

        num_kv_heads = execution.num_kv_heads
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        if execution.max_exact_seq_len == 0:
            prefix_lse = torch.full(
                (num_query_heads, num_tokens),
                float("-inf"),
                dtype=torch.float32,
                device=query.device,
            )
            return torch.zeros_like(query), prefix_lse

        num_queries_per_kv = num_query_heads // num_kv_heads
        grouped_query_chunks: list[torch.Tensor] = []
        query_offset = 0

        for query_len in query_lens:
            request_query = query[query_offset : query_offset + query_len]
            request_query = request_query.reshape(
                query_len,
                num_kv_heads,
                num_queries_per_kv,
                head_size,
            )
            grouped_query_chunks.append(
                request_query.permute(1, 0, 2, 3).reshape(
                    num_kv_heads * query_len,
                    num_queries_per_kv,
                    head_size,
                )
            )
            query_offset += query_len

        grouped_query = torch.cat(grouped_query_chunks).contiguous()
        grouped_query_lens = torch.tensor(
            query_lens,
            dtype=torch.int32,
            device=query.device,
        ).repeat_interleave(num_kv_heads)

        grouped_cu_seqlens_q = torch.zeros(
            grouped_query_lens.numel() + 1,
            dtype=torch.int32,
            device=query.device,
        )
        torch.cumsum(
            grouped_query_lens,
            dim=0,
            out=grouped_cu_seqlens_q[1:],
        )

        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        grouped_output, grouped_lse = flash_attn_varlen_func(
            q=grouped_query,
            k=execution.keys,
            v=execution.values,
            out=None,
            cu_seqlens_q=grouped_cu_seqlens_q,
            cu_seqlens_k=execution.cu_seqlens_k,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=execution.max_exact_seq_len,
            softmax_scale=impl.scale,
            causal=False,
            alibi_slopes=None,
            window_size=[-1, -1],
            block_table=None,
            softcap=0.0,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            num_splits=0,
            s_aux=None,
        )

        grouped_lse = grouped_lse.as_strided(
            (num_queries_per_kv, grouped_query.shape[0]),
            (grouped_query.shape[0], 1),
        )

        has_prefix = torch.repeat_interleave(
            execution.exact_seq_lens > 0,
            grouped_query_lens.to(torch.int64),
        )
        grouped_output.masked_fill_(~has_prefix[:, None, None], 0)
        grouped_lse.masked_fill_(~has_prefix.unsqueeze(0), float("-inf"))

        prefix_output_chunks: list[torch.Tensor] = []
        prefix_lse_chunks: list[torch.Tensor] = []
        grouped_offset = 0

        for query_len in query_lens:
            grouped_count = num_kv_heads * query_len
            request_output = grouped_output[
                grouped_offset : grouped_offset + grouped_count
            ]
            request_output = (
                request_output.reshape(
                    num_kv_heads,
                    query_len,
                    num_queries_per_kv,
                    head_size,
                )
                .permute(1, 0, 2, 3)
                .reshape(query_len, num_query_heads, head_size)
            )
            prefix_output_chunks.append(request_output)

            request_lse = grouped_lse[
                :,
                grouped_offset : grouped_offset + grouped_count,
            ]
            request_lse = (
                request_lse.reshape(
                    num_queries_per_kv,
                    num_kv_heads,
                    query_len,
                )
                .permute(2, 1, 0)
                .reshape(query_len, num_query_heads)
                .transpose(0, 1)
            )
            prefix_lse_chunks.append(request_lse)
            grouped_offset += grouped_count

        prefix_output = torch.cat(prefix_output_chunks).contiguous()
        prefix_lse = torch.cat(prefix_lse_chunks, dim=1).contiguous()
        return prefix_output, prefix_lse

    @staticmethod
    def _run_full_local_attention(
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        num_tokens = query.shape[0]
        num_query_heads = query.shape[1]

        local_output, local_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=None,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_query_len,
            softmax_scale=impl.scale,
            causal=True,
            alibi_slopes=None,
            window_size=[-1, -1],
            block_table=None,
            softcap=0.0,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            num_splits=0,
            s_aux=None,
        )

        local_lse = local_lse.as_strided(
            (num_query_heads, num_tokens),
            (num_tokens, 1),
        )
        return local_output, local_lse

    def _full_verification_forward(
        self,
        layer_name: str,
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        batch = self.full_verification_batch
        if batch is None:
            raise RuntimeError("No full-verification batch is active")
        if not isinstance(self.index, RetroSpecSegmentedTokenIndex):
            raise RuntimeError("Full verification requires the segmented index")
        if self.exact_execution_buffer is None:
            raise RuntimeError("RetroSpec exact execution buffer is not initialized")
        if not attn_metadata.causal:
            raise RuntimeError("Full verification requires causal decoder attention")

        num_actual_tokens = attn_metadata.num_actual_tokens
        if len(batch.request_ids) != attn_metadata.seq_lens.shape[0]:
            raise RuntimeError(
                "Full-verification request count does not match attention metadata"
            )
        if sum(batch.query_lens) != num_actual_tokens:
            raise RuntimeError(
                "Full-verification query lengths do not match num_actual_tokens"
            )
        if max(batch.query_lens) != attn_metadata.max_query_len:
            raise RuntimeError(
                "Full-verification query lengths do not match max_query_len"
            )

        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(0)

        plan = self.index.build_full_verification_plan(
            request_ids=batch.request_ids,
            layer_name=layer_name,
            seq_lens=batch.context_lens,
            key_cache=key_cache,
            block_table=attn_metadata.block_table,
        )
        source, _ = self._resolve_exact_kv_source(
            selection=plan,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
        )
        execution = self.exact_execution_buffer.pack(source)

        prefix_output, prefix_lse = self._run_full_prefix_attention(
            impl,
            query,
            batch.query_lens,
            execution,
        )
        local_output, local_lse = self._run_full_local_attention(
            impl,
            query,
            key,
            value,
            attn_metadata,
        )

        merge_attn_states(
            output[:num_actual_tokens],
            prefix_output,
            prefix_lse,
            local_output,
            local_lse,
        )
        return output

    def _resolve_exact_kv_source(
        self,
        selection: RetroSpecTokenAttentionSelection | RetroSpecFullVerificationPlan,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[RetroSpecExactKVSource, RetroSpecResolvedClusterPages | None]:
        if not isinstance(self.index, RetroSpecSegmentedTokenIndex):
            raise RuntimeError("Token selection requires the segmented index")

        full_verification = isinstance(selection, RetroSpecFullVerificationPlan)

        if full_verification:
            layer_name = selection.layer_name
            primary_token_indices = selection.primary_exact_token_indices
            primary_token_mask = selection.primary_exact_token_mask
            exact_cluster_ids = selection.exact_cluster_ids
            exact_page_ids = selection.exact_page_ids
            exact_page_token_counts = selection.exact_page_token_counts
            resolved_pages = None
        else:
            layer_name = selection.plan.layer_name
            primary_token_indices = selection.plan.primary_exact_token_indices
            primary_token_mask = selection.plan.primary_exact_token_mask
            exact_cluster_ids = selection.exact_cluster_ids
            exact_page_ids = selection.exact_page_ids
            exact_page_token_counts = selection.exact_page_token_counts
            resolved_pages = selection.resolved_pages

        if resolved_pages is None and exact_page_ids.numel():
            if full_verification:
                resolved_pages = (
                    self.index.cluster_store.resolve_full_verification_blocks(
                        layer_name=layer_name,
                        cluster_ids=exact_cluster_ids,
                        logical_page_ids=exact_page_ids,
                    )
                )
            else:
                resolved_pages = self.index.cluster_store.resolve_cluster_blocks(
                    layer_name=layer_name,
                    cluster_ids=exact_cluster_ids,
                    logical_page_ids=exact_page_ids,
                    mode="verification",
                )

        page_sources: tuple[RetroSpecExactPageKVSource, ...] = ()

        if resolved_pages is not None:
            page_sources = (
                RetroSpecExactPageKVSource(
                    key_pages=resolved_pages.resident_key_pages,
                    value_pages=resolved_pages.resident_value_pages,
                    page_ids=resolved_pages.resident_page_ids,
                    ready_event=resolved_pages.resident_ready_event,
                ),
                RetroSpecExactPageKVSource(
                    key_pages=resolved_pages.staging_key_pages,
                    value_pages=resolved_pages.staging_value_pages,
                    page_ids=resolved_pages.staging_page_ids,
                    ready_event=resolved_pages.staging_ready_event,
                ),
            )

        source = RetroSpecExactKVSource(
            primary=RetroSpecExactPrimaryKVSource(
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                token_indices=primary_token_indices,
                token_mask=primary_token_mask,
            ),
            page_token_counts=exact_page_token_counts,
            page_sources=page_sources,
        )

        return source, resolved_pages

    def _run_exact_attention(
        self,
        impl: FlashAttentionImpl,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(selection, RetroSpecTokenAttentionSelection):
            if not isinstance(
                self.index,
                RetroSpecSegmentedTokenIndex,
            ):
                raise RuntimeError("Token selection requires the segmented index")

            if query.device.type != "cuda" or query.dtype not in (
                torch.float16,
                torch.bfloat16,
            ):
                exact_keys, exact_values, exact_token_mask = (
                    self.index.materialize_exact_reference(
                        selection,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table,
                    )
                )

                if (
                    self.mode
                    in (
                        RetroSpecAttentionMode.SPARSE_VERIFY,
                        RetroSpecAttentionMode.EXPANDED_VERIFY,
                    )
                    and query.device.type == "cuda"
                    and self.index.cluster_store.is_cpu_backed
                    and selection.exact_cluster_ids.numel()
                ):
                    self.index.cluster_store.admit_resident_clusters(
                        layer_name=selection.plan.layer_name,
                        cluster_ids=selection.exact_cluster_ids,
                        page_ids=selection.exact_page_ids,
                    )

                return self._run_grouped_reference_attention(
                    impl,
                    query,
                    exact_keys,
                    exact_values,
                    exact_token_mask.to(torch.int32),
                )

            if self.exact_execution_buffer is None:
                raise RuntimeError(
                    "RetroSpec exact execution buffer is not initialized"
                )

            source, resolved_pages = self._resolve_exact_kv_source(
                selection=selection,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table,
            )
            execution = self.exact_execution_buffer.pack(source)

            if (
                self.mode
                in (
                    RetroSpecAttentionMode.SPARSE_VERIFY,
                    RetroSpecAttentionMode.EXPANDED_VERIFY,
                )
                and self.index.cluster_store.is_cpu_backed
                and selection.exact_cluster_ids.numel()
            ):
                if resolved_pages is None:
                    raise RuntimeError(
                        "Verification cache update requires resolved cluster pages"
                    )

                self.index.cluster_store.admit_staged_clusters(
                    layer_name=selection.plan.layer_name,
                    cluster_ids=selection.exact_cluster_ids,
                    logical_page_ids=selection.exact_page_ids,
                    staging_page_ids=resolved_pages.staging_page_ids,
                    staging_key_pages=resolved_pages.staging_key_pages,
                    staging_value_pages=resolved_pages.staging_value_pages,
                )

            return self._run_grouped_flash_exact_attention(
                impl,
                query,
                execution,
            )

        # block_mean mode continues to use the primary vLLM paged KV cache.
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
        )

        batch_size = query.shape[0]
        descale_shape = (batch_size, impl.num_kv_heads)

        q_descale = layer._q_scale.expand(descale_shape)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        exact_output, exact_lse = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=None,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=selection.exact_seq_lens,
            max_seqlen_k=(selection.exact_block_table.shape[1] * self.block_size),
            softmax_scale=impl.scale,
            causal=attn_metadata.causal,
            alibi_slopes=None,
            window_size=[-1, -1],
            block_table=selection.exact_block_table,
            softcap=0.0,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=0,
            s_aux=None,
        )

        return exact_output, exact_lse

    @staticmethod
    def _get_grouped_estimation(
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return estimation tensors in grouped KV-head layout."""
        if isinstance(selection, RetroSpecTokenAttentionSelection):
            return (
                selection.estimation_keys,
                selection.estimation_values,
                selection.estimation_token_counts,
            )

        # block_mean stores centroids as:
        #
        #   [batch, estimates, kv_heads, head_size]
        #
        # Both the reference and fused weighted paths consume:
        #
        #   [batch, kv_heads, estimates, head_size]
        estimation_keys = selection.estimation_keys.permute(
            0,
            2,
            1,
            3,
        ).contiguous()
        estimation_values = selection.estimation_values.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        num_kv_heads = estimation_keys.shape[1]
        estimation_token_counts = (
            selection.estimation_token_counts[:, None, :]
            .expand(-1, num_kv_heads, -1)
            .contiguous()
        )

        return (
            estimation_keys,
            estimation_values,
            estimation_token_counts,
        )

    @classmethod
    def _run_estimation_attention(
        cls,
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the reference weighted-centroid attention path."""
        (
            estimation_keys,
            estimation_values,
            estimation_token_counts,
        ) = cls._get_grouped_estimation(selection)

        return cls._run_grouped_reference_attention(
            impl,
            query,
            estimation_keys,
            estimation_values,
            estimation_token_counts,
        )

    def _sparse_forward(
        self,
        layer_name: str,
        impl: FlashAttentionImpl,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert self.active_mask is not None
        assert self.step_index >= 0

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens != self.batch_size:
            raise RuntimeError(
                "RetroSpec attention requires exactly one query token per request."
            )
        if attn_metadata.max_query_len != 1:
            raise RuntimeError("RetroSpec attention requires max_query_len=1.")

        query = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(0)

        if self.mode == RetroSpecAttentionMode.DRAFT:
            if isinstance(
                self.index,
                RetroSpecSegmentedTokenIndex,
            ):
                selection = self.index.select_segmented(
                    request_ids=self.proposal_request_ids,
                    layer_name=layer_name,
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    active_mask=self.active_mask,
                    scale=impl.scale,
                )
            else:
                selection = self.index.select(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    self.active_mask,
                    impl.scale,
                )
            self.selection_plans[self.step_index][layer_name] = selection.plan
        else:
            try:
                plan = self.selection_plans[self.step_index][layer_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"No draft selection plan for step "
                    f"{self.step_index}, layer {layer_name!r}."
                ) from exc

            if self.mode == RetroSpecAttentionMode.SPARSE_VERIFY:
                level = RetroSpecAttentionLevel.SPARSE
            elif self.mode == RetroSpecAttentionMode.EXPANDED_VERIFY:
                level = RetroSpecAttentionLevel.EXPANDED
            else:
                raise RuntimeError(f"Unexpected RetroSpec attention mode: {self.mode}")

            selection = self.index.materialize(
                plan, level, key_cache, value_cache, attn_metadata.block_table
            )

        exact_output, exact_lse = self._run_exact_attention(
            impl,
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            selection,
        )

        can_use_fused_estimation = query.device.type == "cuda" and query.dtype in (
            torch.float16,
            torch.bfloat16,
        )

        if can_use_fused_estimation:
            (
                estimation_keys,
                estimation_values,
                estimation_token_counts,
            ) = self._get_grouped_estimation(selection)

            merge_weighted_estimation(
                output=output[:num_actual_tokens],
                query=query,
                estimation_keys=estimation_keys,
                estimation_values=estimation_values,
                estimation_token_counts=estimation_token_counts,
                exact_output=exact_output,
                exact_lse=exact_lse,
                scale=impl.scale,
            )
        else:
            estimation_output, estimation_lse = self._run_estimation_attention(
                impl,
                query,
                selection,
            )

            merge_attn_states(
                output[:num_actual_tokens],
                exact_output,
                exact_lse,
                estimation_output,
                estimation_lse,
            )

        self.attention_mass_sum[: self.batch_size].add_(selection.attention_mass)
        self.attention_mass_layer_count += 1

        return output
