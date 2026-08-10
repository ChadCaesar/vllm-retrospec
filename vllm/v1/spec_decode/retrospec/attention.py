# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from .index import RetroSpecAttentionSelection, RetroSpecBlockIndex

LayerForward = Callable[..., torch.Tensor]


class _RetroSpecLayerForward:
    def __init__(
        self,
        controller: "RetroSpecSparseAttention",
        original_forward: LayerForward,
    ) -> None:
        self.controller = controller
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

        self.index = RetroSpecBlockIndex(
            block_size=block_size,
            num_speculative_tokens=config.num_speculative_tokens,
            retrieval_ratio=config.retrospec_retrieval_ratio,
            estimation_ratio=config.retrospec_estimation_ratio,
        )

        self.enabled = False
        self.step_active = False
        self.active_mask: torch.Tensor | None = None
        self.batch_size = 0
        self.hit_attn_layer_count = 0

        self.hit_attn_sum = torch.zeros(
            self.max_batch_size, dtype=torch.float32, device=device
        )

        self.original_forwards: dict[str, tuple[FlashAttentionImpl, LayerForward]] = {}
        self.forward_wrappers: dict[str, _RetroSpecLayerForward] = {}

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
            wrapper = _RetroSpecLayerForward(self, original_forward)

            self.original_forwards[layer_name] = (impl, original_forward)
            self.forward_wrappers[layer_name] = wrapper
            impl.forward = wrapper  # type: ignore[method-assign]

    def uninstall(self) -> None:
        if self.enabled:
            raise RuntimeError("Cannot uninstall RetroSpec attention during drafting.")

        for impl, original_forward in self.original_forwards.values():
            impl.forward = original_forward  # type: ignore[method-assign]

        self.original_forwards.clear()
        self.forward_wrappers.clear()

    @contextmanager
    def draft_context(self) -> Iterator[None]:
        if self.enabled:
            raise RuntimeError("RetroSpec draft attention context cannot be nested.")
        if not self.original_forwards:
            raise RuntimeError("RetroSpec attention must be installed before drafting.")

        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False
            self.step_active = False
            self.active_mask = None
            self.batch_size = 0
            self.hit_attn_layer_count = 0

    def begin_step(
        self,
        active_mask: torch.Tensor,
    ) -> None:
        if not self.enabled:
            raise RuntimeError("begin_step must be called inside draft_context.")
        if self.step_active:
            raise RuntimeError("The previous RetroSpec draft step is still active.")
        if active_mask.ndim != 1 or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a one-dimensional boolean tensor.")
        if active_mask.device != self.device:
            raise ValueError(
                f"active_mask must be on {self.device}, but is on {active_mask.device}."
            )
        if active_mask.shape[0] > self.max_batch_size:
            raise ValueError("active_mask exceeds the configured maximum batch size.")

        self.batch_size = active_mask.shape[0]
        self.active_mask = active_mask
        self.hit_attn_sum[: self.batch_size].zero_()
        self.hit_attn_layer_count = 0
        self.step_active = True

    def end_step(self) -> torch.Tensor:
        if not self.step_active:
            raise RuntimeError("No RetroSpec draft attention step is active.")
        if self.hit_attn_layer_count == 0:
            raise RuntimeError(
                "No attention layer ran during the RetroSpec draft step."
            )

        hit_attn = self.hit_attn_sum[: self.batch_size] / self.hit_attn_layer_count

        self.step_active = False
        self.active_mask = None
        self.batch_size = 0
        self.hit_attn_layer_count = 0
        return hit_attn

    def forward(
        self,
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
        if not self.enabled:
            return original_forward(
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

        if not self.step_active or self.active_mask is None:
            raise RuntimeError("RetroSpec attention ran without an active draft step.")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "RetroSpec sparse attention does not support fused output quantization."
            )
        if output is None:
            raise RuntimeError("RetroSpec FlashAttention requires an output buffer.")
        if attn_metadata is None:
            raise RuntimeError("RetroSpec draft attention requires attention metadata.")

        impl = getattr(layer, "impl", None)
        if not isinstance(impl, FlashAttentionImpl):
            raise RuntimeError(
                "RetroSpec attention wrapper received an incompatible layer."
            )

        return self._draft_forward(impl, layer, query, kv_cache, attn_metadata, output)

    def _run_exact_attention(
        self,
        impl: FlashAttentionImpl,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        selection: RetroSpecAttentionSelection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The FlashAttention function is platform-specific and is not defined
        # by fa_utils on CPU. Import it only on the GPU execution path so that
        # RetroSpec state and policy tests remain importable on CPU.
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

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
    def _run_estimation_attention(
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        selection: RetroSpecAttentionSelection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_query_heads, head_size = query.shape
        num_kv_heads = selection.estimation_keys.shape[2]

        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads."
            )

        num_queries_per_kv = num_query_heads // num_kv_heads
        grouped_query = query.float().view(
            batch_size, num_kv_heads, num_queries_per_kv, head_size
        )

        logits = torch.einsum(
            "bhgd,bmhd->bhgm", grouped_query, selection.estimation_keys
        )
        logits *= impl.scale

        token_counts = selection.estimation_token_counts.float()
        estimation_mask = token_counts > 0

        logits += torch.log(token_counts.clamp_min(1))[:, None, None, :]
        logits.masked_fill_(~estimation_mask[:, None, None, :], float("-inf"))

        has_estimation = estimation_mask.any(dim=1)
        safe_logits = torch.where(
            has_estimation[:, None, None, None], logits, torch.zeros_like(logits)
        )

        estimation_lse = torch.logsumexp(safe_logits, dim=-1)
        estimation_lse = torch.where(
            has_estimation[:, None, None],
            estimation_lse,
            torch.full_like(estimation_lse, float("-inf")),
        )

        safe_normalizer = torch.where(
            has_estimation[:, None, None],
            estimation_lse,
            torch.zeros_like(estimation_lse),
        )
        weights = torch.exp(logits - safe_normalizer.unsqueeze(-1))
        weights.masked_fill_(~estimation_mask[:, None, None, :], 0.0)

        estimation_output = torch.einsum(
            "bhgm,bmhd->bhgd", weights, selection.estimation_values
        )
        estimation_output = estimation_output.reshape(
            batch_size, num_query_heads, head_size
        )
        estimation_output = estimation_output.to(query.dtype)

        estimation_lse = estimation_lse.reshape(batch_size, num_query_heads)
        estimation_lse = estimation_lse.transpose(0, 1).contiguous()

        return estimation_output, estimation_lse

    def _draft_forward(
        self,
        impl: FlashAttentionImpl,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert self.active_mask is not None

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens != self.batch_size:
            raise RuntimeError(
                "RetroSpec sparse draft attention requires exactly one "
                "query token per request."
            )
        if attn_metadata.max_query_len != 1:
            raise RuntimeError(
                "RetroSpec sparse draft attention requires max_query_len=1."
            )

        query = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(0)

        selection = self.index.select(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            active_mask=self.active_mask,
            scale=impl.scale,
        )

        exact_output, exact_lse = self._run_exact_attention(
            impl, layer, query, key_cache, value_cache, attn_metadata, selection
        )
        estimation_output, estimation_lse = self._run_estimation_attention(
            impl, query, selection
        )

        merge_attn_states(
            output[:num_actual_tokens],
            exact_output,
            exact_lse,
            estimation_output,
            estimation_lse,
        )

        self.hit_attn_sum[: self.batch_size].add_(selection.hit_attn)
        self.hit_attn_layer_count += 1

        return output
