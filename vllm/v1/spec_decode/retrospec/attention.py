# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum

import torch

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention import Attention
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from .capacity import get_retrospec_exact_attention_partition_capacity
from .cluster_store import (
    RetroSpecCompactResolvedClusterPages,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecResidentPrefetchInput,
    RetroSpecResolvedClusterPages,
)
from .execution import (
    RetroSpecCompactExactPageTable,
    RetroSpecCompactKVSource,
    RetroSpecEstimationKVSource,
    RetroSpecExactAttentionWorkspace,
    RetroSpecExactKVSource,
    RetroSpecExactPageKVSource,
    RetroSpecExactPrimaryKVSource,
    RetroSpecFullVerificationKVSource,
)
from .index import RetroSpecAttentionLevel
from .performance import RetroSpecPerformanceStats
from .pipeline import RetroSpecAttentionMassStats
from .segmented_index import (
    RetroSpecIndexedTokenAttentionSelection,
    RetroSpecSegmentedTokenIndex,
    RetroSpecTokenAttentionSelection,
)
from .selection_kernels import add_indexed_values
from .workspace import exact_attention_query_capacity

RetroSpecSelection = (
    RetroSpecTokenAttentionSelection | RetroSpecIndexedTokenAttentionSelection
)


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
        self.max_verification_tokens = exact_attention_query_capacity(
            self.max_batch_size, self.num_speculative_tokens
        )

        self.performance_stats = RetroSpecPerformanceStats(
            device=device,
            log_interval_seconds=getattr(
                config,
                "retrospec_stats_interval_seconds",
                0.0,
            ),
        )
        cpu_page_slab_size_mib = getattr(
            config, "retrospec_cpu_page_slab_size_mib", 256
        )
        cpu_page_initial_slab_size_mib = getattr(
            config,
            "retrospec_cpu_page_initial_slab_size_mib",
            min(8, cpu_page_slab_size_mib),
        )

        self.index = RetroSpecSegmentedTokenIndex(
            block_size=block_size,
            num_speculative_tokens=config.num_speculative_tokens,
            retrieval_ratio=config.retrospec_retrieval_ratio,
            estimation_ratio=config.retrospec_estimation_ratio,
            prefill_segment_size_tokens=config.retrospec_index_segment_size,
            generation_update_interval=config.retrospec_index_update_interval,
            blocks_per_cluster=config.retrospec_blocks_per_cluster,
            num_kmeans_iterations=config.retrospec_kmeans_iterations,
            max_model_len=vllm_config.model_config.max_model_len,
            max_pending_cluster_builds=config.retrospec_max_pending_cluster_builds,
            cpu_page_build_workers=config.retrospec_cpu_page_build_workers,
            cache_ratio=config.retrospec_cache_ratio,
            pin_memory=device.type == "cuda" and is_pin_memory_available(),
            max_resident_requests=vllm_config.scheduler_config.max_num_seqs,
            first_draft_warmup_multiplier=getattr(
                config,
                "retrospec_first_draft_warmup_multiplier",
                4,
            ),
            cpu_page_initial_slab_bytes=(cpu_page_initial_slab_size_mib * MiB_bytes),
            cpu_page_slab_bytes=cpu_page_slab_size_mib * MiB_bytes,
            max_pinned_memory_bytes=int(
                getattr(config, "retrospec_max_pinned_memory", 1.0) * GiB_bytes
            ),
            max_gpu_index_memory_bytes=int(
                getattr(config, "retrospec_max_gpu_index_memory", 4.0) * GiB_bytes
            ),
            performance_stats=(
                self.performance_stats if self.performance_stats.enabled else None
            ),
        )

        exact_partition_capacity = get_retrospec_exact_attention_partition_capacity(
            vllm_config, block_size
        )
        self.exact_attention_workspace = RetroSpecExactAttentionWorkspace(
            page_size=block_size,
            max_num_queries=self.max_verification_tokens,
            partition_capacity=exact_partition_capacity,
        )

        self.proposal_request_ids: tuple[str, ...] = ()
        self.proposal_context_lens: tuple[int, ...] = ()

        self.full_verification_batch: _RetroSpecFullVerificationBatch | None = None

        # request_id -> exclusive logical block boundary already retired.
        # Block 0 remains the permanent exact sink.
        self._retired_block_ends: dict[str, int] = {}

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
        self._draft_layout_generation = 0
        self.active_mask: torch.Tensor | None = None
        self.batch_size = 0
        self.parallel_request_indices: torch.Tensor | None = None
        self.parallel_token_indices: torch.Tensor | None = None

        self.attention_mass_layer_count = 0
        self.attention_mass_sum = torch.zeros(
            self.max_parallel_tokens, dtype=torch.float32, device=device
        )

        self.original_forwards: dict[str, tuple[FlashAttentionImpl, LayerForward]] = {}
        self.forward_wrappers: dict[str, _RetroSpecLayerForward] = {}

        self._resident_prefetch_wave: list[RetroSpecResidentPrefetchInput] = []

    @property
    def uses_full_verification_offload(self) -> bool:
        return True

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
        first_draft_warmup_request_ids = tuple(
            request_ids[row] for row in build_rows if prefill_complete[row]
        )

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
            self.index.mark_first_draft_warmup(
                first_draft_warmup_request_ids,
                tuple(self.original_forwards),
            )
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
        """Stage one complete prompt layer without using forward hooks."""
        if self.in_proposal:
            raise RuntimeError("Cannot build a prefill index during a proposal")
        if self.index_update_active:
            raise RuntimeError(
                "Layer-major prefill cannot overlap a normal index update"
            )
        if self.index.has_staged_request_layer(layer_name, request_id):
            raise RuntimeError(
                f"Layer {layer_name!r} is already staged for {request_id!r}"
            )

        return self.index.build_or_update(
            layer_name=layer_name,
            request_ids=(request_id,),
            seq_lens=(seq_len,),
            is_prefill=(True,),
            rows=(0,),
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            defer_cpu_store=True,
            prefill_complete=(True,),
        )

    def commit_layer_major_prefill(
        self,
        request_id: str,
        layer_names: Sequence[str],
    ) -> None:
        """Publish every layer after the complete request succeeds."""
        self.index.flush_staged_updates()
        self.index.mark_first_draft_warmup((request_id,), tuple(layer_names))

    def abort_layer_major_prefill(self) -> None:
        self.index.discard_staged_updates()

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
        self.index.close()

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

        self.index.prepare_full_verification(
            request_ids,
            context_lens,
            tuple(self.original_forwards),
        )
        self.index.begin_full_verification_residency(request_ids)
        pipeline_started = False
        try:
            if self.device.type == "cuda":
                layer_num_kv_heads = {
                    layer_name: impl.num_kv_heads
                    for layer_name, (impl, _) in self.original_forwards.items()
                }
                self.index.begin_full_verification_pipeline(
                    request_ids,
                    layer_num_kv_heads,
                    self.device,
                )
                pipeline_started = True
            self.performance_stats.add_counter(
                "full_verify_requests",
                len(request_ids),
            )
            self.full_verification_batch = _RetroSpecFullVerificationBatch(
                request_ids=request_ids,
                context_lens=context_lens,
                query_lens=query_lens,
            )
            self.mode = RetroSpecAttentionMode.FULL_VERIFY
            yield
        finally:
            self.mode = RetroSpecAttentionMode.PASSTHROUGH
            self.full_verification_batch = None
            if pipeline_started:
                self.index.end_full_verification_pipeline()
            self.index.end_full_verification_residency()
            self.performance_stats.maybe_log()

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
            self._resident_prefetch_wave.clear()
            self._draft_layout_generation = 0

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
            self._resident_prefetch_wave.clear()

            try:
                self.index.flush_sparse_verification_prefetch()
            finally:
                self.index.end_proposal()
                self.proposal_request_ids = ()
                self.proposal_context_lens = ()

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
        if mode == RetroSpecAttentionMode.DRAFT and self._resident_prefetch_wave:
            raise RuntimeError("The previous draft prefetch wave was not submitted.")
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
        if mode == RetroSpecAttentionMode.DRAFT:
            self._draft_layout_generation += 1
        self.batch_size = active_mask.shape[0]
        self.active_mask = active_mask
        self.parallel_request_indices = None
        self.parallel_token_indices = None

        self.attention_mass_sum[: self.batch_size].zero_()
        self.attention_mass_layer_count = 0
        self.step_active = True

    def begin_parallel_step(
        self,
        mode: RetroSpecAttentionMode,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
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
        self.index.prepare_indexed_plan_workspace(num_tokens)

        self.mode = mode
        self.step_index = -1
        self.batch_size = num_tokens
        self.active_mask = torch.ones(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.parallel_request_indices = request_indices
        self.parallel_token_indices = token_indices

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

        if self.mode == RetroSpecAttentionMode.DRAFT:
            assert self.active_mask is not None
            self.index.complete_first_draft_warmup(
                self.proposal_request_ids,
                tuple(self.original_forwards),
                self.active_mask,
            )
            prefetch_wave = tuple(self._resident_prefetch_wave)
            self._resident_prefetch_wave.clear()
            if prefetch_wave:
                with self.performance_stats.cpu_timer("draft_prefetch_wave_submit"):
                    self.index.submit_sparse_verification_prefetch_wave(prefetch_wave)

        self.mode = RetroSpecAttentionMode.PASSTHROUGH
        self.step_active = False
        self.step_index = -1
        self.active_mask = None
        self.batch_size = 0
        self.parallel_request_indices = None
        self.parallel_token_indices = None
        self.attention_mass_layer_count = 0

        return RetroSpecAttentionMassStats(
            value_sum=attention_mass_sum,
            layer_count=layer_count,
        )

    def end_step(self) -> torch.Tensor:
        return self.end_step_statistics().mean()

    def flush_sparse_verification_prefetch(self) -> None:
        self.index.flush_sparse_verification_prefetch()

    def maybe_prime_full_verification(self, num_candidate_tokens: int) -> bool:
        """Start speculative full-verification H2D while sparse verify runs."""
        if not self.in_proposal:
            raise RuntimeError("Full-verification priming requires a proposal")
        if num_candidate_tokens <= 0:
            return False
        if self.device.type != "cuda" or not self.index.cluster_store.pin_memory:
            return False
        if not self.proposal_context_lens:
            return False
        if max(self.proposal_context_lens) < self.index.prefill_segment_size_tokens:
            self.performance_stats.add_counter("full_verify_prime_context_skipped")
            return False

        layer_num_kv_heads = {
            layer_name: impl.num_kv_heads
            for layer_name, (impl, _) in self.original_forwards.items()
        }
        return self.index.prime_full_verification_pipeline(
            self.proposal_request_ids,
            layer_num_kv_heads,
            self.device,
        )

    def _get_indexed_selection(
        self, layer_name: str, level: RetroSpecAttentionLevel
    ) -> RetroSpecIndexedTokenAttentionSelection:
        request_indices = self.parallel_request_indices
        token_indices = self.parallel_token_indices
        if request_indices is None or token_indices is None:
            raise RuntimeError("No parallel verification plan is active")
        return self.index.get_indexed_selection(
            layer_name, level, request_indices, token_indices
        )

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
            defer_cpu_store=True,
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
            self._maybe_update_index(layer_name, kv_cache, attn_metadata)
            return result

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

        full_timer = self.performance_stats.start_cuda_timer("full_verify_layer")
        self.performance_stats.add_counter("full_verify_layers")

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
        clustered = plan.clustered_kv
        if clustered is None and any(
            descriptor.num_tokens for descriptor in plan.clustered_descriptors
        ):
            clustered = self.index.cluster_store.resolve_full_verification_tokens(
                layer_name=layer_name,
                descriptors=plan.clustered_descriptors,
            )
        clustered_source = None
        if clustered is not None:
            clustered_source = RetroSpecCompactKVSource(
                key_tokens=clustered.key_tokens,
                value_tokens=clustered.value_tokens,
                token_offsets=clustered.token_offsets,
                token_counts=clustered.token_counts,
                max_tokens_per_head=clustered.max_tokens_per_head,
                ready_event=clustered.ready_event,
            )
        source = RetroSpecFullVerificationKVSource(
            primary=RetroSpecExactPrimaryKVSource(
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table,
                token_indices=plan.primary_exact_token_indices,
                token_mask=plan.primary_exact_token_mask,
            ),
            clustered=clustered_source,
        )
        cluster_output, cluster_lse, native_output, native_lse = (
            self.exact_attention_workspace.run_parallel_full_verification(
                source,
                query,
                key,
                value,
                impl.scale,
                attn_metadata.query_start_loc,
                attn_metadata.max_query_len,
            )
        )

        merge_attn_states(
            output[:num_actual_tokens],
            cluster_output,
            cluster_lse,
            native_output,
            native_lse,
        )
        self.performance_stats.stop_cuda_timer(full_timer)
        return output

    def _resolve_exact_kv_source(
        self,
        selection: RetroSpecSelection,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[
        RetroSpecExactKVSource,
        RetroSpecResolvedClusterPages
        | RetroSpecCompactResolvedClusterPages
        | RetroSpecCompactVerificationResolvedPages
        | None,
    ]:
        indexed = isinstance(selection, RetroSpecIndexedTokenAttentionSelection)
        if indexed:
            layer_name = selection.layer_name
            primary_token_indices = selection.primary_exact_token_indices
            primary_token_mask = selection.primary_exact_token_mask
            plan_row_indices = selection.plan_row_indices
            if self.mode not in (
                RetroSpecAttentionMode.SPARSE_VERIFY,
                RetroSpecAttentionMode.EXPANDED_VERIFY,
            ):
                raise RuntimeError("Indexed plans are only valid during verification")
            resolved_pages = self.index.resolve_indexed_verification_pages(selection)
            if resolved_pages is None:
                exact_page_token_counts = torch.empty(
                    plan_row_indices.shape[0],
                    primary_token_indices.shape[1],
                    0,
                    dtype=torch.int32,
                    device=primary_token_indices.device,
                )
            else:
                exact_page_token_counts = resolved_pages.page_token_counts
        else:
            layer_name = selection.plan.layer_name
            primary_token_indices = selection.plan.primary_exact_token_indices
            primary_token_mask = selection.plan.primary_exact_token_mask
            plan_row_indices = None
            resolved_pages = selection.resolved_pages
            exact_cluster_ids = selection.exact_cluster_ids
            exact_page_ids = selection.exact_page_ids
            exact_page_token_counts = selection.exact_page_token_counts
            if resolved_pages is None and exact_page_ids.numel():
                resolved_pages = self.index.cluster_store.resolve_cluster_blocks(
                    layer_name=layer_name,
                    cluster_ids=exact_cluster_ids,
                    logical_page_ids=exact_page_ids,
                    mode="verification",
                )

        resident_pages = None
        staging_pages = None
        compact_pages = None
        if resolved_pages is not None:
            if isinstance(
                resolved_pages,
                (
                    RetroSpecCompactResolvedClusterPages,
                    RetroSpecCompactVerificationResolvedPages,
                ),
            ):
                if resolved_pages.resident_key_pages.shape[0] > 0:
                    resident_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.resident_key_pages,
                        value_pages=resolved_pages.resident_value_pages,
                        page_ids=resolved_pages.resident_page_ids,
                    )
                if (
                    isinstance(
                        resolved_pages, RetroSpecCompactVerificationResolvedPages
                    )
                    and resolved_pages.staging_key_pages.shape[0] > 0
                ):
                    staging_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.staging_key_pages,
                        value_pages=resolved_pages.staging_value_pages,
                        page_ids=resolved_pages.staging_page_ids,
                        ready_event=resolved_pages.staging_ready_event,
                    )
                compact_pages = RetroSpecCompactExactPageTable(
                    page_counts=resolved_pages.page_counts
                )
            else:
                if resolved_pages.resident_key_pages.shape[0] > 0:
                    resident_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.resident_key_pages,
                        value_pages=resolved_pages.resident_value_pages,
                        page_ids=resolved_pages.resident_page_ids,
                        ready_event=resolved_pages.resident_ready_event,
                    )
                if resolved_pages.staging_key_pages.shape[0] > 0:
                    staging_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.staging_key_pages,
                        value_pages=resolved_pages.staging_value_pages,
                        page_ids=resolved_pages.staging_page_ids,
                        ready_event=resolved_pages.staging_ready_event,
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
            resident_pages=resident_pages,
            staging_pages=staging_pages,
            plan_row_indices=plan_row_indices,
            compact_pages=compact_pages,
        )

        return source, resolved_pages

    def _run_exact_attention(
        self,
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                and selection.exact_page_ids.numel()
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

        stage_name = {
            RetroSpecAttentionMode.DRAFT: "draft",
            RetroSpecAttentionMode.SPARSE_VERIFY: "sparse_verify",
            RetroSpecAttentionMode.EXPANDED_VERIFY: "expanded_verify",
        }[self.mode]
        with (
            self.performance_stats.cpu_timer(f"{stage_name}_page_resolve_wall"),
            self.performance_stats.cuda_timer(f"{stage_name}_page_resolve"),
        ):
            source, resolved_pages = self._resolve_exact_kv_source(
                selection=selection,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table,
            )
        try:
            with self.performance_stats.cuda_timer(f"{stage_name}_exact_attention"):
                exact_output = self.exact_attention_workspace.run(
                    source, query, impl.scale
                )
        finally:
            if resolved_pages is not None and resolved_pages.read_lease is not None:
                resolved_pages.read_lease.release()

        miss_admission = getattr(resolved_pages, "miss_admission", None)
        if miss_admission is not None:
            with self.performance_stats.cpu_timer(
                f"{stage_name}_resident_admit_submit"
            ):
                self.index.cluster_store.admit_verification_misses(miss_admission)

        return exact_output

    def _run_fused_proposal_attention(
        self,
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        selection: RetroSpecSelection,
        output: torch.Tensor,
    ) -> torch.Tensor:
        stage_name = {
            RetroSpecAttentionMode.DRAFT: "draft",
            RetroSpecAttentionMode.SPARSE_VERIFY: "sparse_verify",
            RetroSpecAttentionMode.EXPANDED_VERIFY: "expanded_verify",
        }[self.mode]
        with (
            self.performance_stats.cpu_timer(f"{stage_name}_page_resolve_wall"),
            self.performance_stats.cuda_timer(f"{stage_name}_page_resolve"),
        ):
            source, resolved_pages = self._resolve_exact_kv_source(
                selection=selection,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table,
            )

        estimation_keys, estimation_values, estimation_token_counts = (
            self._get_grouped_estimation(selection)
        )
        estimation = RetroSpecEstimationKVSource(
            keys=estimation_keys,
            values=estimation_values,
            token_counts=estimation_token_counts,
            plan_row_indices=None,
        )

        try:
            with self.performance_stats.cuda_timer(f"{stage_name}_fused_attention"):
                self.exact_attention_workspace.run_proposal(
                    source=source,
                    estimation=estimation,
                    query=query,
                    scale=impl.scale,
                    output=output,
                )
        finally:
            if resolved_pages is not None and resolved_pages.read_lease is not None:
                resolved_pages.read_lease.release()

        miss_admission = getattr(resolved_pages, "miss_admission", None)
        if miss_admission is not None:
            with self.performance_stats.cpu_timer(
                f"{stage_name}_resident_admit_submit"
            ):
                self.index.cluster_store.admit_verification_misses(miss_admission)

        return output

    @staticmethod
    def _get_grouped_estimation(
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return estimation tensors in grouped KV-head layout."""
        return (
            selection.estimation_keys,
            selection.estimation_values,
            selection.estimation_token_counts,
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
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                None,
                None,
            )
            self.attention_mass_sum[: self.batch_size].add_(1.0)
            self.attention_mass_layer_count += 1
            return result

        query = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(0)

        if self.mode == RetroSpecAttentionMode.DRAFT:
            with self.performance_stats.cuda_timer("draft_selection_total"):
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
                    warm_first_draft=True,
                    plan_slot=self.step_index,
                    layout_generation=self._draft_layout_generation,
                )
        else:
            if self.mode == RetroSpecAttentionMode.SPARSE_VERIFY:
                level = RetroSpecAttentionLevel.SPARSE
            elif self.mode == RetroSpecAttentionMode.EXPANDED_VERIFY:
                level = RetroSpecAttentionLevel.EXPANDED
            else:
                raise RuntimeError(f"Unexpected RetroSpec attention mode: {self.mode}")

            if has_parallel_plan:
                with self.performance_stats.cuda_timer("verification_plan_index"):
                    selection = self._get_indexed_selection(layer_name, level)
                if query.device.type != "cuda" or query.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    selection = self.index.materialize_indexed_reference(selection)
            else:
                plan = self.index.get_selection_plan(layer_name, self.step_index)
                with self.performance_stats.cuda_timer("verification_plan_materialize"):
                    selection = self.index.materialize(
                        plan,
                        level,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table,
                    )

        stage_name = {
            RetroSpecAttentionMode.DRAFT: "draft",
            RetroSpecAttentionMode.SPARSE_VERIFY: "sparse_verify",
            RetroSpecAttentionMode.EXPANDED_VERIFY: "expanded_verify",
        }[self.mode]
        can_use_fused_proposal = query.device.type == "cuda" and query.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        if can_use_fused_proposal:
            self._run_fused_proposal_attention(
                impl=impl,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                selection=selection,
                output=output[:num_actual_tokens],
            )
        else:
            exact_output, exact_lse = self._run_exact_attention(
                impl,
                query,
                key_cache,
                value_cache,
                attn_metadata,
                selection,
            )
            with self.performance_stats.cuda_timer(f"{stage_name}_estimation_merge"):
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

        if isinstance(selection, RetroSpecIndexedTokenAttentionSelection):
            add_indexed_values(
                self.attention_mass_sum[: self.batch_size],
                selection.attention_mass,
                selection.plan_row_indices,
            )
        else:
            self.attention_mass_sum[: self.batch_size].add_(selection.attention_mass)
        self.attention_mass_layer_count += 1

        if self.mode == RetroSpecAttentionMode.DRAFT:
            record = self.index.build_sparse_verification_prefetch(
                selection=selection, active_mask=self.active_mask
            )
            if record is not None:
                self._resident_prefetch_wave.append(record)

        return output
