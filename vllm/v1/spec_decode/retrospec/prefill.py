# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.core.sched.output import (
    RetroSpecLayerMajorPrefillDescriptor,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.outputs import RetroSpecLayerMajorPrefillCompletion


@runtime_checkable
class RetroSpecLayerModel(Protocol):
    """Decoder model interface required by layer-major prefill."""

    start_layer: int
    end_layer: int

    def forward_layer(
        self,
        layer_index: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **extra_layer_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def resolve_retrospec_layer_model(model: torch.nn.Module) -> RetroSpecLayerModel:
    """Resolve the decoder model implementing the layer execution interface."""
    if isinstance(model, RetroSpecLayerModel):
        return model

    layer_model = getattr(model, "model", None)
    if isinstance(layer_model, RetroSpecLayerModel):
        return layer_model

    raise TypeError(
        "RetroSpec layer-major prefill requires a model exposing "
        "start_layer, end_layer, and forward_layer()"
    )


@dataclass(frozen=True)
class RetroSpecLayerPrefillTile:
    """One token tile backed by the reusable layer-prefill KV workspace."""

    kv_cache: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduled_start: int
    scheduled_end: int
    context_len: int

    @property
    def num_scheduled_tokens(self) -> int:
        return self.scheduled_end - self.scheduled_start


class RetroSpecLayerPrefillWorkspace:
    """Reusable paged-KV storage for one layer-major prefill request.

    ``kv_cache_spec`` must describe the kernel-level cache layout. In
    particular, its block size must match the block size consumed by the
    attention backend, rather than a larger scheduler-level virtual block.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        backend: type[AttentionBackend],
        kv_cache_spec: AttentionSpec,
        cache_dtype: str,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                "RetroSpec layer-prefill workspace only supports CUDA devices"
            )
        if kv_cache_spec.block_size <= 0:
            raise ValueError("Layer-prefill KV block size must be positive")
        if (
            kv_cache_spec.page_size_padded is not None
            and kv_cache_spec.page_size_padded != kv_cache_spec.real_page_size_bytes
        ):
            raise NotImplementedError(
                "Layer-prefill workspace does not support padded KV pages"
            )

        self.device = device
        self.backend = backend
        self.kv_cache_spec = kv_cache_spec
        self.cache_dtype = cache_dtype

        self._storage: torch.Tensor | None = None
        self._kv_cache: torch.Tensor | None = None
        self._block_table: torch.Tensor | None = None
        self._slot_mapping: torch.Tensor | None = None

        self._num_tokens = 0
        self._num_blocks = 0
        self._active_layer_name: str | None = None
        self._reuse_ready_event: torch.cuda.Event | None = None

    @property
    def block_size(self) -> int:
        return self.kv_cache_spec.block_size

    @property
    def capacity_tokens(self) -> int:
        return self._num_blocks * self.block_size

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def active_layer_name(self) -> str | None:
        return self._active_layer_name

    @property
    def kv_cache(self) -> torch.Tensor:
        if self._kv_cache is None:
            raise RuntimeError("Layer-prefill workspace has not been prepared")
        return self._kv_cache

    def _allocate_kv_cache(self, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        logical_shape = self.backend.get_kv_cache_shape(
            num_blocks,
            self.block_size,
            self.kv_cache_spec.num_kv_heads,
            self.kv_cache_spec.head_size,
            cache_dtype_str=self.cache_dtype,
        )

        try:
            stride_order = self.backend.get_kv_cache_stride_order()
            if tuple(sorted(stride_order)) != tuple(range(len(logical_shape))):
                raise ValueError(
                    "Attention backend returned an invalid KV stride order"
                )
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(logical_shape)))

        physical_shape = tuple(logical_shape[index] for index in stride_order)
        storage = torch.empty(
            physical_shape,
            dtype=self.kv_cache_spec.dtype,
            device=self.device,
        )
        inverse_order = [
            stride_order.index(index) for index in range(len(stride_order))
        ]
        return storage, storage.permute(*inverse_order)

    def _synchronize_reuse(self) -> None:
        if self._reuse_ready_event is None:
            return

        self._reuse_ready_event.synchronize()
        self._reuse_ready_event = None

    def _wait_for_reuse(self) -> None:
        if self._reuse_ready_event is None:
            return

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._reuse_ready_event)
        self._reuse_ready_event = None

    def prepare(self, num_tokens: int) -> None:
        """Reserve enough paged KV storage for one complete prompt."""
        if num_tokens <= 0:
            raise ValueError("Layer-prefill token count must be positive")
        if self._active_layer_name is not None:
            raise RuntimeError("Cannot resize an active layer-prefill workspace")

        required_blocks = cdiv(num_tokens, self.block_size)
        if required_blocks <= self._num_blocks:
            self._num_tokens = num_tokens
            return

        # Growth is rare and must not temporarily retain both the old and new
        # allocations while an asynchronous source read is still in flight.
        self._synchronize_reuse()

        self._storage = None
        self._kv_cache = None
        self._block_table = None
        self._slot_mapping = None

        storage, kv_cache = self._allocate_kv_cache(required_blocks)
        self._storage = storage
        self._kv_cache = kv_cache
        self._block_table = torch.arange(
            required_blocks,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(0)
        self._slot_mapping = torch.arange(
            required_blocks * self.block_size,
            dtype=torch.int64,
            device=self.device,
        )
        self._num_tokens = num_tokens
        self._num_blocks = required_blocks

    def begin_layer(self, layer_name: str) -> None:
        """Acquire the shared workspace for one decoder layer."""
        if not layer_name:
            raise ValueError("Layer name must not be empty")
        if self._kv_cache is None:
            raise RuntimeError("Layer-prefill workspace must be prepared before use")
        if self._active_layer_name is not None:
            raise RuntimeError(
                f"Layer-prefill workspace is already owned by {self._active_layer_name}"
            )

        self._wait_for_reuse()
        self._active_layer_name = layer_name

    def tile(
        self,
        scheduled_start: int,
        scheduled_end: int,
    ) -> RetroSpecLayerPrefillTile:
        """Return the paged-cache descriptors for one prompt tile."""
        if self._active_layer_name is None:
            raise RuntimeError("Layer-prefill workspace has no active layer")
        if scheduled_start < 0:
            raise ValueError("Layer-prefill tile start must be non-negative")
        if scheduled_end <= scheduled_start:
            raise ValueError("Layer-prefill tile end must be greater than its start")
        if scheduled_end > self._num_tokens:
            raise ValueError("Layer-prefill tile exceeds the prepared prompt length")

        assert self._block_table is not None
        assert self._slot_mapping is not None

        num_context_blocks = cdiv(scheduled_end, self.block_size)
        return RetroSpecLayerPrefillTile(
            kv_cache=self.kv_cache,
            block_table=self._block_table[:, :num_context_blocks],
            slot_mapping=self._slot_mapping[scheduled_start:scheduled_end],
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            context_len=scheduled_end,
        )

    @contextmanager
    def bind_layer(
        self,
        layer_name: str,
        attention_layer: Any,
    ) -> Iterator[None]:
        """Temporarily bind the current Attention module to this workspace."""
        if self._active_layer_name != layer_name:
            raise RuntimeError(
                "Layer-prefill workspace is owned by "
                f"{self._active_layer_name}, not {layer_name}"
            )

        original_kv_cache = attention_layer.kv_cache
        if len(original_kv_cache) != 1:
            raise NotImplementedError(
                "Layer-prefill workspace currently requires one virtual engine"
            )

        attention_layer.kv_cache = [self.kv_cache]
        try:
            yield
        finally:
            attention_layer.kv_cache = original_kv_cache

    def end_layer(
        self,
        reuse_ready_event: torch.cuda.Event | None = None,
    ) -> None:
        """Release the active layer after all workspace source reads are queued."""
        if self._active_layer_name is None:
            raise RuntimeError("Layer-prefill workspace has no active layer")

        if reuse_ready_event is None:
            reuse_ready_event = torch.cuda.Event()
            reuse_ready_event.record(torch.cuda.current_stream(self.device))

        self._reuse_ready_event = reuse_ready_event
        self._active_layer_name = None

    def abort_layer(self) -> None:
        """Recover workspace ownership after an exceptional layer execution."""
        if self._active_layer_name is None:
            return

        torch.cuda.synchronize(self.device)
        self._active_layer_name = None
        self._reuse_ready_event = None

    def close(self) -> None:
        if self._active_layer_name is not None:
            raise RuntimeError("Cannot close an active layer-prefill workspace")

        self._synchronize_reuse()
        self._storage = None
        self._kv_cache = None
        self._block_table = None
        self._slot_mapping = None
        self._num_tokens = 0
        self._num_blocks = 0


class RetroSpecLayerMajorPrefillProtocol:
    """Validate the scheduler/worker layer-major prefill contract."""

    @staticmethod
    def validate_scheduler_output(scheduler_output: SchedulerOutput) -> None:
        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is None:
            return

        request_id = descriptor.request_id
        scheduled_request_ids = set(scheduler_output.num_scheduled_tokens)
        if scheduled_request_ids != {request_id}:
            raise RuntimeError(
                "RetroSpec layer-major prefill must be scheduled as an exclusive "
                f"batch, got {scheduled_request_ids}"
            )

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]
        if num_scheduled_tokens != descriptor.num_scheduled_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill descriptor does not match "
                "num_scheduled_tokens: "
                f"descriptor={descriptor.num_scheduled_tokens}, "
                f"scheduled={num_scheduled_tokens}"
            )

        if scheduler_output.total_num_scheduled_tokens != num_scheduled_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill total token count is inconsistent"
            )

        if scheduler_output.scheduled_spec_decode_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill cannot contain speculative tokens"
            )

        if scheduler_output.scheduled_encoder_inputs:
            raise RuntimeError(
                "RetroSpec layer-major prefill does not support encoder inputs"
            )

    @staticmethod
    def validate_cached_prompt(
        descriptor: RetroSpecLayerMajorPrefillDescriptor,
        cached_prompt_num_tokens: int,
    ) -> None:
        if cached_prompt_num_tokens != descriptor.prompt_num_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill prompt descriptor is stale: "
                f"descriptor={descriptor.prompt_num_tokens}, "
                f"worker={cached_prompt_num_tokens}"
            )

    @classmethod
    def complete_scheduled_range(
        cls, scheduler_output: SchedulerOutput
    ) -> RetroSpecLayerMajorPrefillCompletion | None:
        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is None:
            return None

        cls.validate_scheduler_output(scheduler_output)
        return RetroSpecLayerMajorPrefillCompletion(
            request_id=descriptor.request_id,
            num_completed_prompt_tokens=descriptor.scheduled_end,
        )
