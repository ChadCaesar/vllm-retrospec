# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_pp_group,
    get_tp_group,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from .prefill import RetroSpecLayerModel


@dataclass(frozen=True)
class RetroSpecPipelineStage:
    """Description of the model layers owned by one PP rank."""

    rank: int
    world_size: int
    start_layer: int
    end_layer: int

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.world_size - 1

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer

    @classmethod
    def from_model(cls, layer_model: RetroSpecLayerModel) -> "RetroSpecPipelineStage":
        pp_group = get_pp_group()
        start_layer = int(layer_model.start_layer)
        end_layer = int(layer_model.end_layer)
        if start_layer < 0 or end_layer <= start_layer:
            raise ValueError(
                "RetroSpec pipeline stage must own a non-empty layer range, "
                f"got [{start_layer}, {end_layer})"
            )

        return cls(
            rank=pp_group.rank_in_group,
            world_size=pp_group.world_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )


@dataclass(frozen=True)
class RetroSpecAttentionMassStats:
    """TP-normalized attention-mass sum over locally owned layers.

    The value may be a view of the attention controller's fixed workspace and
    must be consumed before the next RetroSpec attention step begins.
    """

    value_sum: torch.Tensor
    layer_count: int

    def __post_init__(self) -> None:
        if self.value_sum.ndim != 1:
            raise ValueError("value_sum must be a one-dimensional tensor")
        if self.value_sum.dtype != torch.float32:
            raise ValueError("value_sum must have dtype torch.float32")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be greater than zero")

    def mean(self) -> torch.Tensor:
        return self.value_sum / self.layer_count


@dataclass(frozen=True)
class RetroSpecPipelineControlState:
    """Request-level state broadcast by the final PP rank."""

    token_ids: torch.Tensor
    stages: torch.Tensor
    draft_counts: torch.Tensor
    pending_counts: torch.Tensor
    active_mask: torch.Tensor


@dataclass(frozen=True)
class RetroSpecPipelineModelOutput:
    """Model-step results broadcast by the final PP rank."""

    token_ids: torch.Tensor
    margin: torch.Tensor | None


class RetroSpecPipelineProtocol:
    """Fixed-capacity GPU communication protocol for RetroSpec PP."""

    _HIDDEN_STATES_KEY = "retrospec_hidden_states"

    _TOKEN_IDS_COLUMN = 0
    _STAGES_COLUMN = 1
    _DRAFT_COUNTS_COLUMN = 2
    _PENDING_COUNTS_COLUMN = 3
    _NUM_INTEGER_COLUMNS = 4

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        max_batch_size: int,
        max_parallel_tokens: int,
        max_sampled_tokens: int,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if max_parallel_tokens < max_batch_size:
            raise ValueError("max_parallel_tokens must be at least max_batch_size")
        if max_sampled_tokens <= 0:
            raise ValueError("max_sampled_tokens must be greater than zero")

        self.vllm_config = vllm_config
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_parallel_tokens = max_parallel_tokens
        self.max_sampled_tokens = max_sampled_tokens

        self._integer_control = torch.empty(
            (max_batch_size, self._NUM_INTEGER_COLUMNS),
            dtype=torch.int32,
            device=device,
        )
        self._active_mask = torch.empty(max_batch_size, dtype=torch.bool, device=device)
        self._target_sampled_token_ids = torch.empty(
            (max_batch_size, max_sampled_tokens),
            dtype=torch.int32,
            device=device,
        )
        self._model_token_ids = torch.empty(
            max_parallel_tokens, dtype=torch.int32, device=device
        )
        self._model_margin = torch.empty(
            max_parallel_tokens, dtype=torch.float32, device=device
        )

        # The final element carries the local layer count, allowing the
        # numerator and denominator to use one PP all-reduce.
        self._attention_reduce = torch.empty(
            max_parallel_tokens + 1, dtype=torch.float32, device=device
        )

    @property
    def pp_group(self) -> GroupCoordinator:
        return get_pp_group()

    def describe_stage(
        self,
        layer_model: RetroSpecLayerModel,
        layer_names: Sequence[str],
    ) -> RetroSpecPipelineStage:
        stage = RetroSpecPipelineStage.from_model(layer_model)
        if len(layer_names) != stage.num_layers:
            raise RuntimeError(
                "RetroSpec local attention-layer count does not match the owned "
                f"model range: names={len(layer_names)}, "
                f"range=[{stage.start_layer}, {stage.end_layer})"
            )
        return stage

    def prepare_layer_prefill_input(
        self,
        stage: RetroSpecPipelineStage,
        layer_model: RetroSpecLayerModel,
        prompt_token_ids: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        prompt_num_tokens: int,
    ) -> torch.Tensor:
        if prompt_num_tokens <= 0:
            raise ValueError("prompt_num_tokens must be greater than zero")

        if stage.is_first:
            if prompt_token_ids is None:
                raise RuntimeError(
                    "The first RetroSpec PP rank requires prompt token IDs"
                )
            if intermediate_tensors is not None:
                raise RuntimeError(
                    "The first RetroSpec PP rank must not receive intermediate tensors"
                )
            if prompt_token_ids.shape != (prompt_num_tokens,):
                raise ValueError(
                    "Prompt token IDs do not match the layer-prefill descriptor"
                )
            return layer_model.embed_input_ids(prompt_token_ids)

        if prompt_token_ids is not None:
            raise RuntimeError(
                "Only the first RetroSpec PP rank may consume prompt token IDs"
            )
        if intermediate_tensors is None:
            raise RuntimeError(
                "A non-first RetroSpec PP rank requires intermediate tensors"
            )

        hidden_states = intermediate_tensors.tensors.get(self._HIDDEN_STATES_KEY)
        if hidden_states is None:
            raise RuntimeError("RetroSpec PP payload does not contain hidden states")
        if hidden_states.ndim != 2 or hidden_states.shape[0] != prompt_num_tokens:
            raise ValueError(
                "RetroSpec PP hidden-state payload has an invalid shape: "
                f"{tuple(hidden_states.shape)}"
            )
        if hidden_states.device != self.device:
            raise ValueError("RetroSpec PP hidden-state payload is on the wrong device")
        return hidden_states

    def make_layer_prefill_output(
        self,
        stage: RetroSpecPipelineStage,
        hidden_states: torch.Tensor,
    ) -> IntermediateTensors:
        if stage.is_last:
            raise RuntimeError(
                "The last RetroSpec PP rank must produce logits, not a PP payload"
            )
        if hidden_states.ndim != 2:
            raise ValueError("RetroSpec PP hidden states must be two-dimensional")
        if hidden_states.device != self.device:
            raise ValueError("RetroSpec PP hidden states are on the wrong device")

        return IntermediateTensors(
            {self._HIDDEN_STATES_KEY: hidden_states.contiguous()}
        )

    def _proposal_all_gather_tensors(self, num_tokens: int) -> dict[str, bool]:
        return {
            "residual": not is_residual_scattered_for_sp(self.vllm_config, num_tokens)
        }

    def receive_model_input(
        self,
        stage: RetroSpecPipelineStage,
        num_tokens: int,
    ) -> IntermediateTensors | None:
        if stage.is_first:
            return None

        tensor_dict = self.pp_group.recv_tensor_dict(
            all_gather_group=get_tp_group(),
            all_gather_tensors=self._proposal_all_gather_tensors(num_tokens),
        )
        if tensor_dict is None:
            raise RuntimeError(
                "RetroSpec PP stage did not receive intermediate tensors"
            )

        for name, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                raise RuntimeError(
                    f"RetroSpec PP tensor {name!r} is on the wrong device"
                )
        return IntermediateTensors(tensor_dict)

    def send_model_output(
        self,
        stage: RetroSpecPipelineStage,
        output: IntermediateTensors,
        num_tokens: int,
    ) -> None:
        if stage.is_last:
            raise RuntimeError(
                "The final RetroSpec PP stage must not send intermediate output"
            )

        self.pp_group.send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=self._proposal_all_gather_tensors(num_tokens),
        )

    def broadcast_target_sampled_token_ids(
        self,
        batch_size: int,
        sampled_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not 0 <= batch_size <= self.max_batch_size:
            raise ValueError(f"batch_size must be in [0, {self.max_batch_size}]")

        output = self._target_sampled_token_ids[:batch_size]
        if self.pp_group.is_last_rank:
            if sampled_token_ids is None:
                raise RuntimeError(
                    "The final RetroSpec PP rank must provide sampled token IDs"
                )
            if sampled_token_ids.ndim != 2:
                raise ValueError("sampled_token_ids must be two-dimensional")
            if sampled_token_ids.shape[0] != batch_size:
                raise ValueError("sampled_token_ids must match the proposal batch size")
            if sampled_token_ids.shape[1] > self.max_sampled_tokens:
                raise ValueError("sampled_token_ids exceeds the pipeline workspace")
            if sampled_token_ids.device != self.device:
                raise ValueError("sampled_token_ids is on the wrong device")

            output.fill_(-1)
            output[:, : sampled_token_ids.shape[1]].copy_(sampled_token_ids)

        if batch_size > 0 and self.pp_group.world_size > 1:
            torch.distributed.broadcast(
                output,
                src=self.pp_group.last_rank,
                group=self.pp_group.device_group,
            )
        return output

    def broadcast_model_output(
        self,
        num_tokens: int,
        token_ids: torch.Tensor | None,
        margin: torch.Tensor | None,
        compute_margin: bool,
    ) -> RetroSpecPipelineModelOutput:
        if not 0 < num_tokens <= self.max_parallel_tokens:
            raise ValueError(f"num_tokens must be in [1, {self.max_parallel_tokens}]")

        output_token_ids = self._model_token_ids[:num_tokens]
        output_margin = self._model_margin[:num_tokens] if compute_margin else None

        if self.pp_group.is_last_rank:
            if token_ids is None or token_ids.shape != (num_tokens,):
                raise ValueError(
                    f"token_ids must have shape ({num_tokens},) on the final rank"
                )
            if token_ids.device != self.device:
                raise ValueError("token_ids is on the wrong device")
            output_token_ids.copy_(token_ids)

            if compute_margin:
                if margin is None or margin.shape != (num_tokens,):
                    raise ValueError(
                        f"margin must have shape ({num_tokens},) on the final rank"
                    )
                if margin.device != self.device:
                    raise ValueError("margin is on the wrong device")
                assert output_margin is not None
                output_margin.copy_(margin)

        if self.pp_group.world_size > 1:
            torch.distributed.broadcast(
                output_token_ids,
                src=self.pp_group.last_rank,
                group=self.pp_group.device_group,
            )
            if output_margin is not None:
                torch.distributed.broadcast(
                    output_margin,
                    src=self.pp_group.last_rank,
                    group=self.pp_group.device_group,
                )

        return RetroSpecPipelineModelOutput(output_token_ids, output_margin)

    def reduce_attention_mass(self, stats: RetroSpecAttentionMassStats) -> torch.Tensor:
        if stats.value_sum.device != self.device:
            raise ValueError("Attention-mass statistics are on the wrong device")

        batch_size = stats.value_sum.numel()
        if batch_size > self.max_parallel_tokens:
            raise ValueError(
                "Attention-mass batch size "
                f"{batch_size} exceeds {self.max_parallel_tokens}"
            )

        reduction = self._attention_reduce[: batch_size + 1]
        reduction[:batch_size].copy_(stats.value_sum)
        reduction[batch_size].fill_(stats.layer_count)

        reduction = self.pp_group.all_reduce(reduction)
        return reduction[:batch_size] / reduction[batch_size]

    def _copy_integer_control(
        self,
        name: str,
        destination: torch.Tensor,
        source: torch.Tensor,
        batch_size: int,
    ) -> None:
        if source.shape != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},)")
        if source.device != self.device:
            raise ValueError(f"{name} is on the wrong device")
        if source.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{name} must use an integer dtype")
        destination.copy_(source)

    def broadcast_control_state(
        self,
        batch_size: int,
        state: RetroSpecPipelineControlState | None,
    ) -> RetroSpecPipelineControlState:
        if not 0 <= batch_size <= self.max_batch_size:
            raise ValueError(f"batch_size must be in [0, {self.max_batch_size}]")

        integer_control = self._integer_control[:batch_size]
        active_mask = self._active_mask[:batch_size]

        if self.pp_group.is_last_rank:
            if state is None:
                raise RuntimeError(
                    "The final RetroSpec PP rank must provide control state"
                )

            self._copy_integer_control(
                "token_ids",
                integer_control[:, self._TOKEN_IDS_COLUMN],
                state.token_ids,
                batch_size,
            )
            self._copy_integer_control(
                "stages",
                integer_control[:, self._STAGES_COLUMN],
                state.stages,
                batch_size,
            )
            self._copy_integer_control(
                "draft_counts",
                integer_control[:, self._DRAFT_COUNTS_COLUMN],
                state.draft_counts,
                batch_size,
            )
            self._copy_integer_control(
                "pending_counts",
                integer_control[:, self._PENDING_COUNTS_COLUMN],
                state.pending_counts,
                batch_size,
            )

            if state.active_mask.shape != (batch_size,):
                raise ValueError(f"active_mask must have shape ({batch_size},)")
            if state.active_mask.device != self.device:
                raise ValueError("active_mask is on the wrong device")
            if state.active_mask.dtype != torch.bool:
                raise ValueError("active_mask must have dtype torch.bool")
            active_mask.copy_(state.active_mask)

        if batch_size > 0 and self.pp_group.world_size > 1:
            torch.distributed.broadcast(
                integer_control,
                src=self.pp_group.last_rank,
                group=self.pp_group.device_group,
            )
            torch.distributed.broadcast(
                active_mask,
                src=self.pp_group.last_rank,
                group=self.pp_group.device_group,
            )

        # These views remain valid only until the next control broadcast.
        return RetroSpecPipelineControlState(
            token_ids=integer_control[:, self._TOKEN_IDS_COLUMN],
            stages=integer_control[:, self._STAGES_COLUMN],
            draft_counts=integer_control[:, self._DRAFT_COUNTS_COLUMN],
            pending_counts=integer_control[:, self._PENDING_COUNTS_COLUMN],
            active_mask=active_mask,
        )
