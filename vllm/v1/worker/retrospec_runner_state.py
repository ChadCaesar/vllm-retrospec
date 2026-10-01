# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.outputs import ECConnectorOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


class RetroSpecPipelineProposalState(NamedTuple):
    """State retained by non-final PP ranks until sample_tokens()."""

    scheduler_output: "SchedulerOutput"
    spec_decode_metadata: SpecDecodeMetadata | None
    common_attn_metadata: CommonAttentionMetadata
