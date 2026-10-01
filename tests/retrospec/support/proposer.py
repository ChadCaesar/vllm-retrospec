# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.retrospec import (
    RetroSpecAttentionMassStats,
    RetroSpecPipelineStage,
    RetroSpecProposer,
)
from vllm.v1.spec_decode.retrospec.proposer import (
    RetroSpecParallelVerificationOutput,
    RetroSpecVerificationResult,
)


@pytest.fixture(autouse=True)
def disable_pin_memory_for_cpu_tests(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.is_pin_memory_available",
        lambda: False,
    )
    pp_group = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        is_last_rank=True,
        last_rank=0,
        device_group=None,
        all_reduce=lambda tensor: tensor,
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.pipeline.get_pp_group", lambda: pp_group
    )


def make_vllm_config(
    *,
    max_model_len: int = 16,
    async_scheduling: bool = False,
    **spec_overrides: Any,
) -> VllmConfig:
    spec_values = {
        "method": "retrospec",
        "num_speculative_tokens": 4,
        "retrospec_max_draft_tokens": 4,
        **spec_overrides,
    }
    return cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SpeculativeConfig(**spec_values),
            scheduler_config=SimpleNamespace(
                max_num_seqs=8,
                async_scheduling=async_scheduling,
            ),
            model_config=SimpleNamespace(
                dtype=torch.float32,
                max_model_len=max_model_len,
                get_hidden_size=Mock(return_value=4),
            ),
            cache_config=SimpleNamespace(block_size=4),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                data_parallel_size=1,
            ),
        ),
    )


def make_runner(**overrides: Any) -> Any:
    values = {
        "supports_mm_inputs": False,
        "uses_mrope": False,
        "uses_xdrope_dim": 0,
        **overrides,
    }
    return SimpleNamespace(**values)


def make_common_metadata(seq_lens: list[int]) -> CommonAttentionMetadata:
    metadata = SimpleNamespace(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        batch_size=lambda: len(seq_lens),
    )
    return cast(CommonAttentionMetadata, metadata)


def make_sampling_metadata(*, all_greedy: bool) -> SamplingMetadata:
    max_batch_size = 8
    return SamplingMetadata(
        temperature=None if all_greedy else torch.ones(max_batch_size),
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(max_batch_size),
        presence_penalties=torch.zeros(max_batch_size),
        repetition_penalties=torch.ones(max_batch_size),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def run_proposal(
    proposer: RetroSpecProposer,
    next_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    common_attn_metadata: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
    request_ids: list[str] | None = None,
    committed_positions: list[int] | None = None,
    proposal_active_mask: torch.Tensor | None = None,
    remaining_generation_tokens: list[int] | None = None,
    valid_sampled_tokens_count: torch.Tensor | None = None,
) -> list[list[int]]:
    initialize_single_pipeline_stage(proposer)
    batch_size = common_attn_metadata.batch_size()
    if request_ids is None:
        request_ids = [f"request-{index}" for index in range(batch_size)]
    if committed_positions is None:
        committed_positions = common_attn_metadata.seq_lens.tolist()
    if proposal_active_mask is None:
        proposal_active_mask = torch.ones(batch_size, dtype=torch.bool)
    if remaining_generation_tokens is None:
        remaining_generation_tokens = [proposer.num_speculative_tokens] * batch_size
    if valid_sampled_tokens_count is None:
        valid_sampled_tokens_count = torch.zeros(batch_size, dtype=torch.int32)

    return proposer.propose(
        request_ids=request_ids,
        committed_positions=committed_positions,
        next_token_ids=next_token_ids,
        sampling_metadata=sampling_metadata,
        common_attn_metadata=common_attn_metadata,
        proposal_active_mask=proposal_active_mask,
        remaining_generation_tokens=remaining_generation_tokens,
        valid_sampled_tokens_count=valid_sampled_tokens_count,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )


def initialize_single_pipeline_stage(proposer: RetroSpecProposer) -> None:
    if proposer.pipeline_stage is None:
        num_layers = max(len(proposer.attn_layer_names), 1)
        proposer.pipeline_stage = RetroSpecPipelineStage(0, 1, 0, num_layers)


def attention_stats(size: int) -> RetroSpecAttentionMassStats:
    return RetroSpecAttentionMassStats(torch.ones(size), layer_count=1)


def mock_proposal_execution(
    proposer: RetroSpecProposer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids, _context_lens=None: nullcontext(),
    )
    monkeypatch.setattr(
        proposer,
        "_verify_draft_tokens",
        lambda *args: RetroSpecVerificationResult(
            verified_counts=proposer.state.draft_counts.clone(),
            require_full=proposer.state.active_mask.clone(),
        ),
    )


def initialize_verification(
    proposer: RetroSpecProposer,
    draft_token_ids: torch.Tensor,
    draft_counts: torch.Tensor,
    pending_counts: torch.Tensor | None = None,
) -> None:
    batch_size = draft_token_ids.shape[0]
    proposer.sparse_attention.maybe_prime_full_verification = Mock(return_value=False)
    proposer.state.begin_batch(batch_size)
    proposer.index_update_state.begin_batch(
        [f"request-{index}" for index in range(batch_size)],
        [1] * batch_size,
    )
    proposer.state.add_draft_counts(draft_counts)
    if pending_counts is not None:
        proposer.state.set_pending_counts(pending_counts)
    proposer._draft_token_ids[:batch_size].copy_(draft_token_ids)
    proposer.proposal_input_ids[:batch_size].copy_(
        torch.arange(7, 7 + batch_size, dtype=torch.int32)
    )
    proposer.proposal_start_positions[:batch_size].fill_(1)


def make_parallel_verification_output(
    request_indices: list[int],
    token_indices: list[int],
    token_ids: list[int],
    margin: list[float] | None = None,
    attention_mass: list[float] | None = None,
) -> RetroSpecParallelVerificationOutput:
    if attention_mass is None:
        attention_mass = [1.0] * len(request_indices)
    return RetroSpecParallelVerificationOutput(
        request_indices=torch.tensor(request_indices, dtype=torch.int64),
        token_indices=torch.tensor(token_indices, dtype=torch.int64),
        token_ids=torch.tensor(token_ids, dtype=torch.int32),
        margin=None if margin is None else torch.tensor(margin),
        attention_mass=torch.tensor(attention_mass),
    )
