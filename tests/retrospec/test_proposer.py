# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.retrospec import (
    RetroSpecAttentionMode,
    RetroSpecProposer,
)
from vllm.v1.spec_decode.retrospec.proposer import RetroSpecVerificationResult
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


@pytest.fixture(autouse=True)
def disable_pin_memory_for_cpu_tests(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.is_pin_memory_available",
        lambda: False,
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
            ),
            cache_config=SimpleNamespace(block_size=4),
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
    return cast(
        SamplingMetadata,
        SimpleNamespace(all_greedy=all_greedy),
    )


def run_proposal(
    proposer: RetroSpecProposer,
    next_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    common_attn_metadata: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
    request_ids: list[str] | None = None,
    committed_positions: list[int] | None = None,
) -> list[list[int]]:
    batch_size = common_attn_metadata.batch_size()
    if request_ids is None:
        request_ids = [f"request-{index}" for index in range(batch_size)]
    if committed_positions is None:
        committed_positions = common_attn_metadata.seq_lens.tolist()

    return proposer.propose(
        request_ids=request_ids,
        committed_positions=committed_positions,
        next_token_ids=next_token_ids,
        sampling_metadata=sampling_metadata,
        common_attn_metadata=common_attn_metadata,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )


def mock_proposal_execution(
    proposer: RetroSpecProposer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids: nullcontext(),
    )
    monkeypatch.setattr(
        proposer,
        "_verify_draft_tokens",
        lambda *args: RetroSpecVerificationResult(
            verified_counts=proposer.state.draft_counts.clone(),
            require_full=proposer.state.active_mask.clone(),
        ),
    )


def test_retrospec_proposer_initialization():
    vllm_config = make_vllm_config()
    runner = make_runner()
    device = torch.device("cpu")

    proposer = RetroSpecProposer(vllm_config, device, runner)

    assert proposer.vllm_config is vllm_config
    assert proposer.device == device
    assert proposer.runner is runner
    assert proposer.model is None
    assert proposer.num_speculative_tokens == 4
    assert proposer.max_batch_size == 8
    assert proposer.policy.max_draft_tokens == 4
    assert proposer.state.max_batch_size == 8
    assert proposer.state.device == device
    assert proposer.attn_metadata_builder is None
    assert proposer.attn_layer_names == []


def test_retrospec_proposer_loads_target_model():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    target_model = Mock()

    install = Mock()
    proposer.sparse_attention.install = install
    attention_layer = Mock()

    with patch(
        "vllm.v1.spec_decode.retrospec.proposer.get_layers_from_vllm_config",
        return_value={"model.layers.0.self_attn.attn": attention_layer},
    ):
        proposer.load_model(target_model)

    assert proposer.model is target_model
    assert proposer.attn_layer_names == ["model.layers.0.self_attn.attn"]
    install.assert_called_once_with({"model.layers.0.self_attn.attn": attention_layer})


@pytest.mark.parametrize(
    ("config", "runner", "message"),
    [
        (
            make_vllm_config(disable_padded_drafter_batch=True),
            make_runner(),
            "padded drafter batches",
        ),
        (
            make_vllm_config(async_scheduling=True),
            make_runner(),
            "async scheduling",
        ),
        (
            make_vllm_config(),
            make_runner(supports_mm_inputs=True),
            "multimodal models",
        ),
        (
            make_vllm_config(),
            make_runner(uses_mrope=True),
            "one-dimensional RoPE",
        ),
        (
            make_vllm_config(retrospec_cache_mode="cpu_offload"),
            make_runner(),
            "CPU-offloaded KV cache",
        ),
    ],
)
def test_retrospec_proposer_rejects_unsupported_features(
    config: VllmConfig,
    runner: Any,
    message: str,
):
    with pytest.raises(NotImplementedError, match=message):
        RetroSpecProposer(config, torch.device("cpu"), runner)


def test_propose_rejects_random_sampling():
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )

    with pytest.raises(NotImplementedError, match="greedy decoding only"):
        run_proposal(
            proposer,
            torch.tensor([1], dtype=torch.int32),
            make_sampling_metadata(all_greedy=False),
            make_common_metadata([1]),
        )


def test_propose_stops_requests_independently_on_draft_margin(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_draft_margin_threshold=1.0),
        torch.device("cpu"),
        make_runner(),
    )
    expected_masks = [
        [True, True, True],
        [True, False, True],
        [False, False, True],
    ]
    margins = [
        [2.0, 0.5, 2.0],
        [0.5, 0.0, 2.0],
        [0.0, 0.0, 0.5],
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        assert batch_size == 3
        assert active_mask.tolist() == expected_masks[draft_index]
        token_base = 10 * (draft_index + 1)
        token_ids = torch.tensor(
            [token_base + i for i in range(batch_size)], dtype=torch.int32
        )
        draft_margin = torch.tensor(margins[draft_index])
        hit_attn = torch.ones(batch_size)
        return token_ids, draft_margin, hit_attn

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1, 1]),
    )

    assert result == [[10, 20], [11], [12, 22, 32]]


def test_propose_stops_at_configured_max_draft_tokens(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=3),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1]),
    )

    assert result == [[1, 2, 3], [1, 2, 3]]


def test_propose_respects_per_request_generation_limit(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(max_model_len=6),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        token_base = 10 * (draft_index + 1)
        return (
            torch.tensor(
                [token_base + i for i in range(batch_size)], dtype=torch.int32
            ),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([4, 3, 5]),
    )

    assert result == [[10], [11, 21], []]


def test_propose_rolls_back_rejected_tokens_before_drafting(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(max_model_len=6),
        torch.device("cpu"),
        make_runner(),
    )

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.tensor([draft_index + 1], dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([5]),
        num_rejected_tokens_gpu=torch.tensor([2], dtype=torch.int32),
    )

    assert result == [[1, 2]]


def test_run_draft_step_preserves_attention_seq_lens_dtype(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(),
        torch.device("cpu"),
        make_runner(),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
    )

    class FakeBuilder:
        def build_for_drafting(self, metadata, draft_index):
            assert metadata.seq_lens.dtype == common_attn_metadata.seq_lens.dtype
            return SimpleNamespace()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds):
            return torch.zeros((input_ids.shape[0], 4))

        def compute_logits(self, hidden_states):
            return torch.tensor([[0.0, 1.0]])

    proposer.model = FakeModel()
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.attn_metadata_builder = cast(Any, FakeBuilder())
    proposer.runner.sampler = lambda **kwargs: SimpleNamespace(
        sampled_token_ids=torch.tensor([[1]], dtype=torch.int32)
    )
    proposer.sparse_attention.begin_step = Mock()
    proposer.sparse_attention.end_step = Mock(return_value=torch.ones(1))
    proposer.input_ids[0] = 1
    proposer.positions[0] = 3
    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.proposer.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    proposer._run_draft_step(
        batch_size=1,
        draft_index=0,
        common_attn_metadata=common_attn_metadata,
        active_mask=torch.tensor([True]),
        sampling_metadata=make_sampling_metadata(all_greedy=True),
    )

    proposer.sparse_attention.begin_step.assert_called_once()
    proposer.sparse_attention.end_step.assert_called_once_with()


def test_propose_stops_requests_independently_on_hit_attention(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_hit_attn_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    expected_masks = [
        [True, True],
        [True, False],
    ]
    hit_attn_values = [
        [0.8, 0.4],
        [0.3, 1.0],
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        assert active_mask.tolist() == expected_masks[draft_index]
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.tensor(hit_attn_values[draft_index]),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([1, 2], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1, 1]),
    )

    assert result == [[1, 2], [1]]


def initialize_verification(
    proposer: RetroSpecProposer,
    draft_token_ids: torch.Tensor,
    draft_counts: torch.Tensor,
    pending_counts: torch.Tensor | None = None,
) -> None:
    batch_size = draft_token_ids.shape[0]
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


def test_propose_stops_draft_at_index_update_boundary(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=2),
        torch.device("cpu"),
        make_runner(),
    )
    observed_indices: list[int] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        observed_indices.append(draft_index)
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    mock_proposal_execution(proposer, monkeypatch)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([1]),
        committed_positions=[1],
    )

    assert observed_indices == [0, 1]
    assert result == [[1, 2]]


def test_sparse_verification_requires_full_at_index_update_boundary(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=2),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    observed_indices: list[int] = []

    def fake_run_verification_step(
        batch_size,
        draft_index,
        input_ids,
        active_mask,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        observed_indices.append(draft_index)
        return (
            proposer._draft_token_ids[:batch_size, draft_index].clone(),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(
        proposer,
        "_run_verification_step",
        fake_run_verification_step,
    )

    verification = proposer._verify_draft_tokens(
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed_indices == [0, 1]
    assert verification.verified_counts.tolist() == [2]
    assert verification.require_full.tolist() == [True]


def test_remove_requests_resets_proposer_index_update_boundary():
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_index_update_interval=4),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.index_update_state.begin_batch(["request"], [10])

    proposer.remove_requests({"request"})
    proposer.index_update_state.begin_batch(["request"], [100])

    assert proposer.index_update_state.next_update_positions.tolist() == [104]


def test_verify_unchanged_sparse_tokens_keeps_complete_prefix(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    observed_inputs: list[list[int]] = []

    def fake_run_verification_step(
        batch_size,
        draft_index,
        input_ids,
        active_mask,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        assert attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
        assert active_mask.tolist() == [True]
        observed_inputs.append(input_ids[:batch_size].tolist())
        return (
            proposer._draft_token_ids[:batch_size, draft_index].clone(),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_verification_step", fake_run_verification_step)
    verification = proposer._verify_draft_tokens(
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [3]
    assert not verification.require_full.any()
    assert proposer._draft_token_ids[0, :3].tolist() == [10, 20, 30]
    assert observed_inputs == [[7], [10], [20]]
    assert proposer.state.stage.tolist() == [int(RetroSpecStage.DRAFT)]


def test_sparse_token_change_is_corrected_and_truncates_prefix(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    observed_modes: list[RetroSpecAttentionMode] = []

    def fake_run_verification_step(
        batch_size,
        draft_index,
        input_ids,
        active_mask,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        observed_modes.append(attention_mode)
        return torch.tensor([11], dtype=torch.int32), None, torch.ones(1)

    monkeypatch.setattr(proposer, "_run_verification_step", fake_run_verification_step)
    verification = proposer._verify_draft_tokens(
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert not verification.require_full.any()
    assert proposer._draft_token_ids[0, 0].item() == 11
    assert observed_modes == [
        RetroSpecAttentionMode.SPARSE_VERIFY,
        RetroSpecAttentionMode.EXPANDED_VERIFY,
    ]


def test_expanded_verification_passes_or_stops_requests_independently(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(
            retrospec_sparse_margin_threshold=0.5,
            retrospec_expanded_margin_threshold=0.5,
        ),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, -1], [11, 21, 31, -1]], dtype=torch.int32),
        torch.tensor([3, 3], dtype=torch.int32),
    )
    expanded_masks: list[list[bool]] = []

    def fake_run_verification_step(
        batch_size,
        draft_index,
        input_ids,
        active_mask,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        token_ids = proposer._draft_token_ids[:batch_size, draft_index].clone()
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            margin = torch.tensor([0.1, 0.1])
            return token_ids, margin, torch.ones(batch_size)

        expanded_masks.append(active_mask.tolist())
        margin = torch.tensor([0.9, 0.1])
        return token_ids, margin, torch.ones(batch_size)

    monkeypatch.setattr(proposer, "_run_verification_step", fake_run_verification_step)
    verification = proposer._verify_draft_tokens(
        2,
        torch.zeros(2, dtype=torch.int32),
        make_common_metadata([1, 1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [3, 1]
    assert verification.require_full.tolist() == [False, True]
    assert expanded_masks == [[True, True], [True, False], [True, False]]
    assert proposer.state.stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.FULL_VERIFY),
    ]


def test_expanded_token_change_replaces_current_token_before_truncation(
    monkeypatch,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, -1, -1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )

    def fake_run_verification_step(
        batch_size,
        draft_index,
        input_ids,
        active_mask,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        if attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY:
            return torch.tensor([10]), torch.tensor([0.1]), torch.ones(1)
        return torch.tensor([99]), None, torch.ones(1)

    monkeypatch.setattr(proposer, "_run_verification_step", fake_run_verification_step)
    verification = proposer._verify_draft_tokens(
        1,
        torch.zeros(1, dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert verification.verified_counts.tolist() == [1]
    assert verification.require_full.tolist() == [True]
    assert proposer._draft_token_ids[0, 0].item() == 99


def test_verify_only_processes_current_logical_draft_interval(monkeypatch):
    proposer = RetroSpecProposer(make_vllm_config(), torch.device("cpu"), make_runner())
    initialize_verification(
        proposer,
        torch.tensor([[10, 20, 30, 40]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        pending_counts=torch.tensor([2], dtype=torch.int32),
    )
    observed_indices: list[int] = []
    observed_inputs: list[list[int]] = []

    def fake_run_verification_step(
        batch_size,
        draft_index,
        input_ids,
        active_mask,
        common_attn_metadata,
        sampling_metadata,
        attention_mode,
    ):
        assert attention_mode == RetroSpecAttentionMode.SPARSE_VERIFY
        observed_indices.append(draft_index)
        observed_inputs.append(input_ids[:batch_size].tolist())
        return (
            proposer._draft_token_ids[:batch_size, draft_index].clone(),
            None,
            torch.ones(batch_size),
        )

    monkeypatch.setattr(proposer, "_run_verification_step", fake_run_verification_step)
    verification = proposer._verify_draft_tokens(
        1,
        torch.tensor([2], dtype=torch.int32),
        make_common_metadata([1]),
        make_sampling_metadata(all_greedy=True),
    )

    assert observed_indices == [2, 3]
    assert observed_inputs == [[20], [30]]
    assert verification.verified_counts.tolist() == [2]
    assert verification.require_full.tolist() == [True]


def test_propose_accumulates_multiple_draft_rounds(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=2),
        torch.device("cpu"),
        make_runner(),
    )
    round_starts: list[list[int]] = []

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        return (
            torch.full((batch_size,), draft_index + 1, dtype=torch.int32),
            None,
            torch.ones(batch_size),
        )

    def fake_verify(
        batch_size,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        round_starts.append(round_start_counts.tolist())
        verified_counts = proposer.state.draft_counts.clone()
        require_full = (
            round_start_counts + verified_counts >= proposer.policy.pending_limit
        )
        return RetroSpecVerificationResult(verified_counts, require_full)

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2]),
    )

    assert round_starts == [[0], [2]]
    assert result == [[1, 2, 3, 4]]
    assert proposer.state.pending_counts.tolist() == [4]
    assert proposer.state.stage.tolist() == [int(RetroSpecStage.FULL_VERIFY)]


def test_propose_handles_different_round_offsets_in_one_buffer(monkeypatch):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_max_draft_tokens=2),
        torch.device("cpu"),
        make_runner(),
    )
    round_starts: list[list[int]] = []
    draft_calls: list[tuple[int, list[bool], list[int]]] = []
    verified_by_round = [
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    ]

    def fake_run_draft_step(
        batch_size,
        draft_index,
        common_attn_metadata,
        active_mask,
        sampling_metadata,
    ):
        draft_calls.append(
            (
                draft_index,
                active_mask.tolist(),
                proposer.positions[:batch_size].tolist(),
            )
        )
        return (
            torch.tensor(
                [draft_index * 10 + row for row in range(batch_size)],
                dtype=torch.int32,
            ),
            None,
            torch.ones(batch_size),
        )

    def fake_verify(
        batch_size,
        round_start_counts,
        common_attn_metadata,
        sampling_metadata,
    ):
        round_index = len(round_starts)
        round_starts.append(round_start_counts.tolist())
        verified_counts = verified_by_round[round_index]
        require_full = (
            round_start_counts + verified_counts >= proposer.policy.pending_limit
        )
        return RetroSpecVerificationResult(verified_counts, require_full)

    monkeypatch.setattr(
        proposer.sparse_attention,
        "proposal_context",
        lambda _request_ids: nullcontext(),
    )
    monkeypatch.setattr(proposer, "_run_draft_step", fake_run_draft_step)
    monkeypatch.setattr(proposer, "_verify_draft_tokens", fake_verify)

    result = run_proposal(
        proposer,
        torch.tensor([7, 8], dtype=torch.int32),
        make_sampling_metadata(all_greedy=True),
        make_common_metadata([2, 4]),
    )

    assert round_starts == [[0, 0], [1, 2], [3, 4]]
    assert [(index, mask) for index, mask, _ in draft_calls] == [
        (0, [True, True]),
        (1, [True, True]),
        (1, [True, False]),
        (2, [True, True]),
        (3, [False, True]),
        (3, [True, False]),
    ]
    assert draft_calls[2][2] == [3, 6]
    assert draft_calls[-1][2] == [5, 8]
    assert result == [[0, 10, 20, 30], [1, 11, 21, 31]]
    assert proposer.state.pending_counts.tolist() == [4, 4]


@pytest.mark.parametrize(
    ("attention_mode", "expected_compute_margin"),
    [
        (RetroSpecAttentionMode.SPARSE_VERIFY, True),
        (RetroSpecAttentionMode.EXPANDED_VERIFY, False),
    ],
)
def test_run_verification_step_uses_stage_margin_and_original_positions(
    monkeypatch,
    attention_mode,
    expected_compute_margin,
):
    proposer = RetroSpecProposer(
        make_vllm_config(retrospec_sparse_margin_threshold=0.5),
        torch.device("cpu"),
        make_runner(),
    )
    proposer.proposal_start_positions[:2].copy_(torch.tensor([3, 5]))
    run_model_step = Mock(
        return_value=(torch.zeros(2, dtype=torch.int32), None, torch.ones(2))
    )
    monkeypatch.setattr(proposer, "_run_model_step", run_model_step)

    proposer._run_verification_step(
        batch_size=2,
        draft_index=1,
        input_ids=torch.tensor([10, 11], dtype=torch.int32),
        active_mask=torch.tensor([True, False]),
        common_attn_metadata=make_common_metadata([3, 5]),
        sampling_metadata=make_sampling_metadata(all_greedy=True),
        attention_mode=attention_mode,
    )

    call_kwargs = run_model_step.call_args.kwargs
    assert call_kwargs["positions"].tolist() == [4, 6]
    assert call_kwargs["compute_margin"] is expected_compute_margin
