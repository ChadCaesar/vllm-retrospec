# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.retrospec.decision import (
    RetroSpecDecisionPolicy,
    RetroSpecMetrics,
)
from vllm.v1.spec_decode.retrospec.state import (
    RetroSpecBatchState,
    RetroSpecStage,
)


def make_state(max_batch_size: int = 4) -> RetroSpecBatchState:
    return RetroSpecBatchState(
        max_batch_size=max_batch_size,
        device=torch.device("cpu"),
    )


def test_state_initialization():
    state = make_state()

    assert state.max_batch_size == 4
    assert state.device == torch.device("cpu")
    assert state.batch_size == 0
    assert state.stage.numel() == 0
    assert state.draft_counts.numel() == 0
    assert state.pending_counts.numel() == 0
    assert state.active_mask.numel() == 0


def test_begin_batch_initializes_active_requests():
    state = make_state()

    state.begin_batch(3)

    assert state.batch_size == 3
    assert state.stage.tolist() == [int(RetroSpecStage.DRAFT)] * 3
    assert state.draft_counts.tolist() == [0, 0, 0]
    assert state.pending_counts.tolist() == [0, 0, 0]
    assert state.active_mask.tolist() == [True, True, True]


def test_begin_empty_batch():
    state = make_state()

    state.begin_batch(0)

    assert state.batch_size == 0
    assert state.active_mask.numel() == 0


def test_begin_batch_rejects_invalid_size():
    state = make_state()

    with pytest.raises(ValueError, match="batch_size"):
        state.begin_batch(5)


def test_set_stage_only_changes_selected_requests():
    state = make_state()
    state.begin_batch(3)

    state.set_stage(
        torch.tensor([False, True, False]),
        RetroSpecStage.EXPANDED_VERIFY,
    )

    assert state.stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.EXPANDED_VERIFY),
        int(RetroSpecStage.DRAFT),
    ]


def test_set_stages_replaces_all_request_stages():
    state = make_state()
    state.begin_batch(3)

    state.set_stages(
        torch.tensor(
            [
                int(RetroSpecStage.DRAFT),
                int(RetroSpecStage.SPARSE_VERIFY),
                int(RetroSpecStage.FULL_VERIFY),
            ],
            dtype=torch.int32,
        )
    )

    assert state.stage.dtype == torch.int8
    assert state.stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.SPARSE_VERIFY),
        int(RetroSpecStage.FULL_VERIFY),
    ]


def test_decision_next_stage_can_be_applied_to_state():
    state = make_state()
    state.begin_batch(2)
    state.add_draft_counts(torch.tensor([4, 2], dtype=torch.int32))
    policy = RetroSpecDecisionPolicy(
        SpeculativeConfig(
            method="retrospec",
            num_speculative_tokens=8,
            retrospec_max_draft_tokens=4,
        )
    )

    decision = policy.evaluate(
        current_stage=RetroSpecStage.DRAFT,
        request_stages=state.stage,
        metrics=RetroSpecMetrics(),
        draft_counts=state.draft_counts,
        pending_counts=state.pending_counts,
        active_mask=state.active_mask,
    )
    state.set_stages(decision.next_stage)

    assert state.stage.tolist() == [
        int(RetroSpecStage.SPARSE_VERIFY),
        int(RetroSpecStage.DRAFT),
    ]


def test_add_counts_accumulates_per_request():
    state = make_state()
    state.begin_batch(3)

    state.add_draft_counts(torch.tensor([1, 2, 3], dtype=torch.int32))
    state.add_draft_counts(torch.tensor([1, 0, 1], dtype=torch.int32))
    state.add_pending_counts(torch.tensor([0, 2, 1], dtype=torch.int32))

    assert state.draft_counts.tolist() == [2, 2, 4]
    assert state.pending_counts.tolist() == [0, 2, 1]


def test_finish_requests_clears_only_selected_requests():
    state = make_state()
    state.begin_batch(3)
    state.add_draft_counts(torch.tensor([1, 2, 3], dtype=torch.int32))
    state.add_pending_counts(torch.tensor([3, 2, 1], dtype=torch.int32))
    state.set_stage(
        torch.tensor([False, True, True]),
        RetroSpecStage.FULL_VERIFY,
    )

    state.finish_requests(torch.tensor([False, True, False]))

    assert state.stage.tolist() == [
        int(RetroSpecStage.DRAFT),
        int(RetroSpecStage.IDLE),
        int(RetroSpecStage.FULL_VERIFY),
    ]
    assert state.draft_counts.tolist() == [1, 0, 3]
    assert state.pending_counts.tolist() == [3, 0, 1]
    assert state.active_mask.tolist() == [True, False, True]


def test_new_batch_clears_previous_state():
    state = make_state()
    state.begin_batch(3)
    state.add_draft_counts(torch.tensor([1, 2, 3], dtype=torch.int32))
    state.finish_requests(torch.tensor([True, False, False]))

    state.begin_batch(2)

    assert state.stage.tolist() == [int(RetroSpecStage.DRAFT)] * 2
    assert state.draft_counts.tolist() == [0, 0]
    assert state.pending_counts.tolist() == [0, 0]
    assert state.active_mask.tolist() == [True, True]


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([True]),
        torch.tensor([1, 0], dtype=torch.int32),
    ],
)
def test_invalid_masks_raise(mask: torch.Tensor):
    state = make_state()
    state.begin_batch(2)

    with pytest.raises(ValueError, match="mask"):
        state.set_stage(mask, RetroSpecStage.FULL_VERIFY)


@pytest.mark.parametrize(
    "counts",
    [
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1.0, 2.0]),
    ],
)
def test_invalid_counts_raise(counts: torch.Tensor):
    state = make_state()
    state.begin_batch(2)

    with pytest.raises(ValueError, match="counts"):
        state.add_draft_counts(counts)


@pytest.mark.parametrize(
    "stages",
    [
        torch.tensor([int(RetroSpecStage.DRAFT)], dtype=torch.int8),
        torch.tensor([1.0, 2.0]),
    ],
)
def test_invalid_stages_raise(stages: torch.Tensor):
    state = make_state()
    state.begin_batch(2)

    with pytest.raises(ValueError, match="stages"):
        state.set_stages(stages)
