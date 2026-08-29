# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.outputs import KVCacheRetirement
from vllm.v1.spec_decode.retrospec import RetroSpecProposer
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_retrospec_proposal_runner(
    partial_prefill_mask: list[bool],
) -> tuple[GPUModelRunner, RetroSpecProposer]:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    drafter = RetroSpecProposer.__new__(RetroSpecProposer)
    drafter.prepare_next_token_ids_padded = Mock(
        return_value=(
            torch.tensor([10, 11], dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
        )
    )
    drafter.propose = Mock(return_value=[[], [12]])

    runner.speculative_config = SimpleNamespace(
        method="retrospec",
        disable_padded_drafter_batch=False,
    )
    runner.drafter = drafter
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["prefill", "decode"],
    )
    runner.requests = {
        "prefill": SimpleNamespace(num_computed_tokens=8),
        "decode": SimpleNamespace(num_computed_tokens=4),
    }
    runner.discard_request_mask = SimpleNamespace(
        np=np.array(partial_prefill_mask, dtype=np.bool_),
        gpu=torch.tensor(partial_prefill_mask, dtype=torch.bool),
    )
    runner._copy_valid_sampled_token_count = Mock()
    return runner, drafter


def call_retrospec_proposal(runner: GPUModelRunner) -> list[list[int]]:
    return runner.propose_draft_token_ids(
        scheduler_output=SimpleNamespace(total_num_scheduled_tokens=2),
        sampled_token_ids=torch.tensor([[10], [11]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(),
        hidden_states=torch.empty(0),
        sample_hidden_states=torch.empty(0),
        aux_hidden_states=None,
        spec_decode_metadata=None,
        common_attn_metadata=SimpleNamespace(),
        slot_mappings=None,
    )


def test_all_partial_prefill_rows_skip_retrospec_proposal():
    runner, drafter = make_retrospec_proposal_runner([True, True])

    result = call_retrospec_proposal(runner)

    assert result == [[], []]
    drafter.prepare_next_token_ids_padded.assert_not_called()
    drafter.propose.assert_not_called()


def test_mixed_batch_only_activates_decode_rows_for_retrospec():
    runner, drafter = make_retrospec_proposal_runner([True, False])

    result = call_retrospec_proposal(runner)

    assert result == [[], [12]]
    proposal_active_mask = drafter.propose.call_args.args[-2]
    assert proposal_active_mask.tolist() == [False, True]


def test_completed_prefill_rows_can_start_retrospec_proposal():
    runner, drafter = make_retrospec_proposal_runner([False, False])

    call_retrospec_proposal(runner)

    proposal_active_mask = drafter.propose.call_args.args[-2]
    assert proposal_active_mask.tolist() == [True, True]


def test_list_draft_tokens_keep_proposal_request_order():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._draft_token_ids = [[1, 2], [3]]
    runner._draft_token_req_ids = ["request-a", "request-b"]
    runner.input_batch = SimpleNamespace(req_ids=["request-b", "request-c"])

    draft_token_ids, req_ids = runner._get_draft_token_ids_cpu()

    assert draft_token_ids == [[1, 2], [3]]
    assert req_ids == ["request-a", "request-b"]


def test_block_table_retires_standard_manager_blocks():
    table = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    table.add_row([7, 8, 9, 10, 11], row_idx=0)

    table.retire_blocks(row_idx=0, start_block=1, end_block=4)

    assert table.get_numpy_array()[0, :5].tolist() == [7, 0, 0, 0, 11]
    assert table.num_blocks_per_row[0] == 5


def test_block_table_retires_hybrid_manager_blocks():
    table = BlockTable(
        block_size=4,
        max_num_reqs=2,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    table.add_row([7, 8, 9], row_idx=0)

    table.retire_blocks(row_idx=0, start_block=1, end_block=2)

    assert table.get_numpy_array()[0, :6].tolist() == [14, 15, 0, 1, 18, 19]
    assert table.num_blocks_per_row[0] == 6


def test_gpu_model_runner_applies_retirement_to_cached_and_batched_state():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    request = SimpleNamespace(block_ids=([7, 8, 9, 10, 11],))
    block_table = Mock()
    runner.requests = {"request": request}
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"request": 2},
        block_table=block_table,
    )
    retirement = KVCacheRetirement(
        request_id="request",
        kv_cache_group_id=0,
        start_block=1,
        end_block=4,
    )

    runner._apply_retrospec_kv_retirements([retirement])

    assert request.block_ids[0] == [7, 0, 0, 0, 11]
    block_table.retire_blocks.assert_called_once_with(
        kv_cache_group_id=0,
        row_idx=2,
        start_block=1,
        end_block=4,
    )
