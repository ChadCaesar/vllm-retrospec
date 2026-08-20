# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.v1.outputs import KVCacheRetirement
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


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
