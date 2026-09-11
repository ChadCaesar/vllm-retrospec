# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.outputs import KVCacheRetirement
from vllm.v1.spec_decode.retrospec import RetroSpecProposer
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_retrospec_registers_capture_sizes_before_spec_decode_rounding():
    class FakeMetadataBuilder:
        @classmethod
        def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
            return AttentionCGSupport.ALWAYS

    class FakeAttentionBackend:
        @classmethod
        def get_builder_cls(cls):
            return FakeMetadataBuilder

    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        max_cudagraph_capture_size=16,
    )

    def adjust_capture_sizes(uniform_decode_query_len, tensor_parallel_size):
        assert uniform_decode_query_len == 5
        assert tensor_parallel_size == 1
        compilation_config.cudagraph_capture_sizes = [5, 10, 15]
        compilation_config.max_cudagraph_capture_size = 15

    compilation_config.adjust_cudagraph_sizes_for_spec_decode = Mock(
        side_effect=adjust_capture_sizes
    )

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.compilation_config = compilation_config
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    runner.uniform_decode_query_len = 5
    runner.cudagraph_dispatcher = Mock()
    runner.speculative_config = SimpleNamespace(use_eagle=lambda: False)
    runner.drafter = RetroSpecProposer.__new__(RetroSpecProposer)
    runner.drafter.initialize_cudagraph_keys = Mock()

    runner._check_and_update_cudagraph_mode(
        [{FakeAttentionBackend}],
        [SimpleNamespace(kv_cache_spec=object())],
    )

    runner.cudagraph_dispatcher.initialize_cudagraph_keys.assert_called_once_with(
        CUDAGraphMode.FULL_AND_PIECEWISE, 5
    )
    runner.drafter.initialize_cudagraph_keys.assert_called_once_with(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        (1, 2, 4, 8, 16),
        16,
    )


def test_capture_cudagraphs_forces_registered_descriptor(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.is_global_first_rank", lambda: False
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.parallel_config = SimpleNamespace(use_ubatching=False)
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=0)
    runner.load_config = SimpleNamespace(use_tqdm_on_load=False)
    runner.lora_config = None
    runner._dummy_run = Mock()
    runner.maybe_remove_all_loras = Mock()
    descriptor = BatchDescriptor(4).relax_for_mixed_batch_cudagraphs()

    runner._capture_cudagraphs([descriptor], CUDAGraphMode.PIECEWISE)

    runner._dummy_run.assert_called_once()
    call = runner._dummy_run.call_args
    assert call.args == (4,)
    assert call.kwargs["cudagraph_capture_descriptor"] == descriptor
    assert call.kwargs["cudagraph_runtime_mode"] == CUDAGraphMode.PIECEWISE
    assert call.kwargs["is_graph_capturing"] is True


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


def test_layer_major_prefill_copies_only_resident_workspace_blocks():
    workspace_cache = torch.arange(2 * 5 * 2, dtype=torch.float32).view(2, 5, 2)
    workspace = SimpleNamespace(kv_cache=workspace_cache)
    native_cache = torch.full((2, 8, 2), -1.0)

    GPUModelRunner._copy_retrospec_prefill_resident_blocks(
        workspace,
        native_cache,
        torch.tensor([0, 3, 4]),
        torch.tensor([2, 5, 7]),
    )

    torch.testing.assert_close(native_cache[:, 2], workspace_cache[:, 0])
    torch.testing.assert_close(native_cache[:, 5], workspace_cache[:, 3])
    torch.testing.assert_close(native_cache[:, 7], workspace_cache[:, 4])
    assert torch.all(native_cache[:, [0, 1, 3, 4, 6]] == -1)


def test_estimate_layer_prefill_activation_bytes_uses_tp_local_widths():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(
        dtype=torch.float16,
        hf_text_config=SimpleNamespace(intermediate_size=14336),
        get_hidden_size=lambda: 4096,
        get_head_size=lambda: 128,
        get_num_attention_heads=lambda parallel_config: 16,
        get_num_kv_heads=lambda parallel_config: 4,
    )
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=2)

    estimated_bytes = runner._estimate_retrospec_prefill_activation_bytes_per_token()

    activation_elements = 4 * 4096 + 2 * 7168 + (16 + 2 * 4) * 128
    assert estimated_bytes == 2 * 2 * activation_elements


def test_build_layer_prefill_tile_plan_builds_metadata_once_per_tile():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._build_retrospec_prefill_tile_metadata = Mock(
        side_effect=lambda tile, builder: f"metadata-{tile.scheduled_start}"
    )
    tiles = {
        (0, 4): SimpleNamespace(scheduled_start=0),
        (4, 8): SimpleNamespace(scheduled_start=4),
        (8, 10): SimpleNamespace(scheduled_start=8),
        (0, 10): SimpleNamespace(scheduled_start=0),
    }
    workspace = SimpleNamespace(
        tile=Mock(side_effect=lambda start, end: tiles[start, end])
    )
    builder = Mock()

    tile_plan, full_prompt_tile = runner._build_retrospec_prefill_tile_plan(
        workspace,
        prompt_num_tokens=10,
        tile_size=4,
        builder=builder,
    )

    assert [tile.scheduled_start for tile, _ in tile_plan] == [0, 4, 8]
    assert [metadata for _, metadata in tile_plan] == [
        "metadata-0",
        "metadata-4",
        "metadata-8",
    ]
    assert full_prompt_tile is tiles[0, 10]
    assert runner._build_retrospec_prefill_tile_metadata.call_count == 3
