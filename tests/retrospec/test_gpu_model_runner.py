# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_list_draft_tokens_keep_proposal_request_order():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._draft_token_ids = [[1, 2], [3]]
    runner._draft_token_req_ids = ["request-a", "request-b"]
    runner.input_batch = SimpleNamespace(req_ids=["request-b", "request-c"])

    draft_token_ids, req_ids = runner._get_draft_token_ids_cpu()

    assert draft_token_ids == [[1, 2], [3]]
    assert req_ids == ["request-a", "request-b"]
