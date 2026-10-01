# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_cluster_tokens_kernel(
    assignments,
    token_offsets,
    cluster_offsets,
    token_indices,
    assignment_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    cluster_stride: tl.constexpr,
    output_stride: tl.constexpr,
    num_tokens: tl.constexpr,
    first_token: tl.constexpr,
    block_size: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.program_id(1) * block_size + tl.arange(0, block_size)
    valid = tokens < num_tokens
    clusters = tl.load(
        assignments + head * assignment_stride + tokens, mask=valid, other=0
    )
    offsets = tl.load(
        token_offsets + head * offset_stride + tokens, mask=valid, other=0
    )
    starts = tl.load(
        cluster_offsets + head * cluster_stride + clusters, mask=valid, other=0
    )
    tl.store(
        token_indices + head * output_stride + starts + offsets,
        first_token + tokens,
        mask=valid,
    )
