# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture small native KV ranges into a private packed snapshot."""

from vllm.triton_utils import tl, triton


@triton.jit
def _capture_native_kv_kernel(
    plan,
    block_table,
    output,
    valid,
    table_length,
    table_stride,
    BLOCK: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    part = tl.program_id(0)
    tile = tl.program_id(1)
    row = plan + part * 17
    dtype = output.dtype.element_ty
    keys = tl.load(row).to(tl.pointer_type(dtype))
    values = tl.load(row + 1).to(tl.pointer_type(dtype))
    ks0, ks1 = tl.load(row + 2), tl.load(row + 3)
    ks2, ks3 = tl.load(row + 4), tl.load(row + 5)
    vs0, vs1 = tl.load(row + 6), tl.load(row + 7)
    vs2, vs3 = tl.load(row + 8), tl.load(row + 9)
    start, length = tl.load(row + 10), tl.load(row + 11)
    heads, dimension = tl.load(row + 12), tl.load(row + 13)
    destination = tl.load(row + 14)
    blocks, block_size = tl.load(row + 15), tl.load(row + 16)

    # Validate the entire range once, including logical table bounds. Invalid
    # native rows are never dereferenced or published to the CPU target.
    if tile == 0:
        tokens = tl.arange(0, MAX_TOKENS)
        logical = (start + tokens) // block_size
        in_table = (logical >= 0) & (logical < table_length)
        physical = tl.load(
            block_table + logical * table_stride,
            mask=(tokens < length) & in_table,
            other=-1,
        )
        ok = in_table & (physical >= 0) & (physical < blocks)
        tl.store(valid + part, tl.sum(((tokens < length) & ~ok).to(tl.int32)) == 0)

    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    count = heads * length * dimension
    token = offsets // dimension % length
    head = offsets // (length * dimension)
    dim = offsets % dimension
    logical = (start + token) // block_size
    in_table = (logical >= 0) & (logical < table_length)
    physical = tl.load(
        block_table + logical * table_stride,
        mask=(offsets < count) & in_table,
        other=-1,
    ).to(tl.int64)
    native_offset = (start + token) % block_size
    mask = (offsets < count) & in_table & (physical >= 0) & (physical < blocks)
    key = tl.load(
        keys + physical * ks0 + native_offset * ks1 + head * ks2 + dim * ks3,
        mask=mask,
        other=0,
    )
    value = tl.load(
        values + physical * vs0 + native_offset * vs1 + head * vs2 + dim * vs3,
        mask=mask,
        other=0,
    )
    tl.store(output + destination + offsets, key, mask=offsets < count)
    tl.store(output + destination + count + offsets, value, mask=offsets < count)
