# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Copy fragmented CUDA staging pages without one driver call per span."""

from collections.abc import Sequence

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

from .command_buffers import RetroSpecCommandBufferPool


@triton.jit
def _copy_fragmented_pages_kernel(
    key_source,
    value_source,
    key_destination,
    value_destination,
    mapping,
    PAGE_BYTES: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    page = tl.program_id(0)
    source_page = tl.load(mapping + page * 2)
    destination_page = tl.load(mapping + page * 2 + 1)
    offsets = tl.program_id(1) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    source = source_page * PAGE_BYTES + offsets
    destination = destination_page * PAGE_BYTES + offsets
    mask = offsets < PAGE_BYTES
    # Byte pointers preserve every bit, including NaNs and quantized payloads.
    keys = tl.load(key_source + source, mask=mask, other=0)
    values = tl.load(value_source + source, mask=mask, other=0)
    tl.store(key_destination + destination, keys, mask=mask)
    tl.store(value_destination + destination, values, mask=mask)


def copy_resident_pages(
    key_source: torch.Tensor,
    value_source: torch.Tensor,
    key_destination: torch.Tensor,
    value_destination: torch.Tensor,
    source_pages: Sequence[int],
    destination_pages: Sequence[int],
    page_bytes: int,
    commands: RetroSpecCommandBufferPool,
) -> tuple[int, bool]:
    """Return span count and whether a fragmented copy used the batched kernel.

    Sources must be separate CUDA staging storage. Ordered/aliased writes keep
    the legacy implementation, as do small batches and large contiguous spans.
    All decisions and index checks use CPU metadata; no CUDA readback is added.
    """
    count = len(source_pages)
    if len(destination_pages) != count:
        raise ValueError("Source page and destination slot counts must match")
    if not key_source.is_cuda or count < 32:
        mapping = torch.tensor(
            tuple(zip(source_pages, destination_pages, strict=True)),
            dtype=torch.int64,
            device="cpu",
        ).view(count, 2)
        return _copy_coalesced(
            key_source,
            value_source,
            key_destination,
            value_destination,
            page_bytes,
            mapping,
        )

    # Build the mapping once. Vectorized CPU planning also feeds the memcpy
    # fallback, so long contiguous batches do not pay for a Python page scan
    # followed by another tensor construction.
    mapping = torch.empty((count, 2), dtype=torch.int64, device="cpu")
    rows = mapping.numpy()
    rows[:, 0] = source_pages
    rows[:, 1] = destination_pages
    contiguous = (rows[1:, 0] == rows[:-1, 0] + 1) & (rows[1:, 1] == rows[:-1, 1] + 1)
    spans = count - int(np.count_nonzero(contiguous))
    if spans < 32 or count * page_bytes > spans * (64 << 10):
        return _copy_coalesced(
            key_source,
            value_source,
            key_destination,
            value_destination,
            page_bytes,
            mapping,
        )
    tensors = (key_source, value_source, key_destination, value_destination)
    if page_bytes <= 0:
        raise ValueError("Page bytes must be positive")
    if any(t.device != key_destination.device for t in tensors):
        raise ValueError("Batched K/V pages must use the same CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("Batched K/V pages must be contiguous")
    if any(t.dtype != key_source.dtype for t in tensors):
        raise ValueError("K/V source and destination dtypes must match")
    if (
        key_source.nbytes != value_source.nbytes
        or key_destination.nbytes != value_destination.nbytes
    ):
        raise ValueError("K/V sources and destinations must have matching capacities")
    if key_source.nbytes % page_bytes or key_destination.nbytes % page_bytes:
        raise ValueError("K/V capacities must contain complete pages")
    if rows[:, 0].min() < 0 or rows[:, 0].max() >= key_source.nbytes // page_bytes:
        raise RuntimeError("source block index is out of range")
    if rows[:, 1].min() < 0 or rows[:, 1].max() >= key_destination.nbytes // page_bytes:
        raise RuntimeError("destination block index is out of range")
    if np.unique(rows[:, 1]).size != count:
        return _copy_coalesced(
            key_source,
            value_source,
            key_destination,
            value_destination,
            page_bytes,
            mapping,
        )
    sources = {
        key_source.untyped_storage().data_ptr(),
        value_source.untyped_storage().data_ptr(),
    }
    key_storage = key_destination.untyped_storage().data_ptr()
    value_storage = value_destination.untyped_storage().data_ptr()
    if (
        key_storage in sources
        or value_storage in sources
        or key_storage == value_storage
    ):
        return _copy_coalesced(
            key_source,
            value_source,
            key_destination,
            value_destination,
            page_bytes,
            mapping,
        )

    stream = torch.cuda.current_stream(key_destination.device)
    with commands.upload(rows, (count, 2), stream) as device_mapping:
        _copy_fragmented_pages_kernel[(count, triton.cdiv(page_bytes, 4096))](
            key_source.view(torch.uint8),
            value_source.view(torch.uint8),
            key_destination.view(torch.uint8),
            value_destination.view(torch.uint8),
            device_mapping,
            PAGE_BYTES=page_bytes,
            BLOCK_BYTES=min(4096, triton.next_power_of_2(page_bytes)),
            num_warps=4,
        )
    return spans, True


def _copy_coalesced(
    keys: torch.Tensor,
    values: torch.Tensor,
    target_keys: torch.Tensor,
    target_values: torch.Tensor,
    page_bytes: int,
    mapping: torch.Tensor,
) -> tuple[int, bool]:
    return ops.copy_kv_blocks_coalesced(
        keys, values, target_keys, target_values, page_bytes, mapping
    ), False
