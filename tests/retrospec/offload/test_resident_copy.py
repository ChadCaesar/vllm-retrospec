# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.offload.command_buffers import (
    RetroSpecCommandBufferPool,
)
from vllm.v1.spec_decode.retrospec.offload.resident_copy import (
    copy_resident_pages,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "dtype,page_bytes",
    [
        (torch.uint8, 6),
        (torch.bfloat16, 256),
        (torch.bfloat16, 4096),
        (torch.float32, 8192),
    ],
)
def test_fragmented_copy_preserves_raw_bits_and_cross_stream_dependencies(
    dtype, page_bytes
):
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device)
    producer = torch.cuda.Stream(device=device)
    copier = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    # Arbitrary bytes include float NaN payloads. Compare bytes, not float values.
    raw = (torch.arange(64 * page_bytes, dtype=torch.int64) % 251).to(torch.uint8)
    raw = raw.view(64, page_bytes)
    with torch.cuda.stream(producer):
        keys = raw.to(device).view(dtype)
        values = raw.flip(1).contiguous().to(device).view(dtype)
        target_k = torch.full_like(keys.view(torch.uint8), 0xAA).view(dtype)
        target_v = torch.full_like(values.view(torch.uint8), 0xBB).view(dtype)
        ready = torch.cuda.Event()
        ready.record(producer)
    source = list(reversed(range(32)))
    destination = [(i * 5) % 64 for i in range(32)]
    copier.wait_event(ready)
    with torch.cuda.stream(copier), torch.device(device):
        spans, batched = copy_resident_pages(
            keys, values, target_k, target_v, source, destination, page_bytes, pool
        )
        copied = torch.cuda.Event()
        copied.record(copier)
    assert spans == 32 and batched
    consumer.wait_event(copied)
    with torch.cuda.stream(consumer):
        actual_k = target_k.view(torch.uint8).clone()
        actual_v = target_v.view(torch.uint8).clone()
    consumer.synchronize()
    expected_k = torch.full_like(raw, 0xAA)
    expected_v = torch.full_like(raw, 0xBB)
    for s, d in zip(source, destination, strict=True):
        expected_k[d] = raw[s]
        expected_v[d] = raw[s].flip(0)
    assert torch.equal(actual_k.cpu(), expected_k)
    assert torch.equal(actual_v.cpu(), expected_v)


@pytest.mark.parametrize("case", ["contiguous", "small", "duplicate", "alias", "h2d"])
def test_copy_fallback_keeps_ordered_memcpy_semantics(case):
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device)
    raw = torch.arange(64 * 128, dtype=torch.int64).view(64, 128)
    keys = raw.to(device)
    values = (raw + 1000).to(device)
    target_k = torch.full_like(keys, -1)
    target_v = torch.full_like(values, -2)
    sources = list(reversed(range(32)))
    destinations = list(range(32))
    if case == "contiguous":
        sources = list(range(32))
    elif case == "small":
        sources = sources[:4]
        destinations = destinations[:4]
    elif case == "duplicate":
        destinations[-1] = destinations[0]
    elif case == "alias":
        target_k, target_v = keys, values
    elif case == "h2d":
        keys, values = keys.cpu().pin_memory(), values.cpu().pin_memory()
    expected_k, expected_v = target_k.cpu().clone(), target_v.cpu().clone()
    source_k, source_v = keys.cpu(), values.cpu()
    if case == "alias":
        source_k, source_v = expected_k, expected_v
    for s, d in zip(sources, destinations, strict=True):
        expected_k[d] = source_k[s].clone()
        expected_v[d] = source_v[s].clone()
    _, batched = copy_resident_pages(
        keys, values, target_k, target_v, sources, destinations, 1024, pool
    )
    assert not batched
    assert torch.equal(target_k.cpu(), expected_k)
    assert torch.equal(target_v.cpu(), expected_v)


@pytest.mark.parametrize("invalid_source", [-1, 32])
def test_fragmented_copy_rejects_out_of_bounds_without_writing(invalid_source):
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device)
    source = torch.zeros((32, 128), dtype=torch.bfloat16, device=device)
    target = torch.full_like(source, 7)
    source_ids = list(reversed(range(32)))
    source_ids[0] = invalid_source
    with pytest.raises(RuntimeError, match="source block index"):
        copy_resident_pages(
            source,
            source.clone(),
            target,
            target.clone(),
            source_ids,
            list(range(32)),
            256,
            pool,
        )
    assert torch.equal(target.cpu(), torch.full((32, 128), 7, dtype=torch.bfloat16))
