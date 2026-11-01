# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm.v1.spec_decode.retrospec.offload.command_buffers import (
    RetroSpecCommandBufferPool,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_commands_stay_on_cpu_under_a_cuda_default_device():
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device)
    stream = torch.cuda.Stream(device=device)
    with torch.device(device), pool.upload([3, 5, 7], (3,), stream) as commands:
        result = commands.clone()
    stream.synchronize()
    assert pool._buffers[0].host.device.type == "cpu"
    assert result.cpu().tolist() == [3, 5, 7]


@pytest.mark.parametrize("shape", [(32,), (8192,), (5462, 6)])
def test_metadata_sources_survive_a_delayed_h2d_under_cuda_default(shape):
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device)
    stream = torch.cuda.Stream(device=device)
    size = shape[0] if len(shape) == 1 else shape[0] * shape[1]
    results = []
    for value in (17, 29):
        values = (
            [value] * size
            if len(shape) == 1
            else [[value] * shape[1] for _ in range(shape[0])]
        )
        with torch.device(device), torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            with pool.stage_metadata(values, shape) as commands:
                results.append((commands.clone(), value))
        # In the small path the temporary pinned source has already been
        # released. Its allocator must keep it alive until the queued H2D.
        for _ in range(4):
            torch.empty(size, dtype=torch.int64, device="cpu", pin_memory=True).fill_(
                -999
            )
        stream.synchronize()
    for result, value in results:
        assert tuple(result.shape) == shape
        assert result.cpu().reshape(-1).tolist() == [value] * size


def test_busy_commands_do_not_wait_or_reuse_before_the_kernel_consumes_them():
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device, max_buffers=1, max_elements=128)
    first = torch.cuda.Stream(device=device)
    second = torch.cuda.Stream(device=device)
    output = torch.empty(32, dtype=torch.int64, device=device)
    other = torch.empty_like(output)
    with pool.upload([1] * 32, (32,), first) as commands:
        pointer = commands.data_ptr()
        output.copy_(commands)
    first.synchronize()
    # Warm the temporary batch's allocator size as well. This assertion tests
    # pool synchronization, not a first cudaHostAlloc/cudaMalloc invocation.
    temporary_host = torch.empty(32, dtype=torch.int64, pin_memory=True)
    temporary_gpu = torch.empty(32, dtype=torch.int64, device=device)
    del temporary_host, temporary_gpu
    with pool.upload([17] * 32, (32,), first) as commands:
        assert commands.data_ptr() == pointer
        # H2D alone is insufficient: leave the command reader pending.
        torch.cuda._sleep(500_000_000)
        output.copy_(commands)
    with pool.upload([29] * 32, (32,), second) as commands:
        assert commands.data_ptr() != pointer
        other.copy_(commands)
    second.synchronize()
    assert not pool._buffers[0].ready.query(), "A busy slot was synchronized"
    for _ in range(16):
        torch.empty(32, dtype=torch.int64, pin_memory=True).fill_(-999)
    first.synchronize()
    assert output.cpu().tolist() == [17] * 32
    assert other.cpu().tolist() == [29] * 32
    assert pool.fallbacks == 1
    assert pool.retained_bytes <= 128 * 8
    with pool.upload([41] * 32, (32,), first) as commands:
        assert commands.data_ptr() == pointer
        output.copy_(commands)
    first.synchronize()
    assert output.cpu().tolist() == [41] * 32


def test_command_pool_handles_concurrent_streams_and_capacity_growth():
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device, max_buffers=2, max_elements=4096)

    def submit(worker):
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            outputs = []
            for i in range(16):
                size = 32 if i % 2 else 1031
                value = worker * 100 + i
                target = torch.empty(size, dtype=torch.int64, device=device)
                with pool.upload([value] * size, (size,), stream) as commands:
                    target.copy_(commands)
                outputs.append((target, value))
            stream.synchronize()
            return [(t.cpu().tolist(), v) for t, v in outputs]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, (1, 2)))
    for result in results:
        for values, expected in result:
            assert values == [expected] * len(values)
    assert pool.retained_bytes <= 4096 * 8
    assert len(pool._buffers) <= 2


def test_oversized_and_invalid_commands_leave_reusable_buffers_intact():
    device = torch.device("cuda", torch.cuda.current_device())
    pool = RetroSpecCommandBufferPool(device, max_buffers=1, max_elements=32)
    stream = torch.cuda.Stream(device=device)
    with (
        pytest.raises(ValueError, match="must not be empty"),
        pool.upload([], (0,), stream),
    ):
        pass
    with pool.upload(list(range(65)), (65,), stream) as commands:
        result = commands.clone()
    stream.synchronize()
    assert result.cpu().tolist() == list(range(65))
    assert pool.retained_bytes == 0
    with pytest.raises(ValueError), pool.upload([1, 2], (3,), stream):
        pass
    stream.synchronize()
    with pool.upload([7] * 3, (3,), stream) as commands:
        result = commands.clone()
    stream.synchronize()
    assert result.cpu().tolist() == [7] * 3
