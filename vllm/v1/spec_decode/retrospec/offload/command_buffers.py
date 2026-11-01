# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded, nonblocking staging for short CPU-to-GPU command batches."""

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from math import prod
from threading import Lock
from weakref import WeakValueDictionary

import numpy as np
import torch


@dataclass(eq=False)
class _CommandBuffer:
    host: torch.Tensor
    device: torch.Tensor
    ready: torch.cuda.Event
    capacity: int
    busy: bool = False
    views: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor, np.ndarray]] = field(
        default_factory=dict
    )


class RetroSpecCommandBufferPool:
    """Reuse commands only after their last CUDA consumer has completed.

    Metadata has its own small bound, independent of offload payload staging:
    at most 8 MiB each of pinned and device storage per CUDA device. A busy or
    oversized batch uses a temporary allocation instead of waiting under an
    admission/arena lock. PyTorch's allocator protects those temporary sources.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_buffers: int = 8,
        max_elements: int = 1 << 20,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Command buffers require a CUDA device")
        if max_buffers <= 0 or max_elements <= 0:
            raise ValueError("Command-buffer limits must be positive")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_buffers = max_buffers
        self.max_elements = max_elements
        self._buffers: list[_CommandBuffer] = []
        self._elements = 0
        self._pending_allocations = 0
        self._lock = Lock()
        self.reuses = 0
        self.allocations = 0
        self.fallbacks = 0

    @property
    def retained_bytes(self) -> int:
        """Retained bytes on each of CPU and CUDA, excluding temporary batches."""
        with self._lock:
            return self._elements * 8

    def _acquire(self, elements: int) -> tuple[_CommandBuffer, bool]:
        capacity = max(min(1024, self.max_elements), 1 << (elements - 1).bit_length())
        replace = None
        pooled = False
        previous_capacity = 0
        with self._lock:
            # Capacity is immutable CPU metadata. Query only fitting slots,
            # stopping at the first completed consumer instead of scanning all.
            fitting = sorted(
                (b for b in self._buffers if not b.busy and b.capacity >= elements),
                key=lambda b: b.capacity,
            )
            for buffer in fitting:
                if buffer.ready.query():
                    buffer.busy = True
                    self.reuses += 1
                    return buffer, True
            smaller = sorted(
                (b for b in self._buffers if not b.busy and b.capacity < elements),
                key=lambda b: b.capacity,
            )
            replace = next((b for b in smaller if b.ready.query()), None)
            if replace is not None:
                previous_capacity = replace.capacity
            has_slot = (
                replace is not None
                or len(self._buffers) + self._pending_allocations < self.max_buffers
            )
            if (
                has_slot
                and self._elements - previous_capacity + capacity <= self.max_elements
            ):
                pooled = True
                if replace is not None:
                    replace.busy = True
                else:
                    self._pending_allocations += 1
                self._elements += capacity - previous_capacity
            else:
                self.fallbacks += 1
                capacity = elements

        # Allocation can enter the CUDA driver. Do not hold the pool lock while
        # allocating, and never synchronize a busy slot to make room.
        try:
            host = torch.empty(
                capacity, dtype=torch.int64, device="cpu", pin_memory=True
            )
            gpu = torch.empty(capacity, dtype=torch.int64, device=self.device)
            buffer = _CommandBuffer(
                host, gpu, torch.cuda.Event(), capacity=capacity, busy=True
            )
        except BaseException:
            if pooled:
                with self._lock:
                    self._elements -= capacity - previous_capacity
                    if replace is not None:
                        replace.busy = False
                    else:
                        self._pending_allocations -= 1
            raise
        with self._lock:
            self.allocations += 1
            if pooled:
                if replace is not None:
                    self._buffers[self._buffers.index(replace)] = buffer
                else:
                    self._pending_allocations -= 1
                    self._buffers.append(buffer)
        return buffer, pooled

    def stage_metadata(
        self,
        values: Sequence,
        shape: tuple[int, ...],
    ) -> AbstractContextManager[torch.Tensor]:
        """Stage metadata on this device's current stream.

        Values must already have the supplied layout, as handle and binding
        batches do. Small batches keep the allocator-managed path: bookkeeping costs
        more than their tensor construction. Large batches amortize the pool's
        completion checks and reuse storage. Consumers run on the same stream.
        """
        elements = prod(shape)
        if elements <= 0:
            raise ValueError("A command batch must not be empty")
        if elements < (8192 if len(shape) == 1 else 32768):
            host = torch.tensor(
                values, dtype=torch.int64, device="cpu", pin_memory=True
            )
            return nullcontext(host.to(device=self.device, non_blocking=True))
        return self.upload(values, shape, torch.cuda.current_stream(self.device))

    @contextmanager
    def upload(
        self,
        values: Sequence,
        shape: tuple[int, ...],
        stream: torch.cuda.Stream,
    ) -> Iterator[torch.Tensor]:
        elements = prod(shape)
        if elements <= 0:
            raise ValueError("A command batch must not be empty")
        if stream.device != self.device:
            raise ValueError("Commands and consumer stream must use one CUDA device")
        buffer, pooled = self._acquire(elements)
        try:
            # NumPy fills existing pinned storage directly from Python metadata;
            # no new tensor or pageable-to-pinned intermediate is needed.
            views = buffer.views.get(shape)
            if views is None:
                host = buffer.host[:elements].view(shape)
                commands = buffer.device[:elements].view(shape)
                array = host.numpy()
                # Bound Python metadata as well as backing storage. Each view
                # shares this slot; the busy flag guards its cache and writes.
                if len(buffer.views) == 4:
                    buffer.views.pop(next(iter(buffer.views)))
                buffer.views[shape] = views = (host, commands, array)
            host, commands, array = views
            array[...] = values
            with torch.cuda.stream(stream):
                commands.copy_(host, non_blocking=True)
                # The pool may be released before a non-default stream finishes.
                commands.record_stream(stream)
                yield commands
        finally:
            # This event follows the caller's kernel, not just the H2D copy.
            # Keep it private: published arena events must not be re-recorded
            # when another layer reuses this command slot.
            if pooled:
                buffer.ready.record(stream)
                with self._lock:
                    buffer.busy = False


_pools: WeakValueDictionary[torch.device, RetroSpecCommandBufferPool] = (
    WeakValueDictionary()
)
_pools_lock = Lock()


def get_command_buffer_pool(device: torch.device) -> RetroSpecCommandBufferPool:
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with _pools_lock:
        pool = _pools.get(device)
        if pool is None:
            pool = RetroSpecCommandBufferPool(device)
            _pools[device] = pool
        return pool
