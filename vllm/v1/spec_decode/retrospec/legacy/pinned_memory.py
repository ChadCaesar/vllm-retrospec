# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from math import prod
from threading import Lock

import torch


class RetroSpecPinnedMemoryManager:
    """Account for all reusable RetroSpec pinned CPU allocations."""

    def __init__(self, enabled: bool, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self.enabled = enabled
        self.max_bytes = max_bytes
        self._allocated_bytes = 0
        self._allocations: dict[int, int] = {}
        self._lock = Lock()

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return self._allocated_bytes

    @property
    def available_bytes(self) -> int:
        with self._lock:
            return self.max_bytes - self._allocated_bytes

    @staticmethod
    def _required_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
        shape = tuple(shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError("Pinned allocation dimensions must be non-negative")
        return prod(shape) * dtype.itemsize

    def empty(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        label: str,
    ) -> torch.Tensor:
        return self.replace(None, shape, dtype, label)

    def replace(
        self,
        current: torch.Tensor | None,
        shape: Sequence[int],
        dtype: torch.dtype,
        label: str,
    ) -> torch.Tensor:
        shape = tuple(shape)
        if not self.enabled:
            return torch.empty(shape, dtype=dtype, device="cpu")

        required_bytes = self._required_bytes(shape, dtype)
        with self._lock:
            current_bytes = (
                0 if current is None else self._allocations.get(id(current), 0)
            )
            available_bytes = self.max_bytes - self._allocated_bytes + current_bytes
            if required_bytes > available_bytes:
                raise RuntimeError(
                    f"RetroSpec pinned allocation {label!r} requires "
                    f"{required_bytes} bytes, but only {available_bytes} bytes "
                    "remain; increase retrospec_max_pinned_memory or reduce "
                    "the clustering/verification working set"
                )

            replacement = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            if current is not None:
                self._allocations.pop(id(current), None)
            self._allocations[id(replacement)] = required_bytes
            self._allocated_bytes = (
                self._allocated_bytes - current_bytes + required_bytes
            )
            return replacement

    def release(self, tensor: torch.Tensor | None) -> None:
        if tensor is None or not self.enabled:
            return

        with self._lock:
            allocation_bytes = self._allocations.pop(id(tensor), 0)
            self._allocated_bytes -= allocation_bytes

    def assert_empty(self) -> None:
        if self.allocated_bytes:
            raise RuntimeError(
                "RetroSpec pinned-memory allocations remain after shutdown: "
                f"{self.allocated_bytes} bytes"
            )
