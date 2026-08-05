# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import IntEnum

import torch


class RetroSpecStage(IntEnum):
    IDLE = 0
    DRAFT = 1
    SPARSE_VERIFY = 2
    EXPANDED_VERIFY = 3
    FULL_VERIFY = 4


class RetroSpecBatchState:
    def __init__(self, max_batch_size: int, device: torch.device) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")

        self.max_batch_size = max_batch_size
        self.device = device
        self.batch_size = 0

        self._stage = torch.full(
            (max_batch_size,),
            int(RetroSpecStage.IDLE),
            dtype=torch.int8,
            device=device,
        )
        self._draft_counts = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._pending_counts = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._active_mask = torch.zeros(max_batch_size, dtype=torch.bool, device=device)

    @property
    def stage(self) -> torch.Tensor:
        return self._stage[: self.batch_size]

    @property
    def draft_counts(self) -> torch.Tensor:
        return self._draft_counts[: self.batch_size]

    @property
    def pending_counts(self) -> torch.Tensor:
        return self._pending_counts[: self.batch_size]

    @property
    def active_mask(self) -> torch.Tensor:
        return self._active_mask[: self.batch_size]

    def begin_batch(self, batch_size: int) -> None:
        if not 0 <= batch_size <= self.max_batch_size:
            raise ValueError(f"batch_size must be in [0, {self.max_batch_size}]")

        self.batch_size = batch_size

        self._stage.fill_(int(RetroSpecStage.IDLE))
        self._draft_counts.zero_()
        self._pending_counts.zero_()
        self._active_mask.zero_()

        if batch_size == 0:
            return

        self._stage[:batch_size].fill_(int(RetroSpecStage.DRAFT))
        self._active_mask[:batch_size].fill_(True)

    def _validate_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != (self.batch_size,):
            raise ValueError(f"mask must have shape ({self.batch_size},)")
        if mask.device != self.device:
            raise ValueError("mask must be on the state device")
        if mask.dtype != torch.bool:
            raise ValueError("mask must have dtype torch.bool")

    def _validate_counts(self, name: str, counts: torch.Tensor) -> None:
        if counts.shape != (self.batch_size,):
            raise ValueError(f"{name} must have shape ({self.batch_size},)")
        if counts.device != self.device:
            raise ValueError(f"{name} must be on the state device")
        if counts.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use an integer dtype")

    def set_stage(self, mask: torch.Tensor, stage: RetroSpecStage) -> None:
        self._validate_mask(mask)
        self.stage.masked_fill_(mask, int(stage))

    def set_stages(self, stages: torch.Tensor) -> None:
        if stages.shape != (self.batch_size,):
            raise ValueError(f"stages must have shape ({self.batch_size},)")
        if stages.device != self.device:
            raise ValueError("stages must be on the state device")
        if stages.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("stages must use an integer dtype")

        self.stage.copy_(stages.to(dtype=self.stage.dtype))

    def add_draft_counts(self, counts: torch.Tensor) -> None:
        self._validate_counts("counts", counts)
        self.draft_counts.add_(counts)

    def add_pending_counts(self, counts: torch.Tensor) -> None:
        self._validate_counts("counts", counts)
        self.pending_counts.add_(counts)

    def finish_requests(self, mask: torch.Tensor) -> None:
        self._validate_mask(mask)

        self.stage.masked_fill_(mask, int(RetroSpecStage.IDLE))
        self.draft_counts.masked_fill_(mask, 0)
        self.pending_counts.masked_fill_(mask, 0)
        self.active_mask.masked_fill_(mask, False)
