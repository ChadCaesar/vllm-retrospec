# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Sequence
from enum import IntEnum

import torch


class RetroSpecStage(IntEnum):
    IDLE = 0
    DRAFT = 1
    SPARSE_VERIFY = 2
    EXPANDED_VERIFY = 3
    FULL_VERIFY = 4


class RetroSpecIndexUpdateState:
    """Track the next index-update boundary for each request.

    Positions are sequence lengths rather than zero-based token positions.
    Request state is stored by request ID so batch reordering and preemption
    do not associate an update boundary with the wrong request.
    """

    def __init__(
        self,
        max_batch_size: int,
        update_interval: int,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if update_interval <= 0:
            raise ValueError("update_interval must be greater than zero")

        self.max_batch_size = max_batch_size
        self.update_interval = update_interval
        self.device = device
        self.pin_memory = pin_memory

        self._next_update_by_request: dict[str, int] = {}

        self._next_update_positions_cpu = torch.empty(
            max_batch_size,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._next_update_positions = torch.empty(
            max_batch_size,
            dtype=torch.int64,
            device=device,
        )

        self.batch_size = 0

    @property
    def next_update_positions(self) -> torch.Tensor:
        return self._next_update_positions[: self.batch_size]

    def begin_batch(
        self,
        request_ids: Sequence[str],
        committed_positions: Sequence[int],
    ) -> None:
        batch_size = len(request_ids)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch size {batch_size} exceeds maximum {self.max_batch_size}"
            )
        if len(committed_positions) != batch_size:
            raise ValueError(
                "committed_positions must have the same length as request_ids"
            )
        if len(set(request_ids)) != batch_size:
            raise ValueError("request_ids must be unique within a batch")

        self.batch_size = batch_size

        for request_index, (request_id, position) in enumerate(
            zip(request_ids, committed_positions)
        ):
            if position < 0:
                raise ValueError("committed positions must be non-negative")

            next_update_position = self._next_update_by_request.get(request_id)

            if next_update_position is None:
                # The current prompt and committed output are already covered
                # by the index available when the request first enters RetroSpec.
                next_update_position = position + self.update_interval
            elif position >= next_update_position:
                # A previous full-verification pass committed enough tokens to
                # cross one or more index-update boundaries.
                crossed_intervals = (
                    position - next_update_position
                ) // self.update_interval + 1
                next_update_position += crossed_intervals * self.update_interval

            self._next_update_by_request[request_id] = next_update_position
            self._next_update_positions_cpu[request_index] = next_update_position

        if batch_size == 0:
            return

        self._next_update_positions[:batch_size].copy_(
            self._next_update_positions_cpu[:batch_size],
            non_blocking=self.pin_memory,
        )

    def requires_update(
        self,
        candidate_positions: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_positions.shape != (self.batch_size,):
            raise ValueError("candidate_positions must match the current batch size")
        if candidate_positions.device != self.device:
            raise ValueError("candidate_positions must be on the state device")
        if candidate_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_positions must use an integer dtype")

        if active_mask.shape != (self.batch_size,):
            raise ValueError("active_mask must match the current batch size")
        if active_mask.device != self.device:
            raise ValueError("active_mask must be on the state device")
        if active_mask.dtype != torch.bool:
            raise ValueError("active_mask must have dtype torch.bool")

        return active_mask & (candidate_positions >= self.next_update_positions)

    def remove_requests(self, request_ids: Collection[str]) -> None:
        for request_id in request_ids:
            self._next_update_by_request.pop(request_id, None)


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

    def reset_draft_counts(self, mask: torch.Tensor) -> None:
        self._validate_mask(mask)
        self.draft_counts.masked_fill_(mask, 0)

    def add_draft_counts(self, counts: torch.Tensor) -> None:
        self._validate_counts("counts", counts)
        self.draft_counts.add_(counts)

    def add_pending_counts(self, counts: torch.Tensor) -> None:
        self._validate_counts("counts", counts)
        self.pending_counts.add_(counts)

    def set_pending_counts(self, counts: torch.Tensor) -> None:
        self._validate_counts("counts", counts)
        self.pending_counts.copy_(counts.to(self.pending_counts.dtype))

    def finish_requests(self, mask: torch.Tensor) -> None:
        self._validate_mask(mask)

        self.stage.masked_fill_(mask, int(RetroSpecStage.IDLE))
        self.draft_counts.masked_fill_(mask, 0)
        self.pending_counts.masked_fill_(mask, 0)
        self.active_mask.masked_fill_(mask, False)
