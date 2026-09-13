# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Mapping, Sequence

import torch

from vllm.logger import init_logger

from .decision import RetroSpecReason
from .state import RetroSpecStage

logger = init_logger(__name__)


class RetroSpecTransitionTracer:
    """Emit compact request-level traces for RetroSpec stage transitions.

    Tracing is intentionally synchronous and is intended only for deterministic
    diagnostics. The disabled path returns before allocating or copying any
    temporary tensor.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    @staticmethod
    def _validate_vector(
        name: str,
        value: torch.Tensor,
        num_rows: int,
        device: torch.device,
    ) -> None:
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")
        if value.shape[0] != num_rows:
            raise ValueError(
                f"{name} has {value.shape[0]} rows, but expected {num_rows}"
            )
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}")

    @staticmethod
    def _reason_names(value: int) -> list[str]:
        return [
            reason.name
            for reason in RetroSpecReason
            if reason != RetroSpecReason.NONE and value & int(reason)
        ]

    @staticmethod
    def _stage_name(value: int) -> str:
        try:
            return RetroSpecStage(value).name
        except ValueError:
            return f"UNKNOWN_{value}"

    def record_masked(
        self,
        request_ids: Sequence[str],
        phase: str,
        proposal_round: int,
        mask: torch.Tensor,
        integer_fields: Mapping[str, torch.Tensor],
        float_fields: Mapping[str, torch.Tensor | None] | None = None,
    ) -> None:
        if not self.enabled:
            return

        num_rows = len(request_ids)
        self._validate_vector("mask", mask, num_rows, mask.device)
        if mask.dtype != torch.bool:
            raise ValueError("mask must use boolean dtype")

        for name, value in integer_fields.items():
            self._validate_vector(name, value, num_rows, mask.device)

        if float_fields is not None:
            for name, value in float_fields.items():
                if value is not None:
                    self._validate_vector(name, value, num_rows, mask.device)

        request_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if request_indices.numel() == 0:
            return

        compact_integer_fields = {
            name: value.index_select(0, request_indices)
            for name, value in integer_fields.items()
        }
        compact_float_fields = None
        if float_fields is not None:
            compact_float_fields = {
                name: None if value is None else value.index_select(0, request_indices)
                for name, value in float_fields.items()
            }

        self.record_compact(
            request_ids=request_ids,
            phase=phase,
            proposal_round=proposal_round,
            request_indices=request_indices,
            integer_fields=compact_integer_fields,
            float_fields=compact_float_fields,
        )

    def record_compact(
        self,
        request_ids: Sequence[str],
        phase: str,
        proposal_round: int,
        request_indices: torch.Tensor,
        integer_fields: Mapping[str, torch.Tensor],
        float_fields: Mapping[str, torch.Tensor | None] | None = None,
    ) -> None:
        if not self.enabled:
            return

        if request_indices.ndim != 1:
            raise ValueError("request_indices must be one-dimensional")
        if request_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("request_indices must use an integer dtype")

        num_rows = request_indices.shape[0]
        if num_rows == 0:
            return

        device = request_indices.device
        for name, value in integer_fields.items():
            self._validate_vector(name, value, num_rows, device)

        integer_names = list(integer_fields)
        integer_columns = [request_indices.to(dtype=torch.int64)]
        integer_columns.extend(
            value.to(dtype=torch.int64) for value in integer_fields.values()
        )
        integer_rows = torch.stack(integer_columns, dim=1).detach().cpu().tolist()

        float_names: list[str] = []
        float_columns: list[torch.Tensor] = []
        if float_fields is not None:
            for name, value in float_fields.items():
                if value is None:
                    continue
                self._validate_vector(name, value, num_rows, device)
                float_names.append(name)
                float_columns.append(value.to(dtype=torch.float32))

        float_rows: list[list[float]]
        if float_columns:
            float_rows = torch.stack(float_columns, dim=1).detach().cpu().tolist()
        else:
            float_rows = [[] for _ in range(num_rows)]

        records: list[dict[str, object]] = []
        for integer_row, float_row in zip(integer_rows, float_rows):
            request_index = int(integer_row[0])
            if request_index < 0 or request_index >= len(request_ids):
                raise ValueError(
                    f"request index {request_index} is outside the current batch"
                )

            record: dict[str, object] = {
                "request_id": request_ids[request_index],
                "request_index": request_index,
            }
            for name, value in zip(integer_names, integer_row[1:]):
                record[name] = int(value)
            for name, value in zip(float_names, float_row):
                record[name] = float(value)

            reasons = record.get("reasons")
            if reasons is not None:
                record["reason_names"] = self._reason_names(int(reasons))

            request_stage = record.get("request_stage")
            if request_stage is not None:
                record["request_stage_name"] = self._stage_name(int(request_stage))

            next_stage = record.get("next_stage")
            if next_stage is not None:
                record["next_stage_name"] = self._stage_name(int(next_stage))

            records.append(record)

        payload = {
            "phase": phase,
            "proposal_round": proposal_round,
            "records": records,
        }
        logger.info(
            "RetroSpec transition trace: %s",
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
