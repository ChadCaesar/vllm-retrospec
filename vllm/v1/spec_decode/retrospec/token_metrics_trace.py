# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in diagnostics for accepted tokens and the first target mismatch."""

import json
from collections import defaultdict, deque
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class _TokenMetrics:
    token_index: int
    position: int
    token_id: int
    origin: str
    draft_margin: float | None = None
    hit_attn: float | None = None


class RetroSpecTokenMetricsTracer:
    """Synchronous diagnostics; disabled tracing never copies device tensors.

    Keep a FIFO of submitted proposals: the next proposal can be generated
    before bookkeeping reports the outcome of the previous target pass.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._starts: dict[str, int] = {}
        self._active: dict[str, dict[int, _TokenMetrics]] = {}
        self._sequence: dict[str, int] = defaultdict(int)
        self._pending: dict[str, deque[tuple[int, list[_TokenMetrics]]]] = defaultdict(
            deque
        )

    def begin(self, request_ids: Sequence[str], positions: torch.Tensor) -> None:
        if not self.enabled:
            return
        for request_id, position in zip(request_ids, positions.cpu().tolist()):
            self._starts[request_id] = int(position)
            self._active[request_id] = {}

    def record_draft(
        self,
        request_ids: Sequence[str],
        token_index: int,
        mask: torch.Tensor,
        token_ids: torch.Tensor,
        margin: torch.Tensor | None,
        hit_attn: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        if margin is None:
            raise RuntimeError("Token-metric tracing requires draft margins")
        rows = mask.nonzero(as_tuple=False).flatten()
        values = torch.stack(
            [token_ids.double(), margin.double(), hit_attn.double()], dim=1
        ).index_select(0, rows)
        for row, (token_id, draft_margin, hit) in zip(
            rows.cpu().tolist(), values.cpu().tolist()
        ):
            request_id = request_ids[row]
            self._active[request_id][token_index] = _TokenMetrics(
                token_index=token_index,
                position=self._starts[request_id] + token_index + 1,
                token_id=int(token_id),
                origin="draft",
                draft_margin=float(draft_margin),
                hit_attn=float(hit),
            )

    def record_replacements(
        self,
        request_ids: Sequence[str],
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        previous_ids: torch.Tensor,
        token_ids: torch.Tensor,
        origin: str,
    ) -> None:
        if not self.enabled:
            return
        changed = previous_ids != token_ids
        self._record_generated(
            request_ids,
            request_indices[changed],
            token_indices[changed],
            token_ids[changed],
            origin,
        )

    def record_bonus(
        self,
        request_ids: Sequence[str],
        mask: torch.Tensor,
        token_indices: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        rows = mask.nonzero(as_tuple=False).flatten()
        self._record_generated(
            request_ids,
            rows,
            token_indices.index_select(0, rows),
            token_ids.index_select(0, rows),
            "sparse_bonus",
        )

    def _record_generated(
        self,
        request_ids: Sequence[str],
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        token_ids: torch.Tensor,
        origin: str,
    ) -> None:
        values = torch.stack([request_indices, token_indices, token_ids], dim=1)
        for row, token_index, token_id in values.cpu().tolist():
            request_id = request_ids[row]
            self._active[request_id][token_index] = _TokenMetrics(
                token_index=token_index,
                position=self._starts[request_id] + token_index + 1,
                token_id=token_id,
                origin=origin,
            )

    def submit(
        self, request_ids: Sequence[str], token_ids: Sequence[Sequence[int]]
    ) -> None:
        if not self.enabled:
            return
        for request_id, tokens in zip(request_ids, token_ids):
            if not tokens:
                continue
            records = [self._active[request_id][i] for i in range(len(tokens))]
            if [r.token_id for r in records] != list(tokens):
                raise RuntimeError("Token-metric records do not match the proposal")
            sequence = self._sequence[request_id]
            self._sequence[request_id] += 1
            self._pending[request_id].append((sequence, records))

    @staticmethod
    def _emit(payload: dict) -> None:
        logger.info(
            "RetroSpec token metrics: %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def record_outcomes(
        self,
        request_ids: Sequence[str],
        proposed_counts: Sequence[int],
        valid_counts: Sequence[int],
        sampled_token_ids: Sequence[Sequence[int]] | None,
    ) -> None:
        if not self.enabled:
            return
        if sampled_token_ids is None:
            raise ValueError("Token-metric tracing requires target sampled token IDs")
        for request_id, proposed, valid, sampled in zip(
            request_ids, proposed_counts, valid_counts, sampled_token_ids
        ):
            if proposed <= 0:
                continue
            if not self._pending[request_id]:
                raise RuntimeError("No traced proposal for the target outcome")
            sequence, records = self._pending[request_id].popleft()
            if proposed > len(records) or len(sampled) != valid:
                raise RuntimeError("Target outcome has an inconsistent proposal size")
            accepted = min(proposed, max(int(valid) - 1, 0))
            if [r.token_id for r in records[:accepted]] != list(sampled[:accepted]):
                raise RuntimeError("Target accepted prefix does not match the proposal")
            mismatch = None
            if valid > 0 and accepted < proposed:
                mismatch = asdict(records[accepted])
                mismatch["target_token_id"] = int(sampled[accepted])
                if mismatch["token_id"] == mismatch["target_token_id"]:
                    raise RuntimeError("First target mismatch has matching token IDs")
            self._emit(
                {
                    "event": "target_outcome",
                    "request_id": request_id,
                    "proposal_id": sequence,
                    "proposed_count": proposed,
                    "submitted_count": len(records),
                    "valid_count": valid,
                    "accepted_count": accepted,
                    "accepted_tokens": [asdict(r) for r in records[:accepted]],
                    "first_mismatch": mismatch,
                }
            )

    def remove_requests(self, request_ids: Collection[str]) -> None:
        if not self.enabled:
            return
        for request_id in request_ids:
            pending = self._pending.pop(request_id, ())
            if pending:
                self._emit(
                    {
                        "event": "unverified_proposals",
                        "request_id": request_id,
                        "proposal_ids": [sequence for sequence, _ in pending],
                    }
                )
            self._active.pop(request_id, None)
            self._starts.pop(request_id, None)
            self._sequence.pop(request_id, None)
