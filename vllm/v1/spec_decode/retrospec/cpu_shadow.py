# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded CPU lookahead which corrects candidates, never commits outputs."""

from collections import deque
from time import perf_counter

import torch
import torch.distributed as dist

from .cpu_risk import RetroSpecCPURiskCoordinator, RiskTask


class RetroSpecCPUShadowCoordinator(RetroSpecCPURiskCoordinator):
    def __init__(self, proposer):
        super().__init__(proposer)
        self.predictions: dict[int, int] = {}
        self.anchor = None
        self.snapshot = None
        self.checked = 0
        self.next_result = 0
        self.sparse_end = 0
        self.limit = 0
        self.end = 0
        self.stopped = False
        self.retired = True
        self.correction: tuple[int, int] | None = None
        # Bounded, opt-in diagnostics. No token tracing on the default hot path.
        self.events: deque[dict] = deque(maxlen=4096)

    def _event(self, kind, **values):
        if self.stats.enabled:
            self.events.append(
                dict(
                    kind=kind,
                    time=perf_counter(),
                    base=self.base,
                    session=self.session,
                    epoch=self.epoch,
                    **values,
                )
            )

    def begin_round(self, request_id, base, seed, budget):
        super().begin_round(request_id, base, seed)
        self.end = min(
            base + budget,
            self.proposer.max_model_len - 1,
            int(self.proposer.index_update_state.next_update_positions[0].item()),
        )
        self.sparse_end = 0
        self.retired = False
        params = self.proposer.runner.requests[request_id].sampling_params
        self.stop_ids = set(params.stop_token_ids or [])
        if not params.ignore_eos:
            self.stop_ids.update(params._all_stop_token_ids)
        self._queue_anchor(0)
        self._event("begin", end=self.end)

    def _queue_anchor(self, index):
        if self.retired or self.alert or self.base + index >= self.end:
            return
        token = self.seed if index == 0 else self.tokens[index - 1]
        self.queue.append(RiskTask(index, token, perf_counter()))

    def _cancel_active(self, completed=False):
        if self.active is None:
            return
        _, session, epoch = self.active
        self.active = None
        if self.worker is not None:
            self.worker.send(("release", session, epoch))
        self._pending_release = (session, epoch)
        if not completed:
            self.stats.add_counter("cpu_shadow_cancelled")

    def _invalidate_predictions(self):
        self._cancel_active()
        self.predictions.clear()
        self.anchor = self.snapshot = None
        self.checked = self.next_result = 0
        self.stopped = False
        self.queue.clear()
        self.correction = None
        self.alert = False

    def _validate_snapshot(self):
        if self.anchor is None:
            return
        invalid = torch.tensor([not self.snapshot.valid()], dtype=torch.int32)
        if self.group.world_size > 1:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.group.cpu_group)
        if invalid.item():
            self.mirror_length = 0
            self._invalidate_predictions()
            self._queue_anchor(self.sparse_end)
            self.stats.add_counter("cpu_shadow_snapshot_invalidated")

    def invalidate_kv(self, start):
        self.mirror_length = min(self.mirror_length, start)
        if not self.retired:
            self._event("sparse_start", start=start)
        if self.preparing is not None and self.preparing.position > start:
            self._invalidate_preparation(start)
        if self.anchor is not None and self.base + self.anchor > start:
            self._invalidate_predictions()
            self.stats.add_counter("cpu_shadow_snapshot_invalidated")

    def update_candidates(self, tokens):
        common = 0
        for old, new in zip(self.tokens, tokens):
            if old != new:
                break
            common += 1
        if self.anchor is not None and common < self.anchor:
            self.mirror_length = min(self.mirror_length, self.base + common)
            self._invalidate_predictions()
        if self.preparing is not None and common < self.preparing.task.index:
            self._invalidate_preparation(self.base + common)
            self.queue.clear()
        if self.queue and common < self.queue[0].index:
            self.queue.clear()
        self.tokens = list(tokens)
        self.sparse_end = len(tokens)
        if self.anchor is not None:
            self.checked = max(self.anchor, min(self.checked, common))
        self._event("sparse_end", count=len(tokens), common=common)
        self._validate_snapshot()
        self._compare()
        if self.alert:
            return
        # A rollback can shrink the lookahead allowance. Retire the old reader
        # before re-anchoring rather than letting it run beyond the new bound.
        if self.active is not None and self.limit > self._allowed_limit():
            self._invalidate_predictions()
            self.stats.add_counter("cpu_shadow_lead_restarts")
        if self.active is None and not self.queue:
            self._queue_anchor(len(tokens))

    def draft(self, index, token, margin, metadata, *, allow_start=True):
        if index != len(self.tokens):
            raise RuntimeError("CPU shadow candidate positions are not contiguous")
        self.tokens.append(token)
        self._event("draft", index=index, token=token, margin=margin)
        return self.pump(metadata, allow_start=allow_start)

    def _compare(self):
        if self.retired or self.alert or self.anchor is None:
            return
        while self.checked < min(self.next_result, len(self.tokens)):
            index = self.checked
            token = self.predictions[index]
            self.stats.add_counter("cpu_shadow_checked")
            if token != self.tokens[index]:
                self.correction = (index, token)
                self.alert = True
                self.stats.add_counter("cpu_shadow_corrections")
                self.stats.add_counter(
                    "cpu_shadow_alert_after_sparse"
                    if index < self.sparse_end
                    else "cpu_shadow_alert_before_sparse"
                )
                self.stats.add_counter(
                    "cpu_shadow_discarded_draft_tokens", len(self.tokens) - index - 1
                )
                self._event(
                    "correction",
                    index=index,
                    token=token,
                    candidate=self.tokens[index],
                    already_sparse=index < self.sparse_end,
                    discarded=len(self.tokens) - index - 1,
                )
                self._cancel_active()
                return
            self.checked += 1
            self.stats.add_counter("cpu_shadow_matches")

    def _allowed_limit(self):
        return min(
            self.end,
            self.base
            + len(self.tokens)
            + self.proposer.speculative_config.retrospec_max_draft_tokens,
        )

    def _extend(self):
        if self.active is None or self.stopped:
            return
        allowed = self._allowed_limit()
        # An extension does not touch inputs/cache or interrupt the current
        # forward. Avoid dispatching control messages while CPU is far behind.
        if self.limit < allowed and self.base + self.next_result + 1 >= self.limit:
            self.limit = allowed
            if self.worker is not None:
                self.worker.send(("extend", self.session, self.epoch, self.limit))
            self.stats.add_counter("cpu_shadow_extensions")
            self._event("extend", limit=self.limit)

    def pump(self, metadata, *, allow_start=True):
        payload = None
        if self.worker is not None:
            try:
                payload = (None, self.worker.ready, self.worker.poll())
            except Exception as exc:
                payload = (str(exc), False, [])
        error, self.ready, messages = self._control(payload)
        if error:
            raise RuntimeError(error)
        for message in messages:
            if message[0] == "released" and message[1:3] == self._pending_release:
                self._pending_release = None
            if message[0] == "interrupted":
                self.stats.add_counter("cpu_shadow_interrupted")
                self.stats.record_cpu_time("cpu_shadow_cancelled_forward", message[4])
            if message[0] != "token":
                continue
            if self.retired or self.active is None or message[1:3] != self.active[1:3]:
                self.stats.add_counter("cpu_shadow_stale_results")
                continue
            index = message[3] - self.base - 1
            if index != self.next_result or message[3] > self.limit:
                raise RuntimeError("CPU shadow worker returned the wrong position")
            self.predictions[index] = message[4]
            self.next_result += 1
            self.stopped = message[4] in self.stop_ids
            self.stats.record_cpu_time("cpu_shadow_forward", message[5])
            self.stats.add_counter("cpu_shadow_predicted")
            self.stats.observe_peak(
                "cpu_shadow_lead_tokens", max(0, self.next_result - len(self.tokens))
            )
            self._event(
                "token",
                index=index,
                token=message[4],
                started=message[6] if len(message) > 6 else None,
                finished=message[7] if len(message) > 7 else None,
            )
            if message[3] == self.limit:
                self.stats.add_counter("cpu_shadow_limit_hits")
        # All ranks share this progress through the worker-message broadcast.
        # With no prediction to compare, there is no snapshot result to consume.
        # Sparse boundaries still validate unconditionally in update_candidates;
        # otherwise fence every newly comparable prediction before using it.
        if self.checked < min(self.next_result, len(self.tokens)):
            self._validate_snapshot()
        self._compare()
        if self.retired:
            # Cancellation fences the GPU sources immediately, but the old
            # background writer must still finish before its buffer is reused.
            if self.preparing is not None:
                self._poll_preparation(False, None)
            return False
        if self.alert:
            return self.alert
        self._extend()
        if self.active is not None or self._pending_release is not None:
            return False
        if self.preparing is not None:
            self._poll_preparation(allow_start, None)
            if self.preparing is not None or self.active is not None:
                return False
        if self.queue and self.ready and allow_start:
            self._start(self.queue[0], metadata)
        return False

    def _start(self, task, metadata):
        self._event("prepare_start", index=task.index, start=self.mirror_length)
        super()._start(task, metadata)

    def _dispatch_preparation(self, preparation):
        task = preparation.task
        self.queue.popleft()
        self.active = (task, preparation.session, preparation.epoch)
        self.anchor = self.checked = self.next_result = task.index
        self.predictions.clear()
        self.snapshot = preparation.kv
        self.stopped = False
        self.limit = self._allowed_limit()
        token = self.seed if task.index == 0 else self.tokens[task.index - 1]
        if self.worker is not None:
            self.worker.send(
                (
                    "predict",
                    self.session,
                    self.epoch,
                    self.storage.paths,
                    self.storage.shape,
                    preparation.position,
                    token,
                    self.limit,
                    tuple(self.stop_ids),
                )
            )
        self.stats.add_counter("cpu_shadow_started")
        self._event("predict_start", index=task.index, limit=self.limit)

    def finish_round(self):
        if not self.retired:
            self._event("finish")
        super().finish_round()
        self.predictions.clear()
        self.anchor = self.snapshot = None
        self.correction = None
        self.alert = False
        self.retired = True
        self.checked = self.next_result = 0
