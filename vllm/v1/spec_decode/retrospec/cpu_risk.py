# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FIFO, advisory-only CPU checks over GPU-origin KV snapshots."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.distributed as dist

from .cpu_coordinator import RetroSpecCPUCoordinator


@dataclass(frozen=True)
class RiskTask:
    index: int
    token: int
    queued_at: float


@dataclass
class _RiskPreparation:
    task: RiskTask
    session: int
    epoch: int
    start: int
    position: int
    initial: bool
    started: float
    kv: object
    valid_length: int
    obsolete: bool = False
    completed: bool = False


class RetroSpecCPURiskCoordinator(RetroSpecCPUCoordinator):
    def __init__(self, proposer):
        super().__init__(proposer)
        self.queue: deque[RiskTask] = deque()
        self.active = None
        self.base = 0
        self.seed = 0
        self.tokens: list[int] = []
        self.mirror_length = 0
        self.alert = False
        self.ready = False
        self.preparing = None
        self._mirror_dirty = None

    @property
    def stats(self):
        return self.proposer.performance_stats

    def _cancel_active(self, completed=False):
        if self.active is None:
            return
        self.active = None
        if self.worker is not None:
            self.worker.send(("release", self.session, self.epoch))
        self._pending_release = (self.session, self.epoch)
        if not completed:
            self.stats.add_counter("cpu_risk_cancelled")

    def begin_round(self, request_id, base, seed):
        if request_id != self.request_id or base < self.base:
            # A rewound prefix may cut through a per-head packed segment.
            # Truncating its row count cannot recover the logical prefix.
            self.release()
            self.request_id = request_id
            self.session += 1
        self.finish_round()
        self.base, self.seed = base, seed
        self.mirror_length = min(self.mirror_length, base)
        self.tokens = []
        self.alert = False

    def invalidate_kv(self, start):
        self.mirror_length = min(self.mirror_length, start)
        self._mirror_dirty = (
            start if self._mirror_dirty is None else min(self._mirror_dirty, start)
        )
        if self.preparing is not None:
            self.preparing.kv.fence_sources()

    def update_candidates(self, tokens):
        common = 0
        for old, new in zip(self.tokens, tokens):
            if old != new:
                break
            common += 1
        if common < len(self.tokens):
            before = len(self.queue)
            self.queue = deque(task for task in self.queue if task.index < common)
            self.stats.add_counter(
                "cpu_risk_prefix_invalidated", before - len(self.queue)
            )
            if self.active is not None and self.active[0].index >= common:
                self._cancel_active()
            if self.preparing is not None and self.preparing.task.index >= common:
                self._invalidate_preparation(self.base + common)
        self.tokens = list(tokens)

    def draft(
        self,
        index,
        token,
        margin,
        metadata,
        *,
        allow_start=True,
        kv_rewrite_start=None,
    ):
        if index != len(self.tokens):
            raise RuntimeError("CPU risk candidate positions are not contiguous")
        self.tokens.append(token)
        if (
            margin
            < self.proposer.speculative_config.retrospec_cpu_risk_margin_threshold
        ):
            self.queue.append(RiskTask(index, token, perf_counter()))
            self.stats.add_counter("cpu_risk_enqueued")
            self.stats.observe_peak("cpu_risk_queue_depth", len(self.queue))
        return self.pump(
            metadata, allow_start=allow_start, kv_rewrite_start=kv_rewrite_start
        )

    def pump(self, metadata, *, allow_start=True, kv_rewrite_start=None):
        payload = None
        if self.worker is not None:
            try:
                messages = self.worker.poll()
                payload = (None, self.worker.ready, messages)
            except Exception as exc:
                payload = (str(exc), False, [])
        error, self.ready, messages = self._control(payload)
        if error:
            raise RuntimeError(error)
        for message in messages:
            if message[0] == "interrupted":
                self.stats.add_counter("cpu_risk_interrupted")
                self.stats.record_cpu_time("cpu_risk_cancelled_forward", message[4])
            if message[0] == "released" and message[1:3] == self._pending_release:
                self._pending_release = None
            if message[0] != "token":
                continue
            if self.active is None or message[1:3] != self.active[1:3]:
                self.stats.add_counter("cpu_risk_stale_results")
                continue
            task = self.active[0]
            if message[3] != self.base + task.index + 1:
                raise RuntimeError("CPU risk worker returned the wrong position")
            if not self.queue or self.queue[0] is not task:
                raise RuntimeError("CPU risk FIFO ownership changed")
            self.queue.popleft()
            self.stats.record_cpu_time("cpu_risk_forward", message[5])
            self.stats.record_cpu_time(
                "cpu_risk_task_wall", perf_counter() - task.queued_at
            )
            self.stats.add_counter("cpu_risk_checked")
            if message[4] != task.token:
                self.alert = True
                self.stats.add_counter("cpu_risk_disagreements")
            else:
                self.stats.add_counter("cpu_risk_matches")
            self._cancel_active(completed=True)
        if self.alert or self.active is not None or self._pending_release is not None:
            return self.alert
        if self.preparing is not None:
            self._poll_preparation(allow_start, kv_rewrite_start)
            if self.preparing is not None or self.active is not None:
                return self.alert
        if self.queue and self.ready:
            task = self.queue[0]
            if not allow_start:
                self.stats.add_counter("cpu_risk_dispatch_deferred_boundary")
            else:
                self._start(task, metadata)
        elif self.queue and not self.ready:
            self.stats.add_counter("cpu_risk_not_ready_polls")
        return self.alert

    def _start(self, task, metadata):
        position = self.base + task.index
        started = perf_counter()
        initial = self.storage is None
        if self.storage is None:
            request = self.proposer.runner.requests[self.request_id]
            capacity = min(
                self.proposer.max_model_len,
                request.num_prompt_tokens + request.sampling_params.max_tokens,
            )
            # Small headroom avoids remapping a long prefix for a one-token
            # difference between successive requests, while retaining a bound.
            capacity = min(self.proposer.max_model_len, (capacity + 255) // 256 * 256)
            shape = (
                2,
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
                capacity,
                getattr(self.config, "head_dim", None)
                or self.config.hidden_size // self.config.num_attention_heads,
            )
            self._acquire_storage(shape)
            self.mirror_length = 0
        self.stats.record_cpu_time(
            "cpu_risk_queue_wait", perf_counter() - task.queued_at
        )
        start = min(self.mirror_length, position)
        self._mirror_dirty = None
        self.epoch += 1
        preparation = None
        error = None
        try:
            preparation = self.bridge.prepare_risk(
                self.storage, metadata, self.request_id, start, position
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            self._preparation_error(error)
        except BaseException:
            if preparation is not None:
                preparation.cancel()
            raise
        self.preparing = _RiskPreparation(
            task,
            self.session,
            self.epoch,
            start,
            position,
            initial,
            started,
            preparation,
            position,
        )
        self.stats.record_cpu_time(
            "cpu_risk_prepare_submit_wall",
            perf_counter() - started,
        )
        self.stats.add_counter("cpu_risk_prepare_submitted")

    def _preparation_error(self, error):
        """Coordinate local capture failures; no collectives run in workers."""
        failed = torch.tensor([int(error is not None)], dtype=torch.int32)
        if self.group.world_size > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.group.cpu_group)
        if failed.item():
            errors = [error]
            if self.group.world_size > 1:
                errors = [None] * self.group.world_size
                dist.all_gather_object(errors, error, group=self.group.cpu_group)
            raise RuntimeError(
                "CPU KV preparation failed: " + next(e for e in errors if e)
            )

    def _cancel_preparation(self):
        preparation = self.preparing
        if preparation is not None and not preparation.kv.cancelled.is_set():
            preparation.kv.cancel()
            self.stats.add_counter("cpu_risk_prepare_cancelled")

    def _invalidate_preparation(self, start):
        preparation = self.preparing
        preparation.valid_length = min(preparation.valid_length, start)
        preparation.obsolete = True
        preparation.kv.fence_sources()
        # Finish the stable historical prefix rather than repeatedly restart a
        # long initial gather at every sparse boundary. No CPU reader or new
        # writer is admitted until this future ends; only the tail is retried.
        self.stats.add_counter("cpu_risk_prepare_tail_invalidated")

    def _poll_preparation(self, allow_start, kv_rewrite_start):
        preparation = self.preparing
        kv = preparation.kv
        done = kv.future.done()
        error = None
        if done:
            try:
                kv.future.result()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        snapshot_valid = kv.valid()
        if not snapshot_valid:
            self.mirror_length = 0
        valid = (
            not kv.cancelled.is_set()
            and preparation.session == self.session
            and preparation.epoch == self.epoch
            and snapshot_valid
        )
        obsolete = (
            preparation.obsolete
            or not self.queue
            or self.queue[0] is not preparation.task
        )
        # Poll all ranks, rather than wait at an export barrier. A ready rank
        # cannot expose the shared buffer until every writer has completed.
        status = torch.tensor(
            [
                not done,
                error is not None,
                not valid,
                obsolete,
                not snapshot_valid,
                -preparation.valid_length,
            ],
            dtype=torch.int64,
        )
        with self.stats.cpu_timer("cpu_risk_prepare_poll_wall"):
            if self.group.world_size > 1:
                dist.all_reduce(
                    status, op=dist.ReduceOp.MAX, group=self.group.cpu_group
                )
        pending, failed, invalid, obsolete, version_changed, negative_length = (
            status.tolist()
        )
        if version_changed:
            self.mirror_length = 0
        if failed:
            self._cancel_preparation()
            self._preparation_error(error)
        if invalid:
            self._cancel_preparation()
        if pending:
            return
        if obsolete:
            # All ranks finished writing a snapshot whose immutable prefix is
            # still useful. Refresh only the invalidated suffix on the retry.
            self.mirror_length = 0 if invalid else -negative_length
            if self._mirror_dirty is not None:
                self.mirror_length = min(self.mirror_length, self._mirror_dirty)
            if not preparation.completed:
                self.stats.add_counter(
                    "cpu_risk_kv_refresh_tokens",
                    max(0, self.mirror_length - preparation.start),
                )
                self.stats.record_cpu_time(
                    "cpu_risk_initial_prepare"
                    if preparation.initial
                    else "cpu_risk_delta_prepare",
                    perf_counter() - preparation.started,
                )
            self.preparing = None
            self.stats.add_counter("cpu_risk_prepare_prefix_reused")
            return
        if invalid:
            self.preparing = None
            self.stats.add_counter("cpu_risk_prepare_discarded")
            return
        if not preparation.completed:
            preparation.completed = True
            self.mirror_length = min(
                preparation.position,
                self._mirror_dirty
                if self._mirror_dirty is not None
                else preparation.position,
            )
            self.stats.add_counter("cpu_risk_prepare_ready")
            self.stats.add_counter(
                "cpu_risk_kv_refresh_tokens", preparation.position - preparation.start
            )
            self.stats.record_cpu_time(
                "cpu_risk_initial_prepare"
                if preparation.initial
                else "cpu_risk_delta_prepare",
                perf_counter() - preparation.started,
            )
        if not allow_start:
            return
        self.preparing = None
        self._dispatch_preparation(preparation)

    def _dispatch_preparation(self, preparation):
        # The worker writes the input row at position. It is outside this
        # exclusive valid prefix and will be restored before a later query.
        task = preparation.task
        position = preparation.position
        self.active = (task, preparation.session, preparation.epoch)
        token = self.seed if task.index == 0 else self.tokens[task.index - 1]
        if self.worker is not None:
            self.worker.send(
                (
                    "check",
                    self.session,
                    self.epoch,
                    self.storage.paths,
                    self.storage.shape,
                    position,
                    token,
                )
            )
        self.stats.add_counter("cpu_risk_started")

    def finish_round(self):
        self.stats.add_counter("cpu_risk_full_covered", len(self.queue))
        self.queue.clear()
        self._cancel_active()
        self._cancel_preparation()

    def close(self):
        self._cancel_preparation()
        try:
            super().close()
        finally:
            self.preparing = None

    def release(self):
        self.finish_round()
        if self.storage is not None:
            self._pooled_storage = self.storage
            self.storage = None
        self.request_id = None
        self.mirror_length = 0
        self.tokens = []
        self.alert = False
