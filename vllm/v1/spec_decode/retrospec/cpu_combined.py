# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One CPU target: continuous authority, preempted by FIFO risk checks."""

from collections import deque
from time import perf_counter

import torch
import torch.distributed as dist

from .cpu_coordinator import RetroSpecCPUCoordinator
from .cpu_risk import RiskTask
from .cpu_worker import RetroSpecCPUStorage


class RetroSpecCPUCombinedCoordinator(RetroSpecCPUCoordinator):
    def __init__(self, proposer):
        super().__init__(proposer)
        self.queue = deque()
        self.candidates = {}
        self.active = self.preparing = None
        self.risk_tail = self.continuous_tail = None
        self.mode = "continuous"
        self.alert = False
        self._continuous_epoch = 0
        self._pause_epoch = None
        self._tail_installed = False
        self._suspend_nonce = 0
        self.common_end = 0
        self._deferred_messages = []
        self._restore_pending = None

    @property
    def stats(self):
        return self.proposer.performance_stats

    def _allocate_tails(self):
        shape = (
            *self.storage.shape[:-2],
            self.proposer.num_speculative_tokens + 1,
            self.storage.shape[-1],
        )
        for name in ("risk_tail", "continuous_tail"):
            tail = getattr(self, name)
            description = None
            if self.group.rank_in_group == 0:
                try:
                    if tail is None or not tail.can_reuse(shape):
                        if tail is not None:
                            tail.close()
                        tail = RetroSpecCPUStorage(shape, journal_tokens=0)
                        self.bridge.export_placement.bind_memory(tail.kv)
                    description = (None, tail.directory, tail.shape)
                except Exception as exc:
                    description = (str(exc), None, None)
            error, directory, actual_shape = self._control(description)
            if error:
                raise RuntimeError(error)
            if self.group.rank_in_group != 0 and (
                tail is None or tail.directory != directory
            ):
                if tail is not None:
                    tail.close()
                tail = RetroSpecCPUStorage(tuple(actual_shape), directory)
            setattr(self, name, tail)
            tail.offset = self.common_end

    def begin(self, request_id, base_position, seed, budget, metadata, sampling_params):
        self._drain_preparation()
        super().begin(
            request_id, base_position, seed, budget, metadata, sampling_params
        )
        if self._pause_fenced:
            # The pause follows restore in the worker's command FIFO. Its ACK
            # fences both the previous risk-tail reader and the journal import.
            self._restore_pending = None
        self._continuous_epoch = self.epoch
        self.mode = "continuous"
        self._tail_installed = False
        self._pause_epoch = None
        self.common_end = base_position
        self._allocate_tails()
        self.alert = False
        self._drop_covered(base_position)

    def _cancel_active(self):
        if self.active is None:
            return
        self.active = None
        self.epoch += 1
        self._pause_epoch = self.epoch
        if self.worker is not None:
            self.worker.send(("pause", self.epoch))
        self.stats.add_counter("cpu_risk_cancelled")

    def _cancel_preparation(self):
        if self.preparing is not None:
            self.preparing[1].cancel()

    def _drop_covered(self, end):
        before = len(self.queue)
        self.queue = deque(task for task in self.queue if task.index >= end)
        self.stats.add_counter("cpu_risk_cpu_covered", before - len(self.queue))
        if self.active is not None and self.active[0].index < end:
            self._cancel_active()
        if self.preparing is not None and self.preparing[0].index < end:
            self._cancel_preparation()
        self.candidates = {
            p: t for p, t in self.candidates.items() if p >= self.common_end
        }

    def update_candidates(self, window):
        current = {
            window.base_position + i: token for i, token in enumerate(window.tokens)
        }
        changed = [p for p, t in self.candidates.items() if current.get(p) != t]
        if changed:
            first = min(changed)
            before = len(self.queue)
            self.queue = deque(task for task in self.queue if task.index < first)
            self.stats.add_counter(
                "cpu_risk_prefix_invalidated", before - len(self.queue)
            )
            if self.active is not None and self.active[0].index >= first:
                self._cancel_active()
            if self.preparing is not None and self.preparing[0].index >= first:
                self._cancel_preparation()
        self.candidates = current
        self._drop_covered(window.base_position + window.confirmed)

    def draft(self, window, margin):
        self.update_candidates(window)
        if (
            margin
            < self.proposer.speculative_config.retrospec_cpu_risk_margin_threshold
        ):
            position = window.base_position + window.draft_end - 1
            if position >= self.authoritative_length:
                self.queue.append(RiskTask(position, window.tokens[-1], perf_counter()))
                self.stats.add_counter("cpu_risk_enqueued")
                self.stats.observe_peak("cpu_risk_queue_depth", len(self.queue))

    def invalidate_kv(self, start):
        # Existing captures are immutable. Future tasks capture their entire
        # bounded tail afresh, so they cannot inherit an indefinitely stale mirror.
        if self.preparing is not None:
            self.preparing[1].fence_sources()

    def poll(self, window, *, accept_risk=True, read_worker=True):
        messages = self._deferred_messages + (
            self._poll_messages() if read_worker else []
        )
        self._deferred_messages = []
        for message in messages:
            if message[0] == "suspended" and message[3] == self._suspend_nonce:
                if self.mode == "suspending":
                    self.mode = "risk"
                    self.stats.add_counter("cpu_continuous_suspensions")
                    if len(message) > 5:
                        self.stats.record_cpu_time("cpu_tail_save_wall", message[5])
            elif message[0] == "restored" and message[1:3] == self._restore_pending:
                self._restore_pending = None
                if self.mode == "restoring":
                    self.mode = "continuous"
                    self._tail_installed = False
                    self.stats.add_counter("cpu_continuous_restores")
                    if len(message) > 3:
                        self.stats.record_cpu_time("cpu_tail_restore_wall", message[3])
            elif message[0] == "paused" and message[2] == self._pause_epoch:
                self._pause_epoch = None
        continuous = [
            m
            for m in messages
            if m[0] in ("token", "interrupted")
            and m[1:3] == (self.session, self._continuous_epoch)
        ]
        before = window.version
        changed = self._consume_continuous(window, continuous, self._continuous_epoch)
        self.update_candidates(window)
        for message in messages:
            if message[0] == "interrupted" and message not in continuous:
                self.stats.add_counter("cpu_risk_interrupted")
                self.stats.record_cpu_time("cpu_risk_cancelled_forward", message[4])
            if message[0] == "token" and message not in continuous:
                if not accept_risk:
                    self._deferred_messages.append(message)
                    continue
                if self.active is None or message[1:3] != (
                    self.session,
                    self.active[1],
                ):
                    self.stats.add_counter("cpu_risk_stale_results")
                    continue
                task = self.active[0]
                if message[3] != task.index + 1:
                    raise RuntimeError("CPU risk worker returned the wrong position")
                if not self.queue or self.queue[0] is not task:
                    raise RuntimeError("CPU risk FIFO ownership changed")
                self.queue.popleft()
                self.active = None
                self.stats.add_counter("cpu_risk_checked")
                self.stats.record_cpu_time("cpu_risk_forward", message[5])
                self.stats.record_cpu_time(
                    "cpu_risk_task_wall", perf_counter() - task.queued_at
                )
                if message[4] != task.token:
                    self.alert = True
                    self.stats.add_counter("cpu_risk_disagreements")
                else:
                    self.stats.add_counter("cpu_risk_matches")
        if before != window.version:
            self.stats.add_counter("cpu_combined_candidate_changes")
        return changed

    def pump(self, metadata, *, allow_start=True):
        self._metadata = metadata
        if (
            self.alert
            or self.active is not None
            or self._pause_epoch is not None
            or self._restore_pending is not None
        ):
            return self.alert
        if self.preparing is not None:
            task, kv, started = self.preparing
            error = None
            done = kv.future.done()
            if done:
                try:
                    kv.future.result()
                except Exception as exc:
                    error = str(exc)
            status = torch.tensor([not done, error is not None], dtype=torch.int32)
            if self.group.world_size > 1:
                dist.all_reduce(
                    status, op=dist.ReduceOp.MAX, group=self.group.cpu_group
                )
            if status[1].item():
                errors = [None] * self.group.world_size
                if self.group.world_size > 1:
                    dist.all_gather_object(errors, error, group=self.group.cpu_group)
                else:
                    errors[0] = error
                raise RuntimeError(
                    "CPU risk preparation failed: " + next(e for e in errors if e)
                )
            if status[0].item():
                return self.alert
            if kv.cancelled.is_set() or not self.queue or self.queue[0] is not task:
                self.preparing = None
            elif allow_start:
                if self.mode == "continuous":
                    self.epoch += 1
                    self._suspend_nonce = self.epoch
                    if self.worker is not None:
                        self.worker.send(
                            (
                                "suspend",
                                self.epoch,
                                self.continuous_tail.paths[0],
                                self.continuous_tail.shape,
                                self.common_end,
                            )
                        )
                    self.mode = "suspending"
                elif self.mode == "risk":
                    self.epoch += 1
                    self.active = (task, self.epoch)
                    self._tail_installed = True
                    token = (
                        self._seed
                        if task.index == self._base
                        else self.candidates[task.index - 1]
                    )
                    if self.worker is not None:
                        self.worker.send(
                            (
                                "check_shared",
                                self.session,
                                self.epoch,
                                self.risk_tail.paths[0],
                                self.risk_tail.shape,
                                self.common_end,
                                task.index,
                                token,
                            )
                        )
                    self.preparing = None
                    self.stats.add_counter("cpu_risk_started")
                    self.stats.record_cpu_time(
                        "cpu_risk_prepare_wall", perf_counter() - started
                    )
        if (
            self.preparing is None
            and self.active is None
            and self.queue
            and allow_start
        ):
            task = self.queue[0]
            started = perf_counter()
            error = None
            kv = None
            try:
                kv = self.bridge.prepare_risk(
                    self.risk_tail,
                    metadata,
                    self.request_id,
                    self.common_end,
                    task.index,
                    native_only=True,
                )
            except Exception as exc:
                error = str(exc)
            failed = torch.tensor([error is not None], dtype=torch.int32)
            if self.group.world_size > 1:
                dist.all_reduce(
                    failed, op=dist.ReduceOp.MAX, group=self.group.cpu_group
                )
            if failed.item():
                errors = [None] * self.group.world_size
                if self.group.world_size > 1:
                    dist.all_gather_object(errors, error, group=self.group.cpu_group)
                else:
                    errors[0] = error
                if kv is not None:
                    kv.cancel()
                    kv.future.result()
                raise RuntimeError(
                    "CPU risk preparation failed: " + next(e for e in errors if e)
                )
            self.preparing = (task, kv, started)
            self.stats.add_counter("cpu_risk_prepare_submitted")
        if (
            not self.queue
            and self.preparing is None
            and self.mode == "risk"
            and self._pause_epoch is None
        ):
            self._restore(True)
        return self.alert

    def start_window(self, window, seed, metadata):
        self._base, self._seed, self._metadata = window.base_position, seed, metadata
        self.update_candidates(window)

    def _restore(self, run):
        if self.mode in ("risk", "suspending"):
            self._restore_pending = (self.session, self._continuous_epoch)
            if self.worker is not None:
                self.worker.send(("restore", run))
            self.mode = "restoring"

    def synchronize(self, metadata, background=False):
        self.invalidate_kv(self.staged_length)
        storage = self.storage
        if self._tail_installed:
            storage = self.continuous_tail
        if self.authoritative_length > self.staged_length:
            self.bridge.stage_cpu(
                storage,
                metadata,
                self.staged_length,
                self.authoritative_length,
                background,
            )
            self.staged_length = self.authoritative_length
        self.bridge.publish(wait=not background)
        if not background:
            self.needs_publication = False

    def freeze(self, window, metadata, *, carry=False):
        self._cancel_preparation()
        self.active = None
        if not carry:
            self.stats.add_counter("cpu_risk_full_covered", len(self.queue))
            self.queue.clear()
            self.candidates.clear()
        self._restore(False)
        super().freeze(window, metadata)
        self._pause_epoch = None

    def _drain_preparation(self):
        if self.preparing is not None:
            kv = self.preparing[1]
            kv.cancel()
            kv.future.result()
            self.preparing = None
            # All TP writers must stop before a shared tail can be reused.
            self.group.barrier()

    def release(self):
        # Base.begin() can call this before the first combined window exists.
        if hasattr(self, "queue"):
            self.queue.clear()
            self.candidates.clear()
            self._cancel_preparation()
            self._drain_preparation()
            self.active = None
            # Acquiring the released full workspace fences the worker before
            # either auxiliary tail can be admitted on the next request.
            self._restore_pending = None
        super().release()

    def close(self):
        self._cancel_preparation()
        try:
            super().close()
        finally:
            for name in ("risk_tail", "continuous_tail"):
                tail = getattr(self, name, None)
                if tail is not None:
                    tail.close()
                    setattr(self, name, None)
