# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FIFO, advisory-only CPU checks over GPU-origin KV snapshots."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

from .cpu_coordinator import RetroSpecCPUCoordinator


@dataclass(frozen=True)
class RiskTask:
    index: int
    token: int
    queued_at: float


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
        if self.active is not None and self.base + self.active[0].index > start:
            # Retain the head task for retry if its candidate prefix survives.
            self._cancel_active()
            self.stats.add_counter("cpu_risk_retries")

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
        self.tokens = list(tokens)

    def draft(self, index, token, margin, metadata):
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
        return self.pump(metadata)

    def pump(self, metadata):
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
        if self.queue and self.ready:
            self._start(self.queue[0], metadata)
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
        with self.stats.cpu_timer("cpu_risk_kv_export"):
            self.bridge.export_risk(
                self.storage, metadata, self.request_id, start, position
            )
            self.group.barrier()
        self.stats.record_cpu_time(
            "cpu_risk_initial_prepare" if initial else "cpu_risk_delta_prepare",
            perf_counter() - started,
        )
        self.stats.add_counter("cpu_risk_kv_refresh_tokens", position - start)
        # The worker writes the input row at position. It is outside this
        # exclusive valid prefix and will be restored before a later query.
        self.mirror_length = position
        self.epoch += 1
        self.active = (task, self.session, self.epoch)
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

    def release(self):
        self.finish_round()
        if self.storage is not None:
            self._pooled_storage = self.storage
            self.storage = None
        self.request_id = None
        self.mirror_length = 0
        self.tokens = []
        self.alert = False
