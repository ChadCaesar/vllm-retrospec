# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import perf_counter

from vllm.distributed import get_tp_group

from .cpu_bridge import RetroSpecCPUKVBridge
from .cpu_worker import RetroSpecCPUStorage, RetroSpecCPUWorker


class RetroSpecCPUCoordinator:
    def __init__(self, proposer):
        self.proposer = proposer
        self.group = get_tp_group()
        model_config = proposer.vllm_config.model_config
        self.config = model_config.hf_config
        self.worker = (
            RetroSpecCPUWorker(
                model_config.model,
                model_config.revision,
                proposer.speculative_config.retrospec_cpu_threads,
                self.config,
            )
            if self.group.rank_in_group == 0
            else None
        )
        self.bridge = RetroSpecCPUKVBridge(proposer)
        self.storage = None
        # One retained mapping, bounded by max_model_len. It is quarantined
        # until the worker acknowledges the matching release session/epoch.
        self._pooled_storage = None
        self._pending_release = None
        self.request_id = None
        self.session = 0
        self.epoch = 0
        self.authoritative_length = 0
        self.initial_length = 0
        self.staged_length = 0
        self.stop_ids: set[int] = set()
        self.committed_tokens: dict[str, list[int]] = {}
        self.deferred_window = None
        self.deferred_plans = None

    def _control(self, value):
        return self.group.broadcast_object(value, src=0)

    def _acquire_storage(self, shape):
        description = None
        if self.group.rank_in_group == 0:
            try:
                if self._pending_release is not None:
                    started = perf_counter()
                    self.worker.wait_for_release(*self._pending_release)
                    self.proposer.performance_stats.record_cpu_time(
                        "cpu_kv_release_wait_wall", perf_counter() - started
                    )
                storage = self._pooled_storage
                reused = storage is not None and storage.can_reuse(shape)
                if not reused:
                    if storage is not None:
                        storage.close()
                        self._pooled_storage = None
                    storage = RetroSpecCPUStorage(shape)
                    self._pooled_storage = storage
                description = (None, storage.directory, storage.shape, reused)
            except Exception as exc:
                description = (str(exc), None, None, False)
        error, directory, actual_shape, reused = self._control(description)
        if error is not None:
            raise RuntimeError(error)
        # No TP rank may overwrite the mapping before this broadcast confirms
        # the CPU fence. Keep the allocated capacity/strides for shorter inputs.
        self._pending_release = None
        storage = self._pooled_storage
        if self.group.rank_in_group != 0 and (
            storage is None or storage.directory != directory
        ):
            if storage is not None:
                storage.close()
            self._pooled_storage = RetroSpecCPUStorage(tuple(actual_shape), directory)
        self.storage = self._pooled_storage
        self._pooled_storage = None
        self.proposer.performance_stats.add_counter(
            "cpu_kv_buffer_reuses" if reused else "cpu_kv_buffer_allocations"
        )
        self.proposer.performance_stats.observe_peak(
            "cpu_kv_buffer_capacity_tokens", self.storage.shape[-2]
        )

    def begin(self, request_id, base_position, seed, budget, metadata, sampling_params):
        initial = (
            self.storage is None
            or self.request_id != request_id
            or base_position < self.authoritative_length
        )
        if initial:
            self.release()
            self.session += 1
            self.request_id = request_id
            self.initial_length = base_position
            # The mapping belongs to the whole request, while budget is only
            # the bounded lookahead of this proposal round.
            request = self.proposer.runner.requests[request_id]
            capacity = min(
                self.proposer.max_model_len,
                request.num_prompt_tokens + sampling_params.max_tokens,
            )
            shape = (
                2,
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
                capacity,
                getattr(self.config, "head_dim", None)
                or self.config.hidden_size // self.config.num_attention_heads,
            )
            self._acquire_storage(shape)
            self.stop_ids = set(sampling_params.stop_token_ids or [])
            if not sampling_params.ignore_eos:
                self.stop_ids.update(sampling_params._all_stop_token_ids)
        self.epoch += 1
        self.bridge.export_gpu(
            self.storage,
            metadata,
            request_id,
            0 if initial else self.authoritative_length,
            base_position,
            initial,
        )
        self.group.barrier()
        self.authoritative_length = self.staged_length = base_position
        self.committed_tokens[request_id] = []
        if self.worker is not None:
            if initial:
                command = (
                    "initialize",
                    self.session,
                    self.epoch,
                    self.storage.paths,
                    self.storage.shape,
                    base_position,
                    seed,
                    base_position + budget,
                    tuple(self.stop_ids),
                )
            else:
                command = (
                    "resume",
                    self.session,
                    self.epoch,
                    self._resume_start,
                    base_position,
                    seed,
                    base_position + budget,
                )
            self.worker.send(command)
        self._resume_start = base_position

    def poll(self, window):
        payload = None
        if self.worker is not None:
            try:
                payload = (None, self.worker.poll())
            except Exception as exc:
                payload = (str(exc), [])
        error, messages = self._control(payload)
        if error:
            raise RuntimeError(error)
        changed = False
        for message in messages:
            if message[0] == "interrupted":
                self.proposer.performance_stats.add_counter("cpu_cancelled_forwards")
            if message[0] != "token":
                continue
            session, epoch, position, token = message[1:5]
            if len(message) > 5:
                self.proposer.performance_stats.record_cpu_time(
                    "cpu_full_forward", message[5]
                )
            if session != self.session or epoch != self.epoch:
                self.proposer.performance_stats.add_counter("cpu_stale_results")
                continue
            if position > window.base_position + self.proposer.num_speculative_tokens:
                continue
            offset = position - window.base_position - 1
            mismatch = offset < window.draft_end and window.tokens[offset] != token
            if mismatch:
                self.proposer.performance_stats.add_counter(
                    "cpu_discarded_draft_tokens", window.draft_end - offset
                )
            corrected = window.confirm(position, token)
            changed |= corrected
            self.authoritative_length = window.base_position + window.confirmed
            self._resume_start = self.authoritative_length
            self.proposer.performance_stats.add_counter("cpu_confirmed_tokens")
            if mismatch:
                self.proposer.performance_stats.add_counter("cpu_corrections")
            elif corrected:
                self.proposer.performance_stats.add_counter("cpu_ahead_tokens")
        return changed

    def synchronize(self, metadata, background=False):
        if self.authoritative_length > self.staged_length:
            self.bridge.stage_cpu(
                self.storage,
                metadata,
                self.staged_length,
                self.authoritative_length,
                background,
            )
            self.staged_length = self.authoritative_length
        self.bridge.publish(wait=not background)

    def freeze(self, window, metadata):
        # Do not wait for an in-flight CPU forward; its old-epoch result is stale.
        self.epoch += 1
        if self.worker is not None:
            self.worker.send(("pause", self.epoch))
        self.synchronize(metadata, background=False)
        self.committed_tokens[self.request_id] = window.tokens[: window.confirmed]
        self._resume_start = self.authoritative_length

    def release(self):
        self.deferred_window = None
        self.deferred_plans = None
        if self.storage is not None:
            self.epoch += 1
            if self.worker is not None:
                self.worker.send(("release", self.session, self.epoch))
            self._pending_release = (self.session, self.epoch)
            # Drain GPU publications before retaining this rank's mapping.
            # CPU acknowledgement can overlap the next request's prefill.
            self.bridge.close()
            self._pooled_storage = self.storage
            self.storage = None
        self.request_id = None

    def close(self):
        try:
            if self.worker is not None:
                self.worker.request_close()
        finally:
            try:
                self.bridge.close()
            finally:
                try:
                    # Unlink before waiting for model teardown: the executor
                    # may terminate this GPU process after its shutdown grace.
                    # Existing CPU mappings stay valid until the child exits;
                    # shutdown never reuses or truncates their backing files.
                    for storage in (self.storage, self._pooled_storage):
                        if storage is not None:
                            storage.close()
                finally:
                    self.storage = self._pooled_storage = None
                    self._pending_release = None
                    self.request_id = None
                    self.deferred_window = self.deferred_plans = None
                    if self.worker is not None:
                        self.worker.close()
