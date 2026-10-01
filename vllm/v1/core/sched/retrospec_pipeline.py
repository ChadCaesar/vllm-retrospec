# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterable

from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import RetroSpecLayerMajorPrefillCompletion
from vllm.v1.request import RequestStatus


class RetroSpecSchedulerPipelineMixin:
    def _get_retrospec_pp_request_budget(self) -> int | None:
        if not self.enable_retrospec_pp_batching:
            return None

        free_pipeline_slots = self.retrospec_pp_depth - len(
            self._retrospec_in_flight_batches
        )
        if free_pipeline_slots <= 0:
            return 0

        eligible_running = sum(
            request.request_id not in self._retrospec_in_flight_req_ids
            for request in self.running
        )
        available_running_slots = max(0, self.max_num_running_reqs - len(self.running))
        eligible_waiting = min(len(self.waiting), available_running_slots)
        eligible_requests = eligible_running + eligible_waiting
        if eligible_requests == 0:
            return 0

        return cdiv(eligible_requests, free_pipeline_slots)

    @staticmethod
    def _retrospec_pp_request_budget_reached(
        request_budget: int | None,
        num_scheduled_requests: int,
    ) -> bool:
        return request_budget is not None and num_scheduled_requests >= request_budget

    def _register_retrospec_pp_batch(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if (
            not self.enable_retrospec_pp_batching
            or not scheduler_output.num_scheduled_tokens
        ):
            return

        request_ids = frozenset(scheduler_output.num_scheduled_tokens)
        overlap = request_ids & self._retrospec_in_flight_req_ids
        if overlap:
            raise RuntimeError(
                "RetroSpec PP scheduled requests with unresolved dependencies: "
                f"{sorted(overlap)}"
            )
        if len(self._retrospec_in_flight_batches) >= self.retrospec_pp_depth:
            raise RuntimeError("RetroSpec PP in-flight batch capacity exceeded")

        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is not None:
            if request_ids != frozenset((descriptor.request_id,)):
                raise RuntimeError(
                    "RetroSpec layer-major prefill must own an exclusive PP batch"
                )
            if self._retrospec_layer_major_prefill_in_flight_req_id is not None:
                raise RuntimeError(
                    "A RetroSpec layer-major prefill is already in flight"
                )

        batch_id = self._retrospec_next_pp_batch_id
        self._retrospec_next_pp_batch_id += 1
        scheduler_output.retrospec_pp_batch_id = batch_id
        self._retrospec_in_flight_batches.append((batch_id, request_ids))
        self._retrospec_in_flight_req_ids.update(request_ids)

        if descriptor is not None:
            self._retrospec_layer_major_prefill_in_flight_req_id = descriptor.request_id

    def _complete_retrospec_pp_batch(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        batch_id = scheduler_output.retrospec_pp_batch_id
        if not self.enable_retrospec_pp_batching:
            if batch_id is not None:
                raise RuntimeError(
                    "Received a RetroSpec PP batch ID while PP batching is disabled"
                )
            return

        if batch_id is None:
            if scheduler_output.num_scheduled_tokens:
                raise RuntimeError("Non-empty RetroSpec PP output has no batch ID")
            return

        if not self._retrospec_in_flight_batches:
            raise RuntimeError("RetroSpec PP completed an unknown batch")

        expected_batch_id, expected_request_ids = self._retrospec_in_flight_batches[0]
        request_ids = frozenset(scheduler_output.num_scheduled_tokens)
        if batch_id != expected_batch_id:
            raise RuntimeError(
                "RetroSpec PP batches completed out of order: "
                f"expected={expected_batch_id}, got={batch_id}"
            )
        if request_ids != expected_request_ids:
            raise RuntimeError("RetroSpec PP batch request set changed while in flight")

        self._retrospec_in_flight_batches.popleft()
        self._retrospec_in_flight_req_ids.difference_update(request_ids)

        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is not None:
            if (
                self._retrospec_layer_major_prefill_in_flight_req_id
                != descriptor.request_id
            ):
                raise RuntimeError(
                    "RetroSpec layer-major prefill completion is not active"
                )
            self._retrospec_layer_major_prefill_in_flight_req_id = None

        deferred_by_status: dict[RequestStatus, list[str]] = defaultdict(list)
        for request_id in request_ids:
            status = self._retrospec_deferred_finish_status.pop(request_id, None)
            if status is not None:
                deferred_by_status[status].append(request_id)

        for status, deferred_request_ids in deferred_by_status.items():
            self._finish_requests_now(deferred_request_ids, status)

    def _get_retrospec_generation_token_budgets(
        self, scheduled_req_ids: Iterable[str]
    ) -> dict[str, int] | None:
        if not getattr(self, "is_retrospec", False):
            return None

        budgets: dict[str, int] = {}
        for req_id in scheduled_req_ids:
            request = self.requests[req_id]
            if request.sampling_params is None:
                continue
            budgets[req_id] = max(
                request.max_tokens - request.num_output_tokens,
                0,
            )
        return budgets

    def _update_retrospec_layer_major_prefill_completion(
        self,
        scheduler_output: SchedulerOutput,
        completion: RetroSpecLayerMajorPrefillCompletion | None,
    ) -> None:
        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is None:
            if completion is not None:
                raise RuntimeError(
                    "Received a layer-major prefill completion without a descriptor"
                )
            return

        if completion is None:
            raise RuntimeError(
                "RetroSpec layer-major prefill did not return a completion"
            )

        if completion.request_id != descriptor.request_id:
            raise RuntimeError(
                "RetroSpec layer-major prefill completion request ID mismatch: "
                f"expected={descriptor.request_id}, got={completion.request_id}"
            )

        completed = completion.num_completed_prompt_tokens
        if completed != descriptor.prompt_num_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill did not complete its atomic "
                f"prompt transaction: expected={descriptor.prompt_num_tokens}, "
                f"completed={completed}"
            )

        request = self.requests.get(descriptor.request_id)
        if request is None or request.is_finished():
            return

        # _update_after_schedule() already advanced this request optimistically
        # to scheduled_end. Replace that value with the worker's actual result.
        request.num_computed_tokens = completed
        request.is_prefill_chunk = completed < (
            request.num_tokens + request.num_output_placeholders
        )
