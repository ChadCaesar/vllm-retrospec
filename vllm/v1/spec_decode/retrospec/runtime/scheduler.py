# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterable

from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import RetroSpecLayerMajorPrefillCompletion
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.retrospec.capacity import (
    RetroSpecGPUIndexFootprint,
    estimate_retrospec_gpu_index_arena_bytes,
    estimate_retrospec_gpu_index_footprint,
)


class RetroSpecSchedulerAdmissionMixin:
    """RetroSpec SchedulerAdmission helpers."""

    def _get_retrospec_gpu_index_footprint(
        self, request: Request
    ) -> RetroSpecGPUIndexFootprint:
        max_context_tokens = min(
            request.num_prompt_tokens + request.max_tokens, self.max_model_len
        )
        footprint = estimate_retrospec_gpu_index_footprint(
            self.vllm_config, self.kv_cache_config, max_context_tokens
        )

        current = self._retrospec_gpu_index_footprints.get(request.request_id)
        if current is None:
            return footprint

        # Resumable input may increase the request bound. Never shrink an
        # existing ticket because worker arenas retain their high-water mark.
        return RetroSpecGPUIndexFootprint(
            cluster_capacity=max(current.cluster_capacity, footprint.cluster_capacity),
            page_capacity=max(current.page_capacity, footprint.page_capacity),
        )

    def _requires_retrospec_gpu_index_ticket(self, request: Request) -> bool:
        if not self.is_retrospec or request.pooling_params is not None:
            return False

        footprint = self._get_retrospec_gpu_index_footprint(request)
        return footprint.cluster_capacity > 0

    def _estimate_retrospec_gpu_index_bytes_with(self, request: Request) -> int:
        footprints = dict(self._retrospec_gpu_index_footprints)
        footprints[request.request_id] = self._get_retrospec_gpu_index_footprint(
            request
        )
        return estimate_retrospec_gpu_index_arena_bytes(
            self.vllm_config, self.kv_cache_config, footprints.values()
        )

    def _validate_retrospec_gpu_index_ticket(self, request: Request) -> None:
        footprint = self._get_retrospec_gpu_index_footprint(request)
        required_bytes = estimate_retrospec_gpu_index_arena_bytes(
            self.vllm_config, self.kv_cache_config, (footprint,)
        )
        if required_bytes > self._retrospec_gpu_index_budget_bytes:
            raise ValueError(
                "RetroSpec request cannot fit the configured GPU index memory "
                f"budget: request={request.request_id!r}, required={required_bytes}, "
                f"budget={self._retrospec_gpu_index_budget_bytes} bytes"
            )

    def _can_admit_retrospec_gpu_index(self, request: Request) -> bool:
        return (
            self._estimate_retrospec_gpu_index_bytes_with(request)
            <= self._retrospec_gpu_index_budget_bytes
        )

    def _reserve_retrospec_gpu_index_ticket(self, request: Request) -> None:
        if not self._can_admit_retrospec_gpu_index(request):
            raise RuntimeError(
                "RetroSpec GPU index admission changed during scheduling"
            )

        self._retrospec_gpu_index_footprints[request.request_id] = (
            self._get_retrospec_gpu_index_footprint(request)
        )

    def _release_retrospec_gpu_index_ticket(self, request_id: str) -> None:
        self._retrospec_gpu_index_footprints.pop(request_id, None)

    def _is_retrospec_layer_major_prefill_candidate(self, request: Request) -> bool:
        if not self.enable_retrospec_layer_major_prefill:
            return False

        if request.status not in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
            RequestStatus.RUNNING,
        ):
            return False

        sampling_params = request.sampling_params
        return (
            request.num_prompt_tokens > self.retrospec_layer_major_prefill_threshold
            and request.pooling_params is None
            and request.prompt_token_ids is not None
            and request.prompt_embeds is None
            and not request.has_encoder_inputs
            and request.lora_request is None
            and request.num_output_tokens == 0
            and request.num_computed_tokens == 0
            and not request.spec_token_ids
            and (sampling_params is None or sampling_params.prompt_logprobs is None)
        )

    def _can_admit_retrospec_layer_major_prefill(self, request: Request) -> bool:
        if (
            request.status != RequestStatus.RUNNING
            and len(self.running) >= self.max_num_running_reqs
        ):
            return False
        if not self._can_admit_retrospec_gpu_index(request):
            return False

        num_recent_blocks = cdiv(self.num_spec_tokens, self.block_size) + 1
        return self.kv_cache_manager.can_allocate_retrospec_prefill_slots(
            request,
            prompt_num_tokens=request.num_prompt_tokens,
            num_recent_blocks=num_recent_blocks,
            blocks_per_cluster=self.retrospec_blocks_per_cluster,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )

    def _select_retrospec_layer_major_prefill_request(self) -> str | None:
        # Layer-major prefill reuses one global workspace. Wait until every
        # earlier prefill and PP batch has released its dependencies.
        if self._retrospec_layer_major_prefill_in_flight_req_id is not None:
            return None
        if self._retrospec_in_flight_batches:
            return None

        request_id = self._retrospec_layer_major_prefill_req_id
        if request_id is not None:
            request = self.requests.get(request_id)
            if request is not None and self._is_retrospec_layer_major_prefill_candidate(
                request
            ):
                if self._can_admit_retrospec_layer_major_prefill(request):
                    return request_id

                # Keep the request selected, but let running decode requests
                # release native blocks before retrying the exclusive prefill.
                return None
            self._retrospec_layer_major_prefill_req_id = None

        if not self.waiting:
            return None

        request = self.waiting.peek_request()
        if not self._is_retrospec_layer_major_prefill_candidate(request):
            return None
        if not self._can_admit_retrospec_layer_major_prefill(request):
            return None

        request_id = request.request_id
        self._retrospec_layer_major_prefill_req_id = request_id
        return request_id


class RetroSpecSchedulerPipelineMixin:
    """RetroSpec SchedulerPipeline helpers."""

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
