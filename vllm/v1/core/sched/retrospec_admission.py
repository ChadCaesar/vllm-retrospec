# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.utils.math_utils import cdiv
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.retrospec.capacity import (
    RetroSpecGPUIndexFootprint,
    estimate_retrospec_gpu_index_arena_bytes,
    estimate_retrospec_gpu_index_footprint,
)


class RetroSpecSchedulerAdmissionMixin:
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
            retain_all_kv=True,
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
