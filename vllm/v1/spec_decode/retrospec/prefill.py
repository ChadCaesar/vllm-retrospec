# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.sched.output import (
    RetroSpecLayerMajorPrefillDescriptor,
    SchedulerOutput,
)
from vllm.v1.outputs import RetroSpecLayerMajorPrefillCompletion


class RetroSpecLayerMajorPrefillProtocol:
    """Validate the scheduler/worker layer-major prefill contract."""

    @staticmethod
    def validate_scheduler_output(scheduler_output: SchedulerOutput) -> None:
        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is None:
            return

        request_id = descriptor.request_id
        scheduled_request_ids = set(scheduler_output.num_scheduled_tokens)
        if scheduled_request_ids != {request_id}:
            raise RuntimeError(
                "RetroSpec layer-major prefill must be scheduled as an exclusive "
                f"batch, got {scheduled_request_ids}"
            )

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]
        if num_scheduled_tokens != descriptor.num_scheduled_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill descriptor does not match "
                "num_scheduled_tokens: "
                f"descriptor={descriptor.num_scheduled_tokens}, "
                f"scheduled={num_scheduled_tokens}"
            )

        if scheduler_output.total_num_scheduled_tokens != num_scheduled_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill total token count is inconsistent"
            )

        if scheduler_output.scheduled_spec_decode_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill cannot contain speculative tokens"
            )

        if scheduler_output.scheduled_encoder_inputs:
            raise RuntimeError(
                "RetroSpec layer-major prefill does not support encoder inputs"
            )

    @staticmethod
    def validate_cached_prompt(
        descriptor: RetroSpecLayerMajorPrefillDescriptor,
        cached_prompt_num_tokens: int,
    ) -> None:
        if cached_prompt_num_tokens != descriptor.prompt_num_tokens:
            raise RuntimeError(
                "RetroSpec layer-major prefill prompt descriptor is stale: "
                f"descriptor={descriptor.prompt_num_tokens}, "
                f"worker={cached_prompt_num_tokens}"
            )

    @classmethod
    def complete_scheduled_range(
        cls, scheduler_output: SchedulerOutput
    ) -> RetroSpecLayerMajorPrefillCompletion | None:
        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is None:
            return None

        cls.validate_scheduler_output(scheduler_output)
        return RetroSpecLayerMajorPrefillCompletion(
            request_id=descriptor.request_id,
            num_completed_prompt_tokens=descriptor.scheduled_end,
        )
