# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.kv_cache_utils import KV_CACHE_NULL_BLOCK_ID
from vllm.v1.spec_decode.retrospec import RetroSpecProposer
from vllm.v1.spec_decode.retrospec.capacity import (
    estimate_retrospec_gpu_index_arena_bytes,
    estimate_retrospec_gpu_index_footprint,
)
from vllm.v1.spec_decode.retrospec.prefill import (
    RetroSpecLayerPrefillTile,
    RetroSpecLayerPrefillWorkspace,
    resolve_retrospec_layer_model,
)
from vllm.v1.worker.retrospec_runner_state import ExecuteModelState

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class RetroSpecRunnerPrefillMixin:
    def _build_retrospec_prefill_tile_metadata(
        self,
        tile,
        builder: AttentionMetadataBuilder,
    ) -> AttentionMetadata:
        query_start_loc_cpu = torch.tensor(
            [0, tile.num_scheduled_tokens], dtype=torch.int32
        )
        query_start_loc = query_start_loc_cpu.to(device=self.device, non_blocking=True)
        common_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=torch.tensor(
                [tile.context_len], dtype=torch.int32, device=self.device
            ),
            num_reqs=1,
            num_actual_tokens=tile.num_scheduled_tokens,
            max_query_len=tile.num_scheduled_tokens,
            max_seq_len=tile.context_len,
            block_table_tensor=tile.block_table,
            slot_mapping=tile.slot_mapping,
            causal=True,
        )
        return builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata,
        )

    @staticmethod
    def _copy_retrospec_prefill_resident_blocks(
        workspace: RetroSpecLayerPrefillWorkspace,
        native_kv_cache: torch.Tensor,
        source_block_ids: torch.Tensor,
        destination_block_ids: torch.Tensor,
    ) -> None:
        source = workspace.kv_cache.index_select(1, source_block_ids)
        native_kv_cache.index_copy_(1, destination_block_ids, source)

    def _build_retrospec_post_prefill_metadata(
        self,
        block_table: torch.Tensor,
        prompt_num_tokens: int,
        last_prompt_slot: int,
    ) -> CommonAttentionMetadata:
        return CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor(
                [prompt_num_tokens], dtype=torch.int32, device=self.device
            ),
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=prompt_num_tokens,
            block_table_tensor=block_table,
            slot_mapping=torch.tensor(
                [last_prompt_slot], dtype=torch.int64, device=self.device
            ),
            causal=True,
        )

    def _estimate_retrospec_prefill_activation_bytes_per_token(self) -> int:
        hidden_size = self.model_config.get_hidden_size()
        head_size = self.model_config.get_head_size()
        num_query_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        intermediate_size = int(
            getattr(
                self.model_config.hf_text_config,
                "intermediate_size",
                4 * hidden_size,
            )
        )
        local_intermediate_size = cdiv(
            intermediate_size,
            self.parallel_config.tensor_parallel_size,
        )

        activation_elements = (
            4 * hidden_size
            + 2 * local_intermediate_size
            + (num_query_heads + 2 * num_kv_heads) * head_size
        )
        return 2 * get_dtype_size(self.model_config.dtype) * activation_elements

    def _estimate_retrospec_prefill_future_memory_bytes(
        self,
        prompt_num_tokens: int,
    ) -> int:
        footprint = estimate_retrospec_gpu_index_footprint(
            self.vllm_config,
            self.kv_cache_config,
            prompt_num_tokens,
        )
        return estimate_retrospec_gpu_index_arena_bytes(
            self.vllm_config,
            self.kv_cache_config,
            (footprint,),
        )

    def _build_retrospec_prefill_tile_plan(
        self,
        workspace: RetroSpecLayerPrefillWorkspace,
        prompt_num_tokens: int,
        tile_size: int,
        builder: AttentionMetadataBuilder,
    ) -> tuple[
        tuple[tuple[RetroSpecLayerPrefillTile, AttentionMetadata], ...],
        RetroSpecLayerPrefillTile,
    ]:
        tiles = tuple(
            workspace.tile(
                tile_start,
                min(tile_start + tile_size, prompt_num_tokens),
            )
            for tile_start in range(0, prompt_num_tokens, tile_size)
        )
        tile_plan = tuple(
            (
                tile,
                self._build_retrospec_prefill_tile_metadata(tile, builder),
            )
            for tile in tiles
        )
        return tile_plan, workspace.tile(0, prompt_num_tokens)

    def _execute_retrospec_layer_major_prefill(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is None:
            raise RuntimeError("Missing layer-major prefill descriptor")
        drafter = self.drafter
        if not isinstance(drafter, RetroSpecProposer):
            raise RuntimeError("Layer-major prefill requires RetroSpec")

        stats = drafter.performance_stats
        stats.add_counter("layer_prefill_requests")
        stats.add_counter("layer_prefill_prompt_tokens", descriptor.prompt_num_tokens)
        with stats.cpu_timer("layer_prefill_total_wall"):
            return self._execute_retrospec_layer_major_prefill_impl(
                scheduler_output, intermediate_tensors
            )

    def _execute_retrospec_layer_major_prefill_impl(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
        descriptor = scheduler_output.retrospec_layer_major_prefill
        if descriptor is None:
            raise RuntimeError("Missing layer-major prefill descriptor")

        drafter = self.drafter
        if not isinstance(drafter, RetroSpecProposer):
            raise RuntimeError("Layer-major prefill requires RetroSpec")
        workspace = self.retrospec_layer_prefill_workspace
        if workspace is None:
            raise RuntimeError("Layer-major prefill workspace is unavailable")

        request = self.requests[descriptor.request_id]

        layer_model = resolve_retrospec_layer_model(self.model)
        layer_names = drafter.attn_layer_names
        protocol = drafter.pipeline_protocol
        stage = protocol.describe_stage(layer_model, layer_names)
        layer_indices = tuple(range(layer_model.start_layer, layer_model.end_layer))
        if len(layer_names) != len(layer_indices):
            raise RuntimeError(
                "Model layer count and registered attention layer count differ"
            )

        kv_cache_group_id = drafter.kv_cache_group_id
        if kv_cache_group_id is None:
            raise RuntimeError("RetroSpec KV-cache group is unavailable")

        self.input_batch.block_table.commit_block_table(1)
        native_block_table = self.input_batch.block_table[
            kv_cache_group_id
        ].get_device_tensor(1)

        prompt_num_tokens = descriptor.prompt_num_tokens
        workspace.prepare(prompt_num_tokens)
        prompt_block_count = cdiv(prompt_num_tokens, workspace.block_size)
        source_blocks = (
            0,
            *range(descriptor.resident_start_block, prompt_block_count),
        )
        request_block_ids = request.block_ids[kv_cache_group_id]
        destination_blocks = tuple(
            request_block_ids[logical_block] for logical_block in source_blocks
        )
        if any(block_id == KV_CACHE_NULL_BLOCK_ID for block_id in destination_blocks):
            raise RuntimeError("A required resident prompt block is not allocated")

        source_block_ids = torch.tensor(
            source_blocks, dtype=torch.int64, device=self.device
        )
        destination_block_ids = torch.tensor(
            destination_blocks, dtype=torch.int64, device=self.device
        )
        positions = torch.arange(
            prompt_num_tokens, dtype=torch.int64, device=self.device
        )
        prompt_token_ids: torch.Tensor | None = None
        if stage.is_first:
            if request.prompt_token_ids is None:
                raise RuntimeError(
                    "The first RetroSpec PP rank requires prompt token IDs"
                )
            prompt_token_ids = torch.tensor(
                request.prompt_token_ids, dtype=torch.int64, device=self.device
            )
        hidden_states, residual_states = protocol.prepare_layer_prefill_input(
            stage=stage,
            layer_model=layer_model,
            prompt_token_ids=prompt_token_ids,
            intermediate_tensors=intermediate_tensors,
            prompt_num_tokens=prompt_num_tokens,
        )
        has_residual_input = residual_states is not None
        if residual_states is None:
            residual_states = torch.empty_like(hidden_states)
        builder = drafter.get_attention_metadata_builder()
        tile_planner = self.retrospec_layer_prefill_tile_planner
        if tile_planner is None:
            raise RuntimeError("Layer-major prefill tile planner is unavailable")

        future_memory_reserve_bytes = (
            self._estimate_retrospec_prefill_future_memory_bytes(prompt_num_tokens)
        )
        tile_selection = tile_planner.select(
            prompt_num_tokens,
            future_memory_reserve_bytes=future_memory_reserve_bytes,
        )
        tile_size_tensor = torch.tensor(
            tile_selection.tile_size,
            dtype=torch.int64,
            device=self.device,
        )
        if get_tp_group().world_size > 1:
            torch.distributed.all_reduce(
                tile_size_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        tile_size = int(tile_size_tensor.item())

        stats = drafter.performance_stats
        stats.observe_peak("layer_prefill_tile_tokens", tile_size)
        stats.observe_peak(
            "layer_prefill_activation_estimate_bytes",
            tile_size * tile_planner.activation_bytes_per_token,
        )
        stats.observe_peak(
            "layer_prefill_available_memory_bytes",
            tile_selection.available_memory_bytes,
        )
        stats.observe_peak(
            "layer_prefill_reserve_memory_bytes",
            tile_selection.reserve_memory_bytes,
        )
        stats.observe_peak(
            "layer_prefill_future_reserve_bytes",
            tile_selection.future_memory_reserve_bytes,
        )
        logger.debug(
            "RetroSpec layer-prefill selected %d-token tiles for %d tokens "
            "(available=%d, reserve=%d, future_reserve=%d, "
            "activation_estimate=%d)",
            tile_size,
            prompt_num_tokens,
            tile_selection.available_memory_bytes,
            tile_selection.reserve_memory_bytes,
            tile_selection.future_memory_reserve_bytes,
            tile_size * tile_planner.activation_bytes_per_token,
        )

        tile_plan: (
            tuple[tuple[RetroSpecLayerPrefillTile, AttentionMetadata], ...] | None
        ) = None
        full_prompt_tile: RetroSpecLayerPrefillTile | None = None

        try:
            for layer_index, layer_name in zip(layer_indices, layer_names, strict=True):
                attention_layer = self.compilation_config.static_forward_context[
                    layer_name
                ]
                if not isinstance(attention_layer, Attention):
                    raise TypeError(
                        f"Layer {layer_name!r} is not a vLLM Attention module"
                    )

                workspace.begin_layer(layer_name)
                compute_timer = stats.start_cuda_timer("layer_prefill_compute")
                try:
                    if tile_plan is None:
                        tile_plan, full_prompt_tile = (
                            self._build_retrospec_prefill_tile_plan(
                                workspace,
                                prompt_num_tokens,
                                tile_size,
                                builder,
                            )
                        )
                        stats.add_counter(
                            "layer_prefill_metadata_builds", len(tile_plan)
                        )
                        stats.add_counter(
                            "layer_prefill_tile_executions",
                            len(tile_plan) * len(layer_names),
                        )

                    for tile, attn_metadata in tile_plan:
                        tile_start = tile.scheduled_start
                        tile_end = tile.scheduled_end
                        per_layer_metadata = {layer_name: attn_metadata}
                        per_layer_slot_mapping = {layer_name: tile.slot_mapping}
                        with (
                            workspace.bind_layer(layer_name, attention_layer),
                            set_forward_context(
                                per_layer_metadata,
                                self.vllm_config,
                                num_tokens=tile.num_scheduled_tokens,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                slot_mapping=per_layer_slot_mapping,
                            ),
                        ):
                            tile_hidden, tile_residual = layer_model.forward_layer(
                                layer_index,
                                positions[tile_start:tile_end],
                                hidden_states[tile_start:tile_end],
                                residual_states[tile_start:tile_end]
                                if has_residual_input
                                else None,
                            )

                        if tile_residual is None:
                            raise RuntimeError(
                                "Layer-major prefill layer did not return residual"
                            )
                        # The first layer may return its input hidden slice as residual.
                        residual_states[tile_start:tile_end].copy_(tile_residual)
                        hidden_states[tile_start:tile_end].copy_(tile_hidden)

                    stats.stop_cuda_timer(compute_timer)
                    compute_timer = None
                    has_residual_input = True

                    assert full_prompt_tile is not None
                    key_cache, value_cache = workspace.kv_cache.unbind(0)
                    drafter.stage_layer_major_prefill_layer(
                        layer_name=layer_name,
                        request_id=descriptor.request_id,
                        seq_len=prompt_num_tokens,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=full_prompt_tile.block_table,
                    )
                    native_kv_cache = attention_layer.kv_cache[0]
                    self._copy_retrospec_prefill_resident_blocks(
                        workspace,
                        native_kv_cache,
                        source_block_ids,
                        destination_block_ids,
                    )
                    workspace.end_layer()
                except BaseException:
                    stats.stop_cuda_timer(compute_timer)
                    workspace.abort_layer()
                    raise

            with stats.cpu_timer("layer_prefill_commit_wall"):
                drafter.commit_layer_major_prefill(descriptor.request_id)
        except BaseException:
            drafter.abort_layer_major_prefill()
            raise

        req_index = self.input_batch.req_id_to_index[descriptor.request_id]
        request.num_computed_tokens = prompt_num_tokens
        self.input_batch.num_computed_tokens_cpu[req_index] = prompt_num_tokens
        self.discard_request_mask.np[:1] = False
        self.discard_request_mask.copy_to_gpu(1)

        last_prompt_position = prompt_num_tokens - 1
        last_prompt_block = last_prompt_position // workspace.block_size
        last_prompt_block_id = request_block_ids[last_prompt_block]
        last_prompt_slot = (
            last_prompt_block_id * workspace.block_size
            + last_prompt_position % workspace.block_size
        )
        common_metadata = self._build_retrospec_post_prefill_metadata(
            native_block_table,
            prompt_num_tokens,
            last_prompt_slot,
        )
        slot_mappings = {
            layer_name: common_metadata.slot_mapping for layer_name in layer_names
        }

        if not stage.is_last:
            self._save_retrospec_pipeline_proposal_state(
                scheduler_output,
                spec_decode_metadata=None,
                common_attn_metadata=common_metadata,
            )
            return protocol.make_layer_prefill_output(
                stage, hidden_states, residual_states
            )

        last_hidden_state = layer_model.finalize_hidden_states(
            hidden_states[-1:], residual_states[-1:]
        )
        del hidden_states, residual_states
        logits = self.model.compute_logits(last_hidden_state)
        if logits is None:
            raise RuntimeError("Layer-major prefill did not produce logits")

        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=common_metadata,
            hidden_states=last_hidden_state,
            sample_hidden_states=last_hidden_state,
            aux_hidden_states=None,
            ec_connector_output=None,
            cudagraph_stats=None,
            slot_mappings=slot_mappings,
        )
        self.kv_connector_output = None
        return None
