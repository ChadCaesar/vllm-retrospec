# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU-only clustered index and sparse attention for RetroSpec.

The native vLLM KV blocks remain the sole exact-token storage. The reverse
cluster layout contains logical token indices, not copies of K/V vectors.
"""

import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.native.kernels import (
    _merge_native_attention_partitions,
    _native_grouped_sparse_attention_kernel,
    _native_sparse_attention_kernel,
    _NativeAttentionWorkspace,
    _NativeRankedPlan,
)


class RetroSpecNativeAttentionMixin:
    def _attention_workspace(
        self, query: torch.Tensor, partitions: int
    ) -> _NativeAttentionWorkspace:
        workspace = self._shared_attention_workspace
        rows, heads, head_size = query.shape
        if (
            workspace is None
            or workspace.output.shape[0] < rows
            or workspace.output.shape[1] != heads
            or workspace.output.shape[2] < partitions
            or workspace.output.shape[3] != head_size
            or workspace.output.device != query.device
        ):
            row_capacity = rows
            if workspace is not None:
                previous_rows = workspace.output.shape[0]
                row_capacity = max(
                    rows, previous_rows * 2 if rows > previous_rows else previous_rows
                )
            workspace = _NativeAttentionWorkspace(
                output=torch.empty(
                    row_capacity,
                    heads,
                    partitions,
                    head_size,
                    device=query.device,
                    dtype=torch.float32,
                ),
                maximum=torch.empty(
                    row_capacity,
                    heads,
                    partitions,
                    device=query.device,
                    dtype=torch.float32,
                ),
                denominator=torch.empty(
                    row_capacity,
                    heads,
                    partitions,
                    device=query.device,
                    dtype=torch.float32,
                ),
            )
            self._shared_attention_workspace = workspace
        return workspace

    def forward(
        self,
        *,
        layer_name: str,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        active_mask: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        step: int,
        expanded: bool = False,
        sparse_verify: bool = False,
        request_indices: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        bonus_start_index: int | None = None,
    ) -> torch.Tensor:
        if not self._request_ids:
            raise RuntimeError("GPU-native attention requires an active proposal")
        with self._cuda_timer("gpu_native_batch_layout"):
            layout = self._batch_layer(layer_name, query.device, query.dtype)
        draft_attention = request_indices is None
        if draft_attention:
            attention_timer_name = "gpu_native_draft_attention"
            request_slots = self._request_slots
            if (
                request_slots is None
                or request_slots.shape[0] < query.shape[0]
                or request_slots.device != query.device
            ):
                request_slots = torch.arange(
                    query.shape[0], dtype=torch.int32, device=query.device
                )
                self._request_slots = request_slots
            request_indices = request_slots[: query.shape[0]]
            with self._cuda_timer("gpu_native_rank"):
                plan = self._rank_draft(
                    layer_name, query, scale, active_mask, step, layout
                )
            mass = plan.sparse_mass
        else:
            attention_timer_name = (
                "gpu_native_expanded_verify_attention"
                if expanded
                else "gpu_native_sparse_verify_attention"
                if sparse_verify
                else "gpu_native_verify_attention"
            )
            if token_indices is None:
                raise ValueError("Verification requires draft token indices")
            if bonus_start_index is not None and (
                not sparse_verify or not 0 < bonus_start_index < query.shape[0]
            ):
                raise ValueError("Bonus rows require a sparse verification suffix")
            # Draft rows reuse their ranking; only bonus rows lack a plan.
            if sparse_verify and bonus_start_index is not None:
                with self._cuda_timer("gpu_native_sparse_rank"):
                    self._rank_parallel_bonus(
                        layer_name,
                        query[bonus_start_index:],
                        scale,
                        request_indices[bonus_start_index:],
                        token_indices[bonus_start_index:],
                        active_mask[bonus_start_index:],
                        layout,
                    )
            plans = self._plans.get(layer_name, {})
            if not plans:
                raise RuntimeError("Verification has no GPU-native draft plan")
            workspace = self._active_plan_workspaces[layer_name]
            request_rows = request_indices.long()
            draft_steps = token_indices.long()
            mass_table = (
                workspace.expanded_mass
                if expanded
                else workspace.verification_mass
                if sparse_verify
                else workspace.sparse_mass
            )
            selected_mass = mass_table[draft_steps, request_rows]
            plan = _NativeRankedPlan(
                workspace.ranked[draft_steps, request_rows],
                workspace.candidate_counts[draft_steps, request_rows],
                selected_mass,
                selected_mass,
                selected_mass,
            )
            mass = plan.sparse_mass

        heads = layout.counts.shape[1]
        if query.shape[1] % heads:
            raise ValueError("GPU-native query and KV heads are incompatible")
        cluster_count = self._active_cluster_counts[layer_name]
        num_partitions = (
            8 if cluster_count >= 4096 else 4 if cluster_count >= 1536 else 1
        )
        attention_workspace = (
            self._attention_workspace(query, num_partitions)
            if num_partitions > 1
            else None
        )
        partial_output = attention_workspace.output if attention_workspace else output
        partial_maximum = attention_workspace.maximum if attention_workspace else output
        partial_denominator = (
            attention_workspace.denominator if attention_workspace else output
        )
        queries_per_kv = query.shape[1] // heads
        grouped_attention = (
            sparse_verify
            and 4 <= queries_per_kv <= 16
            and query.dtype in (torch.float16, torch.bfloat16)
            and key_cache.dtype == query.dtype
            and value_cache.dtype == query.dtype
            and layout.keys.dtype == query.dtype
            and layout.values.dtype == query.dtype
            and 32 <= query.shape[2] <= 256
        )
        attention_kernel = (
            _native_grouped_sparse_attention_kernel
            if grouped_attention
            else _native_sparse_attention_kernel
        )
        attention_heads = heads if grouped_attention else query.shape[1]
        with self._cuda_timer(attention_timer_name):
            attention_kernel[(query.shape[0], attention_heads, num_partitions)](
                query,
                key_cache,
                value_cache,
                block_table,
                layout.token_indices,
                layout.cluster_offsets,
                layout.keys,
                layout.values,
                layout.counts,
                plan.ranked,
                plan.candidate_counts,
                layout.indexed_ends,
                seq_lens,
                request_indices,
                layout.request_slots
                if layout.request_slots is not None
                else request_indices,
                output,
                partial_output,
                partial_maximum,
                partial_denominator,
                *query.stride(),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                block_table.stride(0),
                block_table.stride(1),
                layout.token_indices.stride(0),
                layout.token_indices.stride(1),
                layout.cluster_offsets.stride(0),
                layout.cluster_offsets.stride(1),
                layout.keys.stride(0),
                layout.keys.stride(1),
                layout.counts.stride(0),
                layout.counts.stride(1),
                plan.ranked.stride(0),
                plan.ranked.stride(1),
                plan.candidate_counts.stride(0),
                *output.stride(),
                *partial_output.stride()[:3],
                partial_maximum.stride(0),
                partial_maximum.stride(1),
                scale,
                self.retrieval_ratio,
                self.estimation_ratio,
                expanded,
                sparse_verify,
                heads,
                queries_per_kv,
                layout.counts.shape[2],
                plan.ranked.shape[2],
                self.block_size,
                query.shape[2],
                triton.next_power_of_2(query.shape[2]),
                32,
                32,
                num_partitions,
            )
            if attention_workspace is not None:
                _merge_native_attention_partitions[(query.shape[0], query.shape[1])](
                    attention_workspace.output,
                    attention_workspace.maximum,
                    attention_workspace.denominator,
                    output,
                    *attention_workspace.output.stride()[:3],
                    attention_workspace.maximum.stride(0),
                    attention_workspace.maximum.stride(1),
                    *output.stride(),
                    query.shape[2],
                    num_partitions,
                    triton.next_power_of_2(query.shape[2]),
                )
        return mass
