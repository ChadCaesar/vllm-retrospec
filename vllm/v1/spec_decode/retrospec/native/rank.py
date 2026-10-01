# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU-only clustered index and sparse attention for RetroSpec.

The native vLLM KV blocks remain the sole exact-token storage. The reverse
cluster layout contains logical token indices, not copies of K/V vectors.
"""

from math import ceil

import torch

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.native.kernels import (
    _cluster_mass_kernel,
    _cluster_rank_dot_kernel,
    _cluster_rank_int8_kernel,
    _cluster_scores_kernel,
    _NativeBatchLayer,
    _NativePlanWorkspace,
    _NativeRankedPlan,
)


class RetroSpecNativeRankMixin:
    def _prepare_plan_workspace(
        self,
        layer_name: str,
        layout: _NativeBatchLayer,
        batch: int,
        device: torch.device,
    ) -> None:
        heads, clusters = layout.counts.shape[1:]
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        width = sparse_width + estimation_width
        workspace = self._plan_workspaces.get(layer_name)
        if (
            workspace is None
            or workspace.ranked.device.type != device.type
            or (
                device.index is not None
                and workspace.ranked.device.index != device.index
            )
            or workspace.ranked.shape[1] < batch
            or workspace.ranked.shape[2] != heads
            or workspace.ranked.shape[3] < width
            or workspace.scores.shape[2] < clusters
        ):
            batch_capacity = layout.keys.shape[0]
            plan_shape = (self.num_speculative_tokens, batch_capacity, heads, width)
            workspace = _NativePlanWorkspace(
                ranked=torch.empty(plan_shape, dtype=torch.int64, device=device),
                candidate_counts=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    heads,
                    dtype=torch.int32,
                    device=device,
                ),
                sparse_mass=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    dtype=torch.float32,
                    device=device,
                ),
                verification_mass=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    dtype=torch.float32,
                    device=device,
                ),
                expanded_mass=torch.empty(
                    self.num_speculative_tokens,
                    batch_capacity,
                    dtype=torch.float32,
                    device=device,
                ),
                scores=torch.empty(
                    batch_capacity,
                    heads,
                    clusters,
                    dtype=torch.float32,
                    device=device,
                ),
                topk_values=torch.empty(
                    batch_capacity,
                    heads,
                    width,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            self._plan_workspaces[layer_name] = workspace
        workspace.candidate_counts[:, :batch].zero_()
        workspace.sparse_mass[:, :batch].fill_(1.0)
        workspace.verification_mass[:, :batch].fill_(1.0)
        workspace.expanded_mass[:, :batch].fill_(1.0)
        self._active_plan_workspaces[layer_name] = workspace

    def _compute_rank_draft(
        self,
        query: torch.Tensor,
        scale: float,
        active_mask: torch.Tensor,
        layout: _NativeBatchLayer,
        output: _NativeRankedPlan | None = None,
        score_buffer: torch.Tensor | None = None,
        topk_values: torch.Tensor | None = None,
        logits_buffer: torch.Tensor | None = None,
        request_rows: torch.Tensor | None = None,
        exact: bool = False,
    ) -> _NativeRankedPlan:
        batch, query_heads, head_size = query.shape
        if request_rows is not None:
            if request_rows.shape != (batch,):
                raise ValueError("Mapped rank request rows must match the query batch")
            if request_rows.device != query.device or request_rows.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("Mapped rank request rows must be GPU integers")
        mapped_rows = request_rows
        if layout.request_slots is not None:
            mapped_rows = (
                layout.request_slots.index_select(0, request_rows.long())
                if request_rows is not None
                else layout.request_slots[:batch]
            )
        heads, clusters = layout.counts.shape[1:]
        group_size = query_heads // heads
        if (
            clusters > 8192
            or triton.next_power_of_2(group_size) * triton.next_power_of_2(clusters)
            > 65536
        ):
            mapped_layout = layout
            if mapped_rows is not None:
                mapped_layout = _NativeBatchLayer(
                    keys=layout.keys.index_select(0, mapped_rows.long()),
                    values=layout.values,
                    counts=layout.counts.index_select(0, mapped_rows.long()),
                    token_indices=layout.token_indices,
                    cluster_offsets=layout.cluster_offsets,
                    indexed_ends=layout.indexed_ends,
                )
            result = self._compute_rank_draft_torch(
                query, scale, active_mask, mapped_layout
            )
            if output is None:
                return result
            output.ranked.copy_(result.ranked)
            output.candidate_counts.copy_(result.candidate_counts)
            output.sparse_mass.copy_(result.sparse_mass)
            output.verification_mass.copy_(result.verification_mass)
            output.expanded_mass.copy_(result.expanded_mass)
            return output
        grouped_query = query.reshape(batch, heads, group_size, head_size)
        if (
            not exact
            and self.draft_rank_dtype == "int8"
            and layout.quantized_keys is not None
            and layout.key_scales is not None
            and query.dtype in (torch.float16, torch.bfloat16)
            and group_size <= 16
            and 32 <= head_size <= 256
        ):
            logits = logits_buffer
            if logits is None:
                logits = torch.empty(
                    batch,
                    heads,
                    group_size,
                    clusters,
                    dtype=torch.float32,
                    device=query.device,
                )
            if mapped_rows is None:
                mapped_rows = torch.arange(
                    batch, dtype=torch.int32, device=query.device
                )
            with self._cuda_timer("gpu_native_rank_dot"):
                _cluster_rank_int8_kernel[(triton.cdiv(clusters, 128), batch * heads)](
                    grouped_query,
                    layout.quantized_keys,
                    layout.key_scales,
                    logits,
                    mapped_rows,
                    *grouped_query.stride()[:3],
                    *layout.quantized_keys.stride()[:3],
                    *layout.key_scales.stride()[:2],
                    *logits.stride()[:3],
                    heads,
                    group_size,
                    clusters,
                    head_size,
                    128,
                    max(32, triton.next_power_of_2(head_size)),
                    num_warps=4,
                )
        elif (
            query.dtype in (torch.float16, torch.bfloat16)
            and layout.keys.dtype == query.dtype
            and group_size <= 16
            and head_size <= 256
        ):
            logits = logits_buffer
            if logits is None:
                logits = torch.empty(
                    batch,
                    heads,
                    group_size,
                    clusters,
                    dtype=torch.float32,
                    device=query.device,
                )
            with self._cuda_timer("gpu_native_rank_dot"):
                _cluster_rank_dot_kernel[(triton.cdiv(clusters, 128), batch * heads)](
                    grouped_query,
                    layout.keys,
                    logits,
                    mapped_rows if mapped_rows is not None else query,
                    *grouped_query.stride()[:3],
                    *layout.keys.stride()[:3],
                    *logits.stride()[:3],
                    heads,
                    group_size,
                    clusters,
                    head_size,
                    128,
                    triton.next_power_of_2(head_size),
                    mapped_rows is not None,
                    num_warps=4,
                )
        else:
            keys = (
                layout.keys.index_select(0, mapped_rows.long())
                if mapped_rows is not None
                else layout.keys[:batch]
            )
            logits = torch.einsum(
                "bhgd,bhcd->bhgc", grouped_query.float(), keys.float()
            )
        scores = score_buffer
        if scores is None:
            scores = torch.empty(
                batch, heads, clusters, dtype=torch.float32, device=query.device
            )
        candidates = (
            output.candidate_counts
            if output is not None
            else torch.empty(batch, heads, dtype=torch.int32, device=query.device)
        )
        _cluster_scores_kernel[(batch, heads)](
            logits,
            layout.counts,
            active_mask,
            scores,
            candidates,
            mapped_rows if mapped_rows is not None else query,
            *logits.stride(),
            layout.counts.stride(0),
            layout.counts.stride(1),
            scores.stride(0),
            scores.stride(1),
            candidates.stride(0),
            group_size,
            clusters,
            scale,
            triton.next_power_of_2(group_size),
            triton.next_power_of_2(clusters),
            mapped_rows is not None,
            num_warps=8 if clusters <= 2048 else 16,
        )
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        if output is None:
            with self._cuda_timer("gpu_native_rank_topk"):
                ranked_scores, ranked = torch.topk(
                    scores, k=sparse_width + estimation_width, dim=2, sorted=True
                )
            sparse_mass = torch.empty(batch, dtype=torch.float32, device=query.device)
            verification_mass = torch.empty_like(sparse_mass)
            expanded_mass = torch.empty_like(sparse_mass)
        else:
            assert topk_values is not None
            with self._cuda_timer("gpu_native_rank_topk"):
                ranked_scores, ranked = torch.topk(
                    scores,
                    k=sparse_width + estimation_width,
                    dim=2,
                    sorted=True,
                    out=(topk_values, output.ranked),
                )
            sparse_mass = output.sparse_mass
            verification_mass = output.verification_mass
            expanded_mass = output.expanded_mass
        _cluster_mass_kernel[(batch,)](
            ranked_scores,
            candidates,
            active_mask,
            sparse_mass,
            verification_mass,
            expanded_mass,
            *ranked_scores.stride(),
            candidates.stride(0),
            heads,
            ranked_scores.shape[2],
            self.retrieval_ratio,
            self.estimation_ratio,
            triton.next_power_of_2(heads),
            triton.next_power_of_2(ranked_scores.shape[2]),
            num_warps=4,
        )
        return output or _NativeRankedPlan(
            ranked, candidates, sparse_mass, verification_mass, expanded_mass
        )

    def _compute_rank_draft_torch(
        self,
        query: torch.Tensor,
        scale: float,
        active_mask: torch.Tensor,
        layout: _NativeBatchLayer,
    ) -> _NativeRankedPlan:
        batch, query_heads, head_size = query.shape
        heads, clusters = layout.counts.shape[1:]
        group_size = query_heads // heads
        logits = (
            torch.einsum(
                "bhgd,bhcd->bhgc",
                query.reshape(batch, heads, group_size, head_size).float(),
                (
                    layout.keys.index_select(0, layout.request_slots[:batch].long())
                    if layout.request_slots is not None
                    else layout.keys[:batch]
                ).float(),
            )
            * scale
        )
        counts = (
            layout.counts.index_select(0, layout.request_slots[:batch].long())
            if layout.request_slots is not None
            else layout.counts[:batch]
        )
        valid = (counts > 0) & active_mask[:, None, None]
        logits += counts.clamp_min(1).float().log()[:, :, None, :]
        probabilities = torch.softmax(
            logits.masked_fill(~valid[:, :, None, :], -1e30), dim=-1
        )
        scores = probabilities.mean(2).masked_fill(~valid, float("-inf"))
        candidates = valid.sum(2, dtype=torch.int32)
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        ranked = torch.topk(
            scores, k=sparse_width + estimation_width, dim=2, sorted=True
        ).indices.to(torch.int32)
        retrieval = torch.ceil(candidates.float() * self.retrieval_ratio).int()
        estimation = torch.ceil(candidates.float() * self.estimation_ratio).int()
        estimation = torch.minimum(estimation, candidates - retrieval)
        rank_ids = torch.arange(ranked.shape[2], device=query.device)
        ranked_scores = scores.gather(2, ranked.long()).clamp_min(0)
        sparse_mass = (
            (ranked_scores * (rank_ids < retrieval[:, :, None])).sum(2).mean(1)
        )
        verification_end = torch.minimum(retrieval * 2, retrieval + estimation)
        verification_mass = (
            (ranked_scores * (rank_ids < verification_end[:, :, None])).sum(2).mean(1)
        )
        expanded_end = torch.minimum(retrieval * 3, retrieval + estimation)
        expanded_mass = (
            (ranked_scores * (rank_ids < expanded_end[:, :, None])).sum(2).mean(1)
        )
        sparse_mass = torch.where(active_mask, sparse_mass, 1.0)
        verification_mass = torch.where(active_mask, verification_mass, 1.0)
        expanded_mass = torch.where(active_mask, expanded_mass, 1.0)
        return _NativeRankedPlan(
            ranked, candidates, sparse_mass, verification_mass, expanded_mass
        )

    def _rank_draft(
        self,
        layer_name: str,
        query: torch.Tensor,
        scale: float,
        active_mask: torch.Tensor,
        step: int,
        layout: _NativeBatchLayer,
    ) -> _NativeRankedPlan:
        batch = query.shape[0]
        workspace = self._active_plan_workspaces[layer_name]
        heads, clusters = layout.counts.shape[1:]
        group_size = query.shape[1] // heads
        sparse_width = min(ceil(clusters * self.retrieval_ratio), clusters)
        estimation_width = min(
            ceil(clusters * self.estimation_ratio), clusters - sparse_width
        )
        width = sparse_width + estimation_width
        logits = None
        if (
            clusters <= 8192
            and triton.next_power_of_2(group_size) * triton.next_power_of_2(clusters)
            <= 65536
            and query.dtype in (torch.float16, torch.bfloat16)
            and layout.keys.dtype == query.dtype
            and group_size <= 16
            and query.shape[2] <= 256
        ):
            logits = self._logit_workspaces.get(layer_name)
            if (
                logits is None
                or logits.shape[0] < batch
                or logits.shape[1] != heads
                or logits.shape[2] != group_size
                or logits.shape[3] < clusters
                or logits.device != query.device
            ):
                logits = torch.empty(
                    batch,
                    heads,
                    group_size,
                    clusters,
                    dtype=torch.float32,
                    device=query.device,
                )
                self._logit_workspaces[layer_name] = logits
        slot = _NativeRankedPlan(
            workspace.ranked[step, :batch, :, :width],
            workspace.candidate_counts[step, :batch],
            workspace.sparse_mass[step, :batch],
            workspace.verification_mass[step, :batch],
            workspace.expanded_mass[step, :batch],
        )
        plan = self._compute_rank_draft(
            query,
            scale,
            active_mask,
            layout,
            output=slot,
            score_buffer=workspace.scores[:batch, :, :clusters],
            topk_values=workspace.topk_values[:batch, :, :width],
            logits_buffer=logits[:batch, :, :, :clusters]
            if logits is not None
            else None,
        )
        self._plans.setdefault(layer_name, {})[step] = plan
        return plan

    def _rank_parallel_bonus(
        self,
        layer_name: str,
        query: torch.Tensor,
        scale: float,
        request_rows: torch.Tensor,
        token_steps: torch.Tensor,
        active_mask: torch.Tensor,
        layout: _NativeBatchLayer,
    ) -> _NativeRankedPlan:
        if (
            request_rows.shape != (query.shape[0],)
            or token_steps.shape != request_rows.shape
        ):
            raise ValueError("Bonus plan indices must match the query batch")
        if token_steps.device != query.device or token_steps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Bonus plan steps must be GPU integers")
        plan = self._compute_rank_draft(
            query, scale, active_mask, layout, request_rows=request_rows, exact=True
        )
        workspace = self._active_plan_workspaces[layer_name]
        rows = request_rows.long()
        steps = token_steps.long()
        workspace.ranked[steps, rows, :, : plan.ranked.shape[-1]] = plan.ranked
        workspace.candidate_counts[steps, rows] = plan.candidate_counts
        workspace.sparse_mass[steps, rows] = plan.sparse_mass
        workspace.verification_mass[steps, rows] = plan.verification_mass
        workspace.expanded_mass[steps, rows] = plan.expanded_mass
        return plan
