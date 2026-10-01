# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from math import prod

import torch

from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.legacy.cluster_store import (
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecVerificationResolveRequest,
)
from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecResidentBatchView,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.types import (
    RetroSpecIndexedTokenAttentionSelection,
    RetroSpecRankedSelectionPlan,
    RetroSpecTokenAttentionSelection,
    RetroSpecTokenSelectionPlan,
    _DraftSelectionScratch,
    _IndexedVerificationTransaction,
    _IndexedVerificationWorkspace,
    _PackedClusterZones,
    _PreparedIndexedVerificationLayer,
    _SelectionPlanTable,
    _SelectionStepWorkspace,
)
from vllm.v1.spec_decode.retrospec.legacy.selection_kernels import (
    emit_primary_exact_token_plan,
    gather_resident_estimation,
    gather_resident_exact_pages,
    pack_ranked_verification_exact_plan,
    pack_ranked_verification_plan,
)
from vllm.v1.spec_decode.retrospec.workspace import (
    exact_attention_primary_token_capacity,
)


class _RetroSpecSegmentedTokenIndexPlanningMixin:
    @staticmethod
    def _pack_bounded_mask_indices(
        mask: torch.Tensor,
        output_width: int,
        output_indices: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
        topk_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack True positions using a synchronization-free upper bound."""
        if output_width < 0:
            raise ValueError("output_width must be non-negative")
        supplied_outputs = (output_indices, output_mask, topk_order)
        if any(output is not None for output in supplied_outputs) and not all(
            output is not None for output in supplied_outputs
        ):
            raise ValueError("Bounded packing outputs must be supplied together")

        max_num_items = mask.shape[-1]
        leading_shape = mask.shape[:-1]
        active_width = min(output_width, max_num_items)

        if output_indices is None:
            output_indices = torch.empty(
                (*leading_shape, active_width),
                dtype=torch.int64,
                device=mask.device,
            )
            output_mask = torch.empty_like(output_indices, dtype=torch.bool)
            topk_order = torch.empty_like(output_indices)
        else:
            expected_shape = (*leading_shape, output_width)
            if output_indices.shape != expected_shape:
                raise ValueError("Bounded packing index output has an invalid shape")
            if output_mask is None or output_mask.shape != expected_shape:
                raise ValueError("Bounded packing mask output has an invalid shape")
            if topk_order is None or topk_order.shape != expected_shape:
                raise ValueError("Bounded packing scratch output has an invalid shape")
            output_indices.zero_()
            output_mask.zero_()

        assert output_mask is not None
        assert topk_order is not None
        if active_width == 0:
            return output_indices, output_mask

        logical_indices = torch.arange(
            max_num_items,
            dtype=torch.int64,
            device=mask.device,
        )
        sentinel = torch.full(
            (),
            max_num_items,
            dtype=torch.int64,
            device=mask.device,
        )

        candidates = torch.where(
            mask,
            logical_indices,
            sentinel,
        )

        packed_indices = output_indices[..., :active_width]
        packed_order = topk_order[..., :active_width]
        torch.topk(
            candidates,
            k=active_width,
            dim=-1,
            largest=False,
            sorted=True,
            out=(packed_indices, packed_order),
        )

        packed_mask = output_mask[..., :active_width]
        torch.lt(packed_indices, max_num_items, out=packed_mask)
        packed_indices.clamp_(
            min=0,
            max=max_num_items - 1,
        )

        return output_indices, output_mask

    @staticmethod
    def _sum_selected_scores(
        selected_scores: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sum probability mass already aligned with a ranked cluster zone."""
        if selected_scores.shape != selected_mask.shape:
            raise ValueError("Selected cluster scores and mask must match")

        if selected_scores.shape[2] == 0:
            return torch.zeros(
                selected_scores.shape[:2],
                dtype=selected_scores.dtype,
                device=selected_scores.device,
            )

        return selected_scores.masked_fill(~selected_mask, 0.0).sum(dim=2)

    def _get_draft_selection_scratch(
        self,
        batch_size: int,
        num_kv_heads: int,
        sparse_retrieval_width: int,
        device: torch.device,
    ) -> _DraftSelectionScratch:
        scratch = self._draft_selection_scratch
        matches = scratch is not None and scratch.matches(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            sparse_retrieval_width=sparse_retrieval_width,
            device=device,
        )
        if matches:
            return scratch

        batch_capacity = batch_size
        compatible_scratch = (
            scratch is not None
            and scratch.draft_exact_cluster_handles.shape[1] == num_kv_heads
            and scratch.draft_exact_cluster_handles.device == device
        )
        if compatible_scratch:
            assert scratch is not None
            batch_capacity = max(
                batch_capacity,
                scratch.draft_exact_cluster_handles.shape[0],
            )
            old_retrieval_width = scratch.draft_exact_cluster_handles.shape[2]
            sparse_retrieval_width = max(sparse_retrieval_width, old_retrieval_width)
        scratch = _DraftSelectionScratch.allocate(
            batch_capacity=batch_capacity,
            num_kv_heads=num_kv_heads,
            sparse_retrieval_width=sparse_retrieval_width,
            device=device,
        )
        self._draft_selection_scratch = scratch
        return scratch

    def _get_indexed_verification_workspace(
        self,
        pair_capacity: int,
        num_kv_heads: int,
        exact_width: int,
        estimation_width: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _IndexedVerificationWorkspace:
        workspace = self._indexed_verification_workspace
        matches = workspace is not None and workspace.matches(
            pair_capacity=pair_capacity,
            num_kv_heads=num_kv_heads,
            exact_width=exact_width,
            estimation_width=estimation_width,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        if matches:
            return workspace

        if workspace is not None:
            pair_capacity = max(pair_capacity, workspace.keys.shape[0])
            exact_width = max(exact_width, workspace.exact_cluster_indices.shape[2])
            estimation_width = max(estimation_width, workspace.keys.shape[2])
        workspace = _IndexedVerificationWorkspace.allocate(
            pair_capacity=pair_capacity,
            num_kv_heads=num_kv_heads,
            exact_width=exact_width,
            estimation_width=estimation_width,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        self._indexed_verification_workspace = workspace
        return workspace

    def _get_selection_plan_step(
        self,
        layer_name: str,
        plan_slot: int,
        view: RetroSpecResidentBatchView,
        batch_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[
        RetroSpecRankedSelectionPlan,
        _SelectionStepWorkspace,
        _SelectionPlanTable,
    ]:
        if not 0 <= plan_slot < self.num_speculative_tokens:
            raise ValueError("plan_slot is outside the speculative range")

        sparse_retrieval_width, sparse_estimation_width, expanded_retrieval_width = (
            self._maximum_zone_widths(view.max_num_clusters)
        )
        primary_exact_width = exact_attention_primary_token_capacity(
            max_model_len=self.max_model_len,
            prefill_segment_size=self.prefill_segment_size_tokens,
            generation_update_interval=self.generation_update_interval,
            num_speculative_tokens=self.num_speculative_tokens,
            block_size=self.block_size,
        )
        max_access_width = sparse_retrieval_width
        if device.type == "cuda":
            self.cluster_store.reserve_resident_access_ring(
                device,
                batch_size * num_kv_heads * max_access_width,
            )
            self.cluster_store.reserve_verification_resolve_workspace(
                device,
                batch_size
                * self.num_speculative_tokens
                * num_kv_heads
                * expanded_retrieval_width,
                view.max_pages_per_cluster,
                batch_size * self.num_speculative_tokens * num_kv_heads,
                batch_size
                * self.num_speculative_tokens
                * num_kv_heads
                * expanded_retrieval_width
                * view.max_pages_per_cluster,
            )

        table = self._selection_plan_tables.get(layer_name)
        matches = table is not None and table.matches(
            num_steps=self.num_speculative_tokens,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            primary_exact_width=primary_exact_width,
            sparse_retrieval_width=sparse_retrieval_width,
            prefetch_width=max_access_width,
            expanded_retrieval_width=expanded_retrieval_width,
            sparse_estimation_width=sparse_estimation_width,
            max_pages_per_cluster=view.max_pages_per_cluster,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        if not matches:
            if layer_name in self._selection_plan_written_layers:
                raise RuntimeError(
                    "Selection-plan shape changed during an active proposal"
                )
            batch_capacity = batch_size
            if table is not None:
                batch_capacity = max(batch_capacity, table.batch_capacity)
            table = _SelectionPlanTable.allocate(
                layer_name=layer_name,
                num_steps=self.num_speculative_tokens,
                batch_capacity=batch_capacity,
                num_kv_heads=num_kv_heads,
                primary_exact_width=primary_exact_width,
                sparse_retrieval_width=sparse_retrieval_width,
                prefetch_width=max_access_width,
                expanded_retrieval_width=expanded_retrieval_width,
                sparse_estimation_width=sparse_estimation_width,
                max_pages_per_cluster=view.max_pages_per_cluster,
                head_size=head_size,
                dtype=dtype,
                device=device,
            )
            self._selection_plan_tables[layer_name] = table

        assert table is not None
        scratch = self._get_draft_selection_scratch(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            sparse_retrieval_width=sparse_retrieval_width,
            device=device,
        )
        plan = table.ranked_plan(plan_slot, batch_size)
        workspace = table.step_workspace(plan_slot, batch_size, scratch)
        return plan, workspace, table

    @staticmethod
    def _build_resident_exact_cluster_selection(
        view: RetroSpecResidentBatchView,
        packed_cluster_indices: torch.Tensor,
        packed_cluster_mask: torch.Tensor,
        selected_cluster_ids: torch.Tensor | None = None,
        selected_page_ids: torch.Tensor | None = None,
        selected_page_token_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, max_selected = packed_cluster_indices.shape
        max_pages = view.max_pages_per_cluster
        device = packed_cluster_indices.device
        output_tensors = (
            selected_cluster_ids,
            selected_page_ids,
            selected_page_token_counts,
        )
        if any(output is None for output in output_tensors):
            if not all(output is None for output in output_tensors):
                raise ValueError("Exact selection outputs must be supplied together")
            selected_cluster_ids = torch.empty(
                (batch_size, num_kv_heads, max_selected),
                dtype=torch.int64,
                device=device,
            )
            selected_page_ids = torch.empty(
                (batch_size, num_kv_heads, max_selected, max_pages),
                dtype=torch.int64,
                device=device,
            )
            selected_page_token_counts = torch.empty(
                selected_page_ids.shape, dtype=torch.int32, device=device
            )

        assert selected_cluster_ids is not None
        assert selected_page_ids is not None
        assert selected_page_token_counts is not None
        if view.arena is None or max_selected == 0 or max_pages == 0:
            selected_cluster_ids.fill_(-1)
            selected_page_ids.fill_(-1)
            selected_page_token_counts.zero_()
            return selected_cluster_ids, selected_page_ids, selected_page_token_counts

        arena = view.arena
        if device.type == "cuda":
            gather_resident_exact_pages(
                cluster_ids=arena.cluster_ids,
                cluster_page_starts=arena.cluster_page_starts,
                cluster_page_counts=arena.cluster_page_counts,
                page_ids=arena.page_ids,
                page_token_counts=arena.page_token_counts,
                cluster_offsets=arena.cluster_offsets,
                page_offsets=arena.page_offsets,
                request_slot_ids=view.request_slot_ids,
                selected_indices=packed_cluster_indices,
                selected_mask=packed_cluster_mask,
                output_cluster_ids=selected_cluster_ids,
                output_page_ids=selected_page_ids,
                output_page_token_counts=selected_page_token_counts,
            )
            return selected_cluster_ids, selected_page_ids, selected_page_token_counts

        slots = view.request_slot_ids.clamp_min(0)
        head_indices = torch.arange(num_kv_heads, dtype=torch.int64, device=device)[
            None, :, None
        ].expand_as(packed_cluster_indices)
        valid_clusters = packed_cluster_mask & (
            view.request_slot_ids[:, None, None] >= 0
        )
        request_cluster_offsets = arena.cluster_offsets.index_select(0, slots)
        absolute_cluster_indices = (
            request_cluster_offsets[:, None, None] + packed_cluster_indices
        )
        absolute_cluster_indices.clamp_(min=0, max=arena.cluster_ids.shape[1] - 1)
        gathered_cluster_ids = arena.cluster_ids[head_indices, absolute_cluster_indices]
        valid_clusters &= gathered_cluster_ids >= 0
        selected_cluster_ids.copy_(gathered_cluster_ids)
        selected_cluster_ids.masked_fill_(~valid_clusters, -1)

        page_starts = arena.cluster_page_starts[head_indices, absolute_cluster_indices]
        page_counts = arena.cluster_page_counts[head_indices, absolute_cluster_indices]
        page_offsets = torch.arange(max_pages, dtype=torch.int64, device=device)
        request_page_offsets = arena.page_offsets.index_select(0, slots)
        flat_page_indices = (
            request_page_offsets[:, None, None, None]
            + page_starts.unsqueeze(-1)
            + page_offsets
        )
        valid_pages = valid_clusters.unsqueeze(-1) & (
            page_offsets < page_counts.unsqueeze(-1)
        )
        flat_page_indices.clamp_(min=0, max=arena.page_ids.shape[1] - 1)
        page_heads = head_indices.unsqueeze(-1).expand_as(flat_page_indices)
        selected_page_ids.copy_(arena.page_ids[page_heads, flat_page_indices])
        selected_page_token_counts.copy_(
            arena.page_token_counts[page_heads, flat_page_indices]
        )
        selected_page_ids.masked_fill_(~valid_pages, -1)
        selected_page_token_counts.masked_fill_(~valid_pages, 0)

        return selected_cluster_ids, selected_page_ids, selected_page_token_counts

    @staticmethod
    def _build_resident_estimation_selection(
        view: RetroSpecResidentBatchView,
        packed_indices: torch.Tensor,
        packed_mask: torch.Tensor,
        head_size: int,
        dtype: torch.dtype,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, max_selected_clusters = packed_indices.shape
        output_shape = (
            batch_size,
            num_kv_heads,
            max_selected_clusters,
            head_size,
        )
        output_tensors = (keys, values, counts)
        if any(output is None for output in output_tensors):
            if not all(output is None for output in output_tensors):
                raise ValueError("Estimation outputs must be supplied together")
            keys = torch.empty(output_shape, dtype=dtype, device=packed_indices.device)
            values = torch.empty_like(keys)
            counts = torch.empty(
                output_shape[:-1], dtype=torch.int32, device=packed_indices.device
            )

        assert keys is not None
        assert values is not None
        assert counts is not None
        if view.arena is None or max_selected_clusters == 0:
            keys.zero_()
            values.zero_()
            counts.zero_()
            return keys, values, counts

        arena = view.arena
        if packed_indices.device.type == "cuda":
            gather_resident_estimation(
                cluster_keys=arena.cluster_keys,
                cluster_values=arena.cluster_values,
                cluster_token_counts=arena.cluster_token_counts,
                cluster_offsets=arena.cluster_offsets,
                request_slot_ids=view.request_slot_ids,
                selected_indices=packed_indices,
                selected_mask=packed_mask,
                output_keys=keys,
                output_values=values,
                output_token_counts=counts,
            )
            return keys, values, counts

        slots = view.request_slot_ids.clamp_min(0)
        head_indices = torch.arange(
            num_kv_heads, dtype=torch.int64, device=packed_indices.device
        )[None, :, None].expand_as(packed_indices)
        valid = packed_mask & (view.request_slot_ids[:, None, None] >= 0)
        request_cluster_offsets = arena.cluster_offsets.index_select(0, slots)
        absolute_indices = request_cluster_offsets[:, None, None] + packed_indices
        absolute_indices.clamp_(min=0, max=arena.cluster_ids.shape[1] - 1)
        keys.copy_(arena.cluster_keys[head_indices, absolute_indices])
        values.copy_(arena.cluster_values[head_indices, absolute_indices])
        counts.copy_(arena.cluster_token_counts[head_indices, absolute_indices])
        keys.masked_fill_(~valid.unsqueeze(-1), 0.0)
        values.masked_fill_(~valid.unsqueeze(-1), 0.0)
        counts.masked_fill_(~valid, 0)

        return keys, values, counts

    def _prepare_plan_step(
        self,
        layer_name: str,
        plan_slot: int,
        seq_lens: torch.Tensor,
        indexed_starts: torch.Tensor,
        indexed_ends: torch.Tensor,
        indexed_requests: torch.Tensor,
        max_num_tokens: int,
        view: RetroSpecResidentBatchView,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[
        RetroSpecRankedSelectionPlan,
        _SelectionStepWorkspace,
        _SelectionPlanTable,
    ]:
        plan, output_workspace, table = self._get_selection_plan_step(
            layer_name=layer_name,
            plan_slot=plan_slot,
            view=view,
            batch_size=seq_lens.shape[0],
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=seq_lens.device,
        )

        emit_primary_exact_token_plan(
            seq_lens=seq_lens,
            indexed_starts=indexed_starts,
            indexed_ends=indexed_ends,
            indexed_requests=indexed_requests,
            num_kv_heads=num_kv_heads,
            max_num_tokens=max_num_tokens,
            block_size=self.block_size,
            num_recent_blocks=self.num_recent_blocks,
            output_indices=plan.primary_exact_token_indices,
            output_mask=plan.primary_exact_token_mask,
        )
        return plan, output_workspace, table

    def _publish_plan_step(
        self,
        layer_name: str,
        plan_slot: int,
        active_mask: torch.Tensor,
        table: _SelectionPlanTable,
    ) -> None:
        table.valid_rows[plan_slot, : active_mask.shape[0]].copy_(active_mask)
        self._selection_plan_written_layers.add(layer_name)

    def _make_reference_plan(
        self,
        layer_name: str,
        plan_slot: int,
        active_mask: torch.Tensor,
        forced_exact_mask: torch.Tensor,
        cluster_zones: _PackedClusterZones,
        ranked_indices: torch.Tensor,
        candidate_counts: torch.Tensor,
        sparse_attn: torch.Tensor,
        expanded_attn: torch.Tensor,
        view: RetroSpecResidentBatchView,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[RetroSpecTokenSelectionPlan, _SelectionStepWorkspace]:
        ranked_plan, output_workspace, table = self._get_selection_plan_step(
            layer_name=layer_name,
            plan_slot=plan_slot,
            view=view,
            batch_size=forced_exact_mask.shape[0],
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=forced_exact_mask.device,
        )
        if ranked_indices.shape != ranked_plan.ranked_cluster_indices.shape:
            raise ValueError("Reference ranking does not match the plan journal")
        ranked_plan.ranked_cluster_indices.copy_(ranked_indices)
        ranked_plan.candidate_counts.copy_(candidate_counts)

        per_head_forced_exact = forced_exact_mask.unsqueeze(1).expand(
            -1, num_kv_heads, -1
        )
        packed_indices, packed_mask = self._pack_bounded_mask_indices(
            per_head_forced_exact,
            ranked_plan.primary_exact_token_indices.shape[-1],
        )
        ranked_plan.primary_exact_token_indices.zero_()
        ranked_plan.primary_exact_token_mask.zero_()
        packed_width = packed_indices.shape[-1]
        ranked_plan.primary_exact_token_indices[..., :packed_width].copy_(
            packed_indices
        )
        ranked_plan.primary_exact_token_mask[..., :packed_width].copy_(packed_mask)

        ranked_plan.request_slot_ids.copy_(view.request_slot_ids)
        if view.arena is None:
            ranked_plan.request_slot_generations.fill_(-1)
        else:
            valid_slots = view.request_slot_ids >= 0
            safe_slots = view.request_slot_ids.clamp_min(0)
            ranked_plan.request_slot_generations.copy_(
                view.arena.generations.index_select(0, safe_slots)
            )
            ranked_plan.request_slot_generations.masked_fill_(~valid_slots, -1)

        sparse_exact = cluster_zones.sparse_retrieval_indices.masked_fill(
            ~cluster_zones.sparse_retrieval_mask, -1
        )
        sparse_estimation = cluster_zones.sparse_estimation_indices.masked_fill(
            ~cluster_zones.sparse_estimation_mask, -1
        )
        expanded_exact = cluster_zones.expanded_retrieval_indices.masked_fill(
            ~cluster_zones.expanded_retrieval_mask, -1
        )
        expanded_estimation = cluster_zones.expanded_estimation_indices.masked_fill(
            ~cluster_zones.expanded_estimation_mask, -1
        )
        ranked_plan.sparse_attn.copy_(sparse_attn)
        ranked_plan.expanded_attn.copy_(expanded_attn)
        self._publish_plan_step(layer_name, plan_slot, active_mask, table)
        plan = RetroSpecTokenSelectionPlan(
            layer_name=layer_name,
            request_slot_ids=ranked_plan.request_slot_ids,
            request_slot_generations=ranked_plan.request_slot_generations,
            primary_exact_token_indices=ranked_plan.primary_exact_token_indices,
            primary_exact_token_mask=ranked_plan.primary_exact_token_mask,
            sparse_exact_cluster_indices=sparse_exact,
            sparse_estimation_cluster_indices=sparse_estimation,
            expanded_exact_cluster_indices=expanded_exact,
            expanded_estimation_cluster_indices=expanded_estimation,
            sparse_attn=ranked_plan.sparse_attn,
            expanded_attn=ranked_plan.expanded_attn,
        )
        return plan, output_workspace

    @staticmethod
    def _flatten_plan_rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten(0, 1)

    def _materialize_estimation_selection(
        self,
        layer_name: str,
        cluster_indices: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
        head_size: int,
        dtype: torch.dtype,
        plan_row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_rows = (
            cluster_indices.shape[0]
            if plan_row_indices is None
            else plan_row_indices.shape[0]
        )
        num_kv_heads = cluster_indices.shape[1]
        estimation_width = cluster_indices.shape[2]
        workspace = self._get_indexed_verification_workspace(
            pair_capacity=num_rows,
            num_kv_heads=num_kv_heads,
            exact_width=0,
            estimation_width=estimation_width,
            head_size=head_size,
            dtype=dtype,
            device=cluster_indices.device,
        )

        cluster_shape = (num_rows, num_kv_heads, estimation_width)
        summary_shape = (*cluster_shape, head_size)
        num_clusters = prod(cluster_shape)
        num_summary_items = prod(summary_shape)
        selected_indices = workspace.estimation_cluster_indices.view(-1)[
            :num_clusters
        ].view(cluster_shape)
        selected_mask = workspace.estimation_cluster_mask.view(-1)[:num_clusters].view(
            cluster_shape
        )
        packed_slots = workspace.request_slot_ids[:num_rows]
        packed_generations = workspace.request_slot_generations[:num_rows]

        if plan_row_indices is None:
            selected_indices.copy_(cluster_indices)
            packed_slots.copy_(request_slot_ids)
            packed_generations.copy_(request_slot_generations)
        else:
            torch.index_select(
                cluster_indices,
                0,
                plan_row_indices,
                out=selected_indices,
            )
            request_rows = plan_row_indices.remainder(request_slot_ids.shape[0])
            torch.index_select(
                request_slot_ids,
                0,
                request_rows,
                out=packed_slots,
            )
            torch.index_select(
                request_slot_generations,
                0,
                request_rows,
                out=packed_generations,
            )

        torch.ge(selected_indices, 0, out=selected_mask)
        view = self._gpu_index_residency.get_active_view(
            layer_name,
            self._proposal_request_ids,
            cluster_indices.device,
        )
        arena = view.arena
        if arena is not None:
            valid_slots = packed_slots >= 0
            safe_slots = packed_slots.clamp_min(0)
            current_generations = arena.generations.index_select(0, safe_slots)
            descriptors_valid = ~valid_slots | (
                current_generations == packed_generations
            )
            torch._assert_async(
                descriptors_valid.all(),
                f"Stale RetroSpec request descriptor for layer {layer_name!r}",
            )

        row_view = RetroSpecResidentBatchView(
            arena=arena,
            request_slot_ids=packed_slots,
            max_num_clusters=view.max_num_clusters,
            max_pages_per_cluster=view.max_pages_per_cluster,
            max_num_pages=view.max_num_pages,
        )
        keys = workspace.keys.view(-1)[:num_summary_items].view(summary_shape)
        values = workspace.values.view(-1)[:num_summary_items].view(summary_shape)
        counts = workspace.token_counts.view(-1)[:num_clusters].view(cluster_shape)
        self._build_resident_estimation_selection(
            row_view,
            selected_indices,
            selected_mask,
            head_size,
            dtype,
            keys,
            values,
            counts,
        )
        return selected_indices, keys, values, counts

    def get_indexed_selection(
        self,
        layer_name: str,
        level: RetroSpecAttentionLevel,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
        prepared_exact: _PreparedIndexedVerificationLayer | None = None,
    ) -> RetroSpecIndexedTokenAttentionSelection:
        table = self._selection_plan_tables.get(layer_name)
        if table is None:
            raise RuntimeError(
                f"No draft selection plan exists for layer {layer_name!r}"
            )
        if request_indices.shape != token_indices.shape:
            raise ValueError("Indexed plan indices must have equal shapes")
        if request_indices.device != table.valid_rows.device:
            raise ValueError("Indexed request indices use the wrong device")
        if token_indices.device != table.valid_rows.device:
            raise ValueError("Indexed token indices use the wrong device")

        if level == RetroSpecAttentionLevel.SPARSE:
            exact_width = table.sparse_retrieval_width
            expanded = False
            attention_mass = table.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            exact_width = table.expanded_retrieval_width
            expanded = True
            attention_mass = table.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        num_pairs = request_indices.numel()
        num_kv_heads = table.ranked_cluster_indices.shape[2]
        estimation_width = table.sparse_estimation_width
        workspace = self._get_indexed_verification_workspace(
            pair_capacity=num_pairs,
            num_kv_heads=num_kv_heads,
            exact_width=0 if prepared_exact is not None else exact_width,
            estimation_width=estimation_width,
            head_size=table.head_size,
            dtype=table.dtype,
            device=request_indices.device,
        )

        plan_rows = workspace.plan_row_indices[:num_pairs]
        plan_valid_rows = workspace.plan_valid_rows[:num_pairs]
        packed_slots = workspace.request_slot_ids[:num_pairs]
        packed_generations = workspace.request_slot_generations[:num_pairs]
        exact_shape = (
            (num_pairs, num_kv_heads, 0)
            if prepared_exact is not None
            else (num_pairs, num_kv_heads, exact_width)
        )
        estimation_shape = (num_pairs, num_kv_heads, estimation_width)
        num_exact = prod(exact_shape)
        num_estimation = prod(estimation_shape)
        packed_exact = workspace.exact_cluster_indices.view(-1)[:num_exact].view(
            exact_shape
        )
        packed_estimation = workspace.estimation_cluster_indices.view(-1)[
            :num_estimation
        ].view(estimation_shape)
        packed_estimation_mask = workspace.estimation_cluster_mask.view(-1)[
            :num_estimation
        ].view(estimation_shape)
        packed_attention = workspace.attention_mass[:num_pairs]

        flatten = self._flatten_plan_rows
        pack_ranked_verification_plan(
            request_indices=request_indices,
            token_indices=token_indices,
            valid_rows=table.valid_rows,
            request_slot_ids=table.request_slot_ids,
            request_slot_generations=table.request_slot_generations,
            ranked_cluster_indices=flatten(table.ranked_cluster_indices),
            candidate_counts=flatten(table.candidate_counts),
            attention_mass=attention_mass.view(-1),
            output_plan_row_indices=plan_rows,
            output_plan_valid_rows=plan_valid_rows,
            output_request_slot_ids=packed_slots,
            output_request_slot_generations=packed_generations,
            output_exact_cluster_indices=packed_exact,
            output_estimation_cluster_indices=packed_estimation,
            output_estimation_cluster_mask=packed_estimation_mask,
            output_attention_mass=packed_attention,
            retrieval_ratio=self.retrieval_ratio,
            estimation_ratio=self.estimation_ratio,
            expanded=expanded,
        )

        if prepared_exact is not None:
            if prepared_exact.layer_name != layer_name:
                raise RuntimeError("Prepared verification layer does not match")
            plan_rows = prepared_exact.plan_row_indices
            plan_valid_rows = prepared_exact.plan_valid_rows
            packed_slots = prepared_exact.request_slot_ids
            packed_generations = prepared_exact.request_slot_generations
            packed_exact = prepared_exact.exact_cluster_indices
            packed_attention = prepared_exact.attention_mass

        view = self._gpu_index_residency.get_active_view(
            layer_name, self._proposal_request_ids, request_indices.device
        )
        summary_shape = (*estimation_shape, table.head_size)
        num_summary = prod(summary_shape)
        keys = workspace.keys.view(-1)[:num_summary].view(summary_shape)
        values = workspace.values.view(-1)[:num_summary].view(summary_shape)
        counts = workspace.token_counts.view(-1)[:num_estimation].view(estimation_shape)
        if view.arena is None:
            keys.zero_()
            values.zero_()
            counts.zero_()
        else:
            gather_resident_estimation(
                cluster_keys=view.arena.cluster_keys,
                cluster_values=view.arena.cluster_values,
                cluster_token_counts=view.arena.cluster_token_counts,
                cluster_offsets=view.arena.cluster_offsets,
                request_slot_ids=packed_slots,
                selected_indices=packed_estimation,
                selected_mask=packed_estimation_mask,
                output_keys=keys,
                output_values=values,
                output_token_counts=counts,
            )
        return RetroSpecIndexedTokenAttentionSelection(
            layer_name=layer_name,
            plan_row_indices=plan_rows,
            plan_valid_rows=plan_valid_rows,
            request_slot_ids=packed_slots,
            request_slot_generations=packed_generations,
            primary_exact_token_indices=flatten(table.primary_exact_token_indices),
            primary_exact_token_mask=flatten(table.primary_exact_token_mask),
            exact_cluster_indices=packed_exact,
            estimation_cluster_indices=packed_estimation,
            estimation_keys=keys,
            estimation_values=values,
            estimation_token_counts=counts,
            attention_mass=packed_attention,
        )

    def _get_indexed_verification_exact_workspace(
        self,
        layer_name: str,
        level: RetroSpecAttentionLevel,
        pair_capacity: int,
        num_kv_heads: int,
        exact_width: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _IndexedVerificationWorkspace:
        key = (layer_name, level)
        workspace = self._indexed_verification_exact_workspaces.get(key)
        if workspace is not None and workspace.matches(
            pair_capacity=pair_capacity,
            num_kv_heads=num_kv_heads,
            exact_width=exact_width,
            estimation_width=0,
            head_size=head_size,
            dtype=dtype,
            device=device,
        ):
            return workspace
        if workspace is not None:
            pair_capacity = max(pair_capacity, workspace.keys.shape[0])
            exact_width = max(exact_width, workspace.exact_cluster_indices.shape[2])
        workspace = _IndexedVerificationWorkspace.allocate(
            pair_capacity=pair_capacity,
            num_kv_heads=num_kv_heads,
            exact_width=exact_width,
            estimation_width=0,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        self._indexed_verification_exact_workspaces[key] = workspace
        return workspace

    def _prepare_indexed_verification_exact(
        self,
        layer_name: str,
        level: RetroSpecAttentionLevel,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> _PreparedIndexedVerificationLayer:
        table = self._selection_plan_tables.get(layer_name)
        if table is None:
            raise RuntimeError(
                f"No draft selection plan exists for layer {layer_name!r}"
            )
        if level == RetroSpecAttentionLevel.SPARSE:
            exact_width = table.sparse_retrieval_width
            expanded = False
            attention_mass = table.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            exact_width = table.expanded_retrieval_width
            expanded = True
            attention_mass = table.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        num_pairs = request_indices.numel()
        num_kv_heads = table.ranked_cluster_indices.shape[2]
        workspace = self._get_indexed_verification_exact_workspace(
            layer_name=layer_name,
            level=level,
            pair_capacity=num_pairs,
            num_kv_heads=num_kv_heads,
            exact_width=exact_width,
            head_size=table.head_size,
            dtype=table.dtype,
            device=request_indices.device,
        )
        plan_rows = workspace.plan_row_indices[:num_pairs]
        plan_valid_rows = workspace.plan_valid_rows[:num_pairs]
        packed_slots = workspace.request_slot_ids[:num_pairs]
        packed_generations = workspace.request_slot_generations[:num_pairs]
        packed_attention = workspace.attention_mass[:num_pairs]
        exact_items = num_pairs * num_kv_heads * exact_width
        packed_exact = workspace.exact_cluster_indices.view(-1)[:exact_items].view(
            num_pairs, num_kv_heads, exact_width
        )
        empty_items = workspace.estimation_cluster_indices.view(-1)[:0].view(
            num_pairs, num_kv_heads, 0
        )
        empty_mask = workspace.estimation_cluster_mask.view(-1)[:0].view(
            num_pairs, num_kv_heads, 0
        )
        pack_ranked_verification_exact_plan(
            request_indices=request_indices,
            token_indices=token_indices,
            valid_rows=table.valid_rows,
            request_slot_ids=table.request_slot_ids,
            request_slot_generations=table.request_slot_generations,
            ranked_cluster_indices=self._flatten_plan_rows(
                table.ranked_cluster_indices
            ),
            candidate_counts=self._flatten_plan_rows(table.candidate_counts),
            attention_mass=attention_mass.view(-1),
            output_plan_row_indices=plan_rows,
            output_plan_valid_rows=plan_valid_rows,
            output_request_slot_ids=packed_slots,
            output_request_slot_generations=packed_generations,
            output_exact_cluster_indices=packed_exact,
            output_attention_mass=packed_attention,
            empty_estimation_cluster_indices=empty_items,
            empty_estimation_cluster_mask=empty_mask,
            retrieval_ratio=self.retrieval_ratio,
            estimation_ratio=self.estimation_ratio,
            expanded=expanded,
        )
        return _PreparedIndexedVerificationLayer(
            layer_name=layer_name,
            plan_row_indices=plan_rows,
            plan_valid_rows=plan_valid_rows,
            request_slot_ids=packed_slots,
            request_slot_generations=packed_generations,
            exact_cluster_indices=packed_exact,
            attention_mass=packed_attention,
        )

    def begin_indexed_verification_transaction(
        self,
        level: RetroSpecAttentionLevel,
        layer_names: Sequence[str],
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None:
        if self._indexed_verification_transaction is not None:
            raise RuntimeError("Indexed verification transaction is already active")
        layers = tuple(layer_names)
        prepared = {
            layer_name: self._prepare_indexed_verification_exact(
                layer_name, level, request_indices, token_indices
            )
            for layer_name in layers
        }
        requests: list[RetroSpecVerificationResolveRequest] = []
        for layer_name in layers:
            packed = prepared[layer_name]
            view = self._gpu_index_residency.get_active_view(
                layer_name,
                self._proposal_request_ids,
                request_indices.device,
            )
            if view.arena is None or packed.exact_cluster_indices.shape[2] == 0:
                continue
            requests.append(
                RetroSpecVerificationResolveRequest(
                    layer_name=layer_name,
                    selected_cluster_indices=packed.exact_cluster_indices,
                    plan_valid_rows=packed.plan_valid_rows,
                    request_slot_ids=packed.request_slot_ids,
                    request_slot_generations=packed.request_slot_generations,
                    arena=view.arena,
                    max_pages_per_cluster=view.max_pages_per_cluster,
                )
            )
        resolved = self.cluster_store.resolve_verification_cluster_batch(requests)
        for layer_name, pages in resolved.items():
            prepared[layer_name].resolved_pages = pages
        self._indexed_verification_transaction = _IndexedVerificationTransaction(
            level=level,
            layers=layers,
            prepared=prepared,
        )

    def consume_indexed_verification_layer(
        self,
        layer_name: str,
        request_indices: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[
        RetroSpecIndexedTokenAttentionSelection,
        RetroSpecCompactVerificationResolvedPages | None,
    ]:
        transaction = self._indexed_verification_transaction
        if transaction is None:
            raise RuntimeError("No indexed verification transaction is active")
        if transaction.next_layer_index >= len(transaction.layers):
            raise RuntimeError("Indexed verification transaction is exhausted")
        expected_layer = transaction.layers[transaction.next_layer_index]
        if expected_layer != layer_name:
            raise RuntimeError(
                "Verification layer order differs from the installed model"
            )
        packed = transaction.prepared[layer_name]
        selection = self.get_indexed_selection(
            layer_name=layer_name,
            level=transaction.level,
            request_indices=request_indices,
            token_indices=token_indices,
            prepared_exact=packed,
        )
        transaction.next_layer_index += 1
        return selection, packed.resolved_pages

    def end_indexed_verification_transaction(self) -> None:
        transaction = self._indexed_verification_transaction
        if transaction is None:
            return
        self._indexed_verification_transaction = None
        error: BaseException | None = None
        for packed in transaction.prepared.values():
            pages = packed.resolved_pages
            if pages is None:
                continue
            try:
                self.cluster_store.submit_verification_miss_admission(
                    pages.miss_admission
                )
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                pages.read_lease.release()
        if error is not None:
            raise error

    def materialize_indexed_reference(
        self, selection: RetroSpecIndexedTokenAttentionSelection
    ) -> RetroSpecTokenAttentionSelection:
        rows = selection.plan_row_indices

        if not selection.plan_valid_rows.all().item():
            raise RuntimeError(
                f"A draft selection plan is missing for layer {selection.layer_name!r}"
            )

        def gather(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.index_select(0, rows)

        primary_indices = gather(selection.primary_exact_token_indices)
        primary_mask = gather(selection.primary_exact_token_mask)
        cluster_indices = selection.exact_cluster_indices
        estimation_indices = selection.estimation_cluster_indices
        estimation_keys = selection.estimation_keys
        estimation_values = selection.estimation_values
        estimation_counts = selection.estimation_token_counts
        attention_mass = selection.attention_mass
        request_slot_ids = selection.request_slot_ids
        request_slot_generations = selection.request_slot_generations
        cluster_ids, page_ids, page_counts = self._materialize_logical_selection(
            layer_name=selection.layer_name,
            cluster_indices=cluster_indices,
            request_slot_ids=request_slot_ids,
            request_slot_generations=request_slot_generations,
        )
        plan = RetroSpecTokenSelectionPlan(
            layer_name=selection.layer_name,
            request_slot_ids=request_slot_ids,
            request_slot_generations=request_slot_generations,
            primary_exact_token_indices=primary_indices,
            primary_exact_token_mask=primary_mask,
            sparse_exact_cluster_indices=cluster_indices,
            sparse_estimation_cluster_indices=estimation_indices,
            expanded_exact_cluster_indices=cluster_indices,
            expanded_estimation_cluster_indices=estimation_indices,
            sparse_attn=attention_mass,
            expanded_attn=attention_mass,
        )
        return self._selection_from_logical_pages(
            plan=plan,
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            page_token_counts=page_counts,
            estimation_keys=estimation_keys,
            estimation_values=estimation_values,
            estimation_token_counts=estimation_counts,
            attention_mass=attention_mass,
        )
