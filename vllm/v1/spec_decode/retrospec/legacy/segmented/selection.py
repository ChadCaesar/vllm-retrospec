# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.legacy.cluster_store import (
    RetroSpecCompactResolvedClusterPages,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecRankedDraftResolvedClusters,
    RetroSpecResidentPrefetchInput,
)
from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecResidentBatchView,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.types import (
    RetroSpecIndexedTokenAttentionSelection,
    RetroSpecRankedDraftAttentionSelection,
    RetroSpecRankedSelectionPlan,
    RetroSpecTokenAttentionSelection,
    RetroSpecTokenSelectionPlan,
    _SelectionStepWorkspace,
)


class _RetroSpecSegmentedTokenIndexSelectionMixin:
    def resolve_indexed_verification_pages(
        self, selection: RetroSpecIndexedTokenAttentionSelection
    ) -> RetroSpecCompactVerificationResolvedPages | None:
        view = self._gpu_index_residency.get_active_view(
            selection.layer_name,
            self._proposal_request_ids,
            selection.exact_cluster_indices.device,
        )
        if view.arena is None:
            return None
        return self.cluster_store.resolve_verification_cluster_blocks(
            layer_name=selection.layer_name,
            selected_cluster_indices=selection.exact_cluster_indices,
            plan_valid_rows=selection.plan_valid_rows,
            request_slot_ids=selection.request_slot_ids,
            request_slot_generations=selection.request_slot_generations,
            arena=view.arena,
            max_pages_per_cluster=view.max_pages_per_cluster,
        )

    def _gather_selected_tokens(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_kv_heads, max_num_tokens = token_indices.shape

        logical_block_ids = torch.div(
            token_indices,
            self.block_size,
            rounding_mode="floor",
        )
        block_offsets = token_indices % self.block_size

        expanded_block_table = block_table[:, None, :].expand(
            batch_size,
            num_kv_heads,
            -1,
        )
        physical_block_ids = expanded_block_table.gather(
            dim=2,
            index=logical_block_ids,
        ).to(torch.int64)

        head_ids = torch.arange(
            num_kv_heads,
            dtype=torch.int64,
            device=cache.device,
        )[None, :, None].expand(
            batch_size,
            num_kv_heads,
            max_num_tokens,
        )

        selected = cache[
            physical_block_ids,
            block_offsets,
            head_ids,
        ]
        selected.masked_fill_(~token_mask.unsqueeze(-1), 0.0)
        return selected.contiguous()

    def _resolve_ranked_draft_clusters(
        self,
        plan: RetroSpecRankedSelectionPlan,
        output_workspace: _SelectionStepWorkspace,
        view: RetroSpecResidentBatchView,
        active_mask: torch.Tensor,
        ranked_values: torch.Tensor,
        ranked_indices: torch.Tensor,
        candidate_counts: torch.Tensor,
        plan_valid_rows: torch.Tensor,
        capture_request_descriptors: bool,
        emit_misses: bool,
    ) -> RetroSpecRankedDraftResolvedClusters:
        if view.arena is None:
            raise RuntimeError("Ranked draft resolution requires a resident arena")

        return self.cluster_store.resolve_ranked_draft_clusters(
            layer_name=plan.layer_name,
            ranked_values=ranked_values,
            ranked_indices=ranked_indices,
            candidate_counts=candidate_counts,
            arena=view.arena,
            request_slot_ids=view.request_slot_ids,
            active_mask=active_mask,
            retrieval_ratio=self.retrieval_ratio,
            estimation_ratio=self.estimation_ratio,
            expanded_retrieval_width=plan.expanded_retrieval_width,
            max_pages_per_cluster=view.max_pages_per_cluster,
            plan_valid_rows=plan_valid_rows,
            output_request_slot_ids=plan.request_slot_ids,
            output_request_slot_generations=plan.request_slot_generations,
            cluster_handles=output_workspace.draft_exact_cluster_handles,
            resident_bucket_ids=output_workspace.draft_resident_bucket_ids,
            clustered_token_counts=output_workspace.draft_clustered_token_counts,
            attention_mass=output_workspace.draft_attention_mass,
            hit_attention_by_head=output_workspace.draft_hit_attention_by_head,
            selected_cluster_counts=output_workspace.draft_selected_cluster_counts,
            hit_cluster_counts=output_workspace.draft_hit_cluster_counts,
            miss_cluster_counts=output_workspace.draft_miss_cluster_counts,
            hit_gate_ready=output_workspace.draft_hit_gate_ready,
            miss_cluster_ids=output_workspace.draft_prefetch_miss_cluster_ids,
            miss_positions=output_workspace.draft_prefetch_miss_positions,
            miss_count=output_workspace.draft_prefetch_miss_count,
            sparse_attention=plan.sparse_attn,
            expanded_attention=plan.expanded_attn,
            capture_request_descriptors=capture_request_descriptors,
            emit_misses=emit_misses,
        )

    def _trace_draft_selection(
        self,
        request_ids: Sequence[str],
        query: torch.Tensor,
        active_mask: torch.Tensor,
        proposal_round: int,
        draft_step: int,
        snapshot: str,
        physical_source: str,
        index_revisions: Sequence[int],
        plan: RetroSpecRankedSelectionPlan,
        output_workspace: _SelectionStepWorkspace,
        candidate_counts: torch.Tensor,
    ) -> None:
        candidate_counts_i64 = plan.candidate_counts.to(torch.int64)
        retrieval_counts = torch.ceil(
            candidate_counts_i64.float() * self.retrieval_ratio
        ).to(torch.int64)
        retrieval_counts = torch.minimum(retrieval_counts, candidate_counts_i64)
        estimation_counts = torch.ceil(
            candidate_counts_i64.float() * self.estimation_ratio
        ).to(torch.int64)
        estimation_counts = torch.minimum(
            estimation_counts, candidate_counts_i64 - retrieval_counts
        )
        zero_counts = torch.zeros_like(retrieval_counts)
        sparse_exact, sparse_exact_mask = self._slice_rank_range(
            plan.ranked_cluster_indices,
            zero_counts,
            retrieval_counts,
            plan.sparse_retrieval_width,
        )
        sparse_estimation, sparse_estimation_mask = self._slice_rank_range(
            plan.ranked_cluster_indices,
            retrieval_counts,
            retrieval_counts + estimation_counts,
            plan.sparse_estimation_width,
        )
        self.selection_provenance.record_draft_selection(
            request_ids=request_ids,
            layer_name=plan.layer_name,
            proposal_round=proposal_round,
            draft_step=draft_step,
            snapshot=snapshot,
            physical_source=physical_source,
            index_revisions=index_revisions,
            active_mask=active_mask,
            query=query,
            request_slot_ids=plan.request_slot_ids,
            request_slot_generations=plan.request_slot_generations,
            sparse_exact_cluster_indices=sparse_exact.masked_fill(
                ~sparse_exact_mask, -1
            ),
            sparse_estimation_cluster_indices=sparse_estimation.masked_fill(
                ~sparse_estimation_mask, -1
            ),
            exact_cluster_handles=output_workspace.draft_exact_cluster_handles,
            candidate_counts=candidate_counts,
            selected_cluster_counts=(output_workspace.draft_selected_cluster_counts),
            hit_cluster_counts=output_workspace.draft_hit_cluster_counts,
            miss_cluster_counts=output_workspace.draft_miss_cluster_counts,
            hit_gate_ready=output_workspace.draft_hit_gate_ready,
        )

    def _materialize_draft_selection(
        self,
        plan: RetroSpecTokenSelectionPlan | RetroSpecRankedSelectionPlan,
        output_workspace: _SelectionStepWorkspace | None,
        view: RetroSpecResidentBatchView,
        active_mask: torch.Tensor,
        ranked_values: torch.Tensor | None = None,
        ranked_indices: torch.Tensor | None = None,
        candidate_counts: torch.Tensor | None = None,
        plan_valid_rows: torch.Tensor | None = None,
        capture_request_descriptors: bool = False,
        request_ids: Sequence[str] = (),
        query: torch.Tensor | None = None,
        proposal_round: int = 0,
        draft_step: int = 0,
        index_revisions: Sequence[int] = (),
    ) -> RetroSpecTokenAttentionSelection | RetroSpecRankedDraftAttentionSelection:
        """Use resident retrieval clusters and estimate selected cache misses."""
        native_only = (
            view.arena is None
            or output_workspace is None
            or output_workspace.draft_exact_cluster_handles.device.type != "cuda"
        )
        if native_only:
            if not isinstance(plan, RetroSpecTokenSelectionPlan):
                raise RuntimeError("Ranked DRAFT selection requires a resident arena")
            if self.selection_provenance.enabled:
                if query is None:
                    raise RuntimeError("Selection provenance requires the draft query")
                num_requests, num_kv_heads = query.shape[:2]
                if candidate_counts is None:
                    candidate_counts = torch.zeros(
                        num_requests,
                        num_kv_heads,
                        dtype=torch.int32,
                        device=query.device,
                    )
                selected_counts = (plan.sparse_exact_cluster_indices >= 0).sum(
                    dim=2, dtype=torch.int32
                )
                zero_counts = torch.zeros_like(candidate_counts)
                self.selection_provenance.record_draft_selection(
                    request_ids=request_ids,
                    layer_name=plan.layer_name,
                    proposal_round=proposal_round,
                    draft_step=draft_step,
                    snapshot="used",
                    physical_source="native_only",
                    index_revisions=index_revisions,
                    active_mask=active_mask,
                    query=query,
                    request_slot_ids=plan.request_slot_ids,
                    request_slot_generations=plan.request_slot_generations,
                    sparse_exact_cluster_indices=plan.sparse_exact_cluster_indices,
                    sparse_estimation_cluster_indices=(
                        plan.sparse_estimation_cluster_indices
                    ),
                    exact_cluster_handles=torch.full_like(
                        plan.sparse_exact_cluster_indices, -1, dtype=torch.int64
                    ),
                    candidate_counts=candidate_counts,
                    selected_cluster_counts=selected_counts,
                    hit_cluster_counts=zero_counts,
                    miss_cluster_counts=zero_counts,
                    hit_gate_ready=torch.zeros_like(candidate_counts, dtype=torch.bool),
                )
            return self._materialize_token_selection(
                plan,
                RetroSpecAttentionLevel.SPARSE,
            )
        if not isinstance(plan, RetroSpecRankedSelectionPlan):
            raise RuntimeError("Resident DRAFT selection requires a ranked plan")
        if ranked_values is None:
            raise RuntimeError("CUDA draft selection requires ranked scores")
        if ranked_indices is None:
            raise RuntimeError("CUDA draft selection requires ranked indices")
        if candidate_counts is None:
            raise RuntimeError("CUDA draft selection requires candidate counts")
        if plan_valid_rows is None:
            raise RuntimeError("CUDA draft selection requires plan valid rows")
        if self.selection_provenance.enabled:
            if query is None:
                raise RuntimeError("Selection provenance requires the draft query")
            if len(request_ids) != active_mask.shape[0]:
                raise RuntimeError(
                    "Selection provenance request IDs do not match the batch"
                )

        assert query is not None or not self.selection_provenance.enabled

        resolved_clusters = self._resolve_ranked_draft_clusters(
            plan,
            output_workspace,
            view,
            active_mask,
            ranked_values,
            ranked_indices,
            candidate_counts,
            plan_valid_rows,
            capture_request_descriptors,
            emit_misses=self.replay_mode == "ready_selected",
        )

        if self.replay_mode == "ready_selected":
            try:
                self._trace_draft_selection(
                    request_ids=request_ids,
                    query=query,
                    active_mask=active_mask,
                    proposal_round=proposal_round,
                    draft_step=draft_step,
                    snapshot="before_ready",
                    physical_source="resident",
                    index_revisions=index_revisions,
                    plan=plan,
                    output_workspace=output_workspace,
                    candidate_counts=candidate_counts,
                )
            except BaseException:
                resolved_clusters.read_lease.release()
                raise

            resolved_clusters.read_lease.release()
            prefetch = RetroSpecResidentPrefetchInput(
                layer_name=plan.layer_name,
                miss_cluster_ids=output_workspace.draft_prefetch_miss_cluster_ids,
                miss_positions=output_workspace.draft_prefetch_miss_positions,
                miss_count=output_workspace.draft_prefetch_miss_count,
                num_groups=(
                    output_workspace.draft_exact_cluster_handles.shape[0]
                    * output_workspace.draft_exact_cluster_handles.shape[1]
                ),
                num_ranks=output_workspace.draft_exact_cluster_handles.shape[2],
            )
            self.cluster_store.prefetch_resident_cluster_wave((prefetch,))
            self.cluster_store.synchronize_resident_prefetches((plan.layer_name,))
            resolved_clusters = self._resolve_ranked_draft_clusters(
                plan,
                output_workspace,
                view,
                active_mask,
                ranked_values,
                ranked_indices,
                candidate_counts,
                plan_valid_rows,
                False,
                emit_misses=False,
            )

        try:
            self._trace_draft_selection(
                request_ids=request_ids,
                query=query,
                active_mask=active_mask,
                proposal_round=proposal_round,
                draft_step=draft_step,
                snapshot="used",
                physical_source="resident",
                index_revisions=index_revisions,
                plan=plan,
                output_workspace=output_workspace,
                candidate_counts=candidate_counts,
            )
        except BaseException:
            resolved_clusters.read_lease.release()
            raise
        self._proposal_read_leases.append(resolved_clusters.read_lease)

        primary_token_counts = plan.primary_exact_token_mask.sum(
            dim=2,
            dtype=torch.int32,
        )
        exact_token_counts = (
            primary_token_counts + resolved_clusters.clustered_token_counts
        ).contiguous()

        assert view.arena is not None
        return RetroSpecRankedDraftAttentionSelection(
            plan=plan,
            arena=view.arena,
            resolved_clusters=resolved_clusters,
            exact_token_counts=exact_token_counts,
            attention_mass=resolved_clusters.attention_mass,
        )

    def configure_sparse_prefetch_wave(self, max_layers: int) -> None:
        self.cluster_store.configure_resident_prefetch_wave(max_layers)

    def build_sparse_verification_prefetch(
        self,
        selection: RetroSpecTokenAttentionSelection
        | RetroSpecRankedDraftAttentionSelection,
        active_mask: torch.Tensor,
    ) -> RetroSpecResidentPrefetchInput | None:
        """Build one layer record for the current draft prefetch wave."""
        if not self.cluster_store.pin_memory:
            return None

        cluster_ids = selection.prefetch_miss_cluster_ids
        positions = selection.prefetch_miss_positions
        count = selection.prefetch_miss_count
        if cluster_ids is None or positions is None or count is None:
            return None

        if active_mask.ndim != 1 or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a one-dimensional boolean tensor")
        if active_mask.device != cluster_ids.device:
            raise ValueError("active_mask and selection must use one device")
        if active_mask.numel() == 0:
            return None
        if selection.prefetch_num_groups % active_mask.shape[0] != 0:
            raise ValueError("active_mask does not match the prefetch layout")
        if cluster_ids.numel() == 0:
            return None

        return RetroSpecResidentPrefetchInput(
            layer_name=selection.plan.layer_name,
            miss_cluster_ids=cluster_ids,
            miss_positions=positions,
            miss_count=count,
            num_groups=selection.prefetch_num_groups,
            num_ranks=selection.prefetch_num_ranks,
            source="draft",
        )

    def submit_sparse_verification_prefetch_wave(
        self,
        records: Sequence[RetroSpecResidentPrefetchInput],
    ) -> bool:
        return self.cluster_store.prefetch_resident_cluster_wave(records)

    def flush_sparse_verification_prefetch(self) -> None:
        """Submit deferred resident commands before workspace reuse."""
        self.cluster_store.flush_resident_prefetch_commands()

    def prefetch_sparse_verification(
        self,
        selection: RetroSpecTokenAttentionSelection
        | RetroSpecRankedDraftAttentionSelection,
        active_mask: torch.Tensor,
    ) -> bool:
        """Compatibility wrapper for one layer's resident prefetch."""
        record = self.build_sparse_verification_prefetch(selection, active_mask)
        if record is None:
            return False
        return self.submit_sparse_verification_prefetch_wave((record,))

    def _materialize_logical_selection(
        self,
        layer_name: str,
        cluster_indices: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        view = self._gpu_index_residency.get_active_view(
            layer_name,
            self._proposal_request_ids,
            cluster_indices.device,
        )
        if request_slot_ids.shape != (cluster_indices.shape[0],):
            raise ValueError("Request slots do not match logical selection rows")
        if request_slot_generations.shape != request_slot_ids.shape:
            raise ValueError("Request generations do not match request slots")

        arena = view.arena
        if arena is not None:
            valid_slots = request_slot_ids >= 0
            safe_slots = request_slot_ids.clamp_min(0)
            current_generations = arena.generations.index_select(0, safe_slots)
            descriptor_valid = ~valid_slots | (
                current_generations == request_slot_generations
            )
            torch._assert_async(
                descriptor_valid.all(),
                f"Stale RetroSpec request descriptor for layer {layer_name!r}",
            )

        row_view = RetroSpecResidentBatchView(
            arena=arena,
            request_slot_ids=request_slot_ids,
            max_num_clusters=view.max_num_clusters,
            max_pages_per_cluster=view.max_pages_per_cluster,
            max_num_pages=view.max_num_pages,
        )
        return self._build_resident_exact_cluster_selection(
            row_view,
            cluster_indices.clamp_min(0),
            cluster_indices >= 0,
        )

    @staticmethod
    def _selection_from_logical_pages(
        plan: RetroSpecTokenSelectionPlan,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
        estimation_keys: torch.Tensor,
        estimation_values: torch.Tensor,
        estimation_token_counts: torch.Tensor,
        attention_mass: torch.Tensor,
    ) -> RetroSpecTokenAttentionSelection:
        primary_token_counts = plan.primary_exact_token_mask.sum(
            dim=2, dtype=torch.int32
        )
        clustered_token_counts = page_token_counts.sum(dim=(2, 3), dtype=torch.int32)
        exact_token_counts = (
            primary_token_counts + clustered_token_counts
        ).contiguous()
        return RetroSpecTokenAttentionSelection(
            exact_cluster_ids=cluster_ids,
            exact_page_ids=page_ids,
            exact_page_token_counts=page_token_counts,
            exact_token_counts=exact_token_counts,
            estimation_keys=estimation_keys,
            estimation_values=estimation_values,
            estimation_token_counts=estimation_token_counts,
            attention_mass=attention_mass,
            plan=plan,
            resolved_pages=None,
        )

    def _materialize_token_selection(
        self,
        plan: RetroSpecTokenSelectionPlan,
        level: RetroSpecAttentionLevel,
    ) -> RetroSpecTokenAttentionSelection:
        if level == RetroSpecAttentionLevel.SPARSE:
            cluster_indices = plan.sparse_exact_cluster_indices
            estimation_cluster_indices = plan.sparse_estimation_cluster_indices
            attention_mass = plan.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            cluster_indices = plan.expanded_exact_cluster_indices
            estimation_cluster_indices = plan.expanded_estimation_cluster_indices
            attention_mass = plan.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        table = self._selection_plan_tables[plan.layer_name]
        _, estimation_keys, estimation_values, estimation_token_counts = (
            self._materialize_estimation_selection(
                layer_name=plan.layer_name,
                cluster_indices=estimation_cluster_indices,
                request_slot_ids=plan.request_slot_ids,
                request_slot_generations=plan.request_slot_generations,
                head_size=table.head_size,
                dtype=table.dtype,
            )
        )
        cluster_ids, page_ids, page_token_counts = self._materialize_logical_selection(
            layer_name=plan.layer_name,
            cluster_indices=cluster_indices,
            request_slot_ids=plan.request_slot_ids,
            request_slot_generations=plan.request_slot_generations,
        )
        return self._selection_from_logical_pages(
            plan=plan,
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            page_token_counts=page_token_counts,
            estimation_keys=estimation_keys,
            estimation_values=estimation_values,
            estimation_token_counts=estimation_token_counts,
            attention_mass=attention_mass,
        )

    def materialize_exact_reference(
        self,
        selection: RetroSpecTokenAttentionSelection,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Materialize exact KV for CPU tests and unsupported CUDA dtypes."""
        plan = selection.plan

        primary_keys = self._gather_selected_tokens(
            key_cache,
            block_table,
            plan.primary_exact_token_indices,
            plan.primary_exact_token_mask,
        )
        primary_values = self._gather_selected_tokens(
            value_cache,
            block_table,
            plan.primary_exact_token_indices,
            plan.primary_exact_token_mask,
        )

        batch_size, num_kv_heads = plan.primary_exact_token_indices.shape[:2]
        head_size = key_cache.shape[3]

        resolved_pages = selection.resolved_pages
        if isinstance(resolved_pages, RetroSpecCompactResolvedClusterPages):
            page_ids = resolved_pages.resident_page_ids
            page_token_counts = resolved_pages.page_token_counts
            page_size = resolved_pages.resident_key_pages.shape[1]
            safe_page_ids = page_ids.clamp_min(0)
            clustered_keys = resolved_pages.resident_key_pages[safe_page_ids]
            clustered_values = resolved_pages.resident_value_pages[safe_page_ids]
            page_offsets = torch.arange(page_size, device=key_cache.device)
            valid_page_slots = torch.arange(page_ids.shape[2], device=key_cache.device)[
                None, None, :
            ] < resolved_pages.page_counts.unsqueeze(-1)
            clustered_mask = valid_page_slots.unsqueeze(-1) & (
                page_offsets < page_token_counts.unsqueeze(-1)
            )
            clustered_keys = clustered_keys.reshape(
                batch_size, num_kv_heads, -1, head_size
            )
            clustered_values = clustered_values.reshape_as(clustered_keys)
            clustered_mask = clustered_mask.reshape(batch_size, num_kv_heads, -1)
        elif selection.exact_page_ids.numel() == 0:
            clustered_keys = torch.empty(
                batch_size,
                num_kv_heads,
                0,
                head_size,
                dtype=key_cache.dtype,
                device=key_cache.device,
            )
            clustered_values = torch.empty_like(clustered_keys)
            clustered_mask = torch.empty(
                batch_size,
                num_kv_heads,
                0,
                dtype=torch.bool,
                device=key_cache.device,
            )
        else:
            clustered_keys, clustered_values, clustered_mask = (
                self.cluster_store.gather_pages(
                    plan.layer_name,
                    selection.exact_page_ids,
                    selection.exact_page_token_counts,
                )
            )

            if clustered_keys.device != key_cache.device:
                clustered_keys = clustered_keys.to(
                    device=key_cache.device,
                    non_blocking=False,
                )
                clustered_values = clustered_values.to(
                    device=value_cache.device,
                    non_blocking=False,
                )
                clustered_mask = clustered_mask.to(
                    device=key_cache.device,
                    non_blocking=False,
                )

        exact_keys = torch.cat(
            (primary_keys, clustered_keys),
            dim=2,
        ).contiguous()
        exact_values = torch.cat(
            (primary_values, clustered_values),
            dim=2,
        ).contiguous()
        exact_token_mask = torch.cat(
            (
                plan.primary_exact_token_mask,
                clustered_mask,
            ),
            dim=2,
        ).contiguous()

        return exact_keys, exact_values, exact_token_mask
