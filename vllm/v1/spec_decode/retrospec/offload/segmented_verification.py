# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from contextlib import suppress

import torch

from .cluster_store import (
    RetroSpecCompactResolvedClusterPages,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationStaging,
    RetroSpecRankedDraftResolvedClusters,
    RetroSpecResidentPrefetchInput,
    RetroSpecVerificationResolveRequest,
)
from .index import RetroSpecAttentionLevel
from .index_residency import (
    RetroSpecResidentBatchView,
)
from .segmented_types import (
    RetroSpecFullVerificationPlan,
    RetroSpecIndexedTokenAttentionSelection,
    RetroSpecRankedDraftAttentionSelection,
    RetroSpecRankedSelectionPlan,
    RetroSpecTokenAttentionSelection,
    RetroSpecTokenSelectionPlan,
    _IndexedVerificationTransaction,
    _IndexedVerificationWorkspace,
    _PrefetchedFullVerificationLayer,
    _PreparedIndexedVerificationLayer,
    _PrimedFullVerificationPipeline,
    _SelectionStepWorkspace,
)
from .selection_kernels import (
    pack_ranked_verification_exact_plan,
)


class RetroSpecSegmentVerificationMixin:
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

    def _get_full_verification_descriptors(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        num_kv_heads: int,
    ) -> tuple[RetroSpecFullVerificationDescriptor, ...]:
        layer_indices = self._indices.get(layer_name, {})
        descriptors: list[RetroSpecFullVerificationDescriptor] = []
        for request_id in request_ids:
            record = layer_indices.get(request_id)
            descriptor = None if record is None else record.full_verification_descriptor
            if descriptor is None:
                descriptor = RetroSpecFullVerificationDescriptor.empty(num_kv_heads)
            elif descriptor.num_kv_heads != num_kv_heads:
                raise RuntimeError("Full-verification descriptor changed KV-head count")
            descriptors.append(descriptor)
        return tuple(descriptors)

    @staticmethod
    def _canonical_full_verification_device(device: torch.device) -> torch.device:
        if device.type != "cuda":
            raise ValueError("Full-verification pipeline requires a CUDA device")
        if device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    @staticmethod
    def _normalize_full_verification_layers(
        layer_num_kv_heads: Mapping[str, int],
    ) -> tuple[tuple[str, int], ...]:
        if not layer_num_kv_heads:
            raise ValueError("Full-verification pipeline requires model layers")
        layers = tuple(
            (layer_name, int(num_kv_heads))
            for layer_name, num_kv_heads in layer_num_kv_heads.items()
        )
        if any(num_kv_heads <= 0 for _, num_kv_heads in layers):
            raise ValueError("Full-verification KV-head counts must be positive")
        return layers

    def _get_full_verification_revisions(
        self,
        request_ids: Sequence[str],
        layers: Sequence[tuple[str, int]],
    ) -> tuple[tuple[int, ...], ...]:
        revisions: list[tuple[int, ...]] = []
        for layer_name, _ in layers:
            layer_indices = self._indices.get(layer_name, {})
            revisions.append(
                tuple(
                    -1
                    if (record := layer_indices.get(request_id)) is None
                    else record.revision
                    for request_id in request_ids
                )
            )
        return tuple(revisions)

    def _record_full_verification_prime_outcome(self, adopted: bool) -> None:
        sample = 1.0 if adopted else 0.0
        previous = self._full_verify_prime_adoption_ema
        if previous is None:
            self._full_verify_prime_adoption_ema = sample
        else:
            alpha = self._FULL_VERIFY_PRIME_EMA_ALPHA
            self._full_verify_prime_adoption_ema = (
                previous * (1.0 - alpha) + sample * alpha
            )
        self._full_verify_prime_outcomes += 1
        if adopted:
            self._full_verify_prime_skipped_opportunities = 0

        if self.performance_stats is not None:
            counter = (
                "full_verify_prime_adopted"
                if adopted
                else "full_verify_prime_discarded"
            )
            self.performance_stats.add_counter(counter)

    def _should_submit_full_verification_prime(self) -> bool:
        if (
            self._full_verify_prime_outcomes
            < self._FULL_VERIFY_PRIME_BOOTSTRAP_OUTCOMES
        ):
            return True

        adoption_ema = self._full_verify_prime_adoption_ema
        if (
            adoption_ema is None
            or adoption_ema >= self._FULL_VERIFY_PRIME_MIN_ADOPTION_RATE
        ):
            self._full_verify_prime_skipped_opportunities = 0
            return True

        self._full_verify_prime_skipped_opportunities += 1
        if (
            self._full_verify_prime_skipped_opportunities
            >= self._FULL_VERIFY_PRIME_REPROBE_INTERVAL
        ):
            self._full_verify_prime_skipped_opportunities = 0
            return True

        if self.performance_stats is not None:
            self.performance_stats.add_counter("full_verify_prime_policy_skipped")
        return False

    @staticmethod
    def _cancel_full_verification_prefetches(
        prefetched: Sequence[_PrefetchedFullVerificationLayer],
        wait: bool,
    ) -> None:
        tickets = tuple(
            layer.ticket for layer in prefetched if layer.ticket is not None
        )
        for ticket in tickets:
            ticket.cancel()
        if not wait:
            return
        for ticket in tickets:
            with suppress(CancelledError):
                ticket.result()

    def _discard_primed_full_verification(
        self,
        wait: bool,
        record_outcome: bool = True,
    ) -> None:
        primed = self._primed_full_verification
        if primed is None:
            return

        self._primed_full_verification = None
        self._cancel_full_verification_prefetches(primed.prefetched, wait)
        if record_outcome:
            self._record_full_verification_prime_outcome(adopted=False)

    def _discard_primed_full_verification_for_requests(
        self,
        request_ids: Sequence[str],
        wait: bool,
    ) -> None:
        primed = self._primed_full_verification
        if primed is None or set(request_ids).isdisjoint(primed.request_ids):
            return
        self._discard_primed_full_verification(wait=wait)

    def _submit_full_verification_layer(
        self,
        request_ids: Sequence[str],
        layer: tuple[str, int],
        primed: bool,
    ) -> _PrefetchedFullVerificationLayer:
        layer_name, num_kv_heads = layer
        descriptors = self._get_full_verification_descriptors(
            layer_name,
            request_ids,
            num_kv_heads,
        )
        ticket = None
        if any(descriptor.num_tokens for descriptor in descriptors):
            ticket = self.cluster_store.submit_full_verification_tokens(
                layer_name=layer_name,
                descriptors=descriptors,
            )
        return _PrefetchedFullVerificationLayer(
            layer_name=layer_name,
            ticket=ticket,
            primed=primed,
        )

    def _prefetch_full_verification_layer(
        self,
        layer_index: int,
    ) -> _PrefetchedFullVerificationLayer:
        if self._full_verification_device is None:
            raise RuntimeError("Full-verification pipeline has no CUDA device")
        return self._submit_full_verification_layer(
            request_ids=self._full_verification_request_ids,
            layer=self._full_verification_layers[layer_index],
            primed=False,
        )

    def prime_full_verification_pipeline(
        self,
        request_ids: Sequence[str],
        layer_num_kv_heads: Mapping[str, int],
        device: torch.device,
    ) -> bool:
        if not self._proposal_active:
            raise RuntimeError(
                "Full-verification priming must run inside an active proposal"
            )
        if self._full_verification_pipeline_active:
            raise RuntimeError(
                "Cannot prime while a full-verification pipeline is active"
            )

        request_ids = tuple(request_ids)
        if request_ids != self._proposal_request_ids:
            raise ValueError(
                "Full-verification prime requests do not match the proposal batch"
            )

        layers = self._normalize_full_verification_layers(layer_num_kv_heads)
        device = self._canonical_full_verification_device(device)
        revisions = self._get_full_verification_revisions(request_ids, layers)
        current = self._primed_full_verification
        if (
            current is not None
            and current.request_ids == request_ids
            and current.layers == layers
            and current.revisions == revisions
            and current.device == device
        ):
            if self.performance_stats is not None:
                self.performance_stats.add_counter("full_verify_prime_coalesced")
            return True

        if current is not None:
            self._discard_primed_full_verification(wait=False)
        if not self._should_submit_full_verification_prime():
            return False

        prime_depth = min(self._FULL_VERIFY_PRIME_DEPTH, len(layers))
        prefetched: list[_PrefetchedFullVerificationLayer] = []
        try:
            for layer in layers[:prime_depth]:
                prefetched.append(
                    self._submit_full_verification_layer(
                        request_ids=request_ids,
                        layer=layer,
                        primed=True,
                    )
                )
        except BaseException:
            self._cancel_full_verification_prefetches(prefetched, wait=False)
            raise

        if not any(layer.ticket is not None for layer in prefetched):
            if self.performance_stats is not None:
                self.performance_stats.add_counter("full_verify_prime_empty")
            return False

        self._primed_full_verification = _PrimedFullVerificationPipeline(
            request_ids=request_ids,
            layers=layers,
            revisions=revisions,
            device=device,
            prefetched=tuple(prefetched),
        )
        if self.performance_stats is not None:
            self.performance_stats.add_counter("full_verify_prime_submitted")
            self.performance_stats.add_counter(
                "full_verify_prime_layers",
                sum(layer.ticket is not None for layer in prefetched),
            )
        return True

    def _adopt_primed_full_verification(
        self,
        request_ids: tuple[str, ...],
        layers: tuple[tuple[str, int], ...],
        device: torch.device,
    ) -> bool:
        primed = self._primed_full_verification
        if primed is None:
            return False

        revisions = self._get_full_verification_revisions(request_ids, layers)
        matches = (
            primed.request_ids == request_ids
            and primed.layers == layers
            and primed.revisions == revisions
            and primed.device == device
        )
        if not matches:
            self._discard_primed_full_verification(wait=False)
            return False

        self._primed_full_verification = None
        self._full_verification_prefetched.extend(primed.prefetched)
        self._full_verification_next_layer_index = len(primed.prefetched)
        self._record_full_verification_prime_outcome(adopted=True)
        return True

    def begin_full_verification_pipeline(
        self,
        request_ids: Sequence[str],
        layer_num_kv_heads: Mapping[str, int],
        device: torch.device,
    ) -> None:
        if self._full_verification_pipeline_active:
            raise RuntimeError("Full-verification pipeline is already active")

        request_ids = tuple(request_ids)
        layers = self._normalize_full_verification_layers(layer_num_kv_heads)
        device = self._canonical_full_verification_device(device)

        self._full_verification_pipeline_active = True
        self._full_verification_request_ids = request_ids
        self._full_verification_layers = layers
        self._full_verification_layer_cursor = 0
        self._full_verification_next_layer_index = 0
        self._full_verification_device = device
        self._full_verification_prefetched.clear()
        try:
            adopted = self._adopt_primed_full_verification(
                request_ids,
                layers,
                device,
            )
            if not adopted:
                self._full_verification_prefetched.append(
                    self._prefetch_full_verification_layer(0)
                )
                self._full_verification_next_layer_index = 1
        except BaseException:
            self.end_full_verification_pipeline()
            raise

    def consume_full_verification_layer(
        self,
        layer_name: str,
    ) -> RetroSpecFullVerificationStaging | None:
        if not self._full_verification_pipeline_active:
            raise RuntimeError("Full-verification pipeline is not active")
        if not self._full_verification_prefetched:
            raise RuntimeError("Full-verification pipeline has no remaining layer")

        expected_layer_name = self._full_verification_layers[
            self._full_verification_layer_cursor
        ][0]
        if expected_layer_name != layer_name:
            raise RuntimeError(
                "Full-verification layer order differs from the installed model"
            )

        prefetched = self._full_verification_prefetched.popleft()
        if prefetched.layer_name != layer_name:
            raise RuntimeError(
                "Prefetched full-verification layer does not match the model layer"
            )
        if prefetched.primed and prefetched.ticket is not None:
            counter = (
                "full_verify_prime_ready"
                if prefetched.ticket.ready()
                else "full_verify_prime_late"
            )
            if self.performance_stats is not None:
                self.performance_stats.add_counter(counter)

        clustered_kv = None if prefetched.ticket is None else prefetched.ticket.result()
        self._full_verification_layer_cursor += 1
        if (
            not self._full_verification_prefetched
            and self._full_verification_next_layer_index
            < len(self._full_verification_layers)
        ):
            next_layer = self._prefetch_full_verification_layer(
                self._full_verification_next_layer_index
            )
            self._full_verification_prefetched.append(next_layer)
            self._full_verification_next_layer_index += 1
        return clustered_kv

    def end_full_verification_pipeline(self) -> None:
        prefetched = tuple(self._full_verification_prefetched)
        self._full_verification_prefetched.clear()
        self._cancel_full_verification_prefetches(prefetched, wait=False)

        self._full_verification_pipeline_active = False
        self._full_verification_request_ids = ()
        self._full_verification_layers = ()
        self._full_verification_layer_cursor = 0
        self._full_verification_next_layer_index = 0
        self._full_verification_device = None

    def build_full_verification_plan(
        self,
        request_ids: Sequence[str],
        layer_name: str,
        seq_lens: Sequence[int],
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecFullVerificationPlan:
        """Build an exact full-verification view over existing KV storage.

        Complete indexed segments are represented by every cluster page owned
        by the requests. Tokens outside those segments remain primary logical
        token references into the active vLLM KV cache.
        """
        request_ids = tuple(request_ids)
        seq_lens = tuple(int(seq_len) for seq_len in seq_lens)

        if any(staged.layer_name == layer_name for staged in self._staged_segments):
            raise RuntimeError(
                "Cannot build full verification because index updates are staged "
                "for this layer"
            )
        if key_cache.ndim != 4:
            raise ValueError(
                "KV cache must have shape [num_blocks, block_size, kv_heads, head_size]"
            )
        if key_cache.shape[1] != self.block_size:
            raise ValueError("KV cache block size does not match the index")
        if block_table.ndim != 2:
            raise ValueError("block_table must be two-dimensional")
        if block_table.shape[0] != len(request_ids):
            raise ValueError("block_table batch size does not match request_ids")
        if len(seq_lens) != len(request_ids):
            raise ValueError("request_ids and seq_lens must have equal length")
        if block_table.device != key_cache.device:
            raise ValueError("block_table and KV cache must use one device")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("block_table entries must be integral")

        max_num_tokens = block_table.shape[1] * self.block_size
        if any(seq_len < 0 for seq_len in seq_lens):
            raise ValueError("Full-verification context lengths must be non-negative")
        if any(seq_len > max_num_tokens for seq_len in seq_lens):
            raise ValueError(
                "Full-verification sequence length exceeds the block table"
            )

        layer_indices = self._indices.get(layer_name, {})
        primary_token_counts: list[int] = []

        for request_id, seq_len in zip(request_ids, seq_lens):
            record = layer_indices.get(request_id)

            if record is None or not record.segments:
                indexed_token_count = 0
            else:
                if record.indexed_end > seq_len:
                    raise RuntimeError(
                        "Full verification requires rolled-back cluster state "
                        "to be rebuilt first"
                    )
                indexed_token_count = (
                    record.indexed_end - record.segments[0].indexed_start
                )

            primary_token_counts.append(seq_len - indexed_token_count)

        view = self._get_resident_view(layer_name, request_ids, key_cache)

        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int64,
            device=block_table.device,
        )
        logical_token_ids, valid_token_mask, _ = self._build_token_layout(
            block_table,
            seq_lens_tensor,
        )

        # Every committed token not owned by a complete clustered segment is
        # part of the exact primary/steady zone.
        indexed_starts, indexed_ends, indexed_requests = (
            self._get_resident_indexed_bounds(view, block_table.device)
        )
        primary_token_mask = valid_token_mask & (
            ~indexed_requests.unsqueeze(1)
            | (logical_token_ids.unsqueeze(0) < indexed_starts.unsqueeze(1))
            | (logical_token_ids.unsqueeze(0) >= indexed_ends.unsqueeze(1))
        )
        num_kv_heads = key_cache.shape[2]
        per_head_primary_mask = primary_token_mask.unsqueeze(1).expand(
            -1,
            num_kv_heads,
            -1,
        )

        (
            primary_exact_token_indices,
            primary_exact_token_mask,
        ) = self._pack_bounded_mask_indices(
            per_head_primary_mask,
            max(primary_token_counts, default=0),
        )

        primary_exact_token_counts = primary_exact_token_mask.sum(
            dim=2,
            dtype=torch.int32,
        )
        descriptors = self._get_full_verification_descriptors(
            layer_name, request_ids, num_kv_heads
        )
        if self._full_verification_pipeline_active:
            clustered_kv = self.consume_full_verification_layer(layer_name)
            if clustered_kv is not None and clustered_kv.token_counts.shape != (
                len(request_ids),
                num_kv_heads,
            ):
                raise RuntimeError(
                    "Prefetched full-verification staging changed batch shape"
                )
        else:
            clustered_kv = None
        clustered_exact_token_counts = (
            torch.stack(
                tuple(
                    descriptor.head_token_counts_tensor for descriptor in descriptors
                ),
                dim=0,
            )
            .to(
                device=primary_exact_token_counts.device,
                dtype=torch.int32,
            )
            .contiguous()
        )
        if clustered_exact_token_counts.shape != primary_exact_token_counts.shape:
            raise RuntimeError(
                "Full-verification clustered and native counts have different shapes"
            )
        exact_token_counts = (
            primary_exact_token_counts + clustered_exact_token_counts
        ).contiguous()
        return RetroSpecFullVerificationPlan(
            layer_name=layer_name,
            primary_exact_token_indices=primary_exact_token_indices,
            primary_exact_token_mask=primary_exact_token_mask,
            clustered_descriptors=descriptors,
            clustered_kv=clustered_kv,
            exact_token_counts=exact_token_counts,
        )
