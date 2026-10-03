# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.spec_decode.retrospec.offload.cluster_store import (
    RetroSpecCompactResolvedClusterPages,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecResolvedClusterPages,
)
from vllm.v1.spec_decode.retrospec.offload.execution import (
    RetroSpecCompactExactPageTable,
    RetroSpecEstimationKVSource,
    RetroSpecExactKVSource,
    RetroSpecExactPageKVSource,
    RetroSpecExactPrimaryKVSource,
    RetroSpecRankedDraftKVSource,
)
from vllm.v1.spec_decode.retrospec.offload.segmented_index import (
    RetroSpecIndexedTokenAttentionSelection,
    RetroSpecRankedDraftAttentionSelection,
    RetroSpecTokenAttentionSelection,
)
from vllm.v1.spec_decode.retrospec.runtime.attention_types import RetroSpecAttentionMode

RetroSpecSelection = (
    RetroSpecTokenAttentionSelection
    | RetroSpecIndexedTokenAttentionSelection
    | RetroSpecRankedDraftAttentionSelection
)


class RetroSpecAttentionExecutionMixin:
    """RetroSpec AttentionExecution helpers."""

    @staticmethod
    def _run_grouped_reference_attention(
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference grouped attention for weighted centroids and CPU fallback.

        Args:
            query:
                [batch, num_query_heads, head_size]
            keys/values:
                [batch, num_kv_heads, max_num_vectors, head_size]
            token_counts:
                [batch, num_kv_heads, max_num_vectors]. A value of one
                represents an exact token. A value larger than one represents
                an estimation centroid for that many tokens.
        """
        batch_size, num_query_heads, head_size = query.shape
        num_kv_heads = keys.shape[1]

        if values.shape != keys.shape:
            raise ValueError("Reference attention keys and values must match")
        if token_counts.shape != keys.shape[:3]:
            raise ValueError("Reference attention token counts do not match keys")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads"
            )

        num_queries_per_kv = num_query_heads // num_kv_heads
        grouped_query = query.float().view(
            batch_size,
            num_kv_heads,
            num_queries_per_kv,
            head_size,
        )

        logits = torch.einsum(
            "bhgd,bhmd->bhgm",
            grouped_query,
            keys.float(),
        )
        logits *= impl.scale

        token_counts_float = token_counts.float()
        valid_mask = token_counts > 0

        # Exact tokens have count one and therefore receive no correction.
        # Estimation centroids receive the same log(cluster_size) correction
        # as the existing RetroSpec estimation path.
        logits += torch.log(token_counts_float.clamp_min(1)).unsqueeze(2)
        logits.masked_fill_(
            ~valid_mask.unsqueeze(2),
            float("-inf"),
        )

        has_vectors = valid_mask.any(dim=2)
        safe_logits = torch.where(
            has_vectors[:, :, None, None],
            logits,
            torch.zeros_like(logits),
        )

        output_lse = torch.logsumexp(safe_logits, dim=-1)
        output_lse = torch.where(
            has_vectors[:, :, None],
            output_lse,
            torch.full_like(output_lse, float("-inf")),
        )

        safe_normalizer = torch.where(
            has_vectors[:, :, None],
            output_lse,
            torch.zeros_like(output_lse),
        )
        weights = torch.exp(logits - safe_normalizer.unsqueeze(-1))
        weights.masked_fill_(
            ~valid_mask.unsqueeze(2),
            0.0,
        )

        attention_output = torch.einsum(
            "bhgm,bhmd->bhgd",
            weights,
            values.float(),
        )
        attention_output = attention_output.reshape(
            batch_size,
            num_query_heads,
            head_size,
        ).to(query.dtype)

        output_lse = output_lse.reshape(
            batch_size,
            num_query_heads,
        )
        output_lse = output_lse.transpose(0, 1).contiguous()

        return attention_output, output_lse

    def _resolve_exact_kv_source(
        self,
        selection: RetroSpecSelection,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        prepared_pages: RetroSpecCompactVerificationResolvedPages | None = None,
        pages_prepared: bool = False,
    ) -> tuple[
        RetroSpecExactKVSource,
        RetroSpecResolvedClusterPages
        | RetroSpecCompactResolvedClusterPages
        | RetroSpecCompactVerificationResolvedPages
        | None,
    ]:
        if isinstance(selection, RetroSpecRankedDraftAttentionSelection):
            raise RuntimeError(
                "Ranked DRAFT selection must use run_ranked_draft_proposal()"
            )
        indexed = isinstance(selection, RetroSpecIndexedTokenAttentionSelection)
        if indexed:
            layer_name = selection.layer_name
            primary_token_indices = selection.primary_exact_token_indices
            primary_token_mask = selection.primary_exact_token_mask
            plan_row_indices = selection.plan_row_indices
            if self.mode not in (
                RetroSpecAttentionMode.SPARSE_VERIFY,
                RetroSpecAttentionMode.EXPANDED_VERIFY,
            ):
                raise RuntimeError("Indexed plans are only valid during verification")
            resolved_pages = (
                prepared_pages
                if pages_prepared
                else self.index.resolve_indexed_verification_pages(selection)
            )
            if resolved_pages is None:
                exact_page_token_counts = torch.empty(
                    plan_row_indices.shape[0],
                    primary_token_indices.shape[1],
                    0,
                    dtype=torch.int32,
                    device=primary_token_indices.device,
                )
            else:
                exact_page_token_counts = resolved_pages.page_token_counts
        else:
            layer_name = selection.plan.layer_name
            primary_token_indices = selection.plan.primary_exact_token_indices
            primary_token_mask = selection.plan.primary_exact_token_mask
            plan_row_indices = None
            resolved_pages = selection.resolved_pages
            exact_cluster_ids = selection.exact_cluster_ids
            exact_page_ids = selection.exact_page_ids
            exact_page_token_counts = selection.exact_page_token_counts
            if resolved_pages is None and exact_page_ids.numel():
                resolved_pages = self.index.cluster_store.resolve_cluster_blocks(
                    layer_name=layer_name,
                    cluster_ids=exact_cluster_ids,
                    logical_page_ids=exact_page_ids,
                    mode="verification",
                )

        resident_pages = None
        staging_pages = None
        compact_pages = None
        if resolved_pages is not None:
            if isinstance(
                resolved_pages,
                (
                    RetroSpecCompactResolvedClusterPages,
                    RetroSpecCompactVerificationResolvedPages,
                ),
            ):
                if resolved_pages.resident_key_pages.shape[0] > 0:
                    resident_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.resident_key_pages,
                        value_pages=resolved_pages.resident_value_pages,
                        page_ids=resolved_pages.resident_page_ids,
                    )
                if (
                    isinstance(
                        resolved_pages, RetroSpecCompactVerificationResolvedPages
                    )
                    and resolved_pages.staging_key_pages.shape[0] > 0
                ):
                    staging_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.staging_key_pages,
                        value_pages=resolved_pages.staging_value_pages,
                        page_ids=resolved_pages.staging_page_ids,
                        ready_event=resolved_pages.staging_ready_event,
                    )
                compact_pages = RetroSpecCompactExactPageTable(
                    page_counts=resolved_pages.page_counts
                )
            else:
                if resolved_pages.resident_key_pages.shape[0] > 0:
                    resident_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.resident_key_pages,
                        value_pages=resolved_pages.resident_value_pages,
                        page_ids=resolved_pages.resident_page_ids,
                        ready_event=resolved_pages.resident_ready_event,
                    )
                if resolved_pages.staging_key_pages.shape[0] > 0:
                    staging_pages = RetroSpecExactPageKVSource(
                        key_pages=resolved_pages.staging_key_pages,
                        value_pages=resolved_pages.staging_value_pages,
                        page_ids=resolved_pages.staging_page_ids,
                        ready_event=resolved_pages.staging_ready_event,
                    )

        source = RetroSpecExactKVSource(
            primary=RetroSpecExactPrimaryKVSource(
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                token_indices=primary_token_indices,
                token_mask=primary_token_mask,
            ),
            page_token_counts=exact_page_token_counts,
            resident_pages=resident_pages,
            staging_pages=staging_pages,
            plan_row_indices=plan_row_indices,
            compact_pages=compact_pages,
        )

        return source, resolved_pages

    def _run_exact_attention(
        self,
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.device.type != "cuda" or query.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            exact_keys, exact_values, exact_token_mask = (
                self.index.materialize_exact_reference(
                    selection,
                    key_cache,
                    value_cache,
                    attn_metadata.block_table,
                )
            )

            if (
                self.mode
                in (
                    RetroSpecAttentionMode.SPARSE_VERIFY,
                    RetroSpecAttentionMode.EXPANDED_VERIFY,
                )
                and query.device.type == "cuda"
                and selection.exact_page_ids.numel()
            ):
                self.index.cluster_store.admit_resident_clusters(
                    layer_name=selection.plan.layer_name,
                    cluster_ids=selection.exact_cluster_ids,
                    page_ids=selection.exact_page_ids,
                )

            return self._run_grouped_reference_attention(
                impl,
                query,
                exact_keys,
                exact_values,
                exact_token_mask.to(torch.int32),
            )

        stage_name = {
            RetroSpecAttentionMode.DRAFT: "draft",
            RetroSpecAttentionMode.SPARSE_VERIFY: "sparse_verify",
            RetroSpecAttentionMode.EXPANDED_VERIFY: "expanded_verify",
        }[self.mode]
        with (
            self.performance_stats.cpu_timer(f"{stage_name}_page_resolve_wall"),
            self.performance_stats.cuda_timer(f"{stage_name}_page_resolve"),
        ):
            source, resolved_pages = self._resolve_exact_kv_source(
                selection=selection,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table,
            )
        try:
            with self.performance_stats.cuda_timer(f"{stage_name}_exact_attention"):
                exact_output = self.exact_attention_workspace.run(
                    source, query, impl.scale
                )
        finally:
            if resolved_pages is not None and resolved_pages.read_lease is not None:
                resolved_pages.read_lease.release()

        miss_admission = getattr(resolved_pages, "miss_admission", None)
        if miss_admission is not None:
            with self.performance_stats.cpu_timer(
                f"{stage_name}_resident_admit_submit"
            ):
                self.index.cluster_store.admit_verification_misses(miss_admission)

        return exact_output

    def _run_fused_proposal_attention(
        self,
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        selection: RetroSpecSelection,
        output: torch.Tensor,
        prepared_pages: RetroSpecCompactVerificationResolvedPages | None = None,
        pages_prepared: bool = False,
    ) -> torch.Tensor:
        if isinstance(selection, RetroSpecRankedDraftAttentionSelection):
            if self.mode != RetroSpecAttentionMode.DRAFT:
                raise RuntimeError(
                    "Ranked DRAFT selection cannot be used during verification"
                )
            resolved = selection.resolved_clusters
            source = RetroSpecRankedDraftKVSource(
                primary=RetroSpecExactPrimaryKVSource(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=attn_metadata.block_table,
                    token_indices=selection.plan.primary_exact_token_indices,
                    token_mask=selection.plan.primary_exact_token_mask,
                ),
                request_slot_ids=selection.plan.request_slot_ids,
                ranked_cluster_indices=selection.plan.ranked_cluster_indices,
                candidate_counts=selection.plan.candidate_counts,
                resident_bucket_ids=resolved.resident_bucket_ids,
                sparse_retrieval_width=selection.plan.sparse_retrieval_width,
                sparse_estimation_width=selection.plan.sparse_estimation_width,
                retrieval_ratio=self.index.retrieval_ratio,
                estimation_ratio=self.index.estimation_ratio,
                cluster_keys=selection.arena.cluster_keys,
                cluster_values=selection.arena.cluster_values,
                cluster_token_counts=selection.arena.cluster_token_counts,
                cluster_page_starts=selection.arena.cluster_page_starts,
                cluster_page_counts=selection.arena.cluster_page_counts,
                page_token_counts=selection.arena.page_token_counts,
                cluster_offsets=selection.arena.cluster_offsets,
                page_offsets=selection.arena.page_offsets,
                resident_table_page_slots=resolved.resident_table_page_slots,
                resident_key_pages=resolved.resident_key_pages,
                resident_value_pages=resolved.resident_value_pages,
            )
            try:
                with self.performance_stats.cuda_timer("draft_ranked_attention"):
                    self.exact_attention_workspace.run_ranked_draft_proposal(
                        source=source,
                        query=query,
                        scale=impl.scale,
                        output=output,
                    )
            finally:
                resolved.read_lease.release()
            return output

        stage_name = {
            RetroSpecAttentionMode.DRAFT: "draft",
            RetroSpecAttentionMode.SPARSE_VERIFY: "sparse_verify",
            RetroSpecAttentionMode.EXPANDED_VERIFY: "expanded_verify",
        }[self.mode]
        with (
            self.performance_stats.cpu_timer(f"{stage_name}_page_resolve_wall"),
            self.performance_stats.cuda_timer(f"{stage_name}_page_resolve"),
        ):
            if pages_prepared:
                source, resolved_pages = self._resolve_exact_kv_source(
                    selection=selection,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=attn_metadata.block_table,
                    prepared_pages=prepared_pages,
                    pages_prepared=True,
                )
            else:
                source, resolved_pages = self._resolve_exact_kv_source(
                    selection=selection,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=attn_metadata.block_table,
                )

        estimation_keys, estimation_values, estimation_token_counts = (
            self._get_grouped_estimation(selection)
        )
        estimation = RetroSpecEstimationKVSource(
            keys=estimation_keys,
            values=estimation_values,
            token_counts=estimation_token_counts,
            plan_row_indices=None,
        )

        try:
            with self.performance_stats.cuda_timer(f"{stage_name}_fused_attention"):
                self.exact_attention_workspace.run_proposal(
                    source=source,
                    estimation=estimation,
                    query=query,
                    scale=impl.scale,
                    output=output,
                )
        finally:
            if (
                not pages_prepared
                and resolved_pages is not None
                and resolved_pages.read_lease is not None
            ):
                resolved_pages.read_lease.release()

        miss_admission = getattr(resolved_pages, "miss_admission", None)
        if not pages_prepared and miss_admission is not None:
            with self.performance_stats.cpu_timer(
                f"{stage_name}_resident_admit_submit"
            ):
                self.index.cluster_store.admit_verification_misses(miss_admission)

        return output

    @staticmethod
    def _get_grouped_estimation(
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return estimation tensors in grouped KV-head layout."""
        return (
            selection.estimation_keys,
            selection.estimation_values,
            selection.estimation_token_counts,
        )

    @classmethod
    def _run_estimation_attention(
        cls,
        impl: FlashAttentionImpl,
        query: torch.Tensor,
        selection: RetroSpecSelection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the reference weighted-centroid attention path."""
        (
            estimation_keys,
            estimation_values,
            estimation_token_counts,
        ) = cls._get_grouped_estimation(selection)

        return cls._run_grouped_reference_attention(
            impl,
            query,
            estimation_keys,
            estimation_values,
            estimation_token_counts,
        )
