# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from dataclasses import dataclass
from math import prod

import torch

from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.legacy.cluster_store import (
    RetroSpecClusterBlockTable,
    RetroSpecCompactResolvedClusterPages,
    RetroSpecCompactVerificationResolvedPages,
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationStaging,
    RetroSpecFullVerificationTicket,
    RetroSpecRankedDraftResolvedClusters,
    RetroSpecResolvedClusterPages,
)
from vllm.v1.spec_decode.retrospec.legacy.index_residency import (
    RetroSpecClusterSummary,
    RetroSpecResidentLayerArena,
)


@dataclass(frozen=True)
class RetroSpecTokenSelectionPlan:
    layer_name: str

    request_slot_ids: torch.Tensor
    request_slot_generations: torch.Tensor

    primary_exact_token_indices: torch.Tensor
    primary_exact_token_mask: torch.Tensor

    sparse_exact_cluster_indices: torch.Tensor
    sparse_estimation_cluster_indices: torch.Tensor

    expanded_exact_cluster_indices: torch.Tensor
    expanded_estimation_cluster_indices: torch.Tensor

    sparse_attn: torch.Tensor
    expanded_attn: torch.Tensor


@dataclass(frozen=True)
class RetroSpecRankedSelectionPlan:
    layer_name: str

    request_slot_ids: torch.Tensor
    request_slot_generations: torch.Tensor

    primary_exact_token_indices: torch.Tensor
    primary_exact_token_mask: torch.Tensor

    ranked_cluster_indices: torch.Tensor
    candidate_counts: torch.Tensor

    sparse_attn: torch.Tensor
    expanded_attn: torch.Tensor

    sparse_retrieval_width: int
    sparse_estimation_width: int
    expanded_retrieval_width: int


@dataclass(frozen=True)
class RetroSpecTokenAttentionSelection:
    exact_cluster_ids: torch.Tensor
    exact_page_ids: torch.Tensor
    exact_page_token_counts: torch.Tensor
    exact_token_counts: torch.Tensor

    estimation_keys: torch.Tensor
    estimation_values: torch.Tensor
    estimation_token_counts: torch.Tensor

    attention_mass: torch.Tensor
    plan: RetroSpecTokenSelectionPlan
    resolved_pages: (
        RetroSpecResolvedClusterPages | RetroSpecCompactResolvedClusterPages | None
    )
    prefetch_miss_cluster_ids: torch.Tensor | None = None
    prefetch_miss_positions: torch.Tensor | None = None
    prefetch_miss_count: torch.Tensor | None = None
    prefetch_num_groups: int = 0
    prefetch_num_ranks: int = 0

    @property
    def hit_attn(self) -> torch.Tensor:
        return self.attention_mass


@dataclass(frozen=True)
class RetroSpecRankedDraftAttentionSelection:
    """DRAFT selection consumed directly from the resident GPU index."""

    plan: RetroSpecRankedSelectionPlan
    arena: RetroSpecResidentLayerArena
    resolved_clusters: RetroSpecRankedDraftResolvedClusters
    exact_token_counts: torch.Tensor
    attention_mass: torch.Tensor
    prefetch_miss_cluster_ids: torch.Tensor | None = None
    prefetch_miss_positions: torch.Tensor | None = None
    prefetch_miss_count: torch.Tensor | None = None
    prefetch_num_groups: int = 0
    prefetch_num_ranks: int = 0

    @property
    def hit_attn(self) -> torch.Tensor:
        return self.attention_mass


@dataclass(frozen=True)
class RetroSpecIndexedTokenAttentionSelection:
    layer_name: str
    plan_row_indices: torch.Tensor
    plan_valid_rows: torch.Tensor

    request_slot_ids: torch.Tensor
    request_slot_generations: torch.Tensor

    primary_exact_token_indices: torch.Tensor
    primary_exact_token_mask: torch.Tensor

    exact_cluster_indices: torch.Tensor

    estimation_cluster_indices: torch.Tensor
    estimation_keys: torch.Tensor
    estimation_values: torch.Tensor
    estimation_token_counts: torch.Tensor

    attention_mass: torch.Tensor


@dataclass
class _PreparedIndexedVerificationLayer:
    layer_name: str
    plan_row_indices: torch.Tensor
    plan_valid_rows: torch.Tensor
    request_slot_ids: torch.Tensor
    request_slot_generations: torch.Tensor
    exact_cluster_indices: torch.Tensor
    attention_mass: torch.Tensor
    resolved_pages: RetroSpecCompactVerificationResolvedPages | None = None


@dataclass
class _IndexedVerificationTransaction:
    level: RetroSpecAttentionLevel
    layers: tuple[str, ...]
    prepared: dict[str, _PreparedIndexedVerificationLayer]
    next_layer_index: int = 0


@dataclass(frozen=True)
class RetroSpecFullVerificationPlan:
    """Exact committed-prefix layout used by target full verification.

    Clustered stable tokens reference the existing cluster page store.
    Tokens outside complete clustered segments remain primary references into
    the active vLLM KV cache. No second full-prefix KV cache is created.
    """

    layer_name: str

    primary_exact_token_indices: torch.Tensor
    primary_exact_token_mask: torch.Tensor

    clustered_descriptors: tuple[RetroSpecFullVerificationDescriptor, ...]
    clustered_kv: RetroSpecFullVerificationStaging | None
    exact_token_counts: torch.Tensor


@dataclass(frozen=True)
class _RequestLayerSegment:
    indexed_start: int
    indexed_end: int

    cluster_start: int
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_blocks: RetroSpecClusterBlockTable


@dataclass(frozen=True)
class _CompletedRequestLayerSegment:
    cluster_summary: RetroSpecClusterSummary
    cluster_blocks: RetroSpecClusterBlockTable


@dataclass(frozen=True)
class _StagedRequestLayerSegment:
    layer_name: str
    request_id: str

    indexed_start: int
    indexed_end: int
    cluster_start: int

    resident_summary: RetroSpecClusterSummary
    build_future: Future[_CompletedRequestLayerSegment]


@dataclass
class _RequestLayerIndex:
    revision: int
    segments: list[_RequestLayerSegment]
    num_clusters: int
    indexed_end: int
    full_verification_descriptor: RetroSpecFullVerificationDescriptor | None


@dataclass(frozen=True)
class _PrefetchedFullVerificationLayer:
    layer_name: str
    ticket: RetroSpecFullVerificationTicket | None
    primed: bool = False


@dataclass(frozen=True)
class _PrimedFullVerificationPipeline:
    request_ids: tuple[str, ...]
    layers: tuple[tuple[str, int], ...]
    revisions: tuple[tuple[int, ...], ...]
    device: torch.device
    prefetched: tuple[_PrefetchedFullVerificationLayer, ...]


@dataclass(frozen=True)
class _PackedClusterZones:
    sparse_retrieval_indices: torch.Tensor
    sparse_retrieval_mask: torch.Tensor
    sparse_retrieval_scores: torch.Tensor

    sparse_estimation_indices: torch.Tensor
    sparse_estimation_mask: torch.Tensor

    expanded_retrieval_indices: torch.Tensor
    expanded_retrieval_mask: torch.Tensor
    expanded_retrieval_scores: torch.Tensor

    expanded_estimation_indices: torch.Tensor
    expanded_estimation_mask: torch.Tensor


@dataclass(frozen=True)
class _ClusterSelectionWorkspace:
    scores: torch.Tensor
    softmax_lse: torch.Tensor
    tile_max: torch.Tensor
    tile_sum: torch.Tensor
    tile_candidate_counts: torch.Tensor
    candidate_counts: torch.Tensor
    topk_values: torch.Tensor
    topk_indices: torch.Tensor


@dataclass(frozen=True)
class _SelectionStepWorkspace:
    draft_exact_cluster_handles: torch.Tensor
    draft_resident_bucket_ids: torch.Tensor
    draft_clustered_token_counts: torch.Tensor
    draft_attention_mass: torch.Tensor
    draft_hit_attention_by_head: torch.Tensor
    draft_selected_cluster_counts: torch.Tensor
    draft_hit_cluster_counts: torch.Tensor
    draft_miss_cluster_counts: torch.Tensor
    draft_hit_gate_ready: torch.Tensor
    draft_prefetch_miss_cluster_ids: torch.Tensor
    draft_prefetch_miss_positions: torch.Tensor
    draft_prefetch_miss_count: torch.Tensor

    sparse_estimation_width: int
    sparse_retrieval_width: int


@dataclass(frozen=True)
class _DraftSelectionScratch:
    draft_exact_cluster_handles: torch.Tensor
    draft_resident_bucket_ids: torch.Tensor
    draft_clustered_token_counts: torch.Tensor
    draft_attention_mass: torch.Tensor
    draft_hit_attention_by_head: torch.Tensor
    draft_selected_cluster_counts: torch.Tensor
    draft_hit_cluster_counts: torch.Tensor
    draft_miss_cluster_counts: torch.Tensor
    draft_hit_gate_ready: torch.Tensor

    @classmethod
    def allocate(
        cls,
        batch_capacity: int,
        num_kv_heads: int,
        sparse_retrieval_width: int,
        device: torch.device,
    ) -> "_DraftSelectionScratch":
        group_shape = (batch_capacity, num_kv_heads)
        cluster_shape = (*group_shape, sparse_retrieval_width)

        return cls(
            draft_exact_cluster_handles=torch.empty(
                cluster_shape, dtype=torch.int64, device=device
            ),
            draft_resident_bucket_ids=torch.empty(
                cluster_shape, dtype=torch.int32, device=device
            ),
            draft_clustered_token_counts=torch.empty(
                group_shape, dtype=torch.int32, device=device
            ),
            draft_attention_mass=torch.empty(
                batch_capacity, dtype=torch.float32, device=device
            ),
            draft_hit_attention_by_head=torch.empty(
                group_shape, dtype=torch.float32, device=device
            ),
            draft_selected_cluster_counts=torch.empty(
                group_shape, dtype=torch.int32, device=device
            ),
            draft_hit_cluster_counts=torch.empty(
                group_shape, dtype=torch.int32, device=device
            ),
            draft_miss_cluster_counts=torch.empty(
                group_shape, dtype=torch.int32, device=device
            ),
            draft_hit_gate_ready=torch.empty(
                group_shape, dtype=torch.bool, device=device
            ),
        )

    def matches(
        self,
        batch_size: int,
        num_kv_heads: int,
        sparse_retrieval_width: int,
        device: torch.device,
    ) -> bool:
        return (
            self.draft_exact_cluster_handles.shape[0] >= batch_size
            and self.draft_exact_cluster_handles.shape[1] == num_kv_heads
            and self.draft_exact_cluster_handles.shape[2] >= sparse_retrieval_width
            and self.draft_exact_cluster_handles.device == device
        )


@dataclass(frozen=True)
class _SelectionPlanTable:
    layer_name: str
    valid_rows: torch.Tensor

    request_slot_ids: torch.Tensor
    request_slot_generations: torch.Tensor

    primary_exact_token_indices: torch.Tensor
    primary_exact_token_mask: torch.Tensor

    ranked_cluster_indices: torch.Tensor
    candidate_counts: torch.Tensor

    draft_prefetch_miss_cluster_ids: torch.Tensor
    draft_prefetch_miss_positions: torch.Tensor
    draft_prefetch_miss_counts: torch.Tensor

    sparse_attn: torch.Tensor
    expanded_attn: torch.Tensor

    ranking_width: int
    sparse_estimation_width: int
    sparse_retrieval_width: int
    expanded_retrieval_width: int
    prefetch_width: int
    max_pages_per_cluster: int
    head_size: int
    dtype: torch.dtype

    @classmethod
    def allocate(
        cls,
        layer_name: str,
        num_steps: int,
        batch_capacity: int,
        num_kv_heads: int,
        primary_exact_width: int,
        sparse_retrieval_width: int,
        prefetch_width: int,
        expanded_retrieval_width: int,
        sparse_estimation_width: int,
        max_pages_per_cluster: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "_SelectionPlanTable":
        prefix = (num_steps, batch_capacity, num_kv_heads)
        ranking_width = sparse_retrieval_width + sparse_estimation_width
        ranked_cluster_shape = (*prefix, ranking_width)
        primary_shape = (*prefix, primary_exact_width)
        prefetch_capacity = batch_capacity * num_kv_heads * prefetch_width

        return cls(
            layer_name=layer_name,
            valid_rows=torch.zeros(
                (num_steps, batch_capacity), dtype=torch.bool, device=device
            ),
            request_slot_ids=torch.empty(
                batch_capacity, dtype=torch.int64, device=device
            ),
            request_slot_generations=torch.empty(
                batch_capacity, dtype=torch.int64, device=device
            ),
            primary_exact_token_indices=torch.empty(
                primary_shape, dtype=torch.int64, device=device
            ),
            primary_exact_token_mask=torch.empty(
                primary_shape, dtype=torch.bool, device=device
            ),
            ranked_cluster_indices=torch.empty(
                ranked_cluster_shape, dtype=torch.int64, device=device
            ),
            candidate_counts=torch.empty(prefix, dtype=torch.int32, device=device),
            draft_prefetch_miss_cluster_ids=torch.empty(
                (num_steps, prefetch_capacity), dtype=torch.int64, device=device
            ),
            draft_prefetch_miss_positions=torch.empty(
                (num_steps, prefetch_capacity), dtype=torch.int64, device=device
            ),
            draft_prefetch_miss_counts=torch.zeros(
                (num_steps, 1), dtype=torch.int32, device=device
            ),
            sparse_attn=torch.empty(
                (num_steps, batch_capacity), dtype=torch.float32, device=device
            ),
            expanded_attn=torch.empty(
                (num_steps, batch_capacity), dtype=torch.float32, device=device
            ),
            ranking_width=ranking_width,
            sparse_estimation_width=sparse_estimation_width,
            sparse_retrieval_width=sparse_retrieval_width,
            expanded_retrieval_width=expanded_retrieval_width,
            prefetch_width=prefetch_width,
            max_pages_per_cluster=max_pages_per_cluster,
            head_size=head_size,
            dtype=dtype,
        )

    @property
    def batch_capacity(self) -> int:
        return self.valid_rows.shape[1]

    def matches(
        self,
        num_steps: int,
        batch_size: int,
        num_kv_heads: int,
        primary_exact_width: int,
        sparse_retrieval_width: int,
        prefetch_width: int,
        expanded_retrieval_width: int,
        sparse_estimation_width: int,
        max_pages_per_cluster: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            self.valid_rows.shape[0] == num_steps
            and self.batch_capacity >= batch_size
            and self.primary_exact_token_indices.shape[2:]
            == (num_kv_heads, primary_exact_width)
            and self.ranked_cluster_indices.shape[2:]
            == (
                num_kv_heads,
                sparse_retrieval_width + sparse_estimation_width,
            )
            and self.candidate_counts.shape[2:] == (num_kv_heads,)
            and self.prefetch_width == prefetch_width
            and self.expanded_retrieval_width == expanded_retrieval_width
            and self.max_pages_per_cluster == max_pages_per_cluster
            and self.head_size == head_size
            and self.dtype == dtype
            and self.valid_rows.device == device
        )

    def ranked_plan(
        self, step_index: int, batch_size: int
    ) -> RetroSpecRankedSelectionPlan:
        if not 0 <= step_index < self.valid_rows.shape[0]:
            raise IndexError("Selection-plan step is out of range")
        if not 0 < batch_size <= self.batch_capacity:
            raise ValueError("Selection-plan batch exceeds table capacity")

        return RetroSpecRankedSelectionPlan(
            layer_name=self.layer_name,
            request_slot_ids=self.request_slot_ids[:batch_size],
            request_slot_generations=self.request_slot_generations[:batch_size],
            primary_exact_token_indices=(
                self.primary_exact_token_indices[step_index, :batch_size]
            ),
            primary_exact_token_mask=(
                self.primary_exact_token_mask[step_index, :batch_size]
            ),
            ranked_cluster_indices=(
                self.ranked_cluster_indices[step_index, :batch_size]
            ),
            candidate_counts=self.candidate_counts[step_index, :batch_size],
            sparse_attn=self.sparse_attn[step_index, :batch_size],
            expanded_attn=self.expanded_attn[step_index, :batch_size],
            sparse_retrieval_width=self.sparse_retrieval_width,
            sparse_estimation_width=self.sparse_estimation_width,
            expanded_retrieval_width=self.expanded_retrieval_width,
        )

    def step_workspace(
        self,
        step_index: int,
        batch_size: int,
        scratch: _DraftSelectionScratch,
    ) -> _SelectionStepWorkspace:
        if not 0 <= step_index < self.valid_rows.shape[0]:
            raise IndexError("Selection-plan step is out of range")
        if not 0 < batch_size <= self.batch_capacity:
            raise ValueError("Selection-plan batch exceeds table capacity")

        num_kv_heads = self.ranked_cluster_indices.shape[2]
        retrieval_width = self.sparse_retrieval_width
        group_shape = (batch_size, num_kv_heads)
        cluster_shape = (*group_shape, retrieval_width)

        def prefix_view(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
            num_items = prod(shape)
            return tensor.view(-1)[:num_items].view(shape)

        return _SelectionStepWorkspace(
            draft_exact_cluster_handles=prefix_view(
                scratch.draft_exact_cluster_handles, cluster_shape
            ),
            draft_resident_bucket_ids=prefix_view(
                scratch.draft_resident_bucket_ids, cluster_shape
            ),
            draft_clustered_token_counts=(
                prefix_view(scratch.draft_clustered_token_counts, group_shape)
            ),
            draft_attention_mass=prefix_view(
                scratch.draft_attention_mass, (batch_size,)
            ),
            draft_hit_attention_by_head=(
                prefix_view(scratch.draft_hit_attention_by_head, group_shape)
            ),
            draft_selected_cluster_counts=(
                prefix_view(scratch.draft_selected_cluster_counts, group_shape)
            ),
            draft_hit_cluster_counts=(
                prefix_view(scratch.draft_hit_cluster_counts, group_shape)
            ),
            draft_miss_cluster_counts=(
                prefix_view(scratch.draft_miss_cluster_counts, group_shape)
            ),
            draft_hit_gate_ready=(
                prefix_view(scratch.draft_hit_gate_ready, group_shape)
            ),
            draft_prefetch_miss_cluster_ids=(
                self.draft_prefetch_miss_cluster_ids[step_index]
            ),
            draft_prefetch_miss_positions=(
                self.draft_prefetch_miss_positions[step_index]
            ),
            draft_prefetch_miss_count=self.draft_prefetch_miss_counts[step_index],
            sparse_estimation_width=self.sparse_estimation_width,
            sparse_retrieval_width=self.sparse_retrieval_width,
        )


@dataclass(frozen=True)
class _IndexedVerificationWorkspace:
    plan_row_indices: torch.Tensor
    plan_valid_rows: torch.Tensor
    request_slot_ids: torch.Tensor
    request_slot_generations: torch.Tensor
    exact_cluster_indices: torch.Tensor
    estimation_cluster_indices: torch.Tensor
    estimation_cluster_mask: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    token_counts: torch.Tensor
    attention_mass: torch.Tensor

    @classmethod
    def allocate(
        cls,
        pair_capacity: int,
        num_kv_heads: int,
        exact_width: int,
        estimation_width: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "_IndexedVerificationWorkspace":
        exact_shape = (pair_capacity, num_kv_heads, exact_width)
        estimation_shape = (pair_capacity, num_kv_heads, estimation_width)
        summary_shape = (*estimation_shape, head_size)
        return cls(
            plan_row_indices=torch.empty(
                pair_capacity, dtype=torch.int64, device=device
            ),
            plan_valid_rows=torch.empty(pair_capacity, dtype=torch.bool, device=device),
            request_slot_ids=torch.empty(
                pair_capacity, dtype=torch.int64, device=device
            ),
            request_slot_generations=torch.empty(
                pair_capacity, dtype=torch.int64, device=device
            ),
            exact_cluster_indices=torch.empty(
                exact_shape, dtype=torch.int32, device=device
            ),
            estimation_cluster_indices=torch.empty(
                estimation_shape, dtype=torch.int32, device=device
            ),
            estimation_cluster_mask=torch.empty(
                estimation_shape, dtype=torch.bool, device=device
            ),
            keys=torch.empty(summary_shape, dtype=dtype, device=device),
            values=torch.empty(summary_shape, dtype=dtype, device=device),
            token_counts=torch.empty(
                estimation_shape, dtype=torch.int32, device=device
            ),
            attention_mass=torch.empty(
                pair_capacity, dtype=torch.float32, device=device
            ),
        )

    def matches(
        self,
        pair_capacity: int,
        num_kv_heads: int,
        exact_width: int,
        estimation_width: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            self.keys.shape[0] >= pair_capacity
            and self.keys.shape[1] == num_kv_heads
            and self.exact_cluster_indices.shape[2] >= exact_width
            and self.keys.shape[2] >= estimation_width
            and self.keys.shape[3] == head_size
            and self.keys.dtype == dtype
            and self.keys.device == device
        )
