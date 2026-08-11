# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .clustering import segmented_kmeans_assignments
from .index import (
    RetroSpecAttentionLevel,
    RetroSpecAttentionSelection,
    RetroSpecBlockIndex,
    RetroSpecSelectionPlan,
)


@dataclass(frozen=True)
class RetroSpecSegmentedSelectionPlan(RetroSpecSelectionPlan):
    sparse_estimation_keys: torch.Tensor
    sparse_estimation_values: torch.Tensor
    sparse_estimation_token_counts: torch.Tensor

    expanded_estimation_keys: torch.Tensor
    expanded_estimation_values: torch.Tensor
    expanded_estimation_token_counts: torch.Tensor


@dataclass(frozen=True)
class _RequestLayerIndex:
    logical_block_ids: torch.Tensor
    block_cluster_ids: torch.Tensor

    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor

    indexed_end: int


@dataclass(frozen=True)
class _PackedSegmentedIndex:
    indexed_block_mask: torch.Tensor
    block_cluster_ids: torch.Tensor

    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_mask: torch.Tensor


class RetroSpecSegmentedBlockIndex(RetroSpecBlockIndex):
    """Segmented block-cluster index compatible with vLLM paged KV cache."""

    def __init__(
        self,
        block_size: int,
        num_speculative_tokens: int,
        retrieval_ratio: float,
        estimation_ratio: float,
        segment_size_tokens: int,
        blocks_per_cluster: int,
        num_kmeans_iterations: int,
    ) -> None:
        super().__init__(
            block_size=block_size,
            num_speculative_tokens=num_speculative_tokens,
            retrieval_ratio=retrieval_ratio,
            estimation_ratio=estimation_ratio,
        )

        if segment_size_tokens % block_size != 0:
            raise ValueError("segment_size_tokens must be divisible by block_size")

        blocks_per_segment = segment_size_tokens // block_size
        if blocks_per_segment % blocks_per_cluster != 0:
            raise ValueError(
                "blocks per segment must be divisible by blocks_per_cluster"
            )
        if num_kmeans_iterations <= 0:
            raise ValueError("num_kmeans_iterations must be positive")

        self.segment_size_tokens = segment_size_tokens
        self.blocks_per_segment = blocks_per_segment
        self.blocks_per_cluster = blocks_per_cluster
        self.num_kmeans_iterations = num_kmeans_iterations

        # layer_name -> request_id -> index
        self._indices: dict[str, dict[str, _RequestLayerIndex]] = {}

        self._proposal_request_ids: tuple[str, ...] = ()
        self._proposal_packed_indices: dict[
            str,
            _PackedSegmentedIndex,
        ] = {}

    def _desired_indexed_end(self, seq_len: int) -> int:
        """Return the exclusive logical-block index covered by clustering."""
        full_block_count = seq_len // self.block_size

        # Block zero is the attention sink. The last blocks remain in the
        # steady exact zone and are not clustered until they become old.
        stable_end = max(
            full_block_count - self.num_recent_blocks,
            1,
        )
        indexable_blocks = stable_end - 1

        complete_segments = indexable_blocks // self.blocks_per_segment
        return 1 + complete_segments * self.blocks_per_segment

    def needs_update(
        self,
        request_id: str,
        seq_len: int,
        layer_names: Sequence[str],
    ) -> bool:
        desired_end = self._desired_indexed_end(seq_len)

        for layer_name in layer_names:
            layer_indices = self._indices.get(layer_name)
            if layer_indices is None or request_id not in layer_indices:
                return True
            if layer_indices[request_id].indexed_end != desired_end:
                return True

        return False

    def remove_requests(self, request_ids: Sequence[str]) -> None:
        for layer_indices in self._indices.values():
            for request_id in request_ids:
                layer_indices.pop(request_id, None)

    def begin_proposal(self, request_ids: Sequence[str]) -> None:
        if self._proposal_request_ids:
            raise RuntimeError("Segmented-index proposal is already active")

        self._proposal_request_ids = tuple(request_ids)
        self._proposal_packed_indices.clear()

    def end_proposal(self) -> None:
        self._proposal_request_ids = ()
        self._proposal_packed_indices.clear()

    @staticmethod
    def _empty_index(
        key_cache: torch.Tensor,
        indexed_end: int = 1,
    ) -> _RequestLayerIndex:
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        return _RequestLayerIndex(
            logical_block_ids=torch.empty(
                0,
                dtype=torch.int64,
                device=key_cache.device,
            ),
            block_cluster_ids=torch.empty(
                0,
                dtype=torch.int64,
                device=key_cache.device,
            ),
            cluster_keys=torch.empty(
                0,
                num_kv_heads,
                head_size,
                dtype=key_cache.dtype,
                device=key_cache.device,
            ),
            cluster_values=torch.empty(
                0,
                num_kv_heads,
                head_size,
                dtype=key_cache.dtype,
                device=key_cache.device,
            ),
            cluster_token_counts=torch.empty(
                0,
                dtype=torch.int32,
                device=key_cache.device,
            ),
            indexed_end=indexed_end,
        )

    def build_or_update(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        rows: Sequence[int],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        """Build missing prefill segments or append newly stable segments."""
        if len(request_ids) != len(seq_lens):
            raise ValueError("request_ids and seq_lens must have equal length")

        layer_indices = self._indices.setdefault(layer_name, {})

        for row in rows:
            request_id = request_ids[row]
            seq_len = seq_lens[row]
            desired_end = self._desired_indexed_end(seq_len)

            previous = layer_indices.get(request_id)
            if previous is not None and desired_end < previous.indexed_end:
                # A sequence-length rollback means the request was recomputed
                # with a new paged-KV allocation. Discard summaries derived
                # from the old physical blocks before rebuilding the index.
                previous = None
                layer_indices.pop(request_id)

            indexed_start = 1 if previous is None else previous.indexed_end

            if desired_end <= indexed_start:
                if previous is None:
                    layer_indices[request_id] = self._empty_index(
                        key_cache,
                        indexed_end=1,
                    )
                continue

            num_new_blocks = desired_end - indexed_start
            if num_new_blocks % self.blocks_per_segment != 0:
                raise RuntimeError("New indexed region must contain complete segments")

            logical_block_ids = torch.arange(
                indexed_start,
                desired_end,
                dtype=torch.int64,
                device=block_table.device,
            )
            physical_block_ids = (
                block_table[row]
                .index_select(
                    0,
                    logical_block_ids,
                )
                .to(torch.int64)
            )

            key_blocks = key_cache.index_select(
                0,
                physical_block_ids,
            )
            value_blocks = value_cache.index_select(
                0,
                physical_block_ids,
            )

            # Every indexed block is complete, so every block contributes
            # exactly block_size tokens.
            block_key_means = key_blocks.float().mean(dim=1)
            block_value_means = value_blocks.float().mean(dim=1)

            # A single cluster assignment is shared by all KV heads because
            # FlashAttention uses one block table per request. Averaging the
            # KV-head features gives one assignment while cluster summaries
            # below remain head-specific.
            clustering_features = block_key_means.mean(dim=1)

            local_assignments, cluster_block_counts = segmented_kmeans_assignments(
                features=clustering_features,
                segment_size=self.blocks_per_segment,
                items_per_cluster=self.blocks_per_cluster,
                num_iterations=self.num_kmeans_iterations,
            )

            num_new_clusters = cluster_block_counts.numel()
            num_kv_heads = key_cache.shape[2]
            head_size = key_cache.shape[3]

            cluster_key_sums = torch.zeros(
                num_new_clusters,
                num_kv_heads,
                head_size,
                dtype=torch.float32,
                device=key_cache.device,
            )
            cluster_value_sums = torch.zeros_like(cluster_key_sums)

            cluster_key_sums.index_add_(
                0,
                local_assignments,
                block_key_means,
            )
            cluster_value_sums.index_add_(
                0,
                local_assignments,
                block_value_means,
            )

            safe_counts = cluster_block_counts.clamp_min(1).to(torch.float32)
            cluster_keys = (cluster_key_sums / safe_counts[:, None, None]).to(
                key_cache.dtype
            )
            cluster_values = (cluster_value_sums / safe_counts[:, None, None]).to(
                value_cache.dtype
            )
            cluster_token_counts = (cluster_block_counts * self.block_size).to(
                torch.int32
            )

            if previous is None:
                cluster_offset = 0
                previous = self._empty_index(key_cache)
            else:
                cluster_offset = previous.cluster_keys.shape[0]

            block_cluster_ids = local_assignments + cluster_offset

            layer_indices[request_id] = _RequestLayerIndex(
                logical_block_ids=torch.cat(
                    [
                        previous.logical_block_ids,
                        logical_block_ids,
                    ]
                ),
                block_cluster_ids=torch.cat(
                    [
                        previous.block_cluster_ids,
                        block_cluster_ids,
                    ]
                ),
                cluster_keys=torch.cat(
                    [
                        previous.cluster_keys,
                        cluster_keys,
                    ]
                ),
                cluster_values=torch.cat(
                    [
                        previous.cluster_values,
                        cluster_values,
                    ]
                ),
                cluster_token_counts=torch.cat(
                    [
                        previous.cluster_token_counts,
                        cluster_token_counts,
                    ]
                ),
                indexed_end=desired_end,
            )

        self._proposal_packed_indices.pop(layer_name, None)

    def _pack_indices(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> _PackedSegmentedIndex:
        cached = self._proposal_packed_indices.get(layer_name)
        if cached is not None:
            return cached

        batch_size, max_num_blocks = block_table.shape
        layer_indices = self._indices.get(layer_name, {})

        records = [layer_indices.get(request_id) for request_id in request_ids]
        max_num_clusters = max(
            1,
            max(
                (record.cluster_keys.shape[0] if record is not None else 0)
                for record in records
            ),
        )

        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        indexed_block_mask = torch.zeros(
            batch_size,
            max_num_blocks,
            dtype=torch.bool,
            device=key_cache.device,
        )
        block_cluster_ids = torch.full(
            (batch_size, max_num_blocks),
            -1,
            dtype=torch.int64,
            device=key_cache.device,
        )

        cluster_keys = torch.zeros(
            batch_size,
            max_num_clusters,
            num_kv_heads,
            head_size,
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
        cluster_values = torch.zeros_like(cluster_keys)
        cluster_token_counts = torch.zeros(
            batch_size,
            max_num_clusters,
            dtype=torch.int32,
            device=key_cache.device,
        )
        cluster_mask = torch.zeros(
            batch_size,
            max_num_clusters,
            dtype=torch.bool,
            device=key_cache.device,
        )

        for row, record in enumerate(records):
            if record is None:
                continue

            valid_blocks = record.logical_block_ids < max_num_blocks
            logical_block_ids = record.logical_block_ids[valid_blocks]
            cluster_ids = record.block_cluster_ids[valid_blocks]

            indexed_block_mask[row, logical_block_ids] = True
            block_cluster_ids[row, logical_block_ids] = cluster_ids

            num_clusters = record.cluster_keys.shape[0]
            cluster_keys[row, :num_clusters].copy_(record.cluster_keys)
            cluster_values[row, :num_clusters].copy_(record.cluster_values)
            cluster_token_counts[row, :num_clusters].copy_(record.cluster_token_counts)
            cluster_mask[row, :num_clusters].copy_(record.cluster_token_counts > 0)

        packed = _PackedSegmentedIndex(
            indexed_block_mask=indexed_block_mask,
            block_cluster_ids=block_cluster_ids,
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            cluster_token_counts=cluster_token_counts,
            cluster_mask=cluster_mask,
        )
        self._proposal_packed_indices[layer_name] = packed
        return packed

    @staticmethod
    def _expand_cluster_mask_to_blocks(
        cluster_mask: torch.Tensor,
        packed: _PackedSegmentedIndex,
    ) -> torch.Tensor:
        safe_cluster_ids = packed.block_cluster_ids.clamp_min(0)
        selected_blocks = cluster_mask.gather(
            1,
            safe_cluster_ids,
        )
        return selected_blocks & packed.indexed_block_mask

    def _make_plan(
        self,
        logical_block_ids: torch.Tensor,
        valid_token_counts: torch.Tensor,
        forced_exact_mask: torch.Tensor,
        sparse_retrieval_clusters: torch.Tensor,
        sparse_estimation_clusters: torch.Tensor,
        expanded_retrieval_clusters: torch.Tensor,
        expanded_estimation_clusters: torch.Tensor,
        sparse_attn: torch.Tensor,
        expanded_attn: torch.Tensor,
        packed: _PackedSegmentedIndex,
    ) -> RetroSpecSegmentedSelectionPlan:
        sparse_retrieval_blocks = self._expand_cluster_mask_to_blocks(
            sparse_retrieval_clusters,
            packed,
        )
        expanded_retrieval_blocks = self._expand_cluster_mask_to_blocks(
            expanded_retrieval_clusters,
            packed,
        )

        sparse_exact_indices, sparse_exact_mask = self._pack_logical_indices(
            forced_exact_mask | sparse_retrieval_blocks,
            logical_block_ids,
        )
        expanded_exact_indices, expanded_exact_mask = self._pack_logical_indices(
            forced_exact_mask | expanded_retrieval_blocks,
            logical_block_ids,
        )

        max_num_clusters = packed.cluster_mask.shape[1]
        cluster_ids = torch.arange(
            max_num_clusters,
            dtype=torch.int64,
            device=packed.cluster_mask.device,
        )

        sparse_estimation_indices, sparse_estimation_mask = self._pack_logical_indices(
            sparse_estimation_clusters,
            cluster_ids,
        )
        expanded_estimation_indices, expanded_estimation_mask = (
            self._pack_logical_indices(
                expanded_estimation_clusters,
                cluster_ids,
            )
        )

        (
            sparse_estimation_keys,
            sparse_estimation_values,
            sparse_estimation_token_counts,
        ) = self._build_estimation_selection(
            packed.cluster_keys,
            packed.cluster_values,
            packed.cluster_token_counts,
            sparse_estimation_indices,
            sparse_estimation_mask,
        )
        (
            expanded_estimation_keys,
            expanded_estimation_values,
            expanded_estimation_token_counts,
        ) = self._build_estimation_selection(
            packed.cluster_keys,
            packed.cluster_values,
            packed.cluster_token_counts,
            expanded_estimation_indices,
            expanded_estimation_mask,
        )

        return RetroSpecSegmentedSelectionPlan(
            sparse_exact_indices=sparse_exact_indices,
            sparse_exact_mask=sparse_exact_mask,
            sparse_estimation_indices=sparse_estimation_indices,
            sparse_estimation_mask=sparse_estimation_mask,
            expanded_exact_indices=expanded_exact_indices,
            expanded_exact_mask=expanded_exact_mask,
            expanded_estimation_indices=expanded_estimation_indices,
            expanded_estimation_mask=expanded_estimation_mask,
            valid_token_counts=valid_token_counts,
            sparse_attn=sparse_attn,
            expanded_attn=expanded_attn,
            sparse_estimation_keys=sparse_estimation_keys,
            sparse_estimation_values=sparse_estimation_values,
            sparse_estimation_token_counts=(sparse_estimation_token_counts),
            expanded_estimation_keys=expanded_estimation_keys,
            expanded_estimation_values=expanded_estimation_values,
            expanded_estimation_token_counts=(expanded_estimation_token_counts),
        )

    def _materialize_segmented(
        self,
        plan: RetroSpecSegmentedSelectionPlan,
        level: RetroSpecAttentionLevel,
        block_table: torch.Tensor,
    ) -> RetroSpecAttentionSelection:
        if level == RetroSpecAttentionLevel.SPARSE:
            exact_indices = plan.sparse_exact_indices
            exact_mask = plan.sparse_exact_mask
            estimation_keys = plan.sparse_estimation_keys
            estimation_values = plan.sparse_estimation_values
            estimation_token_counts = plan.sparse_estimation_token_counts
            attention_mass = plan.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            exact_indices = plan.expanded_exact_indices
            exact_mask = plan.expanded_exact_mask
            estimation_keys = plan.expanded_estimation_keys
            estimation_values = plan.expanded_estimation_values
            estimation_token_counts = plan.expanded_estimation_token_counts
            attention_mass = plan.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        exact_block_table = block_table.gather(
            1,
            exact_indices,
        )
        exact_block_table.masked_fill_(~exact_mask, 0)
        exact_block_table = exact_block_table.to(torch.int32).contiguous()

        exact_token_counts = plan.valid_token_counts.gather(
            1,
            exact_indices,
        )
        exact_token_counts *= exact_mask
        exact_seq_lens = exact_token_counts.sum(dim=1).to(torch.int32).contiguous()

        return RetroSpecAttentionSelection(
            exact_block_table=exact_block_table,
            exact_seq_lens=exact_seq_lens,
            estimation_keys=estimation_keys,
            estimation_values=estimation_values,
            estimation_token_counts=estimation_token_counts,
            attention_mass=attention_mass,
            plan=plan,
        )

    def select_segmented(
        self,
        request_ids: Sequence[str],
        layer_name: str,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        active_mask: torch.Tensor,
        scale: float,
    ) -> RetroSpecAttentionSelection:
        self._validate_inputs(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            active_mask,
        )

        if tuple(request_ids) != self._proposal_request_ids:
            raise RuntimeError(
                "Segmented index request order does not match proposal order"
            )

        packed = self._pack_indices(
            layer_name,
            request_ids,
            key_cache,
            block_table,
        )

        (
            logical_block_ids,
            valid_block_mask,
            valid_token_counts,
            forced_exact_mask,
        ) = self._build_block_layout(block_table, seq_lens)

        # Blocks not covered by a complete clustered segment remain in the
        # exact steady zone.
        forced_exact_mask |= valid_block_mask & ~packed.indexed_block_mask

        cluster_scores = self._score_blocks(
            query,
            packed.cluster_keys,
            packed.cluster_mask,
            packed.cluster_token_counts,
            scale,
        )

        (
            sparse_retrieval_clusters,
            sparse_estimation_clusters,
            expanded_retrieval_clusters,
            expanded_estimation_clusters,
        ) = self._select_zone_masks(
            cluster_scores,
            packed.cluster_mask,
        )

        sparse_attn = (cluster_scores * sparse_retrieval_clusters).sum(dim=1)
        expanded_attn = (cluster_scores * expanded_retrieval_clusters).sum(dim=1)

        has_clusters = packed.cluster_mask.any(dim=1)
        sparse_attn = torch.where(
            has_clusters,
            sparse_attn,
            torch.ones_like(sparse_attn),
        )
        expanded_attn = torch.where(
            has_clusters,
            expanded_attn,
            torch.ones_like(expanded_attn),
        )

        sparse_attn = torch.where(
            active_mask,
            sparse_attn,
            torch.ones_like(sparse_attn),
        )
        expanded_attn = torch.where(
            active_mask,
            expanded_attn,
            torch.ones_like(expanded_attn),
        )

        plan = self._make_plan(
            logical_block_ids=logical_block_ids,
            valid_token_counts=valid_token_counts,
            forced_exact_mask=forced_exact_mask,
            sparse_retrieval_clusters=sparse_retrieval_clusters,
            sparse_estimation_clusters=sparse_estimation_clusters,
            expanded_retrieval_clusters=expanded_retrieval_clusters,
            expanded_estimation_clusters=expanded_estimation_clusters,
            sparse_attn=sparse_attn,
            expanded_attn=expanded_attn,
            packed=packed,
        )

        return self._materialize_segmented(
            plan,
            RetroSpecAttentionLevel.SPARSE,
            block_table,
        )

    def materialize(
        self,
        plan: RetroSpecSelectionPlan,
        level: RetroSpecAttentionLevel,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecAttentionSelection:
        del key_cache, value_cache

        if not isinstance(plan, RetroSpecSegmentedSelectionPlan):
            raise TypeError("Segmented index requires a segmented selection plan")

        return self._materialize_segmented(
            plan,
            level,
            block_table,
        )
