# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .clustering import segmented_kmeans_assignments
from .index import (
    RetroSpecAttentionLevel,
    RetroSpecBlockIndex,
    RetroSpecSelectionPlan,
)


@dataclass(frozen=True)
class RetroSpecTokenSelectionPlan:
    sparse_exact_token_indices: torch.Tensor
    sparse_exact_token_mask: torch.Tensor
    sparse_estimation_keys: torch.Tensor
    sparse_estimation_values: torch.Tensor
    sparse_estimation_token_counts: torch.Tensor

    expanded_exact_token_indices: torch.Tensor
    expanded_exact_token_mask: torch.Tensor
    expanded_estimation_keys: torch.Tensor
    expanded_estimation_values: torch.Tensor
    expanded_estimation_token_counts: torch.Tensor

    sparse_attn: torch.Tensor
    expanded_attn: torch.Tensor


@dataclass(frozen=True)
class RetroSpecTokenAttentionSelection:
    exact_keys: torch.Tensor
    exact_values: torch.Tensor
    exact_token_mask: torch.Tensor
    exact_token_counts: torch.Tensor

    estimation_keys: torch.Tensor
    estimation_values: torch.Tensor
    estimation_token_counts: torch.Tensor

    attention_mass: torch.Tensor
    plan: RetroSpecTokenSelectionPlan

    @property
    def hit_attn(self) -> torch.Tensor:
        return self.plan.sparse_attn


@dataclass(frozen=True)
class _RequestLayerSegment:
    logical_token_ids: torch.Tensor
    token_cluster_ids: torch.Tensor

    cluster_start: int
    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor


@dataclass
class _RequestLayerIndex:
    segments: list[_RequestLayerSegment]
    num_clusters: int
    indexed_end: int


@dataclass(frozen=True)
class _PackedSegmentedIndex:
    indexed_token_mask: torch.Tensor
    token_cluster_ids: torch.Tensor

    cluster_keys: torch.Tensor
    cluster_values: torch.Tensor
    cluster_token_counts: torch.Tensor
    cluster_mask: torch.Tensor


@dataclass(frozen=True)
class _PackedSegmentedIndexCacheEntry:
    request_ids: tuple[str, ...]
    max_num_tokens: int
    packed: _PackedSegmentedIndex


class RetroSpecSegmentedTokenIndex(RetroSpecBlockIndex):
    """Reference token-level segmented index backed by the main KV cache."""

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
        if blocks_per_cluster <= 0:
            raise ValueError("blocks_per_cluster must be positive")
        if num_kmeans_iterations <= 0:
            raise ValueError("num_kmeans_iterations must be positive")

        tokens_per_cluster = blocks_per_cluster * block_size
        if segment_size_tokens % tokens_per_cluster != 0:
            raise ValueError(
                "segment_size_tokens must be divisible by "
                "blocks_per_cluster * block_size"
            )

        self.segment_size_tokens = segment_size_tokens
        self.tokens_per_cluster = tokens_per_cluster
        self.num_kmeans_iterations = num_kmeans_iterations

        # layer_name -> request_id -> token-level index
        self._indices: dict[str, dict[str, _RequestLayerIndex]] = {}

        self._proposal_active = False
        self._proposal_request_ids: tuple[str, ...] = ()

        # Each layer retains only the most recently packed request batch.
        self._packed_index_cache: dict[
            str,
            _PackedSegmentedIndexCacheEntry,
        ] = {}

    def _desired_indexed_end(self, seq_len: int) -> int:
        """Return the exclusive logical-token boundary covered by clustering."""
        full_block_count = seq_len // self.block_size

        # The first block remains an exact attention sink. Recent complete
        # blocks and the current partial block remain in the exact steady zone.
        stable_end_block = max(
            full_block_count - self.num_recent_blocks,
            1,
        )
        indexable_tokens = (stable_end_block - 1) * self.block_size

        complete_segments = indexable_tokens // self.segment_size_tokens
        return self.block_size + complete_segments * self.segment_size_tokens

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
        request_ids = tuple(request_ids)

        for layer_name, layer_indices in self._indices.items():
            layer_changed = False

            for request_id in request_ids:
                if layer_indices.pop(request_id, None) is not None:
                    layer_changed = True

            if layer_changed:
                self._packed_index_cache.pop(layer_name, None)

    def begin_proposal(self, request_ids: Sequence[str]) -> None:
        if self._proposal_active:
            raise RuntimeError("Segmented token index proposal is already active")

        self._proposal_active = True
        self._proposal_request_ids = tuple(request_ids)

    def end_proposal(self) -> None:
        if not self._proposal_active:
            raise RuntimeError("Segmented token index proposal is not active")

        self._proposal_active = False
        self._proposal_request_ids = ()

    def _empty_index(
        self,
        indexed_end: int | None = None,
    ) -> _RequestLayerIndex:
        if indexed_end is None:
            indexed_end = self.block_size

        return _RequestLayerIndex(
            segments=[],
            num_clusters=0,
            indexed_end=indexed_end,
        )

    @staticmethod
    def _cluster_means(
        vectors: torch.Tensor,
        assignments: torch.Tensor,
        cluster_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce [heads, tokens, dim] vectors into per-head cluster means."""
        num_kv_heads, _, head_size = vectors.shape
        num_clusters = cluster_counts.shape[1]

        expanded_assignments = assignments.unsqueeze(-1).expand(
            -1,
            -1,
            head_size,
        )

        cluster_sums = torch.zeros(
            num_kv_heads,
            num_clusters,
            head_size,
            dtype=torch.float32,
            device=vectors.device,
        )
        cluster_sums.scatter_add_(
            dim=1,
            index=expanded_assignments,
            src=vectors.float(),
        )

        return (
            cluster_sums / cluster_counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
        ).to(vectors.dtype)

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
        """Build missing complete token segments from the main paged KV cache."""
        if len(request_ids) != len(seq_lens):
            raise ValueError("request_ids and seq_lens must have equal length")
        if block_table.shape[0] != len(request_ids):
            raise ValueError("block_table batch size does not match request_ids")
        if key_cache.shape != value_cache.shape:
            raise ValueError("key_cache and value_cache must have equal shapes")
        if key_cache.shape[1] != self.block_size:
            raise ValueError(
                f"KV cache block size {key_cache.shape[1]} does not match "
                f"configured block size {self.block_size}"
            )

        layer_indices = self._indices.setdefault(layer_name, {})
        layer_changed = False

        for row in rows:
            request_id = request_ids[row]
            seq_len = seq_lens[row]
            desired_end = self._desired_indexed_end(seq_len)

            record = layer_indices.get(request_id)
            if record is not None and desired_end < record.indexed_end:
                record = self._empty_index()
                layer_indices[request_id] = record
                layer_changed = True

            indexed_start = self.block_size if record is None else record.indexed_end

            if desired_end <= indexed_start:
                if record is None:
                    layer_indices[request_id] = self._empty_index()
                    layer_changed = True
                continue

            num_new_tokens = desired_end - indexed_start
            if num_new_tokens % self.segment_size_tokens != 0:
                raise RuntimeError(
                    "New indexed region must contain complete token segments"
                )

            first_logical_block = indexed_start // self.block_size
            logical_block_end = desired_end // self.block_size

            logical_block_ids = torch.arange(
                first_logical_block,
                logical_block_end,
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

            key_blocks = key_cache.index_select(0, physical_block_ids)
            value_blocks = value_cache.index_select(0, physical_block_ids)

            num_kv_heads = key_cache.shape[2]
            head_size = key_cache.shape[3]

            token_keys = (
                key_blocks.reshape(
                    num_new_tokens,
                    num_kv_heads,
                    head_size,
                )
                .transpose(0, 1)
                .contiguous()
            )
            token_values = (
                value_blocks.reshape(
                    num_new_tokens,
                    num_kv_heads,
                    head_size,
                )
                .transpose(0, 1)
                .contiguous()
            )

            local_assignments, cluster_token_counts = segmented_kmeans_assignments(
                features=token_keys,
                segment_size=self.segment_size_tokens,
                items_per_cluster=self.tokens_per_cluster,
                num_iterations=self.num_kmeans_iterations,
            )

            cluster_keys = self._cluster_means(
                token_keys,
                local_assignments,
                cluster_token_counts,
            )
            cluster_values = self._cluster_means(
                token_values,
                local_assignments,
                cluster_token_counts,
            )

            if record is None:
                record = self._empty_index()
                layer_indices[request_id] = record

            cluster_start = record.num_clusters
            token_cluster_ids = local_assignments + cluster_start

            logical_token_ids = torch.arange(
                indexed_start,
                desired_end,
                dtype=torch.int64,
                device=block_table.device,
            )

            record.segments.append(
                _RequestLayerSegment(
                    logical_token_ids=logical_token_ids,
                    token_cluster_ids=token_cluster_ids,
                    cluster_start=cluster_start,
                    cluster_keys=cluster_keys,
                    cluster_values=cluster_values,
                    cluster_token_counts=cluster_token_counts,
                )
            )
            record.num_clusters += cluster_token_counts.shape[1]
            record.indexed_end = desired_end
            layer_changed = True

        if layer_changed:
            self._packed_index_cache.pop(layer_name, None)

    def _pack_indices(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> _PackedSegmentedIndex:
        batch_size, max_num_blocks = block_table.shape
        max_num_tokens = max_num_blocks * self.block_size
        request_ids = tuple(request_ids)

        if len(request_ids) != batch_size:
            raise ValueError("request_ids batch size does not match block_table")

        cached = self._packed_index_cache.get(layer_name)
        if (
            cached is not None
            and cached.request_ids == request_ids
            and cached.max_num_tokens == max_num_tokens
        ):
            return cached.packed

        layer_indices = self._indices.get(layer_name, {})
        records = [layer_indices.get(request_id) for request_id in request_ids]

        max_num_clusters = max(
            1,
            max(
                (
                    record.num_clusters if record is not None else 0
                    for record in records
                ),
                default=0,
            ),
        )

        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        indexed_token_mask = torch.zeros(
            batch_size,
            max_num_tokens,
            dtype=torch.bool,
            device=key_cache.device,
        )
        token_cluster_ids = torch.full(
            (batch_size, num_kv_heads, max_num_tokens),
            -1,
            dtype=torch.int64,
            device=key_cache.device,
        )

        cluster_keys = torch.zeros(
            batch_size,
            num_kv_heads,
            max_num_clusters,
            head_size,
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
        cluster_values = torch.zeros_like(cluster_keys)
        cluster_token_counts = torch.zeros(
            batch_size,
            num_kv_heads,
            max_num_clusters,
            dtype=torch.int32,
            device=key_cache.device,
        )
        cluster_mask = torch.zeros(
            batch_size,
            num_kv_heads,
            max_num_clusters,
            dtype=torch.bool,
            device=key_cache.device,
        )

        for row, record in enumerate(records):
            if record is None:
                continue

            for segment in record.segments:
                valid_tokens = segment.logical_token_ids < max_num_tokens
                logical_token_ids = segment.logical_token_ids[valid_tokens]

                indexed_token_mask[row, logical_token_ids] = True
                token_cluster_ids[row, :, logical_token_ids] = (
                    segment.token_cluster_ids[:, valid_tokens]
                )

                cluster_start = segment.cluster_start
                cluster_end = cluster_start + segment.cluster_keys.shape[1]

                cluster_keys[row, :, cluster_start:cluster_end].copy_(
                    segment.cluster_keys
                )
                cluster_values[row, :, cluster_start:cluster_end].copy_(
                    segment.cluster_values
                )
                cluster_token_counts[row, :, cluster_start:cluster_end].copy_(
                    segment.cluster_token_counts
                )
                cluster_mask[row, :, cluster_start:cluster_end].copy_(
                    segment.cluster_token_counts > 0
                )

        packed = _PackedSegmentedIndex(
            indexed_token_mask=indexed_token_mask,
            token_cluster_ids=token_cluster_ids,
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            cluster_token_counts=cluster_token_counts,
            cluster_mask=cluster_mask,
        )
        self._packed_index_cache[layer_name] = _PackedSegmentedIndexCacheEntry(
            request_ids=request_ids,
            max_num_tokens=max_num_tokens,
            packed=packed,
        )
        return packed

    def _build_token_layout(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_num_tokens = block_table.shape[1] * self.block_size
        logical_token_ids = torch.arange(
            max_num_tokens,
            dtype=torch.int64,
            device=block_table.device,
        )

        bounded_seq_lens = seq_lens.to(torch.int64).clamp(
            min=1,
            max=max_num_tokens,
        )
        valid_token_mask = logical_token_ids.unsqueeze(0) < bounded_seq_lens.unsqueeze(
            1
        )

        valid_block_counts = torch.div(
            bounded_seq_lens + self.block_size - 1,
            self.block_size,
            rounding_mode="floor",
        )
        recent_start_blocks = (valid_block_counts - self.num_recent_blocks).clamp_min(0)

        logical_block_ids = torch.div(
            logical_token_ids,
            self.block_size,
            rounding_mode="floor",
        )

        forced_exact_mask = valid_token_mask & (
            (logical_block_ids.unsqueeze(0) == 0)
            | (logical_block_ids.unsqueeze(0) >= recent_start_blocks.unsqueeze(1))
        )

        return logical_token_ids, valid_token_mask, forced_exact_mask

    @staticmethod
    def _score_clusters(
        query: torch.Tensor,
        cluster_keys: torch.Tensor,
        cluster_mask: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        batch_size, num_query_heads, head_size = query.shape
        num_kv_heads = cluster_keys.shape[1]

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
            "bhgd,bhcd->bhgc",
            grouped_query,
            cluster_keys.float(),
        )
        logits *= scale
        logits += torch.log(cluster_token_counts.clamp_min(1).float()).unsqueeze(2)
        logits.masked_fill_(
            ~cluster_mask.unsqueeze(2),
            float("-inf"),
        )

        has_clusters = cluster_mask.any(dim=-1)
        safe_logits = torch.where(
            has_clusters[:, :, None, None],
            logits,
            torch.zeros_like(logits),
        )

        probabilities = torch.softmax(safe_logits, dim=-1)
        probabilities.masked_fill_(~cluster_mask.unsqueeze(2), 0.0)

        # The same KV head serves a group of query heads. Averaging their
        # probabilities preserves the ranking used by RetroInfer.
        return probabilities.mean(dim=2)

    def _select_cluster_zones(
        self,
        cluster_scores: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, num_clusters = cluster_scores.shape

        flat_scores = cluster_scores.reshape(
            batch_size * num_kv_heads,
            num_clusters,
        )
        flat_mask = cluster_mask.reshape(
            batch_size * num_kv_heads,
            num_clusters,
        )

        zone_masks = self._select_zone_masks(flat_scores, flat_mask)

        return tuple(
            mask.view(batch_size, num_kv_heads, num_clusters) for mask in zone_masks
        )

    @staticmethod
    def _expand_cluster_mask_to_tokens(
        selected_clusters: torch.Tensor,
        packed: _PackedSegmentedIndex,
    ) -> torch.Tensor:
        safe_cluster_ids = packed.token_cluster_ids.clamp_min(0)
        selected_tokens = selected_clusters.gather(
            dim=2,
            index=safe_cluster_ids,
        )

        return selected_tokens & packed.indexed_token_mask.unsqueeze(1)

    @staticmethod
    def _pack_mask_indices(
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack True positions along the final dimension."""
        max_num_items = mask.shape[-1]
        leading_shape = mask.shape[:-1]

        flat_mask = mask.reshape(-1, max_num_items)
        logical_ids = torch.arange(
            max_num_items,
            dtype=torch.int64,
            device=mask.device,
        ).expand(flat_mask.shape[0], -1)

        sentinel = torch.full_like(logical_ids, max_num_items)
        sorted_indices = (
            torch.where(
                flat_mask,
                logical_ids,
                sentinel,
            )
            .sort(dim=1)
            .values
        )

        packed_counts = flat_mask.sum(dim=1)
        max_packed_count = (
            int(packed_counts.max().item()) if packed_counts.numel() else 0
        )

        if max_packed_count == 0:
            empty_shape = (*leading_shape, 0)
            return (
                torch.empty(
                    empty_shape,
                    dtype=torch.int64,
                    device=mask.device,
                ),
                torch.empty(
                    empty_shape,
                    dtype=torch.bool,
                    device=mask.device,
                ),
            )

        packed_indices = sorted_indices[:, :max_packed_count]
        packed_indices.clamp_(min=0, max=max_num_items - 1)

        packed_positions = torch.arange(
            max_packed_count,
            dtype=torch.int64,
            device=mask.device,
        )
        packed_mask = packed_positions.unsqueeze(0) < packed_counts.unsqueeze(1)

        output_shape = (*leading_shape, max_packed_count)
        return (
            packed_indices.view(output_shape).contiguous(),
            packed_mask.view(output_shape).contiguous(),
        )

    @staticmethod
    def _build_estimation_selection(
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        packed_indices: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, max_num_clusters, head_size = cluster_keys.shape
        del max_num_clusters

        max_selected_clusters = packed_indices.shape[2]
        vector_indices = packed_indices.unsqueeze(-1).expand(
            batch_size,
            num_kv_heads,
            max_selected_clusters,
            head_size,
        )

        estimation_keys = cluster_keys.gather(2, vector_indices)
        estimation_values = cluster_values.gather(2, vector_indices)
        estimation_token_counts = cluster_token_counts.gather(
            2,
            packed_indices,
        )

        estimation_keys.masked_fill_(
            ~packed_mask.unsqueeze(-1),
            0.0,
        )
        estimation_values.masked_fill_(
            ~packed_mask.unsqueeze(-1),
            0.0,
        )
        estimation_token_counts.masked_fill_(~packed_mask, 0)

        return (
            estimation_keys.contiguous(),
            estimation_values.contiguous(),
            estimation_token_counts.to(torch.int32).contiguous(),
        )

    def _make_plan(
        self,
        forced_exact_mask: torch.Tensor,
        sparse_retrieval_clusters: torch.Tensor,
        sparse_estimation_clusters: torch.Tensor,
        expanded_retrieval_clusters: torch.Tensor,
        expanded_estimation_clusters: torch.Tensor,
        sparse_attn: torch.Tensor,
        expanded_attn: torch.Tensor,
        packed: _PackedSegmentedIndex,
    ) -> RetroSpecTokenSelectionPlan:
        sparse_retrieval_tokens = self._expand_cluster_mask_to_tokens(
            sparse_retrieval_clusters,
            packed,
        )
        expanded_retrieval_tokens = self._expand_cluster_mask_to_tokens(
            expanded_retrieval_clusters,
            packed,
        )

        per_head_forced_exact = forced_exact_mask.unsqueeze(1).expand(
            -1,
            packed.cluster_keys.shape[1],
            -1,
        )

        sparse_exact_token_indices, sparse_exact_token_mask = self._pack_mask_indices(
            per_head_forced_exact | sparse_retrieval_tokens
        )
        expanded_exact_token_indices, expanded_exact_token_mask = (
            self._pack_mask_indices(per_head_forced_exact | expanded_retrieval_tokens)
        )

        sparse_estimation_indices, sparse_estimation_mask = self._pack_mask_indices(
            sparse_estimation_clusters
        )
        expanded_estimation_indices, expanded_estimation_mask = self._pack_mask_indices(
            expanded_estimation_clusters
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

        return RetroSpecTokenSelectionPlan(
            sparse_exact_token_indices=sparse_exact_token_indices,
            sparse_exact_token_mask=sparse_exact_token_mask,
            sparse_estimation_keys=sparse_estimation_keys,
            sparse_estimation_values=sparse_estimation_values,
            sparse_estimation_token_counts=sparse_estimation_token_counts,
            expanded_exact_token_indices=expanded_exact_token_indices,
            expanded_exact_token_mask=expanded_exact_token_mask,
            expanded_estimation_keys=expanded_estimation_keys,
            expanded_estimation_values=expanded_estimation_values,
            expanded_estimation_token_counts=expanded_estimation_token_counts,
            sparse_attn=sparse_attn,
            expanded_attn=expanded_attn,
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

    def _materialize_token_selection(
        self,
        plan: RetroSpecTokenSelectionPlan,
        level: RetroSpecAttentionLevel,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecTokenAttentionSelection:
        if level == RetroSpecAttentionLevel.SPARSE:
            exact_token_indices = plan.sparse_exact_token_indices
            exact_token_mask = plan.sparse_exact_token_mask
            estimation_keys = plan.sparse_estimation_keys
            estimation_values = plan.sparse_estimation_values
            estimation_token_counts = plan.sparse_estimation_token_counts
            attention_mass = plan.sparse_attn
        elif level == RetroSpecAttentionLevel.EXPANDED:
            exact_token_indices = plan.expanded_exact_token_indices
            exact_token_mask = plan.expanded_exact_token_mask
            estimation_keys = plan.expanded_estimation_keys
            estimation_values = plan.expanded_estimation_values
            estimation_token_counts = plan.expanded_estimation_token_counts
            attention_mass = plan.expanded_attn
        else:
            raise ValueError(f"Unsupported RetroSpec attention level: {level}")

        exact_keys = self._gather_selected_tokens(
            key_cache,
            block_table,
            exact_token_indices,
            exact_token_mask,
        )
        exact_values = self._gather_selected_tokens(
            value_cache,
            block_table,
            exact_token_indices,
            exact_token_mask,
        )
        exact_token_counts = exact_token_mask.sum(dim=2).to(torch.int32)

        return RetroSpecTokenAttentionSelection(
            exact_keys=exact_keys,
            exact_values=exact_values,
            exact_token_mask=exact_token_mask,
            exact_token_counts=exact_token_counts,
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
    ) -> RetroSpecTokenAttentionSelection:
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
                "Segmented token index request order does not match proposal order"
            )

        packed = self._pack_indices(
            layer_name,
            request_ids,
            key_cache,
            block_table,
        )

        _, valid_token_mask, forced_exact_mask = self._build_token_layout(
            block_table,
            seq_lens,
        )

        # All valid tokens not covered by a complete clustered segment remain
        # in the exact steady zone.
        forced_exact_mask |= valid_token_mask & ~packed.indexed_token_mask

        cluster_scores = self._score_clusters(
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
        ) = self._select_cluster_zones(
            cluster_scores,
            packed.cluster_mask,
        )

        sparse_attn_by_head = (cluster_scores * sparse_retrieval_clusters).sum(dim=2)
        expanded_attn_by_head = (cluster_scores * expanded_retrieval_clusters).sum(
            dim=2
        )

        has_clusters = packed.cluster_mask.any(dim=2)
        sparse_attn_by_head = torch.where(
            has_clusters,
            sparse_attn_by_head,
            torch.ones_like(sparse_attn_by_head),
        )
        expanded_attn_by_head = torch.where(
            has_clusters,
            expanded_attn_by_head,
            torch.ones_like(expanded_attn_by_head),
        )

        sparse_attn = sparse_attn_by_head.mean(dim=1)
        expanded_attn = expanded_attn_by_head.mean(dim=1)

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
            forced_exact_mask=forced_exact_mask,
            sparse_retrieval_clusters=sparse_retrieval_clusters,
            sparse_estimation_clusters=sparse_estimation_clusters,
            expanded_retrieval_clusters=expanded_retrieval_clusters,
            expanded_estimation_clusters=expanded_estimation_clusters,
            sparse_attn=sparse_attn,
            expanded_attn=expanded_attn,
            packed=packed,
        )

        return self._materialize_token_selection(
            plan,
            RetroSpecAttentionLevel.SPARSE,
            key_cache,
            value_cache,
            block_table,
        )

    def materialize(
        self,
        plan: RetroSpecSelectionPlan | RetroSpecTokenSelectionPlan,
        level: RetroSpecAttentionLevel,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecTokenAttentionSelection:
        if not isinstance(plan, RetroSpecTokenSelectionPlan):
            raise TypeError("Segmented token index requires a token selection plan")

        return self._materialize_token_selection(
            plan,
            level,
            key_cache,
            value_cache,
            block_table,
        )
