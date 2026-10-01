# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import Literal, TypeAlias

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

RetroSpecReplayMode: TypeAlias = Literal[
    "off",
    "trace",
    "freeze_resident",
    "ready_selected",
]

RETROSPEC_REPLAY_MODES = frozenset(
    {
        "off",
        "trace",
        "freeze_resident",
        "ready_selected",
    }
)


class RetroSpecSelectionProvenanceTracer:
    """Record logical selection and physical resident-source provenance.

    Every enabled mode is intentionally synchronous. These diagnostics must
    remain disabled during quiet performance measurements.
    """

    def __init__(self, mode: RetroSpecReplayMode) -> None:
        if mode not in RETROSPEC_REPLAY_MODES:
            raise ValueError(f"Unsupported RetroSpec replay mode: {mode!r}")
        self.mode = mode

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @staticmethod
    def _tensor_digest(tensors: Sequence[torch.Tensor]) -> str:
        digest = sha256()
        for tensor in tensors:
            cpu_tensor = tensor.detach().to(device="cpu").contiguous()
            digest.update(str(cpu_tensor.dtype).encode())
            digest.update(str(tuple(cpu_tensor.shape)).encode())
            digest.update(cpu_tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()[:16]

    @classmethod
    def _row_digests(cls, tensor: torch.Tensor) -> tuple[str, ...]:
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        return tuple(cls._tensor_digest((row,)) for row in cpu_tensor)

    def record_index_segment(
        self,
        request_id: str,
        layer_name: str,
        indexed_start: int,
        indexed_end: int,
        cluster_start: int,
        assignments: torch.Tensor,
        cluster_sizes: torch.Tensor,
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "phase": "index_segment",
            "mode": self.mode,
            "request_id": request_id,
            "layer_name": layer_name,
            "indexed_start": indexed_start,
            "indexed_end": indexed_end,
            "cluster_start": cluster_start,
            "cluster_end": cluster_start + cluster_sizes.shape[1],
            "num_clusters": cluster_sizes.shape[1],
            "checksum": self._tensor_digest(
                (
                    assignments,
                    cluster_sizes,
                    cluster_keys,
                    cluster_values,
                    token_offsets_in_cluster,
                )
            ),
        }
        logger.info(
            "RetroSpec selection provenance: %s",
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def record_draft_selection(
        self,
        request_ids: Sequence[str],
        layer_name: str,
        proposal_round: int,
        draft_step: int,
        snapshot: str,
        physical_source: str,
        index_revisions: Sequence[int],
        active_mask: torch.Tensor,
        query: torch.Tensor,
        request_slot_ids: torch.Tensor,
        request_slot_generations: torch.Tensor,
        sparse_exact_cluster_indices: torch.Tensor,
        sparse_estimation_cluster_indices: torch.Tensor,
        exact_cluster_handles: torch.Tensor,
        candidate_counts: torch.Tensor,
        selected_cluster_counts: torch.Tensor,
        hit_cluster_counts: torch.Tensor,
        miss_cluster_counts: torch.Tensor,
        hit_gate_ready: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        num_requests = len(request_ids)
        if len(index_revisions) != num_requests:
            raise ValueError("Index revisions must match request IDs")
        if active_mask.shape != (num_requests,):
            raise ValueError("Selection active mask does not match request IDs")

        ordered_topk = torch.cat(
            (
                sparse_exact_cluster_indices,
                sparse_estimation_cluster_indices,
            ),
            dim=2,
        )
        valid_exact = sparse_exact_cluster_indices >= 0
        stable_handles = torch.where(
            valid_exact,
            exact_cluster_handles,
            torch.full_like(exact_cluster_handles, -1),
        )

        query_checksums = self._row_digests(query)
        topk_checksums = self._row_digests(ordered_topk)
        handle_checksums = self._row_digests(stable_handles)

        integer_columns = (
            request_slot_ids.to(torch.int64),
            request_slot_generations.to(torch.int64),
            candidate_counts.sum(dim=1, dtype=torch.int64),
            selected_cluster_counts.sum(dim=1, dtype=torch.int64),
            hit_cluster_counts.sum(dim=1, dtype=torch.int64),
            miss_cluster_counts.sum(dim=1, dtype=torch.int64),
            hit_gate_ready.sum(dim=1, dtype=torch.int64),
        )
        integer_rows = torch.stack(integer_columns, dim=1).detach().cpu().tolist()
        active_rows = active_mask.detach().cpu().tolist()

        records: list[dict[str, object]] = []
        for request_index, active in enumerate(active_rows):
            if not active:
                continue

            (
                request_slot,
                request_slot_generation,
                candidate_count,
                selected_count,
                hit_count,
                miss_count,
                gate_ready_heads,
            ) = integer_rows[request_index]
            records.append(
                {
                    "request_id": request_ids[request_index],
                    "request_index": request_index,
                    "index_revision": index_revisions[request_index],
                    "request_slot": request_slot,
                    "request_slot_generation": request_slot_generation,
                    "query_checksum": query_checksums[request_index],
                    "ordered_topk_checksum": topk_checksums[request_index],
                    "stable_handle_checksum": handle_checksums[request_index],
                    "candidate_count": candidate_count,
                    "selected_count": selected_count,
                    "hit_count": hit_count,
                    "miss_count": miss_count,
                    "gate_ready_heads": gate_ready_heads,
                }
            )

        if not records:
            return

        payload = {
            "phase": "draft_selection",
            "mode": self.mode,
            "proposal_round": proposal_round,
            "draft_step": draft_step,
            "layer_name": layer_name,
            "snapshot": snapshot,
            "physical_source": physical_source,
            "records": records,
        }
        logger.info(
            "RetroSpec selection provenance: %s",
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
