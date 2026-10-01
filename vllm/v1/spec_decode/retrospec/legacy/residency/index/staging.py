# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.v1.spec_decode.retrospec.legacy.residency.index.types import (
    RetroSpecClusterSummary,
    RetroSpecStagedClusterSummary,
    _PinnedSummarySlot,
)


class _IndexResidencyStagingMixin:
    def _get_offload_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        stream = self._offload_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._offload_streams[device] = stream

        return stream

    def _acquire_summary_slot(
        self,
        device: torch.device,
    ) -> _PinnedSummarySlot:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        with self._summary_slot_lock:
            slots = self._summary_slots.setdefault(device, [])
            for slot in slots:
                if not slot.in_use:
                    slot.in_use = True
                    return slot

            if len(slots) >= self.max_summary_slots:
                raise RuntimeError("RetroSpec pinned summary ring is exhausted")

            slot = _PinnedSummarySlot(
                pinned_memory=self._pinned_memory,
                in_use=True,
            )
            slots.append(slot)
            return slot

    def _release_summary_slot(
        self,
        slot: _PinnedSummarySlot | None,
    ) -> None:
        if slot is None:
            return

        with self._summary_slot_lock:
            if not slot.in_use:
                raise RuntimeError("Pinned summary slot was already released")
            slot.in_use = False

    def stage_cluster_summary(
        self,
        cluster_keys: torch.Tensor,
        cluster_values: torch.Tensor,
        cluster_token_counts: torch.Tensor,
    ) -> RetroSpecStagedClusterSummary:
        if cluster_keys.shape != cluster_values.shape:
            raise ValueError("Cluster key/value summary shapes must match")
        if cluster_keys.ndim != 3:
            raise ValueError(
                "Cluster summaries must have shape "
                "[num_kv_heads, num_clusters, head_size]"
            )
        if cluster_token_counts.shape != cluster_keys.shape[:2]:
            raise ValueError("Cluster token counts do not match cluster summaries")

        resident_summary = RetroSpecClusterSummary(
            cluster_keys=cluster_keys,
            cluster_values=cluster_values,
            cluster_token_counts=cluster_token_counts,
        )

        if cluster_keys.device.type != "cuda" or not self.pin_memory:
            host_keys = torch.empty_like(cluster_keys, device="cpu", pin_memory=False)
            host_values = torch.empty_like(
                cluster_values, device="cpu", pin_memory=False
            )
            host_counts = torch.empty_like(
                cluster_token_counts, device="cpu", pin_memory=False
            )
            host_keys.copy_(cluster_keys, non_blocking=False)
            host_values.copy_(cluster_values, non_blocking=False)
            host_counts.copy_(cluster_token_counts, non_blocking=False)
            return RetroSpecStagedClusterSummary(
                cluster_keys=host_keys,
                cluster_values=host_values,
                cluster_token_counts=host_counts,
                resident_summary=resident_summary,
                ready_event=None,
            )

        slot = self._acquire_summary_slot(cluster_keys.device)
        try:
            host_keys, host_values, host_counts = slot.reserve(
                cluster_keys,
                cluster_values,
                cluster_token_counts,
            )
            device = cluster_keys.device
            transfer_stream = self._get_offload_stream(device)
            current_stream = torch.cuda.current_stream(device)
            transfer_stream.wait_stream(current_stream)

            with torch.cuda.stream(transfer_stream):
                timer = (
                    None
                    if self.performance_stats is None
                    else self.performance_stats.start_cuda_timer(
                        "prefill_summary_d2h", transfer_stream
                    )
                )
                try:
                    host_keys.copy_(cluster_keys, non_blocking=True)
                    host_values.copy_(cluster_values, non_blocking=True)
                    host_counts.copy_(cluster_token_counts, non_blocking=True)
                finally:
                    if self.performance_stats is not None:
                        self.performance_stats.stop_cuda_timer(timer, transfer_stream)
                ready_event = torch.cuda.Event()
                ready_event.record(transfer_stream)
        except BaseException:
            slot.release_storage()
            self._release_summary_slot(slot)
            raise

        return RetroSpecStagedClusterSummary(
            cluster_keys=host_keys,
            cluster_values=host_values,
            cluster_token_counts=host_counts,
            resident_summary=resident_summary,
            ready_event=ready_event,
            staging_slot=slot,
        )

    def finish_cluster_summary(
        self,
        staged: RetroSpecStagedClusterSummary,
    ) -> RetroSpecClusterSummary:
        try:
            if staged.ready_event is not None:
                staged.ready_event.synchronize()

            if staged.staging_slot is None:
                return RetroSpecClusterSummary(
                    cluster_keys=staged.cluster_keys,
                    cluster_values=staged.cluster_values,
                    cluster_token_counts=staged.cluster_token_counts,
                )

            cluster_keys = torch.empty_like(
                staged.cluster_keys, device="cpu", pin_memory=False
            )
            cluster_values = torch.empty_like(
                staged.cluster_values, device="cpu", pin_memory=False
            )
            cluster_token_counts = torch.empty_like(
                staged.cluster_token_counts, device="cpu", pin_memory=False
            )
            cluster_keys.copy_(staged.cluster_keys)
            cluster_values.copy_(staged.cluster_values)
            cluster_token_counts.copy_(staged.cluster_token_counts)
            return RetroSpecClusterSummary(
                cluster_keys=cluster_keys,
                cluster_values=cluster_values,
                cluster_token_counts=cluster_token_counts,
            )
        finally:
            self._release_summary_slot(staged.staging_slot)

    def discard_cluster_summary(self, staged: RetroSpecStagedClusterSummary) -> None:
        try:
            if staged.ready_event is not None:
                staged.ready_event.synchronize()
        finally:
            self._release_summary_slot(staged.staging_slot)

    def close(self) -> None:
        with self._summary_slot_lock:
            for slots in self._summary_slots.values():
                for slot in slots:
                    if slot.in_use:
                        raise RuntimeError(
                            "Cannot close GPU index residency with an active "
                            "summary transfer"
                        )
                    slot.release_storage()
            self._summary_slots.clear()
