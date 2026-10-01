# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.v1.spec_decode.retrospec.legacy.residency.index.types import (
    RetroSpecClusterSummary,
    RetroSpecResidentSegment,
    RetroSpecResidentTableBinding,
    _PackedLayerState,
    _ResidentRequestState,
    _ResidentSpanTransaction,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_kernels import (
    publish_resident_table_bindings,
)


class _IndexResidencySegmentsMixin:
    def build_resident_segment(
        self,
        layer_name: str,
        request_id: str,
        indexed_start: int,
        indexed_end: int,
        cluster_start: int,
        resident_summary: RetroSpecClusterSummary,
        cluster_token_counts_cpu: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_page_ids: torch.Tensor,
        cluster_page_token_counts: torch.Tensor,
    ) -> RetroSpecResidentSegment:
        if indexed_start < 0 or indexed_end <= indexed_start:
            raise ValueError("Resident segment token range is invalid")
        if cluster_start < 0:
            raise ValueError("Resident segment cluster offset must be non-negative")

        summary = resident_summary
        device = summary.cluster_keys.device

        if summary.cluster_keys.shape != summary.cluster_values.shape:
            raise ValueError("Resident cluster key/value shapes must match")
        if summary.cluster_keys.ndim != 3:
            raise ValueError(
                "Resident cluster summaries must have shape "
                "[num_kv_heads, num_clusters, head_size]"
            )
        if summary.cluster_token_counts.shape != summary.cluster_keys.shape[:2]:
            raise ValueError("Resident cluster counts do not match summaries")
        if cluster_ids.shape != summary.cluster_token_counts.shape:
            raise ValueError("Resident cluster IDs do not match cluster counts")
        if cluster_page_ids.shape != cluster_page_token_counts.shape:
            raise ValueError("Resident cluster page metadata shapes must match")
        if cluster_page_ids.shape[:-1] != cluster_ids.shape:
            raise ValueError("Resident cluster pages do not match cluster IDs")

        cluster_page_ids_cpu = cluster_page_ids.to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        cluster_page_token_counts_cpu = cluster_page_token_counts.to(
            device="cpu", dtype=torch.int32
        ).contiguous()
        cluster_ids_cpu = cluster_ids.to(device="cpu", dtype=torch.int64)
        cluster_token_counts_cpu = cluster_token_counts_cpu.to(
            device="cpu", dtype=torch.int32
        ).contiguous()
        valid_clusters = cluster_ids_cpu >= 0
        positive_clusters = cluster_token_counts_cpu > 0
        valid_pages = cluster_page_ids_cpu >= 0
        positive_pages = cluster_page_token_counts_cpu > 0

        if not torch.equal(valid_clusters, positive_clusters):
            raise ValueError(
                "Resident cluster IDs and token counts describe different clusters"
            )
        if not torch.equal(valid_pages, positive_pages):
            raise ValueError(
                "Resident cluster page IDs and token counts describe different pages"
            )

        cluster_page_counts_cpu = (cluster_page_ids_cpu >= 0).sum(
            dim=-1, dtype=torch.int32
        )
        if not torch.equal(cluster_page_counts_cpu > 0, valid_clusters):
            raise ValueError("Resident clusters and page descriptors do not match")
        if not torch.equal(
            cluster_page_token_counts_cpu.sum(dim=-1),
            cluster_token_counts_cpu,
        ):
            raise ValueError(
                "Resident cluster token counts do not match page descriptors"
            )

        resident_token_counts = summary.cluster_token_counts.to(
            device=device, dtype=torch.int32
        ).contiguous()

        return RetroSpecResidentSegment(
            layer_name=layer_name,
            request_id=request_id,
            indexed_start=indexed_start,
            indexed_end=indexed_end,
            cluster_start=cluster_start,
            cluster_ids_cpu=cluster_ids_cpu.contiguous(),
            cluster_keys=summary.cluster_keys.contiguous(),
            cluster_values=summary.cluster_values.contiguous(),
            cluster_token_counts=resident_token_counts,
            cluster_page_ids_cpu=cluster_page_ids_cpu,
            cluster_page_token_counts_cpu=cluster_page_token_counts_cpu,
            cluster_page_counts_cpu=cluster_page_counts_cpu.contiguous(),
        )

    def _write_resident_segment(
        self,
        layer_state: _PackedLayerState,
        segment: RetroSpecResidentSegment,
        slot: int,
        previous_state: _ResidentRequestState | None,
    ) -> tuple[_ResidentRequestState, _ResidentSpanTransaction]:
        arena = layer_state.arena
        num_kv_heads, num_segment_clusters = segment.cluster_ids_cpu.shape
        cluster_start = segment.cluster_start
        cluster_end = cluster_start + num_segment_clusters

        if previous_state is None:
            if cluster_start != 0:
                raise RuntimeError(
                    "The first resident segment must start at cluster zero"
                )
            page_counts = (0,) * num_kv_heads
            indexed_start = segment.indexed_start
            previous_indexed_end = segment.indexed_start
            previous_num_clusters = 0
            max_pages_per_cluster = 0
            generation = self._slot_generations[slot]
        else:
            if len(previous_state.page_counts) != num_kv_heads:
                raise RuntimeError("Resident segment changed KV-head count")
            page_counts = previous_state.page_counts
            indexed_start = previous_state.indexed_start
            previous_indexed_end = previous_state.indexed_end
            previous_num_clusters = previous_state.num_clusters
            max_pages_per_cluster = previous_state.max_pages_per_cluster
            generation = previous_state.generation

        if previous_indexed_end != segment.indexed_start:
            raise RuntimeError(
                "Resident segment does not follow the indexed token prefix"
            )
        if previous_num_clusters != cluster_start:
            raise RuntimeError("Resident segment does not follow the cluster prefix")

        next_page_counts = tuple(
            page_counts[head_index]
            + int(segment.cluster_page_counts_cpu[head_index].sum().item())
            for head_index in range(num_kv_heads)
        )
        required_cluster_capacity = self._next_power_of_two(cluster_end)
        required_page_capacity = self._next_power_of_two(max(next_page_counts))

        new_cluster_span = None
        old_cluster_span = None
        new_page_span = None
        old_page_span = None

        if previous_state is None:
            cluster_offset = self._allocate_cluster_span(
                layer_state, required_cluster_capacity
            )
            new_cluster_span = (cluster_offset, required_cluster_capacity)
            try:
                page_offset = self._allocate_page_span(
                    layer_state, required_page_capacity
                )
            except BaseException:
                layer_state.cluster_allocator.release(*new_cluster_span)
                raise
            new_page_span = (page_offset, required_page_capacity)
            cluster_capacity = required_cluster_capacity
            page_capacity = required_page_capacity
        else:
            cluster_offset = previous_state.cluster_offset
            cluster_capacity = previous_state.cluster_capacity
            page_offset = previous_state.page_offset
            page_capacity = previous_state.page_capacity

            if required_cluster_capacity > cluster_capacity:
                new_offset = self._allocate_cluster_span(
                    layer_state, required_cluster_capacity
                )
                new_cluster_span = (new_offset, required_cluster_capacity)
                old_cluster_span = (cluster_offset, cluster_capacity)
                try:
                    source = slice(
                        cluster_offset, cluster_offset + previous_num_clusters
                    )
                    destination = slice(new_offset, new_offset + previous_num_clusters)
                    arena.cluster_ids[:, destination].copy_(
                        arena.cluster_ids[:, source]
                    )
                    arena.cluster_keys[:, destination].copy_(
                        arena.cluster_keys[:, source]
                    )
                    arena.cluster_values[:, destination].copy_(
                        arena.cluster_values[:, source]
                    )
                    arena.cluster_token_counts[:, destination].copy_(
                        arena.cluster_token_counts[:, source]
                    )
                    arena.cluster_page_starts[:, destination].copy_(
                        arena.cluster_page_starts[:, source]
                    )
                    arena.cluster_page_counts[:, destination].copy_(
                        arena.cluster_page_counts[:, source]
                    )
                    arena.resident_table_buckets[:, destination].copy_(
                        arena.resident_table_buckets[:, source]
                    )
                except BaseException:
                    layer_state.cluster_allocator.release(*new_cluster_span)
                    raise
                cluster_offset = new_offset
                cluster_capacity = required_cluster_capacity

            if required_page_capacity > page_capacity:
                try:
                    new_offset = self._allocate_page_span(
                        layer_state, required_page_capacity
                    )
                except BaseException:
                    if new_cluster_span is not None:
                        layer_state.cluster_allocator.release(*new_cluster_span)
                    raise
                new_page_span = (new_offset, required_page_capacity)
                old_page_span = (page_offset, page_capacity)
                try:
                    for head_index, head_page_count in enumerate(page_counts):
                        source = slice(page_offset, page_offset + head_page_count)
                        destination = slice(new_offset, new_offset + head_page_count)
                        arena.page_ids[head_index, destination].copy_(
                            arena.page_ids[head_index, source]
                        )
                        arena.page_token_counts[head_index, destination].copy_(
                            arena.page_token_counts[head_index, source]
                        )
                except BaseException:
                    layer_state.page_allocator.release(*new_page_span)
                    if new_cluster_span is not None:
                        layer_state.cluster_allocator.release(*new_cluster_span)
                    raise
                page_offset = new_offset
                page_capacity = required_page_capacity

        try:
            absolute_cluster_start = cluster_offset + cluster_start
            absolute_cluster_end = cluster_offset + cluster_end
            cluster_slice = slice(absolute_cluster_start, absolute_cluster_end)
            arena.resident_table_buckets[:, cluster_slice].fill_(-1)
            arena.cluster_ids[:, cluster_slice].copy_(segment.cluster_ids_cpu)
            arena.cluster_keys[:, cluster_slice].copy_(segment.cluster_keys)
            arena.cluster_values[:, cluster_slice].copy_(segment.cluster_values)
            arena.cluster_token_counts[:, cluster_slice].copy_(
                segment.cluster_token_counts
            )

            for head_index, page_start in enumerate(page_counts):
                segment_page_counts = segment.cluster_page_counts_cpu[head_index]
                page_starts_cpu = torch.empty(num_segment_clusters, dtype=torch.int64)
                if num_segment_clusters:
                    page_starts_cpu[0] = page_start
                if num_segment_clusters > 1:
                    torch.cumsum(
                        segment_page_counts[:-1].to(torch.int64),
                        dim=0,
                        out=page_starts_cpu[1:],
                    )
                    page_starts_cpu[1:].add_(page_start)

                arena.cluster_page_starts[head_index, cluster_slice].copy_(
                    page_starts_cpu, non_blocking=self.pin_memory
                )
                arena.cluster_page_counts[head_index, cluster_slice].copy_(
                    segment_page_counts, non_blocking=self.pin_memory
                )

                valid_pages = segment.cluster_page_ids_cpu[head_index] >= 0
                flat_page_ids = segment.cluster_page_ids_cpu[head_index].masked_select(
                    valid_pages
                )
                flat_page_token_counts = segment.cluster_page_token_counts_cpu[
                    head_index
                ].masked_select(valid_pages)

                absolute_page_start = page_offset + page_start
                absolute_page_end = absolute_page_start + flat_page_ids.numel()
                arena.page_ids[head_index, absolute_page_start:absolute_page_end].copy_(
                    flat_page_ids
                )
                arena.page_token_counts[
                    head_index, absolute_page_start:absolute_page_end
                ].copy_(flat_page_token_counts)
        except BaseException:
            if new_page_span is not None:
                layer_state.page_allocator.release(*new_page_span)
            if new_cluster_span is not None:
                layer_state.cluster_allocator.release(*new_cluster_span)
            raise

        segment_max_pages = int(segment.cluster_page_counts_cpu.max().item())
        state = _ResidentRequestState(
            slot=slot,
            generation=generation,
            indexed_start=indexed_start,
            indexed_end=segment.indexed_end,
            cluster_offset=cluster_offset,
            cluster_capacity=cluster_capacity,
            num_clusters=cluster_end,
            page_offset=page_offset,
            page_capacity=page_capacity,
            page_counts=next_page_counts,
            max_pages_per_cluster=max(max_pages_per_cluster, segment_max_pages),
        )
        transaction = _ResidentSpanTransaction(
            layer_state=layer_state,
            new_cluster_span=new_cluster_span,
            old_cluster_span=old_cluster_span,
            new_page_span=new_page_span,
            old_page_span=old_page_span,
        )
        return state, transaction

    @staticmethod
    def _rollback_span_transaction(transaction: _ResidentSpanTransaction) -> None:
        if transaction.new_page_span is not None:
            transaction.layer_state.page_allocator.release(*transaction.new_page_span)
        if transaction.new_cluster_span is not None:
            transaction.layer_state.cluster_allocator.release(
                *transaction.new_cluster_span
            )

    @staticmethod
    def _commit_span_transaction(transaction: _ResidentSpanTransaction) -> None:
        if transaction.old_page_span is not None:
            transaction.layer_state.page_allocator.release(*transaction.old_page_span)
        if transaction.old_cluster_span is not None:
            transaction.layer_state.cluster_allocator.release(
                *transaction.old_cluster_span
            )

    def _wait_for_binding_updates(
        self,
        layer_name: str,
        stream: torch.cuda.Stream,
    ) -> None:
        event = self._binding_update_events.get(layer_name)
        if event is not None:
            stream.wait_event(event)

    def _record_arena_ready(
        self,
        layer_name: str,
        stream: torch.cuda.Stream,
    ) -> None:
        event = torch.cuda.Event()
        event.record(stream)
        self._arena_ready_events[layer_name] = event

    def publish_resident_table_bindings(
        self,
        layer_name: str,
        bindings: Sequence[RetroSpecResidentTableBinding],
        stream: torch.cuda.Stream,
    ) -> None:
        bindings = tuple(bindings)
        if not bindings:
            return

        with self._arena_lock:
            layer_state = self._layer_arenas.get(layer_name)
            resident_states = self._resident_states.get(layer_name)
            if layer_state is None or resident_states is None:
                return

            arena = layer_state.arena
            binding_commands: list[tuple[int, int, int, int, int, int]] = []
            num_kv_heads = arena.cluster_ids.shape[0]

            for binding in bindings:
                state = resident_states.get(binding.request_id)
                if state is None:
                    continue
                if not 0 <= binding.kv_head_index < num_kv_heads:
                    continue
                if not 0 <= binding.local_cluster_index < state.num_clusters:
                    continue

                binding_commands.append(
                    (
                        state.slot,
                        state.generation,
                        binding.kv_head_index,
                        binding.local_cluster_index,
                        binding.cluster_handle,
                        binding.table_bucket,
                    )
                )

            if not binding_commands:
                return

            ready_event = self._arena_ready_events.get(layer_name)
            if ready_event is not None:
                stream.wait_event(ready_event)
            previous_update = self._binding_update_events.get(layer_name)
            if previous_update is not None:
                stream.wait_event(previous_update)

            device = arena.cluster_ids.device
            with torch.cuda.device(device), torch.cuda.stream(stream):
                publish_resident_table_bindings(
                    binding_commands=torch.tensor(
                        binding_commands, dtype=torch.int64, device=device
                    ),
                    arena_cluster_ids=arena.cluster_ids,
                    arena_resident_table_buckets=arena.resident_table_buckets,
                    arena_cluster_offsets=arena.cluster_offsets,
                    arena_num_clusters=arena.num_clusters,
                    arena_generations=arena.generations,
                )
                update_event = torch.cuda.Event()
                update_event.record(stream)
                self._binding_update_events[layer_name] = update_event

    def publish_resident_segments(
        self,
        segments: Sequence[RetroSpecResidentSegment],
    ) -> None:
        with self._arena_lock:
            self._publish_resident_segments_locked(segments)
