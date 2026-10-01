# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.v1.spec_decode.retrospec.legacy.residency.index.types import (
    RetroSpecResidentBatchView,
    RetroSpecResidentSegment,
    _ResidentRequestState,
    _ResidentSpanTransaction,
)


class _IndexResidencyViewsMixin:
    def _publish_resident_segments_locked(
        self,
        segments: Sequence[RetroSpecResidentSegment],
    ) -> None:
        if not segments:
            return

        incoming_request_ids = {segment.request_id for segment in segments}
        resident_request_ids = set(self.resident_request_ids)
        new_request_ids = resident_request_ids | incoming_request_ids
        if len(new_request_ids) > self.max_resident_requests:
            raise RuntimeError(
                "RetroSpec persistent GPU index residency exceeds max_num_seqs: "
                f"{len(new_request_ids)} > {self.max_resident_requests}"
            )

        projected_states = {
            layer_name: dict(layer_states)
            for layer_name, layer_states in self._resident_states.items()
        }
        changed_layers: set[str] = set()
        allocated_request_ids: list[str] = []
        transactions: list[_ResidentSpanTransaction] = []

        waited_layers: set[str] = set()
        try:
            for segment in self._coalesce_resident_segments(segments):
                if (
                    segment.cluster_keys.device.type == "cuda"
                    and segment.layer_name not in waited_layers
                ):
                    stream = torch.cuda.current_stream(segment.cluster_keys.device)
                    self._wait_for_binding_updates(segment.layer_name, stream)
                    waited_layers.add(segment.layer_name)
                if segment.request_id not in self._request_slots:
                    allocated_request_ids.append(segment.request_id)
                slot = self._get_or_allocate_request_slot(segment.request_id)
                layer_state = self._get_or_create_arena(segment.layer_name, segment)
                layer_states = projected_states.setdefault(segment.layer_name, {})
                previous_state = layer_states.get(segment.request_id)
                state, transaction = self._write_resident_segment(
                    layer_state=layer_state,
                    segment=segment,
                    slot=slot,
                    previous_state=previous_state,
                )
                layer_states[segment.request_id] = state
                transactions.append(transaction)
                changed_layers.add(segment.layer_name)
        except BaseException:
            for transaction in reversed(transactions):
                self._rollback_span_transaction(transaction)
            for request_id in reversed(allocated_request_ids):
                slot = self._request_slots.pop(request_id, None)
                if slot is not None:
                    self._free_request_slots.append(slot)
            raise

        for transaction in transactions:
            self._commit_span_transaction(transaction)
        self._resident_states = projected_states
        for layer_name in changed_layers:
            arena = self._layer_arenas[layer_name].arena
            for state in self._resident_states[layer_name].values():
                arena.cluster_offsets[state.slot] = state.cluster_offset
                arena.num_clusters[state.slot] = state.num_clusters
                arena.page_offsets[state.slot] = state.page_offset
                arena.num_pages[state.slot].copy_(
                    torch.tensor(
                        state.page_counts,
                        dtype=torch.int32,
                        device=arena.num_pages.device,
                    )
                )
                arena.generations[state.slot] = state.generation
                arena.indexed_starts[state.slot] = state.indexed_start
                arena.indexed_ends[state.slot] = state.indexed_end
            self._active_views.pop(layer_name, None)
            if arena.cluster_ids.device.type == "cuda":
                self._record_arena_ready(
                    layer_name,
                    torch.cuda.current_stream(arena.cluster_ids.device),
                )

    def get_active_view(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        device: torch.device,
    ) -> RetroSpecResidentBatchView:
        request_ids = tuple(request_ids)
        self._validate_active_requests(request_ids)

        cached = self._active_views.get(layer_name)
        if cached is not None and cached.request_slot_ids.device == device:
            return cached

        layer_states = self._resident_states.get(layer_name, {})
        request_slots = [
            -1 if (state := layer_states.get(request_id)) is None else state.slot
            for request_id in request_ids
        ]
        layer_state = self._layer_arenas.get(layer_name)
        arena = (
            None
            if layer_state is None or all(slot < 0 for slot in request_slots)
            else layer_state.arena
        )
        request_slot_ids = torch.tensor(request_slots, dtype=torch.int64, device=device)
        max_num_clusters = max(
            (
                state.num_clusters
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )
        max_pages_per_cluster = max(
            (
                state.max_pages_per_cluster
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )
        max_num_pages = max(
            (
                max(state.page_counts, default=0)
                for request_id in request_ids
                if (state := layer_states.get(request_id)) is not None
            ),
            default=0,
        )

        view = RetroSpecResidentBatchView(
            arena=arena,
            request_slot_ids=request_slot_ids,
            max_num_clusters=max(max_num_clusters, 1),
            max_pages_per_cluster=max_pages_per_cluster,
            max_num_pages=max_num_pages,
        )
        self._active_views[layer_name] = view
        return view

    def get_num_clusters(self, layer_name: str, request_id: str) -> int:
        state = self._resident_states.get(layer_name, {}).get(request_id)
        return 0 if state is None else state.num_clusters

    def get_indexed_end(self, layer_name: str, request_id: str) -> int | None:
        state = self._resident_states.get(layer_name, {}).get(request_id)
        return None if state is None else state.indexed_end

    def invalidate_active_view(self, layer_name: str) -> None:
        self._active_views.pop(layer_name, None)

    def _release_request_state(
        self,
        layer_name: str,
        state: _ResidentRequestState,
    ) -> None:
        layer_state = self._layer_arenas[layer_name]
        layer_state.cluster_allocator.release(
            state.cluster_offset, state.cluster_capacity
        )
        layer_state.page_allocator.release(state.page_offset, state.page_capacity)

        arena = layer_state.arena
        arena.cluster_offsets[state.slot] = 0
        arena.num_clusters[state.slot] = 0
        arena.page_offsets[state.slot] = 0
        arena.num_pages[state.slot].zero_()
        arena.generations[state.slot] = 0
        arena.indexed_starts[state.slot] = 0
        arena.indexed_ends[state.slot] = 0

    def discard_request_layer(self, layer_name: str, request_id: str) -> None:
        with self._arena_lock:
            layer_states = self._resident_states.get(layer_name)
            layer_state = self._layer_arenas.get(layer_name)
            if layer_states is None or layer_state is None:
                return

            state = layer_states.pop(request_id, None)
            if state is None:
                return

            arena = layer_state.arena
            stream = (
                torch.cuda.current_stream(arena.cluster_ids.device)
                if arena.cluster_ids.device.type == "cuda"
                else None
            )
            if stream is not None:
                self._wait_for_binding_updates(layer_name, stream)
            self._release_request_state(layer_name, state)
            self._active_views.pop(layer_name, None)
            if stream is not None:
                self._record_arena_ready(layer_name, stream)

    def invalidate_requests(self, request_ids: Sequence[str]) -> None:
        removed = set(request_ids)
        if not removed:
            return

        with self._arena_lock:
            for layer_name, layer_states in self._resident_states.items():
                layer_state = self._layer_arenas.get(layer_name)
                if layer_state is None:
                    continue
                arena = layer_state.arena
                stream = (
                    torch.cuda.current_stream(arena.cluster_ids.device)
                    if arena.cluster_ids.device.type == "cuda"
                    else None
                )
                waited = False
                changed = False
                for request_id in removed:
                    state = layer_states.pop(request_id, None)
                    if state is None:
                        continue
                    if not waited and stream is not None:
                        self._wait_for_binding_updates(layer_name, stream)
                        waited = True
                    self._release_request_state(layer_name, state)
                    changed = True
                if changed and stream is not None:
                    self._record_arena_ready(layer_name, stream)

            self._active_views.clear()

            for request_id in removed:
                slot = self._request_slots.pop(request_id, None)
                if slot is not None:
                    self._free_request_slots.append(slot)
