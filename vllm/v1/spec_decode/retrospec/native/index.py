# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU-only clustered index and sparse attention for RetroSpec.

The native vLLM KV blocks remain the sole exact-token storage. The reverse
cluster layout contains logical token indices, not copies of K/V vectors.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from vllm.triton_utils import triton
from vllm.v1.spec_decode.retrospec.cluster import segmented_kmeans
from vllm.v1.spec_decode.retrospec.native.kernels import (
    _NativeBatchLayer,
    _NativeLayerRecord,
    _scatter_cluster_tokens_kernel,
)


class RetroSpecNativeIndexMixin:
    @staticmethod
    def _append_record(
        record: _NativeLayerRecord, part: _NativeLayerRecord
    ) -> _NativeLayerRecord:
        old_clusters = record.counts.shape[1]
        old_tokens = record.token_indices.shape[1]
        total_clusters = old_clusters + part.counts.shape[1]
        total_tokens = old_tokens + part.token_indices.shape[1]
        old_key_storage = (
            record.key_storage if record.key_storage is not None else record.keys
        )
        old_value_storage = (
            record.value_storage if record.value_storage is not None else record.values
        )
        old_count_storage = (
            record.count_storage if record.count_storage is not None else record.counts
        )
        old_token_storage = (
            record.token_storage
            if record.token_storage is not None
            else record.token_indices
        )
        old_offset_storage = (
            record.offset_storage
            if record.offset_storage is not None
            else record.cluster_offsets
        )
        cluster_capacity = triton.next_power_of_2(
            max(total_clusters, old_key_storage.shape[1])
        )
        token_capacity = triton.next_power_of_2(
            max(total_tokens, old_token_storage.shape[1])
        )
        if old_key_storage.shape[1] < total_clusters:
            key_storage = record.keys.new_empty(
                record.keys.shape[0], cluster_capacity, record.keys.shape[2]
            )
            value_storage = record.values.new_empty(
                record.values.shape[0], cluster_capacity, record.values.shape[2]
            )
            count_storage = record.counts.new_empty(
                record.counts.shape[0], cluster_capacity
            )
            offset_storage = record.cluster_offsets.new_empty(
                record.cluster_offsets.shape[0], cluster_capacity + 1
            )
            key_storage[:, :old_clusters].copy_(record.keys)
            value_storage[:, :old_clusters].copy_(record.values)
            count_storage[:, :old_clusters].copy_(record.counts)
            offset_storage[:, : old_clusters + 1].copy_(record.cluster_offsets)
        else:
            key_storage = old_key_storage
            value_storage = old_value_storage
            count_storage = old_count_storage
            offset_storage = old_offset_storage
        if old_token_storage.shape[1] < total_tokens:
            token_storage = record.token_indices.new_empty(
                record.token_indices.shape[0], token_capacity
            )
            token_storage[:, :old_tokens].copy_(record.token_indices)
        else:
            token_storage = old_token_storage
        key_storage[:, old_clusters:total_clusters].copy_(part.keys)
        value_storage[:, old_clusters:total_clusters].copy_(part.values)
        count_storage[:, old_clusters:total_clusters].copy_(part.counts)
        token_storage[:, old_tokens:total_tokens].copy_(part.token_indices)
        offset_storage[:, old_clusters + 1 : total_clusters + 1].copy_(
            part.cluster_offsets[:, 1:] + old_tokens
        )
        return _NativeLayerRecord(
            part.indexed_end,
            key_storage[:, :total_clusters],
            value_storage[:, :total_clusters],
            count_storage[:, :total_clusters],
            token_storage[:, :total_tokens],
            offset_storage[:, : total_clusters + 1],
            key_storage,
            value_storage,
            count_storage,
            token_storage,
            offset_storage,
        )

    def has_staged_request_layer(self, layer_name: str, request_id: str) -> bool:
        return (layer_name, request_id) in self._staged

    def _desired_end(
        self,
        seq_len: int,
        indexed_end: int,
        is_prefill: bool,
        prefill_complete: bool,
    ) -> int:
        stable_end = (
            max(seq_len // self.block_size - self.num_recent_blocks, 1)
            * self.block_size
        )
        start = self.block_size if stable_end < indexed_end else indexed_end
        quantum = (
            self.tokens_per_cluster
            if is_prefill and prefill_complete
            else self.prefill_segment_size_tokens
            if is_prefill
            else self.generation_update_interval
        )
        return start + max(stable_end - start, 0) // quantum * quantum

    def needs_update(
        self,
        request_id: str,
        seq_len: int,
        layer_names: Sequence[str],
        is_prefill: bool,
        prefill_complete: bool = False,
    ) -> bool:
        for layer_name in layer_names:
            record = self._records.get(layer_name, {}).get(request_id)
            current_end = self.block_size if record is None else record.indexed_end
            if (
                self._desired_end(seq_len, current_end, is_prefill, prefill_complete)
                != current_end
            ):
                return True
        return False

    def build_or_update(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        seq_lens: Sequence[int],
        is_prefill: Sequence[bool],
        rows: Sequence[int],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        prefill_complete: Sequence[bool] | None = None,
    ) -> None:
        if prefill_complete is None:
            prefill_complete = (False,) * len(request_ids)
        if key_cache.device.type != "cuda":
            raise ValueError("GPU-native RetroSpec requires CUDA KV cache")
        for row in rows:
            request_id = request_ids[row]
            key = (layer_name, request_id)
            if key in self._staged:
                raise RuntimeError(
                    "A GPU-native request/layer update is already staged"
                )
            record = self._records.get(layer_name, {}).get(request_id)
            current_end = self.block_size if record is None else record.indexed_end
            desired_end = self._desired_end(
                int(seq_lens[row]),
                current_end,
                bool(is_prefill[row]),
                bool(prefill_complete[row]),
            )
            if desired_end < current_end:
                record = None
                current_end = self.block_size
            if desired_end <= current_end:
                if request_id in self._records.get(layer_name, {}):
                    self._staged[key] = None
                continue

            phase_start = current_end
            parts: list[_NativeLayerRecord] = []
            regular = (
                self.prefill_segment_size_tokens
                if is_prefill[row]
                else self.generation_update_interval
            )
            while phase_start < desired_end:
                phase_size = min(regular, desired_end - phase_start)
                if phase_size % self.tokens_per_cluster:
                    raise RuntimeError(
                        "GPU-native clustering phase is not cluster-aligned"
                    )
                with self._cuda_timer("gpu_native_index_gather"):
                    logical_blocks = torch.arange(
                        phase_start // self.block_size,
                        (phase_start + phase_size) // self.block_size,
                        dtype=torch.int64,
                        device=block_table.device,
                    )
                    physical_blocks = (
                        block_table[row].index_select(0, logical_blocks).long()
                    )
                    num_heads, head_size = key_cache.shape[2:]
                    token_keys = (
                        key_cache.index_select(0, physical_blocks)
                        .reshape(phase_size, num_heads, head_size)
                        .transpose(0, 1)
                        .contiguous()
                    )
                    token_values = (
                        value_cache.index_select(0, physical_blocks)
                        .reshape(phase_size, num_heads, head_size)
                        .transpose(0, 1)
                        .contiguous()
                    )
                with self._cuda_timer("gpu_native_index_kmeans"):
                    clustered = segmented_kmeans(
                        token_keys,
                        token_values,
                        phase_size,
                        self.tokens_per_cluster,
                        self.num_kmeans_iterations,
                    )
                with self._cuda_timer("gpu_native_index_reverse"):
                    counts = clustered.cluster_sizes.to(torch.int32)
                    offsets = F.pad(counts.cumsum(1, dtype=torch.int32), (1, 0))
                    token_indices = torch.empty_like(clustered.assignments)
                    _scatter_cluster_tokens_kernel[
                        (num_heads, triton.cdiv(phase_size, 256))
                    ](
                        clustered.assignments,
                        clustered.token_offsets_in_cluster,
                        offsets,
                        token_indices,
                        clustered.assignments.stride(0),
                        clustered.token_offsets_in_cluster.stride(0),
                        offsets.stride(0),
                        token_indices.stride(0),
                        phase_size,
                        phase_start,
                        256,
                    )
                parts.append(
                    _NativeLayerRecord(
                        phase_start + phase_size,
                        clustered.cluster_keys,
                        clustered.cluster_values,
                        counts,
                        token_indices,
                        offsets,
                    )
                )
                phase_start += phase_size

            with self._cuda_timer("gpu_native_index_append"):
                for part in parts:
                    if record is None:
                        record = part
                    else:
                        record = self._append_record(record, part)
            assert record is not None
            self._staged[key] = record
        return None

    def flush_staged_updates(self) -> None:
        for (layer_name, request_id), record in self._staged.items():
            if record is None:
                self._records.get(layer_name, {}).pop(request_id, None)
            else:
                self._records.setdefault(layer_name, {})[request_id] = record
        self._staged.clear()

    def discard_staged_updates(self) -> None:
        self._staged.clear()

    def get_fully_stored_indexed_end(
        self, request_id: str, layer_names: Sequence[str]
    ) -> int:
        return (
            min(
                (
                    self._records.get(name, {}).get(request_id).indexed_end
                    if request_id in self._records.get(name, {})
                    else self.block_size
                )
                for name in layer_names
            )
            if layer_names
            else self.block_size
        )

    def remove_requests(self, request_ids: Sequence[str]) -> None:
        removed = set(request_ids)
        for layer_name, records in self._records.items():
            for request_id in removed:
                records.pop(request_id, None)
                slot = self._arena_slots.get(layer_name, {}).pop(request_id, None)
                if slot is not None:
                    self._arena_free_slots.setdefault(layer_name, []).append(slot)
                    self._arena_versions.get(layer_name, {}).pop(request_id, None)
                    workspace = self._workspaces.get(layer_name)
                    if workspace is not None:
                        workspace.counts[slot].zero_()
                        workspace.indexed_ends[slot] = self.block_size
        for key in tuple(self._staged):
            if key[1] in removed:
                del self._staged[key]

    def has_cluster_pages(self, layer_name: str, request_ids: Sequence[str]) -> bool:
        return any(
            (record := self._records.get(layer_name, {}).get(request_id)) is not None
            and record.counts.shape[1] > 0
            for request_id in request_ids
        )

    def begin_proposal(self, request_ids: Sequence[str]) -> None:
        if self._request_ids or self._staged:
            raise RuntimeError("GPU-native proposal cannot overlap another transaction")
        self._request_ids = tuple(request_ids)
        self._batch_layers.clear()
        self._active_cluster_counts.clear()
        self._plans.clear()
        self._active_plan_workspaces.clear()

    def end_proposal(self) -> None:
        self._request_ids = ()
        self._batch_layers.clear()
        self._active_cluster_counts.clear()
        self._plans.clear()
        self._active_plan_workspaces.clear()

    def _batch_layer(
        self, layer_name: str, device: torch.device, dtype: torch.dtype
    ) -> _NativeBatchLayer:
        cached = self._batch_layers.get(layer_name)
        if cached is not None:
            return cached
        layer_records = self._records.get(layer_name, {})
        records = [layer_records.get(request_id) for request_id in self._request_ids]
        present = next((record for record in records if record is not None), None)
        if present is None:
            raise RuntimeError("GPU-native layer has no cluster index")
        heads, _, head_size = present.keys.shape
        max_clusters = max(record.counts.shape[1] for record in layer_records.values())
        max_tokens = max(
            record.token_indices.shape[1] for record in layer_records.values()
        )
        active_clusters = max(
            record.counts.shape[1] if record is not None else 0 for record in records
        )
        active_tokens = max(
            record.token_indices.shape[1] if record is not None else 0
            for record in records
        )
        batch = len(records)
        slots = self._arena_slots.setdefault(layer_name, {})
        free_slots = self._arena_free_slots.setdefault(layer_name, [])
        for request_id in layer_records:
            if request_id not in slots:
                slots[request_id] = (
                    free_slots.pop()
                    if free_slots
                    else max(slots.values(), default=0) + 1
                )
        required_slots = max(slots.values(), default=0) + 1
        workspace = self._workspaces.get(layer_name)
        if (
            workspace is None
            or workspace.keys.device.type != device.type
            or (
                device.index is not None and workspace.keys.device.index != device.index
            )
            or workspace.keys.dtype != dtype
            or workspace.keys.shape[0] < required_slots
            or workspace.keys.shape[1] != heads
            or workspace.keys.shape[2] < max_clusters
            or workspace.keys.shape[3] != head_size
            or workspace.token_indices.shape[2] < max_tokens
            or (
                self.draft_rank_dtype == "int8"
                and dtype in (torch.float16, torch.bfloat16)
                and workspace.quantized_keys is None
            )
        ):
            batch_capacity = triton.next_power_of_2(
                max(
                    required_slots,
                    workspace.keys.shape[0] if workspace is not None else 2,
                )
            )
            cluster_capacity = triton.next_power_of_2(
                max(
                    max_clusters,
                    workspace.keys.shape[2] if workspace is not None else 1,
                )
            )
            token_capacity = triton.next_power_of_2(
                max(
                    max_tokens,
                    workspace.token_indices.shape[2] if workspace is not None else 1,
                )
            )
            keys = torch.empty(
                batch_capacity,
                heads,
                cluster_capacity,
                head_size,
                dtype=dtype,
                device=device,
            )
            workspace = _NativeBatchLayer(
                keys=keys,
                values=torch.empty_like(keys),
                counts=torch.empty(
                    batch_capacity,
                    heads,
                    cluster_capacity,
                    dtype=torch.int32,
                    device=device,
                ),
                token_indices=torch.empty(
                    batch_capacity,
                    heads,
                    token_capacity,
                    dtype=torch.int32,
                    device=device,
                ),
                cluster_offsets=torch.empty(
                    batch_capacity,
                    heads,
                    cluster_capacity + 1,
                    dtype=torch.int32,
                    device=device,
                ),
                indexed_ends=torch.empty(
                    batch_capacity,
                    dtype=torch.int32,
                    device=device,
                ),
                quantized_keys=(
                    torch.empty(
                        batch_capacity,
                        heads,
                        cluster_capacity,
                        head_size,
                        dtype=torch.int8,
                        device=device,
                    )
                    if self.draft_rank_dtype == "int8"
                    and dtype in (torch.float16, torch.bfloat16)
                    else None
                ),
                key_scales=(
                    torch.empty(
                        batch_capacity,
                        heads,
                        cluster_capacity,
                        dtype=torch.float32,
                        device=device,
                    )
                    if self.draft_rank_dtype == "int8"
                    and dtype in (torch.float16, torch.bfloat16)
                    else None
                ),
            )
            self._workspaces[layer_name] = workspace
            self._workspace_generations[layer_name] = (
                self._workspace_generations.get(layer_name, 0) + 1
            )
            self._arena_versions[layer_name] = {}
            workspace.counts[0].zero_()
            workspace.indexed_ends[0] = self.block_size
        versions = self._arena_versions.setdefault(layer_name, {})
        for request_id, record in layer_records.items():
            if versions.get(request_id) is record:
                continue
            slot = slots[request_id]
            clusters = record.counts.shape[1]
            num_tokens = record.token_indices.shape[1]
            workspace.counts[slot].zero_()
            workspace.keys[slot, :, :clusters].copy_(record.keys)
            workspace.values[slot, :, :clusters].copy_(record.values)
            if workspace.quantized_keys is not None:
                assert workspace.key_scales is not None
                with self._cuda_timer("gpu_native_index_quantize"):
                    centered = record.keys.float()
                    scales = (centered.abs().amax(dim=-1) / 127.0).clamp_min(1.0e-8)
                    workspace.quantized_keys[slot, :, :clusters].copy_(
                        (centered / scales.unsqueeze(-1))
                        .round()
                        .clamp(-127, 127)
                        .to(torch.int8)
                    )
                    workspace.key_scales[slot, :, :clusters].copy_(scales)
            workspace.counts[slot, :, :clusters].copy_(record.counts)
            workspace.token_indices[slot, :, :num_tokens].copy_(record.token_indices)
            workspace.cluster_offsets[slot, :, : clusters + 1].copy_(
                record.cluster_offsets
            )
            workspace.indexed_ends[slot] = record.indexed_end
            resident_record = _NativeLayerRecord(
                record.indexed_end,
                workspace.keys[slot, :, :clusters],
                workspace.values[slot, :, :clusters],
                workspace.counts[slot, :, :clusters],
                workspace.token_indices[slot, :, :num_tokens],
                workspace.cluster_offsets[slot, :, : clusters + 1],
            )
            layer_records[request_id] = resident_record
            versions[request_id] = resident_record
        request_slots = torch.tensor(
            [slots.get(request_id, 0) for request_id in self._request_ids],
            dtype=torch.int32,
            device=device,
        )
        layout = _NativeBatchLayer(
            workspace.keys[:, :, :active_clusters],
            workspace.values[:, :, :active_clusters],
            workspace.counts[:, :, :active_clusters],
            workspace.token_indices[:, :, :active_tokens],
            workspace.cluster_offsets[:, :, : active_clusters + 1],
            workspace.indexed_ends,
            request_slots,
            workspace.quantized_keys[:, :, :active_clusters]
            if workspace.quantized_keys is not None
            else None,
            workspace.key_scales[:, :, :active_clusters]
            if workspace.key_scales is not None
            else None,
        )
        self._batch_layers[layer_name] = layout
        self._active_cluster_counts[layer_name] = active_clusters
        self._prepare_plan_workspace(layer_name, layout, batch, device)
        return layout
