# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full KV handoff between the native paged cache and the CPU target."""

import torch

from .numa import RetroSpecHostPlacement


class RetroSpecCPUKVBridge:
    def __init__(self, proposer):
        self.proposer = proposer
        self.layers = proposer.cpu_attention_layers
        self.pending_uploads = []
        self.upload_stream = torch.cuda.Stream(device=proposer.device)
        self.export_placement = RetroSpecHostPlacement()
        self.upload_placement = RetroSpecHostPlacement()

    def _head_slice(self, layer, tensor):
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        heads = layer.impl.num_kv_heads
        total = tensor.shape[2]
        start = group.rank_in_group * heads if total >= group.world_size else 0
        return slice(start, start + heads)

    @staticmethod
    def _native_rows(cache, block_table, start: int, end: int):
        positions = torch.arange(start, end, device=cache.device)
        block_size = cache.shape[1]
        blocks = block_table[0].index_select(0, positions // block_size).long()
        if (blocks < 0).any():
            raise RuntimeError("CPU verifier requested retired native KV")
        return blocks, positions % block_size

    def export_gpu(
        self, storage, metadata, request_id: str, start: int, end: int, initial: bool
    ):
        with self.export_placement.bind():
            self._export_gpu(storage, metadata, request_id, start, end, initial)

    def _export_gpu(
        self, storage, metadata, request_id: str, start: int, end: int, initial: bool
    ):
        index = self.proposer.sparse_attention.index
        index.flush_staged_updates()
        destination = storage.kv if initial else storage.journal
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        if destination.shape[2] < group.world_size and group.rank_in_group != 0:
            # Replicated KV heads have one writer in the shared CPU mapping.
            return
        for layer_index, (name, layer) in enumerate(self.layers.items()):
            keys, values = layer.kv_cache[0].unbind(0)
            target = destination[:, layer_index, self._head_slice(layer, destination)]
            native_start = start
            if initial:
                record = index._indices.get(name, {}).get(request_id)
                indexed_end = self.proposer.block_size
                if record is not None and record.segments:
                    descriptor = record.full_verification_descriptor
                    indexed_end = record.indexed_end
                    indexed_tokens = indexed_end - self.proposer.block_size
                    heads, capacity, head_size = target.shape[1:]
                    if (
                        descriptor is None
                        or descriptor.num_kv_heads != heads
                        or indexed_tokens <= 0
                        or indexed_end > min(end, capacity)
                        or descriptor.head_token_counts != (indexed_tokens,) * heads
                    ):
                        raise RuntimeError("CPU verifier received an invalid KV prefix")
                    # Flatten complete head slots so the native gather can write
                    # straight into the shared mapping. Per-head offsets skip
                    # sink rows and leave native tails and future slots intact.
                    token_offsets = (
                        torch.arange(heads, dtype=torch.int64, device="cpu") * capacity
                        + self.proposer.block_size
                    ).unsqueeze(0)
                    index.cluster_store.gather_full_verification_into(
                        name,
                        (descriptor,),
                        token_offsets,
                        target[0].view(-1, head_size),
                        target[1].view(-1, head_size),
                    )
                # Historical post-RoPE KV can be permuted per head: every
                # subsequent CPU query attends to the complete past, unmasked.
                sink_end = min(end, self.proposer.block_size)
                if sink_end:
                    blocks, offsets = self._native_rows(
                        keys, metadata.block_table_tensor, 0, sink_end
                    )
                    target[0, :, :sink_end].copy_(
                        keys[blocks, offsets].transpose(0, 1).cpu()
                    )
                    target[1, :, :sink_end].copy_(
                        values[blocks, offsets].transpose(0, 1).cpu()
                    )
                native_start = min(max(indexed_end, sink_end), end)
            if native_start < end:
                blocks, offsets = self._native_rows(
                    keys, metadata.block_table_tensor, native_start, end
                )
                target[0, :, native_start:end].copy_(
                    keys[blocks, offsets].transpose(0, 1).cpu()
                )
                target[1, :, native_start:end].copy_(
                    values[blocks, offsets].transpose(0, 1).cpu()
                )

    def stage_cpu(self, storage, metadata, start: int, end: int, background: bool):
        with self.upload_placement.bind():
            self._stage_cpu(storage, metadata, start, end, background)

    def export_risk(self, storage, metadata, request_id: str, start: int, end: int):
        """Refresh a GPU-origin mirror, including deltas retired to CPU pages.

        Each indexed segment may permute its rows per head. Rebuild an entire
        intersecting segment, rather than splice packed rows into logical rows.
        All these rows precede the query, so the per-head order is immaterial.
        The caller fences CPU readers before invoking this method.
        """
        from vllm.distributed import get_tp_group

        with self.export_placement.bind():
            index = self.proposer.sparse_attention.index
            index.flush_staged_updates()
            group = get_tp_group()
            if storage.shape[2] < group.world_size and group.rank_in_group != 0:
                return
            for layer_index, (name, layer) in enumerate(self.layers.items()):
                keys, values = layer.kv_cache[0].unbind(0)
                target = storage.kv[:, layer_index, self._head_slice(layer, storage.kv)]
                record = index._indices.get(name, {}).get(request_id)
                segments = () if record is None else record.segments
                native_ranges = []
                cursor = start
                for segment in segments:
                    lo, hi = segment.indexed_start, segment.indexed_end
                    if hi <= start or lo >= end:
                        continue
                    if hi > end:
                        raise RuntimeError(
                            "CPU risk query intersects an indexed segment"
                        )
                    if cursor < lo:
                        native_ranges.append((cursor, lo))
                    descriptor = segment.cluster_blocks.full_verification_descriptor
                    heads, capacity, dimension = target.shape[1:]
                    if descriptor.head_token_counts != (hi - lo,) * heads:
                        raise RuntimeError("Invalid CPU risk segment coverage")
                    offsets = (
                        torch.arange(heads, dtype=torch.int64) * capacity + lo
                    ).unsqueeze(0)
                    index.cluster_store.gather_full_verification_into(
                        name,
                        (descriptor,),
                        offsets,
                        target[0].view(-1, dimension),
                        target[1].view(-1, dimension),
                    )
                    cursor = hi
                if cursor < end:
                    native_ranges.append((cursor, end))
                for lo, hi in native_ranges:
                    blocks, offsets = self._native_rows(
                        keys, metadata.block_table_tensor, lo, hi
                    )
                    target[0, :, lo:hi].copy_(
                        keys[blocks, offsets].transpose(0, 1).cpu()
                    )
                    target[1, :, lo:hi].copy_(
                        values[blocks, offsets].transpose(0, 1).cpu()
                    )

    def _stage_cpu(self, storage, metadata, start: int, end: int, background: bool):
        if end <= start:
            return
        # Staging copies into separate tensors may overlap draft. Publication
        # into native KV happens only on the next safe model boundary.
        current = torch.cuda.current_stream(self.proposer.device)
        self.upload_stream.wait_stream(current)
        for layer_index, layer in enumerate(self.layers.values()):
            target = layer.kv_cache[0]
            source = storage.kv[
                :, layer_index, self._head_slice(layer, storage.kv), start:end
            ]
            pinned = source.contiguous().pin_memory()
            with torch.cuda.stream(self.upload_stream):
                staged = pinned.to(self.proposer.device, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.upload_stream)
            self.pending_uploads.append(
                (target, metadata.block_table_tensor, start, end, pinned, staged, event)
            )
        if not background:
            self.publish(wait=True)

    def publish(self, wait: bool):
        remaining = []
        for target, table, start, end, pinned, staged, event in self.pending_uploads:
            if not wait and not event.query():
                remaining.append((target, table, start, end, pinned, staged, event))
                continue
            stream = torch.cuda.current_stream(self.proposer.device)
            stream.wait_event(event)
            keys, values = target.unbind(0)
            blocks, offsets = self._native_rows(keys, table, start, end)
            keys[blocks, offsets] = staged[0].transpose(0, 1)
            values[blocks, offsets] = staged[1].transpose(0, 1)
            staged.record_stream(stream)
        self.pending_uploads = remaining

    def close(self):
        self.publish(wait=True)
        torch.cuda.current_stream(self.proposer.device).synchronize()
