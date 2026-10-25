# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Background CPU KV preparation from leased pages and private GPU tails."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event
from time import perf_counter

import torch

from vllm import _custom_ops as ops
from vllm.distributed import get_tp_group


@dataclass
class _LayerSnapshot:
    target: torch.Tensor
    source: object = None
    gathers: list = field(default_factory=list)
    native: list = field(default_factory=list)


class RetroSpecCPUKVPreparation:
    def __init__(self, index, request_id, captured):
        self.index = index
        self.request_id = request_id
        self.captured = captured
        self.revisions = []
        self.layers = []
        self.leases = []
        self.cancelled = Event()
        self.future = None

    def valid(self):
        for name, record, revision in self.revisions:
            current = self.index._indices.get(name, {}).get(self.request_id)
            if current is not record or (
                current is not None and current.revision != revision
            ):
                return False
        return True

    def fence_sources(self):
        # This event was recorded before returning to the proposer. A sparse,
        # full, or request transition can now overwrite/retire native sources
        # without waiting for the background CPU gather or D2H publication.
        torch.cuda.current_stream(self.captured.device).wait_event(self.captured)

    def cancel(self):
        self.cancelled.set()
        self.fence_sources()

    def release_pages(self):
        for lease in self.leases:
            lease.release()
        self.leases.clear()


class RetroSpecCPUKVPreparer:
    # One reusable pinned window, independent of context length and layer count.
    _NATIVE_CHUNK_TOKENS = 1024

    def __init__(self, bridge):
        self.bridge = bridge
        self.device = bridge.proposer.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="retrospec-cpu-kv",
            initializer=bridge.export_placement.initialize_worker,
        )
        store = bridge.proposer.sparse_attention.index.cluster_store
        self.gather_workers = store.full_verify_gather_workers
        self.pinned = None

    def submit(self, storage, metadata, request_id, start, end):
        index = self.bridge.proposer.sparse_attention.index
        if index.has_staged_updates:
            raise RuntimeError("CPU risk snapshot requires a published KV index")
        group = get_tp_group()
        captured = torch.cuda.Event()
        preparation = RetroSpecCPUKVPreparation(index, request_id, captured)
        source_ready = torch.cuda.Event()
        source_ready.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(source_ready)
        try:
            with torch.cuda.stream(self.stream):
                if storage.shape[2] >= group.world_size or group.rank_in_group == 0:
                    for layer_index, (name, layer) in enumerate(
                        self.bridge.layers.items()
                    ):
                        target = storage.kv[
                            :, layer_index, self.bridge._head_slice(layer, storage.kv)
                        ]
                        snapshot = _LayerSnapshot(target)
                        preparation.layers.append(snapshot)
                        record = index._indices.get(name, {}).get(request_id)
                        preparation.revisions.append(
                            (name, record, None if record is None else record.revision)
                        )
                        segments = () if record is None else record.segments
                        cursor = start
                        native_ranges = []
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
                            descriptor = (
                                segment.cluster_blocks.full_verification_descriptor
                            )
                            if (
                                descriptor.head_token_counts
                                != (hi - lo,) * target.shape[1]
                            ):
                                raise RuntimeError("Invalid CPU risk segment coverage")
                            snapshot.gathers.append((descriptor, lo))
                            cursor = hi
                        if snapshot.gathers:
                            lease = index.cluster_store.lease_full_verification_sources(
                                name
                            )
                            preparation.leases.append(lease)
                            snapshot.source = lease.source
                        if cursor < end:
                            native_ranges.append((cursor, end))
                        keys, values = layer.kv_cache[0].unbind(0)
                        for lo, hi in native_ranges:
                            positions = torch.arange(lo, hi, device=self.device)
                            blocks = (
                                metadata.block_table_tensor[0]
                                .index_select(0, positions // keys.shape[1])
                                .long()
                            )
                            valid = ((blocks >= 0) & (blocks < keys.shape[0])).all()
                            safe_blocks = blocks.clamp(0, keys.shape[0] - 1)
                            offsets = positions % keys.shape[1]
                            # Only the unindexed tail/sink is copied on device.
                            # Private tensors let sparse overwrite native rows
                            # while D2H and the CPU page gather are still running.
                            data = torch.stack(
                                (
                                    keys[safe_blocks, offsets].transpose(0, 1),
                                    values[safe_blocks, offsets].transpose(0, 1),
                                )
                            )
                            snapshot.native.append((lo, hi, data, valid))
                        keys.record_stream(self.stream)
                        values.record_stream(self.stream)
                    metadata.block_table_tensor.record_stream(self.stream)
                captured.record(self.stream)
            preparation.future = self.executor.submit(self._run, preparation)
            return preparation
        except BaseException:
            captured.record(self.stream)
            preparation.cancel()
            preparation.release_pages()
            raise

    def _run(self, preparation):
        stats = self.bridge.proposer.performance_stats
        started = perf_counter()
        try:
            with torch.inference_mode(), torch.cuda.device(self.device):
                for layer in preparation.layers:
                    if preparation.cancelled.is_set():
                        return
                    with stats.cpu_timer("cpu_risk_compact_gather_wall"):
                        for descriptor, lo in layer.gathers:
                            heads, capacity, dimension = layer.target.shape[1:]
                            offsets = (
                                torch.arange(heads, dtype=torch.int64) * capacity + lo
                            ).unsqueeze(0)
                            ops.retrospec_gather_compact_kv(
                                layer.source.key_slabs,
                                layer.source.value_slabs,
                                (descriptor.range_table,),
                                offsets,
                                0,
                                layer.target[0].view(-1, dimension),
                                layer.target[1].view(-1, dimension),
                                self.gather_workers,
                            )
                    for lo, hi, data, valid in layer.native:
                        if preparation.cancelled.is_set():
                            return
                        with torch.cuda.stream(self.stream):
                            if not valid.item():
                                raise RuntimeError(
                                    "CPU verifier requested retired native KV"
                                )
                        for begin in range(0, hi - lo, self._NATIVE_CHUNK_TOKENS):
                            if preparation.cancelled.is_set():
                                return
                            end = min(begin + self._NATIVE_CHUNK_TOKENS, hi - lo)
                            with stats.cpu_timer("cpu_risk_native_d2h_wall"):
                                with torch.cuda.stream(self.stream):
                                    part = data[:, :, begin:end].contiguous()
                                    if (
                                        self.pinned is None
                                        or self.pinned.numel() < part.numel()
                                    ):
                                        with self.bridge.upload_placement.bind():
                                            self.pinned = torch.empty(
                                                part.numel(),
                                                dtype=part.dtype,
                                                device="cpu",
                                                pin_memory=True,
                                            )
                                    pinned = self.pinned[: part.numel()].view(
                                        part.shape
                                    )
                                    pinned.copy_(part, non_blocking=True)
                                    ready = torch.cuda.Event()
                                    ready.record(self.stream)
                                ready.synchronize()
                                layer.target[:, :, lo + begin : lo + end].copy_(pinned)
        finally:
            preparation.release_pages()
            elapsed = perf_counter() - started
            stats.record_cpu_time("cpu_risk_prepare_background_wall", elapsed)
            stats.record_cpu_time("cpu_risk_kv_export", elapsed)

    def close(self):
        self.executor.shutdown(wait=True)
        self.stream.synchronize()
        self.pinned = None
