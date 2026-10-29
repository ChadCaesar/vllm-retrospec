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
    origin: int = 0
    source: object = None
    gathers: list = field(default_factory=list)
    native: list = field(default_factory=list)


@dataclass
class _NativeBatch:
    data: torch.Tensor
    valid: torch.Tensor
    ready: torch.cuda.Event


class RetroSpecCPUKVPreparation:
    def __init__(self, index, request_id, captured):
        self.index = index
        self.request_id = request_id
        self.captured = captured
        self.revisions = []
        self.layers = []
        self.leases = []
        self.native_batch = None
        self.cancelled = Event()
        self.future = None

    def valid(self):
        # Descriptors/pages and native payloads belong to this capture, not to
        # the latest mutable index revision. Leases fence the old sources until
        # the background writer ends; afterwards only the private CPU copy is read.
        return not self.cancelled.is_set()

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
    _NATIVE_BATCH_BYTES = 8 * 1024 * 1024

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
        self.native_pinned = None
        self.native_valid = None

    def submit(self, storage, metadata, request_id, start, end, *, native_only=False):
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
                native_sources = []
                if storage.shape[2] >= group.world_size or group.rank_in_group == 0:
                    for layer_index, (name, layer) in enumerate(
                        self.bridge.layers.items()
                    ):
                        target = storage.kv[
                            :, layer_index, self.bridge._head_slice(layer, storage.kv)
                        ]
                        origin = getattr(storage, "offset", 0)
                        if not origin <= start <= end <= origin + target.shape[-2]:
                            raise RuntimeError("CPU risk tail capacity exceeded")
                        snapshot = _LayerSnapshot(target, origin)
                        preparation.layers.append(snapshot)
                        record = index._indices.get(name, {}).get(request_id)
                        preparation.revisions.append(
                            (name, record, None if record is None else record.revision)
                        )
                        segments = (
                            () if native_only or record is None else record.segments
                        )
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
                            snapshot.gathers.append((descriptor, lo - origin))
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
                            native_sources.append((snapshot, lo, hi, keys, values))
                        keys.record_stream(self.stream)
                        values.record_stream(self.stream)
                    metadata.block_table_tensor.record_stream(self.stream)
                packed = self._capture_native(native_sources, metadata)
                captured.record(self.stream)
                preparation.native_batch = self._batch_native(preparation, packed)
            preparation.future = self.executor.submit(self._run, preparation)
            return preparation
        except BaseException:
            captured.record(self.stream)
            preparation.cancel()
            preparation.release_pages()
            raise

    def _capture_native(self, sources, metadata):
        elements = 0
        fast = bool(sources)
        for _, lo, hi, keys, _ in sources:
            elements += 2 * keys.shape[2] * (hi - lo) * keys.shape[3]
            fast &= (
                0 < hi - lo <= self._NATIVE_CHUNK_TOKENS
                and keys.dtype == sources[0][3].dtype
                and elements * keys.element_size() <= self._NATIVE_BATCH_BYTES
            )
        if fast:
            from vllm.triton_utils import triton

            from .cpu_kv_kernels import _capture_native_kv_kernel

            # One capture launch writes all layers directly into the D2H
            # payload. No per-layer arange/index/reduction/stack launches, and
            # no second GPU packing pass. Pointer/stride metadata is private
            # to this snapshot; source lifetimes retain their original fence.
            data = torch.empty(elements, dtype=sources[0][3].dtype, device=self.device)
            valid = torch.empty(len(sources), dtype=torch.bool, device=self.device)
            rows = []
            offset = largest = 0
            for part, (layer, lo, hi, keys, values) in enumerate(sources):
                shape = (2, keys.shape[2], hi - lo, keys.shape[3])
                count = shape[1] * shape[2] * shape[3]
                rows.append(
                    (
                        keys.data_ptr(),
                        values.data_ptr(),
                        *keys.stride(),
                        *values.stride(),
                        lo,
                        hi - lo,
                        keys.shape[2],
                        keys.shape[3],
                        offset,
                        keys.shape[0],
                        keys.shape[1],
                    )
                )
                layer.native.append(
                    (
                        lo - layer.origin,
                        hi - layer.origin,
                        data[offset : offset + 2 * count].view(shape),
                        valid[part],
                    )
                )
                largest = max(largest, count)
                offset += 2 * count
            with self.bridge.upload_placement.bind():
                plan = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
            plan = plan.to(self.device, non_blocking=True)
            table = metadata.block_table_tensor[0]
            _capture_native_kv_kernel[(len(rows), triton.cdiv(largest, 1024))](
                plan,
                table,
                data,
                valid,
                table.numel(),
                table.stride(0),
                BLOCK=1024,
                MAX_TOKENS=self._NATIVE_CHUNK_TOKENS,
            )
            return data, valid
        for layer, lo, hi, keys, values in sources:
            positions = torch.arange(lo, hi, device=self.device)
            blocks = (
                metadata.block_table_tensor[0]
                .index_select(0, positions // keys.shape[1])
                .long()
            )
            valid = ((blocks >= 0) & (blocks < keys.shape[0])).all()
            safe_blocks = blocks.clamp(0, keys.shape[0] - 1)
            offsets = positions % keys.shape[1]
            data = torch.stack(
                (
                    keys[safe_blocks, offsets].transpose(0, 1),
                    values[safe_blocks, offsets].transpose(0, 1),
                )
            )
            layer.native.append((lo - layer.origin, hi - layer.origin, data, valid))
        return None

    def _batch_native(self, preparation, packed=None):
        parts = []
        elements = 0
        for layer in preparation.layers:
            for lo, hi, data, valid in layer.native:
                elements += data.numel()
                if (
                    hi - lo > self._NATIVE_CHUNK_TOKENS
                    or elements * data.element_size() > self._NATIVE_BATCH_BYTES
                ):
                    return None
                parts.append((layer, lo, hi, data, valid))
        if not parts:
            return None
        # The private GPU snapshots have already fenced their native sources.
        # Queue the small tails together on the independent stream, so the
        # background gather need not enter CUDA or synchronize once per layer
        # while the launch thread is submitting draft kernels.
        with self.bridge.upload_placement.bind():
            if self.native_pinned is None or self.native_pinned.numel() < elements:
                self.native_pinned = torch.empty(
                    elements, dtype=parts[0][3].dtype, device="cpu", pin_memory=True
                )
            if self.native_valid is None or self.native_valid.numel() < len(parts):
                self.native_valid = torch.empty(
                    len(parts), dtype=torch.bool, device="cpu", pin_memory=True
                )
        data = self.native_pinned[:elements]
        valid = self.native_valid[: len(parts)]
        if packed is None:
            packed = (
                torch.cat([part[3].view(-1) for part in parts]),
                torch.stack([part[4] for part in parts]),
            )
        data.copy_(packed[0], non_blocking=True)
        valid.copy_(packed[1], non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(self.stream)
        stats = self.bridge.proposer.performance_stats
        stats.add_counter("cpu_risk_native_batches")
        stats.add_counter("cpu_risk_native_batch_parts", len(parts))
        stats.observe_peak(
            "cpu_risk_native_batch_bytes", elements * data.element_size()
        )
        return _NativeBatch(data, valid, ready)

    def _run(self, preparation):
        stats = self.bridge.proposer.performance_stats
        started = perf_counter()
        try:
            with torch.inference_mode(), torch.cuda.device(self.device):
                batch = preparation.native_batch
                if batch is not None:
                    if preparation.cancelled.is_set():
                        return
                    with stats.cpu_timer("cpu_risk_native_d2h_wall"):
                        batch.ready.synchronize()
                    if not batch.valid.all().item():
                        raise RuntimeError("CPU verifier requested retired native KV")
                offset = 0
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
                        if batch is not None:
                            count = data.numel()
                            with stats.cpu_timer("cpu_risk_native_d2h_wall"):
                                layer.target[:, :, lo:hi].copy_(
                                    batch.data[offset : offset + count].view(data.shape)
                                )
                            offset += count
                            continue
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
        self.native_pinned = None
        self.native_valid = None
