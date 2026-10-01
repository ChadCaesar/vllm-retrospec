# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import perf_counter

import torch

from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    RetroSpecClusterBlockMetadata,
    RetroSpecClusterBlockTable,
    RetroSpecStagedClusterInput,
    RetroSpecStagedTokenKV,
    _PinnedStagingSlot,
)


class _RetroSpecClusterPageStoreBuildMixin:
    @staticmethod
    def _validate_token_kv_input(
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> None:
        if token_keys.ndim != 3:
            raise ValueError(
                "token_keys must have shape [num_kv_heads, num_tokens, head_size]"
            )
        if token_values.shape != token_keys.shape:
            raise ValueError("token_keys and token_values must have equal shapes")
        if token_values.dtype != token_keys.dtype:
            raise ValueError("token_keys and token_values must have equal dtypes")
        if token_values.device != token_keys.device:
            raise ValueError("Token keys and values must be on one device")

    @classmethod
    def _validate_cluster_input_metadata(
        cls,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> None:
        cls._validate_token_kv_input(token_keys, token_values)

        token_shape = token_keys.shape[:2]
        if assignments.shape != token_shape:
            raise ValueError("assignments must have shape [num_kv_heads, num_tokens]")
        if token_offsets_in_cluster.shape != token_shape:
            raise ValueError(
                "token_offsets_in_cluster must have shape [num_kv_heads, num_tokens]"
            )
        if cluster_token_counts.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if cluster_token_counts.shape[0] != token_keys.shape[0]:
            raise ValueError(
                "cluster_token_counts KV-head count does not match token KV"
            )
        metadata = (
            ("assignments", assignments),
            ("cluster_token_counts", cluster_token_counts),
            ("token_offsets_in_cluster", token_offsets_in_cluster),
        )
        for name, tensor in metadata:
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must use an integral dtype")
            if tensor.device != token_keys.device:
                raise ValueError(f"{name} and token KV must be on one device")

    @staticmethod
    def _validate_staged_cluster_metadata(
        staged_token_kv: RetroSpecStagedTokenKV,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> None:
        token_shape = staged_token_kv.token_keys.shape[:2]

        if assignments.shape != token_shape:
            raise ValueError("assignments must have shape [num_kv_heads, num_tokens]")
        if token_offsets_in_cluster.shape != token_shape:
            raise ValueError(
                "token_offsets_in_cluster must have shape [num_kv_heads, num_tokens]"
            )
        if cluster_token_counts.ndim != 2:
            raise ValueError(
                "cluster_token_counts must have shape [num_kv_heads, num_clusters]"
            )
        if cluster_token_counts.shape[0] != token_shape[0]:
            raise ValueError(
                "cluster_token_counts KV-head count does not match token KV"
            )
        metadata = (
            ("Assignments", assignments),
            ("Cluster counts", cluster_token_counts),
            ("Cluster token offsets", token_offsets_in_cluster),
        )
        for name, tensor in metadata:
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must use an integral dtype")
            if tensor.device != staged_token_kv.source_device:
                raise ValueError(f"{name} must remain on the token KV source device")

    @staticmethod
    def _canonical_cuda_device(device: torch.device) -> torch.device:
        if device.type != "cuda":
            raise ValueError("RetroSpec staging requires a CUDA source device")

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        return torch.device("cuda", device_index)

    def _get_offload_stream(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream:
        canonical_device = self._canonical_cuda_device(device)
        stream = self._offload_streams.get(canonical_device)

        if stream is None:
            stream = torch.cuda.Stream(device=canonical_device)
            self._offload_streams[canonical_device] = stream

        return stream

    def _acquire_pinned_staging_slot(
        self,
        source_device: torch.device,
    ) -> _PinnedStagingSlot:
        canonical_device = self._canonical_cuda_device(source_device)

        with self._pinned_staging_lock:
            slots = self._pinned_staging_slots.setdefault(canonical_device, [])

            for slot in slots:
                if not slot.in_use:
                    slot.in_use = True
                    return slot

            if len(slots) >= self.max_pending_cluster_builds:
                raise RuntimeError(
                    "No RetroSpec cluster-build staging slot is available"
                )

            slot = _PinnedStagingSlot(
                source_device=canonical_device,
                max_bytes=(
                    self.max_pinned_memory_bytes // 2 // self.max_pending_cluster_builds
                ),
                pinned_memory=self._pinned_memory,
                in_use=True,
            )
            slots.append(slot)
            return slot

    def _release_pinned_staging_slot(
        self,
        slot: _PinnedStagingSlot | None,
    ) -> None:
        if slot is None:
            return

        with self._pinned_staging_lock:
            if not slot.in_use:
                raise RuntimeError("Pinned staging slot has already been released")
            slot.in_use = False

    def discard_staged_token_kv(
        self,
        staged: RetroSpecStagedTokenKV,
    ) -> None:
        """Wait for an abandoned token-KV transfer and release its slot."""
        try:
            staged.wait()
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    def discard_staged_clusters(
        self,
        staged: RetroSpecStagedClusterInput,
    ) -> None:
        """Wait for abandoned cluster inputs and release their slot."""
        try:
            staged.wait()
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    def stage_token_kv(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
    ) -> RetroSpecStagedTokenKV:
        """Start staging token KV before segmented clustering."""
        self._validate_token_kv_input(token_keys, token_values)
        source_device = token_keys.device

        if source_device.type == "cpu":
            return RetroSpecStagedTokenKV(
                token_keys=token_keys,
                token_values=token_values,
                source_device=source_device,
                ready_event=None,
            )

        if source_device.type != "cuda":
            raise ValueError("CPU-backed cluster staging requires CPU or CUDA inputs")

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "token_kv_d2h_bytes",
                token_keys.nbytes + token_values.nbytes,
            )

        if not self.pin_memory:
            return RetroSpecStagedTokenKV(
                token_keys=token_keys.to(device="cpu", non_blocking=False),
                token_values=token_values.to(device="cpu", non_blocking=False),
                source_device=source_device,
                ready_event=None,
            )

        staging_slot = self._acquire_pinned_staging_slot(source_device)
        offload_stream: torch.cuda.Stream | None = None

        try:
            staged_token_keys, staged_token_values = staging_slot.reserve_token_kv(
                token_keys,
                token_values,
            )

            offload_stream = self._get_offload_stream(source_device)
            current_stream = torch.cuda.current_stream(source_device)
            offload_stream.wait_stream(current_stream)

            ready_event = torch.cuda.Event()

            with torch.cuda.stream(offload_stream):
                timer = (
                    None
                    if self.performance_stats is None
                    else self.performance_stats.start_cuda_timer(
                        "prefill_token_kv_d2h", offload_stream
                    )
                )
                try:
                    staged_token_keys.copy_(token_keys, non_blocking=True)
                    staged_token_values.copy_(token_values, non_blocking=True)
                finally:
                    if self.performance_stats is not None:
                        self.performance_stats.stop_cuda_timer(timer, offload_stream)
                ready_event.record(offload_stream)

            token_keys.record_stream(offload_stream)
            token_values.record_stream(offload_stream)
        except BaseException:
            if offload_stream is not None:
                offload_stream.synchronize()
            self._release_pinned_staging_slot(staging_slot)
            raise

        return RetroSpecStagedTokenKV(
            token_keys=staged_token_keys,
            token_values=staged_token_values,
            source_device=source_device,
            ready_event=ready_event,
            staging_slot=staging_slot,
        )

    def finish_stage_clusters(
        self,
        staged_token_kv: RetroSpecStagedTokenKV,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> RetroSpecStagedClusterInput:
        """Stage clustering metadata after GPU clustering finishes."""
        self._validate_staged_cluster_metadata(
            staged_token_kv,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )
        source_device = staged_token_kv.source_device

        if source_device.type == "cpu":
            return RetroSpecStagedClusterInput(
                token_keys=staged_token_kv.token_keys,
                token_values=staged_token_kv.token_values,
                assignments=assignments,
                cluster_token_counts=cluster_token_counts,
                token_offsets_in_cluster=token_offsets_in_cluster,
                metadata_device=source_device,
                ready_event=None,
            )

        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "cluster_metadata_d2h_bytes",
                assignments.nbytes
                + cluster_token_counts.nbytes
                + token_offsets_in_cluster.nbytes,
            )

        if not self.pin_memory:
            return RetroSpecStagedClusterInput(
                token_keys=staged_token_kv.token_keys,
                token_values=staged_token_kv.token_values,
                assignments=assignments.to(device="cpu", non_blocking=False),
                cluster_token_counts=cluster_token_counts.to(
                    device="cpu",
                    non_blocking=False,
                ),
                token_offsets_in_cluster=token_offsets_in_cluster.to(
                    device="cpu",
                    non_blocking=False,
                ),
                metadata_device=source_device,
                ready_event=None,
            )

        staging_slot = staged_token_kv.staging_slot
        if staging_slot is None:
            raise RuntimeError("Pinned token KV does not own a staging slot")

        offload_stream = self._get_offload_stream(source_device)

        try:
            (
                staged_assignments,
                staged_cluster_token_counts,
                staged_token_offsets,
            ) = staging_slot.reserve_cluster_metadata(
                assignments,
                cluster_token_counts,
                token_offsets_in_cluster,
            )

            current_stream = torch.cuda.current_stream(source_device)

            # Token-KV copies were enqueued earlier on the same offload stream.
            # This wait delays only metadata D2H until clustering has completed.
            offload_stream.wait_stream(current_stream)

            ready_event = torch.cuda.Event()

            with torch.cuda.stream(offload_stream):
                timer = (
                    None
                    if self.performance_stats is None
                    else self.performance_stats.start_cuda_timer(
                        "prefill_cluster_metadata_d2h", offload_stream
                    )
                )
                try:
                    staged_assignments.copy_(assignments, non_blocking=True)
                    staged_cluster_token_counts.copy_(
                        cluster_token_counts, non_blocking=True
                    )
                    staged_token_offsets.copy_(
                        token_offsets_in_cluster, non_blocking=True
                    )
                finally:
                    if self.performance_stats is not None:
                        self.performance_stats.stop_cuda_timer(timer, offload_stream)
                ready_event.record(offload_stream)

            assignments.record_stream(offload_stream)
            cluster_token_counts.record_stream(offload_stream)
            token_offsets_in_cluster.record_stream(offload_stream)
        except BaseException:
            # Ownership remains with staged_token_kv until this method returns.
            # Synchronize partially queued metadata copies before it is discarded.
            offload_stream.synchronize()
            raise

        return RetroSpecStagedClusterInput(
            token_keys=staged_token_kv.token_keys,
            token_values=staged_token_kv.token_values,
            assignments=staged_assignments,
            cluster_token_counts=staged_cluster_token_counts,
            token_offsets_in_cluster=staged_token_offsets,
            metadata_device=source_device,
            ready_event=ready_event,
            staging_slot=staging_slot,
        )

    def stage_clusters(
        self,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
    ) -> RetroSpecStagedClusterInput:
        """Stage complete inputs when overlap is not controlled by the caller."""
        self._validate_cluster_input_metadata(
            token_keys,
            token_values,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )
        staged_token_kv = self.stage_token_kv(token_keys, token_values)

        try:
            return self.finish_stage_clusters(
                staged_token_kv,
                assignments,
                cluster_token_counts,
                token_offsets_in_cluster,
            )
        except BaseException:
            self.discard_staged_token_kv(staged_token_kv)
            raise

    def store_staged_clusters(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        staged: RetroSpecStagedClusterInput,
    ) -> RetroSpecClusterBlockTable:
        """Wait for staged D2H copies and construct CPU cluster pages."""
        wait_started_at = perf_counter()
        try:
            staged.wait()
            if self.performance_stats is not None:
                self.performance_stats.record_cpu_time(
                    "cluster_build_wait",
                    perf_counter() - wait_started_at,
                )

            build_started_at = perf_counter()
            result = self.store_clusters(
                layer_name=layer_name,
                request_id=request_id,
                cluster_start=cluster_start,
                token_keys=staged.token_keys,
                token_values=staged.token_values,
                assignments=staged.assignments,
                cluster_token_counts=staged.cluster_token_counts,
                token_offsets_in_cluster=staged.token_offsets_in_cluster,
                metadata_device=staged.metadata_device,
            )
            if self.performance_stats is not None:
                self.performance_stats.record_cpu_time(
                    "cluster_page_build",
                    perf_counter() - build_started_at,
                )
                self.performance_stats.add_counter("cluster_builds")
            return result
        finally:
            self._release_pinned_staging_slot(staged.staging_slot)

    @staticmethod
    def _move_to_storage(
        tensor: torch.Tensor,
        storage_device: torch.device,
    ) -> torch.Tensor:
        if tensor.device == storage_device:
            return tensor

        return tensor.to(
            device=storage_device,
            non_blocking=False,
        )

    def store_clusters(
        self,
        layer_name: str,
        request_id: str,
        cluster_start: int,
        token_keys: torch.Tensor,
        token_values: torch.Tensor,
        assignments: torch.Tensor,
        cluster_token_counts: torch.Tensor,
        token_offsets_in_cluster: torch.Tensor,
        metadata_device: torch.device | None = None,
    ) -> RetroSpecClusterBlockTable:
        """Pack token KV into per-head, per-cluster backing pages."""
        if cluster_start < 0:
            raise ValueError("cluster_start must be non-negative")

        self._validate_cluster_input_metadata(
            token_keys,
            token_values,
            assignments,
            cluster_token_counts,
            token_offsets_in_cluster,
        )

        pool = self._get_or_create_pool(
            layer_name,
            token_keys,
            metadata_device=metadata_device,
        )
        storage_keys = self._move_to_storage(token_keys, pool.storage_device)
        storage_values = self._move_to_storage(token_values, pool.storage_device)
        storage_assignments = self._move_to_storage(
            assignments,
            pool.storage_device,
        )
        storage_cluster_counts = self._move_to_storage(
            cluster_token_counts, pool.storage_device
        )
        storage_token_offsets = self._move_to_storage(
            token_offsets_in_cluster,
            pool.storage_device,
        )

        if torch.any(storage_cluster_counts < 0).item():
            raise ValueError("cluster_token_counts must be non-negative")

        cluster_page_counts = torch.div(
            storage_cluster_counts.to(dtype=torch.int64) + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        total_pages = int(cluster_page_counts.sum().item())
        if self.performance_stats is not None:
            self.performance_stats.add_counter(
                "cluster_pages_built",
                total_pages,
            )

        allocated_page_ids = pool.allocate(total_pages)

        try:
            page_ids, page_token_counts, full_descriptor = pool.build_cluster_pages(
                allocated_page_ids=allocated_page_ids,
                token_keys=storage_keys,
                token_values=storage_values,
                assignments=storage_assignments,
                cluster_token_counts=storage_cluster_counts,
                token_offsets_in_cluster=storage_token_offsets,
                num_workers=self.cpu_page_build_workers,
            )
        except Exception:
            pool.free(allocated_page_ids)
            raise

        cluster_ids: torch.Tensor | None = None
        with self._resident_state_lock:
            try:
                cluster_ids = self._allocate_cluster_ids(
                    layer_name=layer_name,
                    request_id=request_id,
                    cluster_start=cluster_start,
                    cluster_token_counts=storage_cluster_counts,
                    page_ids=page_ids,
                    page_token_counts=page_token_counts,
                )
                self._resize_resident_cache(layer_name, pool)
            except Exception:
                if cluster_ids is not None:
                    self._free_cluster_ids(layer_name, cluster_ids)
                pool.free(allocated_page_ids)
                raise

        assert cluster_ids is not None
        return RetroSpecClusterBlockTable(
            cluster_ids=cluster_ids,
            page_metadata=RetroSpecClusterBlockMetadata(
                page_ids=page_ids,
                page_token_counts=page_token_counts,
            ),
            full_verification_descriptor=full_descriptor,
        )

    def free(
        self,
        layer_name: str,
        block_table: RetroSpecClusterBlockTable,
    ) -> None:
        self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
            pool = self._layer_pools.get(layer_name)
            if pool is None:
                raise RuntimeError(
                    f"No RetroSpec page pool exists for layer {layer_name!r}"
                )

            resident_cache = self._resident_caches.get(layer_name)
            if resident_cache is not None:
                with resident_cache.mutation_guard():
                    resident_cache.invalidate(block_table.cluster_ids)

            pool.free(block_table.page_metadata.page_ids)
            self._free_cluster_ids(layer_name, block_table.cluster_ids)
            self._resize_resident_cache(layer_name, pool)

    def gather_pages(
        self,
        layer_name: str,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather selected pages on the backing-store device."""
        if page_ids.shape != page_token_counts.shape:
            raise ValueError("page_ids and page_token_counts must have equal shapes")

        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        key_pages, value_pages = pool.read(page_ids)
        storage_page_token_counts = page_token_counts.to(
            device=pool.storage_device,
            dtype=torch.int32,
        )

        token_offsets = torch.arange(
            self.page_size,
            dtype=torch.int32,
            device=pool.storage_device,
        )
        token_mask = token_offsets.view(
            *((1,) * storage_page_token_counts.ndim),
            self.page_size,
        ) < storage_page_token_counts.unsqueeze(-1)

        batch_size, num_kv_heads = page_ids.shape[:2]

        exact_keys = key_pages.reshape(
            batch_size,
            num_kv_heads,
            -1,
            pool.head_size,
        )
        exact_values = value_pages.reshape_as(exact_keys)
        exact_token_mask = token_mask.reshape(
            batch_size,
            num_kv_heads,
            -1,
        )

        exact_keys.masked_fill_(
            ~exact_token_mask.unsqueeze(-1),
            0.0,
        )
        exact_values.masked_fill_(
            ~exact_token_mask.unsqueeze(-1),
            0.0,
        )

        return (
            exact_keys.contiguous(),
            exact_values.contiguous(),
            exact_token_mask.contiguous(),
        )

    def read_page_storage(
        self,
        layer_name: str,
        page_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read logical page handles from the layer's pageable CPU slabs."""
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        return pool.read(page_ids)

    def get_storage_device(
        self,
        layer_name: str,
    ) -> torch.device:
        pool = self._layer_pools.get(layer_name)
        if pool is None:
            raise RuntimeError(
                f"No RetroSpec page pool exists for layer {layer_name!r}"
            )

        return pool.storage_device
