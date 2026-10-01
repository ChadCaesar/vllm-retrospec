# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future

import torch

from vllm.v1.spec_decode.retrospec.legacy.resident_cache import (
    RetroSpecResidentPageAccess,
)
from vllm.v1.spec_decode.retrospec.legacy.store.types import (
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationStaging,
    RetroSpecFullVerificationTicket,
    RetroSpecVerificationMissAdmission,
)


class _RetroSpecClusterPageStoreAdmissionMixin:
    def resolve_full_verification_tokens(
        self,
        layer_name: str,
        descriptors: Sequence[RetroSpecFullVerificationDescriptor],
    ) -> RetroSpecFullVerificationStaging:
        """Synchronously resolve a full-verification staging request."""
        return self.submit_full_verification_tokens(
            layer_name=layer_name,
            descriptors=descriptors,
        ).result()

    def submit_full_verification_tokens(
        self,
        layer_name: str,
        descriptors: Sequence[RetroSpecFullVerificationDescriptor],
    ) -> RetroSpecFullVerificationTicket:
        """Submit native CPU gather and H2D staging without blocking."""
        self.wait_for_resident_prefetches((layer_name,))

        with self._resident_state_lock:
            pool = self._layer_pools.get(layer_name)
            if pool is None:
                raise RuntimeError(
                    f"No RetroSpec page pool exists for layer {layer_name!r}"
                )

            transfer_buffer = self._get_full_verification_buffer(pool)
            source = pool.snapshot_full_verification_sources()
            return transfer_buffer.submit(source=source, descriptors=descriptors)

    def lookup_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        touch: bool = True,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
                layer_name,
                cluster_ids,
                page_ids,
            )
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)

            return resident_cache.lookup(
                cluster_ids=cluster_ids,
                page_ids=page_ids,
                cluster_groups=self._get_cluster_groups(
                    layer_name,
                    cluster_ids_cpu,
                ),
                allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
                allocated_page_ids=pool.allocated_page_ids,
                touch=touch,
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
            )

    def admit_resident_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
                layer_name,
                cluster_ids,
                page_ids,
            )
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)
            selected_cluster_ids, selected_page_ids = (
                self._select_resident_staging_prefix(
                    pool, cluster_ids_cpu, page_ids_cpu
                )
            )
            (
                source_page_ids,
                source_key_pages,
                source_value_pages,
                transfer_buffer,
                transfer_slot,
            ) = self._stage_resident_pages(pool, selected_page_ids)

            try:
                with resident_cache.mutation_guard():
                    access = resident_cache.admit_staged(
                        cluster_ids=selected_cluster_ids,
                        page_ids=selected_page_ids,
                        cluster_groups=self._get_cluster_groups(
                            layer_name,
                            selected_cluster_ids,
                        ),
                        allocated_cluster_ids=self._get_allocated_cluster_ids(
                            layer_name
                        ),
                        allocated_page_ids=pool.allocated_page_ids,
                        staging_page_ids=source_page_ids,
                        staging_key_pages=source_key_pages,
                        staging_value_pages=source_value_pages,
                        cluster_ids_cpu=selected_cluster_ids,
                        page_ids_cpu=selected_page_ids,
                    )
            except BaseException:
                resident_cache.synchronize_pending_copies()
                if transfer_slot is not None:
                    transfer_buffer.release_cpu_slot(transfer_slot, None)
                raise
            if transfer_slot is not None:
                transfer_buffer.release_cpu_slot(transfer_slot, access.ready_event)
            return resident_cache.lookup(
                cluster_ids=cluster_ids,
                page_ids=page_ids,
                cluster_groups=self._get_cluster_groups(layer_name, cluster_ids_cpu),
                allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
                allocated_page_ids=pool.allocated_page_ids,
                touch=False,
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
            )

    def admit_staged_clusters(
        self,
        layer_name: str,
        cluster_ids: torch.Tensor,
        logical_page_ids: torch.Tensor,
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
    ) -> RetroSpecResidentPageAccess:
        with self._resident_state_lock:
            cluster_ids_cpu, page_ids_cpu = self._validate_cluster_blocks(
                layer_name,
                cluster_ids,
                logical_page_ids,
            )
            pool, resident_cache = self._get_or_create_resident_cache(layer_name)

            with resident_cache.mutation_guard():
                return resident_cache.admit_staged(
                    cluster_ids=cluster_ids,
                    page_ids=logical_page_ids,
                    cluster_groups=self._get_cluster_groups(
                        layer_name,
                        cluster_ids_cpu,
                    ),
                    allocated_cluster_ids=self._get_allocated_cluster_ids(layer_name),
                    allocated_page_ids=pool.allocated_page_ids,
                    staging_page_ids=staging_page_ids,
                    staging_key_pages=staging_key_pages,
                    staging_value_pages=staging_value_pages,
                    cluster_ids_cpu=cluster_ids_cpu,
                    page_ids_cpu=page_ids_cpu,
                )

    @torch.inference_mode()
    def admit_verification_misses(
        self,
        admission: RetroSpecVerificationMissAdmission | None,
    ) -> None:
        """Admit compact verification misses without re-reading GPU metadata."""
        if self._resident_admission_frozen:
            return
        if admission is None or admission.cluster_ids_cpu.numel() == 0:
            return

        with self._resident_state_lock:
            pool, resident_cache = self._get_or_create_resident_cache(
                admission.layer_name
            )
            cluster_groups = self._get_cluster_groups(
                admission.layer_name, admission.cluster_ids_cpu
            )
            with resident_cache.mutation_guard():
                resident_cache.admit_staged(
                    cluster_ids=admission.cluster_ids_cpu,
                    page_ids=admission.logical_page_ids_cpu,
                    cluster_groups=cluster_groups,
                    allocated_cluster_ids=self._get_allocated_cluster_ids(
                        admission.layer_name
                    ),
                    allocated_page_ids=pool.allocated_page_ids,
                    staging_page_ids=admission.staging_page_ids_cpu,
                    staging_key_pages=admission.staging_key_pages,
                    staging_value_pages=admission.staging_value_pages,
                    cluster_ids_cpu=admission.cluster_ids_cpu,
                    page_ids_cpu=admission.logical_page_ids_cpu,
                    reuse_ready_event=admission.staging_ready_event,
                    lookup_after_admit=False,
                )

    def submit_verification_miss_admission(
        self,
        admission: RetroSpecVerificationMissAdmission | None,
    ) -> None:
        """Move post-attention LRU admission off the model thread."""
        self._reap_verification_admissions(wait=False)
        if admission is None or admission.cluster_ids_cpu.numel() == 0:
            return
        future = self._resident_prefetch_executor.submit(
            self._VERIFICATION_ADMISSION_PRIORITY,
            self.admit_verification_misses,
            admission,
        )
        self._verification_admission_futures.append(future)
        if self.performance_stats is not None:
            self.performance_stats.add_counter("verification_async_admissions")

    def wait_for_verification_admissions(self) -> None:
        """Drain sparse admissions before a bandwidth-heavy phase starts."""
        self._reap_verification_admissions(wait=True)

    def _reap_verification_admissions(self, wait: bool) -> None:
        remaining: deque[Future[None]] = deque()
        while self._verification_admission_futures:
            future = self._verification_admission_futures.popleft()
            if wait or future.done():
                future.result()
            else:
                remaining.append(future)
        self._verification_admission_futures = remaining

    def get_resident_page_storage(
        self,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return resident GPU pages after waiting on pending H2D copies.

        The wait is inserted into the current CUDA stream and does not block the
        CPU thread.
        """
        _, resident_cache = self._get_or_create_resident_cache(layer_name)
        resident_cache.wait_for_pending_copies()

        return (
            resident_cache.key_pages,
            resident_cache.value_pages,
        )

    def resident_capacity(
        self,
        layer_name: str,
    ) -> int:
        _, resident_cache = self._get_or_create_resident_cache(layer_name)
        return resident_cache.capacity

    def num_resident_pages(
        self,
        layer_name: str,
    ) -> int:
        resident_cache = self._resident_caches.get(layer_name)
        return 0 if resident_cache is None else resident_cache.num_resident_pages

    def num_resident_clusters(
        self,
        layer_name: str,
    ) -> int:
        resident_cache = self._resident_caches.get(layer_name)
        return 0 if resident_cache is None else resident_cache.num_resident_clusters

    def num_resident_groups(
        self,
        layer_name: str,
    ) -> int:
        resident_cache = self._resident_caches.get(layer_name)
        return 0 if resident_cache is None else resident_cache.num_resident_groups

    def num_allocated_pages(
        self,
        layer_name: str,
    ) -> int:
        pool = self._layer_pools.get(layer_name)
        return 0 if pool is None else pool.num_allocated_pages
