# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType

import torch

from vllm import _custom_ops as ops

from .cluster_identity import RetroSpecClusterGroup
from .resident_cache_types import (
    RetroSpecPreparedResidentAdmission,
    RetroSpecResidentLruCapture,
    RetroSpecResidentPageAccess,
    RetroSpecResolvedResidentLru,
    _ClusterId,
    _LogicalPages,
)


class _ResidentAdmissionMixin:
    """Validate, stage, admit and invalidate complete resident clusters."""

    def _validate_source_pages(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
    ) -> None:
        if key_pages.shape != value_pages.shape:
            raise ValueError("Source key and value page shapes must match")
        if key_pages.ndim != 3:
            raise ValueError(
                "Source pages must have shape [pages, page_size, head_size]"
            )
        if key_pages.shape[1:] != (
            self.page_size,
            self.head_size,
        ):
            raise ValueError("Source page shape does not match resident cache")
        if key_pages.dtype != self.dtype:
            raise ValueError("Source key dtype does not match resident cache")
        if value_pages.dtype != self.dtype:
            raise ValueError("Source value dtype does not match resident cache")
        if key_pages.device != value_pages.device:
            raise ValueError("Source key and value pages must use one device")

    def _validate_backing_pages(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
    ) -> None:
        self._validate_source_pages(key_pages, value_pages)
        if key_pages.device.type != "cpu":
            raise ValueError("Resident cache admission requires CPU backing pages")

    def _copy_cluster_to_slots(
        self,
        source_page_ids: tuple[int, ...],
        slots: tuple[int, ...],
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> None:
        """Enqueue one complete cluster on the dedicated copy stream."""
        self._copy_pages_to_slots(
            source_page_ids,
            slots,
            source_key_pages,
            source_value_pages,
        )

    def _copy_pages_to_slots(
        self,
        source_page_ids: Sequence[int],
        slots: Sequence[int],
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
    ) -> None:
        """Enqueue one coalesced K/V page mapping on the copy stream."""
        if len(source_page_ids) != len(slots):
            raise ValueError("Source page and destination slot counts must match")
        if not source_page_ids:
            return

        block_mapping = torch.tensor(
            tuple(zip(source_page_ids, slots, strict=True)),
            dtype=torch.int64,
            device="cpu",
        )
        block_size = self.page_size * self.head_size * source_key_pages.element_size()
        with torch.cuda.stream(self._copy_stream):
            span_count = ops.copy_kv_blocks_coalesced(
                source_key_pages,
                source_value_pages,
                self.key_pages,
                self.value_pages,
                block_size,
                block_mapping,
            )

        stats = self.performance_stats
        if (
            stats is not None
            and stats.enabled
            and stats.cuda_timing_level == "detailed"
        ):
            stats.add_counter("resident_copy_pages", len(source_page_ids))
            stats.add_counter("resident_copy_spans", span_count)

    def _prepare_admission_from_sources(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        source_page_ids: torch.Tensor,
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
    ) -> RetroSpecPreparedResidentAdmission:
        """Parse immutable metadata without reading live resident state."""
        self._validate_source_pages(source_key_pages, source_value_pages)

        if source_page_ids.shape != page_ids.shape:
            raise ValueError(
                "Source page IDs must have the same shape as logical page IDs"
            )
        if source_page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Source page IDs must use an integral dtype")

        (
            cluster_ids_cpu,
            page_ids_cpu,
            parsed_cluster_ids,
            parsed_cluster_pages,
            valid_positions,
        ) = self._parse_clusters(
            cluster_ids,
            page_ids,
            None,
            None,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )
        self._validate_cluster_groups(parsed_cluster_ids, cluster_groups)

        source_page_ids_cpu = source_page_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        if torch.any(source_page_ids_cpu < -1).item():
            raise ValueError("Source page IDs must be at least -1")

        flat_source_page_ids = source_page_ids_cpu.reshape(
            len(parsed_cluster_ids),
            source_page_ids_cpu.shape[-1],
        )
        cluster_page_map: dict[_ClusterId, _LogicalPages] = {}
        cluster_source_ids: dict[_ClusterId, tuple[int, ...]] = {}

        for cluster_id, logical_pages, positions, source_row in zip(
            parsed_cluster_ids,
            parsed_cluster_pages,
            valid_positions,
            flat_source_page_ids,
            strict=True,
        ):
            if cluster_id is None:
                continue

            cluster_page_map[cluster_id] = logical_pages
            current_source_ids = tuple(
                int(source_row[position]) for position in positions
            )
            previous_source_ids = cluster_source_ids.get(cluster_id)

            # A duplicated cluster may occur in more than one batch position.
            # Prefer an occurrence with a complete source mapping.
            if previous_source_ids is None or any(
                source_page_id < 0 for source_page_id in previous_source_ids
            ):
                cluster_source_ids[cluster_id] = current_source_ids

        requested_clusters = tuple(
            self._priority_ordered_clusters(
                parsed_cluster_ids,
                cluster_ids_cpu.shape,
            )
        )
        referenced_page_ids = frozenset(
            page_id
            for logical_pages in cluster_page_map.values()
            for page_id in logical_pages
        )

        return RetroSpecPreparedResidentAdmission(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=MappingProxyType(dict(cluster_groups)),
            source_page_ids=source_page_ids_cpu,
            source_key_pages=source_key_pages,
            source_value_pages=source_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
            parsed_cluster_ids=tuple(parsed_cluster_ids),
            parsed_cluster_pages=tuple(parsed_cluster_pages),
            valid_positions=tuple(valid_positions),
            requested_clusters=requested_clusters,
            cluster_page_map=MappingProxyType(cluster_page_map),
            cluster_source_ids=MappingProxyType(cluster_source_ids),
            referenced_cluster_ids=frozenset(cluster_page_map),
            referenced_page_ids=referenced_page_ids,
        )

    @staticmethod
    def _validate_prepared_admission_allocations(
        prepared: RetroSpecPreparedResidentAdmission,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
    ) -> None:
        """Reject a descriptor whose request storage was released or reused."""
        missing_cluster_ids = prepared.referenced_cluster_ids.difference(
            allocated_cluster_ids
        )
        if missing_cluster_ids:
            cluster_id = min(missing_cluster_ids)
            raise RuntimeError(
                f"Prepared admission references unallocated cluster {cluster_id}"
            )

        missing_page_ids = prepared.referenced_page_ids.difference(allocated_page_ids)
        if missing_page_ids:
            page_id = min(missing_page_ids)
            raise RuntimeError(
                f"Prepared admission references unallocated logical page {page_id}"
            )

    def prepare_staged_admission(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
    ) -> RetroSpecPreparedResidentAdmission:
        """Prepare staged admission metadata without taking mutation_guard()."""
        self._validate_source_pages(staging_key_pages, staging_value_pages)
        if (
            staging_key_pages.device.type != "cpu"
            and staging_key_pages.device != self.device
        ):
            raise ValueError("Staging pages must use CPU or the resident CUDA device")

        return self._prepare_admission_from_sources(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            source_page_ids=staging_page_ids,
            source_key_pages=staging_key_pages,
            source_value_pages=staging_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )

    def _plan_prepared_admission_targets(
        self,
        prepared: RetroSpecPreparedResidentAdmission,
    ) -> tuple[
        tuple[_ClusterId, ...],
        set[_ClusterId],
        tuple[_ClusterId, ...],
        int,
        dict[RetroSpecClusterGroup, int],
    ]:
        """Plan the priority prefix against the current resident state."""
        target_clusters: list[_ClusterId] = []
        target_page_count = 0
        for cluster_id in prepared.requested_clusters:
            cluster_page_count = len(prepared.cluster_page_map[cluster_id])
            if cluster_page_count > self._logical_capacity:
                continue
            if target_page_count + cluster_page_count > self._logical_capacity:
                break
            target_clusters.append(cluster_id)
            target_page_count += cluster_page_count

        missing_targets = tuple(
            cluster_id
            for cluster_id in target_clusters
            if cluster_id not in self._cluster_to_slots
        )
        required_page_count = sum(
            len(prepared.cluster_page_map[cluster_id]) for cluster_id in missing_targets
        )
        incoming_group_pages: dict[RetroSpecClusterGroup, int] = {}
        for cluster_id in missing_targets:
            group = prepared.cluster_groups[cluster_id]
            incoming_group_pages[group] = incoming_group_pages.get(group, 0) + len(
                prepared.cluster_page_map[cluster_id]
            )

        target_cluster_tuple = tuple(target_clusters)
        return (
            target_cluster_tuple,
            set(target_cluster_tuple),
            missing_targets,
            required_page_count,
            incoming_group_pages,
        )

    def capture_prepared_admission_lru(
        self,
        prepared: RetroSpecPreparedResidentAdmission,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        stream: torch.cuda.Stream,
    ) -> RetroSpecResidentLruCapture | None:
        """Capture stable eviction descriptors without waiting on the GPU."""
        self._validate_prepared_admission_allocations(
            prepared,
            allocated_cluster_ids,
            allocated_page_ids,
        )
        _, _, _, required_page_count, _ = self._plan_prepared_admission_targets(
            prepared
        )
        if self.num_resident_pages + required_page_count <= self._logical_capacity:
            return None

        self._reap_completed_copy_batches()
        stats = self.performance_stats
        if stats is not None and stats.enabled:
            stats.add_counter("resident_lru_snapshot_requested")
        return self._capture_group_lru_from_gpu(self._group_states.keys(), stream)

    def _admit_from_sources(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        source_page_ids: torch.Tensor,
        source_key_pages: torch.Tensor,
        source_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
        prepared: RetroSpecPreparedResidentAdmission | None = None,
        resolved_lru: RetroSpecResolvedResidentLru | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from CPU or GPU source pages."""
        if prepared is None:
            prepared = self._prepare_admission_from_sources(
                cluster_ids=cluster_ids,
                page_ids=page_ids,
                cluster_groups=cluster_groups,
                source_page_ids=source_page_ids,
                source_key_pages=source_key_pages,
                source_value_pages=source_value_pages,
                cluster_ids_cpu=cluster_ids_cpu,
                page_ids_cpu=page_ids_cpu,
            )

        self._validate_prepared_admission_allocations(
            prepared,
            allocated_cluster_ids,
            allocated_page_ids,
        )

        cluster_ids = prepared.cluster_ids
        page_ids = prepared.page_ids
        cluster_groups = prepared.cluster_groups
        source_key_pages = prepared.source_key_pages
        source_value_pages = prepared.source_value_pages
        cluster_ids_cpu = prepared.cluster_ids_cpu
        page_ids_cpu = prepared.page_ids_cpu
        requested_clusters = prepared.requested_clusters
        cluster_page_map = prepared.cluster_page_map
        cluster_source_ids = prepared.cluster_source_ids

        for cluster_id in requested_clusters:
            logical_pages = cluster_page_map[cluster_id]
            resident_pages = self._cluster_to_pages.get(cluster_id)

            if resident_pages is not None:
                if resident_pages != logical_pages:
                    raise RuntimeError(
                        "Cluster ID conflicts with an existing resident descriptor"
                    )
                if self._cluster_to_group.get(cluster_id) != cluster_groups[cluster_id]:
                    raise RuntimeError(
                        "Resident cluster group does not match requested ownership"
                    )

            for logical_page_id in logical_pages:
                resident_owner = self._page_to_cluster.get(logical_page_id)
                if resident_owner is not None and resident_owner != cluster_id:
                    raise RuntimeError(
                        "Logical page ownership conflicts with an existing "
                        "resident cluster"
                    )

        (
            _,
            target_cluster_set,
            missing_targets,
            required_page_count,
            incoming_group_pages,
        ) = self._plan_prepared_admission_targets(prepared)

        # Validate every source before changing resident ownership.
        for cluster_id in missing_targets:
            logical_pages = cluster_page_map[cluster_id]
            source_ids = cluster_source_ids.get(cluster_id)

            if source_ids is None or len(source_ids) != len(logical_pages):
                raise RuntimeError("Missing source-page mapping for resident admission")
            if any(source_page_id < 0 for source_page_id in source_ids):
                raise RuntimeError(
                    "A missing cluster does not have complete source pages"
                )
            if any(
                source_page_id >= source_key_pages.shape[0]
                for source_page_id in source_ids
            ):
                raise RuntimeError("Source page ID exceeds source storage capacity")

        if mutation_stream is None:
            mutation_stream = torch.cuda.current_stream(self.device)
        if reuse_ready_event is not None:
            mutation_stream.wait_event(reuse_ready_event)

        previous_gate_ready = self._snapshot_group_hit_gate_ready(incoming_group_pages)
        victims: tuple[_ClusterId, ...] = ()
        if self.num_resident_pages + required_page_count > self._logical_capacity:
            self._reap_completed_copy_batches()
            snapshot_applied = False
            if resolved_lru is not None:
                snapshot_applied = self._apply_resolved_group_lru(resolved_lru)
                stats = self.performance_stats
                if stats is not None and stats.enabled:
                    stats.add_counter(
                        "resident_lru_snapshot_applied"
                        if snapshot_applied
                        else "resident_lru_snapshot_retried"
                    )
            if not snapshot_applied:
                self._refresh_group_lru_from_gpu(
                    self._group_states.keys(), mutation_stream
                )
            protected_clusters = target_cluster_set | set(self._pending_cluster_events)
            victims = self._select_victim_clusters(
                protected_clusters,
                incoming_group_pages,
                required_page_count,
            )
            victim_page_count = sum(
                len(self._cluster_to_slots[cluster_id]) for cluster_id in victims
            )

            if (
                self.num_resident_pages - victim_page_count + required_page_count
                > self._logical_capacity
                and self._pending_cluster_events
            ):
                self.synchronize_pending_copies()
                self._refresh_group_lru_from_gpu(
                    self._group_states.keys(), mutation_stream
                )
                victims = self._select_victim_clusters(
                    target_cluster_set,
                    incoming_group_pages,
                    required_page_count,
                )
                victim_page_count = sum(
                    len(self._cluster_to_slots[cluster_id]) for cluster_id in victims
                )

            if (
                self.num_resident_pages - victim_page_count + required_page_count
                > self._logical_capacity
            ):
                raise RuntimeError(
                    "Resident cache cannot free enough slots for priority admission"
                )
            for cluster_id in victims:
                group = self._cluster_to_group[cluster_id]
                previous_gate_ready.setdefault(
                    group, self._is_group_hit_gate_ready(group)
                )
            self._evict_clusters(victims, update_stream=mutation_stream)

        copy_scheduled = False
        copied_cluster_ids: tuple[_ClusterId, ...] = ()
        reserved_slots: tuple[int, ...] = ()

        try:
            if missing_targets:
                slots_released_event = torch.cuda.Event()
                slots_released_event.record(mutation_stream)
                self._copy_stream.wait_event(slots_released_event)

            registration_entries: list[
                tuple[
                    _ClusterId,
                    RetroSpecClusterGroup,
                    _LogicalPages,
                    tuple[int, ...],
                ]
            ] = []
            source_page_ids: list[int] = []
            destination_slots: list[int] = []

            reserved_slots = tuple(sorted(self._free_slots)[:required_page_count])
            if len(reserved_slots) != required_page_count:
                raise RuntimeError("Resident cache does not have enough free GPU slots")
            self._free_slots.difference_update(reserved_slots)

            slot_offset = 0
            for cluster_id in missing_targets:
                logical_pages = cluster_page_map[cluster_id]
                cluster_page_count = len(logical_pages)
                slots = reserved_slots[slot_offset : slot_offset + cluster_page_count]
                slot_offset += cluster_page_count
                source_ids = cluster_source_ids[cluster_id]
                source_page_ids.extend(source_ids)
                destination_slots.extend(slots)
                registration_entries.append(
                    (
                        cluster_id,
                        cluster_groups[cluster_id],
                        logical_pages,
                        slots,
                    )
                )

            if missing_targets:
                copy_scheduled = True
                self._copy_pages_to_slots(
                    source_page_ids,
                    destination_slots,
                    source_key_pages,
                    source_value_pages,
                )
                self._register_clusters(registration_entries)
                copied_cluster_ids = tuple(missing_targets)
        except BaseException:
            if not copied_cluster_ids:
                self._free_slots.update(reserved_slots)
            raise
        finally:
            if copy_scheduled:
                try:
                    self._publish_handle_delta(
                        copied_cluster_ids,
                        previous_gate_ready,
                        self._copy_stream,
                    )
                finally:
                    self._record_copy_batch(
                        cluster_ids=copied_cluster_ids,
                        source_key_pages=source_key_pages,
                        source_value_pages=source_value_pages,
                    )
            elif victims:
                self._publish_handle_delta(
                    (),
                    previous_gate_ready,
                    mutation_stream,
                )

        for cluster_id in reversed(requested_clusters):
            if cluster_id in self._cluster_to_slots:
                self._touch_cluster(cluster_id)

        if not lookup_after_admit:
            ready_event = self._pending_event_for_clusters(copied_cluster_ids)
            empty_page_ids = torch.empty(0, dtype=torch.int64, device=self.device)
            empty_mask = torch.empty(0, dtype=torch.bool, device=self.device)
            return RetroSpecResidentPageAccess(
                cache_page_ids=empty_page_ids,
                hit_cluster_mask=empty_mask,
                miss_cluster_mask=empty_mask,
                hit_gate_ready_mask=empty_mask,
                logical_page_ids_cpu=None,
                miss_cluster_mask_cpu=None,
                ready_event=ready_event,
            )

        return self.lookup(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            touch=False,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )

    def admit(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        backing_key_pages: torch.Tensor,
        backing_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from stable CPU backing pages."""
        self._validate_backing_pages(
            backing_key_pages,
            backing_value_pages,
        )

        return self._admit_from_sources(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=page_ids if page_ids_cpu is None else page_ids_cpu,
            source_key_pages=backing_key_pages,
            source_value_pages=backing_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
            reuse_ready_event=reuse_ready_event,
            mutation_stream=mutation_stream,
            lookup_after_admit=lookup_after_admit,
        )

    def admit_staged(
        self,
        cluster_ids: torch.Tensor,
        page_ids: torch.Tensor,
        cluster_groups: Mapping[_ClusterId, RetroSpecClusterGroup],
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        staging_page_ids: torch.Tensor,
        staging_key_pages: torch.Tensor,
        staging_value_pages: torch.Tensor,
        cluster_ids_cpu: torch.Tensor | None = None,
        page_ids_cpu: torch.Tensor | None = None,
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
    ) -> RetroSpecResidentPageAccess:
        """Admit a priority cluster prefix from bounded CPU/GPU staging pages."""
        self._validate_source_pages(
            staging_key_pages,
            staging_value_pages,
        )
        if (
            staging_key_pages.device.type != "cpu"
            and staging_key_pages.device != self.device
        ):
            raise ValueError("Staging pages must use CPU or the resident CUDA device")

        return self._admit_from_sources(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=staging_page_ids,
            source_key_pages=staging_key_pages,
            source_value_pages=staging_value_pages,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
            reuse_ready_event=reuse_ready_event,
            mutation_stream=mutation_stream,
            lookup_after_admit=lookup_after_admit,
        )

    def admit_prepared_staged(
        self,
        prepared: RetroSpecPreparedResidentAdmission,
        allocated_cluster_ids: Collection[int],
        allocated_page_ids: Collection[int],
        reuse_ready_event: torch.cuda.Event | None = None,
        mutation_stream: torch.cuda.Stream | None = None,
        lookup_after_admit: bool = True,
        resolved_lru: RetroSpecResolvedResidentLru | None = None,
    ) -> RetroSpecResidentPageAccess:
        """Commit a previously prepared admission under mutation_guard()."""
        return self._admit_from_sources(
            cluster_ids=prepared.cluster_ids,
            page_ids=prepared.page_ids,
            cluster_groups=prepared.cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            source_page_ids=prepared.source_page_ids,
            source_key_pages=prepared.source_key_pages,
            source_value_pages=prepared.source_value_pages,
            cluster_ids_cpu=prepared.cluster_ids_cpu,
            page_ids_cpu=prepared.page_ids_cpu,
            reuse_ready_event=reuse_ready_event,
            mutation_stream=mutation_stream,
            lookup_after_admit=lookup_after_admit,
            prepared=prepared,
            resolved_lru=resolved_lru,
        )

    def invalidate(
        self,
        cluster_ids: torch.Tensor,
    ) -> None:
        """Evict released cluster IDs before their backing pages are reused."""
        self.synchronize_pending_copies()
        cluster_ids_cpu = cluster_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        released_cluster_ids = tuple(
            sorted(set(cluster_ids_cpu[cluster_ids_cpu >= 0].tolist()))
        )
        resident_cluster_ids = tuple(
            cluster_id
            for cluster_id in released_cluster_ids
            if cluster_id in self._cluster_to_slots
        )
        affected_groups = {
            self._cluster_to_group[cluster_id] for cluster_id in resident_cluster_ids
        }
        previous_gate_ready = self._snapshot_group_hit_gate_ready(affected_groups)
        current_stream = torch.cuda.current_stream(self.device)
        self._evict_clusters(resident_cluster_ids, update_stream=current_stream)
        self._publish_handle_delta((), previous_gate_ready, current_stream)
