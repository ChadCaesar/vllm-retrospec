# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Mapping
from threading import Lock

import torch

from vllm.v1.spec_decode.retrospec.legacy.cluster_identity import RetroSpecClusterGroup
from vllm.v1.spec_decode.retrospec.legacy.residency.cache.types import (
    _PREFETCH_ABSENT,
    _PREFETCH_PENDING,
    _PREFETCH_RESIDENT,
    _ClusterId,
    _resident_handle_hash,
)
from vllm.v1.spec_decode.retrospec.legacy.resident_kernels import (
    update_resident_handles,
)


class _RetroSpecResidentClusterCacheHandlesMixin:
    def requires_resize(
        self,
        capacity: int,
        group_targets: Mapping[RetroSpecClusterGroup, int] | None = None,
    ) -> bool:
        targets = self._group_targets if group_targets is None else group_targets
        return capacity != self._logical_capacity or targets != self._group_targets

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 if value <= 1 else 1 << (value - 1).bit_length()

    def mutation_guard(self) -> Lock:
        """Return the short guard shared by GPU readers and cache mutations."""
        return self._gpu_access_lock

    def _bump_structure_revision(self) -> None:
        self._structure_revision += 1

    def partition_admission_candidates(
        self,
        cluster_ids: Collection[_ClusterId],
    ) -> tuple[tuple[_ClusterId, ...], int, int]:
        """Separate absent clusters from resident and pending clusters.

        The caller must hold ``mutation_guard()``. Input priority order is
        preserved in the returned admission candidates.
        """
        self._reap_completed_copy_batches()

        admission_candidates: list[_ClusterId] = []
        resident_count = 0
        pending_count = 0
        for cluster_id in cluster_ids:
            if cluster_id not in self._cluster_to_slots:
                admission_candidates.append(cluster_id)
            elif cluster_id in self._pending_cluster_events:
                pending_count += 1
            else:
                resident_count += 1

        return tuple(admission_candidates), resident_count, pending_count

    def reserve_prefetch_handle_states(self, required_capacity: int) -> None:
        if required_capacity < 0:
            raise ValueError("Prefetch handle-state capacity must be non-negative")
        if required_capacity <= self._prefetch_handle_states.numel():
            return

        capacity = self._next_power_of_two(max(required_capacity, 1))
        # This shadow is updated by the background worker after CUDA events
        # complete, which can happen outside the model runner's inference-mode
        # scope. Keep its storage as a normal tensor even when the cache is
        # first initialized from an inference-mode prefill call.
        with torch.inference_mode(False):
            states = torch.zeros(capacity, dtype=torch.uint8, device="cpu")
            states[: self._prefetch_handle_states.numel()].copy_(
                self._prefetch_handle_states
            )
        self._prefetch_handle_states = states

    def _set_prefetch_handle_state(
        self, cluster_ids: Collection[_ClusterId], state: int
    ) -> None:
        cluster_ids = tuple(dict.fromkeys(cluster_ids))
        if not cluster_ids:
            return
        if state not in (
            _PREFETCH_ABSENT,
            _PREFETCH_PENDING,
            _PREFETCH_RESIDENT,
        ):
            raise ValueError("Invalid resident prefetch state")

        self.reserve_prefetch_handle_states(max(cluster_ids) + 1)
        indices = torch.tensor(cluster_ids, dtype=torch.int64, device="cpu")
        self._prefetch_handle_states.index_fill_(0, indices, state)

    def prefetch_handle_states(self, required_capacity: int) -> torch.Tensor:
        """Return the CPU planner shadow after reaping completed H2D copies."""
        self._reap_completed_copy_batches()
        self.reserve_prefetch_handle_states(required_capacity)
        return self._prefetch_handle_states[:required_capacity]

    def _find_handle_bucket(self, cluster_id: _ClusterId) -> int | None:
        if self._handle_table_capacity == 0:
            return None

        mask = self._handle_table_capacity - 1
        first_tombstone: int | None = None
        first_bucket = _resident_handle_hash(cluster_id) & mask
        for probe in range(64):
            bucket = (first_bucket + probe) & mask
            stored_handle = self._bucket_handles[bucket]
            if stored_handle == cluster_id:
                return bucket
            if stored_handle == -2 and first_tombstone is None:
                first_tombstone = bucket
            if stored_handle == -1:
                return bucket if first_tombstone is None else first_tombstone
        return first_tombstone

    def _allocate_handle_table(
        self,
        capacity: int,
        max_pages_per_cluster: int,
    ) -> None:
        self._handle_table_capacity = capacity
        self._handle_table_max_pages = max_pages_per_cluster
        self._handle_table_handles = torch.full(
            (capacity,), -1, dtype=torch.int64, device=self.device
        )
        self._handle_table_versions = torch.zeros(
            capacity, dtype=torch.int32, device=self.device
        )
        self._handle_table_page_counts = torch.zeros(
            capacity, dtype=torch.int32, device=self.device
        )
        self._handle_table_page_slots = torch.full(
            (capacity, max_pages_per_cluster),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._handle_table_hit_gate_ready = torch.zeros(
            capacity, dtype=torch.bool, device=self.device
        )
        self._handle_table_last_access_epochs = torch.zeros(
            capacity, dtype=torch.int64, device=self.device
        )
        self._handle_to_bucket.clear()
        self._bucket_handles = [-1] * capacity
        self._handle_table_needs_rebuild = False
        self._bump_structure_revision()

    def _ensure_handle_table(
        self,
        max_pages_per_cluster: int,
        stream: torch.cuda.Stream | None = None,
    ) -> bool:
        required_capacity = self._next_power_of_two(max(2, self._logical_capacity * 2))
        requires_rebuild = (
            self._handle_table_needs_rebuild
            or required_capacity > self._handle_table_capacity
            or max_pages_per_cluster > self._handle_table_max_pages
        )
        if not requires_rebuild:
            return False

        previous_max_pages = self._handle_table_max_pages
        # A pending publication targets the current table allocation. Complete
        # it before replacing that allocation, then republish every live entry.
        self.synchronize_pending_copies()
        update_stream = (
            torch.cuda.current_stream(self.device) if stream is None else stream
        )
        if self._handle_table_capacity and self._group_states:
            self._refresh_group_lru_from_gpu(self._group_states.keys(), update_stream)
        with torch.cuda.stream(update_stream):
            self._allocate_handle_table(
                max(required_capacity, self._handle_table_capacity),
                max(max_pages_per_cluster, self._handle_table_max_pages),
            )
        entries = tuple(
            (
                cluster_id,
                self._cluster_to_slots[cluster_id],
                self._cluster_to_group[cluster_id],
            )
            for cluster_id in self._cluster_to_slots
        )
        published = self._write_handle_entries(entries, update_stream)

        # Readers immediately switch to the new tensor references after this
        # method returns. Complete this rare rebuild before releasing the
        # mutation guard so they cannot observe an uninitialized table.
        update_stream.synchronize()
        if published != len(entries):
            raise RuntimeError(
                "Resident handle table rebuild did not publish every live cluster"
            )

        stats = self.performance_stats
        if stats is not None and stats.enabled:
            stats.add_counter("resident_handle_table_rebuilds")
            if self._handle_table_max_pages > previous_max_pages:
                stats.add_counter("resident_handle_table_width_growths")
            stats.observe_peak(
                "resident_handle_table_max_pages",
                self._handle_table_max_pages,
            )
        return True

    def _publish_handle_entries(
        self,
        entries: Collection[tuple[_ClusterId, tuple[int, ...], RetroSpecClusterGroup]],
        stream: torch.cuda.Stream,
    ) -> int:
        entries = tuple(entries)
        if not entries:
            return 0

        max_pages_per_cluster = max(len(slots) for _, slots, _ in entries)
        if self._ensure_handle_table(max_pages_per_cluster, stream):
            # A rebuild republishes every live cluster, including this delta.
            return len(entries)

        return self._write_handle_entries(entries, stream)

    def _write_handle_entries(
        self,
        entries: Collection[tuple[_ClusterId, tuple[int, ...], RetroSpecClusterGroup]],
        stream: torch.cuda.Stream,
    ) -> int:
        entries = tuple(entries)
        if not entries or self._handle_table_capacity == 0:
            return 0

        bucket_ids: list[int] = []
        cluster_ids: list[int] = []
        page_counts: list[int] = []
        page_slots: list[list[int]] = []
        hit_gate_ready: list[bool] = []
        new_entry_indices: list[int] = []

        for cluster_id, slots, group in entries:
            if len(slots) > self._handle_table_max_pages:
                raise RuntimeError(
                    f"Resident cluster {cluster_id} has {len(slots)} pages, "
                    f"but the handle table width is "
                    f"{self._handle_table_max_pages}"
                )

            bucket = self._handle_to_bucket.get(cluster_id)
            if bucket is None:
                bucket = self._find_handle_bucket(cluster_id)
                if bucket is None:
                    self._handle_table_needs_rebuild = True
                    continue
                self._handle_to_bucket[cluster_id] = bucket
                self._bucket_handles[bucket] = cluster_id
                new_entry_indices.append(len(bucket_ids))

            padded_slots = list(slots)
            padded_slots.extend(
                [-1] * (self._handle_table_max_pages - len(padded_slots))
            )
            bucket_ids.append(bucket)
            cluster_ids.append(cluster_id)
            page_counts.append(len(slots))
            page_slots.append(padded_slots)
            hit_gate_ready.append(self._is_group_hit_gate_ready(group))

        if not cluster_ids:
            return 0

        with torch.cuda.stream(stream):
            bucket_ids_gpu = torch.tensor(
                bucket_ids, dtype=torch.int32, device=self.device
            )
            update_resident_handles(
                bucket_ids=bucket_ids_gpu,
                cluster_handles=torch.tensor(
                    cluster_ids, dtype=torch.int64, device=self.device
                ),
                page_counts=torch.tensor(
                    page_counts, dtype=torch.int32, device=self.device
                ),
                page_slots=torch.tensor(
                    page_slots, dtype=torch.int32, device=self.device
                ),
                hit_gate_ready=torch.tensor(
                    hit_gate_ready, dtype=torch.bool, device=self.device
                ),
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
            )
            if new_entry_indices:
                new_bucket_ids = bucket_ids_gpu.index_select(
                    0,
                    torch.tensor(
                        new_entry_indices,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )
                self._handle_table_last_access_epochs.index_fill_(
                    0, new_bucket_ids.to(torch.int64), self._next_access_epoch
                )
            if self._binding_publisher is not None:
                self._binding_publisher(
                    tuple(cluster_ids),
                    tuple(bucket_ids),
                    stream,
                )

        return len(cluster_ids)

    def _erase_handle_entries(
        self,
        cluster_ids: Collection[_ClusterId],
        stream: torch.cuda.Stream,
    ) -> None:
        if self._handle_table_capacity == 0:
            return

        bucket_ids: list[int] = []
        erased_cluster_ids: list[int] = []
        for cluster_id in cluster_ids:
            bucket = self._handle_to_bucket.pop(cluster_id, None)
            if bucket is None:
                continue
            self._bucket_handles[bucket] = -2
            erased_cluster_ids.append(cluster_id)
            bucket_ids.append(bucket)

        if not bucket_ids:
            return

        num_entries = len(bucket_ids)
        with torch.cuda.stream(stream):
            bucket_ids_gpu = torch.tensor(
                bucket_ids, dtype=torch.int32, device=self.device
            )
            update_resident_handles(
                bucket_ids=bucket_ids_gpu,
                cluster_handles=torch.full(
                    (num_entries,), -2, dtype=torch.int64, device=self.device
                ),
                page_counts=torch.zeros(
                    num_entries, dtype=torch.int32, device=self.device
                ),
                page_slots=torch.full(
                    (num_entries, self._handle_table_max_pages),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                hit_gate_ready=torch.zeros(
                    num_entries, dtype=torch.bool, device=self.device
                ),
                table_handles=self._handle_table_handles,
                table_versions=self._handle_table_versions,
                table_page_counts=self._handle_table_page_counts,
                table_page_slots=self._handle_table_page_slots,
                table_hit_gate_ready=self._handle_table_hit_gate_ready,
            )
            self._handle_table_last_access_epochs.index_fill_(
                0, bucket_ids_gpu.to(torch.int64), 0
            )
            if self._binding_publisher is not None:
                self._binding_publisher(
                    tuple(erased_cluster_ids),
                    (-1,) * len(erased_cluster_ids),
                    stream,
                )

    def republish_handle_bindings(
        self,
        cluster_ids: Collection[_ClusterId],
        stream: torch.cuda.Stream,
    ) -> None:
        """Replay live buckets after a matching index segment is published."""
        if self._binding_publisher is None:
            return

        published_cluster_ids: list[int] = []
        published_buckets: list[int] = []
        seen: set[int] = set()
        for cluster_id in cluster_ids:
            if cluster_id < 0 or cluster_id in seen:
                continue
            seen.add(cluster_id)
            bucket = self._handle_to_bucket.get(cluster_id)
            if bucket is None:
                continue
            published_cluster_ids.append(cluster_id)
            published_buckets.append(bucket)

        if published_cluster_ids:
            self._binding_publisher(
                tuple(published_cluster_ids),
                tuple(published_buckets),
                stream,
            )

    def _snapshot_group_hit_gate_ready(
        self, groups: Collection[RetroSpecClusterGroup]
    ) -> dict[RetroSpecClusterGroup, bool]:
        return {group: self._is_group_hit_gate_ready(group) for group in set(groups)}

    def _publish_handle_delta(
        self,
        new_cluster_ids: Collection[_ClusterId],
        previous_gate_ready: Mapping[RetroSpecClusterGroup, bool],
        stream: torch.cuda.Stream,
    ) -> int:
        entries: list[tuple[_ClusterId, tuple[int, ...], RetroSpecClusterGroup]] = []
        seen: set[_ClusterId] = set()

        for cluster_id in new_cluster_ids:
            slots = self._cluster_to_slots.get(cluster_id)
            group = self._cluster_to_group.get(cluster_id)
            if slots is None or group is None or cluster_id in seen:
                continue
            entries.append((cluster_id, slots, group))
            seen.add(cluster_id)

        ordered_groups = sorted(
            previous_gate_ready,
            key=lambda group: (group.request_id, group.kv_head_index),
        )
        for group in ordered_groups:
            if previous_gate_ready[group] == self._is_group_hit_gate_ready(group):
                continue
            group_state = self._group_states.get(group)
            if group_state is None:
                continue
            for cluster_id in group_state.lru:
                if cluster_id in seen:
                    continue
                entries.append((cluster_id, self._cluster_to_slots[cluster_id], group))
                seen.add(cluster_id)

        published = self._publish_handle_entries(entries, stream)
        stats = self.performance_stats
        if (
            stats is not None
            and stats.enabled
            and stats.cuda_timing_level == "detailed"
        ):
            stats.add_counter("resident_handle_entries_published", published)
        return published
