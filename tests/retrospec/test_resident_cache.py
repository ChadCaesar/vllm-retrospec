# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.retrospec.cluster_identity import RetroSpecClusterGroup
from vllm.v1.spec_decode.retrospec.resident_cache import (
    RetroSpecResidentClusterCache,
    _PendingCopyBatch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the resident cluster cache",
)


def _cluster_ids_from_pages(page_ids: torch.Tensor) -> torch.Tensor:
    cluster_ids = torch.full(
        page_ids.shape[:-1],
        -1,
        dtype=torch.int64,
        device=page_ids.device,
    )
    for page_index in range(page_ids.shape[-1]):
        candidate = page_ids[..., page_index]
        cluster_ids = torch.where(
            (cluster_ids < 0) & (candidate >= 0), candidate, cluster_ids
        )
    return cluster_ids


class _ResidentCacheTestAdapter(RetroSpecResidentClusterCache):
    """Keep legacy test cases concise while supplying stable cluster IDs."""

    _DEFAULT_GROUP = RetroSpecClusterGroup("request", 0)

    @staticmethod
    def _allocated_cluster_ids(
        cluster_ids: torch.Tensor, allocated_page_ids: set[int]
    ) -> set[int]:
        valid_ids = cluster_ids[cluster_ids >= 0].detach().cpu().tolist()
        return set(allocated_page_ids).union(valid_ids)

    @classmethod
    def _cluster_groups(
        cls, cluster_ids: torch.Tensor
    ) -> dict[int, RetroSpecClusterGroup]:
        return {
            cluster_id: cls._DEFAULT_GROUP
            for cluster_id in cluster_ids[cluster_ids >= 0].detach().cpu().tolist()
        }

    def lookup(
        self,
        page_ids,
        allocated_page_ids,
        touch=True,
        include_pending=True,
        *,
        cluster_ids=None,
        cluster_groups=None,
        allocated_cluster_ids=None,
        cluster_ids_cpu=None,
        page_ids_cpu=None,
    ):
        if cluster_ids is None:
            cluster_ids = _cluster_ids_from_pages(page_ids)
        if cluster_groups is None:
            cluster_groups = self._cluster_groups(cluster_ids)
        if allocated_cluster_ids is None:
            allocated_cluster_ids = self._allocated_cluster_ids(
                cluster_ids, allocated_page_ids
            )
        return super().lookup(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            touch=touch,
            include_pending=include_pending,
            cluster_ids_cpu=cluster_ids_cpu,
            page_ids_cpu=page_ids_cpu,
        )

    def admit(
        self,
        page_ids,
        allocated_page_ids,
        backing_key_pages,
        backing_value_pages,
        *,
        cluster_ids=None,
        cluster_groups=None,
        allocated_cluster_ids=None,
    ):
        if cluster_ids is None:
            cluster_ids = _cluster_ids_from_pages(page_ids)
        if cluster_groups is None:
            cluster_groups = self._cluster_groups(cluster_ids)
        if allocated_cluster_ids is None:
            allocated_cluster_ids = self._allocated_cluster_ids(
                cluster_ids, allocated_page_ids
            )
        return super().admit(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            backing_key_pages=backing_key_pages,
            backing_value_pages=backing_value_pages,
        )

    def admit_staged(
        self,
        page_ids,
        allocated_page_ids,
        staging_page_ids,
        staging_key_pages,
        staging_value_pages,
        *,
        cluster_ids=None,
        cluster_groups=None,
        allocated_cluster_ids=None,
    ):
        if cluster_ids is None:
            cluster_ids = _cluster_ids_from_pages(page_ids)
        if cluster_groups is None:
            cluster_groups = self._cluster_groups(cluster_ids)
        if allocated_cluster_ids is None:
            allocated_cluster_ids = self._allocated_cluster_ids(
                cluster_ids, allocated_page_ids
            )
        return super().admit_staged(
            cluster_ids=cluster_ids,
            page_ids=page_ids,
            cluster_groups=cluster_groups,
            allocated_cluster_ids=allocated_cluster_ids,
            allocated_page_ids=allocated_page_ids,
            staging_page_ids=staging_page_ids,
            staging_key_pages=staging_key_pages,
            staging_value_pages=staging_value_pages,
        )


def make_cache(
    capacity: int,
    *,
    group_targets: dict[RetroSpecClusterGroup, int] | None = None,
) -> RetroSpecResidentClusterCache:
    cache = _ResidentCacheTestAdapter(
        page_size=2,
        head_size=1,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    cache.resize(capacity, group_targets=group_targets)
    return cache


def make_backing_pages(num_pages: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.arange(num_pages * 2, dtype=torch.float32).view(num_pages, 2, 1)
    return keys, keys + 100.0


def assert_cached_pages_match_backing(
    cache: RetroSpecResidentClusterCache,
    logical_page_ids: torch.Tensor,
    cache_page_ids: torch.Tensor,
    backing_keys: torch.Tensor,
    backing_values: torch.Tensor,
) -> None:
    cache.wait_for_pending_copies()
    valid = logical_page_ids >= 0
    logical_ids = logical_page_ids[valid].to(torch.int64)
    slot_ids = cache_page_ids[valid].to(device="cuda", dtype=torch.int64)

    torch.testing.assert_close(
        cache.key_pages.index_select(0, slot_ids).cpu(),
        backing_keys.index_select(0, logical_ids),
    )
    torch.testing.assert_close(
        cache.value_pages.index_select(0, slot_ids).cpu(),
        backing_values.index_select(0, logical_ids),
    )


def test_resident_cache_records_copy_batch_and_retains_sources():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    access = cache.admit(
        page_ids,
        {0, 1},
        backing_keys,
        backing_values,
    )

    assert len(cache._pending_copy_batches) == 1
    pending = cache._pending_copy_batches[0]
    assert pending.source_key_pages is backing_keys
    assert pending.source_value_pages is backing_values

    cache.synchronize_pending_copies()
    assert cache.num_pending_copy_batches == 0
    assert_cached_pages_match_backing(
        cache,
        page_ids,
        access.cache_page_ids,
        backing_keys,
        backing_values,
    )


def test_resident_cache_returns_latest_pending_copy_event():
    cache = make_cache(capacity=4)
    backing_keys, backing_values = make_backing_pages()

    # Prevent eager reaping so the assertion is deterministic even when the
    # tiny test copies complete before the host asks for the event.
    cache._reap_completed_copy_batches = Mock()

    cache.admit(
        torch.tensor([[0, 1]], dtype=torch.int64),
        set(range(4)),
        backing_keys,
        backing_values,
    )
    first_event = cache._pending_copy_batches[-1].ready_event

    cache.admit(
        torch.tensor([[2, 3]], dtype=torch.int64),
        set(range(4)),
        backing_keys,
        backing_values,
    )
    latest_event = cache._pending_copy_batches[-1].ready_event

    assert latest_event is not first_event
    assert cache.pending_copy_event() is latest_event
    latest_event.synchronize()


def test_pending_resident_cluster_can_be_hidden_from_draft_lookup():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    # Keep the tiny copy pending from the host's point of view so both lookup
    # modes can be checked deterministically.
    cache._reap_completed_copy_batches = Mock()
    admitted = cache.admit(
        page_ids,
        {0, 1},
        backing_keys,
        backing_values,
    )

    assert admitted.hit_cluster_mask.item()
    assert cache._pending_cluster_events.keys() == {0}

    verification = cache.lookup(
        page_ids,
        {0, 1},
        include_pending=True,
    )
    draft = cache.lookup(
        page_ids,
        {0, 1},
        include_pending=False,
    )

    assert verification.hit_cluster_mask.item()
    assert not verification.miss_cluster_mask.item()
    assert not draft.hit_cluster_mask.item()
    assert draft.miss_cluster_mask.item()
    assert torch.all(draft.cache_page_ids == -1)

    cache.synchronize_pending_copies()
    assert not cache._pending_cluster_events


def test_completed_old_batch_does_not_clear_newer_pending_cluster_event():
    cache = make_cache(capacity=1)
    source_keys, source_values = make_backing_pages(num_pages=1)
    first_event = Mock()
    second_event = Mock()
    first_event.query.return_value = True
    second_event.query.return_value = False

    cache._pending_copy_batches.extend(
        (
            _PendingCopyBatch(
                ready_event=first_event,
                cluster_ids=(7,),
                source_key_pages=source_keys,
                source_value_pages=source_values,
            ),
            _PendingCopyBatch(
                ready_event=second_event,
                cluster_ids=(7,),
                source_key_pages=source_keys,
                source_value_pages=source_values,
            ),
        )
    )
    cache._pending_cluster_events[7] = second_event

    cache._reap_completed_copy_batches()

    assert len(cache._pending_copy_batches) == 1
    assert cache._pending_copy_batches[0].ready_event is second_event
    assert cache._pending_cluster_events[7] is second_event


def test_resident_cache_waits_on_explicit_consumer_stream():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    access = cache.admit(
        page_ids,
        {0, 1},
        backing_keys,
        backing_values,
    )

    consumer_stream = torch.cuda.Stream()
    cache.wait_for_pending_copies(consumer_stream)
    with torch.cuda.stream(consumer_stream):
        slot_ids = access.cache_page_ids[0].to(
            device="cuda",
            dtype=torch.int64,
        )
        copied_keys = cache.key_pages.index_select(0, slot_ids).clone()
        copied_values = cache.value_pages.index_select(0, slot_ids).clone()
    consumer_stream.synchronize()

    torch.testing.assert_close(copied_keys.cpu(), backing_keys[:2])
    torch.testing.assert_close(copied_values.cpu(), backing_values[:2])


def test_resident_cache_growth_preserves_inflight_admission():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    access = cache.admit(
        page_ids,
        {0, 1},
        backing_keys,
        backing_values,
    )

    cache.resize(4)

    assert cache.capacity == 4
    assert cache.physical_capacity == 4
    assert_cached_pages_match_backing(
        cache,
        page_ids,
        access.cache_page_ids,
        backing_keys,
        backing_values,
    )


def test_resident_cache_admits_cluster_atomic_priority_prefix():
    cache = make_cache(capacity=3)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor(
        [
            [0, 1, -1],
            [2, -1, -1],
            [3, 4, -1],
            [-1, -1, -1],
        ],
        dtype=torch.int64,
    )

    before = cache.lookup(page_ids, set(range(5)))
    assert before.hit_cluster_mask.tolist() == [False, False, False, False]
    assert before.miss_cluster_mask.tolist() == [True, True, True, False]
    assert torch.all(before.cache_page_ids == -1)

    admitted = cache.admit(
        page_ids,
        set(range(5)),
        backing_keys,
        backing_values,
    )

    assert admitted.hit_cluster_mask.tolist() == [True, True, False, False]
    assert admitted.miss_cluster_mask.tolist() == [False, False, True, False]
    assert cache.num_resident_pages == 3
    assert cache.num_resident_clusters == 2
    assert torch.all(admitted.cache_page_ids[2:] == -1)
    assert_cached_pages_match_backing(
        cache,
        page_ids[:2],
        admitted.cache_page_ids[:2],
        backing_keys,
        backing_values,
    )


@pytest.mark.parametrize(
    ("touch_second_cluster", "expected_hit_mask"),
    [
        (False, [True, False, True]),
        (True, [False, True, True]),
    ],
)
def test_resident_cache_lookup_controls_lru_order(
    touch_second_cluster: bool,
    expected_hit_mask: list[bool],
):
    cache = make_cache(capacity=4)
    backing_keys, backing_values = make_backing_pages()
    first_two = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    cache.admit(first_two, set(range(6)), backing_keys, backing_values)

    cache.lookup(first_two[1:], set(range(6)), touch=touch_second_cluster)
    cache.admit(
        torch.tensor([[4, 5]], dtype=torch.int64),
        set(range(6)),
        backing_keys,
        backing_values,
    )

    all_clusters = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
    access = cache.lookup(all_clusters, set(range(6)), touch=False)
    assert access.hit_cluster_mask.tolist() == expected_hit_mask
    assert access.miss_cluster_mask.tolist() == [not hit for hit in expected_hit_mask]


def test_resident_cache_interleaves_priority_across_groups():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()

    # Each row is one request/head group and the middle dimension is retrieval
    # rank. Rank-major admission gives both groups their rank-zero cluster.
    page_ids = torch.tensor(
        [
            [[0], [2]],
            [[1], [3]],
        ],
        dtype=torch.int64,
    )
    access = cache.admit(
        page_ids,
        set(range(4)),
        backing_keys,
        backing_values,
    )

    assert access.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert access.miss_cluster_mask.tolist() == [
        [False, True],
        [False, True],
    ]
    resident_logical_page_ids = page_ids.masked_fill(
        access.miss_cluster_mask.unsqueeze(-1),
        -1,
    )
    assert_cached_pages_match_backing(
        cache,
        resident_logical_page_ids,
        access.cache_page_ids,
        backing_keys,
        backing_values,
    )


def test_resident_cache_tracks_group_scoped_lru_in_shared_arena():
    cache = make_cache(capacity=4)
    backing_keys, backing_values = make_backing_pages()
    cluster_ids = torch.tensor([10, 11, 12], dtype=torch.int64)
    page_ids = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 0)
    cluster_groups = {
        10: first_group,
        11: first_group,
        12: second_group,
    }

    cache.admit(
        page_ids,
        set(range(4)),
        backing_keys,
        backing_values,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
    )

    assert cache.num_resident_groups == 2
    assert cache._group_states[first_group].num_pages == 2
    assert cache._group_states[second_group].num_pages == 1
    assert list(cache._group_states[first_group].lru) == [11, 10]
    assert list(cache._group_states[second_group].lru) == [12]
    assert len(set(cache._cluster_to_slots.values())) == 3

    cache.lookup(
        page_ids[1:2],
        set(range(4)),
        cluster_ids=cluster_ids[1:2],
        cluster_groups={11: first_group},
    )

    assert list(cache._group_states[first_group].lru) == [10, 11]
    assert list(cache._group_states[second_group].lru) == [12]


def test_hit_gate_becomes_ready_at_each_group_soft_target():
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 0)
    cache = make_cache(
        capacity=4,
        group_targets={first_group: 2, second_group: 2},
    )
    backing_keys, backing_values = make_backing_pages(num_pages=4)
    cluster_ids = torch.tensor([10, 11, 20, 21], dtype=torch.int64)
    page_ids = torch.arange(4, dtype=torch.int64).unsqueeze(-1)
    cluster_groups = {
        10: first_group,
        11: first_group,
        20: second_group,
        21: second_group,
    }

    cache.admit(
        page_ids[[0, 2, 3]],
        set(range(4)),
        backing_keys,
        backing_values,
        cluster_ids=cluster_ids[[0, 2, 3]],
        cluster_groups={10: first_group, 20: second_group, 21: second_group},
        allocated_cluster_ids=set(cluster_groups),
    )

    access = cache.lookup(
        page_ids,
        set(range(4)),
        touch=False,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
        allocated_cluster_ids=set(cluster_groups),
    )
    assert access.hit_gate_ready_mask.tolist() == [False, False, True, True]

    cache.admit(
        page_ids[1:2],
        set(range(4)),
        backing_keys,
        backing_values,
        cluster_ids=cluster_ids[1:2],
        cluster_groups={11: first_group},
        allocated_cluster_ids=set(cluster_groups),
    )
    access = cache.lookup(
        page_ids,
        set(range(4)),
        touch=False,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
        allocated_cluster_ids=set(cluster_groups),
    )
    assert access.hit_gate_ready_mask.all()

    cache.resize(5, group_targets={first_group: 3, second_group: 2})
    access = cache.lookup(
        page_ids,
        set(range(4)),
        touch=False,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
        allocated_cluster_ids=set(cluster_groups),
    )
    assert access.hit_gate_ready_mask.tolist() == [False, False, True, True]


def test_resident_cache_evicts_from_group_above_its_soft_target():
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 0)
    cache = make_cache(
        capacity=4,
        group_targets={first_group: 1, second_group: 3},
    )
    backing_keys, backing_values = make_backing_pages()
    initial_cluster_ids = torch.tensor([10, 11, 12, 20], dtype=torch.int64)
    initial_page_ids = torch.tensor([[0], [1], [2], [3]], dtype=torch.int64)
    cluster_groups = {
        10: first_group,
        11: first_group,
        12: first_group,
        20: second_group,
    }
    cache.admit(
        initial_page_ids,
        set(range(5)),
        backing_keys,
        backing_values,
        cluster_ids=initial_cluster_ids,
        cluster_groups=cluster_groups,
        allocated_cluster_ids=set(cluster_groups),
    )

    cache.admit(
        torch.tensor([[4]], dtype=torch.int64),
        set(range(5)),
        backing_keys,
        backing_values,
        cluster_ids=torch.tensor([21], dtype=torch.int64),
        cluster_groups={21: second_group},
        allocated_cluster_ids={*cluster_groups, 21},
    )

    all_cluster_ids = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64)
    all_page_ids = torch.arange(5, dtype=torch.int64).unsqueeze(-1)
    all_cluster_groups = {**cluster_groups, 21: second_group}
    access = cache.lookup(
        all_page_ids,
        set(range(5)),
        touch=False,
        cluster_ids=all_cluster_ids,
        cluster_groups=all_cluster_groups,
        allocated_cluster_ids=set(all_cluster_groups),
    )

    assert access.hit_cluster_mask.tolist() == [True, True, False, True, True]
    assert cache._group_states[first_group].num_pages == 2
    assert cache._group_states[second_group].num_pages == 2


def test_resident_cache_falls_back_when_target_group_is_protected():
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 0)
    cache = make_cache(
        capacity=2,
        group_targets={first_group: 1, second_group: 1},
    )
    backing_keys, backing_values = make_backing_pages(num_pages=3)
    cluster_groups = {10: first_group, 11: first_group, 20: second_group}
    cache.admit(
        torch.tensor([[0], [1]], dtype=torch.int64),
        set(range(3)),
        backing_keys,
        backing_values,
        cluster_ids=torch.tensor([10, 20], dtype=torch.int64),
        cluster_groups={10: first_group, 20: second_group},
        allocated_cluster_ids=set(cluster_groups),
    )

    cache.admit(
        torch.tensor([[0], [2]], dtype=torch.int64),
        set(range(3)),
        backing_keys,
        backing_values,
        cluster_ids=torch.tensor([10, 11], dtype=torch.int64),
        cluster_groups={10: first_group, 11: first_group},
        allocated_cluster_ids=set(cluster_groups),
    )

    access = cache.lookup(
        torch.tensor([[0], [2], [1]], dtype=torch.int64),
        set(range(3)),
        touch=False,
        cluster_ids=torch.tensor([10, 11, 20], dtype=torch.int64),
        cluster_groups=cluster_groups,
        allocated_cluster_ids=set(cluster_groups),
    )
    assert access.hit_cluster_mask.tolist() == [True, True, False]
    assert cache._group_states[first_group].num_pages == 2
    assert second_group not in cache._group_states


def test_resident_cache_resize_uses_group_targets_and_local_lru():
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 0)
    cache = make_cache(
        capacity=4,
        group_targets={first_group: 1, second_group: 3},
    )
    backing_keys, backing_values = make_backing_pages(num_pages=4)
    cluster_ids = torch.tensor([10, 11, 12, 20], dtype=torch.int64)
    page_ids = torch.arange(4, dtype=torch.int64).unsqueeze(-1)
    cluster_groups = {
        10: first_group,
        11: first_group,
        12: first_group,
        20: second_group,
    }
    cache.admit(
        page_ids,
        set(range(4)),
        backing_keys,
        backing_values,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
        allocated_cluster_ids=set(cluster_groups),
    )

    cache.resize(2, group_targets={first_group: 1, second_group: 1})

    access = cache.lookup(
        page_ids,
        set(range(4)),
        touch=False,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
        allocated_cluster_ids=set(cluster_groups),
    )
    assert access.hit_cluster_mask.tolist() == [True, False, False, True]
    assert cache._group_states[first_group].num_pages == 1
    assert cache._group_states[second_group].num_pages == 1
    assert cache.physical_capacity == 4


def test_resident_cache_rejects_invalid_group_targets():
    group = RetroSpecClusterGroup("request", 0)
    cache = make_cache(capacity=2)

    with pytest.raises(ValueError, match="must be non-negative"):
        cache.resize(2, group_targets={group: -1})

    with pytest.raises(ValueError, match="cannot exceed layer capacity"):
        cache.resize(2, group_targets={group: 3})

    assert cache.capacity == 2
    assert cache._group_targets == {}


def test_resident_cache_invalidates_clusters_and_removes_empty_groups():
    cache = make_cache(capacity=4)
    backing_keys, backing_values = make_backing_pages()
    cluster_ids = torch.tensor([10, 11, 12], dtype=torch.int64)
    page_ids = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 0)
    cluster_groups = {
        10: first_group,
        11: first_group,
        12: second_group,
    }
    cache.admit(
        page_ids,
        set(range(4)),
        backing_keys,
        backing_values,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
    )

    cache.invalidate(cluster_ids[:2])

    assert first_group not in cache._group_states
    assert cache._group_states[second_group].num_pages == 1
    assert cache.num_resident_groups == 1
    assert cache.num_resident_clusters == 1
    assert cache.num_resident_pages == 1
    assert len(cache._free_slots) == cache.physical_capacity - 1
    assert list(cache._group_states[second_group].lru) == [12]

    cache.invalidate(cluster_ids[2:])

    assert cache.num_resident_groups == 0
    assert cache.num_resident_clusters == 0
    assert cache.num_resident_pages == 0
    assert not cache._group_states


def test_resident_cache_validates_cluster_group_metadata():
    cache = make_cache(capacity=1)
    backing_keys, backing_values = make_backing_pages()
    cluster_ids = torch.tensor([7], dtype=torch.int64)
    page_ids = torch.tensor([[0]], dtype=torch.int64)
    first_group = RetroSpecClusterGroup("first", 0)
    second_group = RetroSpecClusterGroup("second", 0)

    with pytest.raises(RuntimeError, match="Missing resident group metadata"):
        cache.admit(
            page_ids,
            {0},
            backing_keys,
            backing_values,
            cluster_ids=cluster_ids,
            cluster_groups={},
        )

    cache.admit(
        page_ids,
        {0},
        backing_keys,
        backing_values,
        cluster_ids=cluster_ids,
        cluster_groups={7: first_group},
    )

    with pytest.raises(RuntimeError, match="does not match requested ownership"):
        cache.lookup(
            page_ids,
            {0},
            cluster_ids=cluster_ids,
            cluster_groups={7: second_group},
        )

    assert cache._cluster_to_group == {7: first_group}
    assert cache.num_resident_groups == 1


def test_resident_cache_high_priority_misses_displace_lower_priority_hits():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    cache.admit(
        torch.tensor([[2], [3]], dtype=torch.int64),
        set(range(4)),
        backing_keys,
        backing_values,
    )

    page_ids = torch.tensor(
        [
            [[0], [2]],
            [[1], [3]],
        ],
        dtype=torch.int64,
    )
    access = cache.admit(
        page_ids,
        set(range(4)),
        backing_keys,
        backing_values,
    )

    assert access.hit_cluster_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert cache.lookup(
        torch.tensor([[2], [3]]),
        set(range(4)),
        touch=False,
    ).miss_cluster_mask.all()


def test_resident_cache_keeps_rank_zero_cluster_most_recent():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    cache.admit(
        torch.tensor([[0], [1]], dtype=torch.int64),
        set(range(3)),
        backing_keys,
        backing_values,
    )

    cache.admit(
        torch.tensor([[2]], dtype=torch.int64),
        set(range(3)),
        backing_keys,
        backing_values,
    )
    access = cache.lookup(
        torch.tensor([[0], [1], [2]], dtype=torch.int64),
        set(range(3)),
        touch=False,
    )

    assert access.hit_cluster_mask.tolist() == [True, False, True]


def test_resident_cache_admits_from_gpu_staging_pages():
    cache = make_cache(capacity=2)
    source_keys_cpu, source_values_cpu = make_backing_pages(num_pages=2)
    source_keys = source_keys_cpu.cuda()
    source_values = source_values_cpu.cuda()
    logical_page_ids = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
    staging_page_ids = torch.tensor([[1, 0]], dtype=torch.int64, device="cuda")

    access = cache.admit_staged(
        logical_page_ids,
        {0, 1},
        staging_page_ids,
        source_keys,
        source_values,
    )

    assert len(cache._pending_copy_batches) == 1
    pending = cache._pending_copy_batches[0]
    assert pending.source_key_pages is source_keys
    assert pending.source_value_pages is source_values

    logical_keys = source_keys_cpu.index_select(0, torch.tensor([1, 0]))
    logical_values = source_values_cpu.index_select(0, torch.tensor([1, 0]))
    assert_cached_pages_match_backing(
        cache,
        logical_page_ids.cpu(),
        access.cache_page_ids.cpu(),
        logical_keys,
        logical_values,
    )


def test_resident_cache_validates_staging_sources_before_eviction():
    cache = make_cache(capacity=1)
    backing_keys, backing_values = make_backing_pages()
    cache.admit(
        torch.tensor([[1]], dtype=torch.int64),
        {0, 1},
        backing_keys,
        backing_values,
    )

    with pytest.raises(RuntimeError, match="complete source pages"):
        cache.admit_staged(
            torch.tensor([[0]], dtype=torch.int64, device="cuda"),
            {0, 1},
            torch.tensor([[-1]], dtype=torch.int64, device="cuda"),
            backing_keys.cuda(),
            backing_values.cuda(),
        )

    assert cache.lookup(
        torch.tensor([[1]], dtype=torch.int64),
        {0, 1},
        touch=False,
    ).hit_cluster_mask.item()


def test_resident_cache_resize_evicts_whole_clusters_and_preserves_storage():
    cache = make_cache(capacity=3)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor([[0, 1], [2, -1]], dtype=torch.int64)
    cache.admit(page_ids, set(range(3)), backing_keys, backing_values)

    cache.resize(2)
    access = cache.lookup(page_ids, set(range(3)), touch=False)

    assert access.hit_cluster_mask.tolist() == [True, False]
    assert access.miss_cluster_mask.tolist() == [False, True]
    assert cache.capacity == 2
    assert cache.physical_capacity == 3
    assert cache.num_resident_pages == 2

    cache.resize(5)
    assert cache.capacity == 5
    assert cache.physical_capacity == 5
    assert cache.lookup(page_ids[:1], set(range(3))).hit_cluster_mask.item()


def test_resident_cache_invalidation_prevents_stale_page_reuse():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    cache.admit(page_ids, {0, 1}, backing_keys, backing_values)
    assert len(cache._pending_copy_batches) == 1

    cache.invalidate(torch.tensor([0]))
    assert cache.num_pending_copy_batches == 0
    assert cache.num_resident_pages == 0
    assert cache.lookup(page_ids, {0, 1}).miss_cluster_mask.item()

    replacement_keys = backing_keys + 1000.0
    replacement_values = backing_values + 1000.0
    access = cache.admit(
        page_ids,
        {0, 1},
        replacement_keys,
        replacement_values,
    )
    assert_cached_pages_match_backing(
        cache,
        page_ids,
        access.cache_page_ids,
        replacement_keys,
        replacement_values,
    )


def test_resident_cache_rejects_invalid_cluster_ownership():
    cache = make_cache(capacity=4)
    backing_keys, backing_values = make_backing_pages()

    with pytest.raises(ValueError, match="same logical page twice"):
        cache.admit(
            torch.tensor([[0, 0]]),
            set(range(6)),
            backing_keys,
            backing_values,
        )

    with pytest.raises(ValueError, match="multiple clusters"):
        cache.admit(
            torch.tensor([[0, 1], [1, 2]]),
            set(range(6)),
            backing_keys,
            backing_values,
        )

    cache.admit(
        torch.tensor([[0, 1]]),
        set(range(6)),
        backing_keys,
        backing_values,
    )
    with pytest.raises(RuntimeError, match="ownership conflicts"):
        cache.admit(
            torch.tensor([[1, 2]]),
            set(range(6)),
            backing_keys,
            backing_values,
        )

    with pytest.raises(RuntimeError, match="unallocated logical page 6"):
        cache.lookup(torch.tensor([[6]]), set(range(6)))


def test_resident_cache_uses_cluster_id_as_identity_after_page_reuse():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    page_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    group = RetroSpecClusterGroup("request", 0)

    first = RetroSpecResidentClusterCache.admit(
        cache,
        cluster_ids=torch.tensor([7]),
        page_ids=page_ids,
        cluster_groups={7: group},
        allocated_cluster_ids={7},
        allocated_page_ids={0, 1},
        backing_key_pages=backing_keys,
        backing_value_pages=backing_values,
    )
    assert first.hit_cluster_mask.item()

    cache.invalidate(torch.tensor([7]))
    stale = RetroSpecResidentClusterCache.lookup(
        cache,
        cluster_ids=torch.tensor([8]),
        page_ids=page_ids,
        cluster_groups={8: group},
        allocated_cluster_ids={8},
        allocated_page_ids={0, 1},
        touch=False,
    )
    assert stale.miss_cluster_mask.item()


def test_resident_cache_rejects_cluster_id_descriptor_conflicts():
    cache = make_cache(capacity=2)
    backing_keys, backing_values = make_backing_pages()
    group = RetroSpecClusterGroup("request", 0)

    with pytest.raises(ValueError, match="different logical pages"):
        RetroSpecResidentClusterCache.admit(
            cache,
            cluster_ids=torch.tensor([3, 3]),
            page_ids=torch.tensor([[0, 1], [2, 3]]),
            cluster_groups={3: group},
            allocated_cluster_ids={3},
            allocated_page_ids={0, 1, 2, 3},
            backing_key_pages=backing_keys,
            backing_value_pages=backing_values,
        )

    with pytest.raises(RuntimeError, match="unallocated cluster 4"):
        RetroSpecResidentClusterCache.lookup(
            cache,
            cluster_ids=torch.tensor([4]),
            page_ids=torch.tensor([[0]]),
            cluster_groups={4: group},
            allocated_cluster_ids={3},
            allocated_page_ids={0},
        )


def test_gpu_handle_table_tracks_resident_admission_and_invalidation():
    group = RetroSpecClusterGroup("request", 0)
    cache = make_cache(capacity=2, group_targets={group: 2})
    backing_keys, backing_values = make_backing_pages()
    cluster_ids = torch.tensor([[[7]]], dtype=torch.int64, device="cuda")
    logical_page_ids = torch.tensor([[[[0, 1]]]], dtype=torch.int64, device="cuda")
    active_mask = torch.tensor([True], device="cuda")

    def lookup_gpu():
        access = cache.lookup_gpu(
            cluster_ids=cluster_ids,
            page_ids=logical_page_ids,
            active_mask=active_mask,
            cache_page_ids=torch.empty_like(logical_page_ids),
            hit_cluster_mask=torch.empty_like(cluster_ids, dtype=torch.bool),
            miss_cluster_mask=torch.empty_like(cluster_ids, dtype=torch.bool),
            hit_gate_ready_mask=torch.empty_like(cluster_ids, dtype=torch.bool),
            access_kinds=torch.empty_like(cluster_ids, dtype=torch.uint8),
        )
        assert access.read_lease is not None
        access.read_lease.release()
        return access

    cold = lookup_gpu()
    assert cold.miss_cluster_mask.item()
    assert cold.access_kinds is not None
    assert cold.access_kinds.item() == 2

    with cache.mutation_guard():
        RetroSpecResidentClusterCache.admit(
            cache,
            cluster_ids=cluster_ids.cpu().reshape(1),
            page_ids=logical_page_ids.cpu().reshape(1, 2),
            cluster_groups={7: group},
            allocated_cluster_ids={7},
            allocated_page_ids={0, 1},
            backing_key_pages=backing_keys,
            backing_value_pages=backing_values,
        )
    cache.synchronize_pending_copies()

    resident = lookup_gpu()
    assert resident.cache_page_ids.cpu().tolist() == [[[[0, 1]]]]
    assert resident.hit_cluster_mask.item()
    assert resident.hit_gate_ready_mask.item()
    assert resident.access_kinds is not None
    assert resident.access_kinds.item() == 1

    with cache.mutation_guard():
        cache.invalidate(torch.tensor([7]))
    invalidated = lookup_gpu()
    assert invalidated.miss_cluster_mask.item()
    assert invalidated.cache_page_ids.cpu().tolist() == [[[[-1, -1]]]]
