# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.retrospec.resident_cache import (
    RetroSpecResidentClusterCache,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the resident cluster cache",
)


def make_cache(capacity: int) -> RetroSpecResidentClusterCache:
    cache = RetroSpecResidentClusterCache(
        page_size=2,
        head_size=1,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    cache.resize(capacity)
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
