# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _resident_handle_hash(cluster_handle):
    """Match the uint32 avalanche hash used by the CPU table publisher."""
    value = cluster_handle.to(tl.int64)
    value = (value ^ (value >> 32)) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


@triton.jit
def _find_resident_buckets(
    cluster_handles,
    selected,
    bound_buckets,
    table_handles,
    table_versions,
    TABLE_CAPACITY: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    TRACK_STATISTICS: tl.constexpr,
):
    bound_valid = selected & (bound_buckets >= 0) & (bound_buckets < TABLE_CAPACITY)
    safe_bound_buckets = tl.maximum(bound_buckets, 0)
    bound_versions_before = tl.atomic_add(
        table_versions + safe_bound_buckets, 0, mask=bound_valid, sem="acquire"
    )
    bound_handles = tl.load(
        table_handles + safe_bound_buckets, mask=bound_valid, other=-1
    )
    bound_versions_after = tl.atomic_add(
        table_versions + safe_bound_buckets, 0, mask=bound_valid, sem="acquire"
    )
    bound_stable = (bound_versions_before == bound_versions_after) & (
        (bound_versions_before & 1) == 0
    )
    direct_matches = bound_valid & bound_stable & (bound_handles == cluster_handles)

    matched_buckets = tl.where(
        direct_matches,
        bound_buckets,
        tl.full((BLOCK_WIDTH,), -1, tl.int64),
    )
    first_buckets = _resident_handle_hash(cluster_handles) & (TABLE_CAPACITY - 1)
    fallback_lookups = selected & ~direct_matches
    searching = fallback_lookups
    searching_count = tl.sum(searching.to(tl.int32), axis=0)
    probe_counts = tl.zeros((BLOCK_WIDTH,), tl.int32)
    probe = 0
    while tl.condition((probe < 64) & (searching_count > 0), disable_licm=True):
        if TRACK_STATISTICS:
            probe_counts += searching.to(tl.int32)

        buckets = (first_buckets + probe) & (TABLE_CAPACITY - 1)
        versions_before = tl.atomic_add(
            table_versions + buckets, 0, mask=searching, sem="acquire"
        )
        stored_handles = tl.load(table_handles + buckets, mask=searching, other=-1)
        versions_after = tl.atomic_add(
            table_versions + buckets, 0, mask=searching, sem="acquire"
        )
        stable = (versions_before == versions_after) & ((versions_before & 1) == 0)
        matched = searching & stable & (stored_handles == cluster_handles)
        empty = searching & stable & (stored_handles == -1)
        matched_buckets = tl.where(matched, buckets, matched_buckets)
        searching &= ~matched & ~empty
        searching_count = tl.sum(searching.to(tl.int32), axis=0)
        probe += 1

    return matched_buckets, direct_matches, fallback_lookups, probe_counts


@triton.jit
def _record_resident_lookup_statistics(
    statistics_buffer,
    selected,
    bound_buckets,
    direct_matches,
    fallback_lookups,
    probe_counts,
    stable_hits,
    BOUND_DIRECT_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_LOOKUP_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_HIT_COUNTER_INDEX: tl.constexpr,
    HASH_FALLBACK_MISS_COUNTER_INDEX: tl.constexpr,
    HASH_PROBE_STEP_COUNTER_INDEX: tl.constexpr,
    HASH_MAX_PROBE_COUNTER_INDEX: tl.constexpr,
    BINDING_INVALIDATION_COUNTER_INDEX: tl.constexpr,
):
    direct_hits = direct_matches & stable_hits
    fallback_hits = fallback_lookups & stable_hits
    fallback_misses = fallback_lookups & ~stable_hits
    binding_invalidations = selected & (bound_buckets >= 0) & ~direct_hits

    tl.atomic_add(
        statistics_buffer + BOUND_DIRECT_HIT_COUNTER_INDEX,
        tl.sum(direct_hits.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_FALLBACK_LOOKUP_COUNTER_INDEX,
        tl.sum(fallback_lookups.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_FALLBACK_HIT_COUNTER_INDEX,
        tl.sum(fallback_hits.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_FALLBACK_MISS_COUNTER_INDEX,
        tl.sum(fallback_misses.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + HASH_PROBE_STEP_COUNTER_INDEX,
        tl.sum(probe_counts, axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_max(
        statistics_buffer + HASH_MAX_PROBE_COUNTER_INDEX,
        tl.max(probe_counts, axis=0).to(tl.int64),
        sem="relaxed",
    )
    tl.atomic_add(
        statistics_buffer + BINDING_INVALIDATION_COUNTER_INDEX,
        tl.sum(binding_invalidations.to(tl.int32), axis=0).to(tl.int64),
        sem="relaxed",
    )
