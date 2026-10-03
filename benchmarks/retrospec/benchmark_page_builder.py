# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure the CPU cluster-page builder with reproducible full/partial pages."""

import argparse
import hashlib
import json
import statistics
import time

import torch

from vllm import _custom_ops as ops
from vllm.v1.spec_decode.retrospec.offload.cluster_store import (
    RetroSpecClusterPageStore,
)


def make_input(
    layout: str,
    num_heads: int,
    num_clusters: int,
    tokens_per_head: int,
    page_size: int,
    head_size: int,
    slab_pages: int,
) -> tuple[object, ...]:
    if layout not in ("full", "mixed"):
        raise ValueError("layout must be full or mixed")
    if num_clusters % 2:
        raise ValueError("num_clusters must be even")
    if tokens_per_head < num_clusters * 2:
        raise ValueError("tokens_per_head must allow at least two tokens per cluster")

    base, remainder = divmod(tokens_per_head, num_clusters)
    cluster_sizes = torch.full((num_clusters,), base, dtype=torch.int32)
    cluster_sizes[:remainder] += 1
    if layout == "full" and (remainder or base % page_size):
        raise ValueError("Full layout requires page-aligned equal cluster sizes")
    if layout == "mixed":
        cluster_sizes[::2] -= 1
        cluster_sizes[1::2] += 1
    token_count = int(cluster_sizes.sum())
    cluster_ids = torch.repeat_interleave(
        torch.arange(num_clusters, dtype=torch.int32), cluster_sizes
    )
    offsets = torch.cat(
        [torch.arange(int(size), dtype=torch.int32) for size in cluster_sizes]
    )
    permutation = torch.randperm(
        token_count, generator=torch.Generator().manual_seed(0)
    )
    assignments = cluster_ids[permutation].repeat(num_heads, 1).contiguous()
    token_offsets = offsets[permutation].repeat(num_heads, 1).contiguous()
    counts = cluster_sizes.repeat(num_heads, 1).contiguous()

    generator = torch.Generator().manual_seed(1)
    shape = (num_heads, token_count, head_size)
    token_keys = torch.randn(shape, dtype=torch.float16, generator=generator)
    token_values = torch.randn(shape, dtype=torch.float16, generator=generator)
    total_pages = int(
        torch.div(counts + page_size - 1, page_size, rounding_mode="floor").sum()
    )
    key_slabs = []
    value_slabs = []
    page_ids = []
    for slab_id, start in enumerate(range(0, total_pages, slab_pages)):
        slab_capacity = min(slab_pages, total_pages - start)
        key_slab = torch.empty(
            (slab_capacity, page_size, head_size), dtype=torch.float16
        )
        key_slabs.append(key_slab)
        value_slabs.append(torch.empty_like(key_slab))
        page_ids.extend((slab_id << 32) | offset for offset in range(slab_capacity))
    return (
        tuple(key_slabs),
        tuple(value_slabs),
        torch.tensor(page_ids, dtype=torch.int64),
        token_keys,
        token_values,
        assignments,
        counts,
        token_offsets,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("full", "mixed"), required=True)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--tokens-per-head", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--slab-pages", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--registration", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.heads,
            args.clusters,
            args.tokens_per_head,
            args.page_size,
            args.head_size,
            args.slab_pages,
            args.workers,
            args.warmups,
            args.repeats,
        )
        <= 0
    ):
        raise ValueError("All benchmark dimensions and repetitions must be positive")

    inputs = make_input(
        args.layout,
        args.heads,
        args.clusters,
        args.tokens_per_head,
        args.page_size,
        args.head_size,
        args.slab_pages,
    )
    key_slabs, value_slabs, page_ids, keys, values, assignments, counts, offsets = (
        inputs
    )
    call_args = (
        key_slabs,
        value_slabs,
        page_ids,
        keys,
        values,
        assignments,
        counts,
        offsets,
        args.page_size,
        args.workers,
    )
    for _ in range(args.warmups):
        ops.retrospec_build_cluster_pages(*call_args)

    times = []
    result = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = ops.retrospec_build_cluster_pages(*call_args)
        times.append((time.perf_counter() - started) * 1000)

    assert result is not None
    digest = hashlib.sha256()
    for tensor in (*result, *key_slabs, *value_slabs):
        digest.update(tensor.contiguous().numpy().tobytes())

    registration_times = []
    registration_digest = None
    if args.registration:
        for _ in range(args.repeats):
            store = RetroSpecClusterPageStore(page_size=args.page_size)
            started = time.perf_counter()
            cluster_ids = store._allocate_cluster_ids(
                "layer", "request", 0, counts, result[0], result[1]
            )
            registration_times.append((time.perf_counter() - started) * 1000)
        registration_hash = hashlib.sha256()
        arena = store._prefetch_descriptor_arenas["layer"]
        for tensor in (
            cluster_ids,
            arena.page_ids,
            arena.page_counts,
            arena.group_ids,
        ):
            registration_hash.update(tensor.contiguous().numpy().tobytes())
        registration_digest = registration_hash.hexdigest()
    print(
        "RETROSPEC_PAGE_BUILDER_RESULT="
        + json.dumps(
            {
                "layout": args.layout,
                "heads": args.heads,
                "clusters": args.clusters,
                "tokens_per_head": keys.shape[1],
                "pages": page_ids.numel(),
                "slabs": len(key_slabs),
                "slab_pages": args.slab_pages,
                "page_size": args.page_size,
                "head_size": args.head_size,
                "workers": args.workers,
                "warmups": args.warmups,
                "times_ms": times,
                "median_ms": statistics.median(times),
                "output_sha256": digest.hexdigest(),
                "registration_times_ms": registration_times,
                "registration_median_ms": (
                    statistics.median(registration_times)
                    if registration_times
                    else None
                ),
                "registration_sha256": registration_digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
