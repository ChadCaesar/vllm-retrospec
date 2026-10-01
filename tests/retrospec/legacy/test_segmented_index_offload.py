# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future
from unittest.mock import Mock, patch

import pytest
import torch

from tests.retrospec.support.segmented_index import (
    build_index,
    make_cache,
    make_index,
)
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.legacy.segmented import (
    build as segmented_build_module,
)


def test_cpu_offload_appends_resident_segments_across_index_updates():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    first_num_clusters = index._gpu_index_residency.get_num_clusters("layer", "request")
    first_indexed_end = index._gpu_index_residency.get_indexed_end("layer", "request")

    index.begin_proposal(["request"])
    try:
        first_view = index._get_resident_view("layer", ["request"], keys)
        assert first_view.arena is not None
        first_slot = first_view.request_slot_ids.item()
        assert first_view.arena.num_clusters[first_slot].item() == 2
    finally:
        index.end_proposal()

    build_index(index, 14, keys, values, block_table)

    updated_num_clusters = index._gpu_index_residency.get_num_clusters(
        "layer", "request"
    )
    updated_indexed_end = index._gpu_index_residency.get_indexed_end("layer", "request")
    assert first_num_clusters == 2
    assert updated_num_clusters == 4
    assert first_indexed_end == 6
    assert updated_indexed_end == 10

    index.begin_proposal(["request"])
    try:
        updated_view = index._get_resident_view("layer", ["request"], keys)
        assert updated_view.arena is not None
        updated_slot = updated_view.request_slot_ids.item()
        assert updated_view.arena.num_clusters[updated_slot].item() == 4
    finally:
        index.end_proposal()


def test_cpu_offload_rejects_more_requests_than_reserved_capacity():
    index = make_index(max_resident_requests=2)

    with pytest.raises(RuntimeError, match="exceeds max_num_seqs"):
        index.begin_proposal(["first", "second", "third"])


def test_cpu_offload_can_defer_flush_and_discard_index_updates():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )

    assert index.has_staged_updates
    assert index.needs_update("request", 10, ["layer"], True)
    with pytest.raises(RuntimeError, match="staged index updates"):
        index.begin_proposal(["request"])

    # The background worker may finish page construction before the index is
    # published. Its pages remain private to the staged transaction.
    index._staged_segments[0].build_future.result()
    assert index.cluster_store.num_allocated_pages("layer") == 2
    assert index.needs_update("request", 10, ["layer"], True)

    index.discard_staged_updates()

    assert not index.has_staged_updates
    assert index._cluster_build_executor is None
    assert not index._pending_cluster_builds
    assert index.cluster_store.num_allocated_pages("layer") == 0
    assert index.needs_update("request", 10, ["layer"], True)

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )
    index.flush_staged_updates()

    assert not index.has_staged_updates
    assert index._cluster_build_executor is None
    assert not index._pending_cluster_builds
    assert index.cluster_store.num_allocated_pages("layer") == 2
    assert not index.needs_update("request", 10, ["layer"], True)


def test_cpu_offload_direct_build_preserves_synchronous_api():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)

    assert not index.has_staged_updates
    assert index.cluster_store.num_allocated_pages("layer") == 2
    index.begin_proposal(["request"])
    index.end_proposal()


def test_cpu_offload_recreates_builder_for_later_index_transaction():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)
    first_segment = index._indices["layer"]["request"].segments[0]

    assert index._cluster_build_executor is None

    build_index(index, 14, keys, values, block_table)

    record = index._indices["layer"]["request"]
    assert index._cluster_build_executor is None
    assert len(record.segments) == 2
    assert record.segments[0] is first_segment
    assert record.indexed_end == 10
    assert index.cluster_store.num_allocated_pages("layer") == 4


def test_cpu_offload_builds_cluster_pages_on_background_worker(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_started = threading.Event()
    allow_build = threading.Event()
    worker_names: list[str] = []
    original_store = index.cluster_store.store_staged_clusters

    def store_staged_clusters(*args, **kwargs):
        worker_names.append(threading.current_thread().name)
        build_started.set()
        if not allow_build.wait(timeout=5):
            raise RuntimeError("background build was not released")
        return original_store(*args, **kwargs)

    monkeypatch.setattr(
        index.cluster_store,
        "store_staged_clusters",
        store_staged_clusters,
    )

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )

    try:
        assert build_started.wait(timeout=5)
        assert index.has_staged_updates
        assert index.needs_update("request", 10, ["layer"], True)
        assert worker_names[0].startswith("retrospec-cluster-page")

        allow_build.set()
        index._staged_segments[0].build_future.result()
        metadata_lookup = Mock(
            side_effect=AssertionError("publish rebuilt persistent block metadata")
        )
        monkeypatch.setattr(
            index.cluster_store,
            "get_cluster_block_metadata",
            metadata_lookup,
        )
        index.flush_staged_updates()
        metadata_lookup.assert_not_called()
    finally:
        allow_build.set()
        if index.has_staged_updates:
            index.discard_staged_updates()

    assert not index.needs_update("request", 10, ["layer"], True)
    assert index.cluster_store.num_allocated_pages("layer") == 2


def test_cpu_offload_backpressure_waits_for_oldest_pending_build():
    index = make_index(max_pending_cluster_builds=2)
    first_build: Future = Future()
    second_build: Future = Future()
    index._pending_cluster_builds.extend((first_build, second_build))
    wait_started = threading.Event()
    slot_available = threading.Event()
    errors: list[BaseException] = []

    def wait_for_slot():
        wait_started.set()
        try:
            index._wait_for_cluster_build_slot()
        except BaseException as exc:
            errors.append(exc)
        finally:
            slot_available.set()

    waiter = threading.Thread(target=wait_for_slot)
    waiter.start()

    try:
        assert wait_started.wait(timeout=5)
        assert not slot_available.wait(timeout=0.1)
        first_build.set_result(Mock())
        assert slot_available.wait(timeout=5)
    finally:
        if not first_build.done():
            first_build.set_result(Mock())
        waiter.join(timeout=5)

    assert not waiter.is_alive()
    assert not errors
    assert list(index._pending_cluster_builds) == [second_build]


def test_cpu_offload_backpressure_reaps_completed_builds_and_propagates_errors():
    index = make_index(max_pending_cluster_builds=2)
    completed_build: Future = Future()
    failed_build: Future = Future()
    completed_build.set_result(Mock())
    failed_build.set_exception(RuntimeError("background build failed"))
    index._pending_cluster_builds.extend((completed_build, failed_build))

    with pytest.raises(RuntimeError, match="background build failed"):
        index._wait_for_cluster_build_slot()

    assert not index._pending_cluster_builds


def test_cpu_offload_flush_rolls_back_all_layers_after_build_failure(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    original_store = index.cluster_store.store_staged_clusters

    def store_staged_clusters(*args, **kwargs):
        if kwargs["layer_name"] == "failed-layer":
            raise RuntimeError("background build failed")
        return original_store(*args, **kwargs)

    monkeypatch.setattr(
        index.cluster_store,
        "store_staged_clusters",
        store_staged_clusters,
    )

    for layer_name in ("completed-layer", "failed-layer"):
        index.build_or_update(
            layer_name=layer_name,
            request_ids=["request"],
            seq_lens=[10],
            is_prefill=[True],
            rows=[0],
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            defer_cpu_store=True,
        )

    with pytest.raises(RuntimeError, match="background build failed"):
        index.flush_staged_updates()

    assert not index.has_staged_updates
    assert index._cluster_build_executor is None
    assert index.needs_update("request", 10, ["completed-layer"], True)
    assert index.needs_update("request", 10, ["failed-layer"], True)
    assert index.cluster_store.num_allocated_pages("completed-layer") == 0
    assert index.cluster_store.num_allocated_pages("failed-layer") == 0


def test_cpu_offload_stages_token_kv_before_clustering(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    call_order: list[str] = []

    original_stage_token_kv = index.cluster_store.stage_token_kv
    original_finish_stage_clusters = index.cluster_store.finish_stage_clusters
    original_wait_for_slot = index._wait_for_cluster_build_slot

    def wait_for_cluster_build_slot():
        call_order.append("wait_for_cluster_build_slot")
        return original_wait_for_slot()

    def stage_token_kv(*args, **kwargs):
        call_order.append("stage_token_kv")
        return original_stage_token_kv(*args, **kwargs)

    def finish_stage_clusters(*args, **kwargs):
        call_order.append("finish_stage_clusters")
        return original_finish_stage_clusters(*args, **kwargs)

    original_clustering = segmented_build_module.segmented_kmeans

    def segmented_kmeans(*args, **kwargs):
        call_order.append("segmented_kmeans")
        return original_clustering(*args, **kwargs)

    monkeypatch.setattr(index.cluster_store, "stage_token_kv", stage_token_kv)
    monkeypatch.setattr(
        index,
        "_wait_for_cluster_build_slot",
        wait_for_cluster_build_slot,
    )
    monkeypatch.setattr(
        index.cluster_store,
        "finish_stage_clusters",
        finish_stage_clusters,
    )
    monkeypatch.setattr(
        segmented_build_module,
        "segmented_kmeans",
        segmented_kmeans,
    )

    build_index(
        index,
        10,
        keys,
        values,
        block_table,
        defer_cpu_store=True,
    )

    try:
        assert call_order == [
            "wait_for_cluster_build_slot",
            "stage_token_kv",
            "segmented_kmeans",
            "finish_stage_clusters",
        ]
    finally:
        index.discard_staged_updates()


def test_cpu_offload_waits_for_staged_kv_when_clustering_fails(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    staged_token_kv = Mock()
    discard_staged_token_kv = Mock(
        wraps=index.cluster_store.discard_staged_token_kv,
    )

    monkeypatch.setattr(
        index.cluster_store,
        "stage_token_kv",
        Mock(return_value=staged_token_kv),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "discard_staged_token_kv",
        discard_staged_token_kv,
    )

    monkeypatch.setattr(
        segmented_build_module,
        "segmented_kmeans",
        Mock(side_effect=RuntimeError("clustering failed")),
    )

    with pytest.raises(RuntimeError, match="clustering failed"):
        build_index(index, 10, keys, values, block_table)

    staged_token_kv.wait.assert_called_once_with()
    discard_staged_token_kv.assert_called_once_with(staged_token_kv)
    assert not index.has_staged_updates


def test_cpu_offload_discards_staged_kv_when_metadata_staging_fails(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    staged_token_kv = Mock()
    discard_staged_token_kv = Mock(
        wraps=index.cluster_store.discard_staged_token_kv,
    )

    monkeypatch.setattr(
        index.cluster_store,
        "stage_token_kv",
        Mock(return_value=staged_token_kv),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "finish_stage_clusters",
        Mock(side_effect=RuntimeError("metadata staging failed")),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "discard_staged_token_kv",
        discard_staged_token_kv,
    )

    with pytest.raises(RuntimeError, match="metadata staging failed"):
        build_index(index, 10, keys, values, block_table)

    staged_token_kv.wait.assert_called_once_with()
    discard_staged_token_kv.assert_called_once_with(staged_token_kv)
    assert not index.has_staged_updates


def test_cpu_offload_discards_staged_clusters_when_submission_fails(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    discard_staged_clusters = Mock(
        wraps=index.cluster_store.discard_staged_clusters,
    )

    monkeypatch.setattr(
        index,
        "_submit_cluster_build",
        Mock(side_effect=RuntimeError("submission failed")),
    )
    monkeypatch.setattr(
        index.cluster_store,
        "discard_staged_clusters",
        discard_staged_clusters,
    )

    with pytest.raises(RuntimeError, match="submission failed"):
        build_index(index, 10, keys, values, block_table)

    discard_staged_clusters.assert_called_once()
    assert not index.has_staged_updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_offload_draft_estimates_misses_and_uses_resident_hits():
    device = torch.device("cuda")
    index = make_index(
        cache_ratio=0.5,
    )
    keys, values = make_cache()
    keys = keys.to(device=device, dtype=torch.bfloat16)
    values = values.to(device=device, dtype=torch.bfloat16)
    block_table = torch.arange(
        7,
        dtype=torch.int32,
        device=device,
    ).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.begin_proposal(["request"])
    try:
        resident_view = index._gpu_index_residency.get_active_view(
            "layer", ["request"], device
        )
        assert resident_view.arena is not None
        resident_key_ptr = resident_view.arena.cluster_keys.data_ptr()
        assert resident_view.arena.cluster_keys.device.type == "cuda"
        assert resident_view.arena.cluster_values.device.type == "cuda"
        assert resident_view.arena.cluster_token_counts.device.type == "cuda"
        assert resident_view.arena.cluster_ids.device.type == "cuda"
        assert resident_view.arena.page_ids.device.type == "cuda"
    finally:
        index.end_proposal()
    assert index._indices["layer"]["request"].segments[0].cluster_keys.device.type == (
        "cpu"
    )

    selection_kwargs = {
        "request_ids": ["request"],
        "layer_name": "layer",
        "query": torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
        "key_cache": keys,
        "value_cache": values,
        "block_table": block_table,
        "seq_lens": torch.tensor([10], dtype=torch.int32, device=device),
        "active_mask": torch.tensor([True], device=device),
        "scale": 1.0,
    }

    index.begin_proposal(["request"])
    try:
        with patch.object(
            index,
            "_resolve_ranked_draft_clusters",
            wraps=index._resolve_ranked_draft_clusters,
        ) as resolve:
            cold = index.select_segmented(**selection_kwargs)
        assert resolve.call_args.kwargs["emit_misses"] is False
        cold_sparse_indices = cold.plan.ranked_cluster_indices[
            ..., : cold.plan.sparse_retrieval_width
        ].clone()
    finally:
        index.end_proposal()

    index.begin_proposal(["request"])
    try:
        persistent_view = index._gpu_index_residency.get_active_view(
            "layer", ["request"], device
        )
        assert persistent_view.arena is not None
        assert persistent_view.arena.cluster_keys.data_ptr() == resident_key_ptr
    finally:
        index.end_proposal()
    assert cold.resolved_clusters is not None
    assert cold.prefetch_miss_cluster_ids is None
    assert cold.prefetch_miss_positions is None
    assert cold.prefetch_miss_count is None
    assert cold.prefetch_num_groups == 0
    assert cold.prefetch_num_ranks == 0
    assert index.cluster_store.num_resident_pages("layer") == 0
    assert cold.exact_token_counts.tolist() == [[6]]
    assert cold.plan.candidate_counts.gt(0).all()
    assert cold.plan.ranked_cluster_indices.ge(0).all()
    assert cold.resolved_clusters.resident_bucket_ids.tolist() == [[[-1]]]
    assert cold.hit_attn.item() == pytest.approx(1.0)
    assert not cold.resolved_clusters.hit_gate_ready.any()
    assert not cold.resolved_clusters.clustered_token_counts.any()
    scratch = index._draft_selection_scratch
    assert scratch is not None

    index.begin_proposal(["request"])
    try:
        resident_view = index._gpu_index_residency.get_active_view(
            "layer", ["request"], device
        )
        logical_cluster_ids, logical_page_ids, logical_page_token_counts = (
            index._build_resident_exact_cluster_selection(
                resident_view,
                cold_sparse_indices.clamp_min(0),
                cold_sparse_indices >= 0,
            )
        )
    finally:
        index.end_proposal()
    assert logical_page_token_counts.sum().item() == 2

    index.cluster_store.admit_resident_clusters(
        "layer",
        logical_cluster_ids,
        logical_page_ids,
    )
    index.cluster_store.get_resident_page_storage("layer")
    torch.cuda.current_stream().synchronize()

    index.begin_proposal(["request"])
    try:
        warm = index.select_segmented(**selection_kwargs)
        indexed = index.get_indexed_selection(
            "layer",
            RetroSpecAttentionLevel.SPARSE,
            torch.tensor([0], dtype=torch.int64, device=device),
            torch.tensor([0], dtype=torch.int64, device=device),
        )
        verification = index.materialize_indexed_reference(indexed)
    finally:
        index.end_proposal()

    assert warm.resolved_clusters is not None
    assert warm.prefetch_miss_cluster_ids is None
    assert warm.prefetch_miss_positions is None
    assert warm.prefetch_miss_count is None
    assert index.cluster_store.num_resident_pages("layer") == 1
    assert warm.exact_token_counts.tolist() == [[8]]
    assert warm.plan.candidate_counts.gt(0).all()
    assert warm.plan.ranked_cluster_indices.ge(0).all()
    assert warm.resolved_clusters.resident_bucket_ids.ge(0).all()
    assert warm.resolved_clusters.hit_gate_ready.all()
    assert warm.hit_attn.item() == pytest.approx(warm.plan.sparse_attn.item())

    assert verification.resolved_pages is None
    assert verification.exact_token_counts.tolist() == [[8]]
    assert verification.estimation_token_counts.tolist() == [[[2]]]
    assert verification.attention_mass.item() == pytest.approx(
        warm.plan.sparse_attn.item()
    )
