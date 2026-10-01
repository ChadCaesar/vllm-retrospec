# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from tests.retrospec.support.segmented_index import (
    active_residency,
    build_index,
    make_cache,
    make_index,
    materialize_reference,
)
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel


def test_segmented_index_appends_complete_segments_and_handles_rollback():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    build_index(index, 10, keys, values, block_table)
    first_segment = index._indices["layer"]["request"].segments[0]
    assert index.needs_update("request", 14, ["layer"], True)

    build_index(index, 14, keys, values, block_table)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 10
    assert record.num_clusters == 4
    assert len(record.segments) == 2
    assert record.segments[0] is first_segment
    assert [
        (segment.indexed_start, segment.indexed_end) for segment in record.segments
    ] == [(2, 6), (6, 10)]
    assert [segment.cluster_start for segment in record.segments] == [0, 2]
    assert index.cluster_store.num_allocated_pages("layer") == 4

    first_identities = index.cluster_store.get_cluster_identities(
        "layer", record.segments[0].cluster_blocks.cluster_ids
    )
    second_identities = index.cluster_store.get_cluster_identities(
        "layer", record.segments[1].cluster_blocks.cluster_ids
    )
    assert [identity.local_cluster_id for identity in first_identities.values()] == [
        0,
        1,
    ]
    assert [identity.local_cluster_id for identity in second_identities.values()] == [
        2,
        3,
    ]
    assert all(
        identity.group.request_id == "request" and identity.group.kv_head_index == 0
        for identity in (*first_identities.values(), *second_identities.values())
    )

    assert index.needs_update("request", 6, ["layer"], True)
    build_index(index, 6, keys, values, block_table)
    record = index._indices["layer"]["request"]
    assert record.indexed_end == 2
    assert record.num_clusters == 0
    assert record.segments == []
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_rollback_invalidates_active_view_before_rebuild(monkeypatch):
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 14, keys, values, block_table)

    def fail_clustering(**_kwargs):
        raise RuntimeError("clustering failed")

    monkeypatch.setattr(
        "vllm.v1.spec_decode.retrospec.legacy.segmented.build.segmented_kmeans",
        fail_clustering,
    )

    with active_residency(index, ["request"]):
        original = index._get_resident_view("layer", ["request"], keys)
        with pytest.raises(RuntimeError, match="clustering failed"):
            build_index(index, 10, keys, values, block_table)
        rebuilt = index._get_resident_view("layer", ["request"], keys)

    assert rebuilt is not original
    assert rebuilt.request_slot_ids.tolist() == [-1]
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_indexed_tokens_are_materialized_from_secondary_pages():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.begin_proposal(["request"])
    try:
        sparse = index.select_segmented(
            request_ids=["request"],
            layer_name="layer",
            query=torch.ones(1, 1, 1),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([10], dtype=torch.int32),
            active_mask=torch.tensor([True]),
            scale=1.0,
        )

        # The indexed source blocks may be recycled after the prefill copy.
        # Expanded materialization must still read their original KV from the
        # private cluster store, not from the primary cache.
        keys[1:3].fill_(99.0)
        values[1:3].fill_(999.0)
        expanded = index.materialize(
            sparse.plan,
            RetroSpecAttentionLevel.EXPANDED,
            keys,
            values,
            block_table,
        )
    finally:
        index.end_proposal()

    expanded_keys, expanded_values, expanded_mask = materialize_reference(
        index, expanded, keys, values, block_table
    )

    expanded_keys = expanded_keys[0, 0, expanded_mask[0, 0], 0]
    expanded_values = expanded_values[0, 0, expanded_mask[0, 0], 0]

    assert sorted(expanded_keys[-4:].tolist()) == [1.0, 1.0, 2.0, 2.0]
    assert sorted(expanded_values[-4:].tolist()) == [10.0, 10.0, 20.0, 20.0]


def test_segmented_index_removes_finished_request_state():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    index.remove_requests(["request"])

    assert "request" not in index._indices["layer"]
    assert index.needs_update("request", 10, ["layer"], True)
    assert index.cluster_store.num_allocated_pages("layer") == 0


def test_segmented_index_reuses_active_view_until_update():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)
    with active_residency(index, ["request"]):
        first = index._get_resident_view("layer", ["request"], keys)
        second = index._get_resident_view("layer", ["request"], keys)
        assert second is first

        build_index(index, 14, keys, values, block_table)
        after_update = index._get_resident_view("layer", ["request"], keys)

        assert after_update is not first


def test_removing_request_invalidates_active_view():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    with active_residency(index, ["request"]):
        view = index._get_resident_view("layer", ["request"], keys)

        index.remove_requests(["request"])

        assert "request" not in index._indices["layer"]
        rebuilt = index._get_resident_view("layer", ["request"], keys)
        assert rebuilt is not view
        assert rebuilt.request_slot_ids.tolist() == [-1]


def test_active_view_tracks_request_order_without_block_table_width():
    index = make_index(max_resident_requests=2)
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["long", "short"],
        seq_lens=[10, 3],
        is_prefill=[True, True],
        rows=[0, 1],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    with active_residency(index, ["long", "short"]):
        original = index._get_resident_view("layer", ["long", "short"], keys)
        with pytest.raises(RuntimeError, match="request order"):
            index._get_resident_view("layer", ["short", "long"], keys)
        repeated = index._get_resident_view("layer", ["long", "short"], keys)

        assert repeated is original
        assert original.request_slot_ids.shape == (2,)


def test_segmented_index_proposal_lifecycle_tracks_empty_batches():
    index = make_index()

    index.begin_proposal([])
    with pytest.raises(RuntimeError, match="already active"):
        index.begin_proposal([])
    index.end_proposal()

    with pytest.raises(RuntimeError, match="not active"):
        index.end_proposal()


def test_provenance_rejects_index_revision_changes_inside_proposal():
    index = make_index(replay_mode="trace")
    index._indices["layer"] = {"request": index._empty_index()}

    index.begin_proposal(["request"])
    try:
        revisions = index._get_proposal_index_revisions("layer", ["request"])
        assert revisions == (index._indices["layer"]["request"].revision,)

        index._indices["layer"]["request"] = index._empty_index()
        with pytest.raises(RuntimeError, match="changed inside a proposal"):
            index._get_proposal_index_revisions("layer", ["request"])
    finally:
        index.end_proposal()

    assert index._proposal_index_revisions == {}
    index.close()


def test_provenance_records_published_index_segment_ranges():
    index = make_index(replay_mode="trace")
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    record_segment = Mock()
    index.selection_provenance.record_index_segment = record_segment

    build_index(index, 10, keys, values, block_table)

    record_segment.assert_called_once()
    call = record_segment.call_args.kwargs
    assert call["request_id"] == "request"
    assert call["layer_name"] == "layer"
    assert call["indexed_start"] == 2
    assert call["indexed_end"] == 6
    assert call["cluster_start"] == 0
    assert call["assignments"].shape == (1, 4)
    assert call["cluster_sizes"].shape == (1, 2)
    index.close()


def test_freeze_resident_replay_matches_proposal_lifetime(monkeypatch):
    index = make_index(replay_mode="freeze_resident")
    begin_replay = Mock()
    end_replay = Mock()
    monkeypatch.setattr(index.cluster_store, "begin_resident_replay", begin_replay)
    monkeypatch.setattr(index.cluster_store, "end_resident_replay", end_replay)

    index.begin_proposal(["request"])
    begin_replay.assert_called_once_with(())
    assert index._proposal_resident_frozen
    index.end_proposal()

    end_replay.assert_called_once_with()
    assert not index._proposal_resident_frozen
    index.close()


def test_freeze_resident_replay_unwinds_failed_activation(monkeypatch):
    index = make_index(replay_mode="freeze_resident")
    begin_replay = Mock()
    end_replay = Mock()
    monkeypatch.setattr(index.cluster_store, "begin_resident_replay", begin_replay)
    monkeypatch.setattr(index.cluster_store, "end_resident_replay", end_replay)
    monkeypatch.setattr(
        index._gpu_index_residency,
        "activate",
        Mock(side_effect=RuntimeError("activation failed")),
    )

    with pytest.raises(RuntimeError, match="activation failed"):
        index.begin_proposal(["request"])

    begin_replay.assert_called_once_with(())
    end_replay.assert_called_once_with()
    assert not index._proposal_resident_frozen
    assert index._proposal_index_revisions == {}
    assert not index._proposal_active
    index.close()


def test_ready_selected_replay_requires_pinned_staging():
    with pytest.raises(ValueError, match="requires pinned CPU staging memory"):
        make_index(replay_mode="ready_selected")


def test_cpu_offload_keeps_request_indices_after_batch_deactivation():
    index = make_index(max_resident_requests=2)
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).repeat(2, 1)
    index.build_or_update(
        layer_name="layer",
        request_ids=["first", "second"],
        seq_lens=[10, 10],
        is_prefill=[True, True],
        rows=[0, 1],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
    )

    for request_id in ("first", "second"):
        segment = index._indices["layer"][request_id].segments[0]
        assert segment.cluster_keys.device.type == "cpu"
        assert segment.cluster_values.device.type == "cpu"
        assert segment.cluster_token_counts.device.type == "cpu"

    index.begin_proposal(["first", "second"])
    try:
        view = index._gpu_index_residency.get_active_view(
            "layer", ["first", "second"], keys.device
        )
        assert view.arena is not None
        for row, request_id in enumerate(("first", "second")):
            segment = index._indices["layer"][request_id].segments[0]
            slot = int(view.request_slot_ids[row].item())
            num_clusters = segment.cluster_token_counts.shape[1]
            cluster_offset = int(view.arena.cluster_offsets[slot].item())
            cluster_slice = slice(cluster_offset, cluster_offset + num_clusters)
            assert torch.equal(
                view.arena.cluster_keys[:, cluster_slice], segment.cluster_keys
            )
            assert torch.equal(
                view.arena.cluster_values[:, cluster_slice], segment.cluster_values
            )
            assert torch.equal(
                view.arena.cluster_token_counts[:, cluster_slice],
                segment.cluster_token_counts,
            )
    finally:
        index.end_proposal()

    assert index._gpu_index_residency.resident_request_ids == (
        "first",
        "second",
    )
    assert index._gpu_index_residency.num_resident_layers == 1

    index.begin_proposal(["first", "second"])
    try:
        view = index._get_resident_view("layer", ["first", "second"], keys)
        assert view.arena is not None
        assert (view.request_slot_ids >= 0).all()
    finally:
        index.end_proposal()

    assert index._gpu_index_residency.num_resident_layers == 1

    index.remove_requests(["first"])
    assert index._gpu_index_residency.resident_request_ids == ("second",)
    assert index._gpu_index_residency.get_num_clusters("layer", "first") == 0
