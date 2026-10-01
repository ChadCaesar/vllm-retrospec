# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.retrospec.support.segmented_index import (
    build_index,
    make_cache,
    make_index,
)


def test_full_verification_plan_covers_clustered_and_primary_tokens():
    index = make_index(max_resident_requests=2)
    keys, values = make_cache(num_kv_heads=2)
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

    index.begin_full_verification_residency(["long", "short"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["long", "short"],
            layer_name="layer",
            seq_lens=[10, 3],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    assert plan.layer_name == "layer"
    assert plan.exact_token_counts.tolist() == [[10, 10], [3, 3]]
    assert plan.primary_exact_token_mask.sum(dim=2).tolist() == [[6, 6], [3, 3]]
    assert plan.clustered_descriptors[0].head_token_counts == (4, 4)
    assert plan.clustered_descriptors[1].head_token_counts == (0, 0)
    assert plan.primary_exact_token_indices[0, 0].tolist() == [0, 1, 6, 7, 8, 9]
    assert plan.primary_exact_token_mask[1, 0].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    short_primary_indices = plan.primary_exact_token_indices[1, 0][
        plan.primary_exact_token_mask[1, 0]
    ]
    assert short_primary_indices.tolist() == [0, 1, 2]
    assert plan.clustered_kv is None


def test_full_verification_plan_reuses_persistent_cpu_page_descriptor():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    descriptor = index._indices["layer"]["request"].full_verification_descriptor
    assert descriptor is not None

    with patch.object(
        index.cluster_store,
        "get_cluster_block_metadata",
        side_effect=AssertionError("full verify rebuilt the CPU descriptor"),
    ):
        index.begin_full_verification_residency(["request"])
        try:
            plan = index.build_full_verification_plan(
                request_ids=["request"],
                layer_name="layer",
                seq_lens=[10],
                key_cache=keys,
                block_table=block_table,
            )
        finally:
            index.end_full_verification_residency()

    assert plan.clustered_descriptors[0] is descriptor


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for full-verification pipelining",
)
def test_full_verification_pipeline_prefetches_next_layer():
    device = torch.device("cuda", torch.cuda.current_device())
    index = make_index()
    keys, values = make_cache()
    keys = keys.to(device)
    values = values.to(device)
    block_table = torch.arange(7, dtype=torch.int32, device=device).view(1, -1)

    for layer_name in ("layer.0", "layer.1"):
        index.build_or_update(
            layer_name=layer_name,
            request_ids=["request"],
            seq_lens=[10],
            is_prefill=[True],
            rows=[0],
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
        )

    index.begin_full_verification_residency(["request"])
    try:
        with patch.object(
            index.cluster_store,
            "get_cluster_block_metadata",
            side_effect=AssertionError("pipeline rebuilt the CPU descriptor"),
        ):
            index.begin_full_verification_pipeline(
                ["request"],
                {"layer.0": 1, "layer.1": 1},
                device,
            )
            assert len(index._full_verification_prefetched) == 1
            assert index._full_verification_prefetched[0].layer_name == "layer.0"
            assert index._full_verification_prefetched[0].ticket is not None

            first = index.build_full_verification_plan(
                request_ids=["request"],
                layer_name="layer.0",
                seq_lens=[10],
                key_cache=keys,
                block_table=block_table,
            )
            assert first.clustered_kv is not None
            assert len(index._full_verification_prefetched) == 1
            assert index._full_verification_prefetched[0].layer_name == "layer.1"
            assert index._full_verification_prefetched[0].ticket is not None

            second = index.build_full_verification_plan(
                request_ids=["request"],
                layer_name="layer.1",
                seq_lens=[10],
                key_cache=keys,
                block_table=block_table,
            )
            assert second.clustered_kv is not None
            assert not index._full_verification_prefetched
    finally:
        index.end_full_verification_pipeline()
        index.end_full_verification_residency()

    assert first.clustered_kv.ready_event is not None
    assert second.clustered_kv.ready_event is not None
    torch.cuda.current_stream(device).wait_event(first.clustered_kv.ready_event)
    torch.cuda.current_stream(device).wait_event(second.clustered_kv.ready_event)
    torch.cuda.synchronize(device)


def test_full_verification_prime_is_adopted_and_refills_after_two_layers():
    index = make_index()
    tickets = [Mock() for _ in range(3)]
    for ticket_index, ticket in enumerate(tickets):
        ticket.ready.return_value = True
        ticket.result.return_value = f"staging-{ticket_index}"
    index.cluster_store.submit_full_verification_tokens = Mock(side_effect=tickets)

    with patch.object(
        index,
        "_get_full_verification_descriptors",
        return_value=(SimpleNamespace(num_tokens=1),),
    ):
        index.begin_proposal(["request"])
        assert index.prime_full_verification_pipeline(
            ["request"],
            {"layer.0": 1, "layer.1": 1, "layer.2": 1},
            torch.device("cuda", 0),
        )
        index.end_proposal()

        index.begin_full_verification_pipeline(
            ["request"],
            {"layer.0": 1, "layer.1": 1, "layer.2": 1},
            torch.device("cuda", 0),
        )
        assert len(index._full_verification_prefetched) == 2
        assert index.consume_full_verification_layer("layer.0") == "staging-0"
        assert len(index._full_verification_prefetched) == 1
        assert index.cluster_store.submit_full_verification_tokens.call_count == 2

        assert index.consume_full_verification_layer("layer.1") == "staging-1"
        assert len(index._full_verification_prefetched) == 1
        assert index._full_verification_prefetched[0].layer_name == "layer.2"
        assert index.cluster_store.submit_full_verification_tokens.call_count == 3
        assert index.consume_full_verification_layer("layer.2") == "staging-2"
        index.end_full_verification_pipeline()

    assert all(not ticket.cancel.called for ticket in tickets)
    index.close()


def test_full_verification_prime_revision_mismatch_is_discarded():
    index = make_index()
    primed_tickets = [Mock(), Mock()]
    replacement_ticket = Mock()
    index.cluster_store.submit_full_verification_tokens = Mock(
        side_effect=(*primed_tickets, replacement_ticket)
    )

    with patch.object(
        index,
        "_get_full_verification_descriptors",
        return_value=(SimpleNamespace(num_tokens=1),),
    ):
        index.begin_proposal(["request"])
        assert index.prime_full_verification_pipeline(
            ["request"],
            {"layer.0": 1, "layer.1": 1},
            torch.device("cuda", 0),
        )
        index.end_proposal()

        index._indices["layer.0"] = {"request": index._empty_index()}
        index.begin_full_verification_pipeline(
            ["request"],
            {"layer.0": 1, "layer.1": 1},
            torch.device("cuda", 0),
        )

        assert len(index._full_verification_prefetched) == 1
        assert index._full_verification_prefetched[0].ticket is replacement_ticket
        index.end_full_verification_pipeline()

    for ticket in primed_tickets:
        ticket.cancel.assert_called_once_with()
    replacement_ticket.cancel.assert_called_once_with()
    index.close()


def test_full_verification_prime_policy_periodically_reprobes():
    index = make_index()
    for _ in range(index._FULL_VERIFY_PRIME_BOOTSTRAP_OUTCOMES):
        index._record_full_verification_prime_outcome(adopted=False)

    decisions = [
        index._should_submit_full_verification_prime()
        for _ in range(index._FULL_VERIFY_PRIME_REPROBE_INTERVAL)
    ]

    assert decisions == [False] * 7 + [True]
    index.close()


def test_remove_request_drains_affected_full_verification_prime():
    index = make_index()
    ticket = Mock()
    index.cluster_store.submit_full_verification_tokens = Mock(return_value=ticket)

    with patch.object(
        index,
        "_get_full_verification_descriptors",
        return_value=(SimpleNamespace(num_tokens=1),),
    ):
        index.begin_proposal(["request"])
        assert index.prime_full_verification_pipeline(
            ["request"], {"layer": 1}, torch.device("cuda", 0)
        )
        index.end_proposal()
        index.remove_requests(["request"])

    ticket.cancel.assert_called_once_with()
    ticket.result.assert_called_once_with()
    assert index._primed_full_verification is None
    index.close()


def test_full_verification_plan_handles_request_without_cluster_pages():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 3, keys, values, block_table)

    index.begin_full_verification_residency(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[3],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    assert plan.exact_token_counts.tolist() == [[3]]
    assert plan.primary_exact_token_indices.tolist() == [[[0, 1, 2]]]
    assert plan.primary_exact_token_mask.all()
    assert plan.clustered_descriptors[0].num_tokens == 0
    assert plan.clustered_kv is None


def test_cpu_offload_sparse_selection_handles_request_without_cluster_pages():
    index = make_index(replay_mode="trace")
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 3, keys, values, block_table)
    record_selection = Mock()
    index.selection_provenance.record_draft_selection = record_selection

    index.begin_proposal(["request"])
    try:
        selection = index.select_segmented(
            request_ids=["request"],
            layer_name="layer",
            query=torch.ones(1, 1, 1),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([3], dtype=torch.int32),
            active_mask=torch.tensor([True]),
            scale=1.0,
            proposal_round=2,
        )
    finally:
        index.end_proposal()

    assert selection.exact_token_counts.tolist() == [[3]]
    primary_indices = selection.plan.primary_exact_token_indices[
        selection.plan.primary_exact_token_mask
    ]
    assert primary_indices.tolist() == [0, 1, 2]
    assert selection.exact_page_ids.shape == (1, 1, 1, 0)
    assert selection.resolved_pages is None
    record_selection.assert_called_once()
    assert record_selection.call_args.kwargs["physical_source"] == "native_only"
    assert record_selection.call_args.kwargs["proposal_round"] == 2


def test_full_verification_plan_rejects_staged_index_updates():
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

    try:
        with pytest.raises(RuntimeError, match="index updates are staged"):
            index.build_full_verification_plan(
                request_ids=["request"],
                layer_name="layer",
                seq_lens=[10],
                key_cache=keys,
                block_table=block_table,
            )
    finally:
        index.discard_staged_updates()


def test_full_verification_plan_allows_another_layer_to_be_staged():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)
    index.build_or_update(
        layer_name="other-layer",
        request_ids=["request"],
        seq_lens=[10],
        is_prefill=[True],
        rows=[0],
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
        defer_cpu_store=True,
    )

    # This unit exercises the layer-local plan guard directly. Production
    # full verification rejects any unflushed transaction at context entry.
    index._gpu_index_residency.activate(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[10],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index._gpu_index_residency.deactivate()
        index.discard_staged_updates()

    assert plan.exact_token_counts.tolist() == [[10]]


def test_prepare_full_verification_rolls_back_uncommitted_clusters():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    assert index.cluster_store.num_allocated_pages("layer") == 2

    index.prepare_full_verification(
        request_ids=["request"],
        context_lens=[5],
        layer_names=["layer"],
    )
    index.begin_full_verification_residency(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[5],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    record = index._indices["layer"]["request"]
    assert record.segments == []
    assert index.cluster_store.num_allocated_pages("layer") == 0
    assert plan.exact_token_counts.tolist() == [[5]]
    assert plan.primary_exact_token_mask.sum().item() == 5


def test_full_verification_plan_accepts_an_empty_context():
    index = make_index()
    keys, _ = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)

    index.begin_full_verification_residency(["request"])
    try:
        plan = index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[0],
            key_cache=keys,
            block_table=block_table,
        )
    finally:
        index.end_full_verification_residency()

    assert plan.exact_token_counts.tolist() == [[0]]
    assert plan.primary_exact_token_indices.shape == (1, 1, 0)
    assert plan.primary_exact_token_mask.shape == (1, 1, 0)
    assert plan.clustered_descriptors[0].num_tokens == 0
    assert plan.clustered_kv is None


def test_full_verification_plan_rejects_unapplied_rollback():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.arange(7, dtype=torch.int32).view(1, -1)
    build_index(index, 10, keys, values, block_table)

    with pytest.raises(RuntimeError, match="rolled-back cluster state"):
        index.build_full_verification_plan(
            request_ids=["request"],
            layer_name="layer",
            seq_lens=[5],
            key_cache=keys,
            block_table=block_table,
        )
