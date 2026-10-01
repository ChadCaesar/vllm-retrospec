# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch

from tests.retrospec.support.segmented_index import (
    build_index,
    make_cache,
    make_index,
)
from vllm.v1.spec_decode.retrospec.index import RetroSpecAttentionLevel
from vllm.v1.spec_decode.retrospec.segmented_index import (
    RetroSpecSegmentedTokenIndex,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_indexed_verification_transaction_prepares_and_releases_layer_pages():
    device = torch.device("cuda")
    index = make_index(cache_ratio=0.5, pin_memory=True)
    index.configure_sparse_prefetch_wave(1)
    keys, values = make_cache()
    keys = keys.to(device=device, dtype=torch.bfloat16)
    values = values.to(device=device, dtype=torch.bfloat16)
    block_table = torch.arange(7, dtype=torch.int32, device=device).view(1, -1)
    build_index(index, 10, keys, values, block_table)
    request_indices = torch.tensor([0], dtype=torch.int64, device=device)
    token_indices = torch.tensor([0], dtype=torch.int64, device=device)

    index.begin_proposal(["request"])
    try:
        draft_selection = index.select_segmented(
            request_ids=["request"],
            layer_name="layer",
            query=torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
            key_cache=keys,
            value_cache=values,
            block_table=block_table,
            seq_lens=torch.tensor([10], dtype=torch.int32, device=device),
            active_mask=torch.tensor([True], device=device),
            scale=1.0,
        )
        draft_selection.resolved_clusters.read_lease.release()
        index.begin_indexed_verification_transaction(
            RetroSpecAttentionLevel.SPARSE,
            ("layer",),
            request_indices,
            token_indices,
        )
        selection, pages = index.consume_indexed_verification_layer(
            "layer", request_indices, token_indices
        )
        assert selection.plan_valid_rows.tolist() == [True]
        assert pages is not None
        assert pages.miss_admission is not None
        index.end_indexed_verification_transaction()
        assert index._indexed_verification_transaction is None
        index.cluster_store._reap_verification_admissions(wait=True)
        assert index.cluster_store.num_resident_pages("layer") > 0
    finally:
        index.end_proposal()
        index.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ready_selected_replay_admits_current_topk_before_draft_attention():
    device = torch.device("cuda")
    index = make_index(
        cache_ratio=0.5,
        pin_memory=True,
        replay_mode="ready_selected",
    )
    keys, values = make_cache()
    keys = keys.to(device=device, dtype=torch.bfloat16)
    values = values.to(device=device, dtype=torch.bfloat16)
    block_table = torch.arange(7, dtype=torch.int32, device=device).view(1, -1)
    build_index(index, 10, keys, values, block_table)
    record_selection = Mock()
    index.selection_provenance.record_draft_selection = record_selection

    index.begin_proposal(["request"])
    try:
        with patch.object(
            index,
            "_resolve_ranked_draft_clusters",
            wraps=index._resolve_ranked_draft_clusters,
        ) as resolve:
            selection = index.select_segmented(
                request_ids=["request"],
                layer_name="layer",
                query=torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
                key_cache=keys,
                value_cache=values,
                block_table=block_table,
                seq_lens=torch.tensor([10], dtype=torch.int32, device=device),
                active_mask=torch.tensor([True], device=device),
                scale=1.0,
                proposal_round=2,
            )
    finally:
        index.end_proposal()

    assert [call.kwargs["emit_misses"] for call in resolve.call_args_list] == [
        True,
        False,
    ]
    assert index.cluster_store.num_resident_pages("layer") == 1
    assert selection.resolved_clusters is not None
    assert selection.resolved_clusters.hit_gate_ready.all()
    assert selection.resolved_clusters.miss_cluster_counts.sum().item() == 0
    assert selection.prefetch_miss_cluster_ids is None
    assert selection.prefetch_miss_positions is None
    assert selection.prefetch_miss_count is None
    assert selection.prefetch_num_groups == 0
    assert selection.prefetch_num_ranks == 0
    assert [call.kwargs["snapshot"] for call in record_selection.call_args_list] == [
        "before_ready",
        "used",
    ]
    assert all(
        call.kwargs["proposal_round"] == 2 for call in record_selection.call_args_list
    )
    index.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_segmented_index_builds_and_selects_on_cuda():
    device = torch.device("cuda")
    index = make_index()
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
        with (
            patch.object(
                index,
                "_select_cluster_zones",
                side_effect=AssertionError("CUDA path constructed packed zones"),
            ),
            patch.object(
                index,
                "_make_reference_plan",
                side_effect=AssertionError("CUDA path used the reference plan"),
            ),
        ):
            selection = index.select_segmented(
                request_ids=["request"],
                layer_name="layer",
                query=torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
                key_cache=keys,
                value_cache=values,
                block_table=block_table,
                seq_lens=torch.tensor([10], dtype=torch.int32, device=device),
                active_mask=torch.tensor([True], device=device),
                scale=1.0,
            )
        torch.cuda.synchronize()
    finally:
        index.end_proposal()

    assert selection.resolved_clusters.cluster_handles.device.type == "cuda"
    assert selection.resolved_clusters.resident_bucket_ids.device.type == "cuda"
    assert selection.exact_token_counts.tolist() == [[6]]
    assert selection.plan.ranked_cluster_indices[0, 0, 0].item() >= 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prefill_segment_size_tokens": 3}, "divisible by block_size"),
        ({"generation_update_interval": 3}, "divisible by block_size"),
        ({"prefill_segment_size_tokens": 4, "blocks_per_cluster": 3}, "divisible"),
        ({"generation_update_interval": 2, "blocks_per_cluster": 2}, "divisible"),
        ({"blocks_per_cluster": 0}, "positive"),
        ({"num_kmeans_iterations": 0}, "positive"),
        ({"max_pending_cluster_builds": 0}, "positive"),
        ({"cpu_page_build_workers": 0}, "positive"),
        ({"full_verify_gather_workers": 0}, "positive"),
    ],
)
def test_segmented_index_rejects_invalid_configuration(kwargs, message):
    values = {
        "block_size": 2,
        "num_speculative_tokens": 1,
        "retrieval_ratio": 0.5,
        "estimation_ratio": 0.5,
        "prefill_segment_size_tokens": 4,
        "generation_update_interval": 2,
        "blocks_per_cluster": 1,
        "num_kmeans_iterations": 2,
        "max_model_len": 64,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        RetroSpecSegmentedTokenIndex(**values)
