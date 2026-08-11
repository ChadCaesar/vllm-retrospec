# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.index import (
    RetroSpecAttentionLevel,
    RetroSpecBlockIndex,
)


def make_index() -> RetroSpecBlockIndex:
    return RetroSpecBlockIndex(
        block_size=2,
        num_speculative_tokens=1,
        retrieval_ratio=0.34,
        estimation_ratio=0.33,
    )


def make_cache(num_blocks: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.empty(num_blocks, 2, 1, 1)
    values = torch.empty_like(keys)
    for block_id in range(num_blocks):
        keys[block_id].fill_(float(block_id))
        values[block_id].fill_(float(block_id * 10))
    return keys, values


def test_select_builds_ordered_exact_and_estimation_zones():
    index = make_index()
    keys, values = make_cache()

    selection = index.select(
        query=torch.ones(1, 1, 1),
        key_cache=keys,
        value_cache=values,
        block_table=torch.arange(6, dtype=torch.int32).view(1, -1),
        seq_lens=torch.tensor([11], dtype=torch.int32),
        active_mask=torch.tensor([True]),
        scale=1.0,
    )

    # Blocks 0, 4 and 5 are forced exact. Among candidates 1, 2 and 3,
    # blocks 3 and 2 have the highest scores and are retrieved exactly.
    assert selection.exact_block_table.tolist() == [[0, 2, 3, 4, 5, 0]]
    assert selection.exact_seq_lens.tolist() == [9]

    # The remaining candidate block 1 is represented by its summaries.
    assert selection.estimation_token_counts.tolist() == [[2, 0, 0, 0, 0, 0]]
    assert selection.estimation_keys[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert selection.estimation_values[0, 0, 0, 0].item() == pytest.approx(10.0)
    assert 0.0 < selection.hit_attn.item() <= 1.0


def test_expanded_zone_promotes_sparse_estimation_without_growing_coverage():
    index = RetroSpecBlockIndex(
        block_size=2,
        num_speculative_tokens=1,
        retrieval_ratio=0.2,
        estimation_ratio=0.4,
    )
    scores = torch.arange(10, dtype=torch.float32).view(1, -1)
    candidate_mask = torch.ones_like(scores, dtype=torch.bool)

    sparse_exact, sparse_estimation, expanded_exact, expanded_estimation = (
        index._select_zone_masks(scores, candidate_mask)
    )

    assert sparse_exact.sum().item() == 2
    assert sparse_estimation.sum().item() == 4
    assert expanded_exact.sum().item() == 4
    assert expanded_estimation.sum().item() == 2
    assert torch.all(sparse_exact <= expanded_exact)
    assert torch.equal(
        sparse_exact | sparse_estimation,
        expanded_exact | expanded_estimation,
    )
    assert torch.equal(sparse_exact, scores >= 8)
    assert torch.equal(expanded_exact, scores >= 6)


def test_materialize_expanded_plan_preserves_logical_block_order():
    index = make_index()
    keys, values = make_cache()
    block_table = torch.tensor([[5, 3, 1, 4, 2, 0]], dtype=torch.int32)

    sparse = index.select(
        query=torch.ones(1, 1, 1),
        key_cache=keys,
        value_cache=values,
        block_table=block_table,
        seq_lens=torch.tensor([11], dtype=torch.int32),
        active_mask=torch.tensor([True]),
        scale=1.0,
    )
    expanded = index.materialize(
        sparse.plan,
        RetroSpecAttentionLevel.EXPANDED,
        keys,
        values,
        block_table,
    )

    sparse_logical = sparse.plan.sparse_exact_indices[sparse.plan.sparse_exact_mask]
    expanded_logical = expanded.plan.expanded_exact_indices[
        expanded.plan.expanded_exact_mask
    ]
    assert sparse_logical.tolist() == sorted(sparse_logical.tolist())
    assert expanded_logical.tolist() == sorted(expanded_logical.tolist())
    assert set(sparse_logical.tolist()) <= set(expanded_logical.tolist())
    assert expanded.exact_block_table[0, :6].tolist() == [5, 3, 1, 4, 2, 0]
    assert expanded.exact_seq_lens.tolist() == [11]
    assert torch.count_nonzero(expanded.estimation_token_counts) == 0
    assert expanded.attention_mass.item() >= sparse.attention_mass.item()


def test_block_summary_ignores_stale_tokens_after_sequence_end():
    index = make_index()
    keys, _ = make_cache()
    keys[5, 0].fill_(5.0)
    keys[5, 1].fill_(1000.0)

    (
        _,
        valid_block_mask,
        valid_token_counts,
        _,
    ) = index._build_block_layout(
        torch.arange(6, dtype=torch.int32).view(1, -1),
        torch.tensor([11], dtype=torch.int32),
    )
    means = index._compute_block_means(
        keys,
        torch.arange(6, dtype=torch.int32).view(1, -1),
        valid_block_mask,
        valid_token_counts,
    )

    assert means[0, 5, 0, 0].item() == pytest.approx(5.0)


def test_select_keeps_short_context_fully_exact():
    index = make_index()
    keys, values = make_cache()

    selection = index.select(
        query=torch.ones(2, 1, 1),
        key_cache=keys,
        value_cache=values,
        block_table=torch.tensor(
            [[2, 3, 0], [4, 5, 0]],
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor([3, 1], dtype=torch.int32),
        active_mask=torch.tensor([True, False]),
        scale=1.0,
    )

    assert selection.exact_block_table.tolist() == [
        [2, 3, 0],
        [4, 0, 0],
    ]
    assert selection.exact_seq_lens.tolist() == [3, 1]
    assert torch.count_nonzero(selection.estimation_token_counts) == 0
    assert selection.hit_attn.tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("block_size", 0, "block_size"),
        ("num_speculative_tokens", 0, "num_speculative_tokens"),
        ("retrieval_ratio", 0.0, "retrieval_ratio"),
        ("estimation_ratio", -0.1, "estimation_ratio"),
    ],
)
def test_index_rejects_invalid_configuration(field, value, message):
    values = {
        "block_size": 2,
        "num_speculative_tokens": 1,
        "retrieval_ratio": 0.25,
        "estimation_ratio": 0.5,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        RetroSpecBlockIndex(**values)


def test_index_rejects_incompatible_query_and_kv_heads():
    index = make_index()
    keys = torch.zeros(6, 2, 2, 1)
    values = torch.zeros_like(keys)

    with pytest.raises(ValueError, match="query heads"):
        index.select(
            query=torch.ones(1, 3, 1),
            key_cache=keys,
            value_cache=values,
            block_table=torch.arange(6, dtype=torch.int32).view(1, -1),
            seq_lens=torch.tensor([11], dtype=torch.int32),
            active_mask=torch.tensor([True]),
            scale=1.0,
        )
