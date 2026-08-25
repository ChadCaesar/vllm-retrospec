# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm.config import ParallelConfig, SpeculativeConfig


def make_retrospec_config(**overrides: Any) -> SpeculativeConfig:
    values: dict[str, Any] = {
        "method": "retrospec",
        "num_speculative_tokens": 64,
    }
    values.update(overrides)
    return SpeculativeConfig(**values)


def test_retrospec_defaults():
    config = make_retrospec_config()

    assert config.method == "retrospec"
    assert config.model == "retrospec"
    assert config.enforce_eager is True
    assert config.retrospec_retrieval_ratio == pytest.approx(0.018)
    assert config.retrospec_estimation_ratio == pytest.approx(0.232)
    assert config.retrospec_cache_ratio == pytest.approx(0.0)
    assert config.retrospec_index_segment_size == 8192
    assert config.retrospec_blocks_per_cluster == 1
    assert config.retrospec_kmeans_iterations == 10
    assert config.retrospec_max_pending_cluster_builds == 2
    assert config.retrospec_prefill_warmup_multiplier == 4
    assert config.retrospec_index_update_interval == 1024
    assert config.retrospec_min_draft_tokens == 1
    assert config.retrospec_max_draft_tokens == 16
    assert config.retrospec_draft_margin_threshold is None
    assert config.retrospec_sparse_margin_threshold is None
    assert config.retrospec_expanded_margin_threshold is None
    assert config.retrospec_hit_attn_threshold is None
    assert config.retrospec_retrieval_attn_threshold is None
    assert config.retrospec_expanded_attn_threshold is None
    assert config.retrospec_stats_interval_seconds == pytest.approx(0.0)
    assert config.prompt_lookup_min == 0
    assert config.prompt_lookup_max == 0
    assert repr(config) == (
        "SpeculativeConfig(method='retrospec', model=None, num_spec_tokens=64)"
    )


def test_retrospec_requires_num_speculative_tokens():
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        SpeculativeConfig(method="retrospec")


def test_retrospec_draft_range():
    with pytest.raises(
        ValueError,
        match="retrospec_min_draft_tokens",
    ):
        make_retrospec_config(
            retrospec_min_draft_tokens=17,
            retrospec_max_draft_tokens=16,
        )


def test_retrospec_draft_exceeds_pending_limit():
    with pytest.raises(
        ValueError,
        match="retrospec_max_draft_tokens",
    ):
        make_retrospec_config(
            num_speculative_tokens=8,
            retrospec_max_draft_tokens=16,
        )


def test_retrospec_attention_budget():
    with pytest.raises(ValueError, match="must not exceed 1"):
        make_retrospec_config(
            retrospec_retrieval_ratio=0.6,
            retrospec_estimation_ratio=0.5,
        )


def test_retrospec_expanded_budget():
    with pytest.raises(
        ValueError,
        match="estimation_ratio must be greater",
    ):
        make_retrospec_config(
            retrospec_retrieval_ratio=0.2,
            retrospec_estimation_ratio=0.1,
        )


def test_retrospec_requires_eager():
    with pytest.raises(ValueError, match="enforce_eager"):
        make_retrospec_config(enforce_eager=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrospec_retrieval_ratio", 0.0),
        ("retrospec_retrieval_ratio", 1.0),
        ("retrospec_estimation_ratio", -0.01),
        ("retrospec_estimation_ratio", 1.0),
        ("retrospec_cache_ratio", -0.01),
        ("retrospec_cache_ratio", 1.01),
        ("retrospec_index_segment_size", 0),
        ("retrospec_blocks_per_cluster", 0),
        ("retrospec_kmeans_iterations", 0),
        ("retrospec_max_pending_cluster_builds", 0),
        ("retrospec_prefill_warmup_multiplier", 0),
        ("retrospec_index_update_interval", 0),
        ("num_speculative_tokens", 0),
        ("retrospec_min_draft_tokens", 0),
        ("retrospec_max_draft_tokens", 0),
        ("retrospec_draft_margin_threshold", -0.01),
        ("retrospec_sparse_margin_threshold", -0.01),
        ("retrospec_expanded_margin_threshold", -0.01),
        ("retrospec_hit_attn_threshold", -0.01),
        ("retrospec_hit_attn_threshold", 1.01),
        ("retrospec_retrieval_attn_threshold", -0.01),
        ("retrospec_retrieval_attn_threshold", 1.01),
        ("retrospec_expanded_attn_threshold", -0.01),
        ("retrospec_expanded_attn_threshold", 1.01),
        ("retrospec_stats_interval_seconds", -0.01),
    ],
)
def test_retrospec_rejects_out_of_range_values(field: str, value: Any):
    with pytest.raises(ValueError, match=field):
        make_retrospec_config(**{field: value})


@pytest.mark.parametrize(
    "parallel_config",
    [
        ParallelConfig(tensor_parallel_size=2),
        ParallelConfig(pipeline_parallel_size=2),
    ],
)
def test_retrospec_rejects_parallel_execution(
    parallel_config: ParallelConfig,
):
    with pytest.raises(ValueError, match="supports only"):
        make_retrospec_config(target_parallel_config=parallel_config)


def test_retrospec_clears_prompt_lookup_fields():
    config = make_retrospec_config(
        prompt_lookup_min=2,
        prompt_lookup_max=4,
    )

    assert config.prompt_lookup_min == 0
    assert config.prompt_lookup_max == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrospec_retrieval_ratio", 0.02),
        ("retrospec_estimation_ratio", 0.25),
        ("retrospec_cache_ratio", 0.1),
        ("retrospec_index_segment_size", 2048),
        ("retrospec_blocks_per_cluster", 8),
        ("retrospec_kmeans_iterations", 5),
        ("retrospec_index_update_interval", 2048),
        ("retrospec_min_draft_tokens", 2),
        ("retrospec_max_draft_tokens", 20),
        ("num_speculative_tokens", 80),
    ],
)
def test_retrospec_hash_tracks_execution_structure(field: str, value: Any):
    base_hash = make_retrospec_config().compute_hash()
    changed_hash = make_retrospec_config(**{field: value}).compute_hash()

    assert changed_hash != base_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrospec_draft_margin_threshold", 0.1),
        ("retrospec_sparse_margin_threshold", 0.1),
        ("retrospec_expanded_margin_threshold", 0.1),
        ("retrospec_hit_attn_threshold", 0.1),
        ("retrospec_retrieval_attn_threshold", 0.1),
        ("retrospec_expanded_attn_threshold", 0.1),
        ("retrospec_max_pending_cluster_builds", 4),
        ("retrospec_prefill_warmup_multiplier", 8),
        ("retrospec_stats_interval_seconds", 5.0),
    ],
)
def test_retrospec_hash_ignores_runtime_only_fields(field: str, value: Any):
    base_hash = make_retrospec_config().compute_hash()
    changed_hash = make_retrospec_config(**{field: value}).compute_hash()

    assert changed_hash == base_hash
