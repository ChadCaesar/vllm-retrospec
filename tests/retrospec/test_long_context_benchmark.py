# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock

from benchmarks.retrospec.benchmark_long_context import (
    build_speculative_config,
    shutdown_llm,
)
from benchmarks.retrospec.summarize_long_context import summarize


def test_benchmark_profile_controls_observation_interval():
    common = {
        "num_speculative_tokens": 64,
        "retrieval_ratio": 0.018,
        "estimation_ratio": 0.232,
        "cache_ratio": 0.0,
        "index_segment_size": 8192,
        "index_update_interval": 1024,
        "prefill_tile_size": 8192,
        "blocks_per_cluster": 1,
        "kmeans_iterations": 10,
        "cpu_page_build_workers": 4,
        "full_verify_gather_workers": 4,
        "max_pinned_memory": 1.0,
        "max_gpu_index_memory": 4.0,
        "min_draft_tokens": 1,
        "max_draft_tokens": 16,
        "stats_interval": 2.0,
        "profile_level": "detailed",
        "profile_sample_interval": 4,
        "graph": True,
        "sync_attention_gates": False,
    }

    quiet = build_speculative_config(SimpleNamespace(**common, profile=False))
    profile = build_speculative_config(SimpleNamespace(**common, profile=True))

    assert quiet["retrospec_stats_interval_seconds"] == 0.0
    assert profile["retrospec_stats_interval_seconds"] == 2.0
    assert profile["retrospec_stats_cuda_timing_level"] == "detailed"
    assert profile["retrospec_stats_cuda_sample_interval"] == 4
    assert profile["enforce_eager"] is False


def test_benchmark_shutdown_stops_engine_core():
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=Mock()))

    shutdown_llm(llm)

    llm.llm_engine.engine_core.shutdown.assert_called_once_with()


def test_profile_summary_accumulates_intervals_by_rank(tmp_path):
    result = {"actual_prompt_tokens": 29956, "generated_tokens": 96}
    log = tmp_path / "profile.log"
    log.write_text(
        "\n".join(
            (
                "(Worker_TP1 pid=1) RetroSpec performance over 1.00s "
                "(reason=interval): counters={draft_tokens=2}; peaks={none}; "
                "histograms={draft_to_sparse_tokens=[2:1], "
                "sparse_to_expanded_prefix=[]}; derived={x=0}; "
                "cpu_avg={proposal_wall=2.000ms/2}; "
                "cuda_avg={draft_model=3.000ms/4}",
                "(Worker_TP1 pid=1) RetroSpec performance over 0.50s "
                "(reason=request_finished): counters={draft_tokens=3}; "
                "peaks={none}; histograms={draft_to_sparse_tokens=[2:2,4:1]}; "
                "derived={x=0}; cpu_avg={proposal_wall=5.000ms/1}; "
                "cuda_avg={draft_model=1.000ms/2}",
                "RETROSPEC_BENCHMARK_RESULT=" + json.dumps(result),
            )
        ),
        encoding="utf-8",
    )

    output = summarize(log)

    rank = output["ranks"]["1"]
    assert output["result"] == result
    assert rank["counters"]["draft_tokens"] == 5
    assert rank["timings"]["proposal_wall"] == {
        "count": 3,
        "total_ms": 9.0,
        "average_ms": 3.0,
    }
    assert rank["timings"]["draft_model"]["total_ms"] == 14.0
    assert rank["histograms"]["draft_to_sparse_tokens"] == {"2": 3, "4": 1}
    assert rank["flush_reasons"] == {"interval": 1, "request_finished": 1}
