# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.retrospec.prefill import (
    RetroSpecLayerPrefillTilePlanner,
)


def make_planner(
    *,
    target_tile_tokens: int = 8192,
    minimum_tile_tokens: int = 2048,
    activation_bytes_per_token: int = 1024,
) -> RetroSpecLayerPrefillTilePlanner:
    return RetroSpecLayerPrefillTilePlanner(
        device=torch.device("cuda:0"),
        block_size=16,
        target_tile_tokens=target_tile_tokens,
        minimum_tile_tokens=minimum_tile_tokens,
        activation_bytes_per_token=activation_bytes_per_token,
    )


def set_cuda_memory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_memory: int,
    total_memory: int = 8 << 30,
    reserved_memory: int = 0,
    allocated_memory: int = 0,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (free_memory, total_memory),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: reserved_memory,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: allocated_memory,
    )


def test_prefill_tile_planner_builds_block_aligned_doubling_buckets():
    planner = make_planner(
        target_tile_tokens=10000,
        minimum_tile_tokens=2050,
    )

    assert planner.target_tile_tokens == 10000
    assert planner.minimum_tile_tokens == 2064
    assert planner._candidate_sizes(12000) == (2064, 4128, 8256, 10000)
    assert planner._candidate_sizes(3000) == (2064, 3000)


def test_prefill_tile_planner_selects_largest_bucket_with_headroom(monkeypatch):
    set_cuda_memory(
        monkeypatch,
        free_memory=(1 << 30) + (6 << 20),
    )
    planner = make_planner()

    selection = planner.select(120000)

    assert selection.tile_size == 4096
    assert selection.available_memory_bytes == (1 << 30) + (6 << 20)
    assert selection.reserve_memory_bytes == 1 << 30
    assert selection.estimated_activation_bytes == 4 << 20


def test_prefill_tile_planner_counts_reusable_allocator_slack(monkeypatch):
    set_cuda_memory(
        monkeypatch,
        free_memory=1 << 30,
        reserved_memory=16 << 20,
        allocated_memory=4 << 20,
    )
    planner = make_planner()

    selection = planner.select(120000)

    assert selection.tile_size == 8192
    assert selection.available_memory_bytes == (1 << 30) + (12 << 20)


def test_prefill_tile_planner_keeps_known_safe_minimum_without_headroom(
    monkeypatch,
):
    set_cuda_memory(monkeypatch, free_memory=512 << 20)
    planner = make_planner()

    selection = planner.select(120000)

    assert selection.tile_size == 2048


@pytest.mark.parametrize("prompt_num_tokens", [0, -1])
def test_prefill_tile_planner_rejects_invalid_prompt_length(
    monkeypatch,
    prompt_num_tokens: int,
):
    set_cuda_memory(monkeypatch, free_memory=2 << 30)
    planner = make_planner()

    with pytest.raises(ValueError, match="prompt length must be positive"):
        planner.select(prompt_num_tokens)
