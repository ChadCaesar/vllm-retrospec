# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public import and serialization contracts for relocated offload classes."""

import importlib
import pickle

import pytest


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("cluster_store", "RetroSpecClusterPageStore"),
        ("segmented_index", "RetroSpecSegmentedTokenIndex"),
        ("resident_cache", "RetroSpecResidentClusterCache"),
        ("index_residency", "RetroSpecGPUIndexResidencyManager"),
        ("execution", "RetroSpecExactAttentionWorkspace"),
        ("cluster_store", "RetroSpecClusterBlockMetadata"),
        ("segmented_index", "RetroSpecTokenAttentionSelection"),
    ],
)
def test_offload_class_keeps_original_import_and_pickle_path(
    module_name: str, class_name: str
):
    base = "vllm.v1.spec_decode.retrospec"
    original_module = importlib.import_module(f"{base}.{module_name}")
    implementation_module = importlib.import_module(f"{base}.offload.{module_name}")
    original_class = getattr(original_module, class_name)
    implementation_class = getattr(implementation_module, class_name)

    assert original_class is implementation_class
    assert original_class.__module__ == original_module.__name__
    assert pickle.loads(pickle.dumps(original_class)) is original_class
