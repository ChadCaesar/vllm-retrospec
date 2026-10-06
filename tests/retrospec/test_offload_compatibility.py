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
        ("resident_cache", "RetroSpecResidentPageAccess"),
        ("resident_cache", "RetroSpecPreparedResidentAdmission"),
        ("resident_cache", "RetroSpecResidentLruCapture"),
        ("resident_cache", "RetroSpecResolvedResidentLru"),
        ("resident_cache", "RetroSpecCompactResidentPageAccess"),
        ("resident_cache", "RetroSpecRankedDraftResidentAccess"),
        ("resident_cache", "RetroSpecCompactVerificationPageAccess"),
        ("resident_cache", "RetroSpecResidentReadLease"),
        ("index_residency", "RetroSpecGPUIndexResidencyManager"),
        ("index_residency", "RetroSpecClusterSummary"),
        ("index_residency", "RetroSpecStagedClusterSummary"),
        ("index_residency", "RetroSpecResidentSegment"),
        ("index_residency", "RetroSpecResidentTableBinding"),
        ("index_residency", "RetroSpecResidentLayerArena"),
        ("index_residency", "RetroSpecResidentBatchView"),
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


@pytest.mark.parametrize(
    "class_name",
    [
        "_PinnedStagingSlot",
        "RetroSpecResidentPrefetchInput",
        "_PinnedSelectionSlot",
        "_PinnedVerificationMissSlot",
        "_VerificationResolveGPUArena",
        "_CPUPageSlab",
        "_FullVerificationSourceSnapshot",
        "_PinnedPageTransferSlot",
        "RetroSpecCompactTokenRange",
        "RetroSpecFullVerificationDescriptor",
        "RetroSpecFullVerificationStaging",
        "RetroSpecFullVerificationTicket",
        "RetroSpecResolvedClusterPages",
        "RetroSpecCompactResolvedClusterPages",
        "RetroSpecRankedDraftResolvedClusters",
        "RetroSpecCompactVerificationResolvedPages",
        "RetroSpecVerificationMissAdmission",
        "RetroSpecVerificationResolveRequest",
        "_SubmittedVerificationResolve",
    ],
)
def test_cluster_store_support_type_keeps_legacy_identity(class_name: str) -> None:
    base = "vllm.v1.spec_decode.retrospec"
    legacy = importlib.import_module(f"{base}.cluster_store")
    support = importlib.import_module(f"{base}.offload.cluster_store_support")
    staging_names = {
        "_PinnedStagingSlot",
        "RetroSpecResidentPrefetchInput",
        "_PinnedSelectionSlot",
        "_PinnedVerificationMissSlot",
        "_VerificationResolveGPUArena",
        "_CPUPageSlab",
        "_FullVerificationSourceSnapshot",
        "_PinnedPageTransferSlot",
    }
    implementation = (
        "cluster_staging_types"
        if class_name in staging_names
        else "cluster_verification_types"
    )
    relocated = importlib.import_module(f"{base}.offload.{implementation}")
    cls = getattr(legacy, class_name)

    assert getattr(support, class_name) is cls
    assert getattr(relocated, class_name) is cls
    assert cls.__module__ == legacy.__name__
    assert pickle.loads(pickle.dumps(cls)) is cls


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("resident_kernels", "lookup_resident_handles"),
        ("resident_kernels", "resolve_compact_draft_pages"),
        ("resident_kernels", "resolve_compact_verification_pages"),
    ],
)
def test_resident_kernel_entry_keeps_original_import_path(
    module_name: str, function_name: str
):
    base = "vllm.v1.spec_decode.retrospec"
    original_module = importlib.import_module(f"{base}.{module_name}")
    implementation_module = importlib.import_module(f"{base}.offload.{module_name}")
    assert getattr(original_module, function_name) is getattr(
        implementation_module, function_name
    )
