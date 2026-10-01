# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.v1.spec_decode.retrospec.legacy.residency.kernels.common import (
    _find_resident_buckets,
    _record_resident_lookup_statistics,
    _resident_handle_hash,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.kernels.draft import (
    _finalize_ranked_compact_draft_attention_kernel,
    _resolve_compact_draft_pages_kernel,
    _resolve_ranked_draft_buckets_kernel,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.kernels.handles import (
    _compact_resident_misses_kernel,
    _lookup_resident_handles_kernel,
    _publish_resident_table_bindings_kernel,
    _scatter_staging_page_ids_kernel,
    _update_resident_handles_kernel,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.kernels.verification import (
    _map_compact_verification_miss_indices_kernel,
    _reset_verification_miss_hash_kernel,
    _resolve_compact_verification_pages_vector_kernel,
    _scatter_compact_staging_page_ids_kernel,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.ops.draft import (
    resolve_compact_draft_pages,
    resolve_ranked_draft_buckets,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.ops.handles import (
    compact_resident_misses,
    lookup_resident_handles,
    publish_resident_table_bindings,
    scatter_staging_page_ids,
    update_resident_handles,
)
from vllm.v1.spec_decode.retrospec.legacy.residency.ops.verification import (
    resolve_compact_verification_pages,
    scatter_compact_staging_page_ids,
)

_DRAFT_RESOLVE_STATISTIC_COUNT = 11

__all__ = [
    "_DRAFT_RESOLVE_STATISTIC_COUNT",
    "_resident_handle_hash",
    "_find_resident_buckets",
    "_record_resident_lookup_statistics",
    "_resolve_compact_draft_pages_kernel",
    "_resolve_ranked_draft_buckets_kernel",
    "_finalize_ranked_compact_draft_attention_kernel",
    "_reset_verification_miss_hash_kernel",
    "_resolve_compact_verification_pages_vector_kernel",
    "_map_compact_verification_miss_indices_kernel",
    "_scatter_compact_staging_page_ids_kernel",
    "_lookup_resident_handles_kernel",
    "_compact_resident_misses_kernel",
    "_scatter_staging_page_ids_kernel",
    "_update_resident_handles_kernel",
    "_publish_resident_table_bindings_kernel",
    "lookup_resident_handles",
    "compact_resident_misses",
    "resolve_compact_draft_pages",
    "resolve_ranked_draft_buckets",
    "resolve_compact_verification_pages",
    "scatter_compact_staging_page_ids",
    "scatter_staging_page_ids",
    "update_resident_handles",
    "publish_resident_table_bindings",
]
