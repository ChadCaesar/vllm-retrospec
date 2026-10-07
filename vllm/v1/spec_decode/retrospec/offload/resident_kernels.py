# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility entry for resident lookup, draft, and verification launchers."""

import torch as torch

from vllm.triton_utils import triton as triton

from .resident_draft_launchers import (
    _DRAFT_RESOLVE_STATISTIC_COUNT as _DRAFT_RESOLVE_STATISTIC_COUNT,
)
from .resident_draft_launchers import (
    resolve_compact_draft_pages as resolve_compact_draft_pages,
)
from .resident_draft_launchers import (
    resolve_ranked_draft_buckets as resolve_ranked_draft_buckets,
)
from .resident_kernel_impl import (
    _compact_resident_misses_kernel,  # noqa: F401
    _finalize_ranked_compact_draft_attention_kernel,  # noqa: F401
    _find_resident_buckets,  # noqa: F401
    _lookup_resident_handles_kernel,  # noqa: F401
    _map_compact_verification_miss_indices_kernel,  # noqa: F401
    _publish_resident_table_bindings_kernel,  # noqa: F401
    _record_resident_lookup_statistics,  # noqa: F401
    _reset_verification_miss_hash_kernel,  # noqa: F401
    _resident_handle_hash,  # noqa: F401
    _resolve_compact_draft_pages_kernel,  # noqa: F401
    _resolve_compact_verification_pages_vector_kernel,  # noqa: F401
    _resolve_ranked_draft_buckets_kernel,  # noqa: F401
    _scatter_compact_staging_page_ids_kernel,  # noqa: F401
    _scatter_staging_page_ids_kernel,  # noqa: F401
    _update_resident_handles_kernel,  # noqa: F401
)
from .resident_lookup_launchers import (
    compact_resident_misses as compact_resident_misses,
)
from .resident_lookup_launchers import (
    lookup_resident_handles as lookup_resident_handles,
)
from .resident_lookup_launchers import (
    publish_resident_table_bindings as publish_resident_table_bindings,
)
from .resident_lookup_launchers import (
    scatter_staging_page_ids as scatter_staging_page_ids,
)
from .resident_lookup_launchers import (
    update_resident_handles as update_resident_handles,
)
from .resident_verification_launchers import (
    resolve_compact_verification_pages as resolve_compact_verification_pages,
)
from .resident_verification_launchers import (
    scatter_compact_staging_page_ids as scatter_compact_staging_page_ids,
)
