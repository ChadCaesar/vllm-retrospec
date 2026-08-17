# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class RetroSpecClusterGroup:
    """One independent cluster-cache domain within an attention layer.

    The layer name is intentionally excluded because cluster page stores and
    resident caches are already scoped by layer.
    """

    request_id: str
    kv_head_index: int

    def __post_init__(self) -> None:
        if self.kv_head_index < 0:
            raise ValueError("kv_head_index must be non-negative")


@dataclass(frozen=True)
class RetroSpecClusterIdentity:
    """Semantic identity of one cluster within a layer.

    local_cluster_id belongs to one request and KV head. It may therefore be
    reused by another request, another KV head, or another layer.
    """

    group: RetroSpecClusterGroup
    local_cluster_id: int

    def __post_init__(self) -> None:
        if self.local_cluster_id < 0:
            raise ValueError("local_cluster_id must be non-negative")
