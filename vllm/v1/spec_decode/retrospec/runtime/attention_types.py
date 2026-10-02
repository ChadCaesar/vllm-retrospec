# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import IntEnum


class RetroSpecAttentionMode(IntEnum):
    PASSTHROUGH = 0
    DRAFT = 1
    SPARSE_VERIFY = 2
    EXPANDED_VERIFY = 3
    FULL_VERIFY = 4


RetroSpecAttentionMode.__module__ = "vllm.v1.spec_decode.retrospec.attention"
