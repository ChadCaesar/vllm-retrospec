# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility alias for the earlier RetroSpec implementation."""

import sys as _sys

from .legacy import cluster_scoring as _implementation

_sys.modules[__name__] = _implementation
