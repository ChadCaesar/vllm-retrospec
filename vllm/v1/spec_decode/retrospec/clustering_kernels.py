# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility alias for the shared RetroSpec clustering kernels."""

import sys as _sys

from .cluster import kernels as _implementation

_sys.modules[__name__] = _implementation
