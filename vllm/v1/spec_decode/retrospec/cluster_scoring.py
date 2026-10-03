# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports; implementation lives in ``offload.cluster_scoring``."""

from .offload import cluster_scoring as _implementation

for _name, _value in vars(_implementation).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value
del _name, _value


def __getattr__(name: str):
    return getattr(_implementation, name)
