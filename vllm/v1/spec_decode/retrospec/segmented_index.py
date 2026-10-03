# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports; implementation lives in ``offload.segmented_index``."""

from .offload import segmented_index as _implementation
from .offload import segmented_types as _types

for _module in (_types, _implementation):
    for _name, _value in vars(_module).items():
        if not (_name.startswith("__") and _name.endswith("__")):
            globals()[_name] = _value
del _module, _name, _value


def __getattr__(name: str):
    return getattr(_implementation, name)
