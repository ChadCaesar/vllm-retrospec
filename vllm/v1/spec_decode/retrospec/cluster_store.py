# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports; implementation lives in ``offload.cluster_store``."""

from .offload import cluster_store as _implementation
from .offload import cluster_store_support as _support
from .offload import page_pool as _page_pool
from .offload import verification_transfer as _transfer

for _module in (_support, _page_pool, _transfer, _implementation):
    for _name, _value in vars(_module).items():
        if not (_name.startswith("__") and _name.endswith("__")):
            globals()[_name] = _value
del _module, _name, _value


def __getattr__(name: str):
    return getattr(_implementation, name)
