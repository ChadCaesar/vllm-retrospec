# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scoped placement for host KV work; never changes process-wide policy."""

import ctypes
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _numa_library():
    try:
        # These are short thread-local syscalls. CDLL releases the GIL around
        # each call, which can delay its reacquisition by a whole interpreter
        # switch interval while the GPU launch thread is active.
        return ctypes.PyDLL("libnuma.so.1", use_errno=True)
    except OSError:
        return None


def gpu_numa_node(device_index: int) -> int | None:
    """Resolve the visible CUDA ordinal through its UUID, including reordering."""
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import import_pynvml

    try:
        uuid = current_platform.get_device_uuid(device_index)
        nvml = import_pynvml()
        nvml.nvmlInit()
        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
            bus = nvml.nvmlDeviceGetPciInfo(handle).busId
            if isinstance(bus, bytes):
                bus = bus.decode()
            domain, bus, slot = bus.lower().split(":")
            address = f"{int(domain, 16):04x}:{bus}:{slot}"
            node = int(Path(f"/sys/bus/pci/devices/{address}/numa_node").read_text())
            return node if node >= 0 else None
        finally:
            nvml.nvmlShutdown()
    except Exception:
        logger.debug("Could not resolve RetroSpec GPU NUMA node", exc_info=True)
        return None


class RetroSpecHostPlacement:
    def __init__(self, cpus: tuple[int, ...] = (), node: int | None = None):
        self.cpus = cpus
        self.node = node

    def initialize_worker(self):
        """Dedicated workers keep their placement until the thread exits."""
        if self.cpus:
            try:
                os.sched_setaffinity(0, self.cpus)
            except OSError:
                logger.warning_once("Could not place RetroSpec host KV thread")
        if self.node is not None and 0 <= self.node < 1024:
            library = _numa_library()
            if library is not None:
                mask = (ctypes.c_ulong * 16)()
                bits = ctypes.sizeof(ctypes.c_ulong) * 8
                mask[self.node // bits] = 1 << (self.node % bits)
                library.set_mempolicy(1, mask, 1024)

    def bind_memory(self, tensor) -> bool:
        """Bind new shared KV pages without touching or migrating the payload.

        Unlike a temporary thread policy, the shared-memory VMA policy also
        protects subsequent sessions from remote writers' automatic migration.
        Call only for a newly allocated, page-aligned complete mapping.
        """
        if self.node is None or not 0 <= self.node < 1024:
            return False
        library = _numa_library()
        if library is None:
            return False
        mask = (ctypes.c_ulong * 16)()
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask[self.node // bits] = 1 << (self.node % bits)
        return (
            library.mbind(
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_ulong(tensor.numel() * tensor.element_size()),
                2,  # MPOL_BIND
                mask,
                1024,
                0,  # No MPOL_MF_MOVE: no extra copy or prefault/clear pass.
            )
            == 0
        )

    @contextmanager
    def bind(self):
        """Native gather threads inherit this calling thread's placement.

        Preferred memory policy affects new pages only, without migrating or
        clearing reusable slabs. Restore both policies even if a job fails.
        Containers without mempolicy support still use CPU first touch.
        """
        if not self.cpus and self.node is None:
            yield
            return
        original = None
        changed = False
        library = None
        mode = ctypes.c_int()
        mask = (ctypes.c_ulong * 16)()
        try:
            if self.cpus:
                try:
                    original = os.sched_getaffinity(0)
                    if original == set(self.cpus):
                        original = None
                    else:
                        os.sched_setaffinity(0, self.cpus)
                except OSError:
                    logger.warning_once("Could not place RetroSpec host KV thread")
            if self.node is not None and 0 <= self.node < 1024:
                library = _numa_library()
                if (
                    library is not None
                    and library.get_mempolicy(ctypes.byref(mode), mask, 1024, None, 0)
                    == 0
                ):
                    preferred = (ctypes.c_ulong * 16)()
                    bits = ctypes.sizeof(ctypes.c_ulong) * 8
                    preferred[self.node // bits] = 1 << (self.node % bits)
                    changed = library.set_mempolicy(1, preferred, 1024) == 0
            yield
        finally:
            if changed:
                library.set_mempolicy(mode.value, mask, 1024)
            if original is not None:
                os.sched_setaffinity(0, original)

    def run(self, function, *args, **kwargs):
        with self.bind():
            return function(*args, **kwargs)
