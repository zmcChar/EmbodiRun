"""Device memory inspection for the PyTorch backend."""

from __future__ import annotations

import os
from typing import Any

from ..memory import MemoryStats


def _cpu_memory() -> tuple[int | None, int | None]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES")
        available = page_size * os.sysconf("SC_AVPHYS_PAGES")
        return int(total), int(available)
    except (AttributeError, OSError, ValueError):
        return None, None


def memory_stats(torch: Any, device_id: str) -> MemoryStats:
    if device_id == "cpu":
        total, free = _cpu_memory()
        # PyTorch does not expose its CPU allocator accounting as a stable API.
        return MemoryStats(total_bytes=total, free_bytes=free)

    with torch.cuda.device(torch.device(device_id)):
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        try:
            free, total = torch.cuda.mem_get_info()
        except (RuntimeError, NotImplementedError):
            free = total = None
    return MemoryStats(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        total_bytes=None if total is None else int(total),
        free_bytes=None if free is None else int(free),
    )
