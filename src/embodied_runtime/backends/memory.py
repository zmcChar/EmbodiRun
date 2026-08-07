"""Memory counters reported by a loaded backend."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryStats:
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    total_bytes: int | None = None
    free_bytes: int | None = None


__all__ = ["MemoryStats"]
