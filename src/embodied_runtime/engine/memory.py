"""Backend-neutral memory admission policy."""

from __future__ import annotations

from dataclasses import dataclass

from embodied_runtime.backends.memory import MemoryStats

from .errors import MemoryBudgetExceededError


@dataclass(frozen=True, slots=True)
class MemoryBudgetPolicy:
    """Decide whether another request may enter an already-loaded session.

    The engine owns the policy (budget and headroom); the backend owns the numbers
    returned by :meth:`BackendSession.memory_stats`.  This prototype deliberately
    does not allocate or free device memory itself.
    """

    maximum_reserved_bytes: int | None = None
    minimum_free_bytes: int = 0

    def check(self, stats: MemoryStats) -> None:
        if (
            self.maximum_reserved_bytes is not None
            and stats.reserved_bytes > self.maximum_reserved_bytes
        ):
            raise MemoryBudgetExceededError(
                "backend reserved memory exceeds engine budget: "
                f"{stats.reserved_bytes} > {self.maximum_reserved_bytes} bytes"
            )
        if stats.free_bytes is not None and stats.free_bytes < self.minimum_free_bytes:
            raise MemoryBudgetExceededError(
                "backend free memory is below engine headroom: "
                f"{stats.free_bytes} < {self.minimum_free_bytes} bytes"
            )
