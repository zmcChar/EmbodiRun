"""Small standard-library compatibility helpers for supported Python versions."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised on Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return str(self.value)

        @staticmethod
        def _generate_next_value_(
            name: str,
            start: int,
            count: int,
            last_values: list[object],
        ) -> str:
            del start, count, last_values
            return name.lower()


__all__ = ["StrEnum"]
