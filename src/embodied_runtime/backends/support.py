"""Backend compatibility result for one model and device."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SupportReport:
    supported: bool
    reasons: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()

    @classmethod
    def yes(cls, *capabilities: str) -> "SupportReport":
        return cls(True, capabilities=frozenset(capabilities))

    @classmethod
    def no(cls, *reasons: str) -> "SupportReport":
        return cls(False, reasons=tuple(reasons))


__all__ = ["SupportReport"]
