"""Options controlling model compilation for one backend."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied_runtime.types import Metadata


@dataclass(frozen=True, slots=True)
class CompileOptions:
    mode: str = "eager"
    dtype: str | None = None
    dynamic_shapes: bool = False
    options: Metadata = field(default_factory=dict)


__all__ = ["CompileOptions"]
