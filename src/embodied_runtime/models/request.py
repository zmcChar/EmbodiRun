"""Raw model-facing request before task-specific preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from embodied_runtime.types import Metadata


@dataclass(slots=True)
class RawRequest:
    observation: Mapping[str, Any]
    prompt: str | None = None
    metadata: Metadata = field(default_factory=dict)


__all__ = ["RawRequest"]
