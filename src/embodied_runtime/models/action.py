"""Model-native action chunks before task or robot adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied_runtime.types import Metadata, TensorTree


@dataclass(slots=True)
class ActionChunk:
    actions: TensorTree
    request_id: str | None = None
    metadata: Metadata = field(default_factory=dict)


__all__ = ["ActionChunk"]
