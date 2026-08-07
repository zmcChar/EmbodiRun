"""Message envelope exchanged by distributed runtime nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied_runtime.types import Metadata, TensorTree


@dataclass(slots=True)
class Envelope:
    kind: str
    sender: str
    recipient: str | None
    payload: TensorTree
    correlation_id: str | None = None
    metadata: Metadata = field(default_factory=dict)


__all__ = ["Envelope"]
