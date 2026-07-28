"""Configuration for the hardware- and model-neutral execution engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Host-side scheduling and lifecycle policy.

    Dynamic batching is enabled only when ``max_batch_size`` is greater than one
    *and* the engine is constructed with a model-owned ``batcher`` callback.
    Keeping collation outside this configuration prevents the engine from
    knowing about a tensor library or a model family.
    """

    max_queue_size: int = 128
    max_batch_size: int = 1
    max_wait_ms: float = 0.0
    memory_budget_bytes: int | None = None
    minimum_free_bytes: int = 0

    def __post_init__(self) -> None:
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than zero")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be greater than zero")
        if self.minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes cannot be negative")
