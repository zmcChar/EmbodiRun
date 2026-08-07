"""Structural interface for model-owned execution control flow."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionPlan(Protocol):
    @property
    def kind(self) -> str: ...

    def required_entrypoints(self) -> tuple[str, ...]: ...


__all__ = ["ExecutionPlan"]
