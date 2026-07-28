"""Backend-neutral descriptions of model execution control flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionPlan(Protocol):
    """Control-flow contract supplied by a model package and run by the engine."""

    @property
    def kind(self) -> str: ...

    def required_entrypoints(self) -> tuple[str, ...]:
        """Return every model entrypoint needed to execute this plan."""
        ...


@dataclass(frozen=True, slots=True)
class SingleForwardPlan:
    """Execute one model entrypoint exactly once."""

    forward: str = "forward"
    kind: str = field(default="single_forward", init=False)

    def __post_init__(self) -> None:
        if not self.forward:
            raise ValueError("single-forward entrypoint must not be empty")

    def required_entrypoints(self) -> tuple[str, ...]:
        return (self.forward,)


@dataclass(frozen=True, slots=True)
class IterativeFlowPlan:
    """Encode once, integrate a flow field, and finalize the resulting state."""

    encode: str = "encode_prefix"
    initialize: str = "init_state"
    step: str = "denoise_step"
    finalize: str = "finalize"
    default_num_steps: int = 10
    safe_point_after_step: bool = True
    step_returns_state: bool = False
    kind: str = field(default="iterative_flow", init=False)

    def __post_init__(self) -> None:
        names = self.required_entrypoints()
        if any(not name for name in names):
            raise ValueError("iterative-flow entrypoints must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("iterative-flow entrypoints must be distinct")
        if self.default_num_steps <= 0:
            raise ValueError("default_num_steps must be greater than zero")

    def required_entrypoints(self) -> tuple[str, ...]:
        return self.encode, self.initialize, self.step, self.finalize

    def schedule(self, num_steps: int | None = None) -> tuple[tuple[float, float], ...]:
        """Return Euler integration points from normalized time one to zero."""

        steps = self.default_num_steps if num_steps is None else num_steps
        if steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        dt = -1.0 / steps
        return tuple((1.0 + index * dt, dt) for index in range(steps))

    def entrypoint_names(self) -> tuple[str, ...]:
        """Compatibility spelling; new code should use ``required_entrypoints``."""

        return self.required_entrypoints()


# Temporary source-compatibility alias. ModelPackage itself has migrated to
# ``plan=`` and new code should name IterativeFlowPlan explicitly.
FlowRecipe = IterativeFlowPlan
