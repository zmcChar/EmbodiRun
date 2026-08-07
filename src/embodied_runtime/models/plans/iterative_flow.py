"""Encode-integrate-finalize model execution plan."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class IterativeFlowPlan:
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
        steps = self.default_num_steps if num_steps is None else num_steps
        if steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        dt = -1.0 / steps
        return tuple((1.0 + index * dt, dt) for index in range(steps))

    def entrypoint_names(self) -> tuple[str, ...]:
        return self.required_entrypoints()


__all__ = ["IterativeFlowPlan"]
