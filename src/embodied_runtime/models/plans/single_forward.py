"""One-entrypoint model execution plan."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SingleForwardPlan:
    forward: str = "forward"
    kind: str = field(default="single_forward", init=False)

    def __post_init__(self) -> None:
        if not self.forward:
            raise ValueError("single-forward entrypoint must not be empty")

    def required_entrypoints(self) -> tuple[str, ...]:
        return (self.forward,)


__all__ = ["SingleForwardPlan"]
