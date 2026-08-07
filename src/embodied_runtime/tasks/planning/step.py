"""One instruction selected by a task planner."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied_runtime.types import Metadata

from ._validation import identifier, metadata, optional_text, text


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    instruction: str
    skill: str | None = None
    success_criteria: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", identifier("step_id", self.step_id))
        object.__setattr__(self, "instruction", text("instruction", self.instruction))
        if self.skill is not None:
            object.__setattr__(self, "skill", identifier("skill", self.skill))
        object.__setattr__(
            self,
            "success_criteria",
            optional_text("success_criteria", self.success_criteria),
        )
        object.__setattr__(self, "metadata", metadata(self.metadata))


__all__ = ["PlanStep"]
