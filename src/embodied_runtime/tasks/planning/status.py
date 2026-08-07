"""Lifecycle state for one task-plan step."""

from embodied_runtime._compat import StrEnum


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


__all__ = ["PlanStepStatus"]
