"""Errors exposed by the Hugging Face Texas Hold'em planner."""


class PlannerInputError(ValueError):
    """The structured planner request is absent or violates its schema."""


class PlannerOutputError(ValueError):
    """The model response is not a valid, grounded planning result."""


class PlannerRuntimeError(RuntimeError):
    """The Hugging Face runtime could not load or generate a response."""


__all__ = ["PlannerInputError", "PlannerOutputError", "PlannerRuntimeError"]
