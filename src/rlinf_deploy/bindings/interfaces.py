"""Contracts implemented by policy-to-device action bindings."""

from typing import Protocol

from rlinf_deploy.inference import PolicyResult
from rlinf_deploy.robots import RobotAction


class ActionMappingError(RuntimeError):
    """A policy result cannot be represented in a robot action space."""


class PolicyActionMapper(Protocol):
    def map_result(self, result: PolicyResult) -> RobotAction: ...


__all__ = ["ActionMappingError", "PolicyActionMapper"]
