"""Group 2: distributed registration and communication boundaries.

The initial π0.5 slice executes locally. These interfaces reserve the seam for
cloud, edge, and robot runtimes without coupling the local engine to a network
stack.
"""

from .discovery import NodeRole, RuntimeEndpoint
from .failover import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FailoverDecision,
    FailoverMode,
    FallbackReason,
    ResultFuser,
    ResultSource,
)
from .planning import AsyncPlanCoordinator, AsyncPlannerEndpoint, PlanManager
from .routing import EndpointRouter
from .session import RobotSessionIdentity

__all__ = [
    "AsyncFailoverCoordinator",
    "AsyncPlanCoordinator",
    "AsyncPlannerEndpoint",
    "EndpointRouter",
    "FailoverConfig",
    "FailoverDecision",
    "FailoverMode",
    "FallbackReason",
    "NodeRole",
    "PlanManager",
    "ResultFuser",
    "ResultSource",
    "RobotSessionIdentity",
    "RuntimeEndpoint",
]
