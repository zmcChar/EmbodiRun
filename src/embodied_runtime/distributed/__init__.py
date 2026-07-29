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
from .routing import EndpointRouter
from .session import RobotSessionIdentity

__all__ = [
    "AsyncFailoverCoordinator",
    "EndpointRouter",
    "FailoverConfig",
    "FailoverDecision",
    "FailoverMode",
    "FallbackReason",
    "NodeRole",
    "ResultFuser",
    "ResultSource",
    "RobotSessionIdentity",
    "RuntimeEndpoint",
]
