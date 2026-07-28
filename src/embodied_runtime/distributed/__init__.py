"""Group 2: distributed registration and communication boundaries.

The initial π0.5 slice executes locally. These interfaces reserve the seam for
cloud, edge, and robot runtimes without coupling the local engine to a network
stack.
"""

from .discovery import NodeRole, RuntimeEndpoint
from .routing import EndpointRouter

__all__ = ["EndpointRouter", "NodeRole", "RuntimeEndpoint"]
