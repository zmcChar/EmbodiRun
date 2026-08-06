"""Remote installation and lifecycle management for the Go2 edge agent."""

from .cli import main
from .deployer import Go2AgentDeployer, ServiceName, StartOptions
from .transport import (
    CommandResult,
    ParamikoTransport,
    RemoteCommandError,
    RemoteTransport,
    SshConnection,
)

__all__ = [
    "CommandResult",
    "Go2AgentDeployer",
    "ParamikoTransport",
    "RemoteCommandError",
    "RemoteTransport",
    "ServiceName",
    "SshConnection",
    "StartOptions",
    "main",
]
