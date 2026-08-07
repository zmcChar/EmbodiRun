"""OpenPI-compatible WebSocket endpoint for remote robot-policy serving.

Wire dependencies remain optional and load only when the default codec,
connection factory, or action validator is first used.
"""

from .actions import validate_openpi_actions
from .connection import _websocket_connect_options as _websocket_connect_options
from .endpoint import OpenPIWebSocketEndpoint, OpenPiWebSocketEndpoint
from .errors import (
    OpenPIDependencyError,
    OpenPIError,
    OpenPIProtocolError,
    OpenPIRemoteError,
    OpenPITimeoutError,
    OpenPIUnavailableError,
)
from .protocols import (
    OpenPIConnectionFactory,
    OpenPIMessageCodec,
    OpenPIWebSocketConnection,
)

__all__ = [
    "OpenPIConnectionFactory",
    "OpenPIDependencyError",
    "OpenPIError",
    "OpenPIMessageCodec",
    "OpenPIProtocolError",
    "OpenPIRemoteError",
    "OpenPITimeoutError",
    "OpenPIUnavailableError",
    "OpenPIWebSocketConnection",
    "OpenPIWebSocketEndpoint",
    "OpenPiWebSocketEndpoint",
    "validate_openpi_actions",
]
