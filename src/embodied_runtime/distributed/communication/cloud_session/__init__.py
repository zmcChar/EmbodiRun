"""Robot-session-aware endpoint over the prototype TCP/JSON transport."""

from .endpoint import MultiTenantTcpEndpoint
from .errors import CloudSessionProtocolError, CloudSessionRemoteError
from .normalization import json_compatible
from .protocol import MULTI_ROBOT_PROTOCOL_VERSION

__all__ = [
    "MULTI_ROBOT_PROTOCOL_VERSION",
    "CloudSessionProtocolError",
    "CloudSessionRemoteError",
    "MultiTenantTcpEndpoint",
    "json_compatible",
]
