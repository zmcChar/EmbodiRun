"""Launch descriptors for supported external inference servers."""

from .http import http_server_command
from .sglang import sglang_server_command
from .wireless import wireless_server_command

__all__ = [
    "http_server_command",
    "sglang_server_command",
    "wireless_server_command",
]
