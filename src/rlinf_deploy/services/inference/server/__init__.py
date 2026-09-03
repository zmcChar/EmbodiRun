"""Launch descriptors for inference servers implemented by RLinf Inference."""

from .http import http_server_command
from .wireless import wireless_server_command

__all__ = ["http_server_command", "wireless_server_command"]
