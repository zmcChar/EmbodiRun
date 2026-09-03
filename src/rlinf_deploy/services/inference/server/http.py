"""Describe the external VVLA HTTP server without importing its runtime code."""

from __future__ import annotations


def http_server_command(
    *,
    policy: str,
    checkpoint: str,
    bind: str,
    port: int,
    device: str | None = None,
    adapter_config: str | None = None,
) -> tuple[str, ...]:
    """Build the documented ``vvla-http-serve`` command-line contract."""

    argv = [
        "vvla-http-serve",
        "--policy",
        policy,
        "--checkpoint",
        checkpoint,
    ]
    if adapter_config is not None:
        argv.extend(("--adapter-config", adapter_config))
    if device is not None:
        argv.extend(("--device", device))
    argv.extend(("--host", bind, "--port", str(port)))
    return tuple(argv)


__all__ = ["http_server_command"]
