"""Describe the external VVLA WirelessComm server process and its CLI boundary."""

from __future__ import annotations


def wireless_server_command(
    *,
    policy: str,
    checkpoint: str,
    comm_config: str,
    device: str | None = None,
    adapter_config: str | None = None,
) -> tuple[str, ...]:
    """Build the documented ``vvla-wireless-serve`` command-line contract."""

    argv = [
        "vvla-wireless-serve",
        "--policy",
        policy,
        "--checkpoint",
        checkpoint,
    ]
    if adapter_config is not None:
        argv.extend(("--adapter-config", adapter_config))
    if device is not None:
        argv.extend(("--device", device))
    argv.extend(("--comm-config", comm_config))
    return tuple(argv)


__all__ = ["wireless_server_command"]
