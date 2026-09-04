"""Describe an external SGLang action server process."""

from __future__ import annotations

from collections.abc import Sequence


def sglang_server_command(
    *,
    checkpoint: str,
    bind: str,
    port: int,
    pipeline: str | None = None,
    pipeline_config: str | None = None,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build the documented ``sglang serve`` VLA command-line contract."""

    argv = [
        "sglang",
        "serve",
        checkpoint,
        "--model-type",
        "diffusion",
    ]
    if pipeline is not None:
        argv.extend(("--pipeline", pipeline))
    if pipeline_config is not None:
        argv.extend(("--pipeline-config-path", pipeline_config))
    argv.extend(extra_args)
    argv.extend(("--host", bind, "--port", str(port)))
    return tuple(argv)


__all__ = ["sglang_server_command"]
