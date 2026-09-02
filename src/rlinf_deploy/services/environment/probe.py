"""Probe base capabilities on a local or remote deployment node."""

from __future__ import annotations

from dataclasses import dataclass

from ..executor import Command, Executor
from .errors import EnvironmentError


@dataclass(frozen=True, slots=True)
class NodeProbe:
    """Base capabilities discovered before initializing a node."""

    home: str
    platform: str
    machine: str
    python: str | None
    python_version: str | None
    git: str | None
    uv: str | None


def probe_node(executor: Executor, *, require_init_tools: bool = False) -> NodeProbe:
    """Probe connectivity and report optional tools without changing the node."""

    script = """set -u
home=${HOME:-}
platform=$(uname -s 2>/dev/null || true)
machine=$(uname -m 2>/dev/null || true)
python=$(command -v python3 2>/dev/null || true)
git=$(command -v git 2>/dev/null || true)
uv=$(command -v uv 2>/dev/null || true)
if [ -z "$uv" ] && [ -x "$home/.local/bin/uv" ]; then
  uv="$home/.local/bin/uv"
elif [ -z "$uv" ] && [ -x "$home/.cargo/bin/uv" ]; then
  uv="$home/.cargo/bin/uv"
fi
python_version=''
if [ -n "$python" ]; then
  python_version=$("$python" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$home" "$platform" "$machine" "$python" "$python_version" "$git" "$uv"
"""
    result = executor.run(Command(("sh", "-c", script), timeout_s=20.0))
    fields = result.stdout.rstrip("\n").split("\t")
    if len(fields) != 7:
        raise EnvironmentError("node probe returned an invalid response")
    home, platform, machine, python, python_version, git, uv = fields
    if not home.startswith("/") or not platform or not machine:
        raise EnvironmentError("node probe omitted required machine information")
    probe = NodeProbe(
        home=home,
        platform=platform,
        machine=machine,
        python=python or None,
        python_version=python_version or None,
        git=git or None,
        uv=uv or None,
    )
    if require_init_tools:
        missing = [
            name for name in ("python", "git", "uv") if getattr(probe, name) is None
        ]
        if missing:
            raise EnvironmentError(
                "node is missing tools required by init: " + ", ".join(missing)
            )
        if probe.python_version is None:
            raise EnvironmentError("node Python could not report its version")
    return probe


__all__ = ["NodeProbe", "probe_node"]
