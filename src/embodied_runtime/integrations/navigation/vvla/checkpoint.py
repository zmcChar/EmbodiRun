"""Immutable local source/checkpoint verification for the ActiveVLN profile."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from embodied_runtime.policies.navigation.errors import NavigationPolicyError


def _verifier(vvla_root: str | Path) -> Path:
    verifier = (
        Path(vvla_root).expanduser().resolve() / "scripts" / "activevln" / "verify_source_pin.py"
    )
    if not verifier.is_file():
        raise NavigationPolicyError(f"VVLA ActiveVLN verifier is missing: {verifier}")
    return verifier


def _run_verifier(vvla_root: str | Path, flag: str, root: Path, subject: str) -> None:
    result = subprocess.run(
        [sys.executable, str(_verifier(vvla_root)), flag, str(root.resolve())],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown verification error"
        raise NavigationPolicyError(f"ActiveVLN {subject} verification failed: {detail}")


def verify_activevln_source(
    vvla_root: str | Path,
    source_root: str | Path,
) -> None:
    """Verify a local ActiveVLN checkout against VVLA's commit and blob lock."""

    source_path = Path(source_root).expanduser()
    if not source_path.is_dir():
        raise NavigationPolicyError(f"ActiveVLN source is not a directory: {source_path}")
    _run_verifier(vvla_root, "--source-root", source_path, "source")


def verify_activevln_checkpoint(
    vvla_root: str | Path,
    checkpoint: str | Path,
) -> None:
    """Hash a local snapshot with VVLA's lock; remote immutable IDs pass through."""

    checkpoint_text = str(checkpoint)
    checkpoint_path = Path(checkpoint_text).expanduser()
    if not checkpoint_path.exists():
        if checkpoint_path.is_absolute() or checkpoint_text.startswith(("./", "../", "~")):
            raise NavigationPolicyError(
                f"ActiveVLN checkpoint directory does not exist: {checkpoint_path}"
            )
        return
    if not checkpoint_path.is_dir():
        raise NavigationPolicyError(f"ActiveVLN checkpoint is not a directory: {checkpoint_path}")
    _run_verifier(vvla_root, "--checkpoint-root", checkpoint_path, "checkpoint")


__all__ = ["verify_activevln_checkpoint", "verify_activevln_source"]
