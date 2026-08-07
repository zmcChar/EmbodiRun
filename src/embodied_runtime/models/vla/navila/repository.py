"""Activate the official NaVILA source tree without mixing foreign ``llava`` modules."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from .errors import NaVILALoadError

_PATH_LOCK = threading.Lock()
_OFFICIAL_MODULES = (
    "llava.constants",
    "llava.mm_utils",
    "llava.model.builder",
)


def activate_repository(navila_root: str | Path) -> Path:
    root = Path(navila_root).expanduser().resolve()
    with _PATH_LOCK:
        text = str(root)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    return root


def validate_module_origins(navila_root: str | Path, *, require_loaded: bool) -> None:
    root = Path(navila_root).expanduser().resolve()
    for name in _OFFICIAL_MODULES:
        module = sys.modules.get(name)
        if module is None:
            if require_loaded:
                raise NaVILALoadError(f"official NaVILA import did not load {name!r}")
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str) or not source:
            raise NaVILALoadError(f"cannot verify cached module {name!r}")
        resolved = Path(source).expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise NaVILALoadError(f"cached module {name!r} comes from {resolved}, outside {root}")


__all__ = ["activate_repository", "validate_module_origins"]
