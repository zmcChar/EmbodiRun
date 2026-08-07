"""Official StreamVLN source-tree activation and origin checks."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from .errors import StreamVLNLoadError

_REPOSITORY_PATH_LOCK = threading.Lock()
_OFFICIAL_MODULES = (
    "llava",
    "llava.model.multimodal_encoder.siglip_encoder",
    "model.stream_video_vln",
    "utils.utils",
)


def repository_import_paths(streamvln_root: str | Path) -> tuple[Path, Path]:
    """Return the two import roots required by the upstream source layout.

    The repository root exposes ``llava`` and the ``streamvln`` namespace. Its
    nested ``streamvln`` directory exposes the upstream top-level ``model`` and
    ``utils`` namespaces. The official loader needs both.
    """

    root = Path(streamvln_root).expanduser().resolve()
    return root, root / "streamvln"


def activate_repository_imports(streamvln_root: str | Path) -> tuple[Path, Path]:
    """Put the official source roots at deterministic import precedence."""

    root, nested = repository_import_paths(streamvln_root)
    with _REPOSITORY_PATH_LOCK:
        # Inserting root first leaves nested first, which is required for the
        # official ``from model...`` and ``from utils...`` imports.
        for path in (root, nested):
            text = str(path)
            while text in sys.path:
                sys.path.remove(text)
            sys.path.insert(0, text)
    return root, nested


def validate_repository_module_origins(
    streamvln_root: str | Path,
    *,
    require_loaded: bool,
) -> None:
    """Fail fast if generic upstream module names resolve to another project.

    StreamVLN uses the process-global names ``llava``, ``model``, and ``utils``.
    Changing ``sys.path`` cannot replace an object already cached in
    ``sys.modules``, so silently accepting a foreign module would mix model
    implementations in one process. A dedicated model worker remains the
    safest deployment; this check makes an accidental shared-process collision
    explicit.
    """

    root = Path(streamvln_root).expanduser().resolve()
    for name in _OFFICIAL_MODULES:
        module = sys.modules.get(name)
        if module is None:
            if require_loaded:
                raise StreamVLNLoadError(
                    f"official StreamVLN import did not load required module {name!r}"
                )
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str) or not source:
            raise StreamVLNLoadError(f"cannot verify cached module {name!r}: it has no __file__")
        resolved = Path(source).expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise StreamVLNLoadError(
                f"cached module {name!r} comes from {resolved}, outside {root}"
            )


__all__ = [
    "activate_repository_imports",
    "repository_import_paths",
    "validate_repository_module_origins",
]
