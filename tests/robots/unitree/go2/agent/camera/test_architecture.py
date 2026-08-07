from __future__ import annotations

import ast
from pathlib import Path

CAMERA_ROOT = Path("src/embodied_runtime/robots/unitree/go2/agent/camera")


def test_camera_modules_are_focused_and_python38_parseable() -> None:
    paths = sorted(CAMERA_ROOT.glob("*.py"))
    assert paths
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 300, path.name
        ast.parse(source, filename=str(path), feature_version=(3, 8))


def test_removed_camera_facades_do_not_return() -> None:
    assert not (CAMERA_ROOT / "config.py").exists()
    assert not (CAMERA_ROOT / "stores.py").exists()

    types_tree = ast.parse(
        (CAMERA_ROOT / "types.py").read_text(encoding="utf-8"),
        filename=str(CAMERA_ROOT / "types.py"),
    )
    assigned_names = {
        target.id
        for node in types_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "V4L2Format" not in assigned_names
