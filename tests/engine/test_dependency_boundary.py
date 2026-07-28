from __future__ import annotations

import ast
from pathlib import Path


def test_engine_does_not_import_model_or_backend_implementations() -> None:
    engine_dir = Path(__file__).parents[2] / "src" / "embodied_runtime" / "engine"
    forbidden = (
        "embodied_runtime.models",
        "embodied_runtime.backends",
    )

    for path in engine_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.startswith(prefix) for name in imports for prefix in forbidden), (
            f"{path.name} imports a concrete domain: {imports}"
        )
