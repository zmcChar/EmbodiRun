from __future__ import annotations

import ast
from pathlib import Path


def test_engine_does_not_import_model_or_backend_implementations() -> None:
    engine_dir = Path(__file__).parents[2] / "src" / "embodied_runtime" / "engine"
    allowed_domain_imports = {
        "embodied_runtime.backends",
        "embodied_runtime.backends.compile",
        "embodied_runtime.backends.device",
        "embodied_runtime.backends.errors",
        "embodied_runtime.backends.interfaces",
        "embodied_runtime.backends.memory",
        "embodied_runtime.models.interfaces",
        "embodied_runtime.models.package",
        "embodied_runtime.models.plans",
        "embodied_runtime.models.request",
        "embodied_runtime.models.spec",
    }

    for path in engine_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        domain_imports = {
            name
            for name in imports
            if name.startswith(("embodied_runtime.models", "embodied_runtime.backends"))
        }
        assert domain_imports <= allowed_domain_imports, (
            f"{path.relative_to(engine_dir)} imports a concrete implementation: "
            f"{sorted(domain_imports - allowed_domain_imports)}"
        )
