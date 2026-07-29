"""Guard the five-group dependency direction at the source level."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[2] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "embodied_runtime"

DOMAINS = ("models", "distributed", "engine", "backends", "robots")
FORBIDDEN = {
    domain: tuple(f"embodied_runtime.{other}" for other in DOMAINS if other != domain)
    for domain in DOMAINS
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolved_imports(path: Path) -> set[str]:
    module_name = _module_name(path)
    package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                imports.add(importlib.util.resolve_name(relative_name, package))
            elif node.module:
                imports.add(node.module)
    return imports


@pytest.mark.parametrize("domain", sorted(FORBIDDEN))
def test_concrete_domains_do_not_import_each_other(domain: str) -> None:
    violations: list[str] = []
    for path in sorted((PACKAGE_ROOT / domain).rglob("*.py")):
        for imported in sorted(_resolved_imports(path)):
            if any(
                imported == prefix or imported.startswith(prefix + ".")
                for prefix in FORBIDDEN[domain]
            ):
                violations.append(f"{path.relative_to(SOURCE_ROOT)} imports {imported}")

    assert not violations, "cross-domain imports found:\n" + "\n".join(violations)


def test_vllm_omni_provider_does_not_import_local_execution_or_backends() -> None:
    path = PACKAGE_ROOT / "integrations" / "serving" / "gr00t" / "vllm_omni.py"
    forbidden = (
        "embodied_runtime.backends",
        "embodied_runtime.engine",
        "embodied_runtime.integrations.serving.local",
    )
    violations = [
        imported
        for imported in sorted(_resolved_imports(path))
        if any(imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden)
    ]
    assert not violations, f"vLLM-Omni provider imports local execution code: {violations}"
