from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path("src/embodied_runtime")


def test_generic_serving_implementations_have_one_domain_owner() -> None:
    expected = {
        "LocalBackendProvider": SOURCE_ROOT / "engine/providers/local.py",
        "ProviderRegistry": SOURCE_ROOT / "engine/providers/registry.py",
        "MultiTenantInferenceService": (SOURCE_ROOT / "distributed/multitenant/service.py"),
    }
    definitions: dict[str, list[Path]] = {name: [] for name in expected}
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in definitions:
                definitions[node.name].append(path)

    assert definitions == {name: [path] for name, path in expected.items()}


def test_legacy_generic_serving_modules_are_removed() -> None:
    legacy = SOURCE_ROOT / "integrations/serving"
    assert not (legacy / "local.py").exists()
    assert not (legacy / "registry.py").exists()
    assert not (legacy / "multitenant.py").exists()
    assert not (legacy / "__init__.py").exists()
