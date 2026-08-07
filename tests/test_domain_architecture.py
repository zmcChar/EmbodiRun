"""Dependency-direction checks for the domain-owned runtime architecture."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path("src/embodied_runtime")
MAX_RUNTIME_MODULE_LINES = 300


def _imports(root: Path) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    paths = (root,) if root.is_file() else root.rglob("*.py")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result.append((path, node.module))
    return result


def _assert_no_dependencies(root: Path, forbidden: tuple[str, ...]) -> None:
    violations = [(path, module) for path, module in _imports(root) if module.startswith(forbidden)]
    assert not violations


def _relative_import_graph(root: Path) -> dict[str, set[str]]:
    paths = {path.stem: path for path in root.glob("*.py") if path.name != "__init__.py"}
    graph = {module: set() for module in paths}
    for module, path in paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dependency = node.module.split(".", maxsplit=1)[0]
                if dependency in graph:
                    graph[module].add(dependency)
    return graph


def _find_import_cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    state = {module: "unvisited" for module in graph}
    stack: list[str] = []
    cycles: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        if state[module] == "visiting":
            start = stack.index(module)
            cycles.append(tuple(stack[start:] + [module]))
            return
        if state[module] == "visited":
            return
        state[module] = "visiting"
        stack.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        stack.pop()
        state[module] = "visited"

    for module in sorted(graph):
        visit(module)
    return cycles


def test_removed_ambiguous_and_legacy_packages_do_not_return() -> None:
    removed = (
        SOURCE_ROOT / "contracts",
        SOURCE_ROOT / "robots/go2",
        SOURCE_ROOT / "deployment/go2",
        SOURCE_ROOT / "integrations/serving",
        SOURCE_ROOT / "apps/go2_navigation.py",
        SOURCE_ROOT / "tasks/navigation/types.py",
        SOURCE_ROOT / "robots/unitree/go2/state.py",
        SOURCE_ROOT / "robots/unitree/go2/motion.py",
        SOURCE_ROOT / "robots/unitree/go2/limits.py",
    )
    assert all(not path.exists() for path in removed)


def test_runtime_modules_stay_below_size_guard() -> None:
    oversized = []
    for path in SOURCE_ROOT.rglob("*.py"):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_RUNTIME_MODULE_LINES:
            oversized.append((path, line_count))
    assert not oversized, "split modules that exceed the runtime size guard: " + ", ".join(
        f"{path} ({line_count})" for path, line_count in oversized
    )


def test_navigation_policy_families_use_symmetric_packages() -> None:
    root = SOURCE_ROOT / "policies/navigation"
    for family in ("qwen", "streamvln", "internvla"):
        assert (root / family / "__init__.py").is_file()
        assert (root / family / "policy.py").is_file()
    assert not (root / "streamvln.py").exists()
    assert not (root / "internvla.py").exists()


def test_app_settings_do_not_import_concrete_models_or_backends() -> None:
    settings = (
        tuple((SOURCE_ROOT / "apps/cloud_edge").glob("*_settings.py"))
        + tuple((SOURCE_ROOT / "apps/multi_robot").glob("*_settings.py"))
        + tuple((SOURCE_ROOT / "apps/vlabench").rglob("settings.py"))
    )
    violations = [
        (path, module)
        for path in settings
        for imported_path, module in _imports(path)
        if imported_path == path
        and module.startswith(
            (
                "embodied_runtime.backends.torch_cuda",
                "embodied_runtime.models.vla",
                "torch",
            )
        )
    ]
    assert not violations


def test_app_subpackages_have_acyclic_local_dependency_graphs() -> None:
    assert not _find_import_cycles(_relative_import_graph(SOURCE_ROOT / "apps/cloud_edge"))
    assert not _find_import_cycles(_relative_import_graph(SOURCE_ROOT / "apps/multi_robot"))
    assert not _find_import_cycles(
        _relative_import_graph(SOURCE_ROOT / "apps/vlabench/big_small_brain")
    )
    assert not _find_import_cycles(_relative_import_graph(SOURCE_ROOT / "apps/vlabench/skill_gate"))
    assert not _find_import_cycles(
        _relative_import_graph(SOURCE_ROOT / "apps/vlabench/texas_holdem")
    )


def test_vlabench_app_packages_do_not_import_sibling_compositions() -> None:
    root = SOURCE_ROOT / "apps/vlabench"
    package_names = {"big_small_brain", "skill_gate", "texas_holdem"}
    violations = []
    for owner in sorted(package_names):
        for path in (root / owner).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    dependency = node.module.split(".", maxsplit=1)[0]
                    if node.level >= 2 and dependency in package_names and dependency != owner:
                        violations.append((path, dependency))
                    prefix = "embodied_runtime.apps.vlabench."
                    if node.module.startswith(prefix):
                        dependency = node.module.removeprefix(prefix).split(".", maxsplit=1)[0]
                        if dependency in package_names and dependency != owner:
                            violations.append((path, dependency))
    assert not violations


def test_vlabench_integrations_do_not_depend_on_apps() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "integrations/lerobot/vlabench_smolvla",
        ("embodied_runtime.apps",),
    )
    _assert_no_dependencies(
        SOURCE_ROOT / "integrations/planning/hf_texas_holdem",
        ("embodied_runtime.apps",),
    )


def test_vlabench_integration_packages_have_acyclic_local_dependency_graphs() -> None:
    assert not _find_import_cycles(
        _relative_import_graph(SOURCE_ROOT / "integrations/lerobot/vlabench_smolvla")
    )
    assert not _find_import_cycles(
        _relative_import_graph(SOURCE_ROOT / "integrations/planning/hf_texas_holdem")
    )


def test_models_are_native_and_runtime_independent() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "models",
        (
            "embodied_runtime.apps",
            "embodied_runtime.backends",
            "embodied_runtime.deployment",
            "embodied_runtime.distributed",
            "embodied_runtime.engine",
            "embodied_runtime.integrations",
            "embodied_runtime.policies",
            "embodied_runtime.robots",
            "embodied_runtime.simulators",
            "embodied_runtime.tasks",
        ),
    )


def test_navigation_policies_do_not_depend_on_engine_apps_or_embodiment() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "policies/navigation",
        (
            "embodied_runtime.apps",
            "embodied_runtime.deployment",
            "embodied_runtime.engine",
            "embodied_runtime.integrations",
            "embodied_runtime.robots",
            "embodied_runtime.simulators",
        ),
    )


def test_navigation_task_loop_is_model_and_robot_independent() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "tasks/navigation",
        (
            "embodied_runtime.apps",
            "embodied_runtime.backends",
            "embodied_runtime.deployment",
            "embodied_runtime.distributed",
            "embodied_runtime.engine",
            "embodied_runtime.integrations",
            "embodied_runtime.models",
            "embodied_runtime.policies",
            "embodied_runtime.robots",
            "embodied_runtime.simulators",
        ),
    )


def test_go2_driver_implements_tasks_without_model_dependencies() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "robots/unitree/go2",
        (
            "embodied_runtime.apps",
            "embodied_runtime.backends",
            "embodied_runtime.engine",
            "embodied_runtime.integrations",
            "embodied_runtime.models",
            "embodied_runtime.policies",
            "embodied_runtime.simulators",
        ),
    )


def test_simulators_only_import_generic_robot_values() -> None:
    allowed_robot_modules = {
        "embodied_runtime.robots.action",
        "embodied_runtime.robots.observation",
    }
    violations = [
        (path, module)
        for path, module in _imports(SOURCE_ROOT / "simulators")
        if module.startswith("embodied_runtime.robots") and module not in allowed_robot_modules
    ]
    assert not violations


def test_simulators_do_not_depend_on_integration_adapters() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "simulators",
        ("embodied_runtime.integrations",),
    )


def test_simulator_modules_have_an_acyclic_local_dependency_graph() -> None:
    assert not _find_import_cycles(_relative_import_graph(SOURCE_ROOT / "simulators"))


def test_engine_never_imports_concrete_backend_implementations() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "engine",
        (
            "embodied_runtime.backends.ascend",
            "embodied_runtime.backends.horizon",
            "embodied_runtime.backends.torch_cuda",
        ),
    )


def test_torch_cuda_backend_remains_model_family_and_task_independent() -> None:
    _assert_no_dependencies(
        SOURCE_ROOT / "backends/torch_cuda",
        (
            "embodied_runtime.apps",
            "embodied_runtime.models.vla",
            "embodied_runtime.models.vln",
            "embodied_runtime.policies",
            "embodied_runtime.robots",
            "embodied_runtime.simulators",
            "embodied_runtime.tasks",
        ),
    )


def test_torch_cuda_modules_have_an_acyclic_local_dependency_graph() -> None:
    assert not _find_import_cycles(_relative_import_graph(SOURCE_ROOT / "backends/torch_cuda"))
