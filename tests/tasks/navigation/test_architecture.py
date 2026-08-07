from __future__ import annotations

import ast
from pathlib import Path

from embodied_runtime.tasks.navigation import MobileBase, NavigationPolicy, ObservationSource


def test_navigation_task_has_no_concrete_robot_or_integration_imports() -> None:
    root = Path("src/embodied_runtime/tasks/navigation")
    forbidden = (
        "embodied_runtime.robots",
        "embodied_runtime.integrations",
    )
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(name.startswith(forbidden) for name in imported), path


def test_interfaces_are_runtime_checkable_structural_protocols() -> None:
    class Policy:
        async def plan(self, request): ...

    class Source:
        def capture(self, *, episode_id, reset, robot_state): ...

    class Base:
        def state(self): ...

        def preflight(self): ...

        def start_velocity_lease(self, command, *, duration_s): ...

        def update_velocity_lease(self, lease_id, command): ...

        def stop(self): ...

    assert isinstance(Policy(), NavigationPolicy)
    assert isinstance(Source(), ObservationSource)
    assert isinstance(Base(), MobileBase)
