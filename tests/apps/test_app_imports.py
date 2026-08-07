"""Optional model dependencies must stay behind app runtime build paths."""

from __future__ import annotations

import subprocess
import sys


def test_cloud_edge_and_multi_robot_apps_import_without_torch() -> None:
    modules = (
        "embodied_runtime.apps.cloud_edge.pi05_settings",
        "embodied_runtime.apps.cloud_edge.smolvla_settings",
        "embodied_runtime.apps.multi_robot.cloud_settings",
        "embodied_runtime.apps.multi_robot.edge_settings",
        "embodied_runtime.apps.cloud_edge_failover",
        "embodied_runtime.apps.multi_robot_cloud",
        "embodied_runtime.apps.multi_robot_edge",
        "embodied_runtime.apps.pi05_cloud_server",
        "embodied_runtime.apps.pi05_cpu_gpu_collaboration",
        "embodied_runtime.apps.smolvla_pi05_async",
    )
    statements = [f"import {module}" for module in modules]
    statements.append("assert 'torch' not in sys.modules, sorted(sys.modules)")
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; " + "; ".join(statements)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
