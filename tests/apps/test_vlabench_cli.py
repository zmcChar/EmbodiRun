from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "arguments", "expected"),
    [
        (
            "embodied_runtime.apps.vlabench_big_small_brain",
            ["--checkpoint", "checkpoint", "--output-dir", "output", "--seeds", "3,4"],
            {
                "checkpoint": "checkpoint",
                "output_dir": Path("output"),
                "seeds": (3, 4),
                "device": "cuda",
                "backbone_path": None,
                "max_episode_steps": 500,
                "first_subgoal_budget": 350,
                "subgoal_stability_steps": 3,
                "control_period_s": 0.1,
                "skip_warmup": False,
                "allow_download": False,
            },
        ),
        (
            "embodied_runtime.apps.vlabench_texas_holdem",
            ["--checkpoint", "checkpoint", "--output-dir", "output", "--seeds", "3,4"],
            {
                "checkpoint": "checkpoint",
                "output_dir": Path("output"),
                "seeds": (3, 4),
                "device": "cuda",
                "backbone_path": None,
                "max_episode_steps": 800,
                "control_period_s": 0.1,
                "place_lift_height_m": 0.15,
                "place_clearance_m": 0.12,
                "place_retract_height_m": 0.10,
                "place_max_step_m": 0.025,
                "place_open_steps": 5,
                "place_slot_count": 5,
                "place_slot_spacing_m": 0.05,
                "place_workspace_limit_m": 2.0,
                "skip_warmup": False,
                "allow_download": False,
            },
        ),
        (
            "embodied_runtime.apps.vlabench_texas_holdem_skill_gate",
            ["--output-dir", "output", "--seeds", "3,4"],
            {
                "output_dir": Path("output"),
                "seeds": (3, 4),
                "render_size": 96,
                "max_episode_steps": 1200,
                "shadow_max_episode_steps": 2000,
                "settle_repeats": 12,
                "planner_checkpoint": None,
                "planner_device": "cpu",
                "planner_dtype": "float32",
                "planner_max_new_tokens": 192,
                "allow_single_json_fence": False,
            },
        ),
    ],
)
def test_vlabench_cli_contract_is_preserved(
    module_name: str,
    arguments: list[str],
    expected: dict[str, object],
) -> None:
    module = importlib.import_module(module_name)
    assert vars(module.build_parser().parse_args(arguments)) == expected


def test_vlabench_cli_imports_keep_native_model_dependencies_lazy() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(source_root)
    code = """
import sys
import embodied_runtime.apps.vlabench_big_small_brain
import embodied_runtime.apps.vlabench_texas_holdem
import embodied_runtime.apps.vlabench_texas_holdem_skill_gate
assert "torch" not in sys.modules
assert "scipy" not in sys.modules
assert "transformers" not in sys.modules
assert not any(name == "lerobot" or name.startswith("lerobot.") for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=source_root.parent,
        env=environment,
    )
