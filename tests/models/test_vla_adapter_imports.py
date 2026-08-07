from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_canonical_adapters_and_registry_import_without_torch() -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    source = str(repository / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing_pythonpath else os.pathsep.join((source, existing_pythonpath))
    )
    script = """
import sys

from embodied_runtime.models import get_model_adapter
from embodied_runtime.models.vla.gr00t_n17 import Gr00tN17Adapter
from embodied_runtime.models.vla.openvla_oft import OpenVLAOFTAdapter
from embodied_runtime.models.vla.pi05 import Pi05Adapter
from embodied_runtime.models.vla.smolvla import SmolVLAAdapter

assert "torch" not in sys.modules
assert isinstance(get_model_adapter("gr00t_n17"), Gr00tN17Adapter)
assert isinstance(get_model_adapter("openvla_oft"), OpenVLAOFTAdapter)
assert isinstance(get_model_adapter("pi05"), Pi05Adapter)
assert isinstance(get_model_adapter("smolvla"), SmolVLAAdapter)
assert get_model_adapter("gr00t_n17").describe().family == "gr00t_n17"
assert get_model_adapter("openvla_oft").describe().family == "openvla_oft_categorical"
assert get_model_adapter("pi05").describe().family == "pi05_flow"
assert get_model_adapter("smolvla").describe().family == "smolvla_flow"
assert "torch" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
