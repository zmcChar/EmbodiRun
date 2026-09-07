import ast
import subprocess
import sys
from pathlib import Path


def test_model_dependencies_are_isolated_to_the_sglang_service() -> None:
    root = Path(__file__).parents[1] / "src" / "rlinf_deploy"
    forbidden = {"models", "policies", "backends", "engine"}
    assert not forbidden.intersection(path.name for path in root.iterdir())
    integration = root / "services/inference/adapters/sglang/pi05.py"
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                package = module.partition(".")[0]
                assert package not in {"vvla", "transformers"}, source
                if package in {"torch", "sglang", "safetensors"}:
                    # Opt-in server integration, never Host/client runtime code.
                    assert source == integration, source


def test_host_and_client_import_without_inference_frameworks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import rlinf_deploy\n"
                "import rlinf_deploy.services.host.plan\n"
                "import rlinf_deploy.services.inference.adapters.sglang\n"
                "assert not {'torch', 'sglang', 'vvla'} & sys.modules.keys()\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_go2_adapter_is_retained() -> None:
    from rlinf_deploy.robots.unitree.go2 import Go2ControlClient

    assert Go2ControlClient.__name__ == "Go2ControlClient"
