from pathlib import Path


def test_deploy_source_contains_no_model_runtime_domains() -> None:
    root = Path(__file__).parents[1] / "src" / "rlinf_deploy"
    forbidden = {"models", "policies", "backends", "engine"}
    assert not forbidden.intersection(path.name for path in root.iterdir())
    for source in root.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "import torch" not in text
        assert "import transformers" not in text


def test_go2_adapter_is_retained() -> None:
    from rlinf_deploy.robots.unitree.go2 import Go2ControlClient

    assert Go2ControlClient.__name__ == "Go2ControlClient"
