from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from embodied_runtime.apps import go2_navigation as app
from embodied_runtime.robots.go2.session import NavigationSessionEvent


class _Provider:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _Result:
    episode_id: str
    reason: str = "max_runtime"
    elapsed_s: float = 1.25
    inference_count: int = 2
    plans_accepted: int = 2
    control_ticks: int = 12
    motion_commands: int = 0
    last_observation_sequence: int = 7
    events: tuple[object, ...] = ()


def _write_config(tmp_path, text: str):
    path = tmp_path / "navigation.toml"
    path.write_text(text)
    return path


def test_parser_is_dry_run_unless_execute_is_explicit() -> None:
    parser = app.build_parser()
    assert parser.parse_args([]).execute is False
    assert parser.parse_args(["--execute"]).execute is True


def test_toml_load_and_cli_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GO2_CAMERA_TOKEN", "camera-from-env")
    monkeypatch.setenv("GO2_API_TOKEN", "control-from-env")
    path = _write_config(
        tmp_path,
        """
[session]
instruction = "from toml"
episode_id = "episode-a"
max_runtime_s = 90
control_hz = 8

[provider]
backend = "qwen"

[qwen]
base_url = "http://127.0.0.1:15003/v1"
model = "qwen-original"

[go2]
camera_url = "http://192.168.137.34:8765"
control_url = "http://192.168.137.34:8080"

[go2.follower]
max_abs_vx_mps = 0.30
max_abs_vy_mps = 0.20
max_abs_yaw_rate_rps = 0.60
""",
    )
    loaded = app.load_config(path)
    args = app.build_parser().parse_args(
        [
            "--config",
            str(path),
            "--instruction",
            "from cli",
            "--backend",
            "streamvln",
            "--streamvln-root",
            "/models/StreamVLN",
            "--model-path",
            "/models/checkpoint",
            "--device",
            "cuda:1",
            "--control-hz",
            "12",
        ]
    )
    config = app.apply_cli_overrides(loaded, args)

    assert config.run.instruction == "from cli"
    assert config.run.control_hz == 12.0
    assert config.provider.backend == "streamvln"
    assert config.provider.streamvln_root == "/models/StreamVLN"
    assert config.provider.streamvln_model_path == "/models/checkpoint"
    assert config.provider.streamvln_device == "cuda:1"
    assert config.go2.camera_token == "camera-from-env"
    assert config.go2.control_token == "control-from-env"


def test_go2_limits_cannot_exceed_control_service_bounds() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        app.Go2Settings(max_abs_vx_mps=0.36)
    with pytest.raises(ValueError, match="cannot exceed"):
        app.Go2Settings(max_abs_yaw_rate_rps=0.71)


def test_qwen_provider_factory_does_not_manage_server(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def fake_qwen(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(app, "QwenNavigationProvider", fake_qwen)
    settings = app.ProviderSettings(
        backend="qwen",
        qwen_base_url="http://qwen.example/v1",
        qwen_model="qwen-test",
        qwen_api_key="secret",
        qwen_timeout_s=4,
    )

    assert app.build_navigation_provider(settings) is sentinel
    assert calls == [
        {
            "base_url": "http://qwen.example/v1",
            "model": "qwen-test",
            "api_key": "secret",
            "timeout_s": 4.0,
        }
    ]


def test_streamvln_factory_forwards_local_runtime_options(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_streamvln(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(app, "StreamVLNNavigationProvider", fake_streamvln)
    settings = app.ProviderSettings(
        backend="streamvln",
        streamvln_root="/models/StreamVLN",
        streamvln_model_path="/models/checkpoint",
        streamvln_device="cuda:1",
        cuda_memory_fraction=0.4,
        max_new_tokens=32,
    )

    assert app.build_navigation_provider(settings) is sentinel
    assert captured["streamvln_root"] == "/models/StreamVLN"
    assert captured["model_path"] == "/models/checkpoint"
    assert captured["device"] == "cuda:1"
    assert captured["cuda_memory_fraction"] == 0.4


@pytest.mark.parametrize("execute", [False, True])
def test_composition_passes_explicit_execution_policy_and_emits_json(execute) -> None:
    provider = _Provider()
    constructed = {}
    lines: list[str] = []

    class Camera:
        def __init__(self, url, **kwargs):
            constructed["camera"] = (url, kwargs)

    class Control:
        def __init__(self, url, **kwargs):
            constructed["control"] = (url, kwargs)

    class Session:
        def __init__(self, selected, camera, control, **kwargs):
            del camera, control
            assert selected is provider
            constructed["session"] = kwargs

        async def run(self, instruction, *, episode_id):
            assert instruction == "导航到三脚架前"
            self_event = NavigationSessionEvent("plan_accepted", 0.5, 7, 0, "waypoints=2")
            constructed["session"]["event_sink"](self_event)
            return _Result(
                episode_id=episode_id,
                motion_commands=3 if execute else 0,
            )

    config = app.Go2NavigationAppConfig(
        run=app.RunSettings(instruction="导航到三脚架前", control_hz=11),
        provider=app.ProviderSettings(backend="qwen"),
        go2=app.Go2Settings(camera_token="camera-token", control_token="control-token"),
    )
    result = asyncio.run(
        app.run_go2_navigation(
            config,
            execute=execute,
            output=lines.append,
            provider_factory=lambda settings: provider,
            camera_factory=Camera,
            control_factory=Control,
            session_factory=Session,
        )
    )

    assert result.episode_id == "go2-navigation"
    assert provider.closed is True
    assert constructed["camera"] == (
        "http://192.168.137.34:8765",
        {"token": "camera-token", "timeout_s": 2.0},
    )
    assert constructed["control"] == (
        "http://192.168.137.34:8080",
        {"token": "control-token", "timeout_s": 1.0},
    )
    session_config = constructed["session"]["config"]
    assert session_config.execute is execute
    assert session_config.control_hz == 11.0
    assert constructed["session"]["follower"].config.limits == app.GO2_CONTROL_HARD_LIMITS
    decoded = [json.loads(line) for line in lines]
    assert decoded[0]["kind"] == "navigation_start"
    assert decoded[0]["mode"] == ("execute" if execute else "dry-run")
    assert decoded[1]["kind"] == "plan_accepted"
    assert decoded[-1]["kind"] == "navigation_summary"
    assert decoded[-1]["ok"] is True
    assert "events" not in decoded[-1]


def test_missing_instruction_fails_before_robot_io() -> None:
    config = app.Go2NavigationAppConfig(provider=app.ProviderSettings(backend="qwen"))
    called = False

    def provider_factory(settings):
        nonlocal called
        called = True
        return _Provider()

    with pytest.raises(ValueError, match="instruction is required"):
        asyncio.run(app.run_go2_navigation(config, provider_factory=provider_factory))
    assert called is False
