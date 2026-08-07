from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from embodied_runtime.apps import navigation as app
from embodied_runtime.apps.navigation import policies
from embodied_runtime.tasks.navigation import NavigationSessionEvent


class _Policy:
    def __init__(self) -> None:
        self.prepared = False
        self.closed = False

    async def prepare(self) -> None:
        self.prepared = True

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

[policy]
backend = "qwen"

[qwen]
base_url = "http://127.0.0.1:15003/v1"
model = "qwen-original"

[go2]
camera_url = "http://robot.example:8765"
control_url = "http://robot.example:8080"

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
            "--control-timeout-s",
            "3",
        ]
    )
    config = app.apply_cli_overrides(loaded, args)

    assert config.run.instruction == "from cli"
    assert config.run.control_hz == 12.0
    assert config.policy.backend == "streamvln"
    assert config.policy.streamvln_root == "/models/StreamVLN"
    assert config.policy.streamvln_model_path == "/models/checkpoint"
    assert config.policy.streamvln_device == "cuda:1"
    assert config.go2.camera_token == "camera-from-env"
    assert config.go2.control_token == "control-from-env"
    assert config.go2.control_timeout_s == 3.0


def test_robot_host_cli_derives_service_urls_without_stored_ip() -> None:
    args = app.build_parser().parse_args(
        [
            "--robot-host",
            "go2.internal",
            "--camera-port",
            "9001",
            "--control-port",
            "9002",
        ]
    )

    resolved = app.apply_cli_overrides(app.Go2NavigationAppConfig(), args)

    assert resolved.go2.camera_url == "http://go2.internal:9001"
    assert resolved.go2.control_url == "http://go2.internal:9002"


def test_robot_host_rejects_ambiguous_explicit_url() -> None:
    args = app.build_parser().parse_args(
        ["--robot-ip", "10.0.0.2", "--camera-url", "http://camera:8765"]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        app.apply_cli_overrides(app.Go2NavigationAppConfig(), args)


def test_go2_limits_cannot_exceed_control_service_bounds() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        app.Go2Settings(max_abs_vx_mps=0.36)
    with pytest.raises(ValueError, match="cannot exceed"):
        app.Go2Settings(max_abs_yaw_rate_rps=0.71)


def test_qwen_policy_factory_does_not_manage_server(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def fake_qwen(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(policies, "QwenNavigationPolicy", fake_qwen)
    settings = app.PolicySettings(
        backend="qwen",
        qwen_base_url="http://qwen.example/v1",
        qwen_model="qwen-test",
        qwen_api_key="secret",
        qwen_timeout_s=4,
    )

    assert app.build_navigation_policy(settings) is sentinel
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

    monkeypatch.setattr(policies, "StreamVLNNavigationPolicy", fake_streamvln)
    settings = app.PolicySettings(
        backend="streamvln",
        streamvln_root="/models/StreamVLN",
        streamvln_model_path="/models/checkpoint",
        streamvln_device="cuda:1",
        cuda_memory_fraction=0.4,
        max_new_tokens=32,
    )

    assert app.build_navigation_policy(settings) is sentinel
    assert captured["streamvln_root"] == "/models/StreamVLN"
    assert captured["model_path"] == "/models/checkpoint"
    assert captured["device"] == "cuda:1"
    assert captured["cuda_memory_fraction"] == 0.4


@pytest.mark.parametrize("execute", [False, True])
def test_composition_passes_explicit_execution_policy_and_emits_json(execute) -> None:
    policy = _Policy()
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
            assert selected is policy
            constructed["session"] = kwargs

        async def run(self, instruction, *, episode_id):
            assert instruction == "导航到三脚架前"
            assert policy.prepared is True
            self_event = NavigationSessionEvent("plan_accepted", 0.5, 7, 0, "waypoints=2")
            constructed["session"]["event_sink"](self_event)
            return _Result(
                episode_id=episode_id,
                motion_commands=3 if execute else 0,
            )

    config = app.Go2NavigationAppConfig(
        run=app.RunSettings(instruction="导航到三脚架前", control_hz=11),
        policy=app.PolicySettings(backend="qwen"),
        go2=app.Go2Settings(camera_token="camera-token", control_token="control-token"),
    )
    result = asyncio.run(
        app.run_navigation(
            config,
            execute=execute,
            output=lines.append,
            policy_factory=lambda settings: policy,
            camera_factory=Camera,
            control_factory=Control,
            session_factory=Session,
        )
    )

    assert result.episode_id == "go2-navigation"
    assert policy.prepared is True
    assert policy.closed is True
    assert constructed["camera"] == (
        "http://127.0.0.1:8765",
        {"token": "camera-token", "timeout_s": 2.0},
    )
    assert constructed["control"] == (
        "http://127.0.0.1:8080",
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


def test_streamvln_auto_selects_image_reactive_session() -> None:
    policy = _Policy()
    constructed = {}
    lines: list[str] = []

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

    class Session:
        def __init__(self, selected, camera, control, **kwargs):
            del camera, control
            assert selected is policy
            constructed.update(kwargs)

        async def run(self, _instruction, *, episode_id):
            assert policy.prepared is True
            return _Result(episode_id=episode_id)

    config = app.Go2NavigationAppConfig(
        run=app.RunSettings(instruction="find the tripod"),
        policy=app.PolicySettings(backend="streamvln"),
    )
    asyncio.run(
        app.run_navigation(
            config,
            execute=True,
            output=lines.append,
            policy_factory=lambda _settings: policy,
            camera_factory=Client,
            control_factory=Client,
            session_factory=Session,
        )
    )

    assert constructed["config"].execute is True
    assert type(constructed["config"]).__name__ == "ReactiveNavigationSessionConfig"
    assert "follower" not in constructed
    assert json.loads(lines[0])["session_mode"] == "reactive"


def test_missing_instruction_fails_before_robot_io() -> None:
    config = app.Go2NavigationAppConfig(policy=app.PolicySettings(backend="qwen"))
    called = False

    def policy_factory(settings):
        nonlocal called
        called = True
        return _Policy()

    with pytest.raises(ValueError, match="instruction is required"):
        asyncio.run(app.run_navigation(config, policy_factory=policy_factory))
    assert called is False
