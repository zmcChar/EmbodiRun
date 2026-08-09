from __future__ import annotations

import asyncio
import json
import sys
import types
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
    assert config.policy.model == "streamvln"
    assert config.policy.runtime == "transformers"
    assert config.policy.streamvln_root == "/models/StreamVLN"
    assert config.policy.streamvln_model_path == "/models/checkpoint"
    assert config.policy.streamvln_device == "cuda:1"
    assert config.go2.camera_token == "camera-from-env"
    assert config.go2.control_token == "control-from-env"
    assert config.go2.control_timeout_s == 3.0


def test_model_runtime_toml_and_cli_are_independent_axes(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
[policy]
model = "streamvln"
runtime = "vllm-omni"

[vllm_omni]
url = "ws://omni.example:9000/v1/realtime/robot/openpi"
timeout_s = 45
session_id = "toml-session"
""",
    )

    loaded = app.load_config(path)
    assert loaded.policy.model == "streamvln"
    assert loaded.policy.backend == "streamvln"
    assert loaded.policy.runtime == "vllm-omni"
    assert loaded.policy.vllm_omni_url == "ws://omni.example:9000/v1/realtime/robot/openpi"
    assert loaded.policy.vllm_omni_timeout_s == 45.0
    assert loaded.policy.vllm_omni_session_id == "toml-session"

    args = app.build_parser().parse_args(
        [
            "--model",
            "navila",
            "--runtime",
            "transformers",
            "--vllm-omni-url",
            "wss://override.example/v1/realtime/robot/openpi",
            "--vllm-omni-timeout-s",
            "12",
            "--vllm-omni-session-id",
            "cli-session",
        ]
    )
    resolved = app.apply_cli_overrides(loaded, args)

    assert resolved.policy.model == "navila"
    assert resolved.policy.runtime == "transformers"
    assert resolved.policy.vllm_omni_url == "wss://override.example/v1/realtime/robot/openpi"
    assert resolved.policy.vllm_omni_timeout_s == 12.0
    assert resolved.policy.vllm_omni_session_id == "cli-session"


def test_backend_alias_remains_supported_and_conflicts_are_explicit(tmp_path) -> None:
    args = app.build_parser().parse_args(["--backend", "navila"])
    resolved = app.apply_cli_overrides(app.Go2NavigationAppConfig(), args)
    assert resolved.policy.model == "navila"

    conflicting_args = app.build_parser().parse_args(
        ["--model", "streamvln", "--backend", "navila"]
    )
    with pytest.raises(ValueError, match="must match"):
        app.apply_cli_overrides(app.Go2NavigationAppConfig(), conflicting_args)

    path = _write_config(
        tmp_path,
        """
[policy]
model = "streamvln"
backend = "navila"
""",
    )
    with pytest.raises(ValueError, match="must match"):
        app.load_config(path)


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


@pytest.mark.parametrize("model", ["qwen", "internvla"])
def test_explicit_unsupported_vllm_runtime_fails_before_construction(
    model: str,
    monkeypatch,
) -> None:
    def unexpected_policy(**_kwargs):
        raise AssertionError("unsupported runtime must fail before policy construction")

    monkeypatch.setattr(policies, "QwenNavigationPolicy", unexpected_policy)
    monkeypatch.setattr(policies, "InternVLANavigationPolicy", unexpected_policy)

    with pytest.raises(ValueError, match=rf"not supported.*{model!r}"):
        app.build_navigation_policy(app.PolicySettings(backend=model, runtime="vllm-omni"))


@pytest.mark.parametrize(
    ("model", "factory_name"),
    [
        ("streamvln", "VllmOmniStreamVLNNavigationPolicy"),
        ("navila", "VllmOmniNaVILANavigationPolicy"),
    ],
)
def test_vllm_runtime_selects_lazy_remote_policy(
    model: str,
    factory_name: str,
    monkeypatch,
) -> None:
    captured = {}
    sentinel = object()

    def factory(**kwargs):
        captured.update(kwargs)
        return sentinel

    navigation_package = types.ModuleType("embodied_runtime.integrations.navigation")
    navigation_package.__path__ = []  # type: ignore[attr-defined]
    runtime_module = types.ModuleType("embodied_runtime.integrations.navigation.vllm_omni")
    runtime_module.VllmOmniStreamVLNNavigationPolicy = (
        factory if factory_name == "VllmOmniStreamVLNNavigationPolicy" else object
    )
    runtime_module.VllmOmniNaVILANavigationPolicy = (
        factory if factory_name == "VllmOmniNaVILANavigationPolicy" else object
    )
    monkeypatch.setitem(sys.modules, navigation_package.__name__, navigation_package)
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)

    settings = app.PolicySettings(
        backend=model,
        runtime="vllm-omni",
        vllm_omni_url="ws://omni.example:9000/v1/realtime/robot/openpi",
        vllm_omni_timeout_s=8,
        vllm_omni_session_id="episode-7",
    )
    assert app.build_navigation_policy(settings) is sentinel
    assert captured == {
        "url": "ws://omni.example:9000/v1/realtime/robot/openpi",
        "timeout_s": 8.0,
        "session_id": "episode-7",
    }


def test_vvla_runtime_selects_lazy_activevln_policy(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def factory(**kwargs):
        captured.update(kwargs)
        return sentinel

    runtime_module = types.ModuleType("embodied_runtime.integrations.navigation.vvla")
    runtime_module.VvlaActiveVLNNavigationPolicy = factory
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    settings = app.PolicySettings(
        backend="activevln",
        runtime="vvla",
        vvla_root="/src/vvla",
        activevln_checkpoint="/models/activevln",
        activevln_device="cuda:0",
        activevln_dtype="bfloat16",
        activevln_attention="sdpa",
        activevln_max_new_tokens=24,
        activevln_max_context=4096,
    )

    assert app.build_navigation_policy(settings) is sentinel
    assert captured["vvla_root"] == "/src/vvla"
    assert captured["checkpoint"] == "/models/activevln"
    assert captured["dtype"] == "bfloat16"
    assert captured["attention"] == "sdpa"
    assert captured["max_new_tokens"] == 24
    assert captured["max_context"] == 4096


def test_transformers_runtime_selects_lazy_activevln_policy(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def factory(**kwargs):
        captured.update(kwargs)
        return sentinel

    runtime_module = types.ModuleType(
        "embodied_runtime.integrations.navigation.activevln_transformers"
    )
    runtime_module.TransformersActiveVLNNavigationPolicy = factory
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    settings = app.PolicySettings(
        backend="activevln",
        runtime="transformers",
        vvla_root="/src/vvla",
        activevln_checkpoint="/models/activevln",
        activevln_device="cuda:0",
        activevln_dtype="bfloat16",
        activevln_attention="eager",
        activevln_max_new_tokens=24,
        activevln_max_context=4096,
    )

    assert app.build_navigation_policy(settings) is sentinel
    assert captured["vvla_root"] == "/src/vvla"
    assert captured["checkpoint"] == "/models/activevln"
    assert captured["dtype"] == "bfloat16"
    assert captured["attention"] == "eager"
    assert captured["max_new_tokens"] == 24
    assert captured["max_context"] == 4096


def test_activevln_advertises_transformers_and_vvla() -> None:
    assert app.supported_runtimes("activevln") == ("transformers", "vvla")
    with pytest.raises(ValueError, match="eager_bc.*VVLA-only"):
        app.build_navigation_policy(
            app.PolicySettings(
                backend="activevln",
                runtime="transformers",
                activevln_attention="eager_bc",
            )
        )


def test_activevln_cli_overrides_are_model_specific() -> None:
    args = app.build_parser().parse_args(
        [
            "--model",
            "activevln",
            "--runtime",
            "vvla",
            "--vvla-root",
            "/src/vvla",
            "--model-path",
            "/models/activevln",
            "--activevln-revision",
            "a" * 40,
            "--device",
            "cuda:1",
            "--dtype",
            "bfloat16",
            "--attention",
            "sdpa",
            "--max-new-tokens",
            "24",
            "--max-context",
            "4096",
            "--allow-download",
            "--do-sample",
        ]
    )
    policy = app.apply_cli_overrides(app.Go2NavigationAppConfig(), args).policy

    assert policy.backend == "activevln"
    assert policy.runtime == "vvla"
    assert policy.vvla_root == "/src/vvla"
    assert policy.activevln_checkpoint == "/models/activevln"
    assert policy.activevln_revision == "a" * 40
    assert policy.activevln_device == "cuda:1"
    assert policy.activevln_dtype == "bfloat16"
    assert policy.activevln_attention == "sdpa"
    assert policy.activevln_max_new_tokens == 24
    assert policy.activevln_max_context == 4096
    assert policy.activevln_allow_download is True
    assert policy.activevln_do_sample is True


def test_activevln_toml_table_loads_all_vvla_options(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
[policy]
model = "activevln"
runtime = "vvla"

[activevln]
repository = "/src/vvla"
checkpoint = "/models/activevln"
revision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
device = "cuda:1"
dtype = "float16"
attention = "eager_bc"
max_new_tokens = 32
max_context = 8192
allow_download = true
do_sample = true
""",
    )

    policy = app.load_config(path).policy
    assert policy.backend == "activevln"
    assert policy.runtime == "vvla"
    assert policy.vvla_root == "/src/vvla"
    assert policy.activevln_checkpoint == "/models/activevln"
    assert policy.activevln_revision == "b" * 40
    assert policy.activevln_device == "cuda:1"
    assert policy.activevln_dtype == "float16"
    assert policy.activevln_attention == "eager_bc"
    assert policy.activevln_max_new_tokens == 32
    assert policy.activevln_max_context == 8192
    assert policy.activevln_allow_download is True
    assert policy.activevln_do_sample is True


def test_navila_cli_and_factory_use_dedicated_model_settings(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_navila(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(policies, "NaVILANavigationPolicy", fake_navila)
    args = app.build_parser().parse_args(
        [
            "--backend",
            "navila",
            "--navila-root",
            "/models/NaVILA",
            "--model-path",
            "/models/navila-8b",
            "--device",
            "cuda:1",
            "--cuda-memory-fraction",
            "0.4",
            "--max-new-tokens",
            "32",
        ]
    )
    config = app.apply_cli_overrides(app.Go2NavigationAppConfig(), args)

    assert config.policy.navila_root == "/models/NaVILA"
    assert config.policy.navila_model_path == "/models/navila-8b"
    assert config.policy.navila_device == "cuda:1"
    assert config.policy.navila_cuda_memory_fraction == 0.4
    assert config.policy.navila_max_new_tokens == 32
    assert app.build_navigation_policy(config.policy) is sentinel
    assert captured == {
        "navila_root": "/models/NaVILA",
        "model_path": "/models/navila-8b",
        "device": "cuda:1",
        "cuda_memory_fraction": 0.4,
        "max_new_tokens": 32,
        "local_files_only": True,
    }


def test_navila_toml_table_loads_without_reusing_streamvln_options(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
[policy]
backend = "navila"

[navila]
repository = "/models/NaVILA"
checkpoint = "/models/navila-8b"
device = "cuda:1"
cuda_memory_fraction = 0.42
max_new_tokens = 24
local_files_only = true
""",
    )

    policy = app.load_config(path).policy

    assert policy.backend == "navila"
    assert policy.navila_root == "/models/NaVILA"
    assert policy.navila_model_path == "/models/navila-8b"
    assert policy.navila_device == "cuda:1"
    assert policy.navila_cuda_memory_fraction == pytest.approx(0.42)
    assert policy.navila_max_new_tokens == 24
    assert policy.navila_local_files_only is True


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
    assert decoded[0]["model"] == "qwen"
    assert decoded[0]["runtime"] == "transformers"
    assert decoded[0]["backend"] == "qwen"
    assert decoded[0]["mode"] == ("execute" if execute else "dry-run")
    assert decoded[1]["kind"] == "plan_accepted"
    assert decoded[-1]["kind"] == "navigation_summary"
    assert decoded[-1]["ok"] is True
    assert decoded[-1]["model"] == "qwen"
    assert decoded[-1]["runtime"] == "transformers"
    assert decoded[-1]["backend"] == "qwen"
    assert "events" not in decoded[-1]


@pytest.mark.parametrize("backend", ["streamvln", "navila", "activevln"])
def test_image_action_backends_auto_select_reactive_session(backend: str) -> None:
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
        policy=app.PolicySettings(
            backend=backend,
            runtime="vvla" if backend == "activevln" else "transformers",
        ),
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
    expected_max_pulse_s = 2.5 if backend in {"navila", "activevln"} else 1.5
    assert constructed["config"].max_pulse_s == pytest.approx(expected_max_pulse_s)
    assert constructed["config"].max_waypoints_per_observation == (
        3 if backend == "activevln" else 1
    )
    assert constructed["config"].terminal_after_waypoints is (backend == "activevln")
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
