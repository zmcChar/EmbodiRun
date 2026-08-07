from __future__ import annotations

from types import SimpleNamespace

from embodied_runtime.apps import multi_robot_edge as multi_robot_edge_app
from embodied_runtime.apps.cloud_edge.pi05_settings import load_pi05_cpu_gpu_config
from embodied_runtime.apps.cloud_edge_failover import load_demo_config
from embodied_runtime.apps.multi_robot.cloud_settings import load_multi_robot_cloud_config
from embodied_runtime.apps.multi_robot.edge_demo import build_demo_observation
from embodied_runtime.apps.multi_robot.edge_settings import (
    MultiRobotEdgeConfig,
    load_multi_robot_edge_config,
)
from embodied_runtime.distributed import FailoverMode, RobotSessionIdentity


def test_toy_config_loads_async_blend_and_weight(tmp_path) -> None:
    config_path = tmp_path / "blend.toml"
    config_path.write_text(
        """
[failover]
mode = "async_blend"

[fusion]
cloud_weight = 0.25
""",
        encoding="utf-8",
    )

    config = load_demo_config(config_path)

    assert config.failover.mode is FailoverMode.ASYNC_BLEND
    assert config.cloud_weight == 0.25


def test_real_cpu_gpu_config_declares_physical_placements() -> None:
    config = load_pi05_cpu_gpu_config("configs/pi05_cpu_gpu_collaboration.toml")

    assert config.failover.mode is FailoverMode.ASYNC_BLEND
    assert config.cloud_device == "cuda:0"
    assert config.edge_device == "cpu"
    assert config.cloud_weight == 0.5


def test_multi_robot_configs_separate_cloud_service_from_edge_identity() -> None:
    cloud = load_multi_robot_cloud_config("configs/multi_robot_cloud.toml")
    edge = load_multi_robot_edge_config("configs/multi_robot_edge.toml")

    assert cloud.provider.max_batch_size == 1
    assert cloud.provider.model == "toy_single_forward"
    assert edge.identity.robot_id == "robot-demo"
    assert edge.provider.max_batch_size == 1
    assert edge.cloud_host == "localhost"
    assert edge.registration_timeout_s == 2.0
    assert edge.physical_host_id != edge.identity.edge_node_id
    assert edge.observation_mode == "target_vector"
    assert edge.observation_seed == 0
    assert edge.observation_language_length == 48


def test_real_smolvla_topology_configs_share_one_action_contract() -> None:
    cloud = load_multi_robot_cloud_config("configs/smolvla_multi_robot_cloud.toml")
    edge = load_multi_robot_edge_config("configs/smolvla_multi_robot_edge.toml")

    assert cloud.provider.model == edge.provider.model == "smolvla"
    assert cloud.action_space_id == edge.identity.action_space_id
    assert cloud.supported_embodiments == (edge.identity.embodiment,) == ("so100",)
    assert cloud.provider.device == "cpu"
    assert edge.provider.device == "cuda:0"
    assert edge.registration_timeout_s == 2.0
    assert cloud.provider.dtype is edge.provider.dtype is None
    assert cloud.provider.multi_tenant_safe is True
    assert edge.observation_mode == "adapter_synthetic"
    assert cloud.provider.package_options["stats_variant"] == "so100"
    assert edge.provider.package_options["stats_variant"] == "so100"
    assert cloud.provider.package_options["default_num_steps"] == 10


def test_multi_robot_edge_loads_adapter_synthetic_observation_options(
    tmp_path,
) -> None:
    config_path = tmp_path / "edge.toml"
    config_path.write_text(
        """
[edge_provider]
model = "smolvla"
checkpoint = "/models/smolvla"

[edge_provider.package_options]
vlm_base_path = "/models/smolvlm2"
stats_variant = "so100"

[observation]
mode = "adapter_synthetic"
seed = 41
language_length = 32
""",
        encoding="utf-8",
    )

    config = load_multi_robot_edge_config(config_path)

    assert config.provider.checkpoint == "/models/smolvla"
    assert config.provider.package_options == {
        "vlm_base_path": "/models/smolvlm2",
        "stats_variant": "so100",
    }
    assert config.observation_mode == "adapter_synthetic"
    assert config.observation_seed == 41
    assert config.observation_language_length == 32


def test_adapter_synthetic_observation_uses_tick_scoped_seed() -> None:
    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls = []

        def synthetic_batch(self, **options):
            self.calls.append(options)
            return {"synthetic": options["seed"]}

    adapter = RecordingAdapter()
    runtime = SimpleNamespace(
        edge=SimpleNamespace(adapter=adapter),
    )
    config = MultiRobotEdgeConfig(
        identity=RobotSessionIdentity(
            robot_id="robot-synthetic",
            edge_node_id="edge-synthetic",
            session_id="session-synthetic",
            embodiment="so100",
            action_space_id="smolvla-so100-actions-v1",
        ),
        observation_mode="adapter_synthetic",
        observation_seed=100,
        observation_language_length=48,
    )

    observation = build_demo_observation(runtime, config, tick=3)

    assert observation == {"synthetic": 102}
    assert adapter.calls == [
        {
            "batch_size": 1,
            "language_length": 48,
            "seed": 102,
        }
    ]


def test_target_vector_observation_remains_the_default() -> None:
    runtime = SimpleNamespace(
        edge=SimpleNamespace(
            capabilities=SimpleNamespace(
                model=SimpleNamespace(action_dim=2),
            )
        ),
    )
    config = MultiRobotEdgeConfig(
        identity=RobotSessionIdentity(
            robot_id="robot-target",
            edge_node_id="edge-target",
            session_id="session-target",
            embodiment="toy-vector",
            action_space_id="toy-vector-actions-v1",
        ),
        observation_offset=10.0,
    )

    assert build_demo_observation(runtime, config, tick=2) == {"target": [12.0, 12.5]}


def test_multi_robot_edge_cli_overrides_model_and_synthetic_input(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "edge.toml"
    config_path.write_text(
        """
[edge_provider]
model = "smolvla"
checkpoint = "/config/smolvla"

[edge_provider.package_options]
vlm_base_path = "/config/smolvlm2"
stats_variant = "so100"
""",
        encoding="utf-8",
    )
    captured = []

    def fake_run(config):
        captured.append(config)
        return []

    monkeypatch.setattr(multi_robot_edge_app, "run_multi_robot_edge", fake_run)

    assert (
        multi_robot_edge_app.main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                "/cli/smolvla",
                "--vlm-base-path",
                "/cli/smolvlm2",
                "--observation-mode",
                "adapter_synthetic",
                "--observation-seed",
                "73",
                "--observation-language-length",
                "24",
            ]
        )
        == 0
    )

    config = captured[0]
    assert config.provider.checkpoint == "/cli/smolvla"
    assert config.provider.package_options == {
        "vlm_base_path": "/cli/smolvlm2",
        "stats_variant": "so100",
    }
    assert config.observation_mode == "adapter_synthetic"
    assert config.observation_seed == 73
    assert config.observation_language_length == 24
    assert capsys.readouterr().out.strip() == "[]"


def test_multi_robot_edge_cli_can_omit_large_action_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "edge.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        multi_robot_edge_app,
        "run_multi_robot_edge",
        lambda config: [
            {
                "robot_id": config.identity.robot_id,
                "tick": 1,
                "source": "edge",
                "output": {"actions": [[1.0, 2.0]]},
            }
        ],
    )

    assert (
        multi_robot_edge_app.main(
            [
                "--config",
                str(config_path),
                "--omit-output",
            ]
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert '"source": "edge"' in printed
    assert '"output"' not in printed
