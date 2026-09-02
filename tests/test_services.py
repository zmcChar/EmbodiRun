import json
import sys
from pathlib import Path
from threading import Barrier

import pytest

from rlinf_deploy.services.cli import main
from rlinf_deploy.services.cli.command.init import (
    DEPLOY_REPOSITORY,
    INFERENCE_REPOSITORY,
)
from rlinf_deploy.services.config import ConfigError, load_config
from rlinf_deploy.services.environment import (
    ProjectManager,
    UvEnvironmentManager,
    environment_profiles,
)
from rlinf_deploy.services.executor import Command, CommandResult, LocalExecutor
from rlinf_deploy.services.service import ServiceError, ServiceSupervisor, build_plan
from rlinf_deploy.services.state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateError,
    StateStore,
)

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "muti-nodes.example.yaml"


class RecordingExecutor:
    def __init__(self) -> None:
        self.commands = []

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        return CommandResult(0, "ready\n", "")

    def close(self) -> None:
        return None


class ResultExecutor:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.commands = []

    def run(self, command, *, check=True):
        self.commands.append(command)
        return self.results.pop(0)

    def close(self) -> None:
        return None


class FakeNodeExecutor:
    def __init__(self) -> None:
        self.commands = []
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        argv = command.argv
        if argv[:2] == ("sh", "-c") and "uname -s" in argv[2]:
            return CommandResult(
                0,
                "/home/user\tLinux\taarch64\t/usr/bin/python3\t3.10.12\t"
                "/usr/bin/git\t/home/user/.local/bin/uv\n",
                "",
            )
        if "remote" in argv and "get-url" in argv:
            repository = (
                DEPLOY_REPOSITORY
                if argv[2].endswith("/deploy")
                else INFERENCE_REPOSITORY
            )
            return CommandResult(0, f"{repository}\n", "")
        if "rev-parse" in argv and "HEAD" in argv:
            return CommandResult(0, "resolved\nresolved\n", "")
        if argv[:2] == ("sh", "-c"):
            return CommandResult(0, "running\t321\n", "")
        return CommandResult(0, "", "")

    def close(self) -> None:
        self.closed = True


class FailingExecutor:
    def __init__(self) -> None:
        self.closed = False

    def run(self, command, *, check=True):
        raise RuntimeError("connection refused")

    def close(self) -> None:
        self.closed = True


class ServiceReadinessExecutor:
    def __init__(
        self,
        *,
        process_state="running",
        health_exit_code=0,
    ) -> None:
        self.process_state = process_state
        self.health_exit_code = health_exit_code
        self.commands = []
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        if command.argv[:2] == ("test", "-x"):
            return CommandResult(0)
        if command.argv[:2] == ("sh", "-c"):
            if "nohup" in command.argv[2]:
                return CommandResult(0, "running\t321\n")
            return CommandResult(0, f"{self.process_state}\t321\n")
        if any(argument.endswith("/healthz") for argument in command.argv):
            return CommandResult(self.health_exit_code)
        raise AssertionError(f"unexpected command: {command.argv!r}")

    def close(self) -> None:
        self.closed = True


class RuntimeExecutor:
    def __init__(self, result=None) -> None:
        self.commands = []
        self.closed = False
        self.result = result or CommandResult(
            0,
            '{"event":"step","step":1}\n{"event":"complete","steps":1}\n',
        )

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        if command.argv[:2] == ("test", "-x"):
            return CommandResult(0)
        if command.argv[1:4] == (
            "-m",
            "rlinf_deploy.bindings.lerobot.so101.pi05.worker",
            "--spec-json",
        ):
            return self.result
        raise AssertionError(f"unexpected command: {command.argv!r}")

    def close(self) -> None:
        self.closed = True


class DownExecutor:
    def __init__(self) -> None:
        self.commands = []
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        if command.argv[:2] == ("sh", "-c"):
            return CommandResult(0, "stopped\n")
        raise AssertionError(f"unexpected command: {command.argv!r}")

    def close(self) -> None:
        self.closed = True


class ConcurrentInitExecutor(FakeNodeExecutor):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def run(self, command, *, check=True):
        if command.argv[:2] == ("sh", "-c") and "uname -s" in command.argv[2]:
            self.barrier.wait(timeout=1.0)
        return super().run(command, check=check)


class ConcurrentUpExecutor(ServiceReadinessExecutor):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def run(self, command, *, check=True):
        if command.argv[:2] == ("test", "-x"):
            self.barrier.wait(timeout=1.0)
        return super().run(command, check=check)


class ConcurrentDownExecutor(DownExecutor):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def run(self, command, *, check=True):
        if command.argv[:2] == ("sh", "-c"):
            self.barrier.wait(timeout=1.0)
        return super().run(command, check=check)


def two_node_model_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "two-node-models.yaml"
    second_node = """
  jetson-worker:
    type: jetson.test
    connection:
      type: local
"""
    second_model = """
  pi05-02:
    backend: vvla
    transport: http
    type: pi05
    node: jetson-worker
    environment: .venv-vvla
    source: /models/pi05-02
    adapter_config: /configs/pi05-02.json
    server:
      bind: 127.0.0.1
      port: 8001
"""
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8")
        .replace("\nrobots:\n", f"{second_node}\nrobots:\n")
        .replace("\nruntimes:\n", f"{second_model}\nruntimes:\n"),
        encoding="utf-8",
    )
    return config_path


def test_example_resolves_real_thor_environment_and_runtime() -> None:
    config = load_config(EXAMPLE)
    profiles = environment_profiles(config)
    plan = build_plan(config)

    assert config.metadata.name == "thor-so101-pi05"
    assert len(profiles) == 2
    assert {(item.project, item.group) for item in profiles} == {
        ("deploy", "robot-so101"),
        ("inference", "pi05"),
    }
    assert len(plan.services) == 1
    assert plan.services[0].health_endpoint == "http://127.0.0.1:8000/healthz"
    assert plan.services[0].command.argv == (
        "vvla-http-serve",
        "--policy",
        "pi05",
        "--checkpoint",
        "/home/user/models/pi05_so101",
        "--adapter-config",
        "/home/user/.config/rlinf-deploy/pi05_so101_http_serve.json",
        "--device",
        "cuda:0",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    )
    assert {runtime.runtime_id for runtime in plan.runtimes} == {"so101-1-runtime"}
    assert {runtime.model_endpoint for runtime in plan.runtimes} == {
        "http://127.0.0.1:8000"
    }


def test_uv_environment_manager_uses_only_the_selected_group() -> None:
    profile = next(
        item
        for item in environment_profiles(load_config(EXAMPLE))
        if item.project == "deploy"
    )
    executor = RecordingExecutor()
    result = UvEnvironmentManager(executor).prepare(
        profile,
        project_dir="/opt/rlinf-deploy",
    )

    command, check = executor.commands[0]
    assert result.stdout == "ready\n"
    assert check is True
    assert command.argv == (
        "uv",
        "sync",
        "--python",
        "3.12",
        "--frozen",
        "--no-dev",
        "--group",
        "robot-so101",
    )
    assert command.cwd == "/opt/rlinf-deploy"
    assert command.environment == {"UV_PROJECT_ENVIRONMENT": ".venv-robot-so101"}


def test_uv_environment_manager_applies_configured_package_overlay() -> None:
    profile = next(
        item
        for item in environment_profiles(load_config(EXAMPLE))
        if item.project == "inference"
    )
    executor = RecordingExecutor()

    UvEnvironmentManager(executor).prepare(
        profile,
        project_dir="/opt/rlinf-inference",
    )

    assert len(executor.commands) == 2
    command, check = executor.commands[1]
    assert check is True
    assert command.argv == (
        "uv",
        "pip",
        "install",
        "--python",
        "/opt/rlinf-inference/.venv-vvla/bin/python",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
        "torch==2.10.0+cu130",
        "torchvision==0.25.0+cu130",
    )
    assert command.timeout_s == 1800.0


def test_project_manager_clones_and_fetches_only_when_missing() -> None:
    repository = "https://example.com/project.git"
    executor = ResultExecutor(
        CommandResult(0),
        CommandResult(1),
        CommandResult(0),
        CommandResult(0, f"{repository}\n"),
        CommandResult(1),
        CommandResult(0),
        CommandResult(0),
        CommandResult(0, "resolved\nresolved\n"),
    )

    ProjectManager(executor).prepare(
        repository=repository,
        revision="abc1234",
        project_dir="/opt/project",
    )

    argv = [command.argv for command in executor.commands]
    assert ("git", "clone", "--no-checkout", repository, "/opt/project") in argv
    assert ("git", "-C", "/opt/project", "fetch", "origin", "abc1234") in argv


def test_local_executor_preserves_argv_and_explicit_environment(tmp_path) -> None:
    result = LocalExecutor().run(
        Command(
            (
                sys.executable,
                "-c",
                (
                    "import os,sys; print(os.getcwd()); print(sys.argv[1]); "
                    "print(os.environ['RLINF_TEST_VALUE'])"
                ),
                "argument with spaces",
            ),
            cwd=str(tmp_path),
            environment={"RLINF_TEST_VALUE": "value with spaces"},
        )
    )

    assert result.stdout.splitlines() == [
        str(tmp_path),
        "argument with spaces",
        "value with spaces",
    ]


def test_service_supervisor_uses_identity_checked_pid_lifecycle() -> None:
    service = build_plan(load_config(EXAMPLE)).services[0]
    executor = ResultExecutor(
        CommandResult(0, "running\t314\n", ""),
        CommandResult(0, "running\t314\n", ""),
        CommandResult(0, "stopped\n", ""),
    )
    supervisor = ServiceSupervisor(
        executor,
        run_root=".local/state/rlinf-deploy/run",
        log_root=".local/state/rlinf-deploy/logs",
    )

    started = supervisor.start(service)
    running = supervisor.status(service.service_id)
    stopped = supervisor.stop(service.service_id)

    assert started.state == running.state == "running"
    assert started.pid == running.pid == 314
    assert stopped.state == "stopped"
    assert all(command.argv[:2] == ("sh", "-c") for command in executor.commands)
    start_script = executor.commands[0].argv[2]
    assert "RLINF_DEPLOY_SERVICE_ID=pi05-01" in start_script
    assert "PID belongs to another process" in start_script
    assert executor.commands[-1].timeout_s == 10.0


def test_service_supervisor_rejects_unsafe_service_id() -> None:
    supervisor = ServiceSupervisor(
        ResultExecutor(),
        run_root="run",
        log_root="logs",
    )

    with pytest.raises(ServiceError, match="service IDs"):
        supervisor.status("../other-process")


def test_duplicate_yaml_keys_are_rejected(tmp_path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        "metadata:\n"
        "  name: one\n"
        "  name: two\n"
        "  deploy-commit: abc\n"
        "  inference-commit: def\n"
        "nodes: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_config(config_path)


def test_unknown_node_reference_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "unknown-node.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "node: jetson-agx-thor-232",
            "node: missing-node",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="references unknown node"):
        load_config(config_path)


def test_two_robots_cannot_claim_the_same_device_port(tmp_path) -> None:
    config_path = tmp_path / "port-conflict.yaml"
    duplicate_robot = """
  so101-conflict:
    type: lerobot.so101
    node: jetson-agx-thor-232
    port: /dev/ttyACM0
"""
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "\nsensors:\n",
            f"{duplicate_robot}\nsensors:\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="share port"):
        load_config(config_path)


def test_cli_init_then_up_uses_persisted_initialized_state(tmp_path, capsys) -> None:
    executors = []

    def factory(_node):
        executor = FakeNodeExecutor()
        executors.append(executor)
        return executor

    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(tmp_path),
    )
    init_exit_code = main((*base_args, "init"), executor_factory=factory)
    init_output = capsys.readouterr()

    assert init_exit_code == 0
    assert init_output.err == ""
    assert "Init deployment thor-so101-pi05" in init_output.out
    assert "2 environments ready" in init_output.out
    initialized = StateStore(tmp_path / "thor-so101-pi05.json").load()
    assert initialized is not None
    assert {item.group for item in initialized.environments.values()} == {
        "pi05",
        "robot-so101",
    }
    init_argv = [command.argv for command, _check in executors[0].commands]
    assert any(argv[-2:] == ("--group", "pi05") for argv in init_argv)
    assert any(argv[-2:] == ("--group", "robot-so101") for argv in init_argv)
    assert executors[0].closed is True

    up_exit_code = main((*base_args, "up"), executor_factory=factory)
    up_output = capsys.readouterr()

    assert up_exit_code == 0
    assert up_output.err == ""
    assert "Starting model" in up_output.out
    assert "1/1 services running" in up_output.out
    started = StateStore(tmp_path / "thor-so101-pi05.json").load()
    assert started is not None
    assert started.services["pi05-01"] == ServiceState(
        service_id="pi05-01",
        node="jetson-agx-thor-232",
        status="running",
        pid=321,
        endpoint="http://127.0.0.1:8000",
    )
    up_commands = [command for command, _check in executors[1].commands]
    start_script = next(
        command.argv[2] for command in up_commands if command.argv[:2] == ("sh", "-c")
    )
    assert "/sources/inference/.venv-vvla/bin/vvla-http-serve" in start_script
    assert "/home/user/.config/rlinf-deploy/pi05_so101_http_serve.json" in start_script
    health_command, health_check = next(
        (command, check)
        for command, check in executors[1].commands
        if any(argument.endswith("/healthz") for argument in command.argv)
    )
    assert health_command.argv[-2] == "http://127.0.0.1:8000/healthz"
    assert health_check is False
    assert "123456" not in up_output.out


def test_cli_routes_prompt_to_selected_runtime(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert (
        main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor())
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    executor = RuntimeExecutor()

    exit_code = main(
        (
            *base_args,
            "run",
            "--runtime",
            "so101-1-runtime",
            "--prompt",
            "把红色积木放进盒子",
        ),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "Run deployment thor-so101-pi05" in captured.out
    assert "Executing robot-policy loop" in captured.out
    assert "Completed 1 step(s)" in captured.out
    command = executor.commands[-1][0]
    assert command.argv[1:4] == (
        "-m",
        "rlinf_deploy.bindings.lerobot.so101.pi05.worker",
        "--spec-json",
    )
    payload = json.loads(command.argv[4])
    assert payload["prompt"] == "把红色积木放进盒子"
    assert payload["model_endpoint"] == "http://127.0.0.1:8000"
    assert payload["max_steps"] == 1
    assert payload["max_joint_step_deg"] == 5.0
    assert payload["max_gripper_step"] == 10.0
    assert payload["step_limit_mode"] == "clip"
    assert {item["name"] for item in payload["inputs"]} == {
        "observation.images.front",
        "observation.images.wrist",
    }
    assert {item["sensor_id"] for item in payload["inputs"]} == {
        "front-camera",
        "wrist-camera",
    }
    assert {item["type"] for item in payload["inputs"]} == {"v4l2"}
    assert executor.closed is True


def test_cli_runtime_surfaces_remote_binding_error(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert (
        main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor())
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    executor = RuntimeExecutor(
        CommandResult(
            1,
            '{"event":"error","error":"joint step exceeds 12 degrees"}\n',
            "remote warning",
        )
    )

    exit_code = main(
        (
            *base_args,
            "run",
            "--runtime",
            "so101-1-runtime",
            "--prompt",
            "move",
        ),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "joint step exceeds 12 degrees" in captured.err
    runtime_command, check = executor.commands[-1]
    assert runtime_command.argv[1:3] == (
        "-m",
        "rlinf_deploy.bindings.lerobot.so101.pi05.worker",
    )
    assert check is False


@pytest.mark.parametrize(
    ("process_state", "health_exit_code", "error"),
    [
        ("stale", 1, "exited before becoming ready"),
        ("running", 1, "did not become healthy"),
    ],
)
def test_cli_up_fails_until_service_is_healthy(
    tmp_path,
    capsys,
    process_state,
    health_exit_code,
    error,
) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert (
        main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor())
        == 0
    )
    capsys.readouterr()
    executor = ServiceReadinessExecutor(
        process_state=process_state,
        health_exit_code=health_exit_code,
    )

    exit_code = main(
        (*base_args, "up", "--wait-timeout", "0.01"),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert error in captured.err
    assert "/logs/pi05-01.log" in captured.err
    state = StateStore(state_dir / "thor-so101-pi05.json").load()
    assert state is not None
    assert state.services["pi05-01"] == ServiceState(
        service_id="pi05-01",
        node="jetson-agx-thor-232",
        status="failed",
        pid=321,
        endpoint="http://127.0.0.1:8000",
    )
    assert executor.closed is True


def test_cli_probe_only_checks_connectivity_and_does_not_write_state(
    tmp_path, capsys
) -> None:
    executors = []

    def factory(_node):
        executor = FakeNodeExecutor()
        executors.append(executor)
        return executor

    exit_code = main(
        (
            "--config",
            str(EXAMPLE),
            "--state-dir",
            str(tmp_path),
            "probe",
        ),
        executor_factory=factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "Probe deployment thor-so101-pi05" in captured.out
    assert "jetson-agx-thor-232: Reachable" in captured.out
    assert "1/1 nodes ready" in captured.out
    assert len(executors[0].commands) == 1
    assert list(tmp_path.iterdir()) == []


def test_cli_probe_continues_after_one_node_is_unreachable(tmp_path, capsys) -> None:
    config_path = tmp_path / "two-nodes.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "\nrobots:\n",
            "\n  offline-node:\n"
            "    type: test.node\n"
            "    connection:\n"
            "      type: local\n"
            "\nrobots:\n",
        ),
        encoding="utf-8",
    )

    def factory(node):
        if node.node_id == "offline-node":
            return FailingExecutor()
        return FakeNodeExecutor()

    exit_code = main(
        ("--config", str(config_path), "probe"),
        executor_factory=factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "jetson-agx-thor-232: Reachable" in captured.out
    assert "offline-node: Failed: connection refused" in captured.out
    assert "1/2 nodes ready" in captured.out


def test_cli_lifecycle_runs_different_nodes_concurrently(tmp_path, capsys) -> None:
    config_path = two_node_model_config(tmp_path)
    state_dir = tmp_path / "state"
    arguments = ("--config", str(config_path), "--state-dir", str(state_dir))

    init_barrier = Barrier(2)
    assert (
        main(
            (*arguments, "init"),
            executor_factory=lambda _node: ConcurrentInitExecutor(init_barrier),
        )
        == 0
    )
    init_output = capsys.readouterr()
    assert "2/2 nodes ready" in init_output.out

    up_barrier = Barrier(2)
    assert (
        main(
            (*arguments, "up"),
            executor_factory=lambda _node: ConcurrentUpExecutor(up_barrier),
        )
        == 0
    )
    up_output = capsys.readouterr()
    assert "2/2 nodes ready" in up_output.out
    assert "2/2 services running" in up_output.out

    down_barrier = Barrier(2)
    assert (
        main(
            (*arguments, "down"),
            executor_factory=lambda _node: ConcurrentDownExecutor(down_barrier),
        )
        == 0
    )
    down_output = capsys.readouterr()
    assert "2/2 nodes ready" in down_output.out
    assert "2/2 services stopped" in down_output.out


def test_cli_probe_runs_different_nodes_concurrently(tmp_path, capsys) -> None:
    config_path = two_node_model_config(tmp_path)
    barrier = Barrier(2)

    exit_code = main(
        ("--config", str(config_path), "probe"),
        executor_factory=lambda _node: ConcurrentInitExecutor(barrier),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "2/2 nodes ready" in captured.out


def test_cli_up_requires_successful_matching_init(tmp_path, capsys) -> None:
    exit_code = main(
        (
            "--config",
            str(EXAMPLE),
            "--state-dir",
            str(tmp_path),
            "up",
        ),
        executor_factory=lambda _node: FakeNodeExecutor(),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not initialized" in captured.err


def test_cli_down_stops_services_after_configuration_changes(tmp_path, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    state_dir = tmp_path / "state"
    base_args = ("--config", str(config_path), "--state-dir", str(state_dir))
    assert (
        main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor())
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# safety changed\n",
        encoding="utf-8",
    )
    executor = DownExecutor()

    exit_code = main(
        (*base_args, "down"),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "1/1 services stopped" in captured.out
    stopped = StateStore(state_dir / "thor-so101-pi05.json").load()
    assert stopped is not None
    assert stopped.services["pi05-01"].status == "stopped"
    assert "kill -TERM" in executor.commands[0][0].argv[2]
    assert executor.closed is True


def test_cli_up_rejects_config_changed_after_init(tmp_path, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    state_dir = tmp_path / "state"
    arguments = (
        "--config",
        str(config_path),
        "--state-dir",
        str(state_dir),
    )
    factory = lambda _node: FakeNodeExecutor()
    assert main((*arguments, "init"), executor_factory=factory) == 0
    capsys.readouterr()
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )

    exit_code = main((*arguments, "up"), executor_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "configuration changed since init" in captured.err


def test_state_store_round_trip_is_atomic_and_secret_free(tmp_path) -> None:
    state_path = tmp_path / "state" / "deployment.json"
    state = DeploymentState(
        name="lab",
        config_digest="1234",
        deploy_commit="abc",
        inference_commit="def",
        nodes={
            "node": NodeState(
                node_id="node",
                home="/home/user",
                root="/home/user/.local/share/rlinf-deploy/lab",
                deploy_project="/home/user/project/deploy",
                inference_project="/home/user/project/inference",
                platform="Linux",
                machine="aarch64",
                python="/usr/bin/python3",
                python_version="3.10.12",
            )
        },
        environments={
            "node:deploy:robot-so101": EnvironmentState(
                environment_id="node:deploy:robot-so101",
                node="node",
                project="deploy",
                group="robot-so101",
                path=".venv-robot-so101",
                status="ready",
            )
        },
        services={
            "pi05-01": ServiceState(
                service_id="pi05-01",
                node="node",
                status="running",
                pid=123,
                endpoint="http://127.0.0.1:8000",
            )
        },
    )
    store = StateStore(state_path)

    store.save(state)

    assert store.load() == state
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert "password" not in state_path.read_text(encoding="utf-8").lower()


def test_state_store_rejects_unknown_format(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"version": 999}', encoding="utf-8")

    with pytest.raises(StateError, match="unsupported state version"):
        StateStore(state_path).load()
