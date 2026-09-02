import json
import sys
from pathlib import Path

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


def test_example_resolves_shared_environments_and_two_runtimes() -> None:
    config = load_config(EXAMPLE)
    profiles = environment_profiles(config)
    plan = build_plan(config)

    assert config.metadata.name == "thor-dual-so101-pi05-example"
    assert len(profiles) == 2
    assert {(item.project, item.group) for item in profiles} == {
        ("deploy", "robot-so101"),
        ("inference", "pi05"),
    }
    assert len(plan.services) == 1
    assert plan.services[0].command.argv == (
        "vvla-http-serve",
        "--policy",
        "pi05",
        "--checkpoint",
        "/path/to/pi05-so101-checkpoint",
        "--adapter-config",
        "configs/pi05_so101_http_serve.example.json",
        "--device",
        "cuda:0",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    )
    assert {runtime.runtime_id for runtime in plan.runtimes} == {
        "so101-1-runtime",
        "so101-2-runtime",
    }
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
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "port: /dev/ttyACM1",
            "port: /dev/ttyACM0",
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
    init_payload = json.loads(init_output.out)
    assert init_payload["command"] == "init"
    assert {item["group"] for item in init_payload["environments"]} == {
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
    up_payload = json.loads(up_output.out)
    assert up_payload["command"] == "up"
    assert up_payload["services"] == [
        {
            "endpoint": "http://127.0.0.1:8000",
            "node": "jetson-agx-thor-232",
            "pid": 321,
            "service_id": "pi05-01",
            "status": "running",
        }
    ]
    up_commands = [command for command, _check in executors[1].commands]
    start_script = next(
        command.argv[2] for command in up_commands if command.argv[:2] == ("sh", "-c")
    )
    assert "/sources/inference/.venv-vvla/bin/vvla-http-serve" in start_script
    assert "/sources/deploy/configs/pi05_so101_http_serve.example.json" in start_script
    assert "123456" not in up_output.out


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
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["nodes"][0]["reachable"] is True
    assert payload["nodes"][0]["init_ready"] is True
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
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert {node["node_id"] for node in payload["nodes"]} == {
        "jetson-agx-thor-232",
        "offline-node",
    }
    offline = next(
        node for node in payload["nodes"] if node["node_id"] == "offline-node"
    )
    assert offline["reachable"] is False
    assert offline["error"] == "connection refused"


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
