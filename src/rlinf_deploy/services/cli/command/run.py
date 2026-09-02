"""Route a prompt to one configured robot-policy runtime."""

from __future__ import annotations

import argparse
import json
import math
import posixpath
from collections.abc import Mapping
from typing import Any

from rlinf_deploy.robots.lerobot.so101 import SO101Config

from ...config import config_digest
from ...executor import Command, CommandResult
from ...state import StateStore
from ..context import CommandContext


class RunError(RuntimeError):
    """A configured runtime cannot be started safely."""


DEFAULT_MAX_STEPS = 1
DEFAULT_CONTROL_HZ = 5.0
DEFAULT_REQUEST_TIMEOUT_S = 60.0
_SUPPORTED_BINDING = "lerobot.so101.pi05"


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "run",
        help="execute one configured robot-policy runtime",
    )
    parser.add_argument(
        "--runtime",
        required=True,
        help="configured runtime ID to execute",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="instruction sent unchanged to the configured inference runtime",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=DEFAULT_MAX_STEPS,
        metavar="N",
        help=f"maximum actions to execute (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--control-hz",
        type=_positive_number,
        default=DEFAULT_CONTROL_HZ,
        metavar="HZ",
        help=f"maximum control-loop rate (default: {DEFAULT_CONTROL_HZ:g})",
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_number,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        metavar="SECONDS",
        help=(
            "timeout for one inference request "
            f"(default: {DEFAULT_REQUEST_TIMEOUT_S:g})"
        ),
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    if not args.prompt.strip():
        raise RunError("--prompt must not be empty")

    runtime = next(
        (
            item
            for item in context.deployment.runtimes
            if item.runtime_id == args.runtime
        ),
        None,
    )
    if runtime is None:
        available = ", ".join(item.runtime_id for item in context.deployment.runtimes)
        raise RunError(
            f"unknown runtime {args.runtime!r}; available runtimes: "
            f"{available or 'none'}"
        )
    if runtime.binding != _SUPPORTED_BINDING:
        raise RunError(f"runtime binding {runtime.binding!r} has no executable runner")

    progress = context.progress
    progress.begin("run", context.deployment.name)
    progress.add_node(runtime.node, total=3)
    try:
        progress.update(
            runtime.node, "Validating runtime state", detail=runtime.runtime_id
        )
        state = StateStore(context.state_path).load()
        if state is None:
            raise RunError("deployment is not initialized; run `rlinf-deploy ... init`")
        if state.config_digest != config_digest(context.config):
            raise RunError("configuration changed since init; run init again")
        environment = state.environments.get(runtime.environment_id)
        if environment is None or environment.status != "ready":
            raise RunError(
                f"environment {runtime.environment_id!r} is not ready; run init again"
            )
        service = state.services.get(runtime.model)
        if service is None or service.status != "running":
            raise RunError(
                f"model service {runtime.model!r} is not running; "
                "run `rlinf-deploy ... up`"
            )
        node = state.nodes.get(runtime.node)
        if node is None:
            raise RunError(f"initialized state is missing node {runtime.node!r}")

        robot = context.config.robots[runtime.robot]
        try:
            robot_config = SO101Config.from_yaml(robot.robot_id, robot.options)
        except (TypeError, ValueError) as error:
            raise RunError(str(error)) from error
        payload = _so101_pi05_payload(
            runtime_id=runtime.runtime_id,
            prompt=args.prompt,
            model_endpoint=runtime.model_endpoint,
            robot_config=robot_config,
            options=robot.options,
            max_steps=args.max_steps,
            control_hz=args.control_hz,
            request_timeout_s=args.request_timeout,
        )
        progress.advance(runtime.node)

        python = posixpath.join(environment.path, "bin", "python")
        command_timeout_s = max(
            60.0,
            args.max_steps * args.request_timeout
            + max(0, args.max_steps - 1) / args.control_hz
            + 30.0,
        )
        with context.executor(runtime.node) as executor:
            progress.update(runtime.node, "Checking runtime executable")
            available = executor.run(Command(("test", "-x", python)), check=False)
            if available.exit_code != 0:
                raise RunError(
                    f"runtime Python is missing or not executable: {python}; "
                    "run init again"
                )
            progress.advance(runtime.node)

            progress.update(
                runtime.node,
                "Executing robot-policy loop",
                detail=f"{args.max_steps} step(s)",
            )
            result = executor.run(
                Command(
                    (
                        python,
                        "-m",
                        "rlinf_deploy.bindings.lerobot.so101.pi05.runner",
                        "--spec-json",
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    ),
                    cwd=node.deploy_project,
                    timeout_s=command_timeout_s,
                ),
                check=False,
            )
            if result.exit_code != 0:
                raise RunError(
                    f"runtime {runtime.runtime_id!r} failed: {_failure_detail(result)}"
                )
            progress.advance(runtime.node)
        progress.succeed(
            runtime.node,
            detail=f"Completed {args.max_steps} step(s)",
        )
    except BaseException as error:
        progress.fail(runtime.node, error)
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    return 0


def _so101_pi05_payload(
    *,
    runtime_id: str,
    prompt: str,
    model_endpoint: str,
    robot_config: SO101Config,
    options: Mapping[str, Any],
    max_steps: int,
    control_hz: float,
    request_timeout_s: float,
) -> dict[str, object]:
    sensors = options.get("sensors")
    if not isinstance(sensors, Mapping) or not sensors:
        raise RunError(
            f"robot {robot_config.robot_id!r} requires at least one camera sensor"
        )
    cameras: list[dict[str, object]] = []
    for name, value in sensors.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise RunError(
                f"robot {robot_config.robot_id!r} sensors must be named objects"
            )
        if value.get("type") != "v4l2":
            raise RunError(
                f"robot {robot_config.robot_id!r} sensor {name!r} must use type 'v4l2'"
            )
        cameras.append(
            {
                "name": name,
                "device": _option_string(value, "device", f"sensor {name!r}"),
                "width": _option_integer(value, "width", f"sensor {name!r}"),
                "height": _option_integer(value, "height", f"sensor {name!r}"),
                "fps": _option_number(value, "fps", f"sensor {name!r}"),
            }
        )
    return {
        "schema": "rlinf.runtime.so101-pi05.v1",
        "runtime_id": runtime_id,
        "prompt": prompt,
        "model_endpoint": model_endpoint,
        **robot_config.to_runtime_payload(),
        "cameras": cameras,
        "max_steps": max_steps,
        "control_hz": control_hz,
        "request_timeout_s": request_timeout_s,
    }


def _option_string(options: Mapping[str, Any], name: str, context: str) -> str:
    value = options.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RunError(f"{context} requires a non-empty {name!r}")
    return value


def _option_integer(options: Mapping[str, Any], name: str, context: str) -> int:
    value = options.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunError(f"{context} requires a positive integer {name!r}")
    return value


def _option_number(
    options: Mapping[str, Any],
    name: str,
    context: str,
) -> float:
    value = options.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunError(f"{context} requires a positive number {name!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RunError(f"{context} requires a positive number {name!r}")
    return number


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_number(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def _failure_detail(result: CommandResult) -> str:
    for line in reversed(result.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("event") == "error"
            and isinstance(payload.get("error"), str)
            and payload["error"].strip()
        ):
            return payload["error"].strip()
    lines = (result.stderr or result.stdout).strip().splitlines()
    if lines:
        return lines[-1]
    return f"remote process exited with code {result.exit_code}"


__all__ = ["RunError", "register", "run"]
