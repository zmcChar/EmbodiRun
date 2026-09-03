"""Submit one prompt task to a configured control-node service."""

from __future__ import annotations

import argparse
import math
import uuid
from typing import Any

from rlinf_deploy.bindings import binding_definition
from rlinf_deploy.services.control.contracts import TaskRequest

from ...config import config_digest
from ...control import ControlClient
from ...state import StateStore
from ..context import CommandContext


class RunError(RuntimeError):
    """A configured runtime cannot accept or safely execute the requested task."""


DEFAULT_MAX_STEPS = 1
DEFAULT_CHUNK_STEPS = 10
DEFAULT_CONTROL_HZ = 5.0
DEFAULT_REQUEST_TIMEOUT_S = 60.0


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "run",
        help="submit one task to a configured control runtime",
    )
    parser.add_argument("--runtime", required=True, help="configured runtime ID")
    parser.add_argument(
        "--prompt",
        required=True,
        help="instruction sent unchanged through Control to Inference",
    )
    parser.add_argument(
        "--chunk-steps",
        type=_positive_integer,
        default=DEFAULT_CHUNK_STEPS,
        metavar="N",
        help=(
            "actions to execute from each inference chunk "
            f"(default: {DEFAULT_CHUNK_STEPS})"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=DEFAULT_MAX_STEPS,
        metavar="N",
        help=(
            "maximum inference/action chunks to execute "
            f"(default: {DEFAULT_MAX_STEPS})"
        ),
    )
    parser.add_argument(
        "--control-hz",
        type=_positive_number,
        default=DEFAULT_CONTROL_HZ,
        metavar="HZ",
        help=f"action playback rate (default: {DEFAULT_CONTROL_HZ:g})",
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
    definition = binding_definition(runtime.binding)
    if args.chunk_steps > definition.maximum_chunk_steps:
        raise RunError(
            f"--chunk-steps {args.chunk_steps} exceeds binding "
            f"{runtime.binding!r} maximum {definition.maximum_chunk_steps}"
        )

    progress = context.progress
    progress.begin("run", context.deployment.name)
    progress.add_node(runtime.node, total=2)
    try:
        progress.update(
            runtime.node, "Validating control service", detail=runtime.runtime_id
        )
        state = StateStore(context.state_path).load()
        if state is None:
            raise RunError("deployment is not initialized; run `rlinf-deploy ... init`")
        if state.config_digest != config_digest(context.config):
            raise RunError("configuration changed since init; run init again")
        if state.deploy_commit != context.deployment.deploy_commit:
            raise RunError("Deploy revision changed since init; run init again")
        if state.inference_commit != context.deployment.inference_commit:
            raise RunError("Inference revision changed since init; run init again")
        environment = state.environments.get(runtime.environment_id)
        if environment is None or environment.status != "ready":
            raise RunError(
                f"environment {runtime.environment_id!r} is not ready; run init again"
            )
        service = state.services.get(runtime.control_service_id)
        if service is None or service.status != "running":
            raise RunError(
                f"control service {runtime.control_service_id!r} is not running; "
                "run `rlinf-deploy ... up`"
            )
        if service.node != runtime.node or service.endpoint != runtime.control_endpoint:
            raise RunError(
                f"control service {runtime.control_service_id!r} does not match "
                "the configured runtime; restart the deployment"
            )
        if runtime.node not in state.nodes:
            raise RunError(f"initialized state is missing node {runtime.node!r}")
        request = TaskRequest(
            request_id=f"task-{uuid.uuid4().hex}",
            runtime_id=runtime.runtime_id,
            prompt=args.prompt,
            chunk_steps=args.chunk_steps,
            max_steps=args.max_steps,
            control_hz=args.control_hz,
            inference_timeout_s=args.request_timeout,
        )
        progress.advance(runtime.node)

        task_timeout_s = max(
            60.0,
            args.max_steps * args.request_timeout
            + args.max_steps * max(0, args.chunk_steps - 1) / args.control_hz
            + 30.0,
        )
        progress.update(
            runtime.node,
            "Executing control task",
            detail=f"{args.max_steps} chunk(s), {args.chunk_steps} action(s) each",
        )
        with context.executor(runtime.node) as executor:
            result = ControlClient(executor, runtime.control_endpoint).run(
                request,
                timeout_s=task_timeout_s,
            )
        progress.advance(runtime.node)
        progress.succeed(
            runtime.node,
            detail=(
                f"Completed {result.completed_steps} chunk(s), "
                f"{args.chunk_steps} action(s) each"
            ),
        )
    except BaseException as error:
        progress.fail(runtime.node, error)
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    return 0


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


__all__ = ["RunError", "register", "run"]
