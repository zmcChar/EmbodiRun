"""Integrate independently implemented deployment subcommands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..config import ConfigError, NodeConfig, load_config
from ..environment import EnvironmentError
from ..executor import Executor, executor_for
from ..service import ServiceError, build_plan
from .command import down, init, probe, run, up, validate
from .context import CommandContext, state_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlinf-deploy")
    parser.add_argument("--config", required=True, help="deployment YAML path")
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="local state directory (default: ~/.local/state/rlinf-deploy)",
    )
    run.register(parser)
    commands = parser.add_subparsers(dest="command")
    validate.register(commands)
    probe.register(commands)
    init.register(commands)
    up.register(commands)
    down.register(commands)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    executor_factory: Callable[[NodeConfig], Executor] = executor_for,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        if not run.selected(args):
            parser.error("a command or --runtime/--prompt invocation is required")
        args.command_handler = run.run
    elif run.selected(args):
        parser.error("--runtime, --prompt, and --execute cannot be used with a subcommand")
    try:
        config = load_config(args.config)
        deployment = build_plan(config)
    except (ConfigError, EnvironmentError, ServiceError, RuntimeError) as error:
        print(f"rlinf-deploy: error: {error}", file=sys.stderr)
        return 2

    context = CommandContext(
        config=config,
        deployment=deployment,
        state_path=state_path(args.state_dir, deployment.name),
        executor_factory=executor_factory,
    )
    try:
        return args.command_handler(args, context)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"rlinf-deploy: error: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
