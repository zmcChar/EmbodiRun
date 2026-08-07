"""CLI and composition root for the multi-robot inference cloud."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import replace

from .multi_robot.cloud_codec import dispatch_multi_robot_request
from .multi_robot.cloud_health import (
    probe_multi_robot_cloud_health,
    validate_multi_robot_health_response,
)
from .multi_robot.cloud_server import (
    run_multi_robot_cloud,
    serve_multi_robot_cloud,
)
from .multi_robot.cloud_settings import (
    MultiRobotCloudConfig,
    load_multi_robot_cloud_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one model to multiple isolated robot sessions."
    )
    parser.add_argument("--config", default="configs/multi_robot_cloud.toml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vlm-base-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_multi_robot_cloud_config(args.config)
    provider = config.provider
    if args.device is not None:
        provider = replace(provider, device=args.device)
    if args.checkpoint is not None:
        provider = replace(provider, checkpoint=args.checkpoint)
    if args.vlm_base_path is not None:
        provider = replace(
            provider,
            package_options={
                **provider.package_options,
                "vlm_base_path": args.vlm_base_path,
            },
        )
    config = replace(
        config,
        provider=provider,
        host=config.host if args.host is None else args.host,
        port=config.port if args.port is None else args.port,
    )
    run_multi_robot_cloud(config)
    return 0


def build_health_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only health probe for a multi-robot cloud service."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18770)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser


def health_main(argv: Sequence[str] | None = None) -> int:
    args = build_health_parser().parse_args(argv)
    try:
        health = asyncio.run(
            probe_multi_robot_cloud_health(
                args.host,
                args.port,
                timeout_s=args.timeout_s,
            )
        )
    except Exception as error:  # noqa: BLE001 - failed probe means unhealthy
        print(
            json.dumps(
                {
                    "healthy": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(health, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "MultiRobotCloudConfig",
    "dispatch_multi_robot_request",
    "health_main",
    "load_multi_robot_cloud_config",
    "main",
    "probe_multi_robot_cloud_health",
    "run_multi_robot_cloud",
    "serve_multi_robot_cloud",
    "validate_multi_robot_health_response",
]


if __name__ == "__main__":
    raise SystemExit(main())
