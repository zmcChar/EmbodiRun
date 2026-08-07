"""CLI composition root for one robot edge against a shared cloud model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace

from embodied_runtime.distributed.session import RobotSessionIdentity

from .multi_robot.edge_demo import (
    build_demo_observation as _build_demo_observation,
)
from .multi_robot.edge_demo import run_multi_robot_edge
from .multi_robot.edge_runtime import RobotEdgeRuntime
from .multi_robot.edge_settings import (
    OBSERVATION_MODES,
    MultiRobotEdgeConfig,
    load_multi_robot_edge_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one robot-scoped edge model against a shared cloud model."
    )
    parser.add_argument("--config", default="configs/multi_robot_edge.toml")
    parser.add_argument("--robot-id")
    parser.add_argument("--edge-node-id")
    parser.add_argument("--session-id")
    parser.add_argument("--physical-host-id")
    parser.add_argument("--physical-resource-id")
    parser.add_argument("--cloud-host")
    parser.add_argument("--cloud-port", type=int)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vlm-base-path")
    parser.add_argument("--observation-mode", choices=tuple(sorted(OBSERVATION_MODES)))
    parser.add_argument("--observation-offset", type=float)
    parser.add_argument("--observation-seed", type=int)
    parser.add_argument("--observation-language-length", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--disconnect-tick", type=int)
    parser.add_argument("--reconnect-tick", type=int)
    parser.add_argument(
        "--omit-output",
        action="store_true",
        help="omit the action matrix from CLI JSON while retaining source and timing records",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_multi_robot_edge_config(args.config)
    identity = RobotSessionIdentity(
        robot_id=args.robot_id or config.identity.robot_id,
        edge_node_id=args.edge_node_id or config.identity.edge_node_id,
        session_id=args.session_id or config.identity.session_id,
        embodiment=config.identity.embodiment,
        action_space_id=config.identity.action_space_id,
    )
    provider = config.provider
    if args.device is not None:
        provider = replace(provider, device=args.device)
    if args.checkpoint is not None:
        provider = replace(provider, checkpoint=args.checkpoint)
    if args.vlm_base_path is not None:
        provider = replace(
            provider,
            package_options={**provider.package_options, "vlm_base_path": args.vlm_base_path},
        )
    config = replace(
        config,
        identity=identity,
        provider=provider,
        physical_host_id=args.physical_host_id or config.physical_host_id,
        physical_resource_id=args.physical_resource_id or config.physical_resource_id,
        cloud_host=args.cloud_host or config.cloud_host,
        cloud_port=config.cloud_port if args.cloud_port is None else args.cloud_port,
        observation_mode=(
            config.observation_mode if args.observation_mode is None else args.observation_mode
        ),
        observation_offset=(
            config.observation_offset
            if args.observation_offset is None
            else args.observation_offset
        ),
        observation_seed=(
            config.observation_seed if args.observation_seed is None else args.observation_seed
        ),
        observation_language_length=(
            config.observation_language_length
            if args.observation_language_length is None
            else args.observation_language_length
        ),
        ticks=config.ticks if args.ticks is None else args.ticks,
        disconnect_tick=(
            config.disconnect_tick if args.disconnect_tick is None else args.disconnect_tick
        ),
        reconnect_tick=(
            config.reconnect_tick if args.reconnect_tick is None else args.reconnect_tick
        ),
    )
    records = run_multi_robot_edge(config)
    if args.omit_output:
        records = [
            {name: value for name, value in record.items() if name != "output"}
            for record in records
        ]
    print(json.dumps(records, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "MultiRobotEdgeConfig",
    "RobotEdgeRuntime",
    "_build_demo_observation",
    "build_parser",
    "load_multi_robot_edge_config",
    "main",
    "run_multi_robot_edge",
]


if __name__ == "__main__":
    raise SystemExit(main())
