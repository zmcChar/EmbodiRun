"""CLI composition root for pi0.5 GPU and lightweight CPU collaboration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace

from embodied_runtime.distributed import FailoverMode

from .cloud_edge.pi05_collaboration import (
    actions,
    cloud_request,
    edge_request,
    run_pi05_collaboration,
    run_pi05_cpu_gpu_collaboration,
)
from .cloud_edge.pi05_resources import (
    Pi05CollaborationResources,
    build_pi05_collaboration_resources,
)
from .cloud_edge.pi05_settings import (
    Pi05CpuGpuConfig,
    load_pi05_cpu_gpu_config,
)

# Compatibility names retained while implementation ownership lives in cloud_edge.
_Resources = Pi05CollaborationResources
_actions = actions
_build_resources = build_pi05_collaboration_resources
_cloud_request = cloud_request
_edge_request = edge_request
_run_collaboration = run_pi05_collaboration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run π0.5 on CUDA and a lightweight policy on CPU with async fusion."
    )
    parser.add_argument(
        "--config",
        default="configs/pi05_cpu_gpu_collaboration.toml",
    )
    parser.add_argument("--checkpoint", help="override model.checkpoint")
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in FailoverMode),
        help="override coordination.mode",
    )
    parser.add_argument("--num-steps", type=int, help="override cloud.num_steps")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pi05_cpu_gpu_config(args.config)
    if args.checkpoint is not None:
        config = replace(config, checkpoint=args.checkpoint)
    if args.mode is not None:
        config = replace(
            config,
            failover=replace(config.failover, mode=FailoverMode(args.mode)),
        )
    if args.num_steps is not None:
        config = replace(config, num_steps=args.num_steps)
    if args.cuda_graph:
        config = replace(config, cuda_graph=True)
    if args.allow_download:
        config = replace(config, local_files_only=False)

    summary = run_pi05_cpu_gpu_collaboration(config)
    print("π0.5 GPU + lightweight CPU collaboration succeeded")
    for name, value in summary.items():
        print(f"{name}: {value}")
    return 0


__all__ = [
    "Pi05CpuGpuConfig",
    "_Resources",
    "_actions",
    "_build_resources",
    "_cloud_request",
    "_edge_request",
    "_run_collaboration",
    "build_parser",
    "load_pi05_cpu_gpu_config",
    "main",
    "run_pi05_cpu_gpu_collaboration",
]


if __name__ == "__main__":
    raise SystemExit(main())
