"""CLI composition root for SmolVLA edge with pi0.5 cloud authority."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from embodied_runtime.distributed import FailoverConfig, FailoverMode

from .cloud_edge.pi05_codec import pi05_response_to_result
from .cloud_edge.smolvla_runtime import (
    EdgeRuntime,
    Pi05TcpEndpoint,
    build_edge_runtime,
)
from .cloud_edge.smolvla_scenario import run_async_scenario, run_smolvla_pi05_async
from .cloud_edge.smolvla_settings import SmolVLAPi05AsyncConfig

# Compatibility names retained while implementation ownership lives in cloud_edge.
_EdgeRuntime = EdgeRuntime
_build_edge_runtime = build_edge_runtime
_pi05_response_to_result = pi05_response_to_result
_run_async_scenario = run_async_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real SmolVLA through Adapter/Backend/Engine on an edge GPU and "
            "use an asynchronous remote pi0.5 result when available."
        )
    )
    parser.add_argument("--edge-checkpoint", required=True)
    parser.add_argument("--edge-vlm-base-path", required=True)
    parser.add_argument("--cloud-host", required=True)
    parser.add_argument("--cloud-port", type=int, default=18765)
    parser.add_argument("--edge-device", default="cuda:0")
    parser.add_argument("--stats-variant", default="so100")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--language-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tcp-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--mode",
        choices=(
            FailoverMode.ASYNC_CLOUD_PREFERRED.value,
            FailoverMode.EDGE_ONLY.value,
        ),
        default=FailoverMode.ASYNC_CLOUD_PREFERRED.value,
    )
    parser.add_argument("--cloud-result-ttl-s", type=float, default=120.0)
    parser.add_argument("--max-cloud-sequence-lag", type=int, default=4)
    parser.add_argument("--cloud-submit-interval-s", type=float, default=60.0)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SmolVLAPi05AsyncConfig(
        edge_checkpoint=args.edge_checkpoint,
        edge_vlm_base_path=args.edge_vlm_base_path,
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
        edge_device=args.edge_device,
        stats_variant=args.stats_variant,
        num_steps=args.num_steps,
        language_length=args.language_length,
        seed=args.seed,
        local_files_only=not args.allow_download,
        tcp_timeout_s=args.tcp_timeout_s,
        failover=FailoverConfig(
            mode=FailoverMode(args.mode),
            cloud_request_timeout_s=args.tcp_timeout_s,
            cloud_result_ttl_s=args.cloud_result_ttl_s,
            max_cloud_sequence_lag=args.max_cloud_sequence_lag,
            cloud_submit_interval_s=args.cloud_submit_interval_s,
        ),
    )
    summary = run_smolvla_pi05_async(config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "Pi05TcpEndpoint",
    "SmolVLAPi05AsyncConfig",
    "_EdgeRuntime",
    "_build_edge_runtime",
    "_pi05_response_to_result",
    "_run_async_scenario",
    "build_parser",
    "main",
    "run_smolvla_pi05_async",
]


if __name__ == "__main__":
    raise SystemExit(main())
