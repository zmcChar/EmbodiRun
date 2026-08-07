"""CLI composition root for the paired VLABench get-coffee pilot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from embodied_runtime.evaluation import OracleGateConfig

from .vlabench.big_small_brain import VLABenchPilotConfig, run_vlabench_pilot
from .vlabench.cli import parse_seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paired VLABench Edge-only/Oracle-plan gate."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=(1000,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone-path",
        help="optional local SmolVLM2 snapshot used for deterministic offline loading",
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--first-subgoal-budget", type=int, default=350)
    parser.add_argument("--subgoal-stability-steps", type=int, default=3)
    parser.add_argument("--control-period-s", type=float, default=0.1)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="skip the unmeasured policy warm-up before the first paired trial",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model files missing from the local checkpoint/cache to download",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VLABenchPilotConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        seeds=args.seeds,
        device=args.device,
        backbone_path=args.backbone_path,
        max_episode_steps=args.max_episode_steps,
        first_subgoal_budget=args.first_subgoal_budget,
        subgoal_stability_steps=args.subgoal_stability_steps,
        control_period_s=args.control_period_s,
        warmup_policy=not args.skip_warmup,
        local_files_only=not args.allow_download,
    )
    report = run_vlabench_pilot(config)
    print(report.to_json(oracle_gate_config=OracleGateConfig()))
    return 0


__all__ = ["build_parser", "main"]
