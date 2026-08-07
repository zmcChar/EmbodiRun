"""CLI composition root for the matched VLABench Texas Hold'em experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from embodied_runtime.evaluation import OracleGateConfig

from .vlabench.cli import parse_seeds
from .vlabench.texas_holdem import (
    EndpointPlaceControllerConfig,
    TexasHoldemExperimentConfig,
    run_texas_holdem_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matched VLABench Texas Hold'em Edge/Oracle/Wrong-plan controls."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=(1000,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone-path",
        help="optional local SmolVLM2 snapshot used for deterministic offline loading",
    )
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--control-period-s", type=float, default=0.1)
    parser.add_argument("--place-lift-height-m", type=float, default=0.15)
    parser.add_argument("--place-clearance-m", type=float, default=0.12)
    parser.add_argument("--place-retract-height-m", type=float, default=0.10)
    parser.add_argument("--place-max-step-m", type=float, default=0.025)
    parser.add_argument("--place-open-steps", type=int, default=5)
    parser.add_argument("--place-slot-count", type=int, default=5)
    parser.add_argument("--place-slot-spacing-m", type=float, default=0.05)
    parser.add_argument("--place-workspace-limit-m", type=float, default=2.0)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="skip the unmeasured policy warm-up before the first paired trial",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow missing model files to download",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TexasHoldemExperimentConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        seeds=args.seeds,
        device=args.device,
        backbone_path=args.backbone_path,
        max_episode_steps=args.max_episode_steps,
        control_period_s=args.control_period_s,
        warmup_policy=not args.skip_warmup,
        local_files_only=not args.allow_download,
        place_controller=EndpointPlaceControllerConfig(
            lift_height_m=args.place_lift_height_m,
            clearance_m=args.place_clearance_m,
            retract_height_m=args.place_retract_height_m,
            max_translation_step_m=args.place_max_step_m,
            open_steps=args.place_open_steps,
            slot_count=args.place_slot_count,
            slot_spacing_m=args.place_slot_spacing_m,
            workspace_abs_limit_m=args.place_workspace_limit_m,
        ),
    )
    report = run_texas_holdem_experiment(config)
    print(report.to_json(oracle_gate_config=OracleGateConfig()))
    return 0


__all__ = ["build_parser", "main"]
