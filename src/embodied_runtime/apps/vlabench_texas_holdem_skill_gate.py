"""CLI composition root for the VLABench planner skill gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .vlabench.cli import parse_seeds
from .vlabench.skill_gate import PlannerSkillGateConfig, run_planner_skill_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the VLABench Oracle/Wrong/Cloud plan skill-executor gate."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=(1000,))
    parser.add_argument("--render-size", type=int, default=96)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--shadow-max-episode-steps", type=int, default=2000)
    parser.add_argument("--settle-repeats", type=int, default=12)
    parser.add_argument("--planner-checkpoint")
    parser.add_argument("--planner-device", default="cpu")
    parser.add_argument("--planner-dtype", default="float32")
    parser.add_argument("--planner-max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--allow-single-json-fence",
        action="store_true",
        help="accept exactly one bare ```json ... ``` wrapper before schema validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    planner = None
    if args.planner_checkpoint:
        from embodied_runtime.integrations.planning.hf_texas_holdem import (
            HfTexasHoldemPlanner,
            HfTexasHoldemPlannerConfig,
        )

        planner = HfTexasHoldemPlanner(
            HfTexasHoldemPlannerConfig(
                checkpoint=args.planner_checkpoint,
                device=args.planner_device,
                dtype=args.planner_dtype,
                max_new_tokens=args.planner_max_new_tokens,
                allow_single_json_fence=args.allow_single_json_fence,
            )
        )
    config = PlannerSkillGateConfig(
        output_dir=args.output_dir,
        seeds=args.seeds,
        render_resolution=(args.render_size, args.render_size),
        max_episode_steps=args.max_episode_steps,
        shadow_max_episode_steps=args.shadow_max_episode_steps,
        settle_repeats=args.settle_repeats,
    )
    trials = run_planner_skill_gate(config, cloud_planner=planner)
    print(json.dumps([trial.to_dict() for trial in trials], allow_nan=False, indent=2))
    return 0


__all__ = ["build_parser", "main"]
