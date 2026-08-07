#!/usr/bin/env python3
"""Model-only smoke CLI for the shared Go2 navigation contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import time
from pathlib import Path

from embodied_runtime.models.vln.streamvln import StreamVLNRuntime
from embodied_runtime.policies.navigation import (
    InternVLANavigationPolicy,
    QwenNavigationPolicy,
    StreamVLNNavigationPolicy,
)
from embodied_runtime.tasks.navigation import (
    EncodedRGBFrame,
    NavigationObservation,
    NavigationRequest,
    WaypointPlan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "streamvln", "internvla"), required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="Move toward the visible target.")
    parser.add_argument("--episode-id", default="go2-model-smoke")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--streamvln-root", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--internvla-root", type=Path)
    parser.add_argument("--internvla-variant", choices=("dualvln", "navdp"), default="dualvln")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:15003/v1")
    parser.add_argument("--qwen-model", default="qwen3.5-9b")
    parser.add_argument("--api-key")
    return parser


def _stream_runtime(args: argparse.Namespace) -> StreamVLNRuntime:
    if args.streamvln_root is None or args.model_path is None:
        raise SystemExit("StreamVLN requires --streamvln-root and --model-path")
    return StreamVLNRuntime(
        streamvln_root=args.streamvln_root,
        model_path=args.model_path,
        device=args.device,
        cuda_memory_fraction=args.cuda_memory_fraction,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
        warmup=True,
    )


def _policy(args: argparse.Namespace):
    if args.backend == "qwen":
        return QwenNavigationPolicy(
            base_url=args.qwen_base_url,
            model=args.qwen_model,
            api_key=args.api_key,
        )
    if args.backend == "streamvln":
        return StreamVLNNavigationPolicy(runtime=_stream_runtime(args))
    return InternVLANavigationPolicy(
        variant=args.internvla_variant,
        model_path=args.model_path,
        device=args.device,
        internnav_root=args.internvla_root,
    )


def _request(args: argparse.Namespace) -> NavigationRequest:
    if args.image is None:
        raise SystemExit("inference requires --image")
    image_path = args.image.expanduser().resolve()
    data = image_path.read_bytes()
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise SystemExit(f"unsupported image type: {media_type}")
    captured_at = time.time()
    observation = NavigationObservation(
        episode_id=args.episode_id,
        sequence=0,
        reset=True,
        rgb_frames=(
            EncodedRGBFrame(
                sequence=0,
                captured_at_s=captured_at,
                data=data,
                media_type=media_type,
            ),
        ),
    )
    return NavigationRequest(args.prompt, observation)


def _plan_json(plan: WaypointPlan) -> dict[str, object]:
    return {
        "kind": "waypoint_plan",
        "observation_sequence": plan.observation_sequence,
        "frame": plan.frame,
        "waypoints": [
            {"x_m": item.x_m, "y_m": item.y_m, "yaw_rad": item.yaw_rad} for item in plan.waypoints
        ],
        "terminal": plan.terminal,
        "confidence": plan.confidence,
        "valid_for_s": plan.valid_for_s,
    }


async def _infer(args: argparse.Namespace) -> None:
    policy = _policy(args)
    try:
        plan = await policy.plan(_request(args))
        if not isinstance(plan, WaypointPlan):
            raise TypeError(f"policy returned {type(plan).__name__}, not WaypointPlan")
        print(json.dumps(_plan_json(plan), ensure_ascii=False, indent=2))
    finally:
        await policy.aclose()


def main() -> None:
    args = _parser().parse_args()
    if args.load_only:
        if args.backend != "streamvln":
            raise SystemExit("--load-only currently supports StreamVLN")
        runtime = _stream_runtime(args)
        runtime.load()
        print(json.dumps({"loaded": True, "model": str(runtime.config.model_path)}))
        return
    asyncio.run(_infer(args))


if __name__ == "__main__":
    main()
