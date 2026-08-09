#!/usr/bin/env python3
"""Model-only smoke CLI for the shared Go2 navigation contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import time
from pathlib import Path

from embodied_runtime.apps.navigation.policies import supported_runtimes
from embodied_runtime.models.vla.navila import NaVILARuntime
from embodied_runtime.models.vln.streamvln import StreamVLNRuntime
from embodied_runtime.policies.navigation import (
    InternVLANavigationPolicy,
    NaVILANavigationPolicy,
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
    parser.add_argument(
        "--model",
        choices=("qwen", "streamvln", "internvla", "navila", "activevln"),
    )
    parser.add_argument(
        "--backend",
        choices=("qwen", "streamvln", "internvla", "navila", "activevln"),
        help="Compatibility alias for --model.",
    )
    parser.add_argument(
        "--runtime",
        choices=("transformers", "vllm-omni", "vvla"),
        default="transformers",
    )
    parser.add_argument(
        "--vllm-omni-url",
        default="ws://127.0.0.1:8000/v1/realtime/robot/openpi",
    )
    parser.add_argument("--vllm-omni-timeout-s", type=float, default=120.0)
    parser.add_argument("--vllm-omni-session-id")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="Move toward the visible target.")
    parser.add_argument("--episode-id", default="go2-model-smoke")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--streamvln-root", type=Path)
    parser.add_argument("--navila-root", type=Path)
    parser.add_argument("--vvla-root", type=Path, default=Path("third_party/vvla"))
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attention", choices=("eager", "eager_bc", "sdpa"), default="eager")
    parser.add_argument("--revision", default="160987313e3e869705f42400d1b8f28177044518")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--internvla-root", type=Path)
    parser.add_argument("--internvla-variant", choices=("dualvln", "navdp"), default="dualvln")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:15003/v1")
    parser.add_argument("--qwen-model", default="qwen3.5-9b")
    parser.add_argument("--api-key")
    return parser


def _selected_model(args: argparse.Namespace) -> str:
    if args.model is not None and args.backend is not None and args.model != args.backend:
        raise SystemExit("--model and legacy --backend must match when both are supplied")
    model = args.model if args.model is not None else args.backend
    if model is None:
        raise SystemExit("one of --model or --backend is required")
    return model


def _stream_runtime(args: argparse.Namespace) -> StreamVLNRuntime:
    if args.streamvln_root is None or args.model_path is None:
        raise SystemExit("StreamVLN requires --streamvln-root and --model-path")
    return StreamVLNRuntime(
        streamvln_root=args.streamvln_root,
        model_path=args.model_path,
        device=args.device,
        cuda_memory_fraction=args.cuda_memory_fraction,
        max_new_tokens=args.max_new_tokens or 64,
        local_files_only=True,
        warmup=True,
    )


def _navila_runtime(args: argparse.Namespace) -> NaVILARuntime:
    if args.navila_root is None or args.model_path is None:
        raise SystemExit("NaVILA requires --navila-root and --model-path")
    return NaVILARuntime(
        navila_root=args.navila_root,
        model_path=args.model_path,
        device=args.device,
        cuda_memory_fraction=args.cuda_memory_fraction,
        max_new_tokens=args.max_new_tokens or 32,
        local_files_only=True,
    )


def _policy(args: argparse.Namespace):
    supported = supported_runtimes(args.model)
    if args.runtime not in supported:
        choices = ", ".join(supported)
        raise SystemExit(
            f"{args.model} does not support --runtime {args.runtime}; supported runtimes: {choices}"
        )
    if args.model == "activevln":
        if args.model_path is None:
            raise SystemExit("ActiveVLN requires --model-path (or a Hugging Face ID)")
        options = {
            "vvla_root": args.vvla_root,
            "checkpoint": args.model_path,
            "revision": args.revision,
            "device": args.device,
            "dtype": args.dtype,
            "attention": args.attention,
            "max_new_tokens": args.max_new_tokens or 64,
            "max_context": args.max_context,
            "allow_download": args.allow_download,
            "do_sample": False,
        }
        if args.runtime == "vvla":
            from embodied_runtime.integrations.navigation.vvla import (
                VvlaActiveVLNNavigationPolicy,
            )

            return VvlaActiveVLNNavigationPolicy(**options)
        if args.attention == "eager_bc":
            raise SystemExit("--attention eager_bc is VVLA-only; use eager or sdpa")
        from embodied_runtime.integrations.navigation.activevln_transformers import (
            TransformersActiveVLNNavigationPolicy,
        )

        return TransformersActiveVLNNavigationPolicy(**options)
    if args.runtime == "vllm-omni":
        from embodied_runtime.integrations.navigation.vllm_omni import (
            VllmOmniNaVILANavigationPolicy,
            VllmOmniStreamVLNNavigationPolicy,
        )

        options = {
            "url": args.vllm_omni_url,
            "timeout_s": args.vllm_omni_timeout_s,
            "session_id": args.vllm_omni_session_id,
        }
        if args.model == "streamvln":
            return VllmOmniStreamVLNNavigationPolicy(**options)
        if args.model == "navila":
            return VllmOmniNaVILANavigationPolicy(**options)
        raise SystemExit(f"{args.model} does not support --runtime vllm-omni")
    if args.model == "qwen":
        return QwenNavigationPolicy(
            base_url=args.qwen_base_url,
            model=args.qwen_model,
            api_key=args.api_key,
        )
    if args.model == "streamvln":
        return StreamVLNNavigationPolicy(runtime=_stream_runtime(args))
    if args.model == "navila":
        return NaVILANavigationPolicy(runtime=_navila_runtime(args))
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
        await policy.prepare()
        plan = await policy.plan(_request(args))
        if not isinstance(plan, WaypointPlan):
            raise TypeError(f"policy returned {type(plan).__name__}, not WaypointPlan")
        output = _plan_json(plan)
        output.update({"model": args.model, "runtime": args.runtime})
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        await policy.aclose()


async def _load_only(args: argparse.Namespace) -> None:
    local_pair = args.runtime == "transformers" and args.model in {
        "streamvln",
        "internvla",
        "navila",
        "activevln",
    }
    local_pair = local_pair or (args.runtime == "vvla" and args.model == "activevln")
    if not local_pair:
        raise SystemExit("--load-only requires a local model backend")
    policy = _policy(args)
    try:
        await policy.prepare()
        print(
            json.dumps(
                {
                    "prepared": True,
                    "loaded": True,
                    "model": args.model,
                    "runtime": args.runtime,
                    "backend": args.model,
                }
            )
        )
    finally:
        await policy.aclose()


def main() -> None:
    args = _parser().parse_args()
    args.model = _selected_model(args)
    if args.load_only:
        asyncio.run(_load_only(args))
        return
    asyncio.run(_infer(args))


if __name__ == "__main__":
    main()
