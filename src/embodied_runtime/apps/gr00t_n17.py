"""Run GR00T N1.7 through the local HF/NVIDIA reference path."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from embodied_runtime.contracts import InferenceRequest
from embodied_runtime.integrations.serving.gr00t import HfLocalGr00tProvider
from embodied_runtime.models.vla.gr00t_n17 import (
    DEFAULT_CHECKPOINT,
    Gr00tN17Adapter,
    synthetic_droid_request,
)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"action value must be array-like, got {type(value).__name__}")
    return tuple(int(item) for item in shape)


def _preview(value: Any, limit: int = 8) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    flattened = value.reshape(-1)
    return [float(item) for item in flattened[:limit]]


async def run_gr00t_n17(
    *,
    provider_name: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda:0",
    mode: str = "eager",
    timeout_s: float = 300.0,
    prompt: str = "pick up the object",
    image_height: int = 180,
    image_width: int = 320,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Run one synthetic DROID observation and return a machine-readable summary."""

    adapter = Gr00tN17Adapter()
    setup_started = time.perf_counter()
    if provider_name == "hf":
        provider = HfLocalGr00tProvider.from_checkpoint(
            checkpoint,
            device=device,
            mode=mode,
            local_files_only=local_files_only,
            adapter=adapter,
        )
    else:
        raise ValueError(f"unknown GR00T provider: {provider_name!r}")
    setup_time_s = time.perf_counter() - setup_started

    request = InferenceRequest(
        payload=synthetic_droid_request(
            prompt,
            image_height=image_height,
            image_width=image_width,
        ),
        deadline_s=timeout_s,
    )
    try:
        result = await provider.infer_async(request)
        chunk = adapter.postprocess_one(result.output)
        actions = chunk.actions
        if not isinstance(actions, Mapping):
            raise TypeError("GR00T result must contain a named action mapping")
        action_shapes = {str(name): _shape(value) for name, value in actions.items()}
        action_preview = {str(name): _preview(value) for name, value in actions.items()}
        return {
            "ok": True,
            "provider": result.metadata["provider"],
            "provider_runtime": result.metadata["provider_runtime"],
            "model_id": adapter.describe().model_id,
            "requested_checkpoint": checkpoint,
            "device": result.metadata.get("device_id"),
            "request_id": result.request_id,
            "setup_time_s": setup_time_s,
            "inference_time_s": result.execution_time_s,
            "action_shapes": action_shapes,
            "action_preview": action_preview,
        }
    finally:
        await provider.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GR00T N1.7 through the local HF/NVIDIA reference runtime."
    )
    parser.add_argument(
        "--provider",
        choices=("hf",),
        required=True,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("eager",),
        default="eager",
        help="Only the reference eager path is currently supported.",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Permit the HF provider to download the outer GR00T checkpoint; "
            "NVIDIA's nested Cosmos loader follows Hugging Face offline environment flags."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be greater than zero")
    if args.image_height <= 0 or args.image_width <= 0:
        raise SystemExit("--image-height and --image-width must be greater than zero")
    if not args.prompt:
        raise SystemExit("--prompt must not be empty")
    summary = asyncio.run(
        run_gr00t_n17(
            provider_name=args.provider,
            checkpoint=args.checkpoint,
            device=args.device,
            mode=args.mode,
            timeout_s=args.timeout_s,
            prompt=args.prompt,
            image_height=args.image_height,
            image_width=args.image_width,
            local_files_only=not args.allow_download,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
