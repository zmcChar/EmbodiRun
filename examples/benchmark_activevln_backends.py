#!/usr/bin/env python3
"""Benchmark one ActiveVLN backend in an isolated local GPU process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from embodied_runtime.integrations.navigation.vvla import ACTIVEVLN_REVISION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("transformers", "vvla"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--vvla-root", type=Path, default=Path("third_party/vvla"))
    parser.add_argument("--revision", default=ACTIVEVLN_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--turns", type=int, default=2)
    return parser


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "stdev": statistics.pstdev(values),
    }


def _runtime(args: argparse.Namespace):
    options = {
        "vvla_root": args.vvla_root,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "attention": args.attention,
        "max_new_tokens": args.max_new_tokens,
        "max_context": args.max_context,
        "allow_download": False,
        "do_sample": False,
    }
    if args.runtime == "vvla":
        from embodied_runtime.integrations.navigation.vvla import VvlaActiveVLNRuntime

        return VvlaActiveVLNRuntime(**options)
    from embodied_runtime.integrations.navigation.activevln_transformers import (
        TransformersActiveVLNRuntime,
    )

    return TransformersActiveVLNRuntime(**options)


def _predict(runtime, rgb: np.ndarray, instruction: str, episode_id: str):
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    prediction = runtime.predict(rgb, instruction, episode_id=episode_id)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return prediction, wall_ms


def _is_terminal(prediction) -> bool:
    return any(action.name == "stop" for action in prediction.actions)


def main() -> None:
    args = _parser().parse_args()
    for name in ("max_new_tokens", "max_context", "repeats", "turns"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    checkpoint = args.checkpoint.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    if not checkpoint.is_dir() or not image_path.is_file():
        raise SystemExit("checkpoint directory and image file must exist")
    benchmark_device = torch.device(args.device)
    if benchmark_device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("ActiveVLN GPU latency benchmarking requires a visible CUDA device")
    device_index = (
        torch.cuda.current_device() if benchmark_device.index is None else benchmark_device.index
    )
    torch.cuda.set_device(device_index)
    image_bytes = image_path.read_bytes()
    with Image.open(image_path) as image:
        # Both runtimes eventually expose the array through torch.from_numpy().
        # Materialize writable storage so benchmark input handling is warning-free.
        rgb = np.asarray(image.convert("RGB")).copy()
    runtime = _runtime(args)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter_ns()
    try:
        runtime.load()
        torch.cuda.synchronize()
    except BaseException:
        runtime.close()
        raise
    load_ms = (time.perf_counter_ns() - load_started) / 1_000_000.0
    load_memory = {
        "after_allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "after_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    try:
        for repeat in range(args.warmups):
            runtime.reset()
            for turn in range(args.turns):
                prediction, _ = _predict(
                    runtime,
                    rgb,
                    args.instruction,
                    f"warmup-{repeat}",
                )
                if _is_terminal(prediction):
                    break

        torch.cuda.reset_peak_memory_stats()
        records = []
        for repeat in range(args.repeats):
            runtime.reset()
            for turn in range(args.turns):
                prediction, wall_ms = _predict(
                    runtime,
                    rgb,
                    args.instruction,
                    f"measured-{repeat}",
                )
                records.append(
                    {
                        "repeat": repeat,
                        "turn": turn + 1,
                        "predict_wall_ms": wall_ms,
                        "runtime_reported_latency_ms": prediction.latency_ms,
                        "token_count": len(prediction.token_ids),
                        "text": prediction.text,
                        "actions": [
                            {"name": action.name, "value": action.value}
                            for action in prediction.actions
                        ],
                    }
                )
                if _is_terminal(prediction):
                    break
        by_turn = {}
        for turn in range(1, args.turns + 1):
            selected = [item for item in records if item["turn"] == turn]
            if not selected:
                continue
            by_turn[str(turn)] = {
                "predict_wall_ms": _summary([item["predict_wall_ms"] for item in selected]),
                "runtime_reported_latency_ms": _summary(
                    [item["runtime_reported_latency_ms"] for item in selected]
                ),
                "token_counts": [item["token_count"] for item in selected],
            }
        payload = {
            "result_kind": "activevln_backend_latency",
            "runtime": args.runtime,
            "backend_semantics": (
                "stock_transformers_full_history"
                if args.runtime == "transformers"
                else "vvla_incremental_session_kv"
            ),
            "gpu": torch.cuda.get_device_name(device_index),
            "gpu_index": device_index,
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "checkpoint_revision": args.revision,
            "checkpoint_hash_lock_verified": True,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_shape_hwc": list(rgb.shape),
            "instruction": args.instruction,
            "conditions": {
                "dtype": runtime.engine_dtype,
                "attention": args.attention,
                "do_sample": False,
                "use_cache_within_generate": True,
                "repetition_penalty": 1.05,
                "processor_use_fast": False,
                "max_new_tokens": args.max_new_tokens,
                "max_context": args.max_context,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "maximum_turns_per_episode": args.turns,
                "stop_ends_episode": True,
            },
            "comparison_metric": "predict_wall_ms",
            "runtime_reported_latency_scope": (
                "model.generate CUDA interval"
                if args.runtime == "transformers"
                else "VVLA engine-reported chunk interval"
            ),
            "load_ms": load_ms,
            "load_cuda_memory": load_memory,
            "inference_cuda_memory": {
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            },
            "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "by_turn": by_turn,
            "records": records,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
