#!/usr/bin/env python3
"""Run an RLinf-deploy ActiveVLN backend on real local Habitat R2R episodes."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from embodied_runtime.integrations.navigation.activevln_transformers import (
    TransformersActiveVLNRuntime,
)
from embodied_runtime.integrations.navigation.vvla import (
    ACTIVEVLN_REVISION,
    VvlaActiveVLNRuntime,
    validate_activevln_action_text,
    verify_activevln_source,
)
from embodied_runtime.policies.navigation.errors import NavigationPolicyError


class _HabitatPolicyTurn:
    """Structural equivalent of VVLA's evaluator turn without fake logprobs."""

    __slots__ = (
        "action",
        "action_mask",
        "metadata",
        "policy_version",
        "raw_tokens",
        "timing",
        "token_logprobs",
    )

    def __init__(
        self,
        action: Any,
        *,
        timing: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.action = action
        self.raw_tokens: tuple[int, ...] = ()
        self.token_logprobs: tuple[float, ...] = ()
        self.action_mask: tuple[bool, ...] = ()
        self.policy_version = 0
        self.timing = {} if timing is None else timing
        self.metadata = {} if metadata is None else metadata


class _TransformersActiveVLNHabitatDriver:
    """Adapt the stock-HF runtime to VVLA's small Habitat evaluator protocol."""

    def __init__(self, runtime: TransformersActiveVLNRuntime, action_parser: Callable[[str], Any]):
        self.runtime = runtime
        self._action_parser = action_parser

    @staticmethod
    def _rgb(observation: Any) -> np.ndarray:
        images = observation.images
        if hasattr(images, "detach"):
            images = images.detach()
        if hasattr(images, "cpu"):
            images = images.cpu()
        array = np.asarray(images)
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
            raise NavigationPolicyError(
                f"ActiveVLN Habitat observation must be [1,3,H,W], got {array.shape}"
            )
        hwc = np.moveaxis(array[0], 0, -1)
        return np.ascontiguousarray(np.rint(np.clip(hwc, 0.0, 1.0) * 255.0).astype(np.uint8))

    def act(self, observation: Any, session: Any) -> _HabitatPolicyTurn:
        instruction = getattr(observation, "instruction", None)
        if not isinstance(instruction, str) or not instruction.strip():
            raise NavigationPolicyError("ActiveVLN Habitat observation has no instruction")
        prediction = self.runtime.predict(
            self._rgb(observation),
            instruction,
            episode_id=f"{session.env_id}:{session.episode_id}:{session.rollout_id}",
        )
        try:
            # The runtime already parses strictly. Recheck both its typed result
            # and the upstream evaluator object before Habitat can execute it.
            validate_activevln_action_text(prediction.text, prediction.actions)
            parsed = self._action_parser(prediction.text)
            if not getattr(parsed, "valid", False):
                raise ValueError("ActiveVLN evaluator parser rejected canonical action text")
            validate_activevln_action_text(prediction.text, parsed.actions)
        except ValueError as error:
            raise NavigationPolicyError(str(error)) from error
        return _HabitatPolicyTurn(
            action=parsed,
            timing={"e2e_ms": float(prediction.latency_ms)},
            metadata={
                "runtime": "transformers",
                "text": prediction.text,
                "token_ids": tuple(prediction.token_ids),
            },
        )

    def reset_session(self, session: Any) -> None:
        del session
        self.runtime.reset()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("transformers", "vvla"), default="vvla")
    parser.add_argument("--vvla-root", type=Path, default=Path("third_party/vvla"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--activevln-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene-id", default="17DRP5sb8fy")
    parser.add_argument("--episode-ids", help="comma-separated explicit episode ids")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attention", choices=("eager", "eager_bc", "sdpa"), default="eager")
    parser.add_argument("--habitat-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _episode_ids(dataset_root: Path, split: str, scene_id: str | None) -> list[str]:
    path = dataset_root / split / f"{split}.json.gz"
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    return [
        str(item["episode_id"])
        for item in payload["episodes"]
        if scene_id is None or scene_id in str(item["scene_id"])
    ]


def _require_assets(args: argparse.Namespace) -> None:
    vlnce_root = args.activevln_root / "vlnce_server"
    paths = (
        vlnce_root / "client.py",
        vlnce_root / "env.py",
        vlnce_root / "VLN_CE" / "habitat_extensions" / "config" / "vlnce_task_activevln_r2r.yaml",
        args.dataset_root / args.split / f"{args.split}.json.gz",
        args.dataset_root / args.split / f"{args.split}_gt.json.gz",
        args.scenes_dir,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Habitat preflight is missing required assets: " + ", ".join(missing))


def _runtime_from_args(args: argparse.Namespace) -> Any:
    options = {
        "vvla_root": args.vvla_root,
        "checkpoint": args.checkpoint,
        "revision": ACTIVEVLN_REVISION,
        "device": args.device,
        "dtype": args.dtype,
        "attention": args.attention,
        "max_new_tokens": args.max_new_tokens,
        "max_context": args.max_context,
        "allow_download": False,
        "do_sample": False,
    }
    if args.runtime == "transformers":
        return TransformersActiveVLNRuntime(**options)
    return VvlaActiveVLNRuntime(**options, env_id="habitat-r2r")


def main() -> None:
    args = _parser().parse_args()
    if args.max_episodes < 1 or args.max_turns < 1:
        raise SystemExit("episode and turn limits must be positive")
    if args.checkpoint is None and not args.preflight_only:
        raise SystemExit("--checkpoint is required unless --preflight-only is used")
    if args.runtime == "transformers" and args.attention == "eager_bc":
        raise SystemExit("--attention eager_bc is available only with --runtime vvla")
    try:
        verify_activevln_source(args.vvla_root, args.activevln_root)
    except NavigationPolicyError as error:
        raise SystemExit(str(error)) from error
    _require_assets(args)
    available = _episode_ids(args.dataset_root, args.split, args.scene_id)
    if args.episode_ids:
        requested = [item.strip() for item in args.episode_ids.split(",") if item.strip()]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise SystemExit(f"episode ids are absent from the selected scene: {missing}")
        episode_ids = requested
    else:
        episode_ids = available[: args.max_episodes]
    if not episode_ids:
        raise SystemExit("no episodes matched the selected split and scene")

    from vvla.eval.habitat import (
        HabitatLocalR2RConfig,
        HabitatLocalR2REnvironment,
        activevln_local_observation,
    )
    from vvla.eval.vln import GenerationBackendDriver, VLNEvaluator

    config = HabitatLocalR2RConfig(
        activevln_root=args.activevln_root,
        dataset_root=args.dataset_root,
        scenes_dir=args.scenes_dir,
        split=args.split,
        scene_id=args.scene_id,
        gpu_device_id=args.habitat_gpu,
    )
    # Build, render, and close one environment before loading an 8+ GiB model.
    # Missing Habitat packages or licensed assets therefore fail immediately.
    probe = HabitatLocalR2REnvironment(config, episode_ids[0])
    try:
        probe.reset(seed=args.seed)
    except Exception as error:
        raise SystemExit(f"Habitat readiness failed before model load: {error}") from error
    finally:
        probe.close()
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "result_kind": "habitat_r2r_preflight",
                    "ready": True,
                    "episode_id": episode_ids[0],
                    "split": args.split,
                    "scene_id": args.scene_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    runtime = _runtime_from_args(args)
    try:
        runtime.load()
    except BaseException:
        runtime.close()
        raise
    try:
        remaining: Iterator[str] = iter(episode_ids)

        def factory() -> HabitatLocalR2REnvironment:
            return HabitatLocalR2REnvironment(config, next(remaining))

        if args.runtime == "vvla":
            driver = GenerationBackendDriver(runtime.backend)
        else:
            from vvla.policies.activevln.prompt_activevln import parse_r2r_actions

            driver = _TransformersActiveVLNHabitatDriver(runtime, parse_r2r_actions)

        result = VLNEvaluator(
            factory,
            driver,
            activevln_local_observation,
            max_turns=args.max_turns,
        ).run(len(episode_ids), seed=args.seed)
        payload = {
            "result_kind": "real_habitat_r2r",
            "model": "activevln",
            "runtime": args.runtime,
            "checkpoint_revision": ACTIVEVLN_REVISION,
            "episodes": episode_ids,
            "conditions": {
                "device": args.device,
                "dtype": runtime.engine_dtype,
                "attention": args.attention,
                "split": args.split,
                "scene_id": args.scene_id,
                "max_turns": args.max_turns,
                "max_new_tokens": args.max_new_tokens,
                "batch": 1,
            },
            "metrics": result.to_dict(),
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        print(serialized)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
