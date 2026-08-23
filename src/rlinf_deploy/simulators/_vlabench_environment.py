"""Lazy VLABench environment construction and boundary normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VLABENCH_ACTION_DIM = 7


def make_environment(**kwargs: Any) -> Any:
    from .vlabench_camera import make_semantic_vlabench_environment

    return make_semantic_vlabench_environment(**kwargs)


def extract_instruction(environment: Any, *, fallback: str) -> str:
    task_object = getattr(getattr(environment, "_env", None), "task", None)
    get_instruction = getattr(task_object, "get_instruction", None)
    if callable(get_instruction):
        instruction = get_instruction()
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()
        if isinstance(instruction, (tuple, list)):
            for candidate in instruction:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

    for owner in (task_object, environment):
        for name in ("task_description", "language_instruction", "instruction"):
            candidate = getattr(owner, name, None)
            if isinstance(candidate, str) and candidate.strip() and candidate != fallback:
                return candidate.strip()
        candidates = getattr(owner, "instructions", None)
        if isinstance(candidates, (tuple, list)):
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
    return fallback


def action_array(values: Any) -> Any:
    raw = values.get("action") if isinstance(values, Mapping) and "action" in values else values
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "VLABench execution requires NumPy in the isolated simulator environment"
        ) from error
    action = np.asarray(raw, dtype=np.float32)
    if action.shape != (VLABENCH_ACTION_DIM,):
        raise ValueError(
            f"VLABench action must have shape ({VLABENCH_ACTION_DIM},), got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise ValueError("VLABench action must contain only finite values")
    return action


def unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        if not isinstance(info, Mapping):
            raise TypeError("VLABench reset info must be a mapping")
        return observation, info
    return result, {}


def unpack_step(
    result: Any,
) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 5:
        raise ValueError("VLABench step must return five values")
    observation, reward, terminated, truncated, info = result
    if not isinstance(info, Mapping):
        raise TypeError("VLABench step info must be a mapping")
    return observation, float(reward), bool(terminated), bool(truncated), dict(info)


def success_from_info(info: Mapping[str, Any]) -> bool | None:
    value = info.get("is_success", info.get("success"))
    if value is None:
        return None
    return bool(value)


__all__ = ["VLABENCH_ACTION_DIM"]
