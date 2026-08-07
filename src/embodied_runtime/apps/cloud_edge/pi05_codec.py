"""pi0.5 cloud response validation and action-shape utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from embodied_runtime.engine.result import InferenceResult

PI05_ACTION_HORIZON = 50
PI05_ACTION_DIM = 32


def pi05_response_to_result(
    response: Mapping[str, Any],
    expected_request_id: str,
) -> InferenceResult:
    if response.get("ok") is not True:
        error_type = str(response.get("error_type") or "CloudError")
        message = str(response.get("error") or "pi0.5 cloud request failed")
        raise RuntimeError(f"{error_type}: {message}")
    if response.get("kind") != "inference_result":
        raise RuntimeError(f"unexpected cloud response kind: {response.get('kind')!r}")
    if response.get("request_id") != expected_request_id:
        raise RuntimeError("pi0.5 cloud response request_id does not match")

    actions = response.get("actions")
    observed_shape = action_shape(actions)
    if observed_shape == (1, PI05_ACTION_HORIZON, PI05_ACTION_DIM):
        actions = actions[0]
        observed_shape = action_shape(actions)
    if observed_shape != (PI05_ACTION_HORIZON, PI05_ACTION_DIM):
        raise RuntimeError(
            "pi0.5 cloud actions must have shape "
            f"[{PI05_ACTION_HORIZON}, {PI05_ACTION_DIM}], got {observed_shape}"
        )
    validate_finite_numbers(actions)

    reported_shape = response.get("action_shape")
    if reported_shape not in (
        None,
        [PI05_ACTION_HORIZON, PI05_ACTION_DIM],
        [1, PI05_ACTION_HORIZON, PI05_ACTION_DIM],
    ):
        raise RuntimeError(
            f"pi0.5 cloud action_shape disagrees with its actions: {reported_shape!r}"
        )
    execution_time_s = float(response.get("execution_time_s") or 0.0)
    if not math.isfinite(execution_time_s) or execution_time_s < 0:
        raise RuntimeError("pi0.5 cloud execution_time_s must be finite and non-negative")

    return InferenceResult(
        request_id=expected_request_id,
        output={"actions": actions},
        execution_time_s=execution_time_s,
        metadata={
            "model_id": response.get("model_id"),
            "device_id": response.get("device"),
            "action_shape": observed_shape,
            "action_space_id": "pi05_policy_action_space",
        },
    )


def actions(output: Any) -> Any:
    return output["actions"] if isinstance(output, Mapping) else output


def action_shape(output: Any) -> tuple[int, ...]:
    action_values = actions(output)
    shape = getattr(action_values, "shape", None)
    if shape is not None:
        return tuple(int(dimension) for dimension in shape)
    return sequence_shape(action_values)


def sequence_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    length = len(value)
    if length == 0:
        return (0,)
    child_shape = sequence_shape(value[0])
    if any(sequence_shape(child) != child_shape for child in value[1:]):
        raise RuntimeError("action values must form a rectangular tensor")
    return (length, *child_shape)


def validate_finite_numbers(value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            validate_finite_numbers(item)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("pi0.5 cloud actions must contain only numbers")
    if not math.isfinite(float(value)):
        raise RuntimeError("pi0.5 cloud actions contain a non-finite value")
