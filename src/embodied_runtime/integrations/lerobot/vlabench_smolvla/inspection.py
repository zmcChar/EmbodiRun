"""Runtime inspection and action normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .values import LeRobotRunnerError


def feature_shapes(features: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(features, Mapping):
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for name, feature in features.items():
        shape = getattr(feature, "shape", None)
        if shape is not None:
            result[str(name)] = tuple(int(dimension) for dimension in shape)
    return result


def processor_stat_shapes(processor: Any) -> dict[str, tuple[int, ...]]:
    state_dict = getattr(processor, "state_dict", None)
    if not callable(state_dict):
        return {}
    try:
        state = state_dict()
    except (AttributeError, RuntimeError, TypeError):
        return {}
    if not isinstance(state, Mapping):
        return {}

    result: dict[str, tuple[int, ...]] = {}
    for group_name, group in state.items():
        tensors = group if isinstance(group, Mapping) else {str(group_name): group}
        prefix = f"{group_name}." if isinstance(group, Mapping) else ""
        for tensor_name, tensor in tensors.items():
            shape = getattr(tensor, "shape", None)
            if shape is not None:
                result[f"{prefix}{tensor_name}"] = tuple(int(dimension) for dimension in shape)
    return result


def parameter_count(policy: Any) -> int | None:
    parameters = getattr(policy, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return sum(int(parameter.numel()) for parameter in parameters())
    except (AttributeError, RuntimeError, TypeError):
        return None


def policy_action_queue(policy: Any) -> Any | None:
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, Mapping):
        queue = queues.get("action")
        if queue is not None:
            return queue
    return getattr(policy, "_action_queue", None)


def one_cpu_action(action: Any) -> Any:
    shape = _action_shape(action)
    if shape == (1, 7):
        action = action[0]
    elif shape != (7,):
        raise LeRobotRunnerError(
            f"VLABench requires one 7-D action, but the policy returned shape {shape}"
        )

    detach = getattr(action, "detach", None)
    if callable(detach):
        action = detach()
    to = getattr(action, "to", None)
    if callable(to):
        action = to("cpu")
    numpy = getattr(action, "numpy", None)
    if callable(numpy):
        action = numpy()

    if _action_shape(action) != (7,):
        raise LeRobotRunnerError("action conversion did not preserve the required shape (7,)")
    return action


def _action_shape(action: Any) -> tuple[int, ...]:
    shape = getattr(action, "shape", None)
    if shape is None:
        raise LeRobotRunnerError("LeRobot postprocessor returned an action without a shape")
    return tuple(int(dimension) for dimension in shape)
