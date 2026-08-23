"""Deterministic JSON-compatible serialization for episode traces."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Protocol

from rlinf_deploy.robots.action import RobotAction
from rlinf_deploy.robots.observation import RobotObservation

from .trace_values import (
    DisturbanceTraceEvent,
    ResetTraceEvent,
    StepTraceEvent,
    TraceEvent,
)


class EpisodeTraceView(Protocol):
    """Read-only trace shape consumed by deterministic serialization."""

    task: str | None
    seed: int | None
    events: tuple[TraceEvent, ...]

    @property
    def steps(self) -> tuple[StepTraceEvent, ...]: ...

    @property
    def disturbances(self) -> tuple[DisturbanceTraceEvent, ...]: ...


def summarize_episode_trace(trace: EpisodeTraceView) -> dict[str, Any]:
    """Return the exact deterministic structure accepted by strict JSON encoders."""

    steps = trace.steps
    final_outcome = steps[-1].outcome if steps else None
    return {
        "task": trace.task,
        "seed": trace.seed,
        "event_count": len(trace.events),
        "step_count": len(steps),
        "disturbance_count": len(trace.disturbances),
        "total_reward": sum(step.outcome.reward for step in steps),
        "terminated": final_outcome.terminated if final_outcome is not None else False,
        "truncated": final_outcome.truncated if final_outcome is not None else False,
        "success": final_outcome.success if final_outcome is not None else None,
        "events": [_event_summary(event) for event in trace.events],
    }


def _event_summary(event: TraceEvent) -> dict[str, Any]:
    if isinstance(event, ResetTraceEvent):
        return {
            "sequence": event.sequence,
            "kind": event.kind,
            "observation": _observation_summary(event.observation),
        }
    if isinstance(event, StepTraceEvent):
        return {
            "sequence": event.sequence,
            "kind": event.kind,
            "action": _action_summary(event.action),
            "outcome": {
                "observation": _observation_summary(event.outcome.observation),
                "reward": event.outcome.reward,
                "terminated": event.outcome.terminated,
                "truncated": event.outcome.truncated,
                "success": event.outcome.success,
                "subgoal_progress": event.outcome.subgoal_progress,
                "info": _json_compatible(event.outcome.info),
            },
        }
    if not isinstance(event, DisturbanceTraceEvent):  # pragma: no cover - trace validates
        raise TypeError("episode trace contains an unsupported event")
    return {
        "sequence": event.sequence,
        "kind": event.kind,
        "name": event.name,
        "details": _json_compatible(event.details),
    }


def _observation_summary(observation: RobotObservation) -> dict[str, Any]:
    return {
        "timestamp_s": _finite_json_float(observation.timestamp_s),
        "values": _json_compatible(observation.values),
        "metadata": _json_compatible(observation.metadata),
    }


def _action_summary(action: RobotAction) -> dict[str, Any]:
    return {
        "timestamp_s": _finite_json_float(action.timestamp_s),
        "values": _json_compatible(action.values),
        "metadata": _json_compatible(action.metadata),
    }


def _finite_json_float(value: Any) -> float | str:
    normalized = float(value)
    if math.isfinite(normalized):
        return normalized
    if math.isnan(normalized):
        return "NaN"
    return "Infinity" if normalized > 0 else "-Infinity"


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_json_float(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            json_key = str(key)
            if json_key in converted:
                raise ValueError("trace mapping keys collide after JSON conversion")
            converted[json_key] = _json_compatible(value[key])
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted_items = [_json_compatible(item) for item in value]
        return sorted(
            converted_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_compatible(tolist())
    value_type = type(value)
    return {"type": f"{value_type.__module__}.{value_type.__qualname__}"}


__all__ = ["summarize_episode_trace"]
