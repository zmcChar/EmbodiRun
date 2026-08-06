"""Wire schema and validation for Qwen navigation waypoint plans.

The JSON schema constrains the model response at generation time.  The local
validator remains authoritative: remote servers may ignore ``response_format``
or return numerically invalid JSON values.  Successful validation produces the
shared navigation contract instead of a Qwen-specific public result type.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from embodied_runtime.contracts.navigation import (
    MAX_OUTPUT_VALID_FOR_S,
    MAX_WAYPOINT_COORDINATE_M,
    MAX_WAYPOINTS,
    NavigationContractError,
    Waypoint,
    WaypointPlan,
)

WAYPOINT_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})
PLAN_FIELDS = frozenset({"frame", "waypoints", "terminal", "confidence", "valid_for_s"})

WAYPOINT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame": {"type": "string", "enum": ["base_link"]},
        "waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x_m": {
                        "type": "number",
                        "minimum": -MAX_WAYPOINT_COORDINATE_M,
                        "maximum": MAX_WAYPOINT_COORDINATE_M,
                    },
                    "y_m": {
                        "type": "number",
                        "minimum": -MAX_WAYPOINT_COORDINATE_M,
                        "maximum": MAX_WAYPOINT_COORDINATE_M,
                    },
                    "yaw_rad": {
                        "type": "number",
                        "minimum": -math.pi,
                        "maximum": math.pi,
                    },
                },
                "required": sorted(WAYPOINT_FIELDS),
                "additionalProperties": False,
            },
            "maxItems": MAX_WAYPOINTS,
        },
        "terminal": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "valid_for_s": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": MAX_OUTPUT_VALID_FOR_S,
        },
    },
    "required": sorted(PLAN_FIELDS),
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是移动机器人视觉语言导航规划器。结合用户指令、当前前视 RGB 图像、传感器摘要和回合上下文，直接给出当前 robot base_link 坐标系中的短视距空间计划。

坐标约定：x_m 向前为正，y_m 向左为正，yaw_rad 逆时针为正。坐标单位是米和弧度；每个 waypoint 都相对于当前帧的 base_link，不是速度、时长、关节角或电机命令。

规则：
- 只规划当前图像和上下文能够支持的局部、安全 waypoint，不臆测画面之外的通路。
- 避开人、动物、台阶、坑洞和近距离障碍。不能可靠判断安全路径时返回空 waypoints 且 terminal=true。
- 已到达目标或用户要求停止时返回空 waypoints 且 terminal=true；否则 terminal=false 且至少返回一个 waypoint。
- confidence 必须反映当前观测的可靠程度；valid_for_s 是该空间计划建议的短有效期，最大 10 秒。
- frame 必须为 base_link。只返回符合 JSON Schema 的一个对象，不输出 Markdown 或额外文字。
"""


class QwenNavigationError(RuntimeError):
    """A Qwen request or response cannot satisfy the navigation contract."""


class QwenNavigationValidationError(QwenNavigationError):
    """A model response is not a valid shared waypoint plan."""


def _exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = expected.difference(payload)
    unknown = set(payload).difference(expected)
    if missing or unknown:
        raise QwenNavigationValidationError(
            f"{name} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_waypoint_plan(
    payload: Mapping[str, Any],
    *,
    observation_sequence: int,
) -> WaypointPlan:
    """Validate one decoded model object and build the shared result contract."""

    if not isinstance(payload, Mapping):
        raise QwenNavigationValidationError("Qwen navigation response must be an object")
    _exact_fields(payload, PLAN_FIELDS, name="waypoint plan")

    frame = payload["frame"]
    if frame != "base_link":
        raise QwenNavigationValidationError("waypoint plan frame must be 'base_link'")

    raw_waypoints = payload["waypoints"]
    if isinstance(raw_waypoints, (str, bytes)) or not isinstance(raw_waypoints, Sequence):
        raise QwenNavigationValidationError("waypoints must be an array")
    if len(raw_waypoints) > MAX_WAYPOINTS:
        raise QwenNavigationValidationError(f"waypoints may contain at most {MAX_WAYPOINTS} items")

    waypoints: list[Waypoint] = []
    for index, raw_waypoint in enumerate(raw_waypoints):
        if not isinstance(raw_waypoint, Mapping):
            raise QwenNavigationValidationError(f"waypoints[{index}] must be an object")
        _exact_fields(raw_waypoint, WAYPOINT_FIELDS, name=f"waypoints[{index}]")
        try:
            waypoints.append(
                Waypoint(
                    x_m=raw_waypoint["x_m"],
                    y_m=raw_waypoint["y_m"],
                    yaw_rad=raw_waypoint["yaw_rad"],
                )
            )
        except (NavigationContractError, TypeError, ValueError) as error:
            raise QwenNavigationValidationError(
                f"waypoints[{index}] is invalid: {error}"
            ) from error

    terminal = payload["terminal"]
    if type(terminal) is not bool:
        raise QwenNavigationValidationError("terminal must be a boolean")

    try:
        return WaypointPlan(
            observation_sequence=observation_sequence,
            waypoints=tuple(waypoints),
            terminal=terminal,
            confidence=payload["confidence"],
            valid_for_s=payload["valid_for_s"],
            frame=frame,
        )
    except (NavigationContractError, TypeError, ValueError) as error:
        raise QwenNavigationValidationError(f"invalid waypoint plan: {error}") from error


__all__ = [
    "SYSTEM_PROMPT",
    "WAYPOINT_PLAN_SCHEMA",
    "QwenNavigationError",
    "QwenNavigationValidationError",
    "validate_waypoint_plan",
]
