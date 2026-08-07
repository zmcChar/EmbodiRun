"""JSON Schema used for Qwen waypoint-plan generation."""

from __future__ import annotations

import math
from typing import Any

from embodied_runtime.tasks.navigation import (
    MAX_OUTPUT_VALID_FOR_S,
    MAX_WAYPOINT_COORDINATE_M,
    MAX_WAYPOINTS,
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

__all__ = ["PLAN_FIELDS", "WAYPOINT_FIELDS", "WAYPOINT_PLAN_SCHEMA"]
