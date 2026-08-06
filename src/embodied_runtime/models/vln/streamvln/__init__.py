"""Modular StreamVLN real-world runtime for language-guided Go2 navigation.

The package intentionally exposes model-local dataclasses rather than public
navigation contracts.  Serving integrations own that boundary conversion.
"""

from .actions import (
    FORWARD_STEP_M,
    MAX_FUTURE_ACTIONS,
    TURN_STEP_RAD,
    StreamVLNAction,
    StreamVLNNativeOutputError,
    StreamVLNWaypoint,
    StreamVLNWaypointPlan,
    actions_to_cumulative_waypoints,
    normalize_native_actions,
)
from .evaluator import StreamVLNEvaluator, StreamVLNEvaluatorError
from .runtime import (
    DEFAULT_MODEL,
    DEFAULT_SENSOR_CONFIG,
    LazyStreamVLNEngine,
    StreamVLNConfig,
    StreamVLNInferenceError,
    StreamVLNLoadError,
    StreamVLNPrediction,
    StreamVLNRuntime,
    activate_repository_imports,
    embedded_siglip_vision_initialization,
    repository_import_paths,
    validate_repository_module_origins,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SENSOR_CONFIG",
    "FORWARD_STEP_M",
    "MAX_FUTURE_ACTIONS",
    "TURN_STEP_RAD",
    "LazyStreamVLNEngine",
    "StreamVLNAction",
    "StreamVLNConfig",
    "StreamVLNEvaluator",
    "StreamVLNEvaluatorError",
    "StreamVLNInferenceError",
    "StreamVLNLoadError",
    "StreamVLNNativeOutputError",
    "StreamVLNPrediction",
    "StreamVLNRuntime",
    "StreamVLNWaypoint",
    "StreamVLNWaypointPlan",
    "actions_to_cumulative_waypoints",
    "activate_repository_imports",
    "embedded_siglip_vision_initialization",
    "normalize_native_actions",
    "repository_import_paths",
    "validate_repository_module_origins",
]
