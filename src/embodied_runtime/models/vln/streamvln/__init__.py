"""Modular StreamVLN real-world runtime for language-guided Go2 navigation.

The package exposes normalized checkpoint-native actions rather than task-domain
navigation values. Navigation policies own spatial interpretation.
"""

from .actions import (
    MAX_FUTURE_ACTIONS,
    StreamVLNAction,
    StreamVLNNativeOutputError,
    normalize_native_actions,
)
from .config import DEFAULT_MODEL, DEFAULT_SENSOR_CONFIG, StreamVLNConfig
from .errors import StreamVLNInferenceError, StreamVLNLoadError
from .evaluator import StreamVLNEvaluator, StreamVLNEvaluatorError
from .prediction import StreamVLNPrediction
from .repository import (
    activate_repository_imports,
    repository_import_paths,
    validate_repository_module_origins,
)
from .runtime import StreamVLNRuntime
from .vision import embedded_siglip_vision_initialization

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SENSOR_CONFIG",
    "MAX_FUTURE_ACTIONS",
    "StreamVLNAction",
    "StreamVLNConfig",
    "StreamVLNEvaluator",
    "StreamVLNEvaluatorError",
    "StreamVLNInferenceError",
    "StreamVLNLoadError",
    "StreamVLNNativeOutputError",
    "StreamVLNPrediction",
    "StreamVLNRuntime",
    "activate_repository_imports",
    "embedded_siglip_vision_initialization",
    "normalize_native_actions",
    "repository_import_paths",
    "validate_repository_module_origins",
]
