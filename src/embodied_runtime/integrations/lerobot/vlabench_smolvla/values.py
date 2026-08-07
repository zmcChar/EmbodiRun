"""Dependency-light values for the native LeRobot adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embodied_runtime.simulators.vlabench_camera import VLABENCH_DATASET_CAMERA_KEYS

VLABENCH_CAMERA_RENAME_MAP: dict[str, str] = {
    f"observation.images.{source}": f"observation.images.camera{index}"
    for index, source in enumerate(VLABENCH_DATASET_CAMERA_KEYS, start=1)
}


class LeRobotRunnerError(RuntimeError):
    """Raised when the native LeRobot policy runner receives invalid data."""


@dataclass(frozen=True)
class LeRobotBindings:
    """Late-bound LeRobot API surface, injectable for dependency-free tests."""

    pre_trained_config: Any
    vlabench_env_config: Any
    make_policy: Callable[..., Any]
    make_pre_post_processors: Callable[..., tuple[Any, Any]]
    make_env_pre_post_processors: Callable[..., tuple[Any, Any]]
    preprocess_observation: Callable[[dict[str, Any]], dict[str, Any]]
    inference_context: Callable[[str, bool], Any]
    version: str | None = None
    synchronize: Callable[[str], None] | None = None


@dataclass(frozen=True)
class SmolVLAAction:
    """One environment-ready action and the time spent in ``select_action``."""

    action: Any
    inference_latency_s: float
    generated_chunk: bool = False
    queue_remaining: int | None = None


@dataclass(frozen=True)
class SmolVLARuntimeFacts:
    """Shapes reported by the loaded checkpoint, processors, and environment."""

    checkpoint: str
    lerobot_version: str | None
    device: str
    input_feature_shapes: dict[str, tuple[int, ...]]
    output_feature_shapes: dict[str, tuple[int, ...]]
    environment_feature_shapes: dict[str, tuple[int, ...]]
    preprocessor_stat_shapes: dict[str, tuple[int, ...]]
    postprocessor_stat_shapes: dict[str, tuple[int, ...]]
    camera_rename_map: dict[str, str]
    chunk_size: int | None
    n_action_steps: int | None
    model_parameter_count: int | None
