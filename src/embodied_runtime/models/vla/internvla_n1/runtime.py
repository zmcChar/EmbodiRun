"""Serialized stateful runtime for InternVLA-N1 DualVLN and NavDP."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any

from .depth import normalize_depth_meters
from .loader import (
    VARIANT_DUALVLN,
    AgentFactory,
    InternVLAConfig,
    InternVLALoadError,
    VariantSpec,
    load_internvla_agent,
)
from .outputs import (
    InternVLAOutputError,
    NativePrediction,
    is_look_down_request,
    native_prediction_from_official,
)

MAX_LOOK_DOWN_RETRIES = 1


class InternVLARuntimeError(RuntimeError):
    """The loaded stateful model failed to produce a native prediction."""


class InternVLARuntime:
    """Lazy bridge to ``InternVLAN1AsyncAgent`` with episode-safe inference.

    The official agent owns temporal history. Calls to :meth:`predict` and
    :meth:`reset` are consequently serialized. The runtime consumes at most
    one internal LOOK_DOWN retry, reusing the same registered camera pair
    because Go2's camera cannot tilt.
    """

    def __init__(
        self,
        config: InternVLAConfig | None = None,
        *,
        variant: str = VARIANT_DUALVLN,
        model_path: str | None = None,
        device: str = "cuda:0",
        resize_w: int = 384,
        resize_h: int = 384,
        num_history: int = 8,
        plan_step_gap: int = 4,
        camera_intrinsic: Any = None,
        internnav_root: str | Path | None = None,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        if config is None:
            values: dict[str, Any] = {
                "variant": variant,
                "model_path": model_path,
                "device": device,
                "resize_w": resize_w,
                "resize_h": resize_h,
                "num_history": num_history,
                "plan_step_gap": plan_step_gap,
                "internnav_root": internnav_root,
            }
            if camera_intrinsic is not None:
                values["camera_intrinsic"] = camera_intrinsic
            config = InternVLAConfig(**values)
        self.config = config
        self._agent_factory = agent_factory
        self._agent: object | None = None
        self._load_lock = threading.Lock()
        self._state_lock = threading.RLock()

    @property
    def variant(self) -> str:
        return self.config.variant

    @property
    def spec(self) -> VariantSpec:
        return self.config.spec

    @property
    def model_path(self) -> str:
        return self.config.resolved_model_path

    @property
    def loaded(self) -> bool:
        return self._agent is not None

    def _ensure_loaded(self) -> object:
        if self._agent is not None:
            return self._agent
        with self._load_lock:
            if self._agent is None:
                self._agent = load_internvla_agent(
                    self.config,
                    agent_factory=self._agent_factory,
                )
            return self._agent

    def load(self) -> InternVLARuntime:
        """Explicitly materialize the lazy agent and return ``self``."""

        with self._state_lock:
            self._ensure_loaded()
        return self

    def reset(self) -> None:
        """Reset model history without forcing an unloaded model to load."""

        with self._state_lock:
            if self._agent is None:
                return
            try:
                self._agent.reset()
            except Exception as error:
                raise InternVLARuntimeError(f"cannot reset InternVLA-N1: {error}") from error

    @staticmethod
    def _rgb_array(rgb: object, numpy: Any) -> Any:
        try:
            array = numpy.asarray(rgb)
        except Exception as error:
            raise InternVLARuntimeError(f"cannot convert RGB input to an array: {error}") from error
        if array.ndim != 3 or array.shape[2] != 3:
            raise InternVLARuntimeError("RGB input must have shape H x W x 3")
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise InternVLARuntimeError("RGB input dimensions must be positive")
        if array.dtype != numpy.uint8:
            raise InternVLARuntimeError("RGB input must use uint8 samples")
        return numpy.ascontiguousarray(array)

    def predict(
        self,
        rgb: object,
        depth_m: object | None,
        instruction: str,
    ) -> NativePrediction:
        """Run one stateful prediction and return the neutral native union."""

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        with self._state_lock:
            agent = self._ensure_loaded()
            try:
                numpy = importlib.import_module("numpy")
                rgb_array = self._rgb_array(rgb, numpy)
                if self.spec.depth_required:
                    if depth_m is None:
                        raise InternVLARuntimeError(
                            "the NavDP variant requires registered depth in metres"
                        )
                    model_depth = normalize_depth_meters(
                        depth_m,
                        expected_shape=tuple(rgb_array.shape[:2]),
                    )
                else:
                    # DualVLN is RGB-only. The official agent retains a common
                    # RGB-D call signature, so give it a semantically empty array.
                    model_depth = numpy.zeros(rgb_array.shape[:2], dtype=numpy.float32)
                pose = numpy.eye(4, dtype=numpy.float32)
                intrinsic = numpy.asarray(
                    self.config.camera_intrinsic,
                    dtype=numpy.float32,
                )
                for retry_index in range(MAX_LOOK_DOWN_RETRIES + 1):
                    result = agent.step(
                        rgb_array,
                        model_depth,
                        pose,
                        instruction,
                        intrinsic=intrinsic,
                        look_down=retry_index > 0,
                    )
                    native = native_prediction_from_official(result)
                    if not is_look_down_request(native):
                        return native
                raise InternVLARuntimeError(
                    "InternVLA repeatedly requested a look-down observation"
                )
            except (InternVLALoadError, InternVLAOutputError, InternVLARuntimeError):
                raise
            except Exception as error:
                raise InternVLARuntimeError(f"InternVLA inference failed: {error}") from error


__all__ = ["MAX_LOOK_DOWN_RETRIES", "InternVLARuntime", "InternVLARuntimeError"]
