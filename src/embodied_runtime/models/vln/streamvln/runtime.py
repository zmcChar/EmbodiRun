"""Lazy, stateful runtime for the official StreamVLN real-world checkpoint."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .actions import (
    MAX_FUTURE_ACTIONS,
    StreamVLNNativeOutputError,
    normalize_native_actions,
)
from .config import DEFAULT_MODEL, StreamVLNConfig
from .errors import StreamVLNInferenceError, StreamVLNLoadError
from .evaluator import StreamVLNEvaluator
from .loader import materialize_streamvln
from .prediction import StreamVLNPrediction


class StreamVLNRuntime:
    """Lazy owner of one model and one recurrent evaluator.

    Construction performs validation only. PyTorch, Transformers, the official
    repository, and checkpoint weights are first touched by :meth:`load`,
    :meth:`reset_memory`, or :meth:`predict`.
    """

    def __init__(
        self,
        *,
        streamvln_root: str | Path,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cuda:0",
        cuda_memory_fraction: float | None = None,
        num_future_steps: int = MAX_FUTURE_ACTIONS,
        num_frames: int = 32,
        num_history: int = 8,
        model_max_length: int = 4096,
        max_new_tokens: int = 10_000,
        dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
        revision: str | None = None,
        local_files_only: bool = False,
        use_embedded_vision_weights: bool = True,
        warmup: bool = True,
        evaluator_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = StreamVLNConfig(
            streamvln_root=streamvln_root,
            model_path=model_path,
            device=device,
            cuda_memory_fraction=cuda_memory_fraction,
            num_future_steps=num_future_steps,
            num_frames=num_frames,
            num_history=num_history,
            model_max_length=model_max_length,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            attn_implementation=attn_implementation,
            revision=revision,
            local_files_only=local_files_only,
            use_embedded_vision_weights=use_embedded_vision_weights,
            warmup=warmup,
        )
        self._evaluator_factory = evaluator_factory or StreamVLNEvaluator
        self._load_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._evaluator: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None
        self._state_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._evaluator is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def evaluator(self) -> Any | None:
        return self._evaluator

    @property
    def requires_reset(self) -> bool:
        """Whether a failed partial cadence blocks further inference."""

        with self._state_lock:
            return self._state_error is not None

    def _load(self) -> Any:
        if self._evaluator is not None:
            return self._evaluator
        with self._load_lock:
            if self._evaluator is not None:
                return self._evaluator
            try:
                loaded = materialize_streamvln(self.config, self._evaluator_factory)
                self._model = loaded.model
                self._evaluator = loaded.evaluator
                self._load_error = None
            except Exception as error:
                self._load_error = f"{type(error).__name__}: {error}"
                raise StreamVLNLoadError(f"failed to load StreamVLN: {self._load_error}") from error
        return self._evaluator

    def load(self) -> StreamVLNRuntime:
        """Explicitly materialize the lazy runtime and return ``self``."""

        self._load()
        return self

    def reset_memory(self) -> None:
        with self._state_lock:
            evaluator = self._load()
            try:
                evaluator.reset_memory()
            except Exception as error:
                self._state_error = f"reset failed: {type(error).__name__}: {error}"
                raise StreamVLNInferenceError(self._state_error) from error
            self._state_error = None

    def reset(self) -> None:
        """Short spelling used by episode-oriented policies."""

        self.reset_memory()

    def predict(self, rgb_image: Any, instruction: str) -> StreamVLNPrediction:
        """Run one official four-step real-world service cadence."""

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        with self._state_lock:
            if self._state_error is not None:
                raise StreamVLNInferenceError(
                    "StreamVLN state was discarded after a partial inference; "
                    "call reset() before retrying"
                )
            return self._predict_locked(rgb_image, instruction)

    def _discard_failed_state(self, evaluator: Any, error: Exception) -> str:
        detail = f"{type(error).__name__}: {error}"
        try:
            evaluator.reset_memory()
        except Exception as reset_error:  # noqa: BLE001 - preserve the primary failure
            detail += f"; evaluator reset also failed: {type(reset_error).__name__}: {reset_error}"
        # Even when the defensive reset succeeds, callers must explicitly
        # acknowledge that the episode's recurrent context was lost.
        self._state_error = detail
        return detail

    def _predict_locked(self, rgb_image: Any, instruction: str) -> StreamVLNPrediction:
        evaluator = self._load()
        latest_actions: Sequence[object] | None = None
        latest_output = ""
        generation_time_s = 0.0
        try:
            for _ in range(self.config.num_future_steps):
                run_model = evaluator.step_id % self.config.num_future_steps == 0
                actions, generate_time, raw_output = evaluator.step(
                    0,
                    rgb_image,
                    instruction,
                    run_model=run_model,
                )
                if actions is not None:
                    latest_actions = actions
                if raw_output is not None:
                    latest_output = str(raw_output)
                if generate_time is not None:
                    generation_time_s = max(generation_time_s, float(generate_time))
                evaluator.step_id += 1
            if latest_actions is None:
                raise StreamVLNNativeOutputError("StreamVLN evaluator returned no action sequence")

            normalized = normalize_native_actions(
                latest_actions,
                max_actions=self.config.num_future_steps,
            )
            return StreamVLNPrediction(
                actions=normalized,
                raw_output=latest_output,
                generation_time_s=generation_time_s,
            )
        except StreamVLNNativeOutputError as error:
            self._discard_failed_state(evaluator, error)
            raise
        except Exception as error:
            detail = self._discard_failed_state(evaluator, error)
            raise StreamVLNInferenceError(
                f"StreamVLN evaluator failed and requires reset: {detail}"
            ) from error


__all__ = ["StreamVLNRuntime"]
