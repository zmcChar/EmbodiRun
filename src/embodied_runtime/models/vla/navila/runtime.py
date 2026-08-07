"""Thread-safe lazy runtime for stateless NaVILA eight-frame inference."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL, NaVILAConfig
from .errors import NaVILAInferenceError, NaVILALoadError
from .loader import materialize_navila
from .prediction import NaVILAPrediction


class NaVILARuntime:
    def __init__(
        self,
        *,
        navila_root: str | Path,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cuda:0",
        cuda_memory_fraction: float | None = None,
        max_new_tokens: int = 32,
        local_files_only: bool = True,
        materializer: Callable[..., Any] = materialize_navila,
    ) -> None:
        self.config = NaVILAConfig(
            navila_root=navila_root,
            model_path=model_path,
            device=device,
            cuda_memory_fraction=cuda_memory_fraction,
            max_new_tokens=max_new_tokens,
            local_files_only=local_files_only,
        )
        self._materializer = materializer
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()
        self._model: Any | None = None
        self._evaluator: Any | None = None
        self._closed = False
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._evaluator is not None

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def evaluator(self) -> Any | None:
        return self._evaluator

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _load(self) -> Any:
        if self._closed:
            raise NaVILALoadError("NaVILA runtime is closed")
        if self._evaluator is not None:
            return self._evaluator
        with self._load_lock:
            if self._closed:
                raise NaVILALoadError("NaVILA runtime is closed")
            if self._evaluator is None:
                try:
                    loaded = self._materializer(self.config)
                    self._model = loaded.model
                    self._evaluator = loaded.evaluator
                    self._load_error = None
                except Exception as error:
                    self._load_error = f"{type(error).__name__}: {error}"
                    if isinstance(error, NaVILALoadError):
                        raise
                    raise NaVILALoadError(f"failed to load NaVILA: {self._load_error}") from error
        return self._evaluator

    def load(self) -> NaVILARuntime:
        self._load()
        return self

    def predict(
        self,
        rgb_frames: Sequence[Any],
        instruction: str,
    ) -> NaVILAPrediction:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        with self._inference_lock:
            evaluator = self._load()
            try:
                prediction = evaluator.predict(rgb_frames, instruction)
            except (NaVILAInferenceError, ValueError):
                raise
            except Exception as error:
                raise NaVILAInferenceError(f"NaVILA inference failed: {error}") from error
            if not isinstance(prediction, NaVILAPrediction):
                raise NaVILAInferenceError("NaVILA evaluator returned an invalid prediction")
            return prediction

    def close(self) -> None:
        with self._inference_lock:
            self._closed = True
            self._evaluator = None
            self._model = None


__all__ = ["NaVILARuntime"]
