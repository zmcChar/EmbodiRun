"""Session-local CUDA Graph orchestration for selected entrypoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .cuda_graph_cache import CudaGraphCache
from .cuda_graph_capture import (
    CudaGraphWarmupError,
    capture_call,
    refresh_call_inputs,
    replay_call,
)
from .cuda_graph_key import (
    CudaGraphInputError,
    GraphKey,
    prepare_graph_inputs,
    tree_signature,
)


@dataclass(frozen=True, slots=True)
class CudaGraphConfig:
    """Small backend-private configuration carried by a compiled artifact."""

    entrypoints: frozenset[str] = frozenset()
    dynamic_scalar_inputs: Mapping[str, frozenset[str]] = field(default_factory=dict)
    warmup_steps: int = 1
    max_graphs: int = 16

    @property
    def enabled(self) -> bool:
        return bool(self.entrypoints)


class CudaGraphExecutor:
    """Capture and replay selected callables without knowing model semantics."""

    def __init__(
        self,
        torch: Any,
        device: Any,
        config: CudaGraphConfig,
    ) -> None:
        self._torch = torch
        self._device = device
        self._config = config
        self._cache = CudaGraphCache(config.max_graphs)
        self._failures: dict[str, str] = {}
        self._captures = 0
        self._replays = 0
        self._fallbacks = 0
        with torch.cuda.device(device):
            self._stream = torch.cuda.Stream(device=device)

    def handles(self, entrypoint: str) -> bool:
        return entrypoint in self._config.entrypoints

    @property
    def stats(self) -> Mapping[str, int]:
        return {
            "captures": self._captures,
            "replays": self._replays,
            "fallbacks": self._fallbacks,
            "cached_graphs": len(self._cache),
        }

    @property
    def failures(self) -> Mapping[str, str]:
        return dict(self._failures)

    def _fallback(
        self,
        entrypoint: str,
        reason: str,
        function: Callable[[Any], Any],
        inputs: Any,
    ) -> Any:
        self._fallbacks += 1
        self._failures[entrypoint] = reason
        return function(inputs)

    def execute(
        self,
        entrypoint: str,
        inputs: Any,
        function: Callable[[Any], Any],
    ) -> Any:
        try:
            graph_inputs = prepare_graph_inputs(
                self._torch,
                self._device,
                entrypoint,
                inputs,
                self._config.dynamic_scalar_inputs.get(entrypoint, frozenset()),
            )
            signature = tree_signature(self._torch, graph_inputs)
        except Exception as error:  # noqa: BLE001
            return self._fallback(
                entrypoint,
                f"input: {type(error).__name__}: {error}",
                function,
                inputs,
            )
        key: GraphKey = (entrypoint, signature)
        failed_reason = self._cache.failure_reason(key)
        if failed_reason is not None:
            return self._fallback(entrypoint, failed_reason, function, inputs)

        captured = self._cache.get(key)
        cache_hit = captured is not None
        if captured is None:
            if self._cache.full:
                reason = f"cache limit of {self._config.max_graphs} graphs reached"
                return self._fallback(entrypoint, reason, function, inputs)
            try:
                captured = capture_call(
                    self._torch,
                    self._device,
                    self._stream,
                    function,
                    graph_inputs,
                    warmup_steps=self._config.warmup_steps,
                )
                self._cache.put(key, captured)
            except CudaGraphWarmupError as error:
                raise error.original
            except Exception as error:  # noqa: BLE001
                reason = f"capture: {type(error).__name__}: {error}"
                self._cache.remember_failure(key, reason)
                return self._fallback(entrypoint, reason, function, inputs)
        else:
            try:
                refresh_call_inputs(
                    self._torch,
                    self._device,
                    self._stream,
                    captured,
                    graph_inputs,
                )
            except Exception as error:  # noqa: BLE001
                captured.graph.reset()
                self._cache.pop(key)
                reason = f"input copy: {type(error).__name__}: {error}"
                self._cache.remember_failure(key, reason)
                return self._fallback(entrypoint, reason, function, inputs)

        try:
            output = replay_call(
                self._torch,
                self._device,
                self._stream,
                captured,
            )
        except Exception as error:  # noqa: BLE001
            captured.graph.reset()
            self._cache.pop(key)
            reason = f"replay: {type(error).__name__}: {error}"
            self._cache.remember_failure(key, reason)
            return self._fallback(entrypoint, reason, function, inputs)
        if cache_hit:
            self._replays += 1
        else:
            self._captures += 1
        return output

    def close(self) -> None:
        failure: Exception | None = None
        try:
            self._stream.synchronize()
        except Exception as error:  # noqa: BLE001
            failure = error
        try:
            self._cache.close()
        except Exception as error:  # noqa: BLE001
            if failure is None:
                failure = error
        if failure is not None:
            raise failure


__all__ = ["CudaGraphConfig", "CudaGraphExecutor", "CudaGraphInputError"]
