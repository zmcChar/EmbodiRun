"""Session-local CUDA Graph capture for selected backend entrypoints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any


class CudaGraphInputError(TypeError):
    """An input tree cannot be represented by stable CUDA Graph buffers."""


class _CudaGraphWarmupError(RuntimeError):
    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


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


@dataclass(slots=True)
class _CapturedCall:
    graph: Any
    static_inputs: Any
    static_outputs: Any


@dataclass(frozen=True, slots=True)
class _DynamicScalar:
    """A Python scalar that must be copied into a graph-owned device buffer."""

    value: bool | int | float
    dtype: Any
    device: Any


def _mapping_like(original: Mapping[Any, Any], items: list[tuple[Any, Any]]) -> Mapping[Any, Any]:
    if type(original) is dict:
        return dict(items)
    try:
        return type(original)(items)
    except (TypeError, ValueError) as error:
        raise CudaGraphInputError(
            f"mapping type {type(original).__qualname__} cannot be reconstructed"
        ) from error


def _tree_signature(
    torch: Any,
    value: Any,
    seen_storages: set[tuple[str, int]] | None = None,
) -> tuple[Any, ...]:
    if seen_storages is None:
        seen_storages = set()
    if isinstance(value, _DynamicScalar):
        # The scalar value is deliberately absent: its dtype/device define the
        # fixed buffer, while every replay refreshes the buffer's contents.
        return ("dynamic_scalar", str(value.dtype), str(value.device))
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            raise CudaGraphInputError(f"tensor layout {value.layout} is not supported")
        if value.device.type != "cuda":
            raise CudaGraphInputError(f"tensor is on {value.device}, expected a CUDA device")
        if value.storage_offset() != 0:
            raise CudaGraphInputError(
                "tensor views with non-zero storage offsets are not supported"
            )
        if value.numel() > 0:
            storage = (str(value.device), int(value.untyped_storage().data_ptr()))
            if storage in seen_storages:
                raise CudaGraphInputError("tensor leaves must not share storage")
            seen_storages.add(storage)
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (key, _tree_signature(torch, item, seen_storages)) for key, item in value.items()
            ),
        )
    if isinstance(value, tuple):
        return (
            "tuple",
            type(value).__module__,
            type(value).__qualname__,
            tuple(_tree_signature(torch, item, seen_storages) for item in value),
        )
    if isinstance(value, list):
        return (
            "list",
            tuple(_tree_signature(torch, item, seen_storages) for item in value),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    field.name,
                    _tree_signature(torch, getattr(value, field.name), seen_storages),
                )
                for field in fields(value)
            ),
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        # Python leaves are constants during capture, so their values belong in
        # the cache key. Dynamic leaves must be tensors.
        return ("constant", type(value).__module__, type(value).__qualname__, repr(value))
    raise CudaGraphInputError(
        f"leaf type {type(value).__module__}.{type(value).__qualname__} is not supported"
    )


def _make_static_tree(torch: Any, value: Any) -> Any:
    if isinstance(value, _DynamicScalar):
        return torch.tensor(
            value.value,
            dtype=value.dtype,
            device=value.device,
        )
    if isinstance(value, torch.Tensor):
        static = torch.empty_strided(
            tuple(value.shape),
            tuple(value.stride()),
            dtype=value.dtype,
            device=value.device,
        )
        static.copy_(value)
        return static
    if isinstance(value, Mapping):
        return _mapping_like(
            value,
            [(key, _make_static_tree(torch, item)) for key, item in value.items()],
        )
    if isinstance(value, tuple):
        items = tuple(_make_static_tree(torch, item) for item in value)
        if type(value) is tuple:
            return items
        if hasattr(value, "_fields"):
            return type(value)(*items)
        try:
            return type(value)(items)
        except (TypeError, ValueError) as error:
            raise CudaGraphInputError(
                f"tuple type {type(value).__qualname__} cannot be reconstructed"
            ) from error
    if isinstance(value, list):
        return [_make_static_tree(torch, item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _make_static_tree(torch, getattr(value, field.name))
                for field in fields(value)
            },
        )
    return value


def _copy_into_static(torch: Any, static: Any, value: Any) -> None:
    if isinstance(static, torch.Tensor):
        if isinstance(value, _DynamicScalar):
            static.fill_(value.value)
        else:
            static.copy_(value)
        return
    if isinstance(static, Mapping):
        for key in static:
            _copy_into_static(torch, static[key], value[key])
        return
    if isinstance(static, (tuple, list)):
        for target, source in zip(static, value, strict=True):
            _copy_into_static(torch, target, source)
        return
    if is_dataclass(static) and not isinstance(static, type):
        for field in fields(static):
            _copy_into_static(
                torch,
                getattr(static, field.name),
                getattr(value, field.name),
            )


def _clone_output_tree(torch: Any, value: Any) -> Any:
    """Detach returned values from graph-owned buffers."""

    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, Mapping):
        return _mapping_like(
            value,
            [(key, _clone_output_tree(torch, item)) for key, item in value.items()],
        )
    if isinstance(value, tuple):
        items = tuple(_clone_output_tree(torch, item) for item in value)
        if type(value) is tuple:
            return items
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return type(value)(items)
    if isinstance(value, list):
        return [_clone_output_tree(torch, item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _clone_output_tree(torch, getattr(value, field.name))
                for field in fields(value)
            },
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise CudaGraphInputError(
        f"output leaf type {type(value).__module__}.{type(value).__qualname__} is not supported"
    )


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
        self._cache: dict[tuple[str, tuple[Any, ...]], _CapturedCall] = {}
        self._failed: OrderedDict[tuple[str, tuple[Any, ...]], str] = OrderedDict()
        self._failures: dict[str, str] = {}
        self._captures = 0
        self._replays = 0
        self._fallbacks = 0
        with torch.cuda.device(device):
            self._stream = torch.cuda.Stream(device=device)

    def handles(self, entrypoint: str) -> bool:
        return entrypoint in self._config.entrypoints

    def _prepare_inputs(self, entrypoint: str, inputs: Any) -> Any:
        names = self._config.dynamic_scalar_inputs.get(entrypoint, frozenset())
        if not names:
            return inputs
        if not isinstance(inputs, Mapping):
            raise CudaGraphInputError(
                f"dynamic scalar inputs for {entrypoint!r} require a top-level mapping"
            )

        missing = names.difference(inputs)
        if missing:
            formatted = ", ".join(sorted(missing))
            raise CudaGraphInputError(
                f"dynamic scalar input(s) are missing for {entrypoint!r}: {formatted}"
            )

        prepared: list[tuple[Any, Any]] = []
        for key, value in inputs.items():
            if key not in names or isinstance(value, self._torch.Tensor):
                prepared.append((key, value))
                continue
            if isinstance(value, bool):
                dtype = self._torch.bool
            elif isinstance(value, int):
                dtype = self._torch.int64
            elif isinstance(value, float):
                # Match Python float semantics before the model casts to its
                # working dtype; avoiding a float32 intermediate also keeps
                # eager and captured schedules bitwise aligned.
                dtype = self._torch.float64
            else:
                raise CudaGraphInputError(
                    f"dynamic scalar input {key!r} for {entrypoint!r} must be "
                    f"bool, int, float, or Tensor; got {type(value).__qualname__}"
                )
            prepared.append(
                (
                    key,
                    _DynamicScalar(
                        value=value,
                        dtype=dtype,
                        device=self._device,
                    ),
                )
            )
        return _mapping_like(inputs, prepared)

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

    def _remember_failure(
        self,
        key: tuple[str, tuple[Any, ...]],
        reason: str,
    ) -> None:
        self._failed[key] = reason
        self._failed.move_to_end(key)
        while len(self._failed) > self._config.max_graphs:
            self._failed.popitem(last=False)

    def _capture(
        self,
        function: Callable[[Any], Any],
        inputs: Any,
    ) -> _CapturedCall:
        torch = self._torch
        static_inputs = _make_static_tree(torch, inputs)
        with torch.cuda.device(self._device):
            current = torch.cuda.current_stream(self._device)
            self._stream.wait_stream(current)
            try:
                with torch.cuda.stream(self._stream):
                    for _ in range(self._config.warmup_steps):
                        function(static_inputs)
                self._stream.synchronize()
            except Exception as error:
                raise _CudaGraphWarmupError(error) from error

            graph = torch.cuda.CUDAGraph()
            try:
                # Every cached graph owns its pool. Sharing is only safe under
                # stricter replay-order guarantees than this entrypoint cache
                # exposes.
                with torch.cuda.graph(graph, stream=self._stream):
                    static_outputs = function(static_inputs)
                self._stream.synchronize()
            except Exception:
                graph.reset()
                raise
        return _CapturedCall(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
        )

    def execute(
        self,
        entrypoint: str,
        inputs: Any,
        function: Callable[[Any], Any],
    ) -> Any:
        try:
            graph_inputs = self._prepare_inputs(entrypoint, inputs)
            signature = _tree_signature(self._torch, graph_inputs)
        except Exception as error:
            return self._fallback(
                entrypoint,
                f"input: {type(error).__name__}: {error}",
                function,
                inputs,
            )
        key = (entrypoint, signature)
        if key in self._failed:
            return self._fallback(
                entrypoint,
                self._failed[key],
                function,
                inputs,
            )

        captured = self._cache.get(key)
        cache_hit = captured is not None
        if captured is None:
            if len(self._cache) >= self._config.max_graphs:
                reason = f"cache limit of {self._config.max_graphs} graphs reached"
                return self._fallback(
                    entrypoint,
                    reason,
                    function,
                    inputs,
                )
            try:
                captured = self._capture(function, graph_inputs)
                self._cache[key] = captured
            except _CudaGraphWarmupError as error:
                raise error.original
            except Exception as error:
                reason = f"capture: {type(error).__name__}: {error}"
                self._remember_failure(key, reason)
                return self._fallback(
                    entrypoint,
                    reason,
                    function,
                    inputs,
                )
        else:
            try:
                with self._torch.cuda.device(self._device):
                    current = self._torch.cuda.current_stream(self._device)
                    self._stream.wait_stream(current)
                    with self._torch.cuda.stream(self._stream):
                        _copy_into_static(
                            self._torch,
                            captured.static_inputs,
                            graph_inputs,
                        )
            except Exception as error:
                captured.graph.reset()
                self._cache.pop(key, None)
                reason = f"input copy: {type(error).__name__}: {error}"
                self._remember_failure(key, reason)
                return self._fallback(
                    entrypoint,
                    reason,
                    function,
                    inputs,
                )

        try:
            with self._torch.cuda.device(self._device):
                current = self._torch.cuda.current_stream(self._device)
                self._stream.wait_stream(current)
                with self._torch.cuda.stream(self._stream):
                    captured.graph.replay()
                    output = _clone_output_tree(self._torch, captured.static_outputs)
        except Exception as error:
            captured.graph.reset()
            self._cache.pop(key, None)
            reason = f"replay: {type(error).__name__}: {error}"
            self._remember_failure(key, reason)
            return self._fallback(
                entrypoint,
                reason,
                function,
                inputs,
            )
        if cache_hit:
            self._replays += 1
        else:
            self._captures += 1
        return output

    def close(self) -> None:
        failure: Exception | None = None
        try:
            self._stream.synchronize()
        except Exception as error:
            failure = error
        for captured in self._cache.values():
            try:
                captured.graph.reset()
            except Exception as error:
                if failure is None:
                    failure = error
        self._cache.clear()
        self._failed.clear()
        if failure is not None:
            raise failure


__all__ = ["CudaGraphConfig", "CudaGraphExecutor", "CudaGraphInputError"]
