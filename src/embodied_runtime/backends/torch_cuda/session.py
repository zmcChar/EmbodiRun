"""Loaded PyTorch artifact session."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from threading import RLock
from typing import Any

from embodied_runtime.contracts import (
    BackendExecutionError,
    DeviceInfo,
    ExecutionContext,
    MemoryStats,
    RequestCancelledError,
)

from .compiler import TorchArtifactPayload
from .cuda_graph import CudaGraphExecutor
from .memory import memory_stats as read_memory_stats


def _move_tensor_tree(
    torch: Any,
    value: Any,
    *,
    device: Any,
    dtype: Any | None,
    non_blocking: bool,
) -> Any:
    """Recursively move tensors while preserving the surrounding tree shape."""

    if isinstance(value, torch.Tensor):
        target_dtype = dtype if dtype is not None and value.is_floating_point() else None
        return value.to(
            device=device,
            dtype=target_dtype,
            non_blocking=non_blocking,
        )
    if isinstance(value, Mapping):
        return {
            key: _move_tensor_tree(
                torch,
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        moved = tuple(
            _move_tensor_tree(
                torch,
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for item in value
        )
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return moved
    if isinstance(value, list):
        return [
            _move_tensor_tree(
                torch,
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for item in value
        ]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _move_tensor_tree(
                torch,
                getattr(value, field.name),
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for field in fields(value)
        }
        return replace(value, **updates)
    return value


def _add_scaled_tree(torch: Any, state: Any, update: Any, scale: float) -> Any:
    """Apply ``state + scale * update`` without moving arithmetic off-device."""

    if isinstance(state, Mapping):
        if not isinstance(update, Mapping) or state.keys() != update.keys():
            raise TypeError("state and update mappings must have identical keys")
        return type(state)(
            (key, _add_scaled_tree(torch, state[key], update[key], scale)) for key in state
        )
    if isinstance(state, tuple):
        if not isinstance(update, tuple) or len(state) != len(update):
            raise TypeError("state and update tuples must have identical lengths")
        values = tuple(
            _add_scaled_tree(torch, left, right, scale) for left, right in zip(state, update)
        )
        if hasattr(state, "_fields"):
            return type(state)(*values)
        return values
    if isinstance(state, list):
        if not isinstance(update, list) or len(state) != len(update):
            raise TypeError("state and update lists must have identical lengths")
        return [_add_scaled_tree(torch, left, right, scale) for left, right in zip(state, update)]
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
        if not isinstance(update, type(state)) or len(state) != len(update):
            raise TypeError("state and update sequences must have identical lengths")
        return type(state)(
            _add_scaled_tree(torch, left, right, scale) for left, right in zip(state, update)
        )
    if is_dataclass(state) and not isinstance(state, type):
        if not is_dataclass(update) or type(update) is not type(state):
            raise TypeError("state and update dataclasses must have identical types")
        return replace(
            state,
            **{
                field.name: _add_scaled_tree(
                    torch,
                    getattr(state, field.name),
                    getattr(update, field.name),
                    scale,
                )
                for field in fields(state)
            },
        )
    try:
        if isinstance(state, torch.Tensor) and isinstance(update, torch.Tensor):
            # Preserve the engine's declared Euler operation order exactly.
            # ``torch.add(..., alpha=scale)`` may fuse multiply-add and round
            # differently from the model/reference expression.
            return state + update * scale
        return state + update * scale
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(
            f"cannot add scaled leaves {type(state).__name__} and {type(update).__name__}"
        ) from error


class TorchBackendSession:
    """Execute portable package stages on one CPU or CUDA device."""

    def __init__(
        self,
        torch: Any,
        device: DeviceInfo,
        payload: TorchArtifactPayload,
        *,
        release_module: Callable[[], None] | None = None,
    ) -> None:
        self._torch = torch
        self._device_info = device
        self._torch_device = torch.device(payload.device_id)
        # Compilation artifacts are reusable.  All dictionaries changed by
        # deferred fallback or close must therefore belong to this session.
        self._payload = payload.for_session()
        self._release_module = release_module or (lambda: None)
        self._cuda_graph_lock = RLock()
        self._cuda_graphs = (
            CudaGraphExecutor(torch, self._torch_device, self._payload.cuda_graph)
            if self._payload.cuda_graph.enabled
            else None
        )
        self._closed = False

    @property
    def package_id(self) -> str:
        return self._payload.package_id

    @property
    def device(self) -> DeviceInfo:
        return self._device_info

    @property
    def compile_failures(self) -> Mapping[str, str]:
        """Failures that caused an entrypoint to use eager execution."""

        return dict(self._payload.compile_failures)

    @property
    def cuda_graph_stats(self) -> Mapping[str, int]:
        if self._cuda_graphs is None:
            return {
                "captures": 0,
                "replays": 0,
                "fallbacks": 0,
                "cached_graphs": 0,
            }
        return self._cuda_graphs.stats

    @property
    def cuda_graph_failures(self) -> Mapping[str, str]:
        if self._cuda_graphs is None:
            return {}
        return self._cuda_graphs.failures

    @property
    def actual_mode(self) -> str:
        modes = {
            self._payload.entrypoints[name] is self._payload.eager_entrypoints[name]
            for name in self._payload.entrypoints
        }
        if modes == {True}:
            return "eager"
        if modes == {False}:
            return "compile"
        return "mixed"

    def _ensure_open(self) -> None:
        if self._closed:
            raise BackendExecutionError("PyTorch backend session is closed")

    def _invoke(self, entrypoint: str, inputs: Any) -> Any:
        try:
            function = self._payload.entrypoints[entrypoint]
        except KeyError as error:
            available = ", ".join(sorted(self._payload.entrypoints))
            raise BackendExecutionError(
                f"unknown model entrypoint {entrypoint!r}; available: {available}"
            ) from error

        try:
            return function(inputs)
        except Exception as compiled_error:
            eager = self._payload.eager_entrypoints[entrypoint]
            is_compiled = function is not eager
            if not (is_compiled and self._payload.fallback_to_eager):
                raise BackendExecutionError(
                    f"PyTorch entrypoint {entrypoint!r} failed on "
                    f"{self._payload.device_id}: {type(compiled_error).__name__}: "
                    f"{compiled_error}"
                ) from compiled_error
            try:
                output = eager(inputs)
            except Exception as eager_error:
                raise BackendExecutionError(
                    f"compiled and eager execution both failed for {entrypoint!r}; "
                    f"compiled={type(compiled_error).__name__}: {compiled_error}; "
                    f"eager={type(eager_error).__name__}: {eager_error}"
                ) from eager_error
            self._payload.entrypoints[entrypoint] = eager
            self._payload.compile_failures[entrypoint] = (
                f"deferred {type(compiled_error).__name__}: {compiled_error}"
            )
            return output

    def _synchronize_if_needed(self) -> None:
        if self._payload.synchronize_on_submit and self._payload.device_id.startswith("cuda"):
            self._torch.cuda.synchronize(self._torch_device)

    def _execute_moved(self, entrypoint: str, moved_inputs: Any, *, use_cuda_graph: bool) -> Any:
        try:
            if use_cuda_graph:
                assert self._cuda_graphs is not None

                def invoke(inputs: Any) -> Any:
                    return self._invoke(entrypoint, inputs)

                if self._payload.inference_mode:
                    with self._torch.inference_mode():
                        output = self._cuda_graphs.execute(entrypoint, moved_inputs, invoke)
                else:  # rejected when CUDA Graph entrypoints are configured
                    output = self._cuda_graphs.execute(entrypoint, moved_inputs, invoke)
            elif self._payload.inference_mode:
                with self._torch.inference_mode():
                    output = self._invoke(entrypoint, moved_inputs)
            else:
                output = self._invoke(entrypoint, moved_inputs)
        except (BackendExecutionError, RequestCancelledError):
            raise
        except Exception as error:
            raise BackendExecutionError(
                f"PyTorch entrypoint {entrypoint!r} failed: {type(error).__name__}: {error}"
            ) from error
        try:
            self._synchronize_if_needed()
        except Exception as error:
            raise BackendExecutionError(
                f"CUDA synchronization failed after entrypoint {entrypoint!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
        return output

    def _submit_impl(
        self,
        entrypoint: str,
        inputs: Any,
        context: ExecutionContext,
    ) -> Any:
        self._ensure_open()
        if context.cancelled:
            raise RequestCancelledError(f"request cancelled before entrypoint {entrypoint!r}")
        moved_inputs = _move_tensor_tree(
            self._torch,
            inputs,
            device=self._torch_device,
            dtype=self._payload.dtype,
            non_blocking=self._payload.non_blocking,
        )
        use_cuda_graph = self._cuda_graphs is not None and self._cuda_graphs.handles(entrypoint)
        output = self._execute_moved(
            entrypoint,
            moved_inputs,
            use_cuda_graph=use_cuda_graph,
        )
        if context.cancelled:
            raise RequestCancelledError(
                f"request cancelled while entrypoint {entrypoint!r} was running"
            )
        return output

    def submit(
        self,
        entrypoint: str,
        inputs: Any,
        context: ExecutionContext,
    ) -> Any:
        if self._cuda_graphs is not None:
            # Static graph buffers and capture must not overlap another
            # operation in this session. Include non-captured stages, input
            # movement, and final synchronization in the same critical section.
            with self._cuda_graph_lock:
                return self._submit_impl(entrypoint, inputs, context)
        return self._submit_impl(entrypoint, inputs, context)

    def _add_scaled_impl(
        self,
        state: Any,
        update: Any,
        scale: float,
        context: ExecutionContext,
    ) -> Any:
        """Execute an Euler state update on the session's device."""

        self._ensure_open()
        if context.cancelled:
            raise RequestCancelledError("request cancelled before backend state update")
        moved_state = _move_tensor_tree(
            self._torch,
            state,
            device=self._torch_device,
            dtype=self._payload.dtype,
            non_blocking=self._payload.non_blocking,
        )
        moved_update = _move_tensor_tree(
            self._torch,
            update,
            device=self._torch_device,
            dtype=self._payload.dtype,
            non_blocking=self._payload.non_blocking,
        )
        try:
            output = _add_scaled_tree(self._torch, moved_state, moved_update, scale)
            self._synchronize_if_needed()
        except Exception as error:
            raise BackendExecutionError(
                f"PyTorch state update failed on {self._payload.device_id}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if context.cancelled:
            raise RequestCancelledError("request cancelled while backend state update was running")
        return output

    def add_scaled(
        self,
        state: Any,
        update: Any,
        scale: float,
        context: ExecutionContext,
    ) -> Any:
        if self._cuda_graphs is not None:
            with self._cuda_graph_lock:
                return self._add_scaled_impl(state, update, scale, context)
        return self._add_scaled_impl(state, update, scale, context)

    def memory_stats(self) -> MemoryStats:
        if self._cuda_graphs is not None:
            with self._cuda_graph_lock:
                self._ensure_open()
                return read_memory_stats(self._torch, self._payload.device_id)
        self._ensure_open()
        return read_memory_stats(self._torch, self._payload.device_id)

    def close(self) -> None:
        with self._cuda_graph_lock:
            if self._closed:
                return
            is_cuda = self._payload.device_id.startswith("cuda")
            module = self._payload.runtime_module
            failure: Exception | None = None
            try:
                if is_cuda and (
                    self._cuda_graphs is not None
                    or self._payload.offload_module_on_close
                    or self._payload.empty_cache_on_close
                ):
                    self._torch.cuda.synchronize(self._torch_device)
            except Exception as error:
                failure = error
            try:
                if self._cuda_graphs is not None:
                    self._cuda_graphs.close()
            except Exception as error:
                if failure is None:
                    failure = error
            try:
                if is_cuda and self._payload.offload_module_on_close and module is not None:
                    module.to(device=self._torch.device("cpu"))
            except Exception as error:
                if failure is None:
                    failure = error
            try:
                self._payload.entrypoints.clear()
                self._payload.eager_entrypoints.clear()
                self._payload.runtime_module = None
                self._closed = True
                if is_cuda and self._payload.empty_cache_on_close:
                    try:
                        self._torch.cuda.empty_cache()
                    except Exception as error:
                        if failure is None:
                            failure = error
                self._release_module()
            except Exception as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise BackendExecutionError(
                f"could not release PyTorch session on {self._payload.device_id}: "
                f"{type(failure).__name__}: {failure}"
            ) from failure
