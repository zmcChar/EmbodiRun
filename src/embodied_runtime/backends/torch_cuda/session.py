"""Lifecycle and coordination for one loaded PyTorch artifact session."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any

from embodied_runtime.engine.context import ExecutionContext
from embodied_runtime.engine.errors import RequestCancelledError

from ..device import DeviceInfo
from ..errors import BackendExecutionError
from ..memory import MemoryStats
from .compiler import TorchArtifactPayload
from .cuda_graph import CudaGraphExecutor
from .memory import memory_stats as read_memory_stats
from .session_stage import actual_execution_mode, execute_stage, synchronize_if_needed
from .session_transfer import add_scaled_tree, move_tensor_tree


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
        # Artifacts are reusable; deferred fallback and close mutate only this
        # session-owned payload copy.
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
        return actual_execution_mode(self._payload)

    def _ensure_open(self) -> None:
        if self._closed:
            raise BackendExecutionError("PyTorch backend session is closed")

    def _submit_impl(
        self,
        entrypoint: str,
        inputs: Any,
        context: ExecutionContext,
    ) -> Any:
        self._ensure_open()
        if context.cancelled:
            raise RequestCancelledError(f"request cancelled before entrypoint {entrypoint!r}")
        moved_inputs = move_tensor_tree(
            self._torch,
            inputs,
            device=self._torch_device,
            dtype=self._payload.dtype,
            non_blocking=self._payload.non_blocking,
        )
        use_cuda_graph = self._cuda_graphs is not None and self._cuda_graphs.handles(entrypoint)
        output = execute_stage(
            self._torch,
            self._torch_device,
            self._payload,
            self._cuda_graphs,
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
            # Static buffers, transfer, non-captured stages, and final sync all
            # share one session-local critical section.
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
        self._ensure_open()
        if context.cancelled:
            raise RequestCancelledError("request cancelled before backend state update")
        moved_state = move_tensor_tree(
            self._torch,
            state,
            device=self._torch_device,
            dtype=self._payload.dtype,
            non_blocking=self._payload.non_blocking,
        )
        moved_update = move_tensor_tree(
            self._torch,
            update,
            device=self._torch_device,
            dtype=self._payload.dtype,
            non_blocking=self._payload.non_blocking,
        )
        try:
            output = add_scaled_tree(self._torch, moved_state, moved_update, scale)
            synchronize_if_needed(self._torch, self._torch_device, self._payload)
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
            except Exception as error:  # noqa: BLE001
                failure = error
            try:
                if self._cuda_graphs is not None:
                    self._cuda_graphs.close()
            except Exception as error:  # noqa: BLE001
                if failure is None:
                    failure = error
            try:
                if is_cuda and self._payload.offload_module_on_close and module is not None:
                    module.to(device=self._torch.device("cpu"))
            except Exception as error:  # noqa: BLE001
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
                    except Exception as error:  # noqa: BLE001
                        if failure is None:
                            failure = error
                self._release_module()
            except Exception as error:  # noqa: BLE001
                if failure is None:
                    failure = error
        if failure is not None:
            raise BackendExecutionError(
                f"could not release PyTorch session on {self._payload.device_id}: "
                f"{type(failure).__name__}: {failure}"
            ) from failure


__all__ = ["TorchBackendSession"]
