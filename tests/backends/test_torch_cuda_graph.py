"""CUDA Graph cache and orchestration tests without a CUDA runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodied_runtime.backends.torch_cuda import cuda_graph as cuda_graph_module
from embodied_runtime.backends.torch_cuda.cuda_graph import (
    CudaGraphConfig,
    CudaGraphExecutor,
    CudaGraphInputError,
)
from embodied_runtime.backends.torch_cuda.cuda_graph_cache import CudaGraphCache
from embodied_runtime.backends.torch_cuda.cuda_graph_capture import CapturedCall
from embodied_runtime.backends.torch_cuda.cuda_graph_key import (
    CudaGraphInputError as FocusedCudaGraphInputError,
)


class _FakeGraph:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_established_graph_error_import_uses_focused_value() -> None:
    assert CudaGraphInputError is FocusedCudaGraphInputError


def test_cuda_graph_cache_bounds_failures_and_releases_captures() -> None:
    cache = CudaGraphCache(max_graphs=1)
    graph = _FakeGraph()
    captured = CapturedCall(graph=graph, static_inputs={}, static_outputs={})
    key = ("forward", ("constant", "builtins", "int", "1"))
    other_key = ("forward", ("constant", "builtins", "int", "2"))

    cache.put(key, captured)
    cache.remember_failure(key, "first")
    cache.remember_failure(other_key, "second")

    assert cache.full
    assert cache.get(key) is captured
    assert cache.failure_reason(key) is None
    assert cache.failure_reason(other_key) == "second"
    cache.close()
    assert graph.reset_count == 1
    assert len(cache) == 0


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeStream:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


class _FakeCuda:
    def __init__(self) -> None:
        self.stream = _FakeStream()

    @staticmethod
    def device(device):
        del device
        return _Context()

    def Stream(self, *, device):
        del device
        return self.stream


def test_cuda_graph_executor_counts_capture_replay_and_closes_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = SimpleNamespace(cuda=_FakeCuda())
    graph = _FakeGraph()
    captured = CapturedCall(graph=graph, static_inputs={}, static_outputs={})
    captures: list[object] = []
    refreshes: list[object] = []

    monkeypatch.setattr(
        cuda_graph_module,
        "prepare_graph_inputs",
        lambda torch, device, entrypoint, inputs, names: inputs,
    )
    monkeypatch.setattr(cuda_graph_module, "tree_signature", lambda torch, inputs: ("shape",))

    def capture(*args, **kwargs):
        captures.append((args, kwargs))
        return captured

    monkeypatch.setattr(cuda_graph_module, "capture_call", capture)
    monkeypatch.setattr(
        cuda_graph_module,
        "refresh_call_inputs",
        lambda *args: refreshes.append(args),
    )
    monkeypatch.setattr(cuda_graph_module, "replay_call", lambda *args: "output")

    executor = CudaGraphExecutor(
        torch,
        "cuda:0",
        CudaGraphConfig(entrypoints=frozenset({"forward"})),
    )

    assert executor.execute("forward", {"x": 1}, lambda inputs: inputs) == "output"
    assert executor.execute("forward", {"x": 2}, lambda inputs: inputs) == "output"
    assert len(captures) == 1
    assert len(refreshes) == 1
    assert executor.stats == {
        "captures": 1,
        "replays": 1,
        "fallbacks": 0,
        "cached_graphs": 1,
    }
    executor.close()
    assert torch.cuda.stream.synchronize_count == 1
    assert graph.reset_count == 1
