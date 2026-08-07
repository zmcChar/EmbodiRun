"""CUDA Graph capture, input refresh, and replay operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cuda_graph_buffers import clone_output_tree, copy_into_static, make_static_tree


class CudaGraphWarmupError(RuntimeError):
    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


@dataclass(slots=True)
class CapturedCall:
    graph: Any
    static_inputs: Any
    static_outputs: Any


def capture_call(
    torch: Any,
    device: Any,
    stream: Any,
    function: Callable[[Any], Any],
    inputs: Any,
    *,
    warmup_steps: int,
) -> CapturedCall:
    static_inputs = make_static_tree(torch, inputs)
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(warmup_steps):
                    function(static_inputs)
            stream.synchronize()
        except Exception as error:
            raise CudaGraphWarmupError(error) from error

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                static_outputs = function(static_inputs)
            stream.synchronize()
        except Exception:
            graph.reset()
            raise
    return CapturedCall(
        graph=graph,
        static_inputs=static_inputs,
        static_outputs=static_outputs,
    )


def refresh_call_inputs(
    torch: Any,
    device: Any,
    stream: Any,
    captured: CapturedCall,
    inputs: Any,
) -> None:
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            copy_into_static(torch, captured.static_inputs, inputs)


def replay_call(
    torch: Any,
    device: Any,
    stream: Any,
    captured: CapturedCall,
) -> Any:
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            captured.graph.replay()
            return clone_output_tree(torch, captured.static_outputs)
