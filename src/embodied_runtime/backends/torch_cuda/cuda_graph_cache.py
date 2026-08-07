"""Bounded session-local cache for captured CUDA Graph calls."""

from __future__ import annotations

from collections import OrderedDict

from .cuda_graph_capture import CapturedCall
from .cuda_graph_key import GraphKey


class CudaGraphCache:
    def __init__(self, max_graphs: int) -> None:
        self._max_graphs = max_graphs
        self._captured: dict[GraphKey, CapturedCall] = {}
        self._failed: OrderedDict[GraphKey, str] = OrderedDict()

    def __len__(self) -> int:
        return len(self._captured)

    @property
    def full(self) -> bool:
        return len(self._captured) >= self._max_graphs

    def get(self, key: GraphKey) -> CapturedCall | None:
        return self._captured.get(key)

    def put(self, key: GraphKey, captured: CapturedCall) -> None:
        self._captured[key] = captured

    def pop(self, key: GraphKey) -> CapturedCall | None:
        return self._captured.pop(key, None)

    def failure_reason(self, key: GraphKey) -> str | None:
        return self._failed.get(key)

    def remember_failure(self, key: GraphKey, reason: str) -> None:
        self._failed[key] = reason
        self._failed.move_to_end(key)
        while len(self._failed) > self._max_graphs:
            self._failed.popitem(last=False)

    def close(self) -> None:
        failure: Exception | None = None
        for captured in self._captured.values():
            try:
                captured.graph.reset()
            except Exception as error:  # noqa: BLE001
                if failure is None:
                    failure = error
        self._captured.clear()
        self._failed.clear()
        if failure is not None:
            raise failure
