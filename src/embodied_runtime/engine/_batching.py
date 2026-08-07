"""Priority admission queue and compatibility-aware dynamic batch collection."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable, Sequence

from embodied_runtime.types import TensorTree

from ._admission import WorkItem
from .config import EngineConfig
from .request import InferenceRequest

Batcher = Callable[[Sequence[TensorTree]], TensorTree]
Splitter = Callable[[TensorTree, int], Sequence[TensorTree]]
QueueEntry = tuple[int, int, WorkItem]


def batch_key(request: InferenceRequest) -> tuple[int | None, str | None]:
    """Return execution controls that must match for requests to share a batch."""

    seed_isolation = request.request_id if request.seed is not None else None
    return request.num_steps, seed_isolation


class PriorityBatchQueue:
    """Bounded priority queue that preserves exact entries while batching."""

    def __init__(self, config: EngineConfig, *, batching_enabled: bool) -> None:
        self._max_batch_size = config.max_batch_size
        self._max_wait_s = config.max_wait_ms / 1000.0
        self._batching_enabled = batching_enabled and config.max_batch_size > 1
        self._queue: asyncio.PriorityQueue[QueueEntry] = asyncio.PriorityQueue(
            maxsize=config.max_queue_size
        )
        self._sequence = itertools.count()

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def put(self, item: WorkItem) -> None:
        request = item.request
        entry: QueueEntry = (-request.priority, next(self._sequence), item)
        self._queue.put_nowait(entry)

    async def next_batch(self) -> list[QueueEntry]:
        first = await self._queue.get()
        if not self._batching_enabled:
            return [first]

        entries = [first]
        compatible_key = batch_key(first[2].request)
        deadline = asyncio.get_running_loop().time() + self._max_wait_s
        while len(entries) < self._max_batch_size:
            try:
                if self._max_wait_s == 0:
                    candidate = self._queue.get_nowait()
                else:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    candidate = await asyncio.wait_for(self._queue.get(), remaining)
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                break

            if batch_key(candidate[2].request) != compatible_key:
                # The candidate was acquired, so balance unfinished_tasks before
                # returning the exact priority entry to the queue.
                self._queue.task_done()
                self._queue.put_nowait(candidate)
                break
            entries.append(candidate)
        return entries

    def finish(self, entries: Sequence[QueueEntry]) -> None:
        for _ in entries:
            self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


__all__ = [
    "Batcher",
    "PriorityBatchQueue",
    "QueueEntry",
    "Splitter",
    "batch_key",
]
