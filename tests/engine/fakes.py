from __future__ import annotations

import time
from threading import Event, Lock
from typing import Any

from embodied_runtime.contracts import (
    DeviceInfo,
    EntrypointSpec,
    ExecutionContext,
    IterativeFlowPlan,
    MemoryStats,
    ModelPackage,
    ModelSpec,
    RequestCancelledError,
    SingleForwardPlan,
)

FAKE_PACKAGE_ID = "engine-fake-flow-package"
FAKE_FORWARD_PACKAGE_ID = "engine-fake-forward-package"


def make_package(
    *,
    num_steps: int = 4,
    step_returns_state: bool = False,
) -> ModelPackage:
    """Package whose callables fail if the engine bypasses BackendSession."""

    def must_not_be_called(value):
        raise AssertionError("engine called a ModelPackage entrypoint directly")

    plan = IterativeFlowPlan(
        default_num_steps=num_steps,
        step_returns_state=step_returns_state,
    )
    names = plan.required_entrypoints()
    return ModelPackage(
        spec=ModelSpec(
            model_id="fake-flow",
            family="test",
            action_dim=1,
            action_horizon=1,
        ),
        entrypoints={name: must_not_be_called for name in names},
        plan=plan,
        entrypoint_specs={
            name: EntrypointSpec(
                name=name,
                safe_point_after=name == plan.step,
            )
            for name in names
        },
        package_id=FAKE_PACKAGE_ID,
    )


def make_forward_package() -> ModelPackage:
    """Single-stage package used to prove the engine is not flow-specific."""

    def must_not_be_called(value):
        raise AssertionError("engine called a ModelPackage entrypoint directly")

    plan = SingleForwardPlan()
    return ModelPackage(
        spec=ModelSpec(model_id="fake-forward", family="test"),
        entrypoints={plan.forward: must_not_be_called},
        plan=plan,
        entrypoint_specs={plan.forward: EntrypointSpec(plan.forward)},
        package_id=FAKE_FORWARD_PACKAGE_ID,
    )


class FakeSession:
    """CPU-only contract fake; it does not import a model or backend package."""

    def __init__(
        self,
        *,
        memory_stats: MemoryStats | None = None,
        step_delay_s: float = 0.0,
        block_first_encode: bool = False,
        step_returns_state: bool = False,
        honor_context_cancellation: bool = False,
        package_id: str = FAKE_PACKAGE_ID,
    ) -> None:
        self._package_id = package_id
        self._device = DeviceInfo(
            backend="fake",
            device_id="cpu:0",
            kind="test",
            vendor="test",
            name="contract fake",
            total_memory_bytes=1_000_000,
        )
        self._memory_stats = memory_stats or MemoryStats(
            allocated_bytes=10,
            reserved_bytes=20,
            total_bytes=1_000_000,
            free_bytes=999_980,
        )
        self.step_delay_s = step_delay_s
        self.block_first_encode = block_first_encode
        self.step_returns_state = step_returns_state
        self.honor_context_cancellation = honor_context_cancellation
        self.first_encode_started = Event()
        self.release_first_encode = Event()
        self.step_started = Event()
        self.closed = False
        self.calls: list[tuple[str, tuple[str, ...], Any]] = []
        self._lock = Lock()
        self._encode_count = 0

    @property
    def package_id(self) -> str:
        return self._package_id

    @property
    def device(self) -> DeviceInfo:
        return self._device

    def submit(
        self,
        entrypoint: str,
        inputs: Any,
        context: ExecutionContext,
    ) -> Any:
        with self._lock:
            self.calls.append((entrypoint, context.request_ids, inputs))

        if entrypoint == "encode_prefix":
            with self._lock:
                self._encode_count += 1
                should_block = self.block_first_encode and self._encode_count == 1
            if should_block:
                self.first_encode_started.set()
                if not self.release_first_encode.wait(timeout=5):
                    raise TimeoutError("test did not release the first encode")
            return inputs

        if entrypoint == "init_state":
            size = inputs["batch_size"]
            return 0.0 if size == 1 else [0.0] * size

        if entrypoint == "denoise_step":
            self.step_started.set()
            if self.step_delay_s:
                time.sleep(self.step_delay_s)
            if self.honor_context_cancellation and context.cancelled:
                raise RequestCancelledError("fake backend observed cancellation")
            if self.step_returns_state:
                return {"state": _add(inputs["state"], 1.0)}
            # The plan integrates from 1 to 0, so velocity=-prefix yields
            # final state=prefix after the complete Euler loop.
            return {"velocity": _negate(inputs["prefix"])}

        if entrypoint == "finalize":
            return {"actions": inputs["state"]}

        if entrypoint == "forward":
            return inputs

        raise KeyError(entrypoint)

    def add_scaled(
        self,
        state: Any,
        update: Any,
        scale: float,
        context: ExecutionContext,
    ) -> Any:
        with self._lock:
            self.calls.append(("advance_state", context.request_ids, (state, update, scale)))
        if self.honor_context_cancellation and context.cancelled:
            raise RequestCancelledError("fake backend observed cancellation")
        return _add_scaled(state, update, scale)

    def memory_stats(self) -> MemoryStats:
        return self._memory_stats

    def close(self) -> None:
        self.closed = True


def _negate(value: Any) -> Any:
    if isinstance(value, list):
        return [_negate(item) for item in value]
    if isinstance(value, dict):
        return {key: _negate(item) for key, item in value.items()}
    return -value


def _add(value: Any, increment: float) -> Any:
    if isinstance(value, list):
        return [_add(item, increment) for item in value]
    return value + increment


def _add_scaled(state: Any, update: Any, scale: float) -> Any:
    if isinstance(state, list):
        return [_add_scaled(left, right, scale) for left, right in zip(state, update, strict=True)]
    if isinstance(state, dict):
        return {key: _add_scaled(state[key], update[key], scale) for key in state}
    return state + scale * update
