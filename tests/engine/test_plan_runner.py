from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from embodied_runtime.contracts import ExecutionPlan, ModelPackage, ModelSpec, TensorTree
from embodied_runtime.engine import (
    ExecutionEngine,
    PlanRunnerRegistry,
    RunnerHost,
    RunnerRequest,
    UnsupportedExecutionPlanError,
)

from .fakes import FakeSession


@dataclass(frozen=True)
class _UnknownPlan:
    kind: str = "unknown"

    def required_entrypoints(self) -> tuple[str, ...]:
        return ("unknown_stage",)


@dataclass(frozen=True)
class _CustomForwardPlan:
    kind: str = "custom_forward"

    def required_entrypoints(self) -> tuple[str, ...]:
        return ("forward",)


class _CustomForwardRunner:
    def __init__(self, plan: ExecutionPlan, host: RunnerHost) -> None:
        self.plan = plan
        self.host = host

    def run_sync(self, request: RunnerRequest) -> TensorTree:
        return self.host.submit_sync(
            "forward",
            self.host.preprocessed_payload(request),
            request,
            None,
            None,
        )

    async def run_async(
        self,
        requests: Sequence[RunnerRequest],
        payload: TensorTree,
    ) -> TensorTree:
        return await self.host.submit_async("forward", payload, requests, None, None)


def test_plan_runner_registry_rejects_duplicate_kind() -> None:
    registry = PlanRunnerRegistry()

    def factory(plan, engine):
        return object()

    registry.register("custom", factory)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom", factory)

    assert registry.kinds() == ("custom",)


def test_engine_rejects_plan_without_registered_runner() -> None:
    package = ModelPackage(
        spec=ModelSpec(model_id="unknown", family="test"),
        entrypoints={"unknown_stage": lambda value: value},
        plan=_UnknownPlan(),
        package_id="engine-fake-flow-package",
    )

    with pytest.raises(UnsupportedExecutionPlanError, match="'unknown'"):
        ExecutionEngine(package, FakeSession())


def test_custom_runner_receives_narrow_host_facade_not_engine() -> None:
    registry = PlanRunnerRegistry()
    captured_hosts: list[RunnerHost] = []

    def factory(plan: ExecutionPlan, host: RunnerHost) -> _CustomForwardRunner:
        captured_hosts.append(host)
        return _CustomForwardRunner(plan, host)

    registry.register("custom_forward", factory)
    package = ModelPackage(
        spec=ModelSpec(model_id="custom", family="test"),
        entrypoints={"forward": lambda value: value},
        plan=_CustomForwardPlan(),
        package_id="engine-fake-flow-package",
    )
    engine = ExecutionEngine(package, FakeSession(), runner_registry=registry)

    result = engine.infer(4.0)

    assert result.output == 4.0
    assert len(captured_hosts) == 1
    assert isinstance(captured_hosts[0], RunnerHost)
    assert captured_hosts[0] is not engine
