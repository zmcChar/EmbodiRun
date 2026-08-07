from __future__ import annotations

import asyncio
from functools import wraps

import pytest

from embodied_runtime.distributed import RobotSessionIdentity
from embodied_runtime.distributed.multitenant import (
    MultiTenantInferenceService,
    SessionOverloadedError,
    SessionRegistrationError,
    SessionSequenceError,
)
from embodied_runtime.engine import (
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
)
from embodied_runtime.models.request import RawRequest
from embodied_runtime.models.spec import ModelSpec

_ACTION_SPACE_ID = "toy-vector-actions-v1"
_EMBODIMENT = "toy-vector"


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class _EchoProvider:
    def __init__(
        self,
        *,
        wait_for_concurrency: int = 0,
        multi_tenant_safe: bool = True,
    ) -> None:
        self._capabilities = ProviderCapabilities(
            name="cloud",
            runtime="test",
            model=ModelSpec(
                model_id="cloud-model",
                family="test",
                action_dim=2,
                action_horizon=1,
            ),
            is_remote=False,
            features=(frozenset({"multi_tenant_safe"}) if multi_tenant_safe else frozenset()),
        )
        self.requests: list[InferenceRequest] = []
        self.active = 0
        self.max_active = 0
        self.wait_for_concurrency = wait_for_concurrency
        self.concurrent = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active >= self.wait_for_concurrency > 0:
            self.concurrent.set()
        try:
            if self.wait_for_concurrency:
                await self.release.wait()
            raw = request.payload
            assert isinstance(raw, RawRequest)
            return InferenceResult(
                request_id=request.request_id,
                output={"actions": raw.observation["target"]},
                metadata={**request.metadata, "model_id": "cloud-model"},
            )
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True
        self.release.set()


def _identity(
    name: str,
    *,
    session_id: str | None = None,
    embodiment: str = _EMBODIMENT,
    action_space_id: str = _ACTION_SPACE_ID,
) -> RobotSessionIdentity:
    return RobotSessionIdentity(
        robot_id=f"robot-{name}",
        edge_node_id=f"edge-{name}",
        session_id=session_id or f"session-{name}",
        embodiment=embodiment,
        action_space_id=action_space_id,
    )


def _registration_metadata(provider: _EchoProvider) -> dict[str, object]:
    return {
        "edge_action_dim": provider.capabilities.model.action_dim,
        "edge_action_horizon": provider.capabilities.model.action_horizon,
    }


def _service(
    provider: _EchoProvider,
    **options,
) -> MultiTenantInferenceService:
    return MultiTenantInferenceService(
        provider,
        action_space_id=_ACTION_SPACE_ID,
        supported_embodiments=(_EMBODIMENT,),
        **options,
    )


def _request(request_id: str, target: list[float]) -> InferenceRequest:
    return InferenceRequest(
        payload=RawRequest(
            observation={"target": target},
            metadata={"robot_id": "spoofed-robot"},
        ),
        request_id=request_id,
        metadata={"sequence_id": 99},
    )


@async_test
async def test_service_namespaces_ids_and_runs_sessions_concurrently() -> None:
    provider = _EchoProvider(wait_for_concurrency=2)
    service = _service(provider)
    robot_a = _identity("a")
    robot_b = _identity("b")
    registration = _registration_metadata(provider)
    service.register(robot_a, metadata=registration)
    service.register(robot_b, metadata=registration)

    task_a = asyncio.create_task(
        service.infer_async(
            robot_a,
            _request("same-request", [1.0, 2.0]),
            sequence_id=1,
            observation_id="a-observation-1",
            observation_timestamp_s=1.0,
        )
    )
    task_b = asyncio.create_task(
        service.infer_async(
            robot_b,
            _request("same-request", [10.0, 20.0]),
            sequence_id=1,
            observation_id="b-observation-1",
            observation_timestamp_s=1.0,
        )
    )
    await asyncio.wait_for(provider.concurrent.wait(), timeout=1.0)
    provider.release.set()
    result_a, result_b = await asyncio.gather(task_a, task_b)

    assert provider.max_active == 2
    assert result_a.request_id == result_b.request_id == "same-request"
    assert result_a.output == {"actions": [1.0, 2.0]}
    assert result_b.output == {"actions": [10.0, 20.0]}
    assert result_a.metadata["robot_id"] == "robot-a"
    assert result_b.metadata["robot_id"] == "robot-b"
    assert provider.requests[0].request_id != provider.requests[1].request_id
    observed_raw_metadata = {
        request.metadata["robot_id"]: request.payload.metadata["robot_id"]
        for request in provider.requests
    }
    assert observed_raw_metadata == {"robot-a": "robot-a", "robot-b": "robot-b"}

    with pytest.raises(SessionSequenceError, match="expected greater than 1"):
        await service.infer_async(
            robot_a,
            _request("duplicate", [0.0, 0.0]),
            sequence_id=1,
            observation_id="a-observation-duplicate",
            observation_timestamp_s=2.0,
        )
    with pytest.raises(SessionRegistrationError, match="already bound"):
        service.register(
            _identity("other", session_id=robot_a.session_id),
            metadata=registration,
        )

    await service.aclose()
    assert provider.closed


def test_service_requires_explicit_multi_tenant_safety() -> None:
    with pytest.raises(ValueError, match="multi_tenant_safe"):
        _service(_EchoProvider(multi_tenant_safe=False))


def test_registration_rejects_action_and_embodiment_contract_mismatches() -> None:
    provider = _EchoProvider()
    service = _service(provider)
    registration = _registration_metadata(provider)

    with pytest.raises(SessionRegistrationError, match="action_space_id"):
        service.register(
            _identity("wrong-action", action_space_id="other-actions-v1"),
            metadata=registration,
        )
    with pytest.raises(SessionRegistrationError, match="unsupported"):
        service.register(
            _identity("wrong-body", embodiment="other-robot"),
            metadata=registration,
        )
    with pytest.raises(SessionRegistrationError, match="edge_action_dim"):
        service.register(
            _identity("wrong-dim"),
            metadata={**registration, "edge_action_dim": 99},
        )


@async_test
async def test_same_session_overload_and_unregister_drain() -> None:
    provider = _EchoProvider(wait_for_concurrency=1)
    service = _service(provider, max_pending_per_session=1)
    identity = _identity("overload")
    service.register(identity, metadata=_registration_metadata(provider))

    active = asyncio.create_task(
        service.infer_async(
            identity,
            _request("active", [1.0, 2.0]),
            sequence_id=1,
            observation_id="active-observation",
            observation_timestamp_s=1.0,
        )
    )
    await asyncio.wait_for(provider.concurrent.wait(), timeout=1.0)

    with pytest.raises(SessionOverloadedError, match="pending request"):
        await service.infer_async(
            identity,
            _request("overloaded", [3.0, 4.0]),
            sequence_id=2,
            observation_id="overloaded-observation",
            observation_timestamp_s=2.0,
        )

    service.unregister(identity)
    assert service.registered_session_count == 1
    with pytest.raises(SessionRegistrationError, match="unregistering"):
        await service.infer_async(
            identity,
            _request("after-unregister", [5.0, 6.0]),
            sequence_id=2,
            observation_id="after-unregister-observation",
            observation_timestamp_s=3.0,
        )

    provider.release.set()
    assert (await active).output == {"actions": [1.0, 2.0]}
    assert service.registered_session_count == 0
    await service.aclose()


@async_test
async def test_service_close_waits_for_active_inference() -> None:
    provider = _EchoProvider(wait_for_concurrency=1)
    service = _service(provider)
    identity = _identity("close-drain")
    service.register(identity, metadata=_registration_metadata(provider))
    inference = asyncio.create_task(
        service.infer_async(
            identity,
            _request("active-close", [1.0, 2.0]),
            sequence_id=1,
            observation_id="active-close-observation",
            observation_timestamp_s=1.0,
        )
    )
    await asyncio.wait_for(provider.concurrent.wait(), timeout=1.0)

    close = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    assert not close.done()
    assert not provider.closed

    provider.release.set()
    await asyncio.gather(inference, close)
    assert provider.closed


def test_idle_session_ttl_reclaims_capacity() -> None:
    now = [10.0]
    provider = _EchoProvider()
    service = _service(
        provider,
        max_sessions=1,
        session_idle_ttl_s=5.0,
        clock=lambda: now[0],
    )
    first = _identity("ttl-first")
    second = _identity("ttl-second")
    registration = _registration_metadata(provider)
    service.register(first, metadata=registration)

    now[0] = 15.0
    assert service.registered_session_count == 1
    with pytest.raises(SessionRegistrationError, match="capacity"):
        service.register(second, metadata=registration)

    now[0] = 15.001
    service.register(second, metadata=registration)
    snapshots = service.sessions()
    assert len(snapshots) == 1
    assert snapshots[0].identity == second
