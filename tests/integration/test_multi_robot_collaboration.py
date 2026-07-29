from __future__ import annotations

import asyncio
import socket
from functools import wraps

import pytest

from embodied_runtime.apps._local_provider import (
    LocalProviderConfig,
    build_local_provider,
)
from embodied_runtime.apps.multi_robot_cloud import (
    MultiRobotCloudConfig,
    serve_multi_robot_cloud,
)
from embodied_runtime.apps.multi_robot_edge import RobotEdgeRuntime
from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    ModelSpec,
    RawRequest,
)
from embodied_runtime.distributed import (
    FailoverConfig,
    FallbackReason,
    ResultSource,
    RobotSessionIdentity,
)
from embodied_runtime.distributed.communication import (
    CloudSessionProtocolError,
    CloudSessionRemoteError,
    MultiTenantTcpEndpoint,
    TcpJsonProtocolError,
    TcpJsonRequestClient,
)
from embodied_runtime.integrations.serving import ProviderCapabilities
from embodied_runtime.integrations.serving.multitenant import (
    MultiTenantInferenceService,
    SessionOverloadedError,
    SessionRegistrationError,
    SessionSequenceError,
)

_ACTION_SPACE_ID = "toy-vector-actions-v1"
_EMBODIMENT = "toy-vector"


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


@pytest.fixture
def free_tcp_port() -> int:
    """Keep this test independent of third-party pytest socket fixtures."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _EchoProvider:
    def __init__(
        self,
        name: str,
        *,
        wait_for_concurrency: int = 0,
        multi_tenant_safe: bool = False,
    ) -> None:
        self.name = name
        self._capabilities = ProviderCapabilities(
            name=name,
            runtime="test",
            model=ModelSpec(
                model_id=f"{name}-model",
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
                metadata={
                    **request.metadata,
                    "model_id": self.capabilities.model.model_id,
                },
            )
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True
        self.release.set()


class _ScriptedClient:
    def __init__(
        self,
        responses: list[dict[str, object] | BaseException],
    ) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    async def request(self, payload):
        self.requests.append(dict(payload))
        if not self.responses:
            raise AssertionError("scripted client received an unexpected request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _GateRegistrationClient:
    def __init__(self, identity: RobotSessionIdentity) -> None:
        self.identity = identity
        self.requests: list[dict[str, object]] = []
        self.register_started = asyncio.Event()
        self.release_register = asyncio.Event()

    async def request(self, payload):
        request = dict(payload)
        self.requests.append(request)
        if request["kind"] == "register":
            self.register_started.set()
            await self.release_register.wait()
            return _registered_response(self.identity)
        if request["kind"] == "unregister":
            return {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **self.identity.to_wire(),
            }
        raise AssertionError(f"unexpected request kind {request['kind']!r}")


class _BlockingEchoClient:
    def __init__(self, identity: RobotSessionIdentity) -> None:
        self.identity = identity
        self.requests: list[dict[str, object]] = []
        self.infer_started = asyncio.Event()
        self.release = asyncio.Event()

    async def request(self, payload):
        request = dict(payload)
        self.requests.append(request)
        kind = request["kind"]
        if kind == "register":
            return _registered_response(self.identity)
        if kind == "unregister":
            return {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **self.identity.to_wire(),
            }
        if kind != "infer":
            raise AssertionError(f"unexpected request kind {kind!r}")
        self.infer_started.set()
        await self.release.wait()
        observation = request["observation"]
        return _inference_response(
            self.identity,
            request_id=str(request["request_id"]),
            sequence_id=int(request["sequence_id"]),
            observation_id=str(request["observation_id"]),
            output={"actions": observation["target"]},
        )


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


def _registration_metadata(
    provider: _EchoProvider | object,
    **extra: object,
) -> dict[str, object]:
    capabilities = provider.capabilities
    return {
        "edge_action_dim": capabilities.model.action_dim,
        "edge_action_horizon": capabilities.model.action_horizon,
        **extra,
    }


def _service(
    provider,
    **options,
) -> MultiTenantInferenceService:
    return MultiTenantInferenceService(
        provider,
        action_space_id=_ACTION_SPACE_ID,
        supported_embodiments=(_EMBODIMENT,),
        **options,
    )


def _request(
    request_id: str,
    *,
    target: list[float],
    sequence_id: int,
    raw_metadata: dict[str, object] | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        payload=RawRequest(
            observation={"target": target},
            metadata=raw_metadata or {},
        ),
        request_id=request_id,
        metadata={"sequence_id": sequence_id},
    )


def _registered_response(
    identity: RobotSessionIdentity,
    *,
    last_sequence_id: int = 0,
) -> dict[str, object]:
    return {
        "ok": True,
        "kind": "registered",
        "protocol_version": 1,
        **identity.to_wire(),
        "action_space_id": identity.action_space_id,
        "supported_embodiments": [identity.embodiment],
        "action_dim": 2,
        "action_horizon": 1,
        "last_sequence_id": last_sequence_id,
    }


def _inference_response(
    identity: RobotSessionIdentity,
    *,
    request_id: str,
    sequence_id: int,
    observation_id: str,
    status: str = "succeeded",
    output: object = None,
) -> dict[str, object]:
    return {
        "ok": True,
        "kind": "inference_result",
        "protocol_version": 1,
        **identity.to_wire(),
        "request_id": request_id,
        "sequence_id": sequence_id,
        "observation_id": observation_id,
        "status": status,
        "output": {"actions": [[1.0, 2.0]]} if output is None else output,
        "queue_time_s": 0.0,
        "execution_time_s": 0.01,
        "metadata": {},
    }


@async_test
async def test_shared_service_namespaces_ids_and_runs_sessions_concurrently() -> None:
    provider = _EchoProvider(
        "cloud",
        wait_for_concurrency=2,
        multi_tenant_safe=True,
    )
    service = _service(provider)
    robot_a = _identity("a")
    robot_b = _identity("b")
    registration = _registration_metadata(
        provider,
        physical_resource_id="gpu-shared",
    )
    service.register(robot_a, metadata=registration)
    service.register(robot_b, metadata=registration)

    task_a = asyncio.create_task(
        service.infer_async(
            robot_a,
            _request(
                "same-request",
                target=[1.0, 2.0],
                sequence_id=1,
                raw_metadata={"robot_id": "spoofed-robot"},
            ),
            sequence_id=1,
            observation_id="a-observation-1",
            observation_timestamp_s=1.0,
        )
    )
    task_b = asyncio.create_task(
        service.infer_async(
            robot_b,
            _request("same-request", target=[10.0, 20.0], sequence_id=1),
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
    assert observed_raw_metadata == {
        "robot-a": "robot-a",
        "robot-b": "robot-b",
    }

    with pytest.raises(SessionSequenceError, match="expected greater than 1"):
        await service.infer_async(
            robot_a,
            _request("duplicate", target=[0.0, 0.0], sequence_id=1),
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


@async_test
async def test_shared_local_provider_batches_two_robot_observations() -> None:
    provider = build_local_provider(
        LocalProviderConfig(
            model="toy_single_forward",
            device="cpu",
            multi_tenant_safe=True,
            adapter_options={"action_horizon": 1, "action_dim": 2},
            max_batch_size=2,
            max_wait_ms=50.0,
        )
    )
    service = _service(provider)
    robot_a = _identity("batch-a")
    robot_b = _identity("batch-b")
    registration = _registration_metadata(provider)
    service.register(robot_a, metadata=registration)
    service.register(robot_b, metadata=registration)

    result_a, result_b = await asyncio.gather(
        service.infer_async(
            robot_a,
            _request("batch-a-1", target=[1.0, 2.0], sequence_id=1),
            sequence_id=1,
            observation_id="batch-a-observation-1",
            observation_timestamp_s=1.0,
        ),
        service.infer_async(
            robot_b,
            _request("batch-b-1", target=[10.0, 20.0], sequence_id=1),
            sequence_id=1,
            observation_id="batch-b-observation-1",
            observation_timestamp_s=1.0,
        ),
    )

    assert result_a.metadata["batch_size"] == 2
    assert result_b.metadata["batch_size"] == 2
    assert result_a.output["actions"].tolist() == [[1.0, 2.0]]
    assert result_b.output["actions"].tolist() == [[10.0, 20.0]]
    await service.aclose()


async def _connect_with_retry(endpoint: MultiTenantTcpEndpoint) -> None:
    for _ in range(100):
        try:
            await endpoint.connect()
            return
        except (ConnectionError, OSError):
            await asyncio.sleep(0.01)
    pytest.fail("multi-robot cloud server did not become ready")


@async_test
async def test_two_edge_runtimes_keep_sessions_and_failover_independent(
    free_tcp_port: int,
) -> None:
    cloud_provider = _EchoProvider(
        "shared-cloud",
        multi_tenant_safe=True,
    )
    service = _service(cloud_provider)
    cloud_config = MultiRobotCloudConfig(
        provider=LocalProviderConfig(),
        host="127.0.0.1",
        port=free_tcp_port,
        request_timeout_s=2.0,
        shutdown_grace_s=0.1,
    )
    server_task = asyncio.create_task(serve_multi_robot_cloud(service, cloud_config))

    identity_a = _identity("a")
    identity_b = _identity("b")
    endpoint_a = MultiTenantTcpEndpoint(
        TcpJsonRequestClient("127.0.0.1", free_tcp_port, timeout_s=2.0),
        identity_a,
        registration_metadata=_registration_metadata(
            cloud_provider,
            physical_resource_id="gpu-5080-0",
        ),
    )
    endpoint_b = MultiTenantTcpEndpoint(
        TcpJsonRequestClient("127.0.0.1", free_tcp_port, timeout_s=2.0),
        identity_b,
        registration_metadata=_registration_metadata(
            cloud_provider,
            physical_resource_id="gpu-5080-0",
        ),
    )
    await asyncio.gather(
        _connect_with_retry(endpoint_a),
        _connect_with_retry(endpoint_b),
    )
    with pytest.raises(ValueError, match="cannot override bound session_id"):
        await endpoint_a.infer_async(
            InferenceRequest(
                payload=RawRequest(
                    observation={"target": [0.0, 0.0]},
                    metadata={"session_id": identity_b.session_id},
                ),
                request_id="spoofed-request",
                metadata={"sequence_id": 1},
            )
        )

    edge_a = _EchoProvider("edge-a")
    edge_b = _EchoProvider("edge-b")
    runtime_a = RobotEdgeRuntime(
        identity=identity_a,
        edge=edge_a,
        cloud=endpoint_a,
        failover=FailoverConfig(max_cloud_sequence_lag=2),
    )
    runtime_b = RobotEdgeRuntime(
        identity=identity_b,
        edge=edge_b,
        cloud=endpoint_b,
        failover=FailoverConfig(max_cloud_sequence_lag=2),
    )

    try:
        first_a, first_b = await asyncio.gather(
            runtime_a.infer_observation({"target": [1.0, 2.0]}, sequence_id=1),
            runtime_b.infer_observation(
                {"target": [100.0, 200.0]},
                sequence_id=1,
            ),
        )
        assert first_a.source is ResultSource.EDGE
        assert first_b.source is ResultSource.EDGE
        await asyncio.gather(
            runtime_a.wait_for_cloud_idle(),
            runtime_b.wait_for_cloud_idle(),
        )

        second_a, second_b = await asyncio.gather(
            runtime_a.infer_observation({"target": [3.0, 4.0]}, sequence_id=2),
            runtime_b.infer_observation(
                {"target": [300.0, 400.0]},
                sequence_id=2,
            ),
        )
        assert second_a.source is ResultSource.CLOUD
        assert second_b.source is ResultSource.CLOUD
        assert second_a.result.output == {"actions": [1.0, 2.0]}
        assert second_b.result.output == {"actions": [100.0, 200.0]}
        assert second_a.result.metadata["robot_id"] == identity_a.robot_id
        assert second_b.result.metadata["robot_id"] == identity_b.robot_id

        runtime_a.set_cloud_enabled(False)
        disconnected_a, healthy_b = await asyncio.gather(
            runtime_a.infer_observation({"target": [5.0, 6.0]}, sequence_id=3),
            runtime_b.infer_observation(
                {"target": [500.0, 600.0]},
                sequence_id=3,
            ),
        )
        assert disconnected_a.source is ResultSource.EDGE
        assert disconnected_a.fallback_reason is FallbackReason.CLOUD_DISCONNECTED
        assert healthy_b.source is ResultSource.CLOUD
        assert endpoint_b.connected
    finally:
        await asyncio.gather(runtime_a.aclose(), runtime_b.aclose())
        assert service.registered_session_count == 0
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task

    assert edge_a.closed
    assert edge_b.closed
    assert cloud_provider.closed


def test_shared_service_requires_explicit_multi_tenant_safety() -> None:
    provider = _EchoProvider("unsafe-default")

    with pytest.raises(ValueError, match="multi_tenant_safe"):
        _service(provider)

    default_cloud = MultiRobotCloudConfig()
    assert default_cloud.provider.max_batch_size == 1
    assert default_cloud.provider.multi_tenant_safe


def test_registration_rejects_action_and_embodiment_contract_mismatches() -> None:
    provider = _EchoProvider("contract-cloud", multi_tenant_safe=True)
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
            metadata={
                **registration,
                "edge_action_dim": 99,
            },
        )


@async_test
async def test_same_session_overload_and_unregister_drain() -> None:
    provider = _EchoProvider(
        "blocking-cloud",
        wait_for_concurrency=1,
        multi_tenant_safe=True,
    )
    service = _service(provider, max_pending_per_session=1)
    identity = _identity("overload")
    service.register(identity, metadata=_registration_metadata(provider))

    active = asyncio.create_task(
        service.infer_async(
            identity,
            _request("active", target=[1.0, 2.0], sequence_id=1),
            sequence_id=1,
            observation_id="active-observation",
            observation_timestamp_s=1.0,
        )
    )
    await asyncio.wait_for(provider.concurrent.wait(), timeout=1.0)

    with pytest.raises(SessionOverloadedError, match="pending request"):
        await service.infer_async(
            identity,
            _request("overloaded", target=[3.0, 4.0], sequence_id=2),
            sequence_id=2,
            observation_id="overloaded-observation",
            observation_timestamp_s=2.0,
        )

    service.unregister(identity)
    assert service.registered_session_count == 1
    with pytest.raises(SessionRegistrationError, match="unregistering"):
        await service.infer_async(
            identity,
            _request("after-unregister", target=[5.0, 6.0], sequence_id=2),
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
    provider = _EchoProvider(
        "close-drain-cloud",
        wait_for_concurrency=1,
        multi_tenant_safe=True,
    )
    service = _service(provider)
    identity = _identity("close-drain")
    service.register(identity, metadata=_registration_metadata(provider))
    inference = asyncio.create_task(
        service.infer_async(
            identity,
            _request("active-close", target=[1.0, 2.0], sequence_id=1),
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
    provider = _EchoProvider("ttl-cloud", multi_tenant_safe=True)
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


@async_test
async def test_edge_runtime_requires_contiguous_sequence() -> None:
    identity = _identity("strict-sequence")
    endpoint = MultiTenantTcpEndpoint(
        _ScriptedClient([]),
        identity,
    )
    endpoint.set_enabled(False)
    edge = _EchoProvider("strict-edge")
    runtime = RobotEdgeRuntime(
        identity=identity,
        edge=edge,
        cloud=endpoint,
        failover=FailoverConfig(),
    )
    try:
        first = await runtime.infer_observation(
            {"target": [1.0, 2.0]},
            sequence_id=1,
        )
        assert first.source is ResultSource.EDGE
        assert first.result.metadata["sequence_id"] == 1
        with pytest.raises(ValueError, match="contiguous"):
            await runtime.infer_observation(
                {"target": [3.0, 4.0]},
                sequence_id=3,
            )
    finally:
        await runtime.aclose()


@async_test
async def test_edge_runtime_serializes_concurrent_control_ticks() -> None:
    identity = _identity("serialized-ticks")
    endpoint = MultiTenantTcpEndpoint(_ScriptedClient([]), identity)
    endpoint.set_enabled(False)
    edge = _EchoProvider("serialized-edge", wait_for_concurrency=1)
    runtime = RobotEdgeRuntime(
        identity=identity,
        edge=edge,
        cloud=endpoint,
        failover=FailoverConfig(),
    )
    first = asyncio.create_task(
        runtime.infer_observation(
            {"target": [1.0, 2.0]},
            sequence_id=1,
        )
    )
    await asyncio.wait_for(edge.concurrent.wait(), timeout=1.0)
    second = asyncio.create_task(
        runtime.infer_observation(
            {"target": [3.0, 4.0]},
            sequence_id=2,
        )
    )
    await asyncio.sleep(0)
    assert len(edge.requests) == 1

    edge.release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert first_result.sequence_id == 1
    assert second_result.sequence_id == 2
    assert [request.metadata["sequence_id"] for request in edge.requests] == [1, 2]
    await runtime.aclose()


@async_test
async def test_edge_runtime_close_drains_active_control_tick() -> None:
    identity = _identity("runtime-close")
    endpoint = MultiTenantTcpEndpoint(_ScriptedClient([]), identity)
    endpoint.set_enabled(False)
    edge = _EchoProvider("runtime-close-edge", wait_for_concurrency=1)
    runtime = RobotEdgeRuntime(
        identity=identity,
        edge=edge,
        cloud=endpoint,
        failover=FailoverConfig(),
    )
    inference = asyncio.create_task(
        runtime.infer_observation(
            {"target": [1.0, 2.0]},
            sequence_id=1,
        )
    )
    await asyncio.wait_for(edge.concurrent.wait(), timeout=1.0)
    closing = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    assert not edge.closed

    edge.release.set()
    await asyncio.gather(inference, closing)
    assert edge.closed


@async_test
async def test_protocol_error_disconnects_and_advances_epoch() -> None:
    identity = _identity("protocol")
    client = _ScriptedClient(
        [
            _registered_response(identity),
            _inference_response(
                identity,
                request_id="wrong-request",
                sequence_id=1,
                observation_id="observation-1",
            ),
        ]
    )
    endpoint = MultiTenantTcpEndpoint(client, identity)
    await endpoint.connect()
    connected_epoch = endpoint.connection_epoch

    with pytest.raises(CloudSessionProtocolError, match="request_id"):
        await endpoint.infer_async(_request("request-1", target=[1.0, 2.0], sequence_id=1))

    assert not endpoint.connected
    assert endpoint.connection_epoch == connected_epoch + 1
    await endpoint.aclose()


@async_test
async def test_transport_framing_error_disconnects_and_advances_epoch() -> None:
    identity = _identity("framing")
    client = _ScriptedClient(
        [
            _registered_response(identity),
            TcpJsonProtocolError("malformed cloud frame"),
            {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **identity.to_wire(),
            },
        ]
    )
    endpoint = MultiTenantTcpEndpoint(client, identity)
    await endpoint.connect()
    connected_epoch = endpoint.connection_epoch

    with pytest.raises(TcpJsonProtocolError, match="malformed"):
        await endpoint.infer_async(_request("request-1", target=[1.0, 2.0], sequence_id=1))

    assert not endpoint.connected
    assert endpoint.connection_epoch == connected_epoch + 1
    await endpoint.aclose()


@async_test
async def test_missing_cloud_status_is_a_protocol_error() -> None:
    identity = _identity("missing-status")
    response = _inference_response(
        identity,
        request_id="request-1",
        sequence_id=1,
        observation_id="request-1",
    )
    response.pop("status")
    client = _ScriptedClient(
        [
            _registered_response(identity),
            response,
            {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **identity.to_wire(),
            },
        ]
    )
    endpoint = MultiTenantTcpEndpoint(client, identity)
    await endpoint.connect()

    with pytest.raises(CloudSessionProtocolError, match="status"):
        await endpoint.infer_async(_request("request-1", target=[1.0, 2.0], sequence_id=1))

    assert not endpoint.connected
    await endpoint.aclose()


@async_test
async def test_failed_cloud_result_is_rejected_without_becoming_authority() -> None:
    identity = _identity("failed-result")
    client = _ScriptedClient(
        [
            _registered_response(identity),
            _inference_response(
                identity,
                request_id="request-1",
                sequence_id=1,
                observation_id="request-1",
                status="failed",
            ),
            {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **identity.to_wire(),
            },
        ]
    )
    endpoint = MultiTenantTcpEndpoint(client, identity)
    await endpoint.connect()

    with pytest.raises(CloudSessionRemoteError, match="CloudInferenceFailed"):
        await endpoint.infer_async(_request("request-1", target=[1.0, 2.0], sequence_id=1))

    assert endpoint.connected
    await endpoint.aclose()


@async_test
async def test_invalid_cloud_action_shape_disconnects_before_authority() -> None:
    identity = _identity("invalid-actions")
    client = _ScriptedClient(
        [
            _registered_response(identity),
            _inference_response(
                identity,
                request_id="request-1",
                sequence_id=1,
                observation_id="request-1",
                output={"actions": [[1.0, 2.0, 3.0]]},
            ),
            {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **identity.to_wire(),
            },
        ]
    )
    endpoint = MultiTenantTcpEndpoint(client, identity)
    await endpoint.connect()

    with pytest.raises(CloudSessionProtocolError, match="dimension"):
        await endpoint.infer_async(_request("request-1", target=[1.0, 2.0], sequence_id=1))

    assert not endpoint.connected
    await endpoint.aclose()


@async_test
async def test_registration_loss_can_reconnect_after_cloud_restart() -> None:
    identity = _identity("cloud-restart")
    client = _ScriptedClient(
        [
            _registered_response(identity),
            {
                "ok": False,
                "kind": "error",
                "protocol_version": 1,
                "error_type": "SessionRegistrationError",
                "error": "session was lost during cloud restart",
            },
            _registered_response(identity),
            {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **identity.to_wire(),
            },
        ]
    )
    endpoint = MultiTenantTcpEndpoint(client, identity)
    await endpoint.connect()

    with pytest.raises(CloudSessionRemoteError, match="SessionRegistrationError"):
        await endpoint.infer_async(_request("request-1", target=[1.0, 2.0], sequence_id=1))
    assert not endpoint.connected

    await endpoint.connect()
    assert endpoint.connected
    await endpoint.aclose()


@async_test
async def test_cloud_submission_owns_observation_snapshot() -> None:
    identity = _identity("observation-owner")
    client = _BlockingEchoClient(identity)
    endpoint = MultiTenantTcpEndpoint(client, identity)
    edge = _EchoProvider("observation-edge")
    runtime = RobotEdgeRuntime(
        identity=identity,
        edge=edge,
        cloud=endpoint,
        failover=FailoverConfig(),
    )
    target = [1.0, 2.0]
    try:
        await runtime.start()
        decision = await runtime.infer_observation(
            {"target": target},
            sequence_id=1,
        )
        assert decision.source is ResultSource.EDGE

        target[0] = 99.0
        await asyncio.wait_for(client.infer_started.wait(), timeout=1.0)
        client.release.set()
        await runtime.wait_for_cloud_idle()

        infer_request = next(request for request in client.requests if request["kind"] == "infer")
        assert infer_request["observation"]["target"] == [1.0, 2.0]
    finally:
        client.release.set()
        await runtime.aclose()


@async_test
async def test_initial_registration_timeout_never_blocks_local_edge() -> None:
    identity = _identity("slow-registration")
    client = _GateRegistrationClient(identity)
    endpoint = MultiTenantTcpEndpoint(client, identity)
    edge = _EchoProvider("slow-registration-edge")
    runtime = RobotEdgeRuntime(
        identity=identity,
        edge=edge,
        cloud=endpoint,
        failover=FailoverConfig(),
        registration_timeout_s=0.01,
        reconnect_interval_s=0.0,
    )
    try:
        await runtime.start()

        assert isinstance(runtime.last_reconnect_error, asyncio.TimeoutError)
        decision = await runtime.infer_observation(
            {"target": [1.0, 2.0]},
            sequence_id=1,
        )
        assert decision.source is ResultSource.EDGE
        assert decision.fallback_reason is FallbackReason.CLOUD_DISCONNECTED
    finally:
        client.release_register.set()
        await runtime.aclose()


@async_test
async def test_background_registration_and_close_use_control_plane_timeout() -> None:
    identity = _identity("slow-reconnect")
    client = _GateRegistrationClient(identity)
    endpoint = MultiTenantTcpEndpoint(client, identity)
    edge = _EchoProvider("slow-reconnect-edge")
    runtime = RobotEdgeRuntime(
        identity=identity,
        edge=edge,
        cloud=endpoint,
        failover=FailoverConfig(),
        registration_timeout_s=0.01,
        reconnect_interval_s=0.0,
    )
    try:
        decision = await runtime.infer_observation(
            {"target": [1.0, 2.0]},
            sequence_id=1,
        )
        assert decision.source is ResultSource.EDGE
        await asyncio.sleep(0.02)
        assert isinstance(runtime.last_reconnect_error, asyncio.TimeoutError)
    finally:
        await asyncio.wait_for(runtime.aclose(), timeout=0.5)


@async_test
async def test_disable_then_close_still_unregisters_session() -> None:
    identity = _identity("disable-close")
    client = _ScriptedClient(
        [
            _registered_response(identity),
            {
                "ok": True,
                "kind": "unregistered",
                "protocol_version": 1,
                **identity.to_wire(),
            },
        ]
    )
    endpoint = MultiTenantTcpEndpoint(client, identity)
    await endpoint.connect()
    endpoint.set_enabled(False)
    assert not endpoint.connected

    await endpoint.aclose()

    assert [request["kind"] for request in client.requests] == [
        "register",
        "unregister",
    ]


@async_test
async def test_close_serializes_with_in_progress_registration() -> None:
    identity = _identity("connect-close")
    client = _GateRegistrationClient(identity)
    endpoint = MultiTenantTcpEndpoint(client, identity)

    connecting = asyncio.create_task(endpoint.connect())
    await asyncio.wait_for(client.register_started.wait(), timeout=1.0)
    closing = asyncio.create_task(endpoint.aclose())
    await asyncio.sleep(0)
    assert not closing.done()

    client.release_register.set()
    await connecting
    await closing

    assert not endpoint.connected
    assert [request["kind"] for request in client.requests] == [
        "register",
        "unregister",
    ]
    with pytest.raises(RuntimeError, match="closed"):
        await endpoint.connect()
