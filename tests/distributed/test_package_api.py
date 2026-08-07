from __future__ import annotations

from embodied_runtime import distributed
from embodied_runtime.distributed import communication, failover, planning
from embodied_runtime.distributed.communication import cloud_session, openpi


def test_distributed_package_exports_keep_the_established_public_api() -> None:
    assert distributed.AsyncFailoverCoordinator is failover.AsyncFailoverCoordinator
    assert distributed.FailoverConfig is failover.FailoverConfig
    assert distributed.AsyncPlanCoordinator is planning.AsyncPlanCoordinator
    assert distributed.PlanManager is planning.PlanManager


def test_communication_package_exports_keep_the_established_public_api() -> None:
    assert communication.MultiTenantTcpEndpoint is cloud_session.MultiTenantTcpEndpoint
    assert communication.CloudSessionProtocolError is cloud_session.CloudSessionProtocolError
    assert communication.OpenPiWebSocketEndpoint is openpi.OpenPiWebSocketEndpoint
    assert communication.OpenPIProtocolError is openpi.OpenPIProtocolError
