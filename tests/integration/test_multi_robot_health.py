from __future__ import annotations

import asyncio
import json
import socket

import pytest

from embodied_runtime.apps import multi_robot_cloud
from embodied_runtime.apps.multi_robot_cloud import (
    MultiRobotCloudConfig,
    dispatch_multi_robot_request,
    probe_multi_robot_cloud_health,
    serve_multi_robot_cloud,
    validate_multi_robot_health_response,
)
from embodied_runtime.contracts import DeviceInfo, ModelSpec
from embodied_runtime.integrations.serving import ProviderCapabilities
from embodied_runtime.integrations.serving.multitenant import (
    MultiTenantInferenceService,
)


class _HealthProvider:
    def __init__(self) -> None:
        self._capabilities = ProviderCapabilities(
            name="shared-cloud",
            runtime="local_backend",
            model=ModelSpec(
                model_id="smolvla",
                family="vla",
                revision="test-revision",
                action_dim=6,
                action_horizon=50,
            ),
            is_remote=False,
            device=DeviceInfo(
                backend="torch_cuda",
                device_id="cpu",
                kind="cpu",
                vendor="generic",
                name="test CPU",
            ),
            features=frozenset({"multi_tenant_safe"}),
        )

    @property
    def capabilities(self):
        return self._capabilities

    async def infer_async(self, request):
        raise AssertionError("a health probe must not run inference")

    async def aclose(self) -> None:
        pass


def _health_service() -> MultiTenantInferenceService:
    return MultiTenantInferenceService(
        _HealthProvider(),
        action_space_id="smolvla-so100-actions-v1",
        supported_embodiments=("so100",),
    )


def test_ping_is_read_only_and_exposes_operational_contract() -> None:
    service = _health_service()

    response = asyncio.run(
        dispatch_multi_robot_request(
            service,
            {
                "kind": "ping",
                "protocol_version": 1,
            },
        )
    )
    health = validate_multi_robot_health_response(response)

    assert service.registered_session_count == 0
    assert health == {
        "healthy": True,
        "protocol_version": 1,
        "provider": {
            "name": "shared-cloud",
            "runtime": "local_backend",
            "backend": "torch_cuda",
            "device": "cpu",
        },
        "model": {
            "id": "smolvla",
            "family": "vla",
            "revision": "test-revision",
        },
        "action_contract": {
            "action_space_id": "smolvla-so100-actions-v1",
            "supported_embodiments": ["so100"],
            "action_dim": 6,
            "action_horizon": 50,
        },
        "registered_sessions": 0,
    }


def test_health_probe_uses_existing_tcp_json_protocol() -> None:
    async def scenario() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = int(listener.getsockname()[1])
        service = _health_service()
        server = asyncio.create_task(
            serve_multi_robot_cloud(
                service,
                MultiRobotCloudConfig(
                    host="127.0.0.1",
                    port=port,
                    request_timeout_s=1.0,
                    shutdown_grace_s=0.1,
                ),
            )
        )
        try:
            for attempt in range(100):
                try:
                    health = await probe_multi_robot_cloud_health(
                        "127.0.0.1",
                        port,
                        timeout_s=1.0,
                    )
                    break
                except (ConnectionError, OSError):
                    if attempt == 99:
                        raise
                    await asyncio.sleep(0.01)
            assert health["provider"]["name"] == "shared-cloud"
            assert health["registered_sessions"] == 0
        finally:
            server.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"kind": "registered"}, "kind"),
        ({"provider": ""}, "provider"),
        ({"supported_embodiments": []}, "supported_embodiments"),
        ({"registered_sessions": -1}, "registered_sessions"),
    ],
)
def test_health_response_rejects_invalid_service_declarations(
    update,
    message,
) -> None:
    response = {
        "ok": True,
        "kind": "pong",
        "protocol_version": 1,
        "provider": "shared-cloud",
        "provider_runtime": "local_backend",
        "model_id": "smolvla",
        "model_family": "vla",
        "model_revision": None,
        "action_space_id": "actions-v1",
        "supported_embodiments": ["so100"],
        "action_dim": 6,
        "action_horizon": 50,
        "backend": "torch_cuda",
        "device": "cpu",
        "registered_sessions": 0,
        **update,
    }

    with pytest.raises((TypeError, ValueError), match=message):
        validate_multi_robot_health_response(response)


def test_cloud_cli_overrides_checkpoint_and_vlm_path(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "cloud.toml"
    config_path.write_text(
        """
[provider]
model = "smolvla"
checkpoint = "/config/smolvla"
multi_tenant_safe = true

[provider.package_options]
vlm_base_path = "/config/smolvlm2"
stats_variant = "so100"
""",
        encoding="utf-8",
    )
    captured: list[MultiRobotCloudConfig] = []
    monkeypatch.setattr(
        multi_robot_cloud,
        "run_multi_robot_cloud",
        captured.append,
    )

    assert (
        multi_robot_cloud.main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                "/cli/smolvla",
                "--vlm-base-path",
                "/cli/smolvlm2",
            ]
        )
        == 0
    )

    provider = captured[0].provider
    assert provider.checkpoint == "/cli/smolvla"
    assert provider.package_options == {
        "vlm_base_path": "/cli/smolvlm2",
        "stats_variant": "so100",
    }


def test_health_cli_prints_json_and_returns_success(
    monkeypatch,
    capsys,
) -> None:
    async def healthy(*args, **kwargs):
        return {
            "healthy": True,
            "protocol_version": 1,
            "provider": {"name": "cloud"},
            "model": {"id": "smolvla"},
            "action_contract": {"action_space_id": "actions-v1"},
            "registered_sessions": 3,
        }

    monkeypatch.setattr(
        multi_robot_cloud,
        "probe_multi_robot_cloud_health",
        healthy,
    )

    assert multi_robot_cloud.health_main(["--host", "localhost"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["healthy"] is True
    assert output["registered_sessions"] == 3


def test_health_cli_returns_nonzero_for_unreachable_service(
    monkeypatch,
    capsys,
) -> None:
    async def unreachable(*args, **kwargs):
        raise ConnectionError("service unavailable")

    monkeypatch.setattr(
        multi_robot_cloud,
        "probe_multi_robot_cloud_health",
        unreachable,
    )

    assert multi_robot_cloud.health_main([]) == 1
    output = json.loads(capsys.readouterr().err)
    assert output == {
        "error": "service unavailable",
        "error_type": "ConnectionError",
        "healthy": False,
    }
