from __future__ import annotations

from unittest.mock import Mock

import pytest

from embodied_runtime.robots.unitree.go2 import Go2ClientError, Go2ControlClient
from embodied_runtime.tasks.navigation import PlanarVelocityCommand


def test_control_client_parses_sdk_state_and_updates_lease() -> None:
    client = Go2ControlClient("http://go2:8080")
    client.http.request_json = Mock(
        side_effect=[
            {
                "transport": "unitree-sdk2",
                "robot_state_available": True,
                "robot_state_fresh": True,
                "robot_state": {
                    "position": [1.0, 2.0, 0.0],
                    "velocity": [0.1, 0.0, 0.0],
                    "yaw": 0.3,
                    "yaw_rate": 0.0,
                    "sequence": 7,
                    "received_at_unix": 100.0,
                },
                "active_action": None,
            },
            {"action_id": "lease-1"},
            {"accepted": True},
            {"stopped": True},
        ]
    )

    assert client.state().pose.x_m == 1.0
    command = PlanarVelocityCommand(0.2, 0.0, 0.1)
    action_id = client.start_velocity_lease(command)
    client.update_velocity_lease(action_id, command)
    client.stop()

    assert action_id == "lease-1"
    assert client.http.request_json.call_count == 4


def test_preflight_requires_operator_interlock_and_idle_robot() -> None:
    payload = {
        "transport": "unitree-sdk2",
        "operator_motion_ready": True,
        "closing": False,
        "robot_state_available": True,
        "robot_state_fresh": True,
        "robot_state": {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "yaw_rate": 0.0,
            "sequence": 1,
            "received_at_unix": 100.0,
        },
        "active_action": None,
    }
    client = Go2ControlClient("http://go2:8080")
    client.http.request_json = Mock(return_value=payload)

    assert client.preflight().sequence == 1

    payload["operator_motion_ready"] = False
    with pytest.raises(Go2ClientError, match="interlock"):
        client.preflight()
