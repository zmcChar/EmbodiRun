from __future__ import annotations

from unittest.mock import Mock

from embodied_runtime.robots.go2 import BaseVelocityCommand, Go2ControlClient


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
    command = BaseVelocityCommand(0.2, 0.0, 0.1)
    action_id = client.start_velocity_lease(command)
    client.update_velocity_lease(action_id, command)
    client.stop()

    assert action_id == "lease-1"
    assert client.http.request_json.call_count == 4
