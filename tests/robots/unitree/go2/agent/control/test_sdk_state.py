from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from embodied_runtime.robots.unitree.go2.agent.control.sdk_state import (
    SportModeStateStore,
    start_state_subscription,
)


def state_message(**overrides: Any) -> SimpleNamespace:
    values = {
        "position": [1.0, 2.0, 0.3],
        "imu_state": SimpleNamespace(rpy=[0.1, -0.2, 0.4]),
        "velocity": [0.5, 0.0, -0.1],
        "yaw_speed": 0.25,
        "mode": 2,
        "gait_type": 1,
        "error_code": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_state_store_decodes_samples_and_advances_sequence() -> None:
    store = SportModeStateStore()

    store.on_message(state_message())
    first = store.state()
    store.on_message(state_message(position=[3, 4, 5], error_code=7))
    second = store.state()

    assert first is not None
    assert first.sequence == 1
    assert first.position == (1.0, 2.0, 0.3)
    assert first.velocity == (0.5, 0.0, -0.1)
    assert (first.roll, first.pitch, first.yaw) == (0.1, -0.2, 0.4)
    assert first.received_at > 0.0
    assert first.received_at_unix > 1_000_000_000
    assert second is not None
    assert second.sequence == 2
    assert second.position == (3.0, 4.0, 5.0)
    assert second.error_code == 7


def test_malformed_state_is_reported_without_replacing_last_sample(capsys: Any) -> None:
    store = SportModeStateStore()
    store.on_message(state_message())
    valid = store.state()

    store.on_message(state_message(imu_state=SimpleNamespace(rpy=[])))

    assert store.state() is valid
    assert "state callback error:" in capsys.readouterr().err


def test_subscription_is_initialized_with_expected_queue_depth() -> None:
    calls: list[tuple[Any, ...]] = []

    class Subscriber:
        def __init__(self, topic: str, state_type: Any) -> None:
            calls.append(("construct", topic, state_type))

        def Init(self, callback: Any, depth: int) -> None:
            calls.append(("start", callback, depth))

    callback = lambda _message: None
    subscriber = start_state_subscription(Subscriber, "rt/state", object, callback)

    assert isinstance(subscriber, Subscriber)
    assert calls == [
        ("construct", "rt/state", object),
        ("start", callback, 10),
    ]
