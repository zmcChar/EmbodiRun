from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from embodied_runtime.robots.go2.agent.control import (
    UnitreeTransport,
    dds_library_has_unsafe_iceoryx,
)


def recording_sdk_components(
    move_error: Exception | None = None,
    block_started: threading.Event | None = None,
    block_release: threading.Event | None = None,
) -> tuple[tuple[Any, Any, Any, Any], list[tuple[str, tuple[Any, ...], int]]]:
    records: list[tuple[str, tuple[Any, ...], int]] = []
    record_lock = threading.Lock()

    def record(operation: str, *args: Any) -> None:
        with record_lock:
            records.append((operation, args, threading.get_ident()))

    def channel_factory_initialize(domain: int, interface: str) -> None:
        record("channel_factory_initialize", domain, interface)

    class Subscriber:
        def __init__(self, topic: str, state_type: Any) -> None:
            record("subscriber_init", topic, state_type)

        def Init(self, callback: Callable[..., Any], queue_depth: int) -> None:
            record("subscriber_start", callback, queue_depth)

        def __del__(self) -> None:
            record("subscriber_destroyed")

    class Client:
        def __init__(self) -> None:
            record("client_init")

        def SetTimeout(self, timeout_s: float) -> None:
            record("set_timeout", timeout_s)

        def Init(self) -> None:
            record("client_start")

        def Move(self, vx: float, vy: float, yaw_rate: float) -> int:
            record("move", vx, vy, yaw_rate)
            if move_error is not None:
                raise move_error
            return 0

        def StopMove(self) -> int:
            record("stop")
            return 0

        def StandUp(self) -> int:
            record("stand_up")
            return 0

        def StandDown(self) -> int:
            record("stand_down")
            return 0

        def BalanceStand(self) -> int:
            record("balance_stand")
            return 0

        def RecoveryStand(self) -> int:
            record("recovery_stand")
            return 0

        def Block(self) -> int:
            record("block")
            if block_started is not None:
                block_started.set()
            if block_release is not None:
                block_release.wait(timeout=3.0)
            return 0

        def __del__(self) -> None:
            record("client_destroyed")

    return (
        channel_factory_initialize,
        Subscriber,
        Client,
        object,
    ), records


def test_detects_unsafe_iceoryx_publisher_symbol() -> None:
    with tempfile.TemporaryDirectory() as directory:
        safe = Path(directory) / "safe.so"
        unsafe = Path(directory) / "unsafe.so"
        safe.write_bytes(b"ELF-placeholder-without-shm")
        unsafe.write_bytes(b"prefix\x00iox_pub_publish_chunk\x00suffix")

        assert not dds_library_has_unsafe_iceoryx(str(safe))
        assert dds_library_has_unsafe_iceoryx(str(unsafe))


def test_all_sdk_initialization_and_calls_run_on_owner_thread() -> None:
    components, records = recording_sdk_components()
    transport = UnitreeTransport("test0", _sdk_components=components)
    caller_thread_ids = {threading.get_ident()}
    errors: list[Exception] = []

    def issue_move() -> None:
        caller_thread_ids.add(threading.get_ident())
        try:
            transport.move(0.1, 0.0, 0.0)
        except Exception as exc:  # noqa: BLE001 - capture cross-thread assertion result
            errors.append(exc)

    caller = threading.Thread(target=issue_move, name="test-http-worker")
    caller.start()
    caller.join(timeout=1.0)
    try:
        assert not caller.is_alive()
        assert errors == []
        assert transport.stop() == 0
        assert transport.posture("balance_stand") == 0
    finally:
        transport.close()

    operations = [record[0] for record in records]
    assert operations[:10] == [
        "channel_factory_initialize",
        "subscriber_init",
        "subscriber_start",
        "client_init",
        "set_timeout",
        "client_start",
        "move",
        "stop",
        "balance_stand",
        "stop",
    ]
    assert sorted(operations[10:]) == ["client_destroyed", "subscriber_destroyed"]
    sdk_thread_ids = {record[2] for record in records}
    assert len(sdk_thread_ids) == 1
    assert sdk_thread_ids.isdisjoint(caller_thread_ids)
    assert not transport._sdk_thread.is_alive()


def test_sdk_call_error_is_propagated_without_killing_owner() -> None:
    components, records = recording_sdk_components(ValueError("rpc failed"))
    transport = UnitreeTransport("test0", _sdk_components=components)
    try:
        with pytest.raises(RuntimeError, match="Move failed: rpc failed"):
            transport.move(0.1, 0.0, 0.0)
        assert transport.stop() == 0
        assert transport._sdk_thread.is_alive()
    finally:
        transport.close()
    assert "move" in [record[0] for record in records]
    assert "stop" in [record[0] for record in records]


def test_queued_call_is_cancelled_if_it_times_out_before_execution() -> None:
    block_started = threading.Event()
    block_release = threading.Event()
    components, records = recording_sdk_components(
        block_started=block_started,
        block_release=block_release,
    )
    transport = UnitreeTransport("test0", rpc_timeout_s=0.01, _sdk_components=components)
    blocker = transport._enqueue_sdk_call("Block", ())
    try:
        assert block_started.wait(timeout=1.0)
        with pytest.raises(TimeoutError, match="before execution; call cancelled"):
            transport.move(0.1, 0.0, 0.0)
        assert not any(record[0] == "move" for record in records)
        block_release.set()
        assert transport._wait_sdk_call(blocker) == 0
    finally:
        block_release.set()
        transport.close()
    assert not any(record[0] == "move" for record in records)


def test_close_sends_final_stop_and_rejects_later_calls() -> None:
    components, records = recording_sdk_components()
    transport = UnitreeTransport("test0", _sdk_components=components)
    transport.close()
    transport.close()

    assert sum(record[0] == "stop" for record in records) == 1
    with pytest.raises(RuntimeError, match="closed"):
        transport.move(0.1, 0.0, 0.0)


def test_close_never_cancels_final_stop_behind_slow_rpc() -> None:
    block_started = threading.Event()
    block_release = threading.Event()
    components, records = recording_sdk_components(
        block_started=block_started,
        block_release=block_release,
    )
    transport = UnitreeTransport("test0", rpc_timeout_s=0.01, _sdk_components=components)
    transport._enqueue_sdk_call("Block", ())
    assert block_started.wait(timeout=1.0)

    releaser = threading.Thread(target=lambda: (time.sleep(1.2), block_release.set()))
    releaser.start()
    transport.close()
    releaser.join(timeout=1.0)

    operations = [record[0] for record in records]
    assert "block" in operations
    assert operations.count("stop") == 1
    assert operations.index("block") < operations.index("stop")
    assert not transport._sdk_thread.is_alive()


def test_slow_call_fault_prevents_concurrent_later_motion() -> None:
    block_started = threading.Event()
    block_release = threading.Event()
    components, records = recording_sdk_components(
        block_started=block_started,
        block_release=block_release,
    )
    transport = UnitreeTransport("test0", rpc_timeout_s=0.01, _sdk_components=components)
    errors: list[Exception] = []

    def block() -> None:
        try:
            transport._sdk_call("Block")
        except Exception as exc:  # noqa: BLE001 - capture cross-thread assertion result
            errors.append(exc)

    blocker = threading.Thread(target=block)
    blocker.start()
    try:
        assert block_started.wait(timeout=1.0)
        deadline = time.monotonic() + 2.0
        while not errors and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(errors) == 1
        with pytest.raises(RuntimeError, match="transport is faulted"):
            transport.move(0.1, 0.0, 0.0)
        assert not any(record[0] == "move" for record in records)
    finally:
        block_release.set()
        blocker.join(timeout=1.0)
        transport.close()
    assert not any(record[0] == "move" for record in records)
