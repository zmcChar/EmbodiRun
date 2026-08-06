"""Unitree SDK2 transport with strict native-library and thread ownership rules."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from .config import (
    SDK_OWNER_CALL_GRACE_S,
    SDK_OWNER_CLOSE_TIMEOUT_S,
    SDK_OWNER_INIT_TIMEOUT_S,
)
from .transport import RobotTransport
from .types import RobotState

UNSAFE_ICEORYX_PUBLISH_SYMBOL = b"iox_pub_publish_chunk"
CONTROL_MODULE = "embodied_runtime.robots.go2.agent.control"


def loaded_dds_library_path() -> str | None:
    """Return the loaded CycloneDDS C library path on Linux, if available."""

    try:
        with open("/proc/self/maps", encoding="utf-8") as maps_file:
            for line in maps_file:
                path = line.split()[-1]
                if "/libddsc.so" in path:
                    return os.path.realpath(path)
    except (OSError, IndexError):
        return None
    return None


def dds_library_has_unsafe_iceoryx(library_path: str) -> bool:
    """Detect the Iceoryx publisher ABI that is unsafe for this binding."""

    with open(library_path, "rb") as library_file:
        return UNSAFE_ICEORYX_PUBLISH_SYMBOL in library_file.read()


def ensure_cyclonedds_library_dir(
    library_dir: str,
    *,
    reexec_args: list[str] | None = None,
) -> str:
    """Validate the requested libddsc and re-exec with it first on the path.

    The module form is intentional: deployments invoke this package with
    ``python -m`` and relative imports must remain valid after re-exec.
    """

    resolved = os.path.realpath(os.path.abspath(library_dir))
    library = os.path.join(resolved, "libddsc.so.0")
    if not os.path.isfile(library):
        raise RuntimeError(f"CycloneDDS library does not exist: {library}")
    if dds_library_has_unsafe_iceoryx(library):
        raise RuntimeError(
            f"CycloneDDS library contains the incompatible Iceoryx publisher ABI: {library}"
        )

    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if entries and os.path.realpath(entries[0]) == resolved:
        return resolved

    remaining = [entry for entry in entries if os.path.realpath(entry) != resolved]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([resolved, *remaining])
    arguments = sys.argv[1:] if reexec_args is None else reexec_args
    argv = [sys.executable, "-m", CONTROL_MODULE, *arguments]
    os.execve(sys.executable, argv, environment)
    raise RuntimeError("failed to re-exec with the requested CycloneDDS library")


@dataclass
class _SdkCall:
    method: str
    args: tuple[Any, ...]
    future: Future[int]


class UnitreeTransport(RobotTransport):
    """Make one owner thread responsible for every native SDK2 operation."""

    name = "unitree-sdk2"

    def __init__(
        self,
        interface: str,
        state_topic: str = "rt/lf/sportmodestate",
        rpc_timeout_s: float = 2.0,
        *,
        required_dds_lib_dir: str | None = None,
        _sdk_components: tuple[Any, Any, Any, Any] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._state: RobotState | None = None
        self._state_sequence = 0
        self._sdk_submit_lock = threading.Lock()
        self._sdk_lifecycle_lock = threading.Lock()
        self._sdk_queue: queue.Queue[_SdkCall | None] = queue.Queue()
        self._sdk_ready = threading.Event()
        self._sdk_stopped = threading.Event()
        self._sdk_init_error: BaseException | None = None
        self._sdk_closing = False
        self._sdk_closed = False
        self._sdk_faulted: str | None = None
        self._rpc_timeout_s = float(rpc_timeout_s)
        self._subscriber: Any = None
        self._client: Any = None

        if _sdk_components is None:
            # SDK imports stay inside live transport construction so importing
            # the package and running dry-run tests works on macOS and CI.
            try:
                from unitree_sdk2py.core.channel import (
                    ChannelFactoryInitialize,
                    ChannelSubscriber,
                )
                from unitree_sdk2py.go2.sport.sport_client import SportClient
                from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            except ImportError as exc:
                raise RuntimeError(
                    "Unitree SDK2 is unavailable; install unitree_sdk2_python "
                    "in the robot environment"
                ) from exc

            if required_dds_lib_dir is not None:
                expected_dir = os.path.realpath(required_dds_lib_dir)
                loaded_library = loaded_dds_library_path()
                if loaded_library is None or os.path.dirname(loaded_library) != expected_dir:
                    raise RuntimeError(
                        "loaded CycloneDDS library does not match the required "
                        f"directory: loaded={loaded_library!r}, required={expected_dir!r}"
                    )
                if dds_library_has_unsafe_iceoryx(loaded_library):
                    raise RuntimeError(
                        "refusing an Iceoryx-enabled CycloneDDS library that is "
                        "ABI-incompatible with this Python binding"
                    )
            _sdk_components = (
                ChannelFactoryInitialize,
                ChannelSubscriber,
                SportClient,
                SportModeState_,
            )

        self._sdk_thread = threading.Thread(
            target=self._sdk_owner_loop,
            args=(interface, state_topic, _sdk_components),
            daemon=True,
            name="go2-sdk-owner",
        )
        self._sdk_thread.start()
        if not self._sdk_ready.wait(timeout=SDK_OWNER_INIT_TIMEOUT_S):
            with self._sdk_lifecycle_lock:
                self._sdk_closed = True
                self._sdk_queue.put(None)
            raise RuntimeError("timed out while initializing Unitree SDK owner thread")
        if self._sdk_init_error is not None:
            raise RuntimeError(
                f"failed to initialize Unitree SDK: {self._sdk_init_error}"
            ) from self._sdk_init_error

    def _sdk_owner_loop(
        self,
        interface: str,
        state_topic: str,
        components: tuple[Any, Any, Any, Any],
    ) -> None:
        try:
            (
                channel_factory_initialize,
                channel_subscriber,
                sport_client,
                sport_mode_state,
            ) = components
            channel_factory_initialize(0, interface)
            subscriber = channel_subscriber(state_topic, sport_mode_state)
            subscriber.Init(self._on_state, 10)
            client = sport_client()
            client.SetTimeout(self._rpc_timeout_s)
            client.Init()
            self._subscriber = subscriber
            self._client = client
        except BaseException as exc:  # noqa: BLE001 - always release init waiter
            self._sdk_init_error = exc
            self._sdk_ready.set()
            self._sdk_stopped.set()
            return

        self._sdk_ready.set()
        try:
            while True:
                call = self._sdk_queue.get()
                if call is None:
                    break
                if not call.future.set_running_or_notify_cancel():
                    continue
                with self._sdk_lifecycle_lock:
                    faulted = self._sdk_faulted
                if faulted is not None and call.method != "StopMove":
                    call.future.set_exception(
                        RuntimeError(f"Unitree SDK transport is faulted: {faulted}")
                    )
                    continue
                call_started = time.monotonic()
                try:
                    method = getattr(client, call.method)
                    call.future.set_result(int(method(*call.args)))
                except BaseException as exc:  # noqa: BLE001 - propagate through Future
                    call.future.set_exception(exc)
                finally:
                    elapsed = time.monotonic() - call_started
                    timeout_s = max(0.1, self._rpc_timeout_s + SDK_OWNER_CALL_GRACE_S)
                    if elapsed > timeout_s and call.method != "StopMove":
                        with self._sdk_lifecycle_lock:
                            self._sdk_faulted = f"in-flight {call.method} exceeded {timeout_s:.1f}s"
        finally:
            with self._sdk_lifecycle_lock:
                self._sdk_closing = True
                self._sdk_closed = True
            owner_error = RuntimeError("Unitree SDK owner thread exited")
            while True:
                try:
                    pending = self._sdk_queue.get_nowait()
                except queue.Empty:
                    break
                if pending is not None:
                    try:
                        if pending.future.set_running_or_notify_cancel():
                            pending.future.set_exception(owner_error)
                    except RuntimeError:
                        pass
            # Destruction of native-backed objects remains on their owner
            # thread. ChannelFactory is process-wide and is not closed here.
            self._client = None
            self._subscriber = None
            del client
            del subscriber
            self._sdk_stopped.set()

    def _enqueue_sdk_call(
        self,
        method: str,
        args: tuple[Any, ...],
        *,
        allow_closing: bool = False,
    ) -> _SdkCall:
        call = _SdkCall(method=method, args=args, future=Future())
        with self._sdk_lifecycle_lock:
            if self._sdk_closed:
                raise RuntimeError("Unitree SDK transport is closed")
            if self._sdk_closing and not allow_closing:
                raise RuntimeError("Unitree SDK transport is closing")
            if self._sdk_stopped.is_set() or not self._sdk_thread.is_alive():
                raise RuntimeError("Unitree SDK owner thread is unavailable")
            if self._sdk_faulted is not None and method != "StopMove":
                raise RuntimeError(f"Unitree SDK transport is faulted: {self._sdk_faulted}")
            self._sdk_queue.put(call)
        return call

    @staticmethod
    def _completed_sdk_result(call: _SdkCall) -> int:
        try:
            return int(call.future.result())
        except BaseException as exc:
            raise RuntimeError(f"Unitree SDK {call.method} failed: {exc}") from exc

    def _wait_sdk_call(self, call: _SdkCall, *, cancel_if_pending: bool = True) -> int:
        timeout_s = max(0.1, self._rpc_timeout_s + SDK_OWNER_CALL_GRACE_S)
        try:
            return int(call.future.result(timeout=timeout_s))
        except FutureTimeoutError as exc:
            if call.future.done():
                return self._completed_sdk_result(call)
            if cancel_if_pending and call.future.cancel():
                raise TimeoutError(
                    f"Unitree SDK {call.method} timed out before execution; call cancelled"
                ) from exc
            if call.future.done():
                return self._completed_sdk_result(call)
            state = "in-flight" if call.future.running() else "queued"
            fault = f"{state} {call.method} exceeded {timeout_s:.1f}s"
            with self._sdk_lifecycle_lock:
                self._sdk_faulted = fault
            disposition = (
                "transport faulted"
                if cancel_if_pending
                else "transport faulted; safety call remains queued"
            )
            raise TimeoutError(
                f"Unitree SDK {call.method} timed out {state}; {disposition}"
            ) from exc
        except BaseException as exc:
            raise RuntimeError(f"Unitree SDK {call.method} failed: {exc}") from exc

    def _sdk_call(self, method: str, *args: Any) -> int:
        with self._sdk_submit_lock:
            return self._wait_sdk_call(self._enqueue_sdk_call(method, args))

    def _on_state(self, message: Any) -> None:
        try:
            received_at = time.monotonic()
            received_at_unix = time.time()
            position = tuple(float(value) for value in message.position)
            roll = float(message.imu_state.rpy[0])
            pitch = float(message.imu_state.rpy[1])
            yaw = float(message.imu_state.rpy[2])
            velocity = tuple(float(value) for value in message.velocity)
            yaw_rate = float(message.yaw_speed)
            mode = int(message.mode)
            gait_type = int(message.gait_type)
            error_code = int(message.error_code)
            with self._lock:
                self._state_sequence += 1
                self._state = RobotState(
                    received_at=received_at,
                    sequence=self._state_sequence,
                    position=position,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    velocity=velocity,
                    yaw_rate=yaw_rate,
                    received_at_unix=received_at_unix,
                    mode=mode,
                    gait_type=gait_type,
                    error_code=error_code,
                )
        except Exception as exc:  # noqa: BLE001 - SDK callback thread must survive
            # The SDK callback thread must survive a malformed state sample.
            print(f"state callback error: {exc}", file=sys.stderr, flush=True)

    def state(self) -> RobotState | None:
        with self._lock:
            return self._state

    def move(self, vx: float, vy: float, yaw_rate: float) -> int:
        return self._sdk_call("Move", vx, vy, yaw_rate)

    def stop(self) -> int:
        with self._sdk_submit_lock:
            call = self._enqueue_sdk_call("StopMove", ())
            # Never cancel a safety stop merely because it was queued behind a
            # slow native RPC. It remains in the owner-thread FIFO.
            return self._wait_sdk_call(call, cancel_if_pending=False)

    def posture(self, action: str) -> int:
        methods = {
            "stand_up": "StandUp",
            "stand_down": "StandDown",
            "balance_stand": "BalanceStand",
            "recovery_stand": "RecoveryStand",
        }
        return self._sdk_call(methods[action])

    def close(self) -> None:
        with self._sdk_submit_lock:
            with self._sdk_lifecycle_lock:
                if self._sdk_closed:
                    already_closed = True
                    final_stop = None
                else:
                    already_closed = False
                    self._sdk_closing = True
                    if self._sdk_stopped.is_set() or not self._sdk_thread.is_alive():
                        final_stop = None
                    else:
                        final_stop = _SdkCall("StopMove", (), Future())
                        self._sdk_queue.put(final_stop)

            if not already_closed:
                if final_stop is not None:
                    try:
                        self._wait_sdk_call(final_stop, cancel_if_pending=False)
                    except Exception as exc:  # noqa: BLE001 - best-effort final safety stop
                        print(
                            f"final Unitree StopMove failed: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                with self._sdk_lifecycle_lock:
                    self._sdk_closed = True
                    self._sdk_queue.put(None)

            self._sdk_thread.join(timeout=SDK_OWNER_CLOSE_TIMEOUT_S)
        if self._sdk_thread.is_alive():
            print(
                "Unitree SDK owner thread did not exit before close timeout",
                file=sys.stderr,
                flush=True,
            )


__all__ = [
    "UnitreeTransport",
    "dds_library_has_unsafe_iceoryx",
    "ensure_cyclonedds_library_dir",
    "loaded_dds_library_path",
]
