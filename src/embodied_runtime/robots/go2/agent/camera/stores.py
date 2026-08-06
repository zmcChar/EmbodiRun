"""Thread-safe latest-frame stores for the dog-side camera service."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, Optional  # noqa: UP035

from .types import DepthSnapshot, FrameSnapshot, VideoProfile


def _validate_timestamp(name: str, value: Optional[float]) -> None:  # noqa: UP045
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_source_frame_number(value: Optional[int]) -> None:  # noqa: UP045
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("source_frame_number must be a non-negative integer")


class FrameStore:
    """One immutable RGB slot plus capture-health metadata."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        device: Optional[str] = None,  # noqa: UP045
    ) -> None:
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._frame: Optional[FrameSnapshot] = None  # noqa: UP045
        self._sequence = 0
        self._capture_running = False
        self._last_error: Optional[str] = None  # noqa: UP045
        self._started_at = self._clock()
        self._device = device
        self._profile: Optional[VideoProfile] = None  # noqa: UP045

    def set_capture_running(self, running: bool) -> None:
        with self._lock:
            self._capture_running = running

    def set_error(self, error: Optional[str]) -> None:  # noqa: UP045
        with self._lock:
            self._last_error = error

    def set_source_profile(self, device: str, profile: VideoProfile) -> None:
        with self._lock:
            self._device = device
            self._profile = profile

    def update(
        self,
        jpeg: bytes,
        sequence: Optional[int] = None,  # noqa: UP045
        captured_at_unix: Optional[float] = None,  # noqa: UP045
        captured_at_monotonic: Optional[float] = None,  # noqa: UP045
        source_timestamp_ms: Optional[float] = None,  # noqa: UP045
        source_frame_number: Optional[int] = None,  # noqa: UP045
    ) -> FrameSnapshot:
        if not jpeg:
            raise ValueError("jpeg must not be empty")
        for name, value in (
            ("captured_at_unix", captured_at_unix),
            ("captured_at_monotonic", captured_at_monotonic),
            ("source_timestamp_ms", source_timestamp_ms),
        ):
            _validate_timestamp(name, value)
        _validate_source_frame_number(source_frame_number)

        with self._lock:
            next_sequence = self._sequence + 1 if sequence is None else sequence
            if (
                isinstance(next_sequence, bool)
                or not isinstance(next_sequence, int)
                or next_sequence <= self._sequence
            ):
                raise ValueError("frame sequence must increase")
            snapshot = FrameSnapshot(
                jpeg=bytes(jpeg),
                sequence=next_sequence,
                captured_at_unix=(
                    self._wall_clock() if captured_at_unix is None else float(captured_at_unix)
                ),
                captured_at_monotonic=(
                    self._clock() if captured_at_monotonic is None else float(captured_at_monotonic)
                ),
                source_timestamp_ms=(
                    None if source_timestamp_ms is None else float(source_timestamp_ms)
                ),
                source_frame_number=source_frame_number,
            )
            self._sequence = next_sequence
            self._frame = snapshot
            self._last_error = None
            return snapshot

    def latest(self) -> Optional[FrameSnapshot]:  # noqa: UP045
        with self._lock:
            return self._frame

    def frame_age(self, frame: FrameSnapshot) -> float:
        return max(0.0, self._clock() - frame.captured_at_monotonic)

    def health(self, max_frame_age: float) -> Dict[str, object]:  # noqa: UP006
        with self._lock:
            frame = self._frame
            running = self._capture_running
            error = self._last_error
            uptime = max(0.0, self._clock() - self._started_at)
            device = self._device
            profile = self._profile

        age = None if frame is None else self.frame_age(frame)
        fresh = frame is not None and age is not None and age <= max_frame_age
        if running and fresh:
            status = "ok"
        elif running and frame is None and error is None:
            status = "starting"
        else:
            status = "degraded"
        return {
            "status": status,
            "capture_running": running,
            "frame_available": frame is not None,
            "frame_fresh": fresh,
            "frame_sequence": 0 if frame is None else frame.sequence,
            "frame_age_s": None if age is None else round(age, 3),
            "max_frame_age_s": max_frame_age,
            "last_error": error,
            "uptime_s": round(uptime, 3),
            "rgb_device": device,
            "rgb_profile": None if profile is None else profile.as_dict(),
            "source_timestamp_ms": None if frame is None else frame.source_timestamp_ms,
            "source_frame_number": None if frame is None else frame.source_frame_number,
        }


class DepthStore:
    """One immutable depth slot whose readiness always fails closed."""

    def __init__(
        self,
        width: int,
        height: int,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        device: Optional[str] = None,  # noqa: UP045
        backend: str = "v4l2",
    ) -> None:
        self.width = width
        self.height = height
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._snapshot: Optional[DepthSnapshot] = None  # noqa: UP045
        self._sequence = 0
        self._capture_running = False
        self._last_error: Optional[str] = None  # noqa: UP045
        self._device = device
        self._profile: Optional[VideoProfile] = None  # noqa: UP045
        self._backend = backend
        self._registration_verified = False
        self._rgb_depth_aligned = False
        self._depth_scale_m: Optional[float] = None  # noqa: UP045

    def set_capture_running(self, running: bool) -> None:
        with self._lock:
            self._capture_running = running

    def set_error(self, error: Optional[str]) -> None:  # noqa: UP045
        with self._lock:
            self._last_error = error

    def set_source_profile(self, device: str, profile: VideoProfile) -> None:
        with self._lock:
            self._device = device
            self._profile = profile

    def set_registration(
        self,
        verified: bool,
        aligned: bool,
        depth_scale_m: Optional[float],  # noqa: UP045
    ) -> None:
        with self._lock:
            self._registration_verified = verified
            self._rgb_depth_aligned = aligned
            self._depth_scale_m = depth_scale_m

    def update(
        self,
        center_distance_m: Optional[float],  # noqa: UP045
        minimum_distance_m: Optional[float],  # noqa: UP045
        valid_fraction: float,
        error: Optional[str] = None,  # noqa: UP045
        sequence: Optional[int] = None,  # noqa: UP045
        captured_at_unix: Optional[float] = None,  # noqa: UP045
        captured_at_monotonic: Optional[float] = None,  # noqa: UP045
        source_timestamp_ms: Optional[float] = None,  # noqa: UP045
        source_frame_number: Optional[int] = None,  # noqa: UP045
        depth_png: Optional[bytes] = None,  # noqa: UP045
    ) -> DepthSnapshot:
        for name, value in (
            ("center_distance_m", center_distance_m),
            ("minimum_distance_m", minimum_distance_m),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if (
            isinstance(valid_fraction, bool)
            or not isinstance(valid_fraction, (int, float))
            or not math.isfinite(float(valid_fraction))
            or not 0.0 <= float(valid_fraction) <= 1.0
        ):
            raise ValueError("valid_fraction must be between zero and one")
        for name, value in (
            ("captured_at_unix", captured_at_unix),
            ("captured_at_monotonic", captured_at_monotonic),
            ("source_timestamp_ms", source_timestamp_ms),
        ):
            _validate_timestamp(name, value)
        _validate_source_frame_number(source_frame_number)

        normalized_png: Optional[bytes] = None  # noqa: UP045
        if depth_png is not None:
            if not isinstance(depth_png, (bytes, bytearray)) or not depth_png:
                raise ValueError("depth_png must be non-empty bytes or None")
            normalized_png = bytes(depth_png)
            if not normalized_png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("depth_png must have a PNG signature")

        with self._lock:
            next_sequence = self._sequence + 1 if sequence is None else sequence
            if (
                isinstance(next_sequence, bool)
                or not isinstance(next_sequence, int)
                or next_sequence <= self._sequence
            ):
                raise ValueError("depth sequence must increase")
            snapshot = DepthSnapshot(
                center_distance_m=(None if center_distance_m is None else float(center_distance_m)),
                minimum_distance_m=(
                    None if minimum_distance_m is None else float(minimum_distance_m)
                ),
                valid_fraction=float(valid_fraction),
                sequence=next_sequence,
                captured_at_unix=(
                    self._wall_clock() if captured_at_unix is None else float(captured_at_unix)
                ),
                captured_at_monotonic=(
                    self._clock() if captured_at_monotonic is None else float(captured_at_monotonic)
                ),
                depth_png=normalized_png,
                source_timestamp_ms=(
                    None if source_timestamp_ms is None else float(source_timestamp_ms)
                ),
                source_frame_number=source_frame_number,
            )
            self._sequence = next_sequence
            self._snapshot = snapshot
            self._last_error = error
            return snapshot

    def latest(self) -> Optional[DepthSnapshot]:  # noqa: UP045
        with self._lock:
            return self._snapshot

    def snapshot_age(self, snapshot: DepthSnapshot) -> float:
        return max(0.0, self._clock() - snapshot.captured_at_monotonic)

    def status(self, max_depth_age: float) -> Dict[str, object]:  # noqa: UP006
        with self._lock:
            snapshot = self._snapshot
            running = self._capture_running
            error = self._last_error
            device = self._device
            profile = self._profile
            backend = self._backend
            registration_verified = self._registration_verified
            rgb_depth_aligned = self._rgb_depth_aligned
            depth_scale_m = self._depth_scale_m

        age = None if snapshot is None else self.snapshot_age(snapshot)
        values_valid = (
            snapshot is not None
            and snapshot.center_distance_m is not None
            and snapshot.minimum_distance_m is not None
        )
        available = bool(running and values_valid and age is not None and age <= max_depth_age)
        return {
            "available": available,
            "center_distance_m": (None if snapshot is None else snapshot.center_distance_m),
            "minimum_distance_m": (None if snapshot is None else snapshot.minimum_distance_m),
            "captured_at_unix": None if snapshot is None else snapshot.captured_at_unix,
            "age_s": None if age is None else round(age, 3),
            "width": self.width,
            "height": self.height,
            "frame_sequence": 0 if snapshot is None else snapshot.sequence,
            "sequence": 0 if snapshot is None else snapshot.sequence,
            "valid_fraction": (0.0 if snapshot is None else round(snapshot.valid_fraction, 4)),
            "raw_depth_available": bool(snapshot is not None and snapshot.depth_png is not None),
            "capture_running": running,
            "last_error": error,
            "depth_device": device,
            "depth_profile": None if profile is None else profile.as_dict(),
            "backend": backend,
            "registration_verified": registration_verified,
            "rgb_depth_aligned": rgb_depth_aligned,
            "depth_scale_m": depth_scale_m,
            "source_timestamp_ms": (None if snapshot is None else snapshot.source_timestamp_ms),
            "source_frame_number": (None if snapshot is None else snapshot.source_frame_number),
        }


__all__ = ["DepthStore", "FrameStore"]
