from __future__ import annotations

import struct
import zlib
from typing import Dict  # noqa: UP035

import pytest

from embodied_runtime.robots.go2.agent.camera.encoding import (
    analyze_z16_depth,
    encode_z16_as_png,
)
from embodied_runtime.robots.go2.agent.camera.stores import DepthStore, FrameStore
from embodied_runtime.robots.go2.agent.camera.types import CameraStreamError

PNG_FIXTURE = b"\x89PNG\r\n\x1a\nfixture-depth"


class MutableClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_frame_health_transitions_from_starting_to_ok_to_stale() -> None:
    clock = MutableClock()
    store = FrameStore(clock=clock, wall_clock=lambda: 1234.5)
    store.set_capture_running(True)
    assert store.health(2.0)["status"] == "starting"
    assert store.update(b"jpeg").sequence == 1
    assert store.health(2.0)["status"] == "ok"

    clock.value += 2.1
    assert store.health(2.0)["status"] == "degraded"
    assert store.health(2.0)["frame_fresh"] is False


def test_depth_fails_closed_and_sequence_must_increase() -> None:
    store = DepthStore(4, 4)
    store.set_capture_running(True)
    store.update(None, None, 0.0, error="no valid pixels")
    assert store.status(0.5)["available"] is False

    snapshot = store.update(1.0, 0.6, 0.9, depth_png=PNG_FIXTURE)
    assert snapshot.sequence == 2
    assert store.status(0.5)["available"] is True
    assert store.status(0.5)["raw_depth_available"] is True
    with pytest.raises(ValueError, match="increase"):
        store.update(1.0, 0.6, 0.9, sequence=2)
    with pytest.raises(ValueError, match="PNG signature"):
        store.update(1.0, 0.6, 0.9, depth_png=b"not-png")


def test_depth_analysis_reports_median_and_conservative_minimum() -> None:
    values = [0] * 16
    values[5], values[6], values[9], values[10] = 1000, 1500, 500, 0
    analysis = analyze_z16_depth(
        struct.pack("<16H", *values),
        width=4,
        height=4,
        depth_scale=0.001,
        roi_width_ratio=0.5,
        roi_height_ratio=0.5,
        min_valid_fraction=0.5,
        max_depth_m=10.0,
    )
    assert analysis.center_distance_m == pytest.approx(1.0)
    assert analysis.minimum_distance_m == pytest.approx(0.5)
    assert analysis.valid_fraction == pytest.approx(0.75)


def test_z16_png_round_trips_exact_uint16_samples() -> None:
    values = (0, 1, 1000, 65535)
    payload = encode_z16_as_png(struct.pack("<4H", *values), 2, 2)
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")

    chunks: Dict[bytes, bytes] = {}  # noqa: UP006
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])[0]
        assert zlib.crc32(kind + data) & 0xFFFFFFFF == expected_crc
        chunks[kind] = chunks.get(kind, b"") + data
        offset += 12 + length

    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (2, 2, 16, 0, 0, 0, 0)
    scanlines = zlib.decompress(chunks[b"IDAT"])
    assert scanlines[0] == 0
    assert scanlines[5] == 0
    decoded = struct.unpack(">2H", scanlines[1:5]) + struct.unpack(">2H", scanlines[6:10])
    assert decoded == values

    with pytest.raises(CameraStreamError, match="Z16 frame size"):
        encode_z16_as_png(b"\0\0", 2, 2)
