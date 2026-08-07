from __future__ import annotations

import os
import subprocess
import time
from unittest import mock

import pytest

from embodied_runtime.robots.unitree.go2.agent.camera.pipe_io import read_exact_with_timeout
from embodied_runtime.robots.unitree.go2.agent.camera.types import CameraStreamError
from embodied_runtime.robots.unitree.go2.agent.camera.v4l2 import (
    V4L2YUYVSource,
    V4L2Z16Source,
)
from embodied_runtime.robots.unitree.go2.agent.camera.v4l2_format import parse_v4l2_format

FORMAT_OUTPUT = """Format Video Capture:
    Width/Height      : 640/360
    Pixel Format      : 'YUYV' (YUYV 4:2:2)
    Bytes per Line    : 1280
    Size Image        : 460800
"""


def test_persistent_source_command_and_exact_profile_validation() -> None:
    source = V4L2YUYVSource("/dev/video4", 640, 360, 10)
    assert "--stream-count=0" in source.command()
    assert "--stream-to=-" in source.command()
    assert source.frame_bytes == 640 * 360 * 2
    assert "pixelformat=Z16" in " ".join(V4L2Z16Source("/dev/video0", 640, 360, 10).command())

    parsed = parse_v4l2_format(FORMAT_OUTPUT)
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=FORMAT_OUTPUT,
        stderr="",
    )
    with mock.patch(
        "embodied_runtime.robots.unitree.go2.agent.camera.v4l2.subprocess.run",
        return_value=completed,
    ):
        assert source._validate_actual_format() == parsed


def test_driver_coerced_stride_is_rejected_before_stream_start() -> None:
    unsafe = FORMAT_OUTPUT.replace("Bytes per Line    : 1280", "Bytes per Line    : 1344")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=unsafe,
        stderr="",
    )
    source = V4L2YUYVSource("/dev/video4", 640, 360, 10)
    with (
        mock.patch(
            "embodied_runtime.robots.unitree.go2.agent.camera.v4l2.subprocess.run",
            return_value=completed,
        ),
        pytest.raises(CameraStreamError, match="bytes_per_line"),
    ):
        source._validate_actual_format()


def test_pipe_watchdog_reads_exact_payload_and_times_out() -> None:
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "rb", buffering=0)
    try:
        os.write(write_fd, b"abcdef")
        assert read_exact_with_timeout(reader, 6, 0.2) == b"abcdef"
        started = time.monotonic()
        with pytest.raises(CameraStreamError, match="watchdog"):
            read_exact_with_timeout(reader, 1, 0.05)
        assert time.monotonic() - started < 0.5
    finally:
        reader.close()
        os.close(write_fd)
