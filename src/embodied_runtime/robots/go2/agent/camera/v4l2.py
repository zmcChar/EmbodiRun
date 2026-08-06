"""Persistent raw V4L2 sources for dog-side diagnostics.

Split RGB/depth V4L2 nodes cannot prove that the two frames are registered or
captured together. They are therefore retained for diagnostics only; the
RealSense backend is required for a navigation-ready RGB-D observation.
"""

from __future__ import annotations

import collections
import os
import re
import select
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Deque, List, Optional, Sequence  # noqa: UP035

from .types import CameraStreamError, VideoProfile


def parse_v4l2_format(output: str) -> VideoProfile:
    """Parse every raw-video field needed to delimit an unheadered pipe."""

    patterns = {
        "dimensions": r"Width/Height\s*:\s*(\d+)\s*/\s*(\d+)",
        "pixel_format": r"Pixel Format\s*:\s*'([^']+)'",
        "bytes_per_line": r"Bytes per Line\s*:\s*(\d+)",
        "size_image": r"Size Image\s*:\s*(\d+)",
    }
    matches = {name: re.search(pattern, output) for name, pattern in patterns.items()}
    missing = [name for name, match in matches.items() if match is None]
    if missing:
        raise CameraStreamError("v4l2-ctl format output is missing: {}".format(", ".join(missing)))
    dimensions = matches["dimensions"]
    pixel_format = matches["pixel_format"]
    bytes_per_line = matches["bytes_per_line"]
    size_image = matches["size_image"]
    assert dimensions is not None
    assert pixel_format is not None
    assert bytes_per_line is not None
    assert size_image is not None
    return VideoProfile(
        width=int(dimensions.group(1)),
        height=int(dimensions.group(2)),
        pixel_format=pixel_format.group(1).strip(),
        bytes_per_line=int(bytes_per_line.group(1)),
        size_image=int(size_image.group(1)),
    )


def read_exact_with_timeout(stream: object, size: int, timeout: float) -> bytes:
    """Read exactly ``size`` bytes or fail before the frame watchdog expires."""

    if size <= 0 or timeout <= 0:
        raise ValueError("size and timeout must be positive")
    try:
        file_descriptor = stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError) as error:
        raise CameraStreamError("camera pipe has no usable file descriptor") from error

    deadline = time.monotonic() + timeout
    chunks: List[bytes] = []  # noqa: UP006
    remaining = size
    while remaining:
        wait_time = deadline - time.monotonic()
        if wait_time <= 0:
            raise CameraStreamError(f"camera frame watchdog expired after {timeout:.3f}s")
        try:
            readable, _, _ = select.select([file_descriptor], [], [], wait_time)
        except (OSError, ValueError) as error:
            raise CameraStreamError(f"camera pipe polling failed: {error}") from error
        if not readable:
            raise CameraStreamError(f"camera frame watchdog expired after {timeout:.3f}s")
        try:
            chunk = os.read(file_descriptor, remaining)
        except OSError as error:
            raise CameraStreamError(f"camera pipe read failed: {error}") from error
        if not chunk:
            raise CameraStreamError("camera pipe reached EOF mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class V4L2Source:
    """One long-lived ``v4l2-ctl`` process yielding fixed-size raw frames."""

    pixel_format = "YUYV"

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        warmup_frames: int,
        executable: str = "v4l2-ctl",
        frame_timeout: float = 2.0,
        format_timeout: float = 5.0,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.executable = executable
        self.frame_timeout = frame_timeout
        self.format_timeout = format_timeout
        self.frame_bytes = width * height * 2
        self.actual_format: Optional[VideoProfile] = None  # noqa: UP045
        self._process: Optional[subprocess.Popen] = None  # noqa: UP045
        self._process_lock = threading.Lock()
        self._stderr_tail: Deque[str] = collections.deque(maxlen=20)  # noqa: UP006
        self._stderr_thread: Optional[threading.Thread] = None  # noqa: UP045

    def command(self) -> List[str]:  # noqa: UP006
        return [
            self.executable,
            "--device",
            self.device,
            "--set-fmt-video",
            (f"width={self.width},height={self.height},pixelformat={self.pixel_format}"),
            "--stream-mmap=3",
            f"--stream-skip={self.warmup_frames}",
            "--stream-count=0",
            "--stream-to=-",
        ]

    def format_command(self) -> List[str]:  # noqa: UP006
        return [
            self.executable,
            "--device",
            self.device,
            "--set-fmt-video",
            (f"width={self.width},height={self.height},pixelformat={self.pixel_format}"),
            "--get-fmt-video",
        ]

    def _validate_actual_format(self) -> VideoProfile:
        try:
            completed = subprocess.run(
                self.format_command(),
                check=True,
                capture_output=True,
                text=True,
                timeout=self.format_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise CameraStreamError(
                f"V4L2 format validation exceeded {self.format_timeout:.1f}s"
            ) from error
        except subprocess.CalledProcessError as error:
            details = (error.stderr or error.stdout or "unknown V4L2 error").strip()
            raise CameraStreamError(f"V4L2 format validation failed: {details}") from error
        except OSError as error:
            raise CameraStreamError(f"could not validate V4L2 format: {error}") from error

        actual = parse_v4l2_format(f"{completed.stdout or ''}\n{completed.stderr or ''}")
        expected_bytes_per_line = self.width * 2
        expected_size_image = expected_bytes_per_line * self.height
        expected = VideoProfile(
            width=self.width,
            height=self.height,
            pixel_format=self.pixel_format,
            bytes_per_line=expected_bytes_per_line,
            size_image=expected_size_image,
        )
        mismatches = []
        for field in (
            "width",
            "height",
            "pixel_format",
            "bytes_per_line",
            "size_image",
        ):
            if getattr(actual, field) != getattr(expected, field):
                mismatches.append(
                    f"{field}={getattr(actual, field)} (expected {getattr(expected, field)})"
                )
        if mismatches:
            raise CameraStreamError(
                "unsafe V4L2 format negotiation: {}".format("; ".join(mismatches))
            )
        self.actual_format = actual
        return actual

    def open(self) -> None:
        with self._process_lock:
            if self._process is not None:
                raise CameraStreamError("camera source is already open")
        if shutil.which(self.executable) is None:
            raise CameraStreamError(f"{self.executable} is not installed")
        if not Path(self.device).exists():
            raise CameraStreamError(f"camera device does not exist: {self.device}")

        self._validate_actual_format()
        try:
            process = subprocess.Popen(
                self.command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as error:
            raise CameraStreamError(f"could not start v4l2-ctl: {error}") from error

        self._stderr_tail.clear()
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process,),
            name="go2-camera-v4l2-stderr",
            daemon=True,
        )
        try:
            with self._process_lock:
                self._process = process
                self._stderr_thread = stderr_thread
                stderr_thread.start()
        except Exception as error:
            self.close()
            raise CameraStreamError(f"could not start V4L2 stderr monitor: {error}") from error

    def _drain_stderr(self, process: subprocess.Popen) -> None:
        if process.stderr is None:
            return
        try:
            for raw_line in iter(process.stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._stderr_tail.append(line)
        except (OSError, ValueError):
            return

    def read(self) -> bytes:
        with self._process_lock:
            process = self._process
        if process is None or process.stdout is None:
            raise CameraStreamError("camera source is not open")
        try:
            return read_exact_with_timeout(
                process.stdout,
                self.frame_bytes,
                self.frame_timeout,
            )
        except CameraStreamError as error:
            return_code = process.poll()
            details = "; ".join(self._stderr_tail)
            suffix = f"; stderr={details}" if details else ""
            raise CameraStreamError(f"{error}; v4l2-ctl code={return_code}{suffix}") from error

    def close(self) -> None:
        with self._process_lock:
            process = self._process
            self._process = None
            stderr_thread = self._stderr_thread
            self._stderr_thread = None
        if process is None:
            return

        try:
            running = process.poll() is None
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            running = True
        if running:
            try:
                process.terminate()
            except (OSError, RuntimeError, ValueError):
                pass
            try:
                process.wait(timeout=2.0)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                try:
                    process.kill()
                except (OSError, RuntimeError, ValueError):
                    pass
                try:
                    process.wait(timeout=2.0)
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                    pass

        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=0.5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, RuntimeError, ValueError):
                    pass
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=0.5)


class V4L2YUYVSource(V4L2Source):
    pixel_format = "YUYV"


class V4L2Z16Source(V4L2Source):
    pixel_format = "Z16"


__all__: Sequence[str] = (
    "V4L2Source",
    "V4L2YUYVSource",
    "V4L2Z16Source",
    "parse_v4l2_format",
    "read_exact_with_timeout",
)
