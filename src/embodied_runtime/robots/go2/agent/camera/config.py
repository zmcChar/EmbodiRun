"""Configuration and CLI/environment parsing for the Go2 camera agent."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple  # noqa: UP035

ENV_PREFIX = "GO2_CAMERA_"
_PROFILE_PATTERN = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)@(?P<fps>\d+)$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def port_number(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    return parsed


def camera_profile(value: str) -> Tuple[int, int, int]:  # noqa: UP006
    """Parse a compact ``WIDTHxHEIGHT@FPS`` profile."""

    match = _PROFILE_PATTERN.fullmatch(value.strip().lower())
    if match is None:
        raise argparse.ArgumentTypeError("must use WIDTHxHEIGHT@FPS, for example 640x360@15")
    width = int(match.group("width"))
    height = int(match.group("height"))
    fps = int(match.group("fps"))
    if width <= 0 or height <= 0 or fps <= 0:
        raise argparse.ArgumentTypeError("profile dimensions and FPS must be positive")
    return width, height, fps


def _environment_bool(
    environment: Mapping[str, str],
    name: str,
    default: bool = False,
) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, yes/no, on/off, or 1/0")


def is_loopback_bind(host: str) -> bool:
    candidate = host.strip().lower()
    if candidate == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def validate_bind_token(host: str, token: Optional[str]) -> None:  # noqa: UP045
    """Require a strong bearer token when exposing the service off-host."""

    if token is not None:
        try:
            token.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("camera token must contain only ASCII characters") from error
        if len(token) < 32:
            raise ValueError("camera token must contain at least 32 characters")
        if token.strip() != token or any(character.isspace() for character in token):
            raise ValueError("camera token must not contain whitespace")
    if not is_loopback_bind(host) and token is None:
        raise ValueError("camera token is required when binding beyond localhost")


@dataclass(frozen=True)
class CameraServiceConfig:
    backend: str
    host: str
    port: int
    token: Optional[str]  # noqa: UP045
    serial: Optional[str]  # noqa: UP045
    width: int
    height: int
    fps: int
    device: str
    warmup_frames: int
    jpeg_quality: int
    jpeg_fps: float
    retry_delay: float
    frame_timeout: float
    format_timeout: float
    max_frame_age: float
    depth_device: str
    depth_width: int
    depth_height: int
    depth_warmup_frames: int
    depth_scale: Optional[float]  # noqa: UP045
    depth_calibrated: bool
    rgb_depth_alignment_claimed: bool
    depth_fps: float
    max_depth_age: float
    max_depth_m: float
    access_log: bool

    @property
    def profile(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"

    @property
    def initial_device(self) -> str:
        if self.backend != "realsense":
            return self.device
        return "realsense" if not self.serial else f"realsense:{self.serial}"

    def validate(self) -> None:
        if self.backend not in {"v4l2", "realsense"}:
            raise ValueError("backend must be v4l2 or realsense")
        if not self.host.strip():
            raise ValueError("host must not be empty")
        validate_bind_token(self.host, self.token)
        if self.width % 2:
            raise ValueError("width must be even for packed YUYV diagnostics")
        if self.warmup_frames < 0 or self.depth_warmup_frames < 0:
            raise ValueError("warmup frame counts must be non-negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg quality must be between 1 and 100")
        if self.depth_calibrated and self.depth_scale is None:
            raise ValueError("depth calibration requires an explicit depth scale")
        if self.depth_scale is not None and not self.depth_calibrated:
            raise ValueError("depth scale requires explicit depth calibration")
        if self.rgb_depth_alignment_claimed and not self.depth_calibrated:
            raise ValueError("RGB-depth alignment claim requires depth calibration")
        if self.backend == "realsense" and not self.depth_calibrated:
            raise ValueError("realsense backend requires explicit depth calibration")
        if self.backend == "realsense" and (
            self.width != self.depth_width or self.height != self.depth_height
        ):
            raise ValueError("realsense backend requires matching RGB/depth dimensions")


def _env(
    environment: Mapping[str, str],
    name: str,
    default: Optional[str] = None,  # noqa: UP045
) -> Optional[str]:  # noqa: UP045
    return environment.get(f"{ENV_PREFIX}{name}", default)


def _add_boolean_option(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: Optional[str] = None,  # noqa: UP045
) -> None:
    """Add ``--flag``/``--no-flag`` without Python 3.9-only argparse APIs."""

    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}",
        dest=destination,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(**{destination: default})


def parse_args(
    argv: Optional[Sequence[str]] = None,  # noqa: UP045
    *,
    environ: Optional[Mapping[str, str]] = None,  # noqa: UP045
) -> CameraServiceConfig:
    """Resolve CLI values over ``GO2_CAMERA_*`` environment defaults."""

    environment = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        description=(
            "Serve fresh Go2 RGB-D frames. The RealSense backend uses one "
            "pipeline and aligns depth to color; split V4L2 is diagnostic-only."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("v4l2", "realsense"),
        default=_env(environment, "BACKEND", "v4l2"),
    )
    parser.add_argument(
        "--host",
        "--bind",
        dest="host",
        default=_env(environment, "HOST", "127.0.0.1"),
        help="HTTP bind host (GO2_CAMERA_HOST)",
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=_env(environment, "PORT", "8765"),
        help="HTTP port (GO2_CAMERA_PORT)",
    )
    parser.add_argument(
        "--token",
        default=_env(environment, "TOKEN"),
        help="bearer token (prefer GO2_CAMERA_TOKEN to avoid process-list exposure)",
    )
    parser.add_argument(
        "--serial",
        "--realsense-serial",
        dest="serial",
        default=_env(environment, "SERIAL"),
        help="optional RealSense serial (GO2_CAMERA_SERIAL)",
    )
    parser.add_argument(
        "--profile",
        type=camera_profile,
        default=_env(environment, "PROFILE", "640x360@15"),
        metavar="WIDTHxHEIGHT@FPS",
        help="requested RGB-D profile (GO2_CAMERA_PROFILE)",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=_env(environment, "WIDTH"),
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=_env(environment, "HEIGHT"),
    )
    parser.add_argument(
        "--fps",
        "--realsense-fps",
        dest="fps",
        type=positive_int,
        default=_env(environment, "FPS"),
        help="capture FPS override (GO2_CAMERA_FPS)",
    )
    parser.add_argument(
        "--depth-scale",
        type=positive_float,
        default=_env(environment, "DEPTH_SCALE"),
        help="confirmed meters per Z16 unit (GO2_CAMERA_DEPTH_SCALE)",
    )
    try:
        depth_calibrated_default = _environment_bool(
            environment,
            f"{ENV_PREFIX}DEPTH_CALIBRATED",
        )
        aligned_default = _environment_bool(
            environment,
            f"{ENV_PREFIX}RGB_DEPTH_ALIGNED",
        )
        access_log_default = _environment_bool(
            environment,
            f"{ENV_PREFIX}ACCESS_LOG",
        )
    except ValueError as error:
        parser.error(str(error))
    _add_boolean_option(
        parser,
        "depth-calibrated",
        default=depth_calibrated_default,
    )
    _add_boolean_option(
        parser,
        "rgb-depth-aligned",
        default=aligned_default,
        help_text=("diagnostic operator claim; RealSense alignment is verified automatically"),
    )
    _add_boolean_option(
        parser,
        "access-log",
        default=access_log_default,
    )
    parser.add_argument(
        "--device",
        default=_env(environment, "DEVICE", "/dev/video4"),
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=_env(environment, "WARMUP_FRAMES", "10"),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=_env(environment, "JPEG_QUALITY", "85"),
    )
    parser.add_argument(
        "--jpeg-fps",
        type=positive_float,
        default=_env(environment, "JPEG_FPS", "5"),
    )
    parser.add_argument(
        "--retry-delay",
        type=positive_float,
        default=_env(environment, "RETRY_DELAY", "2"),
    )
    parser.add_argument(
        "--frame-timeout",
        type=positive_float,
        default=_env(environment, "FRAME_TIMEOUT", "2"),
    )
    parser.add_argument(
        "--format-timeout",
        type=positive_float,
        default=_env(environment, "FORMAT_TIMEOUT", "5"),
    )
    parser.add_argument(
        "--max-frame-age",
        type=positive_float,
        default=_env(environment, "MAX_FRAME_AGE", "2"),
    )
    parser.add_argument(
        "--depth-device",
        default=_env(environment, "DEPTH_DEVICE", "/dev/video0"),
    )
    parser.add_argument(
        "--depth-width",
        type=positive_int,
        default=_env(environment, "DEPTH_WIDTH"),
    )
    parser.add_argument(
        "--depth-height",
        type=positive_int,
        default=_env(environment, "DEPTH_HEIGHT"),
    )
    parser.add_argument(
        "--depth-warmup-frames",
        type=int,
        default=_env(environment, "DEPTH_WARMUP_FRAMES", "10"),
    )
    parser.add_argument(
        "--depth-fps",
        type=positive_float,
        default=_env(environment, "DEPTH_FPS", "10"),
    )
    parser.add_argument(
        "--max-depth-age",
        type=positive_float,
        default=_env(environment, "MAX_DEPTH_AGE", "0.5"),
    )
    parser.add_argument(
        "--max-depth-m",
        type=positive_float,
        default=_env(environment, "MAX_DEPTH_M", "10"),
    )
    namespace = parser.parse_args(argv)

    profile_width, profile_height, profile_fps = namespace.profile
    width = profile_width if namespace.width is None else namespace.width
    height = profile_height if namespace.height is None else namespace.height
    fps = profile_fps if namespace.fps is None else namespace.fps
    depth_width = width if namespace.depth_width is None else namespace.depth_width
    depth_height = height if namespace.depth_height is None else namespace.depth_height
    config = CameraServiceConfig(
        backend=namespace.backend,
        host=namespace.host,
        port=namespace.port,
        token=namespace.token,
        serial=namespace.serial,
        width=width,
        height=height,
        fps=fps,
        device=namespace.device,
        warmup_frames=namespace.warmup_frames,
        jpeg_quality=namespace.jpeg_quality,
        jpeg_fps=namespace.jpeg_fps,
        retry_delay=namespace.retry_delay,
        frame_timeout=namespace.frame_timeout,
        format_timeout=namespace.format_timeout,
        max_frame_age=namespace.max_frame_age,
        depth_device=namespace.depth_device,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_warmup_frames=namespace.depth_warmup_frames,
        depth_scale=namespace.depth_scale,
        depth_calibrated=namespace.depth_calibrated,
        rgb_depth_alignment_claimed=namespace.rgb_depth_aligned,
        depth_fps=namespace.depth_fps,
        max_depth_age=namespace.max_depth_age,
        max_depth_m=namespace.max_depth_m,
        access_log=namespace.access_log,
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    return config


__all__ = [
    "CameraServiceConfig",
    "camera_profile",
    "is_loopback_bind",
    "parse_args",
    "validate_bind_token",
]
