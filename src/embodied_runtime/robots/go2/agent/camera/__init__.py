"""Dog-side RGB/RGB-D capture service for Unitree Go2 deployments.

The public package intentionally exposes the dependency-free contracts and
service assembly. OpenCV, NumPy, and pyrealsense2 remain lazy hardware-path
imports, so importing this module is safe on deployment controllers.
"""

from .cli import CameraServiceRuntime, build_runtime, main
from .config import CameraServiceConfig, parse_args, validate_bind_token
from .encoding import analyze_z16_depth, encode_z16_as_png
from .realsense import RealSenseWorker
from .server import CameraHTTPServer
from .stores import DepthStore, FrameStore
from .types import (
    CameraStreamError,
    DepthAnalysis,
    DepthSnapshot,
    FrameSnapshot,
    V4L2Format,
    VideoProfile,
)
from .v4l2 import (
    V4L2YUYVSource,
    V4L2Z16Source,
    parse_v4l2_format,
    read_exact_with_timeout,
)
from .workers import CameraWorker, DepthWorker

__all__ = [
    "CameraHTTPServer",
    "CameraServiceConfig",
    "CameraServiceRuntime",
    "CameraStreamError",
    "CameraWorker",
    "DepthAnalysis",
    "DepthSnapshot",
    "DepthStore",
    "DepthWorker",
    "FrameSnapshot",
    "FrameStore",
    "RealSenseWorker",
    "V4L2Format",
    "V4L2YUYVSource",
    "V4L2Z16Source",
    "VideoProfile",
    "analyze_z16_depth",
    "build_runtime",
    "encode_z16_as_png",
    "main",
    "parse_args",
    "parse_v4l2_format",
    "read_exact_with_timeout",
    "validate_bind_token",
]
