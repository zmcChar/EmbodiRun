"""Read-only HTTP API for fresh, matched Go2 camera observations."""

from __future__ import annotations

import base64
import hmac
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple  # noqa: UP035
from urllib.parse import urlsplit

from .config import validate_bind_token
from .stores import DepthStore, FrameStore
from .types import CameraStreamError, DepthSnapshot, FrameSnapshot


class CameraHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],  # noqa: UP006
        store: FrameStore,
        max_frame_age: float,
        depth_store: Optional[DepthStore] = None,  # noqa: UP045
        max_depth_age: float = 0.5,
        api_token: Optional[str] = None,  # noqa: UP045
        calibration_confirmed: bool = False,
        rgb_depth_alignment_claimed: bool = False,
        backend: str = "v4l2",
        registration_verified: bool = False,
        depth_scale_m: Optional[float] = None,  # noqa: UP045
        depth_device: Optional[str] = None,  # noqa: UP045
        access_log: bool = False,
    ) -> None:
        validate_bind_token(str(server_address[0]), api_token)
        self.frame_store = store
        self.max_frame_age = max_frame_age
        self.depth_store = depth_store
        self.max_depth_age = max_depth_age
        self.api_token = api_token
        self.calibration_confirmed = calibration_confirmed
        self.rgb_depth_alignment_claimed = rgb_depth_alignment_claimed
        self.backend = backend
        self.registration_verified = registration_verified
        self.depth_scale_m = depth_scale_m
        self.depth_device = depth_device
        self.access_log = access_log
        super().__init__(server_address, CameraRequestHandler)

    def depth_payload(self) -> Dict[str, object]:  # noqa: UP006
        rgb_health = self.frame_store.health(self.max_frame_age)
        if self.depth_store is None:
            depth: Dict[str, object] = {  # noqa: UP006
                "available": False,
                "center_distance_m": None,
                "minimum_distance_m": None,
                "captured_at_unix": None,
                "age_s": None,
                "width": None,
                "height": None,
                "frame_sequence": 0,
                "sequence": 0,
                "valid_fraction": 0.0,
                "raw_depth_available": False,
                "capture_running": False,
                "last_error": "depth capture is disabled",
                "depth_device": self.depth_device,
                "depth_profile": None,
                "source_timestamp_ms": None,
                "source_frame_number": None,
            }
        else:
            depth = self.depth_store.status(self.max_depth_age)

        backend = str(depth.get("backend", self.backend))
        registration_verified = bool(depth.get("registration_verified", self.registration_verified))
        rgb_depth_aligned = bool(depth.get("rgb_depth_aligned", False))
        if backend != "realsense" or not registration_verified:
            rgb_depth_aligned = False
        depth_scale_m = depth.get("depth_scale_m")
        if depth_scale_m is None:
            depth_scale_m = self.depth_scale_m
        depth_sequence = depth["frame_sequence"]
        rgb_sequence = rgb_health["frame_sequence"]
        pair_matched = bool(
            isinstance(depth_sequence, int)
            and not isinstance(depth_sequence, bool)
            and depth_sequence > 0
            and depth_sequence == rgb_sequence
        )

        navigation_ready = bool(
            depth["available"]
            and self.calibration_confirmed
            and rgb_depth_aligned
            and backend == "realsense"
            and registration_verified
            and pair_matched
            and rgb_health["status"] == "ok"
            and rgb_health["rgb_profile"] is not None
            and depth["depth_profile"] is not None
        )
        depth.update(
            {
                "calibration_confirmed": self.calibration_confirmed,
                "rgb_depth_aligned": rgb_depth_aligned,
                "rgb_depth_alignment_claimed": self.rgb_depth_alignment_claimed,
                "registration_verified": registration_verified,
                "backend": backend,
                "depth_scale_m": depth_scale_m,
                "rgb_device": rgb_health["rgb_device"],
                "rgb_profile": rgb_health["rgb_profile"],
                "pair_sequence_matched": pair_matched,
                "navigation_ready": navigation_ready,
                "observation_ready": bool(
                    navigation_ready and depth.get("raw_depth_available") is True
                ),
            }
        )
        return depth

    def matched_observation(
        self,
    ) -> Tuple[FrameSnapshot, DepthSnapshot, Dict[str, object]]:  # noqa: UP006
        """Read one immutable, fresh, navigation-ready RealSense pair."""

        if self.depth_store is None:
            raise CameraStreamError("aligned depth capture is disabled")

        # Reading both object slots again after metadata construction detects a
        # publish crossing the read without coupling HTTP readers to a lock.
        for _attempt in range(3):
            frame = self.frame_store.latest()
            depth = self.depth_store.latest()
            metadata = self.depth_payload()
            if frame is not self.frame_store.latest() or depth is not self.depth_store.latest():
                continue
            if frame is None or depth is None:
                raise CameraStreamError("no matched RGB-D observation is available")
            if metadata.get("navigation_ready") is not True:
                raise CameraStreamError("matched RGB-D observation is not navigation-ready")
            if depth.depth_png is None:
                raise CameraStreamError("raw aligned depth PNG is unavailable")
            if (
                isinstance(frame.sequence, bool)
                or not isinstance(frame.sequence, int)
                or frame.sequence <= 0
                or frame.sequence != depth.sequence
            ):
                raise CameraStreamError("RGB and depth sequences do not match")
            if metadata.get("sequence") != frame.sequence:
                raise CameraStreamError("RGB-D metadata sequence does not match the snapshot")
            capture_times = (
                frame.captured_at_unix,
                frame.captured_at_monotonic,
                depth.captured_at_unix,
                depth.captured_at_monotonic,
            )
            if not all(math.isfinite(value) for value in capture_times):
                raise CameraStreamError("RGB-D capture timestamps are invalid")
            if (
                frame.captured_at_unix != depth.captured_at_unix
                or frame.captured_at_monotonic != depth.captured_at_monotonic
            ):
                raise CameraStreamError("RGB and depth capture timestamps do not match")
            if (
                frame.source_frame_number is None
                or isinstance(frame.source_frame_number, bool)
                or not isinstance(frame.source_frame_number, int)
                or frame.source_frame_number < 0
                or frame.source_frame_number != depth.source_frame_number
            ):
                raise CameraStreamError("RGB and depth source frame numbers do not match")
            if (
                frame.source_timestamp_ms is None
                or not math.isfinite(frame.source_timestamp_ms)
                or frame.source_timestamp_ms != depth.source_timestamp_ms
            ):
                raise CameraStreamError("RGB and depth source timestamps do not match")
            if self.frame_store.frame_age(frame) > self.max_frame_age:
                raise CameraStreamError("matched RGB frame is stale")
            if self.depth_store.snapshot_age(depth) > self.max_depth_age:
                raise CameraStreamError("matched depth frame is stale")
            if not (frame.jpeg.startswith(b"\xff\xd8") and frame.jpeg.endswith(b"\xff\xd9")):
                raise CameraStreamError("matched RGB payload is not a complete JPEG")
            if not depth.depth_png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise CameraStreamError("matched depth payload is not a PNG")

            depth_scale_m = metadata.get("depth_scale_m")
            if (
                isinstance(depth_scale_m, bool)
                or not isinstance(depth_scale_m, (int, float))
                or not math.isfinite(float(depth_scale_m))
                or float(depth_scale_m) <= 0.0
            ):
                raise CameraStreamError("matched depth scale is invalid")

            rgb_profile = metadata.get("rgb_profile")
            depth_profile = metadata.get("depth_profile")
            if not isinstance(rgb_profile, dict) or not isinstance(depth_profile, dict):
                raise CameraStreamError("matched RGB-D profiles are unavailable")
            rgb_dimensions = (rgb_profile.get("width"), rgb_profile.get("height"))
            depth_dimensions = (
                depth_profile.get("width"),
                depth_profile.get("height"),
            )
            dimensions = (*rgb_dimensions, *depth_dimensions)
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in dimensions
                )
                or rgb_dimensions != depth_dimensions
                or depth_dimensions != (self.depth_store.width, self.depth_store.height)
            ):
                raise CameraStreamError("matched RGB-D dimensions do not agree")
            return frame, depth, metadata

        raise CameraStreamError("RGB-D observation changed while it was being read")

    def observation_payload(self) -> Dict[str, object]:  # noqa: UP006
        frame, depth, metadata = self.matched_observation()
        assert depth.depth_png is not None
        rgb_profile = metadata["rgb_profile"]
        depth_profile = metadata["depth_profile"]
        assert isinstance(rgb_profile, dict)
        assert isinstance(depth_profile, dict)
        sequence = frame.sequence
        return {
            "sequence": sequence,
            "captured_at_unix": frame.captured_at_unix,
            "source_frame_number": frame.source_frame_number,
            "source_timestamp_ms": frame.source_timestamp_ms,
            "pair_sequence_matched": True,
            "navigation_ready": True,
            "rgb": {
                "sequence": sequence,
                "media_type": "image/jpeg",
                "width": rgb_profile["width"],
                "height": rgb_profile["height"],
                "data_url": "data:image/jpeg;base64,{}".format(
                    base64.b64encode(frame.jpeg).decode("ascii")
                ),
            },
            "depth": {
                "sequence": sequence,
                "media_type": "image/png",
                "encoding": "uint16",
                "width": depth_profile["width"],
                "height": depth_profile["height"],
                "scale_m": metadata["depth_scale_m"],
                "registered_to_rgb": True,
                "center_distance_m": depth.center_distance_m,
                "minimum_distance_m": depth.minimum_distance_m,
                "valid_fraction": depth.valid_fraction,
                "data_url": "data:image/png;base64,{}".format(
                    base64.b64encode(depth.depth_png).decode("ascii")
                ),
            },
        }


class CameraRequestHandler(BaseHTTPRequestHandler):
    server_version = "Go2Camera/2"

    @property
    def camera_server(self) -> CameraHTTPServer:
        return self.server  # type: ignore[return-value]

    def _send_json(
        self,
        status: HTTPStatus,
        payload: Dict[str, object],  # noqa: UP006
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = self.camera_server.api_token
        if not token:
            return True
        expected = f"Bearer {token}"
        return hmac.compare_digest(self.headers.get("Authorization", ""), expected)

    def do_GET(self) -> None:
        server = self.camera_server
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        path = urlsplit(self.path).path
        if path == "/health":
            health = server.frame_store.health(server.max_frame_age)
            depth = server.depth_payload()
            health.update(
                {
                    "depth": depth,
                    "calibration_confirmed": depth["calibration_confirmed"],
                    "rgb_depth_aligned": depth["rgb_depth_aligned"],
                    "rgb_depth_alignment_claimed": depth["rgb_depth_alignment_claimed"],
                    "registration_verified": depth["registration_verified"],
                    "backend": depth["backend"],
                    "depth_scale_m": depth["depth_scale_m"],
                    "depth_device": depth["depth_device"],
                    "depth_profile": depth["depth_profile"],
                    "navigation_ready": depth["navigation_ready"],
                    "observation_ready": depth["observation_ready"],
                }
            )
            if not depth["navigation_ready"]:
                health["status"] = "degraded"
            status = HTTPStatus.OK if health["status"] == "ok" else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(status, health)
            return

        if path == "/frame.jpg":
            frame = server.frame_store.latest()
            if frame is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "no camera frame is available yet"},
                )
                return
            age = server.frame_store.frame_age(frame)
            if age > server.max_frame_age:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": "latest camera frame is stale",
                        "frame_age_s": round(age, 3),
                    },
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame.jpeg)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Frame-Sequence", str(frame.sequence))
            self.send_header("X-Frame-Age-Ms", str(int(age * 1000)))
            self.send_header("X-Captured-At-Unix", f"{frame.captured_at_unix:.6f}")
            if frame.source_frame_number is not None:
                self.send_header(
                    "X-Source-Frame-Number",
                    str(frame.source_frame_number),
                )
            if frame.source_timestamp_ms is not None:
                self.send_header(
                    "X-Source-Timestamp-Ms",
                    f"{frame.source_timestamp_ms:.3f}",
                )
            self.end_headers()
            self.wfile.write(frame.jpeg)
            return

        if path == "/depth.json":
            depth = server.depth_payload()
            status = HTTPStatus.OK if depth["navigation_ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(status, depth)
            return

        if path == "/depth.png":
            try:
                frame, depth, metadata = server.matched_observation()
            except CameraStreamError as error:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": str(error), "available": False},
                )
                return
            assert depth.depth_png is not None
            assert server.depth_store is not None
            age = server.depth_store.snapshot_age(depth)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(depth.depth_png)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Frame-Sequence", str(frame.sequence))
            self.send_header("X-Frame-Age-Ms", str(int(age * 1000)))
            self.send_header("X-Captured-At-Unix", f"{frame.captured_at_unix:.6f}")
            self.send_header("X-Depth-Encoding", "uint16")
            self.send_header("X-Depth-Scale-M", f"{metadata['depth_scale_m']:.12g}")
            self.send_header("X-Source-Frame-Number", str(frame.source_frame_number))
            self.send_header(
                "X-Source-Timestamp-Ms",
                f"{frame.source_timestamp_ms:.3f}",
            )
            self.end_headers()
            self.wfile.write(depth.depth_png)
            return

        if path == "/observation.json":
            try:
                observation = server.observation_payload()
            except CameraStreamError as error:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": str(error), "available": False},
                )
                return
            self._send_json(HTTPStatus.OK, observation)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, format_string: str, *args: Any) -> None:
        if self.camera_server.access_log:
            super().log_message(format_string, *args)


__all__ = ["CameraHTTPServer", "CameraRequestHandler"]
