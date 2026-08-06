from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple  # noqa: UP035

import pytest

from embodied_runtime.robots.go2.agent.camera.server import CameraHTTPServer
from embodied_runtime.robots.go2.agent.camera.stores import DepthStore, FrameStore
from embodied_runtime.robots.go2.agent.camera.types import VideoProfile

PNG_FIXTURE = b"\x89PNG\r\n\x1a\nfixture-depth"
JPEG_FIXTURE = b"\xff\xd8fixture-rgb\xff\xd9"
TOKEN = "camera-token-" + "x" * 32


class MutableClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@contextmanager
def running(server: CameraHTTPServer) -> Iterator[str]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def install_pair(
    server: CameraHTTPServer,
    clock: MutableClock,
    *,
    frame_sequence: int = 1,
    depth_sequence: int = 1,
    include_depth_png: bool = True,
) -> DepthStore:
    rgb_profile = VideoProfile(640, 360, "BGR8", 1920, 691200, fps=15)
    depth_profile = VideoProfile(640, 360, "Z16", 1280, 460800, fps=15)
    server.frame_store.set_source_profile("realsense:FAKE", rgb_profile)
    server.frame_store.set_capture_running(True)
    server.frame_store.update(
        JPEG_FIXTURE,
        sequence=frame_sequence,
        captured_at_unix=1234.5,
        captured_at_monotonic=clock.value,
        source_frame_number=9,
        source_timestamp_ms=10.5,
    )
    depth_store = DepthStore(
        640,
        360,
        clock=clock,
        wall_clock=lambda: 1234.5,
        device="realsense:FAKE",
        backend="realsense",
    )
    server.depth_store = depth_store
    server.backend = "realsense"
    server.calibration_confirmed = True
    depth_store.set_source_profile("realsense:FAKE", depth_profile)
    depth_store.set_registration(True, True, 0.001)
    depth_store.set_capture_running(True)
    depth_store.update(
        1.25,
        0.8,
        0.9,
        sequence=depth_sequence,
        captured_at_unix=1234.5,
        captured_at_monotonic=clock.value,
        source_frame_number=9,
        source_timestamp_ms=10.5,
        depth_png=PNG_FIXTURE if include_depth_png else None,
    )
    return depth_store


def request_json(
    url: str,
    *,
    token: Optional[str] = None,  # noqa: UP045
) -> Dict[str, object]:  # noqa: UP006
    request = urllib.request.Request(url)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=2.0) as response:
        return json.loads(response.read())


def error_json(url: str) -> Tuple[int, Dict[str, object]]:  # noqa: UP006
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(url, timeout=2.0)
    error = caught.value
    try:
        return error.code, json.loads(error.read())
    finally:
        error.close()


def test_all_five_endpoints_serve_one_fresh_matched_observation() -> None:
    clock = MutableClock()
    store = FrameStore(clock=clock, wall_clock=lambda: 1234.5)
    server = CameraHTTPServer(("127.0.0.1", 0), store, max_frame_age=2.0)
    install_pair(server, clock)

    with running(server) as base_url:
        health = request_json(f"{base_url}/health")
        assert health["navigation_ready"] is True
        with urllib.request.urlopen(f"{base_url}/frame.jpg", timeout=2.0) as response:
            assert response.headers["X-Frame-Sequence"] == "1"
            assert response.read() == JPEG_FIXTURE
        depth = request_json(f"{base_url}/depth.json")
        assert depth["pair_sequence_matched"] is True
        with urllib.request.urlopen(f"{base_url}/depth.png", timeout=2.0) as response:
            assert response.headers["X-Depth-Encoding"] == "uint16"
            assert float(response.headers["X-Depth-Scale-M"]) == 0.001
            assert response.read() == PNG_FIXTURE
        observation = request_json(f"{base_url}/observation.json")
        assert observation["sequence"] == 1
        assert observation["rgb"]["sequence"] == 1
        assert observation["depth"]["sequence"] == 1
        assert observation["depth"]["registered_to_rgb"] is True
        assert base64.b64decode(observation["rgb"]["data_url"].partition(",")[2]) == JPEG_FIXTURE
        assert base64.b64decode(observation["depth"]["data_url"].partition(",")[2]) == PNG_FIXTURE


def test_freshness_and_sequence_mismatch_fail_closed() -> None:
    clock = MutableClock()
    store = FrameStore(clock=clock, wall_clock=lambda: 1234.5)
    server = CameraHTTPServer(("127.0.0.1", 0), store, max_frame_age=2.0)
    install_pair(server, clock, depth_sequence=2)
    with running(server) as base_url:
        for endpoint in ("depth.json", "depth.png", "observation.json"):
            code, payload = error_json(f"{base_url}/{endpoint}")
            assert code == 503
            assert payload.get("navigation_ready") is not True

    clock = MutableClock()
    store = FrameStore(clock=clock, wall_clock=lambda: 1234.5)
    server = CameraHTTPServer(("127.0.0.1", 0), store, max_frame_age=2.0)
    install_pair(server, clock)
    clock.value += 2.1
    with running(server) as base_url:
        for endpoint in ("health", "frame.jpg", "depth.json", "depth.png", "observation.json"):
            code, _payload = error_json(f"{base_url}/{endpoint}")
            assert code == 503


def test_bearer_token_protects_every_endpoint() -> None:
    server = CameraHTTPServer(
        ("127.0.0.1", 0),
        FrameStore(),
        max_frame_age=1.0,
        api_token=TOKEN,
    )
    with running(server) as base_url:
        for endpoint in ("health", "frame.jpg", "depth.json", "depth.png", "observation.json"):
            code, payload = error_json(f"{base_url}/{endpoint}")
            assert code == 401
            assert payload == {"error": "unauthorized"}

        request = urllib.request.Request(f"{base_url}/health")
        request.add_header("Authorization", f"Bearer {TOKEN}")
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=2.0)
        assert caught.value.code == 503


def test_external_http_server_cannot_be_constructed_without_token() -> None:
    with pytest.raises(ValueError, match="required"):
        CameraHTTPServer(("0.0.0.0", 0), FrameStore(), max_frame_age=1.0)
