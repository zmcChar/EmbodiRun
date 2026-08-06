from __future__ import annotations

import base64
import struct
import threading
import time
from typing import List, Optional, Tuple  # noqa: UP035
from unittest import mock

import pytest

from embodied_runtime.robots.go2.agent.camera.realsense import RealSenseWorker
from embodied_runtime.robots.go2.agent.camera.server import CameraHTTPServer
from embodied_runtime.robots.go2.agent.camera.stores import DepthStore, FrameStore
from embodied_runtime.robots.go2.agent.camera.types import CameraStreamError, DepthAnalysis

PNG_FIXTURE = b"\x89PNG\r\n\x1a\nfixture-depth"
JPEG_FIXTURE = b"\xff\xd8fixture-rgb\xff\xd9"


class FakeRSVideoProfile:
    def __init__(self, width: int, height: int, fps: int, pixel_format: str) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._pixel_format = pixel_format

    def as_video_stream_profile(self) -> FakeRSVideoProfile:
        return self

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def fps(self) -> int:
        return self._fps

    def format(self) -> str:
        return self._pixel_format


class FakeRSDevice:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def first_depth_sensor(self) -> FakeRSDevice:
        return self

    def get_depth_scale(self) -> float:
        return self.scale

    def get_info(self, _key: object) -> str:
        return "FAKE123"


class FakeRSPipelineProfile:
    def __init__(self, module: FakeRSModule) -> None:
        self.module = module
        self.device = FakeRSDevice(module.scale)

    def get_stream(self, stream: str) -> FakeRSVideoProfile:
        pixel_format = (
            self.module.format.bgr8
            if stream == self.module.stream.color
            else self.module.format.z16
        )
        return FakeRSVideoProfile(
            self.module.width,
            self.module.height,
            self.module.fps,
            pixel_format,
        )

    def get_device(self) -> FakeRSDevice:
        return self.device


class FakeRSFrame:
    def __init__(self, width: int, height: int, bytes_per_pixel: int, data: bytes) -> None:
        self.width = width
        self.height = height
        self.bytes_per_pixel = bytes_per_pixel
        self.data = data

    def __bool__(self) -> bool:
        return True

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_stride_in_bytes(self) -> int:
        return self.width * self.bytes_per_pixel

    def get_data(self) -> bytes:
        return self.data

    def get_frame_number(self) -> int:
        return 1

    def get_timestamp(self) -> float:
        return 123.5


class FakeRSFrames:
    def __init__(self, color: FakeRSFrame, depth: FakeRSFrame) -> None:
        self.color = color
        self.depth = depth

    def get_color_frame(self) -> FakeRSFrame:
        return self.color

    def get_depth_frame(self) -> FakeRSFrame:
        return self.depth


class FakeRSConfig:
    def __init__(self) -> None:
        self.streams: List[Tuple[object, ...]] = []  # noqa: UP006
        self.serial: Optional[str] = None  # noqa: UP045

    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, *args: object) -> None:
        self.streams.append(args)


class FakeRSPipeline:
    def __init__(self, module: FakeRSModule) -> None:
        self.module = module
        self.stopped = threading.Event()
        self.frame_delivered = False
        self.stop_calls = 0

    def start(self, config: FakeRSConfig) -> FakeRSPipelineProfile:
        self.module.last_config = config
        return FakeRSPipelineProfile(self.module)

    def wait_for_frames(self, timeout_ms: int) -> FakeRSFrames:
        self.module.last_timeout_ms = timeout_ms
        if not self.frame_delivered:
            self.frame_delivered = True
            color = FakeRSFrame(
                self.module.width,
                self.module.height,
                3,
                b"\x10" * (self.module.width * self.module.height * 3),
            )
            depth = FakeRSFrame(
                self.module.width,
                self.module.height,
                2,
                struct.pack(
                    f"<{self.module.width * self.module.height}H",
                    *([1000] * (self.module.width * self.module.height)),
                ),
            )
            return FakeRSFrames(color, depth)
        self.stopped.wait(timeout=2.0)
        raise RuntimeError("pipeline stopped")

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()


class FakeRSAlign:
    def __init__(self, module: FakeRSModule) -> None:
        self.module = module

    def process(self, frames: FakeRSFrames) -> FakeRSFrames:
        self.module.align_process_count += 1
        return frames


class FakeRSModule:
    class stream:
        color = "color"
        depth = "depth"

    class format:
        bgr8 = "bgr8"
        z16 = "z16"

    class camera_info:
        serial_number = "serial_number"

    def __init__(
        self,
        width: int = 4,
        height: int = 4,
        fps: int = 15,
        scale: float = 0.001,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.scale = scale
        self.align_process_count = 0
        self.last_config: Optional[FakeRSConfig] = None  # noqa: UP045
        self.last_timeout_ms: Optional[int] = None  # noqa: UP045
        self.pipeline_instance = FakeRSPipeline(self)

    def pipeline(self) -> FakeRSPipeline:
        return self.pipeline_instance

    def config(self) -> FakeRSConfig:
        return FakeRSConfig()

    def align(self, stream: str) -> FakeRSAlign:
        assert stream == self.stream.color
        return FakeRSAlign(self)


def make_worker(
    fake_rs: FakeRSModule,
    frame_store: FrameStore,
    depth_store: DepthStore,
) -> RealSenseWorker:
    return RealSenseWorker(
        frame_store=frame_store,
        depth_store=depth_store,
        width=4,
        height=4,
        fps=15,
        expected_depth_scale=0.001,
        jpeg_quality=80,
        jpeg_fps=15.0,
        frame_timeout=0.2,
        retry_delay=0.1,
        encoder=lambda _raw, _width, _height, _quality: JPEG_FIXTURE,
        depth_encoder=lambda raw, _width, _height: PNG_FIXTURE + raw,
        analyzer=lambda *_args: DepthAnalysis(1.0, 1.0, 1.0),
        rs_module=fake_rs,
    )


def test_single_pipeline_aligns_and_publishes_one_atomic_rgbd_pair() -> None:
    fake_rs = FakeRSModule()
    frame_store = FrameStore(device="realsense")
    depth_store = DepthStore(4, 4, device="realsense", backend="realsense")
    worker = make_worker(fake_rs, frame_store, depth_store)
    worker.start()
    deadline = time.monotonic() + 2.0
    while frame_store.latest() is None and time.monotonic() < deadline:
        time.sleep(0.01)

    frame = frame_store.latest()
    depth = depth_store.latest()
    assert frame is not None
    assert depth is not None
    assert frame.sequence == depth.sequence == 1
    assert frame.captured_at_unix == depth.captured_at_unix
    assert frame.source_frame_number == depth.source_frame_number == 1
    status = depth_store.status(1.0)
    assert status["registration_verified"] is True
    assert status["rgb_depth_aligned"] is True
    assert status["depth_scale_m"] == 0.001
    assert fake_rs.align_process_count == 1
    assert fake_rs.last_timeout_ms == 200
    assert fake_rs.last_config is not None
    assert len(fake_rs.last_config.streams) == 2

    server = CameraHTTPServer(
        ("127.0.0.1", 0),
        frame_store,
        max_frame_age=1.0,
        depth_store=depth_store,
        max_depth_age=1.0,
        calibration_confirmed=True,
        backend="realsense",
        depth_scale_m=0.001,
    )
    try:
        metadata = server.depth_payload()
        assert metadata["navigation_ready"] is True
        assert metadata["pair_sequence_matched"] is True
        observation = server.observation_payload()
        assert observation["sequence"] == 1
        assert observation["depth"]["registered_to_rgb"] is True
        encoded_depth = observation["depth"]["data_url"].partition(",")[2]
        assert base64.b64decode(encoded_depth) == depth.depth_png
    finally:
        server.server_close()
        worker.stop()
        worker.join(timeout=2.0)
    assert fake_rs.pipeline_instance.stop_calls >= 1
    assert depth_store.status(1.0)["registration_verified"] is False


@pytest.mark.parametrize(
    ("fake_rs", "message"),
    [(FakeRSModule(scale=0.002), "depth scale"), (FakeRSModule(width=5), "profile")],
)
def test_scale_or_profile_mismatch_publishes_nothing(
    fake_rs: FakeRSModule,
    message: str,
) -> None:
    frame_store = FrameStore(device="realsense")
    depth_store = DepthStore(4, 4, device="realsense", backend="realsense")
    worker = make_worker(fake_rs, frame_store, depth_store)
    with pytest.raises(CameraStreamError, match=message):
        worker._run_once(fake_rs)
    assert frame_store.latest() is None
    assert depth_store.status(1.0)["registration_verified"] is False
    assert fake_rs.pipeline_instance.stop_calls >= 1


def test_pyrealsense_is_loaded_only_when_backend_runs() -> None:
    worker = make_worker(FakeRSModule(), FrameStore(), DepthStore(4, 4))
    worker._rs_module = None
    with (
        mock.patch(
            "embodied_runtime.robots.go2.agent.camera.realsense.importlib.import_module",
            side_effect=ImportError("missing"),
        ),
        pytest.raises(CameraStreamError, match="pyrealsense2"),
    ):
        worker._load_rs()
