import sys
from types import ModuleType, SimpleNamespace

import pytest

from rlinf_deploy.cameras import (
    CameraRig,
    RGBCameraSource,
    opencv_camera,
    realsense_camera,
)
from rlinf_deploy.inference import ImagePayload


class FakeRGBDevice:
    def __init__(self, *, connect_error=None) -> None:
        self.connect_error = connect_error
        self.close_count = 0
        self.timeout_ms = None

    def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error

    def read(self, timeout_ms):
        self.timeout_ms = timeout_ms
        return SimpleNamespace(shape=(2, 3, 3))

    def close(self) -> None:
        self.close_count += 1


def source(name="camera-0", camera=None):
    return RGBCameraSource(
        name,
        camera or FakeRGBDevice(),
        timeout_ms=250,
        jpeg_quality=80,
        encoder=lambda frame, quality: f"{frame.shape}:{quality}".encode(),
    )


def test_generic_rgb_source_captures_an_encoded_image() -> None:
    camera = FakeRGBDevice()
    image_source = source(camera=camera)
    image_source.connect()
    image = image_source.capture()
    assert (image.name, image.mime_type, image.data) == (
        "camera-0",
        "image/jpeg",
        b"(2, 3, 3):80",
    )
    assert camera.timeout_ms == 250
    image_source.close()
    assert camera.close_count == 1


def test_partial_source_connection_is_cleaned_up() -> None:
    camera = FakeRGBDevice(connect_error=RuntimeError("camera unavailable"))
    with pytest.raises(RuntimeError, match="camera unavailable"):
        source(camera=camera).connect()
    assert camera.close_count == 1


def test_camera_rig_captures_in_declared_order_and_closes_in_reverse() -> None:
    events = []

    class Source:
        def __init__(self, name):
            self.name = name

        def connect(self):
            events.append(f"connect:{self.name}")

        def capture(self):
            events.append(f"capture:{self.name}")
            return ImagePayload(self.name, "image/jpeg", self.name.encode())

        def close(self):
            events.append(f"close:{self.name}")

    left, wrist = Source("left"), Source("wrist")
    with CameraRig((left, wrist)) as rig:
        assert tuple(image.name for image in rig.capture()) == ("left", "wrist")
    assert events == [
        "connect:left",
        "connect:wrist",
        "capture:left",
        "capture:wrist",
        "close:wrist",
        "close:left",
    ]


def test_camera_rig_rejects_duplicate_names_and_capture_before_connect() -> None:
    duplicate = source()
    with pytest.raises(ValueError, match="unique"):
        CameraRig((duplicate, duplicate))
    with pytest.raises(RuntimeError, match="not connected"):
        CameraRig((duplicate,)).capture()


@pytest.mark.parametrize(
    ("module_name", "config_name", "camera_name", "factory", "backend_value"),
    (
        (
            "lerobot.cameras.opencv",
            "OpenCVCameraConfig",
            "OpenCVCamera",
            opencv_camera,
            2,
        ),
        (
            "lerobot.cameras.realsense",
            "RealSenseCameraConfig",
            "RealSenseCamera",
            realsense_camera,
            "123456",
        ),
    ),
)
def test_optional_camera_factories_are_lazy_and_configure_rgb_only(
    monkeypatch, module_name, config_name, camera_name, factory, backend_value
) -> None:
    captured = {}
    module = ModuleType(module_name)

    class Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class Camera:
        def __init__(self, config):
            self.is_connected = False

        def connect(self, warmup=True):
            self.is_connected = True

        def async_read(self, timeout_ms=200):
            return SimpleNamespace(shape=(2, 3, 3))

        def disconnect(self):
            self.is_connected = False

    setattr(module, config_name, Config)
    setattr(module, camera_name, Camera)
    monkeypatch.setitem(sys.modules, module_name, module)

    image_source = factory("camera-0", backend_value, fps=30, width=640, height=480)
    assert isinstance(image_source, RGBCameraSource)
    assert captured["color_mode"] == "rgb"
    assert (captured["fps"], captured["width"], captured["height"]) == (
        30,
        640,
        480,
    )
    if factory is realsense_camera:
        assert captured["use_rgb"] is True
        assert captured["use_depth"] is False
