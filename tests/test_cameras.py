from types import SimpleNamespace

from rlinf_deploy.robots.sensors.cameras import (
    CameraFrame,
    V4L2CameraConfig,
    V4L2CameraSource,
)


def test_viewer_reader_stops_with_an_open_or_partial_input_pipe() -> None:
    import os
    from queue import Queue
    from threading import Event, Thread

    from rlinf_deploy.simulators.viewer import _read_frames

    for partial in (b"", b"\x00\x00", b"\x00\x00\x00\x04xy"):
        read_fd, write_fd = os.pipe()
        with (
            os.fdopen(read_fd, "rb") as stream,
            os.fdopen(write_fd, "wb", buffering=0) as writer,
        ):
            frames = Queue(maxsize=1)
            frames.put(b"previous-frame")
            stopped = Event()
            reader = Thread(target=_read_frames, args=(stream, frames, stopped))
            reader.start()
            writer.write(partial)
            stopped.set()
            reader.join(timeout=2)
            assert not reader.is_alive()
            assert frames.get_nowait() is None


class FakeCapture:
    def __init__(self) -> None:
        self.settings = []
        self.grabs = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, name, value):
        self.settings.append((name, value))

    def grab(self):
        self.grabs += 1
        return True

    def retrieve(self):
        return True, SimpleNamespace(shape=(480, 640, 3))

    def release(self):
        self.released = True


class FakeCv2:
    CAP_V4L2 = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5
    IMWRITE_JPEG_QUALITY = 6

    def __init__(self) -> None:
        self.capture = FakeCapture()

    def VideoCapture(self, device, backend):
        assert device == "/dev/video0"
        assert backend == self.CAP_V4L2
        return self.capture

    def imencode(self, extension, frame, options):
        assert extension == ".jpg"
        assert frame.shape == (480, 640, 3)
        assert options == [self.IMWRITE_JPEG_QUALITY, 90]
        return True, SimpleNamespace(tobytes=lambda: b"jpeg")


def test_v4l2_camera_produces_neutral_frame_and_releases_device() -> None:
    cv2 = FakeCv2()
    source = V4L2CameraSource(
        (
            V4L2CameraConfig(
                name="observation.images.front",
                device="/dev/video0",
                width=640,
                height=480,
                fps=30.0,
            ),
        ),
        cv2_module=cv2,
    )

    frames = source.capture()
    source.close()

    assert frames == (CameraFrame("observation.images.front", "image/jpeg", b"jpeg"),)
    assert cv2.capture.grabs == 4
    assert cv2.capture.released is True
