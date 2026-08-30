"""Capture live camera images and execute one action on an SO-101 follower."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlinf_deploy.bindings.lerobot.so101.pi05 import Pi05SO101Runtime
from rlinf_deploy.cameras import CameraRig, opencv_camera, realsense_camera
from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter, SO101Config


def _name_and_value(value: str) -> tuple[str, str]:
    name, separator, backend_value = value.partition("=")
    if not separator or not name or not backend_value:
        raise argparse.ArgumentTypeError("expected NAME=DEVICE")
    return name, backend_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--vvla-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--opencv-camera",
        action="append",
        type=_name_and_value,
        default=[],
        metavar="NAME=DEVICE",
    )
    parser.add_argument(
        "--realsense-camera",
        action="append",
        type=_name_and_value,
        default=[],
        metavar="NAME=SERIAL",
    )
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--token")
    args = parser.parse_args()

    camera_dimensions = (args.camera_fps, args.camera_width, args.camera_height)
    if (
        args.realsense_camera
        and any(value is not None for value in camera_dimensions)
        and any(value is None for value in camera_dimensions)
    ):
        parser.error(
            "RealSense requires --camera-fps, --camera-width, and "
            "--camera-height together"
        )
    camera_profile = {
        "fps": args.camera_fps,
        "width": args.camera_width,
        "height": args.camera_height,
    }
    sources = tuple(
        opencv_camera(
            name,
            int(device) if device.isdigit() else Path(device),
            **camera_profile,
        )
        for name, device in args.opencv_camera
    ) + tuple(
        realsense_camera(name, serial, **camera_profile)
        for name, serial in args.realsense_camera
    )
    if not sources:
        parser.error("provide at least one --opencv-camera or --realsense-camera")

    robot = SO101Adapter(
        SO101Config(
            port=args.robot_port,
            robot_id=args.robot_id,
            calibration_id=args.robot_id,
            calibration_dir=args.calibration_dir,
        )
    )
    try:
        client = VvlaHttpClient(
            args.vvla_url, token=args.token, timeout_s=args.timeout_s
        )
        controller = Pi05SO101Runtime(robot, client, instruction=args.instruction)
        try:
            with CameraRig(sources) as cameras:
                result = controller.step(cameras.capture())
            print(
                f"executed {len(result.actions)} action(s), "
                f"revision={result.session_revision}"
            )
        finally:
            controller.close()
    finally:
        robot.close()


if __name__ == "__main__":
    main()
