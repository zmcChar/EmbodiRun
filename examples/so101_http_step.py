"""Execute one VVLA action on a calibrated LeRobot SO-101 follower."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlinf_deploy.bindings.lerobot.so101.pi05 import Pi05SO101Runtime
from rlinf_deploy.inference import ImagePayload, VvlaHttpClient
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter, SO101Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--vvla-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--token")
    args = parser.parse_args()

    images = tuple(
        ImagePayload(
            name=f"camera-{index}",
            mime_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg",
            data=path.read_bytes(),
        )
        for index, path in enumerate(args.image)
    )
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
            result = controller.step(images)
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
