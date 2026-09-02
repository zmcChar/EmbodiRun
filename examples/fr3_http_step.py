"""Execute one VVLA action chunk on a FR3 robot using encoded camera images."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlinf_deploy.bindings.franka.fr3.pi05 import Pi05FR3Runtime
from rlinf_deploy.inference import ImagePayload, VvlaHttpClient
from rlinf_deploy.robots.franka.fr3 import FR3Adapter, FR3Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-host", required=True)
    parser.add_argument("--vvla-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--token")
    args = parser.parse_args()

    images = tuple(
        ImagePayload(
            name=path.name,
            mime_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg",
            data=path.read_bytes(),
        )
        for path in args.image
    )
    robot = FR3Adapter(FR3Config(host=args.robot_host))
    robot.connect()
    try:
        client = VvlaHttpClient(
            args.vvla_url,
            token=args.token,
            timeout_s=args.timeout_s,
        )
        controller = Pi05FR3Runtime(robot, client, instruction=args.instruction)
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
