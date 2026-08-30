# RLinf Deploy

RLinf Deploy owns the last mile between an inference service and a robot or
simulator. It does not load checkpoints, import model frameworks, build prompts,
or execute neural networks.

```text
camera / robot state
        |
        v
RLinf Deploy -- HTTP --> RLinf Inference (VVLA)
        |
        v
robot-specific safety and motion execution
```

## Repository boundary

| RLinf Deploy owns | RLinf Inference owns |
|---|---|
| Robot and simulator observations | Checkpoints and processors |
| HTTP client, retries, deadlines | Prompt and model forward |
| Session and step identifiers | Session memory and batching |
| Robot action validation | Model-output parsing |
| Go2, FR3, and SO-101 control | DP, TP, CUDA Graph, compilation |
| SSH deployment and service routing | Canonical policy actions |

`third_party/vvla` pins the server implementation for integration and deployment
reproducibility. Runtime communication still crosses the versioned HTTP API; the
deploy package never imports VVLA Python modules.

## HTTP contract

The client expects these endpoints:

```text
GET    /healthz
GET    /v1/capabilities
POST   /v1/sessions
POST   /v1/sessions/{session_id}/steps
POST   /v1/sessions/{session_id}/reset
DELETE /v1/sessions/{session_id}
```

Step requests use `multipart/form-data`: one JSON metadata part followed by
binary image parts. Images are never base64 encoded. `request_id`, `step_id`,
and `session_revision` provide idempotency and ordering.

## Cameras

Camera capture is independent of robot adapters. `rlinf_deploy.cameras`
provides a `CameraSource` contract and a `CameraRig` for ordered multi-camera
capture and cleanup. `RGBCameraSource` can wrap any `RGBFrameDevice`; the first
ready-made factories cover OpenCV devices (USB/UVC, laptop, or phone cameras)
and Intel RealSense RGB:

```bash
python -m pip install -e '.[opencv]'
python -m pip install -e '.[realsense]'
```

Captured RGB frames are JPEG-encoded into the same `ImagePayload` used by every
robot runtime. Camera ownership therefore stays out of SO-101 and FR3 motion
adapters. The ready-made factories currently reuse LeRobot's camera drivers as
an optional implementation detail; custom camera SDKs do not depend on a robot
type and only need to implement `connect()`, `read()`, and `close()`.

## FR3

Install the official Franky binding that matches the robot server version:

```bash
python -m pip install -e '.[fr3]'
```

The adapter follows the official Franky API:

- `Robot(host)` and `recover_from_errors()`
- bounded `JointMotion`
- bounded relative `CartesianMotion`
- optional `Gripper.move`
- state from `robot.state`

Physical execution is intentionally fail-closed. Joint and Cartesian step
limits are checked before Franky receives a command.

See `examples/fr3_http_step.py` for a single HTTP inference step followed by
execution of the returned action chunk.

Quick end-to-end flow:

```bash
# On Thor: run VVLA HTTP service for pi0.5
cd third_party/vvla
vvla-http-serve \
  --policy pi05 \
  --checkpoint <pi05-checkpoint> \
  --adapter-config ../../configs/pi05_http_serve.example.json \
  --host 0.0.0.0 \
  --port 8000 \
  --max-batch 1

# On deploy side: send one image + state and execute one FR3 command
cd ../..
python examples/fr3_http_step.py \
  --robot-host <fr3-ip> \
  --vvla-url http://<thor-ip>:8000 \
  --instruction "pick up the object" \
  --image /path/to/image.jpg
```

## SO-101

SO-101 support wraps LeRobot's official `SO101Follower` API while keeping
policy-to-hardware validation in this package. Install the optional dependency:

```bash
python -m pip install -e '.[so101]'
```

This includes ordinary USB/UVC cameras through LeRobot's OpenCV backend. Add
the RealSense extra when an Intel RealSense camera is used:

```bash
python -m pip install -e '.[so101,realsense]'
```

Configure and calibrate the follower with LeRobot first. The same calibration
ID must be used for setup, calibration, and deployment. The adapter exposes five
joint positions in degrees plus the normalized gripper position in `[0, 100]`.
It rejects non-finite values, wrong action spaces, incomplete commands, and
targets that exceed the configured per-step limits.

Deployment never starts interactive calibration. If the configured calibration
is missing or does not match the motors, the adapter disconnects and fails before
accepting an action; use `lerobot-calibrate` beforehand.

Use `configs/pi05_so101_http_serve.example.json` only with a checkpoint whose
state/action schema and normalization were trained for SO-101. An FR3 checkpoint
is not compatible merely by dropping one action dimension.

```bash
python examples/so101_http_step.py \
  --robot-port /dev/ttyACM0 \
  --robot-id my_follower_arm \
  --vvla-url http://<thor-ip>:8000 \
  --instruction "pick up the object" \
  --image /path/to/image.jpg
```

For live cameras, first list the devices with `lerobot-find-cameras opencv` or
`lerobot-find-cameras realsense`. Camera names must appear in the server
adapter's `image_fields`; that server-side list orders the views and must match
the checkpoint's training setup. For the two-camera example below, set
`image_fields` to `["camera-0", "camera-1"]`:

```bash
python examples/so101_camera_step.py \
  --robot-port /dev/ttyACM0 \
  --robot-id my_follower_arm \
  --vvla-url http://<thor-ip>:8000 \
  --instruction "pick up the object" \
  --opencv-camera camera-0=/dev/video0 \
  --opencv-camera camera-1=/dev/video2
```

Use `--realsense-camera camera-0=<serial>` instead for a RealSense device. The
initial RealSense integration sends its RGB stream only: the current inference
image contract has no depth-image semantics. Multiple cameras are captured
sequentially and are not hardware synchronized.

`stop()` holds the most recently measured pose because SO-101 has no dedicated
stop primitive. This is a software stop, not a safety-rated emergency stop.
Provide a physical power cutoff when the deployment risk requires one.

## Go2

The existing Unitree Go2 camera/control agents and SSH deployer remain under:

```text
src/rlinf_deploy/robots/unitree/go2
src/rlinf_deploy/bindings/unitree/go2/streamvln
```

The robot-resident processes remain isolated from model inference and expose
bounded HTTP control surfaces.
