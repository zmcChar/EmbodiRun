# Multi-robot serving

Three SO-101 arms rolling out the same instruction at the same time against one
inference service, at `batch_size = 3`.

<video controls muted playsinline preload="metadata" width="720">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*74 s. Six camera views: the three front cameras on the top row, the three wrist
cameras below, one column per device. Each column freezes when its own arm
leaves the cube in the bowl, and reports the time it happened.*

## What it shows

One operator session starts three rollout processes at once. All three arms get
the same instruction — **"Pick up the cube and place it in the bowl."** — and
share a single inference service. There is no per-robot service and no
per-robot checkpoint.

| | Device 1 | Device 2 | Device 3 |
|---|---|---|---|
| Compute | Jetson Orin NX | Raspberry Pi 4B (8 GB) | Raspberry Pi 4B (8 GB) |
| Arm | LeRobot SO-101 | SO-101 | SO-101 |
| Cameras | front + wrist, 640×480 | same | same |
| Cube placed at | 14.5 s | 66.0 s | 9.0 s |
| Chunk period, median | 3.59 s | 3.62 s | 3.62 s |
| Dropped observations / actions | 0 / 0 | 0 / 0 | 0 / 0 |

Inference runs on one Jetson AGX Thor, driving 20 Hz position control with a
50-step action chunk (`π0.5`, `num_steps = 10`, bfloat16, CUDA graph enabled).

## What the numbers say

- **Three robots do not slow each other down.** The three chunk periods agree to
  within 1%, so one service at `batch_size = 3` stays inside its budget.
- **Playback is the bottleneck, not inference.** Of the 3.6 s chunk period,
  2.5 s is the arm executing a fixed 50-step chunk. Inference, transport, and
  recording together account for about 1.1 s.
- **The recording path is stable.** Every device reports zero dropped
  observations and zero dropped actions across the whole run.

## Limitations

- The service does not report task success, so completion was verified frame by
  frame rather than read from a status field.
- Device 3's gripper calibration made the commanded closed position physically
  unreachable in this take. The arm still finishes the task, but the calibration
  should be redone before the next run.

## Where the pieces live

The runtime, the robot binding, and the recording format are described in
[Architecture](../architecture.md) and [Configuration](../configuration.md).
The inference service is [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer),
reached over the versioned [inference API](../http_api.md).
