# Pi0.5 with EmbodiInfer and two SO-101 followers

The dual-arm recipe was introduced by these branches:

| Repository | Branch |
| --- | --- |
| EmbodiRun | `feat/pi05-agx-deployment-20260917` |
| EmbodiInfer | `perf/pi05-agx-mixed-graph-20260917` |

Deploy's `third_party/embodiinfer` points to inference commit
`430d4ef660c7d3b086b1c8126109d0ca6ad3a9a0`
([inference PR #27](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/27)).
Host `init` uses the gitlink from `metadata.deploy-commit`; a separate inference
checkout does not override it. After merging, use the Deploy revision containing
this change and its pinned inference revision.

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
git submodule update --init third_party/embodiinfer
# Requires SSH access to the private inference repository through .gitmodules.
uv sync --frozen
cp configs/pi05/bi-so101-vvla.yaml /absolute/path/to/my-pi05.yaml
```

The example runs Compute and Control on the local machine. Replace the two
serial ports, three V4L2 devices, calibration directory, and checkpoint path.
For a remote AGX or Thor, change the node connection to SSH as in the existing
[HTTP example](../configs/http-wireless-inference/http.yaml). To run inference on
Thor and arms on AGX, define separate nodes and assign only `models.pi05.node`
to Thor; keep the robot and all sensors on AGX. Bind the inference server to its
reachable LAN address and keep Control on loopback. The inference API has no
authentication; expose it only on a trusted, restricted network.

## Model and environment

Copy the complete 30,000-step LeRobot checkpoint, including its configuration,
pre/postprocessor configuration and normalization statistics, plus the tokenizer
used during training. Resolve any tokenizer path in the processor configuration
on the new machine. Use a checkpoint with 12 state/action dimensions and the
camera roles `front`, `left_wrist`, `right_wrist`. If checkpoint image feature
names differ, set `models.pi05.image_keys` to map each `observation.images.*`
input to its checkpoint feature name. Do not swap camera roles or recalibrate
the recorded state values to a different convention.

The template passes `attention: eager`, `vision_attention: sdpa`,
`native_embeddings: true`, and `low_cpu_mem_usage: true` to the inference
policy through Host's generated adapter JSON. `--dtype auto` retains the
checkpoint's mixed parameter precision; forcing `--dtype bfloat16` changes that
path. Batch size is 1, with 10 denoising steps and full-loop CUDA Graph capture.
Weights, buffer restoration, processing and model execution remain in inference.

Use uv 0.12.x. Control needs the `robot-so101` group (Python 3.12+), including
OpenCV with V4L2 support on the control node. Inference needs a working CUDA PyTorch
environment compatible with that device, plus the dependencies from the pinned
inference project. AGX Orin and Thor cannot be assumed to use the same wheel.
Host `init` runs the inference project's `uv sync --frozen`; naming an existing
environment in YAML **does not preserve a manually installed Jetson PyTorch**.
Check the inference dependency group against the board's JetPack/CUDA/Python
stack before `init`. This PR does not provide an Orin/Thor wheel installer or
claim that the generic locked environment boots on every Jetson.

## Robot mapping and operation

`lerobot.bi_so101` composes the existing calibrated SO-101 drivers. Each arm
needs its own LeRobot calibration file (`left_follower.json` and
`right_follower.json` in the example), with the six unprefixed motor names and
IDs expected by the single-arm driver. An old combined 12-joint calibration
file is not accepted directly. Calibration, joint signs, motor IDs, and gripper
open/closed direction must match the robot and training data; a larger gripper
number does not by itself identify closure.

State and action order is left six, then right six. Within each arm it is
`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
`gripper`. Wire feature names are `left_arm_<motor>.pos` and
`right_arm_<motor>.pos`; joints use degrees and grippers use the calibrated
0–100 range. Returned features are matched by name, not by their wire order.

The adapter checks both arm targets against fresh feedback before sending any
goal, then rechecks each arm at execution. Default limits reject excessive
steps; `step_limit_mode: clip` explicitly enables per-step clipping. Two serial
buses cannot be written atomically. A send failure attempts measured holds on
both arms and propagates any hold failure. Existing goal initialization,
torque-disconnect behavior and Control arbitration remain in use.

Do not run this direct serial driver concurrently with the older AGX cart/arm
service on the same ports. Stop and release the previous hardware owner before
an operator starts this Control task. This binding commands both arms; it does
not import the old application's left-arm hold, grasp detector, final lift,
base/head control or web UI.

Validate configuration without contacting hardware:

```bash
uv run embodirun --config /absolute/path/to/my-pi05.yaml validate
```

Once device dependencies and paths are ready, `init` prepares projects and
environments; `up` starts services without connecting the arms. `run` is the
explicit motion command and connects both arms. For an operator-authorized
single chunk at the recorded 15 Hz playback rate:

```bash
uv run embodirun --config /absolute/path/to/my-pi05.yaml init
uv run embodirun --config /absolute/path/to/my-pi05.yaml up
# MOTION: issue only after checking calibration, workspace and serial ownership.
uv run embodirun --config /absolute/path/to/my-pi05.yaml run \
  --runtime bi-so101-pi05 --prompt 'Pick up the chips.' \
  --max-steps 1 --chunk-steps 50 --control-hz 15 --request-timeout 120
uv run embodirun --config /absolute/path/to/my-pi05.yaml down
```

Playback rate is not inference frequency. Control completes an action chunk
before obtaining the next prediction; inference latency still creates a pause
between chunks. This integration enables the optimized model settings but adds
no asynchronous prediction or latency compensation.

## Validation scope

Software tests cover the 12-feature mapping, three image roles, generated EmbodiInfer
arguments, both serial-port checks, step limits, and cleanup after a bus failure.
Inference PR #27 records the separate AGX model-output and performance checks.
Those results are not an end-to-end measurement of this new Control binding.
Replay identical observations on the destination device before motion, then
measure camera-to-action latency and task behavior there. The new dual-arm
driver has not been exercised on a physical robot in this PR.
