# Go2 visual navigation runtime

## One-command StreamVLN run

On the dual-4090 host, the wrapper checks or deploys the Go2 camera service,
re-arms the control service for a live run, and starts the image-reactive
StreamVLN loop:

```bash
cd /home/user/go2-nav-runtime/RLinf-deploy
bash nav.sh \
  --prompt "Find the tripod and stop in front of it." \
  --robot-host 192.168.137.34 \
  --ssh-user unitree \
  --camera-serial '<verified RealSense serial>' \
  --depth-scale '<verified meters per raw depth unit>'
```

`nav.sh` first stops only the repository-owned Go2 camera/control services,
installs the current checkout, and then restarts those services. It does not
inspect, stop, or restart a model server. Motion is enabled by default. Pass
`--dry-run` to run camera plus model inference without sending motion commands.
If `GO2_SSH_PASSWORD` is not exported, the wrapper prompts for it without
echoing it. `bash nav.sh --help` lists robot host, SSH, checkpoint, GPU,
timeout, camera calibration, and velocity overrides. Robot host, SSH user,
RealSense serial, and depth scale have no checked-in laboratory defaults; pass
them on the command line or set `GO2_ROBOT_HOST`, `GO2_SSH_USER`,
`GO2_CAMERA_SERIAL`, and `GO2_DEPTH_SCALE`.

The generic Python CLI behaves differently: `examples/go2_navigation.py` is
observation-only unless `--execute` is present.

## Current ownership and closed loop

The runnable composition is assembled in `embodied_runtime.apps.navigation`.
It selects one policy, constructs task sessions, and connects them to the
canonical Go2 clients:

```text
Qwen endpoint
StreamVLN model runtime ─────┐
InternVLA model runtime ─────┼─> policies.navigation
                            │       QwenNavigationPolicy
Go2 RGB or RGB-D image ─────┘       StreamVLNNavigationPolicy
Go2 state -------------------------- InternVLANavigationPolicy
                                      │
                                      v
                         tasks.navigation.WaypointPlan
                                      │
                         task-owned navigation session
                           ┌──────────┴──────────┐
                           v                     v
                  continuous follower    reactive pulse
                           └──────────┬──────────┘
                                      v
                         PlanarVelocityCommand
                          (vx, vy, yaw_rate)
                                      │
                                      v
                     robots.unitree.go2.Go2ControlClient
                                      │ HTTP velocity lease
                                      v
                       Go2 control service → Unitree SDK2
```

The domain paths are:

- `models/vln/streamvln`: StreamVLN loading, recurrent inference, and normalized
  native discrete actions;
- `models/vla/internvla_n1`: InternVLA DualVLN/NavDP loading, depth handling,
  and neutral copies of official native outputs;
- `policies/navigation`: Qwen request adaptation plus StreamVLN/InternVLA
  geometry mapping into task-owned plans;
- `tasks/navigation`: RGB-D observations, `WaypointPlan`, continuous/reactive
  sessions, waypoint tracking, velocity pulses, and lease control;
- `robots/unitree/go2`: host camera/control clients and the robot-resident
  camera/control agent;
- `deployment/unitree/go2`: SSH installation and service lifecycle; and
- `apps/navigation`: configuration, CLI composition, session selection, and
  JSON Lines output.

Qwen calls the existing OpenAI-compatible endpoint and asks for validated
structured output. Closing the policy closes only its client transport; it
does not start, stop, or restart Qwen. StreamVLN and InternVLA load the selected
local checkpoint in the navigation process. InternVLA `dualvln` consumes RGB;
`navdp` additionally requires registered depth.

The application awaits each policy's `prepare()` lifecycle hook before it
starts the task session. StreamVLN checkpoint loading and configured warmup,
and InternVLA checkpoint loading, therefore do not consume `max_runtime_s`.
Qwen preparation intentionally does not probe or manage the external server.

Every policy returns a `WaypointPlan` in `base_link`: `x` is forward, `y` is
left, and positive yaw is counter-clockwise. A policy never emits Unitree SDK
commands. The task session validates the observation sequence and converts the
accepted plan into bounded planar velocity.

## Continuous and reactive sessions

`session.mode = "continuous"` uses `NavigationSession`. It captures image and
state, anchors the base-frame plan at the capture-time odometry pose, and lets a
10 Hz control loop follow the latest accepted plan while the next model
inference runs. This mode requires trustworthy changing planar pose state.

On the current Go2 path, `SportModeState.position` may remain constant while
the robot walks. Therefore `session.mode = "auto"` selects
`ReactiveNavigationSession` for StreamVLN. The reactive loop is:

```text
capture image/state → infer WaypointPlan → use first waypoint
       → bounded timed velocity pulse → stop → settle → capture again
```

The pulse duration is limited, and a new observation is required before the
next action. A terminal plan with no waypoint stops the episode. If a terminal
plan also contains a waypoint, the session executes its first waypoint, lets
the bounded lease expire, and captures and infers again; it stops when a later
plan is empty and terminal.

For Qwen and InternVLA, `auto` currently selects continuous mode. Choose
`reactive` explicitly if the robot does not expose usable translational
odometry. The runtime does not provide a global map, SLAM, or a separate
obstacle-avoidance planner; model waypoints and the available camera/state are
the implemented navigation signal.

## Hosts and services

The navigation process runs on the dual-4090 host and reaches two HTTP services
on the Go2:

```text
192.168.137.44                         192.168.137.34 (Go2)
navigation process                    camera :8765 / control :8080
  capture RGB-D  <------------------- encoded observation
  policy inference
  task session
  velocity lease -------------------> bounded SDK2 motion
```

The camera service owns RealSense/V4L2 capture and image encoding. The control
service owns Unitree SDK2 access, current robot state, the operator-ready
interlock, hard velocity bounds, and lease expiry. The host-side
`Go2ControlClient` uses `stream_move`, `update_move`, and stop through that
service; it never imports the Unitree SDK.

## Motion boundary

The Go2 control API rejects commands above its configured hard maxima:

```text
|vx| <= 0.35 m/s    |vy| <= 0.35 m/s    |yaw_rate| <= 0.7 rad/s
```

The checked-in Python configuration uses those maxima. `nav.sh` chooses lower
live defaults of `0.25 m/s`, `0.25 m/s`, and `0.50 rad/s` unless overridden.
Task-side clamping does not replace the robot-side check.

The Python CLI requires `--execute` before it calls preflight or creates a
velocity lease. `nav.sh` adds `--execute` unless `--dry-run` is supplied. During
a live run, preflight requires fresh robot state, an idle action slot, and the
operator-motion-ready interlock. Completion, timeout, interruption, or failure
stops the active lease. Keep the physical emergency stop available for every
live run.

## Environment

The additive setup script creates a Python 3.10 environment for the task,
policy, and selected local model code. It does not inspect, stop, or restart an
existing Qwen process:

```bash
export GO2_NAV_RUNTIME_ROOT=/home/user/go2-nav-runtime
bash scripts/setup_go2_navigation_env.sh
```

Only the selected local model is loaded by a run; large checkpoints do not all
need to be resident at once.

Pinned upstream inputs:

- StreamVLN source: `e48f6ff7e9201d93aae003e8f64b04c00cec13bc`
- StreamVLN real-world checkpoint:
  `7b0269ad28e7039a32627b950766c09b5a646f8e`
- InternNav source: `1d8d078aa9031a4a02a1ae05844d49a1768a10e4`

## Configure

Edit `configs/go2_navigation.toml` on the 4090 host or override individual
values on the command line. The checked-in service URLs are loopback-only and
therefore inert for a remote Go2. Pass `--robot-host` or set explicit
camera/control URLs when invoking the generic Python CLI.

Keep credentials out of the TOML file when possible:

```bash
export GO2_CAMERA_TOKEN='<camera token>'
export GO2_API_TOKEN='<control token>'
export QWEN_API_KEY='<only when the Qwen endpoint requires it>'
```

Command-line token overrides (`--camera-token`, `--control-token`, and
`--qwen-api-key`) are also supported, but may be visible in the process list.

Relevant overrides include:

```text
--backend qwen|streamvln|internvla
--session-mode auto|reactive|continuous
--instruction TEXT             --episode-id ID
--max-runtime-s SECONDS        --control-hz HZ
--camera-url URL               --control-url URL
--robot-host HOST              --camera-port PORT --control-port PORT
--streamvln-root PATH          --internvla-root PATH
--model-path PATH_OR_MODEL_ID  --device cuda:0
--cuda-memory-fraction 0.45    --max-new-tokens 64
```

`--model-path` applies to the selected backend and acts as the Qwen model ID
when `--backend qwen` is selected. `--cuda-memory-fraction` applies only to a
newly loaded StreamVLN process.

## Dry-run first

From the repository root on `192.168.137.44`:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
  /home/user/go2-nav-runtime/env/bin/python \
  examples/go2_navigation.py \
  --config configs/go2_navigation.toml \
  --backend streamvln \
  --session-mode reactive \
  --device cuda:0 \
  --instruction '导航到画面中的黄色立柱前，保持约 0.8 米距离'
```

This is observation-only because `--execute` is absent. Each event is emitted
as one UTF-8 JSON object followed by `navigation_summary`, so the output is
readable in a terminal and consumable as JSON Lines.

After checking service health, observation freshness, and model plans, enable
the live lease:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
  /home/user/go2-nav-runtime/env/bin/python \
  examples/go2_navigation.py \
  --config configs/go2_navigation.toml \
  --backend streamvln \
  --instruction '导航到画面中的黄色立柱前，保持约 0.8 米距离' \
  --max-runtime-s 120 \
  --execute
```

Qwen uses the already-running endpoint without changing it:

```bash
python examples/go2_navigation.py \
  --backend qwen \
  --qwen-base-url http://127.0.0.1:15003/v1 \
  --qwen-model qwen3.5-9b \
  --instruction '导航到三脚架前'
```

The Qwen command above is also a dry-run because it has no `--execute` flag.

## Model-only StreamVLN check

The dog is not needed for a checkpoint load test:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
  /home/user/go2-nav-runtime/env/bin/python \
  examples/go2_navigation_infer.py \
  --backend streamvln \
  --streamvln-root /home/user/go2-nav-runtime/src/StreamVLN \
  --model-path /home/user/go2-nav-runtime/checkpoints/streamvln-real-world \
  --cuda-memory-fraction 0.45 \
  --load-only
```

Inspect the target GPU before loading the checkpoint. The command does not
reallocate or terminate another process.
