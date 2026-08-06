# Go2 unified navigation runtime

The navigation command composes a Go2 RGB-D camera, odometry/control client,
metric waypoint follower, and exactly one Qwen, StreamVLN, or InternVLA
provider. It runs on the dual-4090 host and talks to the Go2-side services over
HTTP:

```text
192.168.137.44                         192.168.137.34 (Go2)
Go2NavigationSession                  camera :8765 / control :8080
  camera capture   <----------------  RGB-D observation
  provider inference
  waypoint follower
  10 Hz controller ---------------->  bounded SDK2 velocity lease
```

Models return `WaypointPlan` values in `base_link` (`x` forward, `y` left,
yaw counter-clockwise). They do not emit SDK commands. The Go2 follower anchors
each capture-time plan in odometry and continuously converts it to bounded
`vx`, `vy`, and `yaw_rate` while the next inference runs.

## Safety boundary and modes

The command is **dry-run by default**. Dry-run captures camera/state data and
runs the selected model, but never calls `stream_move`, `update_move`, or
`stop`. The only way this command enables control writes is the explicit
`--execute` flag; TOML cannot enable motion.

The deployed control API rejects commands beyond these hard limits, which are
also the checked-in defaults and maximum accepted configuration values:

```text
|vx| <= 0.35 m/s    |vy| <= 0.35 m/s    |yaw_rate| <= 0.7 rad/s
```

During execution, interruption or failure causes the session to stop its
velocity lease. Keep the physical emergency stop available for every live run.

## Environment

The additive setup script does not inspect, stop, or restart Qwen:

```bash
export GO2_NAV_RUNTIME_ROOT=/home/user/go2-nav-runtime
bash scripts/setup_go2_navigation_env.sh
```

The unified Python 3.10 environment contains the common contracts and model
adapters. It does not require every large checkpoint to be resident at once.
The selected local model is loaded in the navigation process. Qwen remains an
external OpenAI-compatible endpoint; closing a Qwen provider only releases its
client transport and never manages the existing server process.

Pinned upstream inputs:

- StreamVLN source: `e48f6ff7e9201d93aae003e8f64b04c00cec13bc`
- StreamVLN real-world checkpoint:
  `7b0269ad28e7039a32627b950766c09b5a646f8e`
- InternNav source: `1d8d078aa9031a4a02a1ae05844d49a1768a10e4`

## Configure

Edit `configs/go2_navigation.toml` on the 4090 host or override individual
values on the command line. The real camera default is
`http://192.168.137.34:8765`; control defaults to port `8080`.

Keep credentials out of the file when possible:

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
--instruction TEXT             --episode-id ID
--max-runtime-s SECONDS        --control-hz HZ
--camera-url URL               --control-url URL
--streamvln-root PATH          --internvla-root PATH
--model-path PATH_OR_MODEL_ID  --device cuda:0
--cuda-memory-fraction 0.45    --max-new-tokens 64
```

`--model-path` applies to the selected backend (and acts as the Qwen model id
for `qwen`). InternVLA `navdp` requires the camera service's registered depth;
`dualvln`, StreamVLN, and Qwen use RGB, while Qwen also receives depth metadata.

## Dry-run first

From the repository root on `192.168.137.44`:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
  /home/user/go2-nav-runtime/env/bin/python \
  examples/go2_navigation.py \
  --config configs/go2_navigation.toml \
  --backend streamvln \
  --device cuda:0 \
  --instruction '导航到画面中的黄色立柱前，保持约 0.8 米距离'
```

This is observation-only because `--execute` is absent. Each event is printed
as one UTF-8 JSON object, followed by `navigation_summary`. This makes the same
output readable in a terminal and consumable as JSON Lines.

After checking camera/state freshness and model plans, enable the live lease:

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

`--cuda-memory-fraction` applies only to the newly selected StreamVLN process.
Inspect the target GPU before loading it; the command never reallocates or
terminates another process.
