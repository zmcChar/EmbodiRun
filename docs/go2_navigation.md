# Go2 unified navigation runtime

The Go2 path uses one Python 3.10 environment for Qwen, StreamVLN, and
InternVLA-N1 integration code. Qwen remains an OpenAI-compatible remote
provider; StreamVLN and InternVLA run locally with PyTorch 2.5.1 and
Transformers 4.51.0.

The unified environment does **not** mean that every checkpoint must be loaded
in one process or on one GPU. StreamVLN and InternVLA have recurrent episode
state and large weights, so each local provider owns one model process. The
composition root selects one provider while the existing Qwen process remains
untouched.

## Why the upstream StreamVLN environment is not installed verbatim

The published `requirements.txt` mixes training, Habitat simulation,
distributed training, depth filtering, and the small real-world RGB inference
path. The real-world checkpoint itself declares Transformers 4.51.0. This
repository ports only that inference path:

- the dependency-trimmed evaluator has no Habitat, ROS, quaternion, or depth
  filtering imports;
- the official `llava` and model sources stay pinned outside this repository;
- both official source roots are activated because upstream imports both
  `llava` and top-level `model` / `utils` namespaces;
- the checkpoint's embedded SigLIP weights initialize the vision tower without
  a second network download;
- SDPA is the default, so StreamVLN does not require FlashAttention.

## Environment

```bash
export GO2_NAV_RUNTIME_ROOT=/home/user/go2-nav-runtime
bash scripts/setup_go2_navigation_env.sh
```

The setup is additive: it creates or reuses only the selected prefix. It does
not inspect, stop, or restart an existing Qwen/vLLM/SGLang process.

Pinned upstream inputs:

- StreamVLN source: `e48f6ff7e9201d93aae003e8f64b04c00cec13bc`
- StreamVLN real-world checkpoint:
  `7b0269ad28e7039a32627b950766c09b5a646f8e`
- InternNav source: `1d8d078aa9031a4a02a1ae05844d49a1768a10e4`

## Shared model contract

All three providers receive a `NavigationRequest` containing an episode id,
monotonic observation sequence, encoded RGB context, optional registered
uint16 depth, and robot state. All providers return a `WaypointPlan`:

```text
frame = base_link
x_m   = forward
y_m   = left
yaw   = counter-clockwise
```

Qwen generates this metric schema directly. StreamVLN converts its native
`STOP / forward 0.25 m / left 15 deg / right 15 deg` vocabulary into cumulative
waypoints. InternVLA converts its native trajectory or discrete actions into
the same contract. No model provider emits Go2 velocity.

The `robots/go2` domain anchors capture-time waypoints in odometry and samples
a bounded velocity controller at 10 Hz. This decouples camera sampling and
model latency from the continuous command rate.

## Model-only StreamVLN check

The dog is not needed for a load test:

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

`--cuda-memory-fraction` caps this new process only. It never reallocates or
stops another process. Before starting a real local model, inspect the selected
GPU and ensure its existing service has enough headroom.
