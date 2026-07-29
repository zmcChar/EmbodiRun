# Embodied runtime prototype

This repository is a provisional prototype; the product name has intentionally
not been decided. Its first vertical slices run π0.5, SmolVLA, OpenVLA-OFT, and
GR00T N1.7 model packages through a hardware-neutral execution engine and a
Torch/CUDA backend.

The neutral Python namespace is `embodied_runtime`; VLA is one model family
under `embodied_runtime.models.vla`, not the boundary of the runtime.

## Five-group boundary

```text
models          model semantics and portable execution recipe
      \                         ModelPackage
       +------------------------------+
                                      v
distributed --> registration --> engine --> BackendSession --> backends
                                      |
                                      v
                                   robots
```

The initial implementation concentrates on Groups 1, 3, and 4:

- `models/vla/pi05`: builds a staged π0.5 package (`encode_prefix`,
  `init_state`, `denoise_step`, `finalize`) and owns reference parity.
- `models/vla/smolvla`: exposes the same staged contract for the pinned
  LeRobot 0.3.3 SmolVLA checkpoint, including its legacy flow schedule and
  robot-native action unnormalization.
- `models/vla/openvla_oft`: builds a full-forward categorical-action package
  and supplies the image/text preprocessing and action-token semantics.
- `models/vla/gr00t_n17`: exposes NVIDIA's official GR00T N1.7 policy as a
  formal single-forward package with a named-action contract.
- `models/base.py`: defines the optional adapter base and the explicit
  `preprocess_one → collate → unbatch → postprocess_one` cardinality boundary.
- `engine`: selects a runner from the package's `ExecutionPlan`, then owns
  request lifecycle, cancellation, cooperative safe points, and memory policy.
- `backends/torch_cuda`: probes devices and compiles, loads, and executes
  package entrypoints and state-update primitives without importing π0.5.

Two plans are implemented: `SingleForwardPlan` and `IterativeFlowPlan`.
Model-family semantics and execution pattern are independent: π0.5 is a VLA
using iterative flow, while OpenVLA-OFT is a real VLA using one causal forward
and a categorical action-token head.

`distributed` now contains asynchronous endpoint coordination and prototype
transports. `robots` and the RLinf integration remain explicit interface
boundaries for their owning groups.

## Development

Use an environment containing PyTorch and, for real π0.5 inference, LeRobot
0.5.1:

```bash
python -m pip install -e '.[dev,torch,pi05]'
pytest
```

For OpenVLA-OFT, install its isolated model extra instead:

```bash
python -m pip install -e '.[dev,torch,openvla_oft]'
```

SmolVLA uses a separate endpoint environment because its official legacy
checkpoint requires LeRobot 0.3.3:

```bash
python -m pip install -e '.[dev,torch,smolvla]'
```

Checkpoints are loaded from user-provided local paths or Hugging Face IDs. This
repository does not contain model weights, and real-model commands are offline
by default at the outer checkpoint boundary. Models with nested upstream
dependencies may require their own cached snapshots; GR00T's Cosmos backbone
is documented explicitly in the GR00T guide.

The dependency-free contracts and core runtime target Python 3.10+. The
`smolvla` endpoint is verified with Python 3.10 and LeRobot 0.3.3. The `pi05`
extra requires Python 3.12+ because that is LeRobot 0.5.1's declared minimum,
so the two real models intentionally run in separate environments.

Run the small cross-group contract fixture:

```bash
python examples/toy_flow_local.py --device cpu
```

Run the same boundary with a non-flow plan:

```bash
python examples/toy_single_forward_local.py --device cpu
```

Run two local engines as asynchronous edge/cloud runtimes behind a controllable
dummy link:

```bash
python examples/cloud_edge_failover.py \
  --config configs/cloud_edge_failover.toml \
  --mode async_blend
```

The edge engine executes on every control tick and is never blocked by cloud
completion. The configurable coordination modes are:

- `edge_only`: do not submit cloud work;
- `async_cloud_preferred`: use a fresh cached cloud result when available,
  otherwise use the current edge result;
- `async_blend`: synchronously combine the current edge result and a fresh
  cached cloud result.

The example disconnects and reconnects the dummy link according to the TOML
schedule. Here, asynchronous means that a control tick does not await cloud
completion. A result fuser is intentionally synchronous, lightweight, and free
of I/O.

Run a tokenizer-free real-weight π0.5 smoke test:

```bash
python examples/pi05_synthetic.py \
  --checkpoint /path/to/lerobot/pi05_base \
  --device cuda:0 \
  --dtype preserve \
  --num-steps 10 \
  --cuda-graph
```

Run the real π0.5 model on a local GPU and a lightweight policy on CPU:

```bash
python examples/pi05_cpu_gpu_collaboration.py \
  --config configs/pi05_cpu_gpu_collaboration.toml \
  --checkpoint /path/to/lerobot/pi05_base \
  --num-steps 1
```

This smoke test uses two local runtimes and a controllable dummy link. The first
tick returns the CPU result while π0.5 runs in the background, a later tick
blends the available action results, and a disconnected tick falls back to
CPU.

For a real two-host link, start π0.5 on the cloud GPU:

```bash
python examples/pi05_cloud_server.py \
  --config configs/pi05_cpu_gpu_collaboration.toml \
  --checkpoint /path/to/lerobot/pi05_base \
  --num-steps 1 \
  --host 0.0.0.0 \
  --port 18765
```

Run the formal SmolVLA endpoint on the edge host against that server:

```bash
python examples/smolvla_pi05_async.py \
  --edge-checkpoint /path/to/lerobot/smolvla_base \
  --edge-vlm-base-path /path/to/SmolVLM2-500M-Video-Instruct \
  --cloud-host <cloud-host> \
  --cloud-port 18765 \
  --edge-device cuda:0 \
  --num-steps 10 \
  --mode async_cloud_preferred
```

This path constructs SmolVLA once through
`SmolVLAAdapter → TorchCudaBackend → ExecutionEngine`. The first tick returns
the edge result while π0.5 remains in flight, the next tick may select a fresh
cached cloud result, and a connection-epoch change immediately falls back to
SmolVLA. The policies retain their native `[50, 6]` and `[50, 32]` action
contracts. `async_blend` is rejected because equal horizon does not imply
compatible robot actions.

Copy the standalone client to an edge host with Python 3.10+ and PyTorch, then
run its local GPU policy while requesting π0.5 asynchronously:

```bash
python wifi_edge_client_py310.py \
  --cloud-host <cloud-lan-ip> \
  --cloud-port 18765 \
  --edge-device cuda:0 \
  --num-steps 1 \
  --cloud-weight 0.5
```

`--failure-probe-port <closed-port>` additionally verifies that a real TCP
failure does not prevent the edge policy from producing an action. The
length-prefixed TCP/JSON transport and standalone client are prototype test
surfaces, not the future registration, discovery, authentication, or streaming
layer.

Run a synthetic-image OpenVLA-OFT smoke test:

```bash
python examples/openvla_oft_synthetic.py \
  --checkpoint /path/to/openvla-oft-checkpoint \
  --prompt "pick up the red block" \
  --device cuda:0
```

The current OpenVLA-OFT slice uses deterministic greedy decoding around one
complete causal-model forward. It establishes the model/engine/backend
contract, but does not yet implement prefix-KV splitting, rollout
log-probability output, or CUDA Graph capture. The reference BF16 checkpoint
is roughly 15.1 GB before activations, so a 16 GB GPU has very little headroom;
this first correctness path should be validated on a larger GPU until a
quantized or memory-optimized backend is added. This slice also accepts exactly
one RGB camera; multi-camera checkpoint layouts are rejected explicitly.

`preserve` retains π0.5's package-defined mixed precision. Explicitly casting
the whole model to FP16 or BF16 is exposed only as an experiment because the
reference vision and normalization paths intentionally retain FP32 parameters.
`--cuda-graph` captures only the repeated denoise entrypoint; the Engine,
initial RNG, Euler update, and model adapter remain unchanged. CUDA Graph
capture failures are reported in the command summary and safely use eager
execution for that exact input signature.

The π0.5 composition explicitly declares `denoise_step.time` as a dynamic
scalar input. Its value is copied into one fixed-address CUDA tensor before
each replay, so the default ten-step schedule captures one graph rather than
ten. Undeclared Python scalars remain capture-time constants and continue to
participate in the graph cache key.

See [architecture](docs/architecture.md) for the ownership boundary and
[verification](docs/verification.md) for the tested environment and results.
The GR00T provider boundary, isolated environment commands, and upstream
support status are documented in [GR00T N1.7](docs/gr00t_n17.md).
