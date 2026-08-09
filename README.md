# Embodied runtime prototype

This repository is a provisional prototype; the product name has intentionally
not been decided. Its first vertical slices expose π0.5, SmolVLA, OpenVLA-OFT,
and GR00T N1.7 through a common inference-provider boundary. A Provider may
compose a local hardware Backend or call an external serving framework such as
vLLM-Omni.

The neutral Python namespace is `embodied_runtime`; VLA is one model family
under `embodied_runtime.models.vla`, not the boundary of the runtime.

## Architecture and ownership

Runtime decisions flow through domain-owned values:

```text
model checkpoint/runtime
        │
        v
     models ── model-native prediction ──> policies
                                                │ task-owned plan
                                                v
                                              tasks
                                          ┌─────┴─────┐
                                          v           v
                                       robots     simulators
```

Observations travel in the reverse direction. `models` owns model semantics,
portable packages, execution plans, and model adapters. `policies` is the
model-coupled translation layer: it turns native predictions into values owned
by a task, such as `WaypointPlan`. `tasks` owns robot-independent requests,
plans, interfaces, controllers, and closed loops. Physical `robots` and
`simulators` implement those task-facing boundaries; neither task sessions nor
models import a concrete Go2 or VLABench implementation.

Local inference has a separate execution path:

```text
ModelAdapter → ModelPackage → ExecutionEngine → BackendSession → device
                                  ^                    ^
                         request lifecycle      hardware execution
```

`engine` owns inference requests/results, the Provider interface, scheduling,
batching, cancellation, memory admission, and execution-plan runners.
`backends` owns devices, compilation artifacts, loaded sessions, memory
statistics, and concrete Torch/CUDA, Ascend, and Horizon implementations. The
Ascend and Horizon backends are currently declared extension points; only the
Torch backend executes packages. External providers such as vLLM-Omni can
implement the same Provider interface without constructing a local engine or
backend.

The remaining packages have narrow ownership:

- `distributed` owns endpoint coordination, routing, failover, session
  identity, and prototype transports; it does not schedule kernels.
- `deployment` installs and supervises robot-resident services; it does not
  decide task actions.
- `evaluation` owns metrics, trial records, comparisons, and replay reports.
- `integrations` adapts external serving, training, planning, and LeRobot
  frameworks to the domain APIs.
- `apps` is the composition root allowed to wire models, policies, tasks,
  robots, simulators, engine, backends, and transports into runnable programs.
- `utils` contains small ownership-free geometry, HTTP, and image helpers.

The current source layout is:

```text
src/embodied_runtime/
├── models/                 # model specs, packages, plans, VLA/VLN runtimes
├── policies/navigation/    # model-coupled navigation task adapters
├── tasks/                  # navigation and high-level planning loops
├── robots/unitree/go2/     # Go2 host clients and robot-resident agent
├── simulators/             # simulator endpoints, traces, VLABench adapters
├── engine/                 # Provider API and single-node execution engine
├── backends/               # hardware interfaces and implementations
├── distributed/            # communication, routing, failover, sessions
├── deployment/unitree/go2/ # Go2 installation and service lifecycle
├── evaluation/             # metrics, reports, and benchmark comparisons
├── integrations/           # GR00T, planning, LeRobot, and RLinf adapters
├── apps/                   # runnable composition roots
└── utils/                  # geometry, HTTP, and image primitives
```

The Go2 navigation composition follows the same ownership chain:

```text
Selected Qwen / StreamVLN / InternVLA / NaVILA / ActiveVLN policy
        → tasks.navigation.WaypointPlan
        → NavigationSession or ReactiveNavigationSession
        → bounded planar velocity
        → robots.unitree.go2.Go2ControlClient
        → robot HTTP control service → Unitree SDK2
```

Go2 navigation selects a model independently from its inference runtime.
`transformers` is the default: StreamVLN, InternVLA, NaVILA, and ActiveVLN load
their selected local runtime in the navigation process, while Qwen calls its
existing OpenAI-compatible endpoint. StreamVLN and NaVILA also expose an
experimental `vllm-omni` protocol option that connects to an already-running external
OpenPI-compatible service. The client does not launch or manage that service,
and the service must implement this repository's expected handshake and
navigation messages; this is not a claim of native upstream vLLM-Omni model
support or acceleration. Every route returns a task-owned `WaypointPlan`; none
emits SDK commands. NaVILA uses seven uniformly sampled historical images plus
the latest observation and normalizes one textual navigation action into the
same plan contract. See [Go2 navigation](docs/go2_navigation.md) for the closed
loop and verified commands.

The optional `third_party/vvla` submodule pins ActiveVLN's source/checkpoint
contract and enables two in-process runtimes. `activevln + transformers` uses
stock-HF full-history generation; `activevln + vvla` uses VVLA's eager,
incremental B=1 session KV. Both enforce the same strict R2R action grammar.
The local RTX 5080 smoke benchmark measured VVLA at about 1.20x/1.22x the
Transformers p50 speed on turns one/two, but the BF16 action texts diverged, so
this is not an output-parity claim. VVLA still does not provide StreamVLN/NaVILA
runtimes, and ActiveVLN is not marked as vLLM-Omni compatible. See
[ActiveVLN navigation through Transformers and VVLA](docs/navigation_vvla_activevln.md)
for both commands, the paired benchmark caveats, and Habitat requirements.

The exact experimental handshake, action encoding, and local RTX 5080
verification are documented in
[Navigation through vLLM-Omni/OpenPI](docs/navigation_vllm_omni.md).

Implemented model execution includes:

- `models/vla/pi05`: staged π0.5 execution (`encode_prefix`, `init_state`,
  `denoise_step`, `finalize`) with reference parity.
- `models/vla/smolvla`: the pinned LeRobot 0.3.3 SmolVLA flow schedule and
  robot-native action unnormalization.
- `models/vla/openvla_oft`: one full causal forward with image/text
  preprocessing and categorical action-token decoding.
- `models/vla/gr00t_n17`: NVIDIA GR00T N1.7 as a single-forward package and a
  native vLLM-Omni/OpenPI provider path.
- `models/vla/navila`: the official NaVILA 8B checkpoint loader, fixed
  eight-frame preprocessing, and native textual navigation action.
- `models/base.py`: the reusable
  `preprocess_one → collate → unbatch → postprocess_one` cardinality boundary.
- `engine/providers`: the backend-injected local Provider and lazy Provider
  registry; `distributed/multitenant` owns shared-session serving, while
  `integrations/gr00t` owns the external GR00T/vLLM-Omni adapter.

`SingleForwardPlan` and `IterativeFlowPlan` have engine runners today.
Model-family semantics and execution pattern remain independent: π0.5 uses an
iterative flow plan, while OpenVLA-OFT uses one causal forward and a categorical
action-token head.

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

NaVILA also requires an independent Python 3.10 environment. Its pinned
[official source](https://github.com/AnjieCheng/NaVILA/tree/76b98f233dd0fff05dfcd69435eec6740febff9d)
uses a specific Transformers replacement, so do not copy that replacement into
the shared Go2/Qwen/StreamVLN environment. The released checkpoint is
[`a8cheng/navila-llama3-8b-8f`](https://huggingface.co/a8cheng/navila-llama3-8b-8f).
Create the isolated, pinned eager-attention environment with:

```bash
bash scripts/setup_navila_navigation_env.sh
```

Checkpoints are loaded from user-provided local paths or Hugging Face IDs. This
repository does not contain model weights, and real-model commands are offline
by default at the outer checkpoint boundary. Models with nested upstream
dependencies may require their own cached snapshots; GR00T's Cosmos backbone
is documented explicitly in the GR00T guide.

The domain-owned values and core runtime target Python 3.10+. The
`smolvla` endpoint is verified with Python 3.10 and LeRobot 0.3.3. The `pi05`
extra requires Python 3.12+ because that is LeRobot 0.5.1's declared minimum,
so the two real models intentionally run in separate environments.

Run the small local engine/backend fixture:

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
CLOUD_HOST=cloud-hostname
python examples/smolvla_pi05_async.py \
  --edge-checkpoint /path/to/lerobot/smolvla_base \
  --edge-vlm-base-path /path/to/SmolVLM2-500M-Video-Instruct \
  --cloud-host "$CLOUD_HOST" \
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
schemas. `async_blend` is rejected because equal horizon does not imply
compatible robot actions.

Copy the standalone client to an edge host with Python 3.10+ and PyTorch, then
run its local GPU policy while requesting π0.5 asynchronously:

```bash
CLOUD_LAN_HOST=cloud-lan-hostname
python wifi_edge_client_py310.py \
  --cloud-host "$CLOUD_LAN_HOST" \
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

## Multi-robot edge/cloud prototype

The multi-robot slice runs one independent edge process, local Provider, and
failover coordinator per robot. All of those processes may connect to one cloud
process that owns one shared Provider and one loaded model:

```text
robot A observation -> edge process A --\
robot B observation -> edge process B ----> one cloud Provider / model queue
robot C observation -> edge process C --/       session A / B / C isolated
```

The executable CLI currently constructs only a local `torch_cuda` Provider; it
does not yet select vLLM-Omni, MaaS, or a vendor Backend from configuration.
The sample uses a PyTorch model fixture. Change `provider.device` or pass
`--device cuda:0` to place it on a cloud GPU:

```bash
PYTHONPATH=src python examples/multi_robot_cloud.py \
  --config configs/multi_robot_cloud.toml \
  --host 0.0.0.0 \
  --port 18770
```

The checked-in default binds only to loopback. `--host 0.0.0.0` is shown solely
for a trusted, isolated two-host experiment: this prototype transport has no
authentication or encryption and must not be exposed to an untrusted network
or connected to physical actuation without a safety layer.

On the 5080 host, start two independent edge processes. They deliberately use
different logical identities and receive fresh automatic session IDs, but may
advertise the same physical resource label because both processes share one
GPU:

```bash
CLOUD_HOST=cloud-hostname
PYTHONPATH=src python examples/multi_robot_edge.py \
  --config configs/multi_robot_edge.toml \
  --robot-id robot-a \
  --edge-node-id edge-a \
  --physical-host-id workstation-5080 \
  --physical-resource-id gpu-5080-0 \
  --device cuda:0 \
  --cloud-host "$CLOUD_HOST" \
  --observation-offset 0

PYTHONPATH=src python examples/multi_robot_edge.py \
  --config configs/multi_robot_edge.toml \
  --robot-id robot-b \
  --edge-node-id edge-b \
  --physical-host-id workstation-5080 \
  --physical-resource-id gpu-5080-0 \
  --device cuda:0 \
  --cloud-host "$CLOUD_HOST" \
  --observation-offset 100
```

The same edge command can represent another robot on a 3070 by changing its
logical IDs and physical resource declaration. Passing `--device cpu` creates a
CPU edge runtime. `physical_resource_id` is declaration-only metadata in this
slice: it performs no locking, placement, or resource arbitration. Likewise,
`cuda:0` is resolved inside one process and is not a globally unique node or
device identity.

Add `--disconnect-tick 3 --reconnect-tick 6` to one edge command to exercise
that robot's logical cloud-link loss and background re-registration while the
other robot remains connected.

The cloud model is loaded once. Each session is ordered and isolated, and the
default configuration admits one pending request per session and uses
`max_batch_size = 1`. Cross-robot batching is therefore disabled by default. A
trusted homogeneous deployment may explicitly opt in only after declaring its
Provider multi-tenant-safe and ensuring that all batched observations have a
compatible model-owned schema and shape.

Every wire request carries `robot_id`, `edge_node_id`, `session_id`,
`sequence_id`, `observation_id`, and the actual JSON numeric observation. The
service namespaces otherwise identical request IDs before they enter the shared
engine and echoes authoritative identity and action-schema metadata on every
result. This JSON path proves that observations cross a real process or host
boundary; it is not a production RGB/depth codec and does not preserve an
efficient tensor representation.

Only an admitted cloud submission snapshots mutable tensor/array storage.
Encoding, transport, and cloud inference then run off the control path. The
local snapshot copy itself is an ownership cost on that tick and must be
budgeted. A real-time capture path can pass an immutable/reference-counted
frame as `owned_cloud_observation` to make that ownership transfer O(1).

Each edge process owns its connection epoch and cached cloud authority. A
single session rejection or disconnect invalidates only that robot's cloud
state. A shared-cloud process or network outage naturally affects every cloud
session, but each edge process continues through its own local Provider.

This first executable topology uses a static cloud address and an explicit
service-local session handshake. It is not dynamic node registration and does
not implement the future leased `RuntimeRegistry`, discovery, or route
selection. The default sample remains a numeric target vector so the topology
can run without checkpoints. The real SmolVLA configuration instead asks the
loaded adapter for model-shaped synthetic observations; a physical robot still
requires its own observation mapper and a production wire codec.

The generic Provider boundary can also represent vLLM-Omni. The current
vLLM-Omni/OpenPI Provider owns WebSocket and reset/session state, however, so it
must be instantiated once per robot session. Those independent
Provider/WebSocket instances may still connect to the same shared vLLM-Omni
server; one stateful Provider instance must not be shared directly by multiple
robot sessions.

A real four-model deployment is also provided: two SmolVLA edge processes on
one RTX 5080, one SmolVLA edge process on an RTX 3070, and one shared CPU
SmolVLA service. See
[`docs/smolvla_four_model_demo.md`](docs/smolvla_four_model_demo.md) for the
fixed model schema, launch commands, health probe, disconnect test, and
acceptance criteria.

Run a synthetic-image OpenVLA-OFT smoke test:

```bash
python examples/openvla_oft_synthetic.py \
  --checkpoint /path/to/openvla-oft-checkpoint \
  --prompt "pick up the red block" \
  --device cuda:0
```

The current OpenVLA-OFT slice uses deterministic greedy decoding around one
complete causal-model forward. It establishes the model/engine/backend
boundary, but does not yet implement prefix-KV splitting, rollout
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
