# Embodied runtime architecture

## Ownership and runtime flow

The runtime is organized by the owner of each value and behavior. Model output
moves through one explicit decision chain:

```text
checkpoint / model server
          │
          v
       models
          │ model-native prediction
          v
       policies
          │ task-owned plan
          v
        tasks
      ┌───┴───────────┐
      v               v
   robots         simulators
```

Observations return in the opposite direction. This diagram is a runtime data
flow, not permission for every package to import every package above it.

- `models` owns `ModelSpec`, `ModelPackage`, execution plans, model adapters,
  model-native outputs, preprocessing, and checkpoint/runtime code.
- `policies` owns model-coupled interpretation. A policy may use model-native
  code and task-owned values, but it does not command a robot. The navigation
  policies normalize Qwen, StreamVLN, InternVLA, and NaVILA output into
  `tasks.navigation.WaypointPlan`.
- `tasks` owns robot-independent goals, observations, plans, interfaces,
  controllers, and closed loops. Navigation and high-level planning live here.
- `robots` owns physical drivers, SDK bindings, hardware limits, host clients,
  and robot-resident services. A robot implements task-facing interfaces; task
  code never imports a concrete robot.
- `simulators` owns simulator endpoints, observation/action mapping, traces,
  and privileged evaluation helpers. It is an alternate execution endpoint,
  not a special model backend.

Shared values live with the domain that defines their meaning. Consumers import
them from that owner: inference envelopes from `engine`, model packages from
`models`, waypoint plans from `tasks.navigation`, and robot observations from
`robots`.

## Current source tree

```text
src/embodied_runtime/
├── models/
│   ├── plans/                 # single-forward and iterative-flow plans
│   ├── vla/                   # π0.5, SmolVLA, OpenVLA-OFT, GR00T, InternVLA, NaVILA
│   └── vln/streamvln/         # StreamVLN runtime and native action validation
├── policies/navigation/
│   ├── qwen/                  # policy, HTTP transport, schema, parsing
│   ├── streamvln/             # policy and native-action geometry
│   ├── internvla/             # policy and native-output mapping
│   └── navila/                # episode history and textual-action geometry
├── tasks/
│   ├── navigation/            # task values, sessions, and controllers
│   └── planning/              # task goals and versioned high-level plans
├── robots/unitree/go2/        # Go2 clients, limits, and resident agent
├── simulators/                # simulator endpoints and evaluation traces
├── engine/
│   ├── providers/             # local Provider and Provider registry
│   └── runners/               # model-plan execution runners
├── backends/                  # Torch/CUDA plus vendor extension points
├── distributed/
│   ├── communication/         # transport endpoints and codecs
│   ├── multitenant/           # shared Provider session isolation
│   └── registry/              # runtime registration data
├── deployment/unitree/go2/    # install/start/stop robot services
├── evaluation/                # metrics, reports, and benchmark comparisons
├── integrations/
│   ├── gr00t/                 # external GR00T/vLLM-Omni adapter
│   ├── lerobot/               # LeRobot/VLABench adapters
│   ├── planning/              # external task planners
│   └── learning/rlinf/        # RLinf integration boundary
├── apps/                      # runnable composition roots
└── utils/                     # geometry, HTTP, and image primitives
```

`apps` is intentionally the broadest package: a runnable application may wire
domains together. `integrations` may adapt an external framework to one or more
owned APIs. Domain packages do not use either package as a shortcut to reach
another domain.

The source dependency direction is:

| Owner | Allowed inward dependencies | Must stay independent of |
| --- | --- | --- |
| `models` | root value helpers and upstream model libraries | policies, tasks, robots, apps |
| `policies` | model runtimes and task-owned values | engine envelopes, concrete robots, simulators, app composition |
| `tasks` | ownership-free utilities | model families, policies, concrete robots |
| `robots` | task interfaces/commands and utilities | model and policy implementations |
| `simulators` | robot observations/actions and simulator libraries | physical robot SDK implementations |
| `engine` | model packages/plans and backend interfaces/values | model-family dispatch and concrete hardware backends |
| `backends` | model packages plus engine execution context/errors | tasks, policies, robot behavior |
| `distributed` | engine envelopes, model request values, task-planning values | model kernels and robot SDKs |
| `deployment` | process, filesystem, SSH, and service configuration | task and policy decisions |
| `evaluation` | task results, traces, and metrics | robot SDKs and model loading |
| `integrations` | only the owned APIs needed by the external framework | acting as a second home for generic domain types |
| `apps` | any domain required by that runnable composition | reusable domain semantics |
| `utils` | Python standard library | every domain package |

When an implementation satisfies an interface owned by another domain, the
implementation imports the interface. For example, the Go2 client imports the
task-owned `MobileBase` vocabulary; the navigation task does not import Go2.

## Execution engine and hardware backends

Local model execution is orthogonal to the task pipeline:

```text
RawRequest
    │
    v
ModelAdapter ──> ModelPackage(entrypoints, ExecutionPlan)
                                  │
                                  v
InferenceRequest ──> ExecutionEngine ──> PlanRunner
                                              │
                                              v
                                       BackendSession
                                              │
                                              v
                                           device
```

`engine` owns `InferenceRequest`, `InferenceResult`, `ExecutionContext`,
`ProviderCapabilities`, the `InferenceProvider` protocol, queueing, dynamic
batching, cancellation, deadlines, metrics, memory admission, and plan-runner
selection. `engine/providers.LocalBackendProvider` composes a model adapter,
package, engine, and selected backend; `engine/providers.ProviderRegistry`
provides lazy Provider factories.

`backends` owns `DeviceInfo`, `CompileOptions`, `ArtifactVariant`,
`BackendSession`, memory statistics, compilation/loading, device transfer, and
hardware execution. `TorchCudaBackend` is the working CPU/CUDA implementation.
The Ascend and Horizon classes currently report structured unsupported results
and reserve their vendor-specific extension points; they do not yet compile or
run a model.

The engine depends only on model plans/packages and backend interfaces or
values. It does not dispatch on a model family or import a concrete backend.
Backends execute named package entrypoints and never choose task semantics.

An external Provider may bypass local execution. The GR00T vLLM-Omni adapter in
`integrations/gr00t` owns its OpenPI transport and reset/session state; it does
not construct `ExecutionEngine` or a local backend. Starting and supervising an
external model server is outside the Provider protocol.

## Model adapters and execution plans

Model semantics and execution pattern are independent axes:

| Adapter | Model family | Current execution plan |
| --- | --- | --- |
| `Pi05Adapter` | VLA | `IterativeFlowPlan` |
| `SmolVLAAdapter` | VLA | `SmolVLAFlowPlan` using the `iterative_flow` runner |
| `OpenVLAOFTAdapter` | VLA | `SingleForwardPlan` |
| `Gr00tN17Adapter` | VLA | `SingleForwardPlan` |
| `ToyFlowAdapter` | test fixture | `IterativeFlowPlan` |
| `ToySingleForwardAdapter` | test fixture | `SingleForwardPlan` |

Built-in adapters share the explicit cardinality path:

```text
preprocess_one(raw)       -> one logical request with tensor B=1
collate(B=1 samples)      -> one tensor payload with B=N
unbatch(B=N output, N)    -> exactly N outputs without the batch axis
postprocess_one(output)   -> one model-family result
```

`ModelPackage.plan` contains backend-neutral data and named entrypoints. The
engine chooses a runner by `plan.kind`. A runner receives the narrow
`RunnerHost` interface, so it can submit stages, update state, and observe safe
points without accessing queue internals.

For iterative flow, the runner executes encode, initialization, repeated model
steps, and finalization. When a model step returns velocity, the state update is
submitted through `BackendSession.add_scaled`:

```text
velocity = denoise_step(state, time, prefix)
state = state + dt * velocity
```

This keeps arithmetic on the concrete device while leaving cancellation and
memory safe points visible to the engine. Only single-forward and iterative-
flow runners are implemented today.

## Navigation policy and Go2 closed loop

The navigation task has one robot-independent policy interface:

```text
plan(NavigationRequest) -> WaypointPlan
```

The four current policies reach it through different model paths:

```text
Qwen HTTP endpoint ───────────────┐
StreamVLNRuntime ─────────────────┼─> policies.navigation
InternVLARuntime (DualVLN/NavDP) ─┤          │
NaVILARuntime ────────────────────┘          │
                                             v
                              tasks.navigation.WaypointPlan
```

Qwen uses validated structured output from an already-running
OpenAI-compatible endpoint. StreamVLN uses
`models.vln.streamvln.StreamVLNRuntime`. InternVLA uses
`models.vla.internvla_n1.InternVLARuntime`; NavDP additionally requires
registered depth. NaVILA uses the released Llama 3 8B checkpoint against
[official source commit `76b98f2`](https://github.com/AnjieCheng/NaVILA/tree/76b98f233dd0fff05dfcd69435eec6740febff9d)
and exactly eight image slots: the policy retains the encoded episode,
uniformly samples seven historical positions plus the latest frame, and
left-pads missing early history with black images. Each policy owns episode
reset/serialization and
returns base-frame metric waypoints. StreamVLN models expose only normalized
native actions, InternVLA models expose only the official native output union,
and NaVILA exposes one textual mid-level action; their action/trajectory
geometry is interpreted in the policy layer.
Policies implement only the task `NavigationPolicy` protocol, not the engine
`InferenceProvider` protocol. None imports the engine, Go2, or sends motion
commands.

The application composition in `apps/navigation` chooses exactly one policy,
constructs the Go2 camera/control clients, selects a task session, and emits
JSON Lines events:

```text
Go2CameraClient.capture + Go2ControlClient.state
                    │
                    v
          NavigationObservation + instruction
                    │
                    v
       Qwen / StreamVLN / InternVLA / NaVILA policy
                    │
                    v
              WaypointPlan (base_link)
                    │
          ┌─────────┴──────────┐
          v                    v
NavigationSession      ReactiveNavigationSession
continuous follower    one bounded velocity pulse
          └─────────┬──────────┘
                    v
        PlanarVelocityCommand(vx, vy, yaw_rate)
                    │
                    v
             Go2ControlClient
                    │ HTTP velocity lease
                    v
       Go2 resident control service → Unitree SDK2
```

The application-owned policy lifecycle calls `prepare()` before constructing
and running the task session. This keeps local checkpoint loading and any
configured warmup outside the session's navigation deadline without
introducing model knowledge into `tasks/navigation`. Qwen's hook is
deliberately local-state-only because the external server has an independent
lifecycle.

NaVILA's accepted native vocabulary is deliberately bounded to `STOP`, forward
25/50/75 cm, and left/right 15/30/45 degrees. The policy maps forward distance
to `x_m`, left turn to positive yaw, right turn to negative yaw, and `STOP` to
an empty terminal `WaypointPlan`. The current Go2 application sizes NaVILA's
reactive lease duration from the configured Go2 speed so the 75 cm envelope is
not silently truncated, then executes the plan through the task controller and
Unitree SDK2 velocity lease.

This is not the complete low-level stack described by NaVILA. The official
two-level system pairs the high-level VLA with a learned real-time locomotion
policy for obstacle avoidance. This repository does not vendor or run the
official
[`legged-loco` policy](https://github.com/yang-zj1026/legged-loco/tree/87b0d3d18404e784abc0a62227bc41c940f29ecc);
therefore its Go2 mapping does not claim the learned obstacle avoidance,
terrain handling, or whole-body behavior of the paper.

`NavigationSession` anchors each capture-time waypoint plan in odometry. Its
inference loop may replace the plan while the control loop samples the current
`WorldWaypointFollower` at the configured rate. This mode requires changing,
trustworthy planar pose state.

`ReactiveNavigationSession` is the default for StreamVLN and NaVILA under
`session.mode = "auto"`. It executes only the first waypoint as a bounded timed
pulse, stops, settles, captures a new frame, and replans. This is the
implemented fallback for the observed Go2 state stream whose x/y position may
remain constant while the robot walks.

Both sessions are observation-only unless the CLI receives `--execute`.
Execution first calls the Go2 preflight check, then uses `stream_move` and
`update_move` leases through the HTTP control service. Completion,
interruption, timeout, or failure sends stop. Robot-side hard bounds remain
independent of task-side clamping.

## Distributed coordination

Cross-runtime coordination is separate from single-endpoint scheduling. The
engine orders work inside one endpoint; `distributed` chooses how independent
endpoint results contribute to a control decision.

```text
tick N:   submit cloud request N ----------------------+
          await edge N -> edge result N                | no waiting
                                                       v
tick N+k: await edge N+k -> edge result N+k + cached cloud result N
                                      |
                                      v
                         select or synchronously fuse
```

`AsyncFailoverCoordinator` keeps at most one cloud request in flight, applies
submission timeout, TTL, sequence-lag, disconnect, and connection-epoch
guards, and never blocks the edge result on cloud completion.

| Mode | Cloud submission | Tick output |
| --- | --- | --- |
| `edge_only` | no | current edge result |
| `async_cloud_preferred` | background | fresh cached cloud result, otherwise edge |
| `async_blend` | background | current edge result plus a fresh cloud result through a fuser |

The current fuser verifies matching shapes, copies the cloud action to the edge
device/dtype, and computes `edge + (cloud - edge) * cloud_weight`. Failure or
shape mismatch returns the current edge result. SmolVLA/π0.5 deliberately does
not enable numeric blending because its native `(50, 6)` and `(50, 32)` actions
do not share an action space.

The two-host smoke transport uses a four-byte big-endian length followed by a
UTF-8 JSON object, with a 64 MiB message limit and one connection per request.
It proves process/host separation; it does not provide authentication,
encryption, persistent streaming, discovery, or model negotiation.

## Multi-robot shared Provider

The N:1 prototype keeps robot state and cloud model ownership separate:

```text
RobotEdgeRuntime A: local Provider A + cloud session A --\
RobotEdgeRuntime B: local Provider B + cloud session B ----+--> distributed.multitenant
RobotEdgeRuntime C: local Provider C + cloud session C --/      one shared Provider/model
```

`RobotSessionIdentity` distinguishes stable robot and edge-node IDs from an
episode-scoped session ID. `distributed/multitenant.MultiTenantInferenceService`
binds each session to one robot/edge identity, serializes requests within the
session, validates sequence and action-space metadata, namespaces request IDs,
and allows independent sessions to make progress.

The safe default remains one pending request per session and
`max_batch_size = 1`. Cross-session batching is an explicit opt-in for a
Provider declared multi-tenant-safe and observations known to share schema and
shape. A cloud outage affects all cloud sessions, but every robot retains its
own local Provider and failover coordinator. A rejection or reconnect in one
session invalidates only that robot's cached cloud authority.

The checked-in CLI currently builds a local Torch Provider for this topology.
It does not select arbitrary external providers from configuration. A stateful
OpenPI Provider requires one client instance per robot session even when those
clients share one external server.

## CUDA Graph ownership

Explicit CUDA Graph capture is private to `backends/torch_cuda`. Callers select
ordinary package entrypoint names with
`CompileOptions.options["cuda_graph_entrypoints"]`; tasks, policies, the engine,
and model adapters do not import CUDA graph machinery.

The backend session owns static buffers, capture streams, graphs, cache keys,
fallback state, replay serialization, and teardown. Cache identity includes
tree structure, tensor shape/stride/dtype/device, and Python constants. Selected
top-level scalar inputs may be materialized at fixed addresses through
`cuda_graph_dynamic_scalar_inputs`; π0.5 uses that mechanism for denoise time so
same-shaped steps can share a graph. Capture/replay failures fall back to eager
for the affected signature.

## Current limits

- Only single-forward and iterative-flow execution plans have engine runners.
- A loaded in-memory Torch module has one active device session; replicas need
  separate packages, and the first successful load fixes its dtype policy.
- OpenVLA-OFT currently performs one greedy causal forward for one RGB camera;
  prefix-KV splitting, rollout log probabilities, multi-camera layouts, and
  CUDA Graph capture are not implemented.
- The multi-robot JSON path carries numeric observations but is not an
  efficient RGB/depth tensor codec.
- Runtime registration, routing, and transport remain prototype surfaces; the
  TCP paths lack production security and persistent streaming.
- Cached cloud actions are bounded by TTL and sequence lag, but task-specific
  policy must still decide whether an older action remains meaningful.
- Shutdown is cooperative. A native backend call that never returns requires a
  process supervisor for a hard termination deadline.
- Physical actuation still requires an external operator and safety process;
  `--execute` is an explicit authorization gate, not a safety certification.
