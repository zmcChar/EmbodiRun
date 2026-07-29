# Prototype architecture

## Inference-provider boundary

Group 3 exposes one minimal interface for complete inference endpoints:

```text
ProviderCapabilities
infer_async(InferenceRequest) -> InferenceResult
aclose()
```

It deliberately does not duplicate scheduling, hardware execution, or
framework-specific session operations:

```text
                         ┌─ LocalBackendProvider
ModelAdapter / request ──┤       ↓
                         │  ExecutionEngine → BackendRegistry → BackendSession
                         │
                         └─ external Provider
                                 ↓
                            vLLM-Omni / MaaS
```

`LocalBackendProvider` receives a fourth-group `BackendRegistry`; it never
constructs a CUDA backend internally. `VllmOmniGr00tProvider` instead connects
to an existing OpenPI service and never imports or invokes a local Backend.
Provider factories are lazy so listing an option does not load weights or open
a connection. Group 2 separately owns runtime-node identity, addresses,
leases, and routing.

Operations such as OpenPI `connect` and `reset` are concrete-provider
extensions rather than requirements on every local, remote, or MaaS Provider.
Starting and supervising an external `vllm serve` process is likewise outside
the Provider protocol.

## Concrete vertical slice

```text
π0.5 checkpoint
      |
      v
Pi05Adapter (Group 1)
      |  ModelPackage:
      |  entrypoints + IterativeFlowPlan
      v
ExecutionEngine (Group 3)
      |  priority, lifecycle, batching, memory policy
      v
IterativeFlowRunner (Group 3)
      |  encode / initialize / step × N / finalize
      v
BackendSession contract
      |
      v
TorchCudaBackend (Group 4)
      |  device move, eager/compile execution, device memory statistics
      v
CPU or NVIDIA CUDA device
```

The iterative-flow plan declares the stage names, step count, safe points, and
whether a step returns velocity or the next state. Its runner owns the
corresponding control flow:

```text
velocity = denoise_step(state, time, prefix)
state = state + dt * velocity
```

The arithmetic itself is submitted through `BackendSession.add_scaled`, so
opaque vendor tensors never need host-language operators. This is deliberately
visible to the engine so a request can be cancelled or cooperatively preempted
between steps. The model group owns the schedule and model semantics; the
backend owns execution and completion fences on a concrete device.

## Adapter and plan axes

Model semantics and execution pattern are deliberately independent:

| Concrete adapter | Semantic family | Execution plan |
| --- | --- | --- |
| `Pi05Adapter` | VLA | `IterativeFlowPlan` |
| `SmolVLAAdapter` | VLA | `SmolVLAFlowPlan` (`iterative_flow`) |
| `OpenVLAOFTAdapter` | VLA | `SingleForwardPlan` |
| `ToyFlowAdapter` | VLA fixture | `IterativeFlowPlan` |
| `ToySingleForwardAdapter` | VLA fixture | `SingleForwardPlan` |

The public `ModelAdapter[RequestT, ResultT]` is a structural protocol. External
plugins do not need to inherit a repository class. Built-in adapters can reuse:

```text
ModelAdapter protocol
        ^
BaseModelAdapter
        ^
VLAAdapterBase
        ^
Pi05Adapter / SmolVLAAdapter / OpenVLAOFTAdapter / fixtures
```

`SmolVLAAdapter` is another sibling under the same VLA family. Its
`SmolVLAFlowPlan` only specializes the timestep schedule needed to reproduce
LeRobot 0.3.3's FP32 scalar accumulation exactly; it retains the generic
`iterative_flow` kind and therefore uses the same Group-3 runner. π0.5 keeps
its analytic Python schedule. The Engine still owns iteration and the Backend
still owns arithmetic on the selected device.

OpenVLA-OFT follows the same family base but selects a different plan:

```text
OpenVLA-OFT checkpoint
        |
        v
OpenVLAOFTAdapter (Group 1)
        |  ModelPackage: forward + SingleForwardPlan
        v
ExecutionEngine / SingleForwardRunner (Group 3)
        |
        v
TorchCudaBackend (Group 4)
```

The cardinality contract is explicit:

```text
preprocess_one(raw)       -> one logical request with tensor B=1
collate(B=1 samples)      -> one tensor payload with B=N
unbatch(B=N output, N)    -> exactly N outputs without the batch axis
postprocess_one(output)   -> one family-specific result
```

`ModelPackage.plan` contains only backend-neutral data and entrypoint names.
The engine selects a runner by `plan.kind`. Registry factories receive the
narrow `RunnerHost` protocol rather than `ExecutionEngine`; runners can submit
stages, update state, and observe safe points without depending on queue
internals or private engine methods. All model execution still crosses
`BackendSession`. Adding an autoregressive or stateful-rollout plan therefore
requires a shared plan contract plus a Group-3 runner, not changes to every
backend.

## Temporary compatibility surface

`FlowRecipe`, the read-only `ModelPackage.recipe` property, and the legacy VLA
`preprocess`/`postprocess` wrappers exist only to ease migration inside this
prototype. Model implementations have one canonical family-qualified path:
π0.5 lives only under `models.vla.pi05`. New code must use
`IterativeFlowPlan`, `ModelPackage.plan`, and the four cardinality-explicit
adapter methods.

In particular, the legacy two-dimensional VLA `postprocess` input is ambiguous:
it may mean `[batch, action_dim]` or `[horizon, action_dim]`. The new
`unbatch → postprocess_one` path has no such ambiguity. These shims are planned
for removal before the first stable external API rather than becoming a second
supported interface.

## Future composition

```text
RLinf / other trainer / MaaS
              |
 cloud runtime+---------------- edge runtime (optional)
              |                         |
              +---------- network ------+
                                        |
                                  robot runtime
                           local model + safety + SDK
```

`distributed` chooses a runtime endpoint. `engine` schedules within one
endpoint. `robots` maps model-independent actions into a physical control
interface.

## Asynchronous cloud/edge coordination

Cross-runtime coordination is deliberately outside `ExecutionEngine`: engine
priority orders work within one endpoint, whereas the coordinator chooses how
results from independent endpoints contribute at a control decision point.

```text
tick N:   submit cloud request N ----------------------+
          await edge N -> edge result N                | no waiting
                                                       v
tick N+k: await edge N+k -> edge result N+k + cached cloud result N
                                      |
                                      v
                         select or synchronously fuse
```

The control tick never waits for cloud. Edge inference remains hot even while a
cloud result has output authority. Only one cloud request may be in flight, and
an in-flight timeout prevents a hung request from occupying that slot forever.
Cloud results are guarded by both a TTL measured from submission and a maximum
sequence lag. Disconnect or connection-epoch change immediately invalidates
the cached result, preventing an old response from taking over after
reconnection.

| Mode | Cloud submission | Tick output |
| --- | --- | --- |
| `edge_only` | No | Current edge result |
| `async_cloud_preferred` | Background | Fresh cached cloud result, otherwise edge |
| `async_blend` | Background | Current edge plus fresh cached cloud through a fuser |

`AsyncFailoverCoordinator` accepts distinct edge and cloud request payloads.
This keeps model-specific preprocessing outside the policy and lets two
different adapters target one common result contract. The action fuser copies
the cloud action to the current edge action's device and dtype, verifies equal
shapes, and computes:

```text
edge + (cloud - edge) * cloud_weight
```

Shape mismatch, an invalid fuser, or any cloud failure returns the current edge
result. The fuser itself must be synchronous, lightweight, free of I/O, and
must not mutate either input. Hierarchical guidance remains a separate future
composition policy rather than another branch inside this selector.

The SmolVLA/π0.5 prototype intentionally permits only `edge_only` and
`async_cloud_preferred`. SmolVLA emits robot-specific `[50, 6]` actions after
checkpoint unnormalization, while the current π0.5 checkpoint emits `[50, 32]`.
The coordinator can transfer output authority between endpoints, but it cannot
make those action spaces semantically compatible by padding or averaging.

### Prototype TCP transport

The real two-host smoke path uses a four-byte big-endian payload length followed
by a UTF-8 JSON object. Messages are limited to 64 MiB, each request opens one
TCP connection, and the π0.5 server returns the final FP32 action array. This
transport proves a physical network boundary without pretending to be the
production communication layer.

## Multi-robot shared-cloud serving

The N:1 prototype separates robot state from cloud model ownership:

```text
RobotEdgeRuntime A: edge Provider A + coordinator A + cloud session A --\
RobotEdgeRuntime B: edge Provider B + coordinator B + cloud session B ----+
RobotEdgeRuntime C: edge Provider C + coordinator C + cloud session C --/
                                                                         |
                                                   MultiTenantInferenceService
                                                                         |
                                                         one cloud Provider
                                                         one model/queue
```

`RobotSessionIdentity` distinguishes a stable `robot_id`, a logical
`edge_node_id`, and an episode-scoped `session_id`. A device selector such as
`cuda:0` is process-local and is never used as node identity. Colocated edge
processes may advertise the same `physical_resource_id`, but this is only a
declarative label in the prototype. It does not discover, lock, schedule, or
arbitrate the underlying device.

The robot runtime sends one `RawRequest` observation with a contiguous,
session-local `sequence_id` starting at 1, so coordinator freshness and wire
observation order use the same clock. `MultiTenantInferenceService`:

- binds a session ID to exactly one robot and edge identity;
- serializes requests within one session while allowing cross-session
  concurrency;
- prefixes external request IDs before submitting them to the shared Provider;
- overwrites identity metadata with server-authoritative values;
- validates the declared embodiment and action-space contract;
- echoes robot, session, sequence, and observation identity on the result; and
- rejects duplicate or out-of-order session sequences.

Each edge keeps its own `AsyncFailoverCoordinator`. Therefore an endpoint
epoch change clears only that robot's cloud authority. The cloud service has no
shared failover cache and does not choose robot actions. A registration or
transport failure confined to one session does not invalidate another robot's
cached result. Conversely, an outage of the one shared cloud process or its
network path makes every cloud session unavailable; the independent edge
Providers remain the per-robot availability baseline.

The safe default is serial session isolation: at most one pending request is
admitted per session, and the shared local Provider uses
`max_batch_size = 1`. Cross-session dynamic batching is an explicit opt-in for
a Provider declared `multi_tenant_safe` and a trusted deployment whose
model-owned observations are shape- and schema-compatible. The generic engine
does not yet infer that compatibility from arbitrary robot payloads.

Registration in this slice is a direct service-local handshake to one static
endpoint, not the future leased discovery plane or dynamic `RuntimeRegistry`.
Per-request TCP connections and JSON numeric observations deliberately keep the
first topology test small. The JSON path demonstrates real observation
transfer, but is not a production RGB/depth/tensor codec. The default listener
is loopback-only; the protocol has no authentication or encryption, so a
non-loopback bind is limited to an isolated test network.

The coordinator invokes the cloud request factory only when it admits a
submission. That factory snapshots mutable tensor/array storage before the
caller may reuse a frame; JSON conversion, network I/O, and cloud inference are
background work. The snapshot copy is still local control-path cost and must be
measured. A capture layer may instead pass an immutable/reference-counted
`owned_cloud_observation`, avoiding the defensive copy on the control path.

The executable multi-robot CLI currently constructs only a local `torch_cuda`
Provider. Its default target-vector observation keeps checkpoint-free tests
small; `adapter_synthetic` delegates model-shaped test input construction to
the loaded adapter and is used by the real SmolVLA four-model deployment.
Physical cameras and state still require a robot-owned mapper. Other Provider
types remain composition options rather than CLI choices. In particular, the
vLLM-Omni/OpenPI Provider owns WebSocket and reset state. A multi-robot
deployment therefore needs one such Provider/WebSocket per robot session,
while all of those independent clients may share the same vLLM-Omni server.

## CUDA Graph boundary

Explicit CUDA Graph capture is private to `backends/torch_cuda`. A caller
selects ordinary package entrypoint names through
`CompileOptions.options["cuda_graph_entrypoints"]`; neither the Engine nor a
model adapter imports CUDA. The session owns static input/output buffers,
capture streams, graph instances, fallback state, and teardown.

Graphs are cached by the exact TensorTree structure, tensor
shape/stride/dtype/device, and Python constant values. Tensor values are copied
into stable buffers before replay. A composition may explicitly list
top-level Python scalar inputs in
`cuda_graph_dynamic_scalar_inputs`; the backend materializes each one as a
fixed-address zero-dimensional CUDA tensor and refreshes its value before
replay. π0.5 uses this only for `denoise_step.time`, allowing all ten
same-shaped flow steps to share one graph. Undeclared Python values remain
capture-time constants.

Returned tensors are cloned so later replays cannot mutate an earlier result.
Capture and replay are serialized inside the session. Allowlisted entrypoints
must be side-effect-free because capture begins with an ordinary warmup call.
Non-zero-offset tensor views or leaves that share storage conservatively fall
back to eager instead of changing alias semantics. The first slice deliberately
keeps RNG initialization, engine-owned state updates, and plan-level control
flow outside the graph.

## Deliberate first-slice limits

- One synchronous `engine.infer` call represents one logical request. The
  π0.5 CLI therefore fixes `batch_size=1`; dynamic batching takes separate
  requests plus the model-owned `collate`/`unbatch` pair.
- Only single-forward and iterative-flow plans have runners today.
  Autoregressive and stateful-rollout plans will be added with their first real
  model rather than specified speculatively.
- The first OpenVLA-OFT slice submits one complete causal forward and decodes
  actions greedily. Prefix-KV splitting, rollout log-probability output, and
  CUDA Graph capture remain later engine/backend work rather than model-ID
  branches in the shared runtime. Its processor accepts one RGB camera;
  multi-image checkpoint configurations fail at package construction rather
  than reaching the vision tower with an invalid channel count.
- A loaded in-memory Torch model has one active device session. Multi-device
  replicas must be built as separate packages until immutable/shareable
  artifacts are introduced. The first successful load also fixes that module
  object's dtype policy (`preserve` or one explicit dtype) for its lifetime;
  changing dtype likewise requires a distinct module replica.
- The Torch backend currently hosts complete staged callables. Its eager/SDPA
  attention operators are parity-tested extension fixtures but are not yet
  injected into π0.5; portable operator slots or an exported IR are the next
  Group 1–Group 4 handoff.
- π0.5 still reuses LeRobot's image preprocessing helper, which follows the
  policy's current device. A production robot input pipeline should make host
  preprocessing and backend transfer two explicit stages.
- Current coordination composes complete model results; it does not split one
  neural network layer-by-layer across hosts.
- The formal SmolVLA/π0.5 path preserves distinct `(50, 6)` and `(50, 32)`
  actions and forbids numeric fusion. The shared-cloud handshake now checks a
  declared action-space ID, action dimension/horizon, and embodiment, but units,
  coordinate frames, and physical robot capabilities are not yet a general
  result contract.
- A cached cloud action may come from an earlier tick. TTL and sequence lag
  bound its age. The multi-robot path now preserves the originating
  `observation_id`, but policy-specific logic must still decide whether that
  older action remains valid for the current control tick.
- The legacy π0.5 smoke server still owns one pre-generated synthetic
  observation. The multi-robot service carries each client's JSON numeric
  observation; an efficient, dtype/shape-aware RGB/depth binary codec remains
  future work.
- The multi-robot TCP handshake provides session registration but not leased
  runtime discovery. TCP/JSON still lacks authentication, encryption,
  compression, persistent connections, streaming, and model-version
  negotiation.
- The synchronous fuser's device copy and arithmetic execute inside the
  decision path. Their cost is small in the recorded smoke run but must be
  budgeted explicitly in a fixed-frequency control loop.
- The multi-robot server drains or cancels active handlers during shutdown,
  but a client disconnect does not yet cancel work already admitted to the
  shared Provider.
- Shutdown grace is cooperative, not a hard kill boundary. A native backend
  call that does not return cannot be safely terminated in-process; production
  deployment needs a process supervisor with a hard termination deadline.
- RLinf, robot SDKs, physical actuators, and a safety supervisor are not yet
  connected. The RTX 3070 laptop represents an edge node, not an embedded
  on-robot deployment target.
