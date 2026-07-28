# Prototype architecture

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
Pi05Adapter / OpenVLAOFTAdapter / ToyFlowAdapter / ToySingleForwardAdapter
```

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

## Asynchronous cloud/edge failover

The first cross-runtime policy is result-level preemption. It is deliberately
outside `ExecutionEngine`: engine priority orders work within one endpoint,
whereas failover chooses which endpoint owns the output at a control decision
point.

```text
tick N observation
      |
      +----> edge Engine ---- await ----> edge result N -----+
      |                                                       |
      +----> dummy/remote link -> cloud Engine                 |
                    background only                           |
                            |                                 v
                            +-> latest fresh cloud result -> selector -> action
```

The selector never waits for cloud. Edge inference remains hot even while a
cloud result has output authority. Only one cloud request may be in flight, and
an in-flight timeout prevents a hung request from occupying that slot forever.
Cloud results are guarded by both a TTL measured from submission and a maximum
sequence lag. Disconnect or connection-epoch change immediately invalidates
cached authority, preventing an old response from taking over after
reconnection.

`AsyncFailoverCoordinator` accepts distinct edge and cloud request payloads.
This keeps model-specific preprocessing outside the policy and allows two
different adapters to produce the same normalized action contract. The current
mode is configurable as `async_cloud_preferred` or `edge_only`; hierarchical
guidance is intended as a separate composition policy rather than another
branch inside the failover selector.

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
