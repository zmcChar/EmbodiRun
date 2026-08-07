# Distributed runtime boundary

This domain owns registration and communication among cloud,
edge, and robot runtimes:

- `registry/`: leases, heartbeats, capability advertisement;
- `communication/`: direct and brokered transports;
- `routing.py`: endpoint placement;
- `failover.py`: asynchronous result authority, staleness, and recovery.

Preemption here operates on results from distributed runtimes, not on
single-device kernels. The first implementation keeps an edge endpoint hot,
allows a fresh cloud result to take output authority, and invalidates that
authority on disconnect or connection-epoch change.

The executable multi-robot slice adds a direct session handshake over the
prototype TCP/JSON link:

```text
N × (RobotSessionIdentity + edge Provider + failover coordinator)
                              |
                              v
             one MultiTenantInferenceService / cloud Provider
```

`robot_id`, `edge_node_id`, `session_id`, `sequence_id`, and `observation_id`
are carried end to end. The cloud namespaces request IDs and prevents one
session from overriding another. Each edge owns its connection epoch and
failover cache, so a rejection or disconnect confined to robot A does not
change robot B. An outage of the one shared cloud process or network path
affects all cloud sessions, while every independent edge process continues
through its own local Provider.

The safe default isolates requests serially: one pending request is admitted per
session, and the shared local Provider uses `max_batch_size = 1`. Cross-robot
batching is an explicit opt-in for a Provider declared `multi_tenant_safe` and a
deployment that already guarantees compatible model-owned observation schemas
and shapes.

This handshake is intentionally not called dynamic node registration. The
cloud address is static and service-local; the leased `RuntimeRegistry`,
heartbeats, discovery, and route selection remain future distributed-runtime work.

`physical_resource_id` is only advertised metadata. It does not reserve or
schedule hardware, and a selector such as `cuda:0` is meaningful only inside
the process that opens the device. The length-prefixed JSON path transports
numeric observations for topology tests, not as a production visual/tensor
codec.

The current multi-robot CLI constructs only a local `torch_cuda` Provider. It
supports a checkpoint-free target vector and adapter-owned synthetic
observations for real-model topology tests; physical sensor mapping is outside
this domain. vLLM-Omni remains available through the generic Provider boundary,
but its OpenPI integration owns WebSocket and reset/session state. Use one
Provider/WebSocket instance per robot session; those instances may all connect
to the same shared vLLM-Omni server.

Distributed code may depend on public envelopes owned by `engine`, `models`,
`tasks`, and `robots`. It must not depend on model-family internals, policy
implementations, simulator details, or vendor robot SDK APIs.
