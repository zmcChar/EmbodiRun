# Domain ownership

| Directory | Primary group | Responsibility |
| --- | --- | --- |
| `src/embodied_runtime/models/` | Group 1 | Model semantics, adapter I/O, package entrypoints, plan selection, reference parity |
| `src/embodied_runtime/distributed/` | Group 2 | Registration, communication, routing, leases, failover |
| `src/embodied_runtime/engine/` | Group 3 | Plan runners, scheduling, batching, lifecycle, memory policy |
| `src/embodied_runtime/backends/` | Group 4 | Device discovery, compilation, device execution, operators |
| `src/embodied_runtime/robots/` | Group 5 | Observation/action mapping, control loop, safety, watchdog |
| `src/embodied_runtime/contracts/` | Shared review | Cross-group interfaces; changes require affected groups to review |
| `src/embodied_runtime/apps/` | Integration | Composition roots; concrete domains meet only here |

Dependency rule:

```text
models ───────┐
distributed ──┤
engine ───────┼──> contracts
backends ─────┤
robots ───────┘

apps/integrations may compose the five domains.
```
