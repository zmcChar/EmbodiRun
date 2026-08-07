# Domain ownership

| Directory | Owner | Responsibility |
| --- | --- | --- |
| `src/embodied_runtime/models/` | Model domain | Model semantics, adapter I/O, packages, execution plans, native outputs, reference parity |
| `src/embodied_runtime/policies/` | Policy domain | Model-coupled translation into task-owned plans; no physical actuation |
| `src/embodied_runtime/tasks/` | Task domain | Robot-independent goals, observations, plans, interfaces, controllers, and closed loops |
| `src/embodied_runtime/robots/` | Robot domain | Physical drivers, SDK bindings, limits, host clients, and robot-resident services |
| `src/embodied_runtime/simulators/` | Simulator domain | Simulator endpoints, mappings, traces, and privileged evaluation helpers |
| `src/embodied_runtime/engine/` | Inference engine | Provider API, request lifecycle, plan runners, batching, cancellation, metrics, and local Provider composition |
| `src/embodied_runtime/backends/` | Hardware backend | Device discovery, compilation, loaded sessions, device execution, memory, and operators |
| `src/embodied_runtime/distributed/` | Distributed runtime | Communication, routing, failover, session identity, registration, and multi-tenant isolation |
| `src/embodied_runtime/deployment/` | Deployment tooling | Installation plus process/service lifecycle for physical robot agents |
| `src/embodied_runtime/evaluation/` | Evaluation domain | Trial records, metrics, reports, and benchmark comparisons |
| `src/embodied_runtime/integrations/` | External integration | GR00T/vLLM-Omni, LeRobot, planning, and RLinf framework adapters |
| `src/embodied_runtime/apps/` | Application composition | Runnable roots that wire the required domains together |
| `src/embodied_runtime/utils/` | Utility primitives | Small geometry, HTTP, encoding, and image helpers with no domain policy |

Runtime decision flow:

```text
models → policies → tasks → robots / simulators
```

Observations return in the opposite direction. Source dependencies follow
ownership: consumers import values and interfaces from their owner. A concrete
robot imports task-owned command/interface types; a task never imports a
concrete robot. `engine` and `backends` meet only through model packages,
backend interfaces/values, and execution context. `apps` is the broad
composition root.

Generic local Provider construction and registration belong to
`engine/providers`. Shared-session isolation belongs to
`distributed/multitenant`. The external GR00T/OpenPI adapter belongs to
`integrations/gr00t`; an external Provider does not pass through a local
hardware backend.

See [`docs/architecture.md`](docs/architecture.md) for the detailed dependency
rules and runtime flows.
