# EmbodiRun

**English** | [简体中文](README.zh-CN.md)

### Embodied AI, Ready to Run.

**An Efficient Deployment & Execution Runtime for Embodied AI.**

**Deploy Models. Accelerate Inference. Run Robots.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-embodirun.readthedocs.io-informational.svg)](https://embodirun.readthedocs.io/)

EmbodiRun is the deployment and execution runtime for embodied AI. It connects
model inference, service deployment, cross-node communication, and robot
execution into one reproducible system: configure a model, a compute node, and
a robot or simulator, then start the services and run a task.

High-performance inference is provided by
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer), which stays an
independent engine and can also be used on its own.

> **Build embodied applications, not integration code.**

---

## Why EmbodiRun

- **Separate control from compute.** Run the robot-facing Control process near
  the robot and place inference on a Jetson, a workstation, or a remote GPU.
  The two sides are configured and maintained independently.
- **Reuse a stable execution path.** Observation → inference request → action
  mapping → bounded execution → feedback is implemented once, with action
  validation, control arbitration, manual takeover, and a software stop.
- **Plug in agents and backends.** A dependency-free public client
  (`observe` / `propose` / `execute` / `inspect` / `cancel` / `stop`) lets an
  agent framework keep its own planning loop, and inference backends are
  registered as replaceable providers.
- **Reproduce a deployment.** A single YAML describes nodes, environments,
  services, robots, sensors, and bindings. The Host CLI validates, prepares,
  starts, and stops them.

EmbodiRun does **not** own checkpoints, model frameworks, prompt construction,
or model-output parsing. It talks to inference services over a versioned HTTP
(or optional WirelessComm) API.

---

## Quick Start

EmbodiRun is installed from source with [uv](https://docs.astral.sh/uv/). The
public lock file does not require any private repository.

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen            # core CLI + development tools
uv run pytest -q            # verify the checkout (CPU only)
```

Every process installs only the capability group it needs:

```bash
uv sync --frozen --no-dev --group host            # SSH orchestration / Host CLI
uv sync --frozen --no-dev --group robot-so101     # SO-101 control
uv sync --frozen --no-dev --group robot-fr3       # Franka FR3 control
uv sync --frozen --no-dev --group robot-arx5      # ARX5 control
uv sync --frozen --no-dev --group sim-libero      # LIBERO simulation
```

Validate a deployment file without contacting any node:

```bash
uv run embodirun --config configs/http-wireless-inference/http.yaml validate
```

### No robot required

Start a robot-only Control service backed by simulated joints and a fake
camera. It opens no hardware, exposes the real Host JSON API, and is a good
first look at the observe/execute path:

```bash
uv run embodirun --config examples/shared-device-fake.yaml validate
uv run embodirun --config examples/shared-device-fake.yaml init
uv run embodirun --config examples/shared-device-fake.yaml up
uv run embodirun --config examples/shared-device-fake.yaml describe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config examples/shared-device-fake.yaml down
```

See [`examples/README.md`](examples/README.md) for the full walkthrough,
including observation, media, execution, and recording.

### A real deployment

1. Copy an example configuration and replace the placeholders (device paths,
   calibration, checkpoint, node addresses).
2. Run the lifecycle:

```bash
uv run embodirun --config my-deployment.yaml validate   # static checks
uv run embodirun --config my-deployment.yaml probe      # connectivity and tools
uv run embodirun --config my-deployment.yaml init       # environments and sources
uv run embodirun --config my-deployment.yaml up         # start services
uv run embodirun --config my-deployment.yaml run \
  --runtime so101-1-runtime \
  --prompt "Pick up the cube and put it into the bowl." \
  --chunk-steps 10
uv run embodirun --config my-deployment.yaml down
```

`run` moves the selected physical robot. It submits one bounded action chunk by
default; `--max-steps`, `--chunk-steps`, and `--control-hz` bound execution.
Every returned row is still subject to the joint and gripper step limits in the
configuration. Clipping is a per-row rate limit, not collision avoidance.

> **Working with a real robot?** Service startup does not connect or move the
> arm, but `run` does. Read [`docs/control.md`](docs/control.md) and
> [`docs/safety.md`](docs/safety.md) first, keep an operator present, and keep
> the physical emergency stop within reach.

More detail: [`docs/installation.md`](docs/installation.md) and
[`docs/quickstart.md`](docs/quickstart.md).

---

## Inference performance

EmbodiRun measures three layers separately so each number keeps its scope:
model inference, the Runtime link, and the application task. Inference is
executed by EmbodiInfer.

One recorded π0.5 offline result:

| Condition | Result |
|---|---|
| Hardware / config | RTX 4090, B=1, BF16, 10 denoising steps |
| Data | LIBERO-10, 1,600 offline observations |
| Mean end-to-end inference latency | **74.33 → 38.89 ms** |
| Throughput | **13.45 → 25.71 observations/s** |

Timing runs from decoded CPU input to CPU action output; it excludes camera
capture, network communication, robot execution, model loading, and the first
warm-up. This is a historical, scoped result against the original
EmbodiInfer/VVLA path, not a claim about every device or task. See
[`EmbodiInfer/benchmarks/pi05-benchmark`](https://github.com/BUAA-CI-LAB/EmbodiInfer)
for the full conditions.

---

## Support status

Support is recorded per **complete combination**, not per component. A single
model, robot, or platform being supported does not mean they work together.

- **Tested** — covered by the repository test suite at the stated level
  (software interface, simulation, or offline model). Real-robot validation
  records are maintained by the deployment owners.
- **Experimental** — implementation is provided as a work-in-progress path with
  known limits.
- **Planned** — no implementation yet.

| Combination | Implementation | Verification in this repository | Public recipe |
|---|---|---|---|
| π0.5 + SO-101 / Bi-SO-101 | Control, bindings, deployment config | Tested — software (CPU suite); offline action checks exist | `configs/pi05/bi-so101-vvla.yaml` |
| LIBERO + π0.5 | Simulator adapter, SGLang and VVLA configs | Tested — software; closed loop needs a GPU and checkpoint | `configs/simulation/libero-pi05-*.yaml` |
| Habitat + StreamVLN | Simulator adapter and config | Experimental | `configs/simulation/habitat-streamvln-vvla.yaml` |
| VLABench + π0.5 | Simulator adapter and config | Experimental | `configs/simulation/vlabench-pi05-*.yaml` |
| Isaac Sim + StreamVLN | Simulator adapter and config | Experimental (requires the NVIDIA Isaac Sim EULA) | `configs/simulation/isaac-streamvln-vvla.yaml` |
| Unitree Go2 + StreamVLN | Robot agent, binding, SSH deployer | Experimental | `src/embodirun/robots/unitree/go2` |
| SGLang HTTP backend | Provider, client, adapter | Tested — software (skipped without `sglang`) | `[sglang]` extra |
| Multi-node shared inference | Config, benchmark, control arbitration | Tested — software benchmark | `configs/http-wireless-inference/` |
| RPent agent adapter | `agents/rpent` public-client session | Experimental, software only | `agents/rpent/README.md` |
| Astra + π0.5 review loop | `agents/astra_pi05` examples | Experimental, mock reviewer | `agents/astra_pi05/README.md` |
| XLeRobot external owner | `integrations/xlerobot_owner` | Experimental, separately installed | `integrations/xlerobot_owner/README.md` |

"Tested — software" means the path is covered by automated tests in this
repository. It does **not** mean a physical robot completed a task.

---

## Architecture

```text
Application / RPent / custom agent
        │  public client: observe / propose / execute / inspect / cancel / stop
        ▼
EmbodiRun
├─ Deployment runtime  — configuration, environments, nodes, service lifecycle
├─ Application runtime — jobs, proposals, execution coordination, model loop
├─ Device runtime      — connection ownership, shared observations, arbitration
├─ Model services      — versioned inference contracts and provider registry
└─ Robot / simulator adapters + policy bindings
        │  HTTP or WirelessComm (versioned policy API)
        ▼
EmbodiInfer (or an external backend such as SGLang)
```

Process boundaries are explicit: the **Host** runs on the operator machine,
**Control** runs beside the robot and owns the hardware, and the **Inference**
service performs model computation. They can share a machine or be split across
nodes. Nodes are selected by configuration; automatic placement is future work.

```text
src/embodirun/
├── client/          # public Agent HTTP client
├── deployment/      # config, planning, state, SSH/process lifecycle
├── application/     # jobs, proposals, execution coordination, runtime loop
├── devices/         # connection ownership, shared observations, arbitration
├── model_services/  # inference contracts, protocols, providers
├── services/        # CLI/HTTP entrypoints and compatibility imports
├── robots/          # robot adapters
├── bindings/        # policy-to-robot mappings and safety horizons
└── simulators/      # simulator adapters
```

---

## Agent integration

An agent keeps its own planning loop and calls the Control service through the
public client:

```python
from embodirun.client import ControlClient

client = ControlClient("http://127.0.0.1:8100")
observation = client.observe(runtime="so101-1-runtime", session_id="task-1")
proposal = client.propose(runtime="so101-1-runtime", session_id="task-1",
                          prompt="Pick up the cube.")
job = client.execute(runtime="so101-1-runtime", session_id="task-1",
                     proposal_id=proposal.proposal_id, chunk_steps=10)
result = client.inspect(job.job_id)
```

The client is dependency-free and documented in
[`agents/CLIENT.md`](agents/CLIENT.md). `observe` and `propose` never move the
robot; `execute` goes through the same authorization, validation, and safety
checks as the CLI. An accepted request is not task success — read the returned
execution evidence and the next observation.

Reference adapters live in [`agents/`](agents/README.md):
[`agents/rpent`](agents/rpent/README.md) for an RPent-style agent and
[`agents/astra_pi05`](agents/astra_pi05/README.md) for a review loop. They are
software examples, outside the installed core package.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Install, extras, optional robot/simulator SDKs |
| [`docs/quickstart.md`](docs/quickstart.md) | First deployment walkthrough |
| [`docs/configuration.md`](docs/configuration.md) | Deployment YAML reference |
| [`docs/control.md`](docs/control.md) | Manual control, arbitration, software stop |
| [`docs/http_api.md`](docs/http_api.md) | Inference service API v1 |
| [`docs/pi05-bi-so101.md`](docs/pi05-bi-so101.md) | Dual SO-101 π0.5 deployment guide |
| [`docs/safety.md`](docs/safety.md) | Safety limits and operator checklist |
| [`docs/architecture.md`](docs/architecture.md) | Runtime domains and process boundaries |
| [`docs/support-matrix.md`](docs/support-matrix.md) | Full support status |

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Model checkpoints, datasets,
robot SDKs, and simulators keep their own licenses and are not distributed here.

---

**EmbodiRun — Embodied AI, Ready to Run.**

**Deploy Models. Accelerate Inference. Run Robots.**
