<p align="center">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" width="440">
</p>

<p align="center"><strong>Deployment and execution for embodied AI</strong></p>
<p align="center">
  <a href="https://embodirun.readthedocs.io/">Documentation</a> ·
  <a href="docs/quickstart.md">Quick start</a> ·
  <a href="#demos">Demos</a> ·
  <a href="docs/support-matrix.md">Support matrix</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Documentation](https://readthedocs.org/projects/embodirun/badge/?version=latest)](https://embodirun.readthedocs.io/)

## Overview

EmbodiRun connects robots and simulators to model inference services. It manages
service deployment, observations, action mapping, and bounded execution so an
application can run the observation–inference–action loop across local or remote
machines.

Use it to deploy a policy with a robot, run a simulator integration, or give an
agent access to observations and execution through a shared client API.
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) provides the first-party
inference engine; external services can connect through a provider integration.

## Features

- **Configuration-driven deployment.** Describe compute nodes, environments,
  devices, model services, and runtime bindings in YAML. Validate, prepare,
  start, inspect, and stop services from the Host CLI.
- **Separate control and compute.** Run Control beside the device and inference
  on a GPU host, using HTTP or the optional WirelessComm transport.
- **Shared device access.** Reuse service-owned observations, recordings, and
  execution arbitration across application clients.
- **Agent integration.** Observe, request a policy proposal, submit bounded
  actions, and inspect or cancel jobs through the public client.
- **Explicit robot bindings.** Keep hardware adapters separate from the
  conversion of model outputs into robot commands.

See [Architecture](docs/architecture.md) for the process boundaries and
[Agent execution workflow](docs/agent-workflow.md) for execution semantics.

## Demos

| Shared inference service | Inference engine comparison |
|---|---|
| [![Three SO-101 recordings](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg)](https://embodirun.readthedocs.io/en/latest/demos/multi-robot-serving/) | [![Engine comparison on SO-101](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/engine_e2e_contrast.jpg)](https://embodirun.readthedocs.io/en/latest/demos/engine-e2e-contrast/) |
| Three SO-101 arms using a shared π0.5 inference service, with one rollout process per device. | π0.5 on a Jetson AGX Thor with an SO-101 arm: EmbodiInfer over HTTP and WirelessComm, SGLang, and native LeRobot. |

Each demo page walks through the setup and measurements.
The engine comparison reports inference latency and full chunk time separately.

## Getting started

Install from source with Python 3.10+ and [uv](https://docs.astral.sh/uv/) 0.12.x:

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
```

Try the local simulated-device walkthrough:

```bash
uv run python examples/run_shared_device_fake.py
```

It starts a local Control service with simulated joints and a fake camera,
exercises observation, execution, recording, and cancellation, and shuts the
service down. No robot, model checkpoint, or GPU is required.

Continue with [Quick start](docs/quickstart.md) to use the deployment CLI.
For hardware, choose a combination from the [support matrix](docs/support-matrix.md),
configure its devices and calibration, and read [Safety](docs/safety.md)
before execution.

## Integrations

| Use case | Starting point |
|---|---|
| π0.5 with SO-101 arms | [Deployment guide](docs/pi05-bi-so101.md) |
| Simulator deployments | [Example configurations](configs/simulation/) and [support matrix](docs/support-matrix.md) |
| An external inference service | [Configuration](docs/configuration.md) and [inference contract](docs/http_api.md) |
| An agent with its own planning loop | [Public client](agents/CLIENT.md) and [execution workflow](docs/agent-workflow.md) |
| RPent | [Integration guide](docs/rpent-integration.md) |
| Optional hardware and model packages | [Integrations](integrations/README.md) |

Support is tracked for complete model/device/backend combinations. Software
tests, offline model checks, and real-robot demonstrations establish different
levels of verification; the support matrix records these distinctions.
Automatic compute placement and cross-model GPU scheduling are not implemented.

## Performance

Deployment performance includes observation capture, inference, transport,
and action playback. Start with the [engine comparison demo](docs/demos/engine-e2e-contrast.md)
for measured chunk times, or the [inference transport experiment](docs/inference-transport.md)
for the HTTP/WirelessComm comparison and its timing breakdown.

For model-only benchmarks, use
[EmbodiInfer's benchmark guide](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/benchmark.md).
Model inference speedups do not directly measure robot task speedups.

## Documentation

| Task | Guide |
|---|---|
| Install and run | [Installation](docs/installation.md) · [Quick start](docs/quickstart.md) |
| Configure a deployment | [Configuration](docs/configuration.md) |
| Operate devices | [Control](docs/control.md) · [Safety](docs/safety.md) |
| Extend the runtime | [Architecture](docs/architecture.md) · [Python API](docs/api.md) |
| Review evidence | [Support matrix](docs/support-matrix.md) · [Experiments](docs/experiments.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks.
Use [GitHub issues](https://github.com/BUAA-CI-LAB/EmbodiRun/issues) for bugs and
feature requests, and follow [SECURITY.md](SECURITY.md) for private vulnerability
reports. Community participation follows our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights, datasets, robot
SDKs, and simulators retain their own licenses.
