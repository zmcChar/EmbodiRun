# EmbodiRun Documentation

**Embodied AI, Ready to Run.**

EmbodiRun is the deployment and execution runtime for embodied AI. It connects
model inference, service deployment, cross-node communication, and robot
execution into one reproducible system: configure a model, a compute node, and a
robot or simulator, then start the services and run a task.

High-performance inference is provided by
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer), which stays an
independent engine and can also be used on its own.

> **Deploy Models. Accelerate Inference. Run Robots.**

## Choose a path

### Run a deployment

1. [Installation](installation.md) — install from source with uv and pick a
   capability group.
2. [Quick start](quickstart.md) — a device-free example first, then a real
   deployment.
3. [Configuration](configuration.md) — the deployment YAML reference.

### Operate a robot

- [Control](control.md) — manual input, control authority, arbitration, and the
  software stop.
- [Safety](safety.md) — operator checklist and failure semantics. Read this
  before a physical deployment.
- [Support matrix](support-matrix.md) — what is tested, experimental, or
  planned, and what evidence each status requires.

### Integrate a model or an agent

- [Inference API v1](http_api.md) — the versioned policy API carried over HTTP
  or WirelessComm.
- [RPent integration](rpent-integration.md) — the Agent boundary, and a
  reproducible software chain against a real π0.5 service.
- [π0.5 with two SO-101 followers](pi05-bi-so101.md) — a dual-arm deployment
  guide.
- [`agents/CLIENT.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)
  — the public Agent client.

### Understand the runtime

- [Architecture](architecture.md) — runtime domains, process boundaries, and
  non-goals.

### Read the engineering notes

- [Experiments](experiments.md) — opt-in hardware experiments. These are
  evidence records, not user guides.

## Documentation map

| Section | Pages | What it covers |
|---|---|---|
| Getting started | Installation, Quick start, Configuration | Install, run a device-free example, write a deployment YAML |
| Operating | Control, Safety, Support matrix | Running and stopping a deployment, what is verified |
| Integrating | Inference API v1, RPent integration, π0.5 with two SO-101 | The policy API and reference integrations |
| Concepts | Architecture | Domains, ownership, and boundaries |
| Experiments | Experiments overview and notes | Opt-in hardware measurements, with their limits |
| Project | Contributing, Code of Conduct, License and relicensing | How to contribute, community rules, licensing |

## Status

The support matrix distinguishes source adapters, CPU-verified software,
real-model runs, and real-robot cases; a combination is never marked **Tested**
from code presence alone. Claims on the experiment pages are scoped to the
configuration that was measured, and an unverified path is labelled as such.

## Community

- [Contributing](contributing.md) — development setup, pull request
  expectations, and the support-matrix rule.
- [Code of Conduct](code-of-conduct.md) — the Contributor Covenant 2.1 adopted
  by this project.
- [Security policy](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/SECURITY.md)
  — report vulnerabilities privately, never in a public issue.
- [License and relicensing](license.md) — Apache-2.0 and the move from MIT.

The repository README is available in
[English](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.md) and
[简体中文](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.zh-CN.md).
