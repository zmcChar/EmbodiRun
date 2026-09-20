# EmbodiRun Documentation

**Embodied AI, Ready to Run.**

**An Efficient Deployment & Execution Runtime for Embodied AI.**

EmbodiRun connects model inference, service deployment, cross-node
communication, and robot execution into one reproducible runtime. High
performance inference is provided by
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer), which can also be
used independently.

> **Deploy Models. Accelerate Inference. Run Robots.**

Start here:

- [Installation](installation.md) — install from source with uv and pick a
  capability group.
- [Quick Start](quickstart.md) — a no-hardware example, then a real deployment.
- [Configuration](configuration.md) — the deployment YAML reference.
- [Architecture](architecture.md) — runtime domains and process boundaries.
- [Support Matrix](support-matrix.md) — what is tested, experimental, or
  planned.

Operating a robot:

- [Control](control.md) — manual takeover, arbitration, and the software stop.
- [Safety](safety.md) — operator checklist and failure semantics.

Integrating:

- [Inference API v1](http_api.md) — the versioned policy API.
- [RPent integration](rpent-integration.md) — the Agent boundary and a
  reproducible software chain against a real π0.5 service.
- [π0.5 with two SO-101 followers](pi05-bi-so101.md) — a dual-arm deployment
  guide.
- [`agents/CLIENT.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)
  — the public Agent client.

The repository README is available in
[English](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.md) and
[简体中文](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.zh-CN.md).

## Community

- [Contributing](contributing.md) — development setup, pull request
  expectations, and the support-matrix rule.
- [Code of Conduct](code-of-conduct.md) — the Contributor Covenant 2.1 adopted
  by this project.
- [Security policy](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/SECURITY.md)
  — report vulnerabilities privately, never in a public issue.
- [License and relicensing](license.md) — Apache-2.0 and the move from MIT.
