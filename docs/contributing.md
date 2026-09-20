# Contributing

EmbodiRun welcomes contributions: bug fixes, new robot and simulator adapters,
policy bindings, model backends, tests, and documentation.

The authoritative guide is
[`CONTRIBUTING.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/CONTRIBUTING.md)
in the repository root. This page summarizes it; if the two disagree, the
repository file wins.

!!! note "Governance documents are English-only"
    `CONTRIBUTING.md`, [`CODE_OF_CONDUCT.md`](code-of-conduct.md), and the
    security policy are maintained in English so that there is a single
    authoritative version. The Chinese README links to them.

## Development setup

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run pytest -q
```

The core suite runs on CPU. Tests that need a GPU, a checkpoint, sglang, or
robot hardware skip themselves with an explicit reason. To run a grouped path,
install the matching group, for example
`uv sync --frozen --dev --group robot-so101`.

## Before opening a pull request

1. Run the checks and make sure they pass:

   ```bash
   uv run ruff check src tests
   uv run ruff format --check src tests
   uv run pytest -q
   ```
2. Keep the process boundaries intact:
    - `client`, `deployment`, `application`, `devices`, `model_services` are the
      canonical domains; `services.*` stays a compatibility layer.
    - `robots/` and `simulators/` hold hardware adapters; `bindings/` holds
      policy-to-robot mappings. Do not merge the two.
    - Do not import inference-engine runtime code into `src/embodirun`; talk to
      it over the versioned API.
3. Keep robot-specific dependencies optional and declared in
   `pyproject.toml`.
4. Do not commit checkpoints, datasets, recordings, or credentials.

## Adding a combination

If you add a robot, simulator, model, or backend combination, update
[Support Matrix](support-matrix.md) with the versions, configuration, hardware,
checkpoint, exact command, and observed result. Do not mark a combination
**Tested** based only on code presence.

## Reporting issues

Use
[GitHub Issues](https://github.com/BUAA-CI-LAB/EmbodiRun/issues) for bugs,
documentation gaps, and support questions. Include the EmbodiRun revision, the
deployment YAML shape with addresses and secrets redacted, the exact command,
the observed result, and whether hardware was involved.

Do not open a public issue for a security problem; see the
[security policy](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/SECURITY.md)
instead.
