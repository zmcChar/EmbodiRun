# Contributing to EmbodiRun

Thanks for your interest. EmbodiRun is licensed under Apache-2.0; by
contributing you agree that your contribution is provided under the same
license (see section 5 of [`LICENSE`](LICENSE)).

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

1. Run `uv run pytest -q` and make sure it passes.
2. Keep the process boundaries intact:
   - `client`, `deployment`, `application`, `devices`, `model_services` are the
     canonical domains; `services.*` stays a compatibility layer.
   - `robots/` and `simulators/` hold hardware adapters; `bindings/` holds
     policy-to-robot mappings. Do not merge the two.
   - Do not import inference-engine runtime code into `src/embodirun`; talk to
     it over the versioned API.
3. Keep robot-specific dependencies optional and declared in `pyproject.toml`.
4. Do not commit checkpoints, datasets, recordings, or credentials.

## Adding a combination

If you add a robot, simulator, model, or backend combination, update
[`docs/support-matrix.md`](docs/support-matrix.md) with the versions,
configuration, hardware, checkpoint, exact command, and observed result. Do not
mark a combination **Tested** based only on code presence.

## Reporting issues

Include the EmbodiRun revision, the deployment YAML shape (redact addresses and
secrets), the exact command, the observed result, and whether hardware was
involved. Never paste tokens, credentials, or personal data.

## Documentation

User documentation lives in `docs/` and is published to ReadTheDocs from
`mkdocs.yml`. Keep the English `README.md` and Chinese `README.zh-CN.md` in
sync when you change positioning, install steps, or the support matrix.
