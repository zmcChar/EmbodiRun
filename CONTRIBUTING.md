# Contributing to EmbodiRun

Thanks for your interest in EmbodiRun. This document covers the development
setup, what a pull request must satisfy, and how to report problems. It is kept
in English so that there is a single authoritative version.

## Code of Conduct

This project follows the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating,
you are expected to uphold it. Report unacceptable behavior to
**cclonelycc@outlook.com**; reports are handled privately. For security
problems, follow [`SECURITY.md`](SECURITY.md) instead of opening a public
issue.

## License of contributions

EmbodiRun is licensed under Apache-2.0. See [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). By
submitting a pull request you agree that your contribution is provided under
the same license, as described in section 5 of the license. Do not contribute
code you cannot license this way, and do not paste third-party code without its
license and attribution.

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
3. Keep robot-specific dependencies optional and declared in `pyproject.toml`.
4. Do not commit checkpoints, datasets, recordings, or credentials.

## Adding a combination

If you add a robot, simulator, model, or backend combination, update
[`docs/support-matrix.md`](docs/support-matrix.md) with the versions,
configuration, hardware, checkpoint, exact command, and observed result. Do not
mark a combination **Tested** based only on code presence.

## Reporting issues

Use [GitHub Issues](https://github.com/BUAA-CI-LAB/EmbodiRun/issues) for bugs,
documentation gaps, and support questions. Include the EmbodiRun revision, the
deployment YAML shape (redact addresses and secrets), the exact command, the
observed result, and whether hardware was involved. Never paste tokens,
credentials, or personal data into an issue.

**Do not open a public issue for a security problem.** Follow
[`SECURITY.md`](SECURITY.md) instead.

## Documentation

User documentation lives in `docs/` and is published to ReadTheDocs from
`mkdocs.yml`. Keep the English `README.md` and Chinese `README.zh-CN.md` in
sync when you change positioning, install steps, or the support matrix.

### Simplified Chinese documentation

`docs/zh/` holds the Chinese pages and `mkdocs.zh.yml` builds them. Read the
Docs serves the result as the `embodirun-zh` project, a **translation** of
`embodirun`, from this same repository and branch: the language prefix and the
flyout menu come from that relationship, not from the build. The `embodirun-zh`
project selects `mkdocs.zh.yml` through the *Build configuration file* field
under **Admin → Settings** (paths in that file are relative to the repository
root, not to the file).

Conventions:

- **English is canonical.** If a Chinese page and an English page disagree, the
  English one is correct and the Chinese one is a bug.
- **Add the navigation entry last.** `nav` in `mkdocs.zh.yml` lists only pages
  that exist; an entry pointing at an untranslated page fails the strict build
  on purpose, so a short navigation is better than a broken link.
- **Translate deliberately.** These pages carry scoped claims — what a result
  does and does not establish — and a translation that smooths those into
  confident statements is worse than no translation.
- **Leave untranslated links absolute.** Point at
  `https://embodirun.readthedocs.io/en/latest/<page>/` rather than at a relative
  path, so the link resolves without an untranslated page.
- `docs/zh/stylesheets` is a symlink to `docs/stylesheets`. `extra_css` cannot
  point outside `docs_dir`, and MkDocs silently emits a `<link>` without copying
  the file when it does, so both sites share the stylesheet this way. Keep
  symlinks enabled in your checkout.

The governance documents — this file, [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md),
and [`SECURITY.md`](SECURITY.md) — are maintained in English only so that there
is one authoritative text. The Chinese README links to them.
