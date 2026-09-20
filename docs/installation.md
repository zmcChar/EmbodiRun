# Installation

EmbodiRun is installed from source with [uv](https://docs.astral.sh/uv/). The
public `uv.lock` does not require access to any private repository.

## Prerequisites

- Linux (x86_64 or ARM64, depending on the capability group).
- Python 3.10 or newer. Some groups require newer Python: `robot-so101`,
  `sim-libero`, `sim-vlabench`, and `sim-isaac` use 3.12; `sim-habitat` uses
  3.11.
- [uv](https://docs.astral.sh/uv/) 0.12.x. The repository pins
  `required-version = ">=0.12.0,<0.13"`.
- `git` and network access to GitHub.

## Source install

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
```

`uv sync --frozen` installs the core package, the `host` group, and the
development tools. It is enough for `embodirun validate`, the Host lifecycle,
and the test suite.

### Capability groups

Each process should install only the group it needs. Groups are not mutually
exclusive, but mixing unrelated robot or simulator stacks is not required.

| Group | Adds | Python |
|---|---|---|
| `host` | SSH orchestration: paramiko, PyYAML, rich | 3.10+ |
| `robot-so101` | Feetech SDK, OpenCV for SO-101 | 3.12+ |
| `robot-fr3` | `franky-control` for Franka FR3 (Linux x86_64) | 3.10+ |
| `robot-arx5` | Pillow, RealSense, OpenCV (vendor motor driver separate) | 3.10+ |
| `robot-go2` | no extra Python dependency (Unitree SDK2 is vendor-supplied) | 3.10+ |
| `robot-xlerobot-external-owner` | no extra dependency in core | 3.10+ |
| `sim-libero` | LeRobot + LIBERO, Pillow | 3.12+ |
| `sim-habitat` | habitat-sim (built from source), OpenCV, Pillow | 3.11 |
| `sim-isaac` | Isaac Sim (NVIDIA EULA), OpenCV, Pillow | 3.12 |
| `sim-vlabench` | LeRobot, VLABench, Pillow | 3.12+ |

Examples:

```bash
uv sync --frozen --no-dev --group host
uv sync --frozen --no-dev --group robot-so101
UV_PROJECT_ENVIRONMENT=.venv-robot-arx5 uv sync --frozen --no-dev --group robot-arx5
```

`robot-arx5` needs the vendor `bimanual` extension built for the Python in its
environment; set the ARX5 robot option `sdk_path` to its absolute import
directory on the control node. ARX5 also needs the vendor native/ROS libraries
and a configured SocketCAN interface.

### Installer scripts

Two helpers cover the common cases:

- `scripts/install.sh` syncs this checkout with a chosen group and extra:

  ```bash
  scripts/install.sh --group robot-so101
  scripts/install.sh --group host --extra sglang
  ```

- `requirements/install.sh` installs a standalone environment at a chosen path
  and Python version, with optional component extras and a selectable PyTorch
  version. Run `bash requirements/install.sh --help` for the full option list.
  `EMBODIRUN_ENV_ROOT` and `EMBODIRUN_PYTORCH_INDEX` override its defaults.

## Extras

| Extra | Contents |
|---|---|
| `sglang` | SGLang diffusion serving (Linux, Python 3.12+, glibc >= 2.34) |
| `wireless` | Marker for the optional WirelessComm data plane (empty) |

`wireless` installs nothing today. WirelessComm is published separately at
[`BUAA-CI-LAB/WirelessComm`](https://github.com/BUAA-CI-LAB/WirelessComm) and
is not on a package index yet, so the extra stays a marker and you install the
released tag into the node environment yourself:

```bash
uv pip install "wireless-comm @ git+https://github.com/BUAA-CI-LAB/WirelessComm@v0.1.0"
```

Then select the transport in the deployment YAML
(`models.<id>.transport: wireless`) as described in
[`http_api.md`](http_api.md). WirelessComm peer IDs and its optional token do
not encrypt or authenticate the link; use it only on a trusted network.

## Managed deployments

`embodirun init` checks out the pinned EmbodiRun and EmbodiInfer revisions below
`~/.local/share/rlinf-deploy/<deployment-name>` on each node and runs
`uv sync --frozen` there. The `rlinf-deploy` path is kept for compatibility
with existing deployments.

The `third_party/embodiinfer` submodule pins the inference server revision used
for reproducibility. Initialize it only if you want the pinned tree:

```bash
git submodule update --init third_party/embodiinfer
```

## PyPI

`embodirun` is not published on PyPI yet. Use the Git + uv installation above.
Publishing is on the roadmap; until then the examples in older material that
show `pip install embodirun` are not valid.

## Upgrading from `rlinf-deploy`

The package was renamed from `rlinf-deploy` / `rlinf_deploy` to `embodirun`.
The old import path and `rlinf-*` console commands remain available, and the
`~/.local/share/rlinf-deploy` and `~/.local/state/rlinf-deploy` paths are
preserved. Uninstall an old `rlinf-deploy` distribution before installing this
checkout; `uv sync --frozen` replaces it automatically.

## Verify the checkout

```bash
uv run pytest -q
```

The CPU suite passes without a GPU, checkpoints, sglang, or robot hardware.
Tests that need those are skipped with an explicit reason.
