# RLinf Deploy

## Repository purpose

RLinf Deploy contains robot and simulator deployment clients for RLinf
inference services.

It owns robot and simulator observations, HTTP inference clients, session and
step coordination, action validation, robot-specific motion execution, and
deployment routing. It does not own model checkpoints, model inference, prompt
construction, model frameworks, or model-output parsing.

Do not import runtime code from `third_party/vvla` into `src/rlinf_deploy`.
Communication with VVLA must go through the versioned HTTP or WirelessComm API.

## Project structure

- `src/rlinf_deploy/services/host`: host CLI, configuration, state, and SSH lifecycle
- `src/rlinf_deploy/services/control`: Host task contracts and control-node runtime
- `src/rlinf_deploy/services/inference`: inference contracts, protocols, backend clients, and launch descriptors
- `src/rlinf_deploy/robots`: robot interfaces and adapters
- `src/rlinf_deploy/bindings`: policy-to-robot bindings
- `src/rlinf_deploy/simulators`: simulator adapters
- `tests`: unit, boundary, and integration tests
- `third_party/vvla`: pinned upstream server implementation
- `configs`: example deployment configurations

## Development commands

Install the package for development:

    uv sync --frozen

Run the full test suite:

    uv run pytest

Run a specific test file:

    uv run pytest tests/<test_file>.py

## Repository conventions

- Support Python 3.10 and newer.
- Keep robot-specific dependencies optional.
- Declare dependencies in `pyproject.toml`.
- Preserve the separation between generic robot adapters and policy-specific
  bindings.
- Do not modify `third_party/vvla` unless the task explicitly targets the pinned
  upstream code.
- Do not perform operations against physical robots unless the user explicitly
  requests them.
