# EmbodiRun

EmbodiRun owns the last mile between an inference service and a robot or
simulator. It does not load checkpoints, import model frameworks, build prompts,
or execute neural networks.

```text
                         management plane
Host ---------------- SSH ----------------> Compute / Control nodes
  |
  +-------- HTTP through SSH -------------> Control service
                                                |
                         inference plane        +--> sensors / robot adapter
                     HTTP or WirelessComm       |
Control service ------------------------------> Inference service
```

## Repository boundary

| EmbodiRun owns | EmbodiInfer owns |
|---|---|
| Robot and simulator observations | Checkpoints and processors |
| Policy clients, retries, deadlines | Prompt and model forward |
| Session and step identifiers | Session memory and batching |
| Robot action validation | Model-output parsing |
| Go2 and FR3 control | DP, TP, CUDA Graph, compilation |
| SSH deployment and service routing | Canonical policy actions |

`third_party/embodiinfer` pins the server implementation for integration and deployment
reproducibility. Runtime communication crosses the versioned policy API over HTTP
or the optional WirelessComm data plane; the deploy package never imports EmbodiInfer
Python modules.

The service directories follow those process boundaries:

```text
services/
├── host/                       # runs only on the operator machine
│   ├── cli/                    # user commands and progress reporting
│   ├── config/                 # typed YAML sections and reference checks
│   ├── executor/               # local/SSH command and tunneled HTTP execution
│   ├── control.py              # Host -> Control task client
│   ├── plan.py                 # resolves config into services and runtimes
│   ├── state.py                # local initialized/running state
│   ├── probe.py                # node capability discovery
│   ├── source.py               # pinned checkouts and development overlays
│   ├── environment.py          # uv environment resolution
│   └── supervisor.py           # identity-checked process lifecycle
├── control/                    # runs beside the robot and sensors
│   ├── contracts.py            # versioned Host <-> Control task messages
│   ├── runtime.py              # observation/inference/action loop
│   └── server.py               # loopback task API and hardware ownership
└── inference/                  # Control <-> Inference boundary
    ├── contracts.py            # versioned policy sessions, steps, and actions
    ├── client/                 # HTTP and WirelessComm clients
    └── server/                 # external EmbodiInfer launch descriptors
```

`bindings/` contains only policy-to-robot mappings and safety horizons; it does
not own a process, network listener, or deployment protocol.

## Environments

Install this checkout as `embodirun`, import `embodirun`, and use the `embodirun` CLI.
Existing `rlinf-*` commands, `rlinf_deploy` imports, configuration keys, and
state/data paths remain supported with no scheduled removal. For pip upgrades,
uninstall `rlinf-deploy` before installing this checkout; `uv sync --frozen`
replaces the old distribution automatically. Keep your existing groups/extras.

Use uv 0.12.x from the repository root. Each process installs only the capability
group it needs:

```bash
uv sync --frozen                                      # core + development tools
uv sync --frozen --extra wireless                     # WirelessComm policy client
uv sync --frozen --no-dev --group host                # SSH orchestration
uv sync --python 3.12 --frozen --no-dev \
  --group robot-so101                                 # SO-101 control agent
uv sync --frozen --no-dev --group robot-fr3           # FR3 control agent
UV_PROJECT_ENVIRONMENT=.venv-robot-arx5 uv sync --frozen --no-dev --group robot-arx5
```

The `wireless` extra pins WirelessComm to a reviewed commit in the
BUAA-CI-LAB repository. Private Git access must be configured for pip or uv.

The optional platform profiles remain separate: the locked `franky-control`
wheel supports Linux x86_64, not ARM64; the locked SGLang and Isaac Sim wheels
require Linux glibc >= 2.34 and >= 2.35 respectively. Habitat uses Python 3.11,
while SO-101, LIBERO, VLABench, and Isaac profiles use Python 3.12. A successful
lock/install-plan check does not verify CUDA, robot SDKs, devices, or checkpoints.

`robot-so101` requires Python 3.12 or newer. The repository does not set a global
Python version because the other environments continue to support Python 3.10.
Groups may be combined when one process genuinely needs multiple capabilities;
they are not mutually exclusive. Model inference dependencies remain owned and
locked by the Inference project, even when inference and robot services run on
the same physical node.

ARX5 uses `type: arx.x5` and binding `arx.x5.dm05`. Its group installs the
RealSense and V4L2 camera dependencies, not the vendor motor driver. Build the
vendor `bimanual` extension for the Python in `.venv-robot-arx5`; set the ARX5
robot option `sdk_path` to its absolute import directory on the control node
(or install the extension in that environment). Its native/ROS libraries and
configured SocketCAN interface must already be available to the control process.
Do not rely on a different shell's Python environment. Physical connection still
requires `operator_confirmed: true` and may enable motors or move to home.

For DM05, use a matching EmbodiInfer revision containing the DM05 policy.
Configure the inference environment's OpenDM installation separately; its
model YAML entry must set `adapter_config: /absolute/path/to/dm05.json` on the
compute node. That JSON is deployment-owned because the norm-stat path depends
on the selected checkpoint:

```json
{"policy_kwargs": {"norm_stats": "/models/dm05/norm_stats.json", "is_history": true, "robot_type": "ARX5", "output_action_dim": 7}}
```

Host synchronizes the core model environment before installing its
`environment_packages` overlay; provision the compatible OpenDM package there,
not only in a pre-existing environment that the sync can clean.
Camera roles are `cam_global`
plus `cam_arm` or `cam_side` (the missing wrist role is explicitly replicated).
The binding selects 25 rows from a 50-row prediction; choose `chunk_steps: 25`
to execute that entire selection rather than the generic 10-row prefix.

## Multi-node configuration

For the 12-dimensional dual SO-101 Pi0.5 checkpoint and three cameras, use
[the EmbodiInfer deployment guide](docs/pi05-bi-so101.md) and
[portable configuration](configs/pi05/bi-so101-vvla.yaml). It includes the
paired inference revision, mixed-precision/CUDA Graph settings, and Jetson
environment prerequisites.

The host CLI reads one deployment YAML and resolves it into nodes, uv
environments, and service/runtime instances. Validate a file without contacting
its nodes:

```bash
uv run embodirun \
  --config configs/http-wireless-inference/http.yaml validate

uv run embodirun \
  --config configs/http-wireless-inference/http.yaml probe
```

The HTTP and WirelessComm lab examples both run PI0.5 on Thor
(`192.168.2.232`), with `so101-1` and its cameras on AGX Orin
(`192.168.2.174`) and `so101-2` and its cameras on Orin NX
(`192.168.2.148`). Both arms share the same model service. Run one protocol
configuration at a time. The
[HTTP / WirelessComm inference benchmark](benchmarks/http-wireless-inference-benchmark/README.md)
includes a read-only recorder for real camera frames and joint state, then
compares both protocols from the two control nodes by replaying fixed
observations. It does not execute robot actions.

The lab examples use `connection.proxy_command` for the Host's SOCKS5 proxy at
`127.0.0.1:1080`; remove that option when Host can reach the LAN directly.
`%h` and `%p` expand to the configured SSH host and port. This only routes SSH;
the nodes exchange inference data directly on their configured LAN addresses.

`probe` checks every node even if another node is unreachable. It reports the
connection address, latency, platform, architecture, and availability of
Python, Git, and uv. It does not clone repositories, install dependencies,
start services, or write deployment state. Nodes are probed concurrently and
the terminal shows each node's current stage and elapsed time.

Initialize the nodes, then start their persistent services:

```bash
# The checked-in configuration uses key-based SSH to Thor and both Orin nodes,
# with the observed serial, calibration, camera, and checkpoint paths.
uv run embodirun \
  --config configs/http-wireless-inference/http.yaml init

# During Deploy-side development, reload Control without restarting pi0.5.
uv run embodirun \
  --config configs/http-wireless-inference/http.yaml down --target control
uv run embodirun \
  --config configs/http-wireless-inference/http.yaml sync --target deploy
uv run embodirun \
  --config configs/http-wireless-inference/http.yaml up

uv run embodirun \
  --config configs/http-wireless-inference/http.yaml down
```

`init` probes Python, Git, and uv; checks configured robot, calibration, model,
and adapter paths; checks out the exact Deploy and Inference commits below
`~/.local/share/rlinf-deploy/<deployment-name>` on each node; and runs the
appropriate `uv sync --frozen` commands. Repeating it is safe: Git checkouts and
uv environments are reused, while uv downloads only missing or changed locked
dependencies. Successful initialization is recorded locally below
`~/.local/state/rlinf-deploy`. Different nodes initialize concurrently; work on
one node remains ordered, and each Deploy or Inference environment appears as a
separate synchronization stage in that node's progress row.

`sync --target deploy` packages the current local `src/embodirun` tree and
uploads it once per Deploy node into a content-addressed development overlay.
It atomically activates that overlay without restarting Inference. A running
Control process has already imported its Python modules, so `sync` requires
Control services to be stopped first. `down --target control` stops only those
services; the following `up` reuses the already-running, healthy model process
and starts Control from the new overlay. If `pyproject.toml` and `uv.lock` are
unchanged, the existing robot environment is reused; otherwise only the affected
Deploy environment groups are synchronized. Use `--source PATH` when invoking
the command outside the repository root. `sync` may still run after unrelated
YAML changes, but it does not apply them to persistent services. Changes to model
service configuration or either pinned revision still require the normal
`down` → `init` → `up` lifecycle.

`up` refuses to run without state from a successful `init`, or if the YAML has
changed since initialization. It starts PID-supervised inference and control
services and waits for their health checks. It is idempotent for processes that
are already running. Starting Control loads static robot, sensor, binding, and
inference routing configuration, but does not connect to or move the arm.
`down` stops identity-checked processes and remains available when the YAML has
changed; `--target control` or `--target model` limits it to one service kind.
Both commands operate on different nodes concurrently.

After `up` reports the services healthy, submit one prompt to a configured
control runtime:

```bash
embodirun \
  --config configs/http-wireless-inference/http.yaml run \
  --runtime so101-1-runtime \
  --prompt "Pick up the cube and put it into the bowl." \
  --chunk-steps 10
```

The Host sends this task as versioned HTTP through the existing authenticated SSH
connection to the Control service's loopback socket; it does not expose a task
port to the Wi-Fi network. The request contains only the prompt and bounded
execution parameters. Static robot, sensor, binding, and inference configuration
stays on Control.

`run` can move the selected physical robot. The default is one inference/action
chunk. `--chunk-steps N` selects how many ordered actions to execute from each
chunk (default 10), `--max-steps N` bounds the number of inference chunks, and
`--control-hz HZ` selects the action playback rate. The SO101/Pi0.5 binding
declares its five arm joints, gripper, image fields, and maximum model horizon;
`up` materializes that declaration as an internal EmbodiInfer adapter file. No
user-maintained model adapter JSON is required. Before connecting the arm,
Control checks model health and opens, warms up, and validates every camera.
V4L2 capture is implemented in the robot sensor camera layer and returns
model-independent camera frames; the SO101/Pi0.5 binding converts those frames
into inference image payloads.
Each returned row is still subject to the SO101 joint and gripper step limits
from the deployment YAML. `step_limit_mode: reject` rejects an oversized target
without sending it. `step_limit_mode: clip` bounds every joint and the gripper
independently before sending that row; the checked-in SO101 configurations use 5
degrees and 10 gripper units for its initial tests. Clipping is a per-row rate
limit, not collision avoidance. A Pi0.5 response has no task-complete signal, so
the chunk bound is always the stopping condition. A requested chunk length above
the binding maximum is rejected before connecting to the robot; a short model
response is rejected before any action from that response is executed.

## Manual control and software emergency stop

The Control service owns the robot connection across model tasks. Task completion
holds the robot through its adapter; it does not disconnect it or automatically
return it to a home pose. Service shutdown releases the hardware.

Manual control takes priority over model actions. Software emergency stop latches
outside the motion queue, cancels active work, and clears pending actions. Reset
does not resume an interrupted model task. Keyboard input works without a
joystick; robot-specific joystick mappings remain in the robot packages.
See [control usage and safety limits](docs/control.md).

## HTTP contract

The client expects these endpoints:

```text
GET    /healthz
GET    /v1/capabilities
POST   /v1/sessions
POST   /v1/sessions/{session_id}/steps
POST   /v1/sessions/{session_id}/reset
DELETE /v1/sessions/{session_id}
```

Step requests use `multipart/form-data`: one JSON metadata part followed by
binary image parts. Images are never base64 encoded. `request_id`, `step_id`,
and `session_revision` provide idempotency and ordering.

## WirelessComm contract

For Host-managed deployment, use
[`configs/http-wireless-inference/wireless.yaml`](configs/http-wireless-inference/wireless.yaml).
It has the same `nodes`, `models`, and `runtimes` structure as HTTP deployments:

- `models.<id>.server` is the actual inference listener for either transport.
- `runtimes.<id>.server` remains the loopback HTTP interface used by Host.
- Wireless runtimes additionally declare `inference_client.bind` and `port`,
  since each client process also listens for WirelessComm traffic. HTTP runtimes
  do not declare this block.
- Optional `transport_options` under a model or `inference_client` are passed to
  that endpoint's native WirelessComm `comm` settings. These are independent
  per-endpoint settings, validated by WirelessComm when the endpoint starts.

For cross-node wildcard listeners, the advertised data address defaults to
`nodes.<id>.connection.host`. Set `nodes.<id>.address` when the SSH host is a
management address or tunnel, or when a `local` node needs to serve another node.
Explicit listener addresses must themselves be reachable by their peers;
loopback listeners cannot serve other nodes. Each listener on the same node
must use a distinct port, including Host-facing HTTP and WirelessComm listeners.

`validate` checks the topology without contacting nodes. `init` prepares the
environments, including the wireless extra. During `up`, Host generates native
WirelessComm configs under each node's deployment `generated/` directory and
passes their absolute paths to EmbodiInfer and the robot/simulator runtime. Peer IDs
identify model/runtime instances: a model lists all its runtimes, while a runtime
lists only its selected model. Do not specify `comm_config`, `client_comm_config`,
or `server_node_id` in the deployment YAML anymore. Changes to a running topology
require `down` → `init` → `up`; generated files are not hot-reloaded.

Wireless steps carry the same versioned session/step fields as HTTP, while image
bytes remain separate WirelessComm payload segments instead of being assembled as
multipart data. One client instance may own multiple policy sessions and should
live for the process lifetime so its connection, response dispatcher, queues and
pacing state are reused. A timeout does not cancel inference; retry an uncertain
step only against the same server/session with the same `request_id`.

WirelessComm peer IDs and the optional token do not encrypt or authenticate the
link. Use this transport only on a trusted isolated network until the data plane
provides TLS or an equivalent authenticated transport.

## FR3

Install the official Franky binding that matches the robot server version:

```bash
uv sync --frozen --no-dev --group robot-fr3
```

The adapter follows the official Franky API:

- `Robot(host)` and `recover_from_errors()`
- bounded `JointMotion`
- bounded relative `CartesianMotion`
- optional `Gripper.move`
- state from `robot.state`

Physical execution is intentionally fail-closed. Joint and Cartesian step
limits are checked before Franky receives a command.

FR3 and SO-101 share `services/control/runtime.py` and the versioned task service.
Their `pi05` binding packages contain only the policy-specific mapper and binding
definition.

## Go2

The existing Unitree Go2 camera/control agents and SSH deployer remain under:

```text
src/embodirun/robots/unitree/go2
src/embodirun/bindings/unitree/go2/streamvln
```

The robot-resident processes remain isolated from model inference and expose
bounded HTTP control surfaces.
