# RLinf Deploy

RLinf Deploy owns the last mile between an inference service and a robot or
simulator. It does not load checkpoints, import model frameworks, build prompts,
or execute neural networks.

```text
camera / robot state
        |
        v
RLinf Deploy -- HTTP / WirelessComm --> RLinf Inference (VVLA)
        |
        v
robot-specific safety and motion execution
```

## Repository boundary

| RLinf Deploy owns | RLinf Inference owns |
|---|---|
| Robot and simulator observations | Checkpoints and processors |
| Policy clients, retries, deadlines | Prompt and model forward |
| Session and step identifiers | Session memory and batching |
| Robot action validation | Model-output parsing |
| Go2 and FR3 control | DP, TP, CUDA Graph, compilation |
| SSH deployment and service routing | Canonical policy actions |

`third_party/vvla` pins the server implementation for integration and deployment
reproducibility. Runtime communication crosses the versioned policy API over HTTP
or the optional WirelessComm data plane; the deploy package never imports VVLA
Python modules.

## Environments

Use uv 0.12.x from the repository root. Each process installs only the capability
group it needs:

```bash
uv sync --frozen                                      # core + development tools
uv sync --frozen --extra wireless                     # WirelessComm policy client
uv sync --frozen --no-dev --group host                # SSH orchestration
uv sync --python 3.12 --frozen --no-dev \
  --group robot-so101                                 # SO-101 control agent
uv sync --frozen --no-dev --group robot-fr3           # FR3 control agent
```

The `wireless` extra pins WirelessComm to a reviewed commit in the
BUAA-CI-LAB repository. Private Git access must be configured for pip or uv.

`robot-so101` requires Python 3.12 or newer. The repository does not set a global
Python version because the other environments continue to support Python 3.10.
Groups may be combined when one process genuinely needs multiple capabilities;
they are not mutually exclusive. Model inference dependencies remain owned and
locked by the Inference project, even when inference and robot services run on
the same physical node.

## Multi-node configuration

The host CLI reads one deployment YAML and resolves it into nodes, uv
environments, and service/runtime instances. Validate a file without contacting
its nodes:

```bash
uv run rlinf-deploy \
  --config examples/muti-nodes.example.yaml validate

uv run rlinf-deploy \
  --config examples/muti-nodes.example.yaml probe
```

`probe` checks every node even if another node is unreachable. It reports the
connection address, latency, platform, architecture, and availability of
Python, Git, and uv. It does not clone repositories, install dependencies,
start services, or write deployment state.

Initialize the nodes, then start their persistent services:

```bash
# First replace the example commit, checkpoint, calibration, and camera values.
export JETSON_AGX_THOR_232_SSH_PASSWORD='<ssh-password>'
uv run rlinf-deploy \
  --config examples/muti-nodes.example.yaml init

uv run rlinf-deploy \
  --config examples/muti-nodes.example.yaml up
```

`init` probes Python, Git, and uv; checks configured robot, calibration, model,
and adapter paths; checks out the exact Deploy and Inference commits below
`~/.local/share/rlinf-deploy/<deployment-name>` on each node; and runs the
appropriate `uv sync --frozen` commands. Repeating it is safe: Git checkouts and
uv environments are reused, while uv downloads only missing or changed locked
dependencies. Successful initialization is recorded locally below
`~/.local/state/rlinf-deploy`.

`up` refuses to run without state from a successful `init`, or if the YAML has
changed since initialization. It starts PID-supervised model services and is
idempotent for services that are already running. The two SO101 binding runtimes
remain independent and will be invoked later by the runtime-oriented `run`
command; `up` does not connect to or move either arm.

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

Install the optional client and give both processes complementary WirelessComm
YAML files whose peer directories contain each other. Start the inference process
with `vvla-wireless-serve`, then create the Deploy client from its local config:

```python
from rlinf_deploy import VvlaWirelessClient

client = VvlaWirelessClient.from_config(
    "configs/wireless.example.yaml",
    server_node_id="inference-1",
    token="shared-token",
    timeout_s=5.0,
)
try:
    capabilities = client.capabilities()
finally:
    client.shutdown()
```

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

See `examples/fr3_http_step.py` for a single HTTP inference step followed by
execution of the returned action chunk.

Quick end-to-end flow:

```bash
# On Thor: run VVLA HTTP service for pi0.5
cd third_party/vvla
vvla-http-serve \
  --policy pi05 \
  --checkpoint <pi05-checkpoint> \
  --adapter-config ../../configs/pi05_http_serve.example.json \
  --host 0.0.0.0 \
  --port 8000 \
  --max-batch 1

# On deploy side: send one image + state and execute one FR3 command
cd ../..
python examples/fr3_http_step.py \
  --robot-host <fr3-ip> \
  --vvla-url http://<thor-ip>:8000 \
  --instruction "pick up the object" \
  --image /path/to/image.jpg
```

## Go2

The existing Unitree Go2 camera/control agents and SSH deployer remain under:

```text
src/rlinf_deploy/robots/unitree/go2
src/rlinf_deploy/bindings/unitree/go2/streamvln
```

The robot-resident processes remain isolated from model inference and expose
bounded HTTP control surfaces.
