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

The `wireless` extra resolves `wireless-comm` from the sibling
`../WirelessComm` checkout used by this workspace.

`robot-so101` requires Python 3.12 or newer. The repository does not set a global
Python version because the other environments continue to support Python 3.10.
Groups may be combined when one process genuinely needs multiple capabilities;
they are not mutually exclusive. Model inference dependencies remain owned and
locked by the Inference project, even when inference and robot services run on
the same physical node.

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
