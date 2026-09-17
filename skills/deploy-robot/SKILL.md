---
name: deploy-robot
description: Use the RLinf Deploy Host JSON commands to inspect a configured robot service, read shared observations and media, submit one bounded action, and inspect or stop that same request.
---

# Deploy robot through Host

Use the Host CLI as the single client boundary for a configured Control
service. It resolves the endpoint from the deployment YAML and initialized
Host state, then uses the SSH plus loopback channel. Do not open a serial or
camera device, call an adapter, or create an ad-hoc HTTP/SSH loop in a skill
workflow.

The Control and recorder commands emit one JSON value on stdout. Progress and errors
from the legacy deployment commands remain human readable. Pass a stable
`--caller-id` and `--session-id` on every invocation; choose a new
`--request-id` for a new action and reuse that exact ID for inspection.

For a complete local software walkthrough, run the public helper first.  It
loads `examples/shared-device-fake.yaml`, starts the real
`ControlHttpServer` on loopback, creates only a temporary Host state fixture,
and drives the normal CLI through describe/observe/media/recording/execute/
inspect/cancel.  It refuses model or physical-device configuration and never
opens a serial, CAN, USB, or robot network resource:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 uv --no-config run --no-project \
  --offline --python 3.12 --with 'uv>=0.12,<0.13' \
  --with 'PyYAML>=6,<7' --with 'rich>=13,<15' --with 'paramiko>=3.4,<5' \
  python examples/run_shared_device_fake.py
```

For a persistent deployment, run `examples/shared-device-fake.yaml` through
the normal `validate`, `init`, `sync --source .`, and `up` lifecycle first.
The source overlay makes the fake adapters and Host API available to the
local service.  The fake robot and camera are in-memory implementations and
report `simulated: true`; this workflow is not evidence about physical
hardware.

```sh
CONFIG=examples/shared-device-fake.yaml
RUNTIME=fake-device
CALLER=agent-demo
SESSION=session-demo

rlinf-deploy --config "$CONFIG" describe \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" --json
rlinf-deploy --config "$CONFIG" observe \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" --json
# Replace this with the observation_id returned by observe when a media store
# is configured; media data is opt-in and remains service-bounded.
rlinf-deploy --config "$CONFIG" media \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" \
  --observation-id observation-id-from-json --include-data --json

printf '%s\n' '{"timestamp_s":0,"values":{"type":"joint_position","joint_positions_deg":[0,0,0,0,0],"gripper_position":25},"metadata":{"action_space":"simulated.so101.position.v1"}}' \
  | rlinf-deploy --config "$CONFIG" execute \
      --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" \
      --request-id action-demo-1 --action - --steps 1 --json
rlinf-deploy --config "$CONFIG" inspect \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" \
  --request-id action-demo-1 --json
rlinf-deploy --config "$CONFIG" observe \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" --json
```

Recorder state is queryable without changing ownership. If recording is
configured, use `recording-start`, `recording-stop`, and `recording-get`; an
unconfigured fake service returns an explicit `unsupported` result rather than
creating a local recorder:

```sh
rlinf-deploy --config "$CONFIG" recording-status \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" --json
rlinf-deploy --config "$CONFIG" recording-get \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" \
  --observation-id observation-id-from-json --json
# Only when recording is configured and explicitly requested:
rlinf-deploy --config "$CONFIG" recording-start \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" --json
rlinf-deploy --config "$CONFIG" recording-stop \
  --runtime "$RUNTIME" --caller-id "$CALLER" --session-id "$SESSION" \
  --recording-timeout 2 --json
```

`observe` returns an `observation_id` and media references when the service
has a shared observation producer. Use the Host client's `media` operation or
the corresponding API route to read a referenced frame; a stale or missing
reference is an explicit service result. The CLI does not copy frames through
an SDK or silently substitute a newer observation.

An accepted response and a completed response are different states. If the
execute request times out, treat the result as `unknown`, keep the original
request ID, and run `inspect` before deciding what to do. Do not resend the
action after a timeout. `cancel` is scoped to the caller/session/request ID;
`stop` requests a stop for that one job. A reader that only observes must not
cancel another caller's job, and a disconnected client must not claim that a
stop was confirmed.

The service is authoritative for action limits, stale-input checks,
unsupported capabilities, busy ownership, stop confirmation, and recording
state. Preserve the returned JSON fields (`status`, `physical_status`,
`error`, and any media or job details) when reporting those outcomes. Do not
take over a serial device, kill the current owner, repeat `stop` or `stop_all`, or invent a
success result when the service reports `unknown`, `uncertain`,
`stop_unconfirmed`, `stale`, `busy`, or `unsupported`.
