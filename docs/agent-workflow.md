# Agent execution workflow

An agent drives a running deployment through the Host JSON commands
(`describe`, `observe`, `media`, `execute`, `inspect`, `cancel`, `stop`, and the
recording commands). This page describes the contract those commands implement.
The wire schemas they use are in [Inference API v1](http_api.md); the
service-side hardware rules are in [Control](control.md) and
[Safety](safety.md).

## Use the Host CLI as the client boundary

The Host CLI resolves the endpoint from the deployment YAML and the initialized
Host state, then reaches the Control service over the configured SSH and
loopback channel. Use it instead of opening a serial or camera device, calling
an adapter, or building an ad-hoc HTTP or SSH loop: those paths bypass
observation sharing, execution arbitration and the identity checks the service
performs.

Control and recorder commands write one JSON value to stdout. Keep `--json` on
and parse it. A self-contained software walkthrough that starts the real
`ControlHttpServer` against simulated adapters is in
[`examples/README.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md).
That walkthrough refuses model or physical-device configuration and opens no
serial, CAN, USB or robot network resource, so its results say nothing about
hardware.

## Carry a stable identity

Pass a stable `--caller-id` and `--session-id` on every invocation. Ownership,
observation scope and cancellation all resolve against them, so changing either
one mid-workflow detaches the workflow from its own state.

Choose a new `--request-id` for a new action and reuse that exact ID for every
later `inspect`, `cancel` or `stop`. The request ID is the idempotency key: the
service commits a session step once per request ID, so a retry that keeps the ID
cannot execute the action twice.

## Read before acting

`describe` reports the runtime's capabilities and current authority. `observe`
returns one shared observation and an `observation_id`. `media` reads the
encoded frames referenced by that ID, but frame data is opt-in: it is only
returned when `--include-data` is passed. A stale or missing reference is an
explicit service result; the CLI never substitutes a newer observation and
never copies frames out through an SDK.

## Submit one bounded action

`execute` submits one action JSON object, read from a path or from stdin with
`--action -`, and applies the binding's limit checks before anything reaches the
robot:

```sh
printf '%s\n' '{"timestamp_s":0,"values":{"type":"joint_position","joint_positions_deg":[0,0,0,0,0],"gripper_position":25},"metadata":{"action_space":"simulated.so101.position.v1"}}' \
  | embodirun --config "$CONFIG" execute --runtime "$RUNTIME" \
      --caller-id "$CALLER" --session-id "$SESSION" \
      --request-id action-1 --action - --steps 1 --json
```

`--observation-id`, `--max-age-ns` and `--max-skew-ns` bound the observation an
action may be based on. When those are omitted the run depends on the latest
shared observation, which is weaker evidence than an explicitly pinned one.

## Accepted is not completed

An accepted response and a completed response are different states, and
`execute` returns before the action has necessarily finished. If the request
times out, treat the outcome as unknown: keep the original request ID, run
`inspect`, and decide from the reported state. Do not resend the action after a
timeout — resending with a new ID can execute it twice, and resending with the
same ID only re-reads the committed result.

`inspect` is the only way to learn the outcome of an action whose submission was
uncertain. Preserve the returned `status`, `physical_status`, `error` and any
job or media detail when reporting the outcome.

## Cancellation is scoped

`cancel` and `stop` act on one caller/session/request ID. A reader that only
observes must not cancel another caller's job, and a client that has lost its
connection must not report that a stop was confirmed. A request to stop is not
evidence that the robot stopped.

## Recording is service-owned

Recorder state is queryable without changing ownership. `recording-status` is a
read-only check; use `recording-start`, `recording-stop` and `recording-get`
only when the service description says recording is available. A service without
a configured recorder returns an explicit `unsupported` result instead of
creating a local one, so no client should maintain a second recorder.

## The service decides

The Control service is authoritative for action limits, stale-input checks,
unsupported capabilities, busy ownership, stop confirmation and recording
state. Do not take over a serial device, kill the current owner, repeat `stop`,
or report success when the service returns `unknown`, `uncertain`,
`stop_unconfirmed`, `stale`, `busy` or `unsupported`. Report those values as
they are; they are results, not failures of the client library.
