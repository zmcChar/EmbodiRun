# Inference API v1

!!! note "Wire schema names"
    The wire schemas keep the `vvla.policy.*` prefix. `vvla` was the former name
    of the inference engine (now [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer)),
    and the prefix is retained so that existing deployments keep working. The
    `vvla` Python package name and CLI aliases are kept for the same reason.

## Agent proposal API

The Control API also exposes `POST /v1/propose` for one non-executing policy
step over a retained shared observation. The JSON body is:

```json
{
  "request_id": "proposal-1",
  "observation_id": "service:1:42",
  "instruction": "move to the table",
  "runtime_id": "runtime-a",
  "timeout_s": 10
}
```

The server checks caller/session authorization and observation freshness, then
uses the selected runtime binding and existing `InferenceClient` session to
return both the raw policy result and binding-mapped `RobotAction` proposals.
The requested observation is read from the shared store; the proposal path
does not recapture cameras, prepare a robot, or submit an action to the
arbiter. It closes only its temporary policy session. The response status is
`proposed`, and this is not a job or an execution receipt. A later
`POST /v1/execute` with a new request ID is required to submit selected rows.

Missing/expired observations, unsupported device-only runtimes, invalid
runtime IDs, authorization failures, and inference errors are rejected; no
action is sent when proposal generation fails. `request_id` correlates the
proposal request and policy step, but does not create a persistent job.

EmbodiRun treats the inference service as a remote policy service. The API is
model-neutral and may be carried by HTTP or WirelessComm:
no checkpoint, tokenizer, prompt, or raw token fields cross the boundary.

## Session ordering

Each step carries a stable `session_id`, monotonic `step_id`, and unique
`request_id`. The server must make repeated request IDs idempotent and commit
session state only once.

## HTTP step request

`POST /v1/sessions/{session_id}/steps` uses multipart form data.

- `metadata`: `application/json`, schema `vvla.policy.step.v1`
- `image_0..N`: encoded JPEG/PNG bytes

## Step response

```json
{
  "request_id": "step-0-...",
  "session_id": "fr3-episode-1",
  "step_id": 0,
  "session_revision": 1,
  "action_space": "pi05.action_chunk.v1",
  "actions": [
    {
      "type": "action_chunk",
      "values": {
        "data": [
          [0.12, -0.03, 0.45, -2.21, 0.08, 1.12, -0.34, 0.04]
        ],
        "feature_names": [
          "joint_1",
          "joint_2",
          "joint_3",
          "joint_4",
          "joint_5",
          "joint_6",
          "joint_7",
          "gripper_width_m"
        ]
      }
    }
  ],
  "timing": {
    "policy_ms": 61.0
  }
}
```

The inference HTTP layer is only responsible for model-native action chunks.
EmbodiRun maps and validates the rows in `pi05.action_chunk.v1` through the
selected binding, executes the `run --chunk-steps` prefix, and plays those robot
commands at the runtime control rate. Robot state fields and action dimensions
are owned by the binding rather than a user-maintained adapter file.

## WirelessComm mapping

WirelessComm uses the same `vvla.policy.session.v1`, `vvla.policy.step.v1` and
result schemas. RPC metadata uses schema `vvla.policy.rpc.v1`, a per-attempt
`rpc_id`, and one of the methods `health`, `capabilities`, `open_session`,
`step`, `reset` or `close`.

For step calls, metadata fields and each encoded image are sent as one structured
payload. Image bytes are native WirelessComm byte segments; they are not encoded
as base64 or assembled into HTTP multipart data. The policy `request_id` remains
the semantic idempotency key and is distinct from the per-attempt `rpc_id`.
