# VVLA policy API v1

RLinf Deploy treats VVLA as a remote policy service. The API is model-neutral
and may be carried by HTTP or WirelessComm:
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

The VVLA HTTP layer is only responsible for model-native action chunks. Deploy
maps `pi05.action_chunk.v1` to FR3 `joint_position` through
`bindings.fr3.pi05`.

## WirelessComm mapping

WirelessComm uses the same `vvla.policy.session.v1`, `vvla.policy.step.v1` and
result schemas. RPC metadata uses schema `vvla.policy.rpc.v1`, a per-attempt
`rpc_id`, and one of the methods `health`, `capabilities`, `open_session`,
`step`, `reset` or `close`.

For step calls, metadata fields and each encoded image are sent as one structured
payload. Image bytes are native WirelessComm byte segments; they are not encoded
as base64 or assembled into HTTP multipart data. The policy `request_id` remains
the semantic idempotency key and is distinct from the per-attempt `rpc_id`.
