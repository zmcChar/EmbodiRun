# VVLA HTTP API v1

RLinf Deploy treats VVLA as a remote policy service. The API is model-neutral:
no checkpoint, tokenizer, prompt, or raw token fields cross the boundary.

## Session ordering

Each step carries a stable `session_id`, monotonic `step_id`, and unique
`request_id`. The server must make repeated request IDs idempotent and commit
session state only once.

## Step request

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

The SO-101 binding uses the same policy action-space version but validates and
maps the six named LeRobot features through `bindings.lerobot.so101.pi05`.
Feature names are part of the safety contract: wire order alone is never used
to decide which SO-101 motor receives a value.
