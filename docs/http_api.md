# VVLA HTTP API v1

RLinf Deploy treats VVLA as a remote policy service. The API is model-neutral:
no checkpoint, tokenizer, prompt, or raw token fields cross the boundary.

## Session ordering

Each step carries a stable `session_id`, monotonically increasing `step_id`, and
unique `request_id`. The server must make repeated request IDs idempotent and
must commit recurrent policy memory exactly once.

## Step request

`POST /v1/sessions/{session_id}/steps` uses multipart form data.

- `metadata`: `application/json`, schema `vvla.policy.step.v1`
- `image_0..N`: encoded JPEG or PNG bytes

## Step response

```json
{
  "request_id": "step-0-...",
  "session_id": "fr3-episode-1",
  "step_id": 0,
  "session_revision": 1,
  "action_space": "franka.fr3.control.v1",
  "actions": [
    {
      "type": "joint_position",
      "values": {
        "joint_positions_rad": [0, 0, 0, -2.2, 0, 2.2, 0.7],
        "gripper_width_m": 0.04
      }
    }
  ],
  "timing": {
    "preprocess_ms": 8.0,
    "kernel_ms": 50.0,
    "policy_ms": 61.0
  }
}
```

The server, not Deploy, maps Pi0.5 or any other model output into the declared
robot action space.
