# RPent integration

RPent owns task planning, skill selection, and business success. EmbodiRun
owns deployment and execution. The two meet at this repository's public Control
HTTP API; neither imports the other's internals.

## Ownership boundary

EmbodiRun does **not** modify the RPent repository. RPent resolves robots from
its own top-level `robots/<name>/` packages, so the RPent-side
`RobotSpec`/`Toolkit` registration belongs to the RPent project or to the
deployment owner. This repository ships the contract and reference client in
[`agents/rpent`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/rpent/README.md).

The maintained RPent-side adapter lives in the **`BUAA-CI-LAB/RPent` fork** on
the `embodirun-integration` branch: `robots/embodirun/` implements
`get_robot_spec`/`get_toolkit`, talks only to the public Control HTTP API, owns
no Env/VLA server or daemon, and returns `([], runtime_kwargs)` from
`init_runtime`.

## What the adapter maps

| RPent surface | EmbodiRun Control API |
|---|---|
| robot identity / capabilities | `GET /v1/describe` (including `binding.kind`, `binding.maximum_chunk_steps`, `binding.action_feature_names`) |
| `reset()` | `observe()` only — zero motion |
| `step` / `chunk_step` | one bounded `POST /v1/execute`, then `observe()` |
| `propose` | `POST /v1/propose` against a retained `observation_id` |
| `inspect` / `cancel` / `stop` | `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel`, `POST /v1/jobs/{id}/stop` |
| task success | an explicit observation/metadata signal only — never an HTTP status |

Execution semantics are the ones documented in
[`agents/CLIENT.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md):
`observe`/`propose` never move the device; only a bounded `execute` moves;
`unknown` outcomes are surfaced as unknown and never auto-resent; `cancel`
discards late results; an unconfirmed stop persists across a client restart.

## Reproducible record — software chain against a real pi0.5 service

This record exercises the RPent adapter against a real EmbodiRun Control
service and a real π0.5 inference service. It is a **software** record:
`physical_robot=false`, the device is an in-memory `simulated.policy_vector`
robot, and the two cameras are synthetic.

Conditions:

| Item | Value |
|---|---|
| Inference host | 2×A100 80GB, service on GPU0, `vvla-http` (`π0.5`) |
| Model adapter | `state_native`; cameras `observation.images.front` + `wrist`; `return_steps=50`; 6 action features |
| Control | local `ControlService` from a deployment YAML; external VVLA model endpoint via an SSH tunnel |
| Robot | `simulated.policy_vector` binding `simulated.policy_vector.pi05` |
| RPent | `BUAA-CI-LAB/RPent` `embodirun-integration`, robot `--robot embodirun` |

Observed result (abridged):

```text
describe.binding: {"action_feature_names": ["shoulder_pan.pos", "shoulder_lift.pos",
  "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
  "kind": "simulated.policy_vector.pi05", "maximum_chunk_steps": 50}
reset.observation_id: 4090:8:...:o2
reset.task_success: unverified
proposal.status: proposed mapped_actions: 50
fresh observation_id: 4090:8:...:o7
chunk_step 5-tuple -> obs 0.0 False False
info: {"cancelled": false, "discarded_late_result": false,
  "execution_evidence_unknown": false,
  "request_id": "rpent-embodirun:execute:...", "status": "completed",
  "task_success": "unverified"}
inspect.status: completed
```

Notes:

- The model round-trip can outlive the proposal's source snapshot. The adapter
  re-observes before executing the mapped prefix (receding horizon); an
  `observation_stale` response is a real, observed failure mode otherwise.
- `task_success` stays `unverified`: EmbodiRun exposes execution evidence, not
  business success.
- The recorded actions ran on the simulated device only. No physical robot was
  connected or moved.

## Known limits

- **Scene reset.** EmbodiRun has no scene-restoring reset route, so the adapter
  sets `supports_exploration=False` and `reset()` never restores a scene.
- **Real-robot guard.** `RobotSpec.is_real_robot` is frozen per package. The
  fork adapter is simulator-oriented; a real-robot package or an explicit
  safeguard decision is required before driving hardware.
- **Normal CLI planner loop.** The RPent CLI discovers `--robot embodirun` and
  its `--control-*` flags, but a full planner run needs an LLM credential and
  is not exercised by the record above.
- **RPent base install.** RPent's base dependency set (for example
  `openai-codex`) must resolve from the index you use; installing from the
  public PyPI works.
