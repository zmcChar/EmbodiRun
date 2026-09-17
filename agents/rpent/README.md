# RPent integration and session entrypoint

RPent remains the owner of task planning, skill selection, and business success.
This package provides a small public-client session adapter for that caller; it
does not port the old robot runtime or become a second device owner.

The prototype lives in the separate RPent-deploy-integration repository.  Use
the repository's configured remote and checkout instructions to inspect its
current files; this package deliberately does not depend on a workstation
absolute path.  The relevant source names are
`robots/mobile_cart/cooperative.py`, `astra_reviewer.py`,
`cooperative_observation.py`, and `cooperative_demo.py`, with tests under
`tests/test_cooperative.py` and `tests/test_astra_reviewer.py`.

`agents.rpent.PublicCooperativeSession` drives the public HTTP boundary. By
default it calls `ControlClient.propose()` for one 50x12 proposal, invokes the
supplied reviewer (for example `AstraCodexReviewer` through its fake-testable
command seam), submits only the reviewer's bounded choice with
`ControlClient.execute()`, and waits for a newer observation. An external
`proposal_provider` may be supplied when RPent already owns proposal creation.

The integration path is:

1. ask Deploy for a fresh shared observation through the Agent API;
2. obtain a pi0.5 proposal and call the Astra reviewer in RPent;
3. pass the structured decision through `agents.astra_pi05.decision`;
4. let an injected robot-specific adapter reject unsupported correction
   capabilities, convert supported waypoints to public joint actions, enforce
   units and limits, and report feedback;
5. discard the unexecuted proposal tail and repeat after a new observation.

The public Deploy client currently covers `describe`, `observe`, `propose`, `execute`,
job inspection/cancellation/stop, and media.  There is no public
`predict` route: `/v1/tasks` is a model task that executes directly. The
`/v1/propose` route is a single non-executing model step bound to a retained
observation; RPent still owns planning, review, and business success.

Run the local software path with:

```text
PYTHONPATH=.:src python -m agents.rpent --demo
PYTHONPATH=.:src python -m agents.rpent --validate-only
```

Both commands use fake services and report `hardware_access: false`.

Use `agents.astra_pi05.BiSO101ActionEncoder` when the Deploy owner exposes the
structured dual-arm SO101 action contract. The encoder is explicit and checks
degrees plus the canonical twelve feature names. `SO101PlanarCorrectionMapper`
also requires the caller's public `control_hz`; it rejects correction
waypoints whose `duration_s` cannot be represented by that fixed public step.

The existing RPent modules depend on the managed-runtime/robot application
stack and contain the original loop, image staging, lifecycle, SSH supervision,
camera recorder, UI, and replay fixtures. The reusable upper-layer ordering and
review contract is represented here; hardware-specific files remain external.
Until a real standalone repository and remote are chosen, this file is the
integration location and migration pointer.
