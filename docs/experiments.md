# Experiments

These pages are engineering notes from opt-in hardware experiments. They record
what was measured, under which configuration, and — explicitly — what the result
does and does not establish.

!!! warning "Not user guides"
    Nothing here is part of the supported surface. A result becomes supported
    only when it also appears in the [Support matrix](support-matrix.md) with
    the versions, configuration, hardware, checkpoint, exact command, and
    observed result. Many of these experiments terminate actions in memory and
    never move a device.

## What each note covers

| Note | Question | Current conclusion |
|---|---|---|
| [Camera-only capture](camera-only-experiments.md) | Can cameras be captured, recorded, and published with no motion path at all? | Works as a read-only publisher; the motion API does not exist by construction. Paired transport measurements and TCP-fault recovery were taken under the recorded load, not as general claims. |
| [Transport candidates](transport-experiments.md) | Do shared memory, raw frames, GStreamer, Zenoh, or NIXL/UCX reduce latency? | Implemented and locally tested as opt-in paths. None has yet demonstrated a speedup or better recovery on the tested hardware. |
| [LightNav-0 with XLeRobot](lightnav0_xlerobot.md) | Can decoded local waypoints drive an XLeRobot base? | The binding and its CPU tests exist. Real-checkpoint evaluation, navigation success, and physical-robot validation are outstanding. |

## How to read a result here

- **Scope is local.** A number belongs to the exact hardware, revisions, and
  load described on that page. Revising any of them invalidates it.
- **A software result is not a robot result.** Unless a page says a physical
  robot ran the loop, treat the evidence as software-only.
- **Equal inputs are checked, not assumed.** Where a comparison claims parity,
  the page states how input fingerprints were verified.
- **Absence of a speedup is a result.** Several candidates were implemented,
  measured, and found not to help; that is recorded rather than dropped.
