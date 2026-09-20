# LightNav-0 with XLeRobot

!!! warning "Engineering note, not a user guide"
    This binding is experimental, is not registered in the default Host
    configuration, and has no real-robot evidence. Read it as a record of an
    unfinished path. See the [Support matrix](support-matrix.md) for the
    status definitions.

This experimental binding consumes decoded local waypoints from the existing
EmbodiInfer HTTP API. EmbodiInfer owns LightNav weights, RGB history, prompts,
RVQ decoding and acceleration. EmbodiRun owns camera capture, session ordering,
action validation, velocity limits, stop feedback and episode metrics.

## Dependencies and inference service

Use a matching EmbodiInfer revision with the `lightnav0` policy and its HTTP
serving adapter. The `third_party/embodiinfer` submodule pin does not yet include
that policy, so the inference service must be provisioned separately. Follow
EmbodiInfer's `docs/proposals/0008-lightnav0-adapter.md` for server setup.

```bash
uv sync --frozen --no-dev --group binding-lightnav0
```

This installs NumPy and Pillow only. The external robot service supplies its
device dependencies. Set `--inference-token-env` to the name
of an environment variable when the inference service uses Bearer authentication.

The binding sends one lossless PNG named `observation.images.rgb`, the navigation
instruction and capture time. It sends no simulator pose, map, target coordinate
or privileged navigation hints to the model. `lightnav0.waypoints.v1` responses
must contain ten finite `[forward_m, left_m, ccw_rad]` cumulative local waypoints
and an explicit boolean `stop`. Existing HTTP session/reset/step APIs are reused.

## Authorized robot client

An already running XLeRobot HTTP service must provide authenticated control
ownership, wheel feedback and fresh timestamped camera frames. This repository
contains its client, not the robot-side HTTP server. Inference readiness is
checked before acquiring the robot control lease.

```bash
.venv/bin/python -m embodirun.bindings.xlerobot.lightnav0.cli \
  --robot-url http://robot-host:8080 --robot-token-env XLEROBOT_TOKEN \
  --inference-url http://inference-host:8050 \
  --inference-token-env INFERENCE_TOKEN \
  --instruction 'go to the door' --camera front --authorize-motion
```

The command explicitly authorizes motion. Without that flag the built-in route
refuses base commands. A custom `--robot-factory module:function` must return an
already connected and authorized robot. Stale camera/feedback, lost ownership,
timeouts and model stop trigger the existing bounded stop/cleanup paths.

## Status and outstanding work

The binding is implemented and covered at the CPU level, but it is not yet
registered in Host configuration and not integrated with the unified control
arbitration path. The matching EmbodiInfer adapter still delegates to the
upstream engine; completing the model computation inside EmbodiInfer is
outstanding.

CPU tests cover local-frame transforms, curvature-preserving velocity limits,
stale observations, authorization, cleanup and HTTP client validation. The
localhost HTTP smoke test uses a fake upstream model. Real-checkpoint HTTP
evaluation, navigation success and physical-robot validation remain outstanding.

Simulation adapters, scene evaluation code, scene assets and recordings are not
part of this binding. Historical model-output parity is not navigation success.
