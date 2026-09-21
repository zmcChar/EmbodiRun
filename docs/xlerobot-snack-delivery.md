# XLeRobot snack delivery recipe

This recipe is a public orchestration example for a mobile manipulation scene:
an XLeRobot follows a route to a pickup table, asks an upper-layer RPent/Astra
agent to review the current camera view, obtains a bounded π0.5/VLA proposal,
returns on a second route, and presents the item with a calibrated arm pose.

The recipe keeps the ownership boundary explicit:

- **EmbodiRun** owns deployment, shared observations, action contracts,
  freshness checks, bounded execution, stop requests, and feedback.
- **The XLeRobot owner** is the only process that reads serial buses and
  cameras. It publishes state age, camera age, scope ownership, stop evidence,
  and task evidence through the external-owner HTTP boundary.
- **RPent/Astra** remains an optional upper-layer planner/reviewer. The adapter
  can be replaced with another factory without changing the runtime or robot
  contract.
- **The VLA service** is an external inference endpoint. Checkpoints and
  credentials are deployment inputs and are not stored in this repository.

## Start from the recipe directory

The complete recipe package is under
[`recipes/xlerobot/snack_delivery`](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/recipes/xlerobot/snack_delivery):

| Need | Document |
|---|---|
| Scene, components, and agent/model split | [`README.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/recipes/xlerobot/snack_delivery/README.md) |
| Hardware bill of materials and planning prices | [`hardware.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/recipes/xlerobot/snack_delivery/hardware.md) |
| Setup, calibration, routes, and supervised execution | [`guide.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/recipes/xlerobot/snack_delivery/guide.md) |
| Public interfaces and evidence contract | [`architecture.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/recipes/xlerobot/snack_delivery/architecture.md) |
| Deployment fields | [`deployment.example.yaml`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/recipes/xlerobot/snack_delivery/deployment.example.yaml), [`config.example.json`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/recipes/xlerobot/snack_delivery/config.example.json) |
| Local setup and launch | [`scripts/recipes/xlerobot_snack_setup.sh`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/scripts/recipes/xlerobot_snack_setup.sh), [`scripts/recipes/xlerobot_snack.sh`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/scripts/recipes/xlerobot_snack.sh) |

The default route files are marked as fixtures and are accepted only by
`dry-run`. For a real scene, record directed route chunks in the target space
or ask RPent to select from locally recorded segments using a route diagram.
An unscaled diagram is never converted directly into motor velocities by this
recipe.

## Software-only verification

The recipe has a deterministic fixture path and a real local HTTP integration
test with a fake owner. These checks exercise the public Control client,
runtime selection, freshness fields, action units, proposal review boundary,
and stop/evidence handling. They do not establish physical task success.

```bash
PYTHONPATH=src:$PWD python -m pytest -q \
  tests/test_snack_delivery_recipe.py \
  tests/test_rpent_snack.py \
  tests/test_snack_http_integration.py
```

For hardware, follow the owner and safety instructions in the recipe guide,
keep an operator present, and validate the robot-specific calibration and
emergency stop before allowing motion. Do not copy a route, calibration file,
camera identity, access file, or recorded scene from another deployment.
