# Support Matrix

Support is recorded per **complete combination**. A model, robot, or platform
being supported does not imply that arbitrary combinations work.

## Status definitions

| Status | Meaning |
|---|---|
| **Tested — software** | Covered by the repository test suite at the software-interface level. |
| **Tested — simulation** | A simulator closed loop has been run with a recorded configuration. |
| **Tested — offline model** | A model was executed offline with recorded inputs and numerical checks. |
| **Tested — real robot** | A physical robot completed the stated task with recorded evidence. |
| **Experimental** | Maintainers provide the path as work-in-progress with known limits. |
| **Planned** | No implementation yet. |

Real-robot and GPU validation records are maintained by the deployment owners
and are not produced by the CPU test suite. "Tested — software" never upgrades
to "Tested — real robot".

## Combinations

| Combination | Implementation | Verification in this repository | Entry point |
|---|---|---|---|
| π0.5 + SO-101 | Control, binding, deployment config | Tested — software; offline action checks exist | `configs/pi05/bi-so101-vvla.yaml` |
| π0.5 + Bi-SO-101 | Dual-arm adapter, physical bus ownership | Tested — software | `configs/pi05/bi-so101-vvla.yaml` |
| LIBERO + π0.5 | Simulator adapter, inference provider | Tested — software; closed loop needs GPU + checkpoint | `configs/simulation/libero-pi05-vvla.yaml` |
| LIBERO + π0.5 (SGLang) | Simulator adapter, SGLang provider | Tested — software (skipped without sglang) | `configs/simulation/libero-pi05-sglang.yaml` |
| VLABench + π0.5 | Simulator adapter | Experimental | `configs/simulation/vlabench-pi05-vvla.yaml` |
| Habitat + StreamVLN | Simulator adapter | Experimental | `configs/simulation/habitat-streamvln-vvla.yaml` |
| Isaac Sim + StreamVLN | Simulator adapter | Experimental (NVIDIA Isaac Sim EULA) | `configs/simulation/isaac-streamvln-vvla.yaml` |
| Franka FR3 + π0.5 | Robot adapter, binding | Tested — software | `src/embodirun/robots/franka/fr3` |
| ARX5 + DM05 | Robot adapter, binding | Experimental (vendor motor driver required) | `src/embodirun/bindings/arx/x5/dm05` |
| Unitree Go2 + StreamVLN | Robot agent, binding, SSH deployer | Experimental | `src/embodirun/robots/unitree/go2` |
| SGLang HTTP backend | Provider, client, adapter | Tested — software | `[sglang]` extra |
| External inference service | `lifecycle: external` model entries | Tested — software | `configs/examples/external.yaml` |
| Multi-node shared inference | Config, control arbitration, benchmark | Tested — software benchmark | `configs/http-wireless-inference/` |
| WirelessComm transport | Client, server config generation | Experimental, transport installed separately | `configs/http-wireless-inference/wireless.yaml` |
| RPent agent adapter | Public-client contract in `agents/rpent`; reference robot in the RPent fork | Experimental, software only; RPent-side registration is owned by the RPent project | `agents/rpent/README.md`, `BUAA-CI-LAB/RPent:embodirun-integration` |
| Astra + π0.5 review loop | Cooperative loop with an injectable reviewer | Experimental, mock reviewer | `agents/astra_pi05/README.md` |
| XLeRobot external owner | Optional integration package | Experimental, separately installed | `integrations/xlerobot_owner/README.md` |
| LightNav-0 + XLeRobot (remote HTTP) | Binding, remote robot client, local segment, teleop | Experimental, software | `docs/lightnav0_xlerobot.md` |
| Camera / transport experiments | GStreamer capture, camera shared memory, NIXL tensors, Zenoh endpoint | Experimental, skipped without the optional dependency | `docs/camera-only-experiments.md`, `docs/transport-experiments.md` |

## Not yet supported

- Automatic compute placement and cross-model GPU scheduling.
- A one-command `embodirun run <recipe>` that also installs and starts
  everything.
- A universal "any backend, any robot" plug-and-play guarantee. Each
  combination above is validated on its own.
- A PyPI release. Install from source with uv.

## Reporting a result

When adding or updating a combination, record the versions, configuration,
hardware, checkpoint/weights, the exact command, and the observed result, then
open a pull request against this table. Do not mark a combination Tested based
only on code presence.
