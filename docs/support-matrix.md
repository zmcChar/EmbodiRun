# Supported platforms and roadmap

Find a simulator, robot, or model below, then choose a deployment recipe.
The roadmap tracks the next integrations separately from the current catalog.

**✓ Software-tested** — implemented with automated interface tests.<br>
**◐ Experimental** — integration code is available; setup is platform-specific.<br>
**○ Planned** — on the roadmap, not yet implemented.

## Current support

<div class="grid cards support-grid" markdown>

-   ### :material-monitor: Simulators {#simulators}

    ---

    **✓ LIBERO**<br>
    Manipulation with π0.5; EmbodiInfer or SGLang.

    **◐ VLABench**<br>
    Manipulation with π0.5.

    **◐ Habitat · Isaac Sim**<br>
    Navigation with StreamVLN.

    [Browse simulator configurations](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/configs/simulation)

-   ### :material-robot: Robots {#robots}

    ---

    **✓ SO-101** — single-arm; [real-robot demos](demos/multi-robot-serving.md).<br>
    **✓ Bi-SO-101** — coordinated dual-arm policy.<br>
    **✓ Franka FR3** — adapter and π0.5 binding.

    **◐ ARX5** — DM0.5 binding.<br>
    **◐ Unitree Go2** — StreamVLN navigation.<br>
    **◐ XLeRobot** — external hardware owner.

    [Explore robot adapters](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots)

-   ### :material-brain: Models {#models}

    ---

    **✓ π0.5**<br>
    SO-101, Bi-SO-101, FR3, and simulator bindings.

    **◐ DM0.5 · StreamVLN**<br>
    ARX5 manipulation and navigation integrations.

    **◐ LightNav-0**<br>
    External XLeRobot binding; model service supplied separately.

    [EmbodiInfer model catalog](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/models.md)

</div>

These statuses describe **EmbodiRun integrations**. EmbodiInfer also runs models
such as GR00T and OpenVLA-OFT through its Python API; deploying one through
EmbodiRun additionally requires a compatible service and device binding.

## Deployment recipes

=== "Simulators"

    | Environment | Policy / backend | Setup |
    |---|---|---|
    | LIBERO | π0.5 / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-vvla.yaml) |
    | LIBERO | π0.5 / SGLang | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-sglang.yaml) |
    | VLABench | π0.5 / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/vlabench-pi05-vvla.yaml) |
    | Habitat | StreamVLN / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/habitat-streamvln-vvla.yaml) |
    | Isaac Sim | StreamVLN / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/isaac-streamvln-vvla.yaml) |

    Prepare the simulator assets, model checkpoint, and environment dependencies
    before starting a closed loop. Isaac Sim uses NVIDIA's Isaac Sim EULA.
    See [Installation](installation.md) for dependency groups.

=== "Robots"

    | Robot | Policy / interface | Setup |
    |---|---|---|
    | SO-101 | π0.5, independent single-arm runtimes | [Shared-service configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/http.yaml) |
    | Bi-SO-101 | π0.5, 12-dimensional dual-arm policy | [Dual-arm guide](pi05-bi-so101.md) |
    | Franka FR3 | π0.5 binding | [Adapter](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/franka/fr3) |
    | ARX5 | DM0.5 binding | [Binding](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/bindings/arx/x5/dm05) |
    | Unitree Go2 | StreamVLN navigation | [Robot integration](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/unitree/go2) |
    | XLeRobot | External hardware owner | [Integration package](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/xlerobot_owner/README.md) |
    | SO-101 (wired teleoperation) | Multi-leader UDP fan-out, follower-side episode collection | [Integration package](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md) |

    SO-101 deployments need calibrated arms, camera mappings, and a checkpoint
    trained for the selected binding. FR3 needs its robot SDK; ARX5 needs the
    vendor motor driver. Source links are starting points for custom integration.

    Follow [Safety](safety.md) and [Manual control](control.md) before operating
    physical hardware. [LightNav-0](lightnav0_xlerobot.md) documents the separate
    experimental XLeRobot waypoint binding.

=== "Models and transports"

    **EmbodiInfer** provides the first-party π0.5, DM0.5, and StreamVLN
    inference services. Its [serving guide](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/serving.md)
    covers checkpoints and model-specific input mappings.

    **SGLang** has a separately installed π0.5 integration. **External
    services** connect through a provider with `lifecycle: external`;
    see [Configuration](configuration.md) and the
    [external-service example](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/examples/external.yaml).

    **HTTP** is the standard transport. **WirelessComm** is an experimental,
    separately installed option; see the
    [WirelessComm configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/wireless.yaml)
    and [transport measurements](inference-transport.md).

## Roadmap

<div class="grid cards support-grid" markdown>

-   :material-robot: __Robots__

    ---

    ○ **AgileX PiperX**

    - [ ] Robot connection and observation adapter
    - [ ] Policy-to-robot action binding
    - [ ] Calibration and deployment example

-   :material-brain: __Models__

    ---

    ○ **SmolVLA · OpenVLA**

    - [ ] EmbodiInfer inference adapters
    - [ ] Serving and observation mappings
    - [ ] EmbodiRun deployment bindings

    OpenVLA here means the base model; OpenVLA-OFT already has an
    EmbodiInfer inference adapter.

-   :material-monitor: __Simulators__

    ---

    ○ **Additional simulators**

    - [ ] Select the next environments
    - [ ] Observation and action adapters
    - [ ] Example deployments and closed-loop checks

</div>

## Agents and extensions

- **MicroDuck VLN:** an experimental [MuJoCo navigation recipe](microduck-vln.md)
  with a separate integration package, HTTP inference, and video recording.
- **Wired SO-101 teleoperation:** an experimental, separately installed
  [multi-leader integration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md)
  with UDP fan-out and follower-side recording.
- **Your planner:** the [public Python client](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)
  exposes observation, proposal, execution, inspection, and cancellation.
- **RPent:** an experimental [agent integration](rpent-integration.md).
- **Astra + π0.5:** a [cooperative review loop](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/astra_pi05/README.md)
  with an injectable reviewer; the example uses a mock reviewer.
- **Camera and transport work:** [camera-only capture](camera-only-experiments.md)
  and [transport experiments](transport-experiments.md) cover optional
  GStreamer, shared-memory, Zenoh, and NIXL paths.

??? info "Tests and integration checks"

    The [fake-device walkthrough](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/run_shared_device_fake.py)
    exercises the public runtime without hardware or a checkpoint.

    Automated coverage includes
    [SO-101](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/tests/test_so101_pi05_runtime.py),
    [dual-arm deployment](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/tests/test_bi_so101_deployment.py),
    [FR3 bindings](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/tests/test_pi05_fr3_binding.py),
    [providers](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/tests/test_inference_providers.py),
    [service configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/tests/test_services.py),
    and [WirelessComm clients](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/tests/test_vvla_wireless.py).

    When contributing a new integration, include its configuration, dependencies,
    test coverage, and a deployment example. For a hardware or simulator result,
    record the versions, checkpoint, command, and observed outcome.

??? info "Runtime availability"

    Install from source with uv; there is no PyPI release yet. Compute placement
    is explicit in configuration. Automatic placement, cross-model GPU scheduling,
    and a single command that installs and runs an entire recipe are not implemented.
