# Verification record

Verified on 2026-07-29 in the local WSL environment:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB |
| Driver | 591.86 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.14.4 |
| LeRobot | 0.5.1 |
| Checkpoint | local `lerobot/pi05_base` snapshot, 14,467,165,872-byte safetensors |

## Automated tests

```text
156 passed, 4 optional real-checkpoint tests skipped
ruff check: passed
ruff format --check: passed
compileall: passed
```

The suite exercises both registered plan runners. It verifies that
single-forward sync, async, and three-request dynamic batching submit one
forward stage and never call the state-update primitive; iterative-flow stage
order, cancellation, deadlines, seed isolation, state replacement, and metrics
retain their previous behavior. Adapter tests enforce the B=1
preprocess/collate/unbatch cardinality contract, including tuple-of-camera
payloads.

The OpenVLA-OFT tests additionally cover action-bin masking and
de-tokenization, RLinf-style BOS/prompt/image preprocessing, the exact causal
logit slice, zero-content action queries, `SingleForwardPlan` packaging, and a
complete CPU Adapter → Engine → Torch backend request using injected tiny
components. Loader regressions cover requested-dtype shard assignment,
materialization of non-persistent Llama rotary buffers after meta
initialization, and explicit rejection of unsupported multi-image checkpoints.

The opt-in checkpoint test was also executed separately. LeRobot loaded and
remapped all 812 checkpoint keys, the adapter verified sentinel weights across
the vision, language, expert, time, input-action, and output-action paths, and
the same-noise 10-step staged/reference and Engine→Backend/reference
comparisons both passed exactly after the Adapter/ExecutionPlan refactor. The
latest opt-in run, including the typed `RunnerHost` facade, completed in 89.07
seconds.

Run that opt-in regression with:

```bash
EMBODIED_RUNTIME_PI05_CHECKPOINT=/path/to/pi05_base \
  pytest -q tests/models/test_pi05_adapter.py::test_real_local_checkpoint_loads_and_matches_reference
```

## OpenVLA-OFT structural verification

The checkpoint index and configuration for the RLinf-compatible
`Haozhan72/Openvla-oft-SFT-libero-goal-traj1` family were inspected without
downloading the 15,082,474,368-byte weight payload. Under timm 0.9.16 and the
installed Transformers 5.3.0, the reference constructor produced a fused
2,176-wide, 256-patch vision tower and a 32-layer, 4,096-wide Llama. The
checkpoint-to-module key comparison was exact:

| Component | Module keys | Checkpoint keys | Missing | Unexpected |
| --- | ---: | ---: | ---: | ---: |
| Vision backbone | 685 | 685 | 0 | 0 |
| Projector | 6 | 6 | 0 | 0 |
| Language model | 291 | 291 | 0 | 0 |

The combined reference graph has 7,541,237,184 parameters. The loader keeps
the 7B Llama on the meta device until safetensor shards are assigned, while
constructing timm's smaller vision leaf directly in BF16 because timm 0.9 uses
`Tensor.item()` during initialization and cannot be built wholly on meta.

The table above is a structural and contract validation. Run the opt-in
package-load check with:

```bash
EMBODIED_RUNTIME_OPENVLA_OFT_CHECKPOINT=/path/to/openvla-oft \
  pytest -q \
  tests/models/test_openvla_oft_adapter.py::test_real_local_checkpoint_builds_a_single_forward_package
```

## Real OpenVLA-OFT checkpoint smoke run

The RLinf-compatible
`Haozhan72/Openvla-oft-SFT-libero-goal-traj1` checkpoint at snapshot
`d20e1d447dfd87c0daa121b0739e2a379f7fe334` was loaded offline from all four
safetensor shards. The shard index and headers contained the same 982 tensor
keys and declared 15,082,474,368 bytes of tensor payload.

One deterministic synthetic 224×224 RGB image and the prompt
`pick up the red block` exercised the real tokenizer, fused DINOv2/SigLIP
preprocessing, Prismatic projector, 7B Llama forward, and categorical action
head on the RTX 5080:

| Metric | Result |
| --- | ---: |
| CPU checkpoint load | 10.453 s |
| Accelerate dispatch | 2.991 s |
| Preprocessing | 0.004 s |
| Model forward | 2.970 s |
| Peak CUDA memory allocated | 9,892.38 MiB |
| Peak CUDA memory reserved | 10,022.00 MiB |
| Action output | `(1, 8, 7)` |
| Action-token output | `(1, 56)` |

The GPU had only about 13.4 GiB free, less than the approximately 14.05 GiB
BF16 tensor payload before activations. For this correctness smoke run,
Accelerate kept Llama layers 20–31 in CPU storage and staged them to CUDA for
execution. This validates the real model adapter and reference graph, but it
does not claim that the current `TorchCudaBackend` supports offload:
`TorchCudaBackend.load()` still moves the complete runtime module to one
device. A production offload policy belongs in the backend rather than in the
OpenVLA-OFT adapter.

## Real π0.5 vertical slice

Both runs used batch size 1, three synthetic 224×224 cameras, eight synthetic
language tokens, offline checkpoint loading, preserved mixed precision, and
the eager Torch/CUDA backend.

| Flow steps | Output | Cold load | Model execution | Result |
| ---: | --- | ---: | ---: | --- |
| 1 | `(50, 32)` | 78.22 s | 0.51 s | passed |
| 10 | `(50, 32)` | 84.15 s | 1.20 s | passed |
| 10, final backend lifecycle | `(50, 32)` | 82.67 s | 0.55 s | passed |

These measurements validate the software path and are not a task-quality or
optimized latency benchmark. Inputs were synthetic, no tokenizer or robot was
connected, and weight loading was intentionally repeated in fresh processes.
The final run additionally exercised synchronized CUDA stage completion,
backend-owned Euler updates, and session-close GPU offload.

## Local GPU/CPU asynchronous collaboration

A separate real-checkpoint smoke run placed π0.5 on the RTX 5080 and a
lightweight single-forward policy on CPU. The two runtimes communicated through
the controllable dummy link so disconnect behavior was deterministic.

| Metric | Result |
| --- | ---: |
| Cloud model/device | π0.5 / `cuda:0` |
| Edge model/device | toy single-forward / CPU |
| Cloud flow steps | 1 |
| Cold load | 81.177 s |
| First tick | edge, 1.982 ms |
| Cloud in flight after first tick | true |
| Second tick | blended |
| Disconnected tick | edge / `cloud_disconnected` |
| Action shape | `(50, 32)` |
| Final action device | CPU |
| Peak CUDA allocation | 9,038.96 MiB |
| Fusion maximum absolute error | `0.0` |

This is the deterministic GPU-plus-CPU integration path. It validates that the
control tick does not await π0.5, that a completed cloud result can later be
fused on the edge device, and that disconnect invalidates cloud authority.

Run the opt-in test with:

```bash
EMBODIED_RUNTIME_PI05_CHECKPOINT=/path/to/pi05_base \
  pytest -q tests/integration/test_pi05_cpu_gpu_collaboration.py
```

## Real SmolVLA adapter and formal two-host path

The edge endpoint used an RTX 3070 Laptop GPU with Python 3.10.12, PyTorch
2.6.0+cu118, LeRobot 0.3.3, the pinned legacy `lerobot/smolvla_base`
checkpoint, and a complete local SmolVLM2 base. The checkpoint loader verified
sentinel tensors in the vision, expert, action-projection, and normalization
paths before exposing the package.

Using the same observations and initial noise, both the four model entrypoints
and the complete Adapter → Torch backend → Engine request matched LeRobot's
original ten-step `sample_actions` result exactly:

| Check | Result |
| --- | ---: |
| Parameters | 450,046,212 |
| Native action | `(50, 6)` |
| Staged/reference maximum absolute error | `0.0` |
| Engine/reference maximum absolute error | `0.0` |

SmolVLA's reference implementation accumulates its timestep as an FP32 scalar,
whereas π0.5 computes each timestep analytically in Python. The model-specific
`SmolVLAFlowPlan` preserves the former sequence while retaining the generic
`iterative_flow` runner. This matters numerically: using the π0.5-style
analytic schedule produced a `0.3854751587` maximum action difference after
unnormalization.

The final real network run kept π0.5 on the RTX 5080 host and SmolVLA on the
RTX 3070 endpoint. Both used ten flow steps and independent synthetic
observations:

| Metric | Result |
| --- | ---: |
| SmolVLA cold load | 7.840 s |
| First tick | edge `[50, 6]`, 0.496 s |
| Cloud in flight after first tick | true |
| π0.5 server execution | 1.019 s |
| Cloud wait outside control path | 0.700 s |
| Second control path | 0.205 s, cached cloud `[50, 32]` selected |
| Disconnected tick | edge `[50, 6]`, 0.209 s |
| Numeric blend | false |

This run validates two persistent real-model runtimes, an actual TCP boundary,
non-blocking cloud submission, later cloud authority, and immediate
connection-epoch fallback. It does not claim that the two policies solve the
same task: their action spaces are explicitly incompatible, and the prototype
rejects `async_blend`.

Run the edge endpoint after starting `examples/pi05_cloud_server.py`:

```bash
python examples/smolvla_pi05_async.py \
  --edge-checkpoint /path/to/smolvla_base \
  --edge-vlm-base-path /path/to/smolvlm2_base \
  --cloud-host <cloud-host> \
  --cloud-port 18765 \
  --num-steps 10 \
  --mode async_cloud_preferred
```

The opt-in exact-parity regression uses:

```bash
EMBODIED_RUNTIME_SMOLVLA_CHECKPOINT=/path/to/smolvla_base \
EMBODIED_RUNTIME_SMOLVLA_VLM_BASE=/path/to/smolvlm2_base \
  pytest -q \
  tests/models/test_smolvla_adapter.py::test_real_checkpoint_matches_reference_and_formal_engine
```

## Real two-host Wi-Fi collaboration

The real network run used the RTX 5080 host as the π0.5 cloud runtime and a
second laptop as the edge runtime:

| Role | Environment |
| --- | --- |
| Cloud | RTX 5080, Python 3.14.4, PyTorch 2.10.0+cu128 |
| Edge | RTX 3070 Laptop GPU, Python 3.10.12, PyTorch 1.11.0+cu115 |
| Edge model | 33,088-parameter temporal MLP, `(50, 32)` action |
| Transport | Length-prefixed TCP/JSON over a direct Tailscale/WireGuard peer |

The laptop's physical interface route to `<cloud-lan-ip>` was `wlp3s0`.
`tailscale ping` reported the cloud peer as direct
`via <cloud-lan-ip>:<peer-udp-port> in 4ms`, rather than through a relay. A
separate 20-packet LAN ICMP sample reported 5% loss and
3.113/10.225/118.309 ms minimum/average/maximum RTT. Direct inbound TCP to the
WSL service through `<cloud-lan-ip>:18765` was blocked by the host firewall, so
the recorded application run used the directly peered Tailscale address
`<cloud-overlay-ip>:18765`; its underlying peer path still traversed the local
Wi-Fi network.

The first direct run reported:

| Metric | Result |
| --- | ---: |
| First tick | edge, 1.469 ms |
| Edge model execution in first tick | 0.441 ms |
| Cloud still in flight after first tick | true |
| π0.5 server execution | 295.567 ms |
| Cloud round trip | 341.272 ms |
| Second tick | blended |
| Fusion time on RTX 3070 | 0.044 ms |
| Action shape/device | `(50, 32)` / `cuda:0` on the RTX 3070 |
| Fusion maximum absolute error | `2.2351741790771484e-08` |
| Request/response JSON payload | 92 / 34,090 bytes |

The same client then submitted a request to the known-closed port `18999`
while running its next edge tick. The connection failed with
`ConnectionRefusedError`, but the tick still returned an edge action in
0.577 ms; the complete failure probe settled in 11.821 ms. A subsequent request
to the real cloud port again produced a blended result, with a 413.130 ms round
trip and 368.830 ms server execution, demonstrating recovery after the failure
probe.

An additional same-address cycle fixed the laptop endpoint at
`127.0.0.1:18766` through an SSH reverse forward carried by the direct Wi-Fi
connection. With that forward removed, the cloud connection was refused while
the first tick still returned edge in 1.513 ms and the second tick remained
edge. Restoring the forward at the same address produced
`cloud_in_flight_after_first=true` and a blended second tick; the recovery
request reported a 1.317 s round trip and 1.167 s first-request server
execution. This separates address-stable disconnect/reconnect behavior from
the bad-port probe above.

The cloud command was:

```bash
python examples/pi05_cloud_server.py \
  --config configs/pi05_cpu_gpu_collaboration.toml \
  --checkpoint /path/to/pi05_base \
  --num-steps 1 \
  --host 0.0.0.0 \
  --port 18765
```

The standalone edge command was:

```bash
python3 wifi_edge_client_py310.py \
  --cloud-host <cloud-overlay-ip> \
  --cloud-port 18765 \
  --edge-device cuda:0 \
  --num-steps 1 \
  --cloud-weight 0.5 \
  --failure-probe-port 18999
```

This validates asynchronous result transport, non-blocking edge execution,
numerical action fusion, real network failure fallback, and later recovery. It
does not validate semantic compatibility or task-quality improvement between
the π0.5 cloud policy and the lightweight edge policy. The server used one
synthetic observation generated at startup, and the edge model was a
deterministic structural fixture rather than a trained robot policy. The
standalone client explicitly waits for the first cloud task outside the
real-time tick before starting its second tick, so this is not yet a sustained
fixed-frequency control-loop benchmark.

## CUDA Graph vertical slice

The backend-local CUDA Graph tests ran on the RTX 5080 and verified:

- first-call warmup/capture followed by changed-value replay;
- changed Python timestep values through one fixed-address scalar buffer and
  one cached graph;
- independent cache entries for shape and Python-constant changes;
- cloned outputs that remain stable after later replays;
- sticky eager fallback for a capture-incompatible callable;
- graph teardown before session cache release.

A real local π0.5 smoke run selected only `denoise_step` and used the default
ten flow steps. It reported `captures=1`, `replays=9`, `fallbacks=0`, and
`cached_graphs=1`: all ten timestep values used one graph. The cold run loaded
the checkpoint in 81.32 seconds and completed its first captured Engine request
in 0.53 seconds. This confirms compatibility and graph reuse, not steady-state
latency.

Same-process steady-state benchmarks compared synchronized eager execution
with CUDA Graph replay on the same loaded weights, synthetic input, and RTX
5080. The first used 30 repeated `denoise_step` calls and 10 complete one-step
Engine requests. A second used two warmups and 10 measured ten-step Engine
requests:

| Path | Eager mean | Graph mean | Speedup | Mean latency reduction |
| --- | ---: | ---: | ---: | ---: |
| `denoise_step` | 12.70 ms | 5.55 ms | 2.29× | 56.28% |
| One-step Engine request | 142.48 ms | 117.65 ms | 1.21× | 17.43% |
| Ten-step Engine request | 261.64 ms | 162.70 ms | 1.61× | 37.81% |

All stage and end-to-end comparisons had `0.0` maximum absolute output
difference. In the ten-step benchmark, two warmups plus ten measured requests
reported one capture, 119 cache-hit replays, one cached graph, and zero
fallbacks. These are local synthetic-input measurements rather than a general
performance guarantee.

## Reference parity

A separate 10-step run compared LeRobot's original `sample_actions` with the
four staged entrypoints using the same loaded weights, observation, and initial
noise:

| Metric | Result |
| --- | ---: |
| Maximum absolute action difference | `0.0` |
| Mean absolute action difference | `0.0` |
| LeRobot reference execution | 0.66 s |
| Staged execution | 0.73 s |
| Peak CUDA memory allocated | 8.84 GiB |
| Peak CUDA memory reserved | 9.13 GiB |

This verifies that exposing the prefix cache and denoise-step safe points did
not change π0.5's eager mixed-precision numerical result in the tested setup.
The opt-in regression additionally routes the same seeded request through
`ExecutionEngine` and `TorchBackendSession`, including backend-owned Euler
updates, with zero tolerance.

## Environment note

The real-weight run reused the existing `vvla` development environment. Its
installed packages are sufficient for this prototype, but `pip check` reports
that the environment is not dependency-clean: notably, LeRobot 0.5.1 requests
`draccus==0.10.0`, `numpy<2.3`, and `pynput>=1.7.8`, while that shared
environment contains different versions. A clean-environment install remains
a release gate; it is not required to reproduce the recorded local smoke run.
