# Verification record

Verified on 2026-07-28 in the local WSL environment:

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
106 passed, 2 optional real-checkpoint tests skipped
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
