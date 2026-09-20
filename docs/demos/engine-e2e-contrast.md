# Engine end-to-end contrast

One arm, one task, one checkpoint, one network — only the inference engine
changes. Four engines run the same task on real hardware, with per-chunk
inference and communication latency annotated in the video.

<video controls muted playsinline preload="metadata" width="720">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/engine_e2e_contrast.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*29 s. Four columns: EmbodiInfer over WirelessComm, EmbodiInfer over HTTP,
SGLang, and the native LeRobot reference. Every column runs the same 15 chunks
and is cut after its own 8th, so the faster engines freeze first.*

## Setup

| | |
|---|---|
| Task | `Pick up the cube and place it in the bowl.` |
| Controlled variable | the inference engine only |
| Arm | SO-101 follower, 5 DoF + gripper |
| Control | 20 Hz, 50-step chunks — 2.45 s of fixed playback per chunk |
| Cameras | front + wrist, 640×480 at 20 Hz |
| Network | Wi-Fi, both machines on the same access point |
| Model | π0.5 on the SO-101 four-arm checkpoint, 10 denoising steps |

## Measurements

Median over 15 chunks per run:

| Engine | Inference | Comm + queue | Playback, fixed | Chunk period |
|---|---|---|---|---|
| **EmbodiInfer, WirelessComm** | **162 ms** | 42 ms | 2455 ms | **2660 ms** |
| EmbodiInfer, HTTP | 170 ms | 45 ms | 2453 ms | 2666 ms |
| SGLang | 194 ms | 65 ms | 2453 ms | 2713 ms |
| Native LeRobot, reference | 1061 ms | 82 ms | 2453 ms | 3592 ms |

## Reading it honestly

This is a **tuned-versus-out-of-the-box** comparison, not an equal-effort one:

- EmbodiInfer runs its full set of optimisations: bfloat16, `inductor`
  compilation, prefix and denoising CUDA graphs, and Triton attention.
- SGLang runs upstream defaults. The one remaining candidate,
  `--enable-torch-compile`, was measured and gave no benefit.
- Native LeRobot deliberately runs plain eager, with no CUDA graph and no
  `torch.compile`. It is the reference point, and its 1061 ms can be compressed
  further.

Two observations matter more than the headline ratio:

- **Communication is not the difference.** 42–82 ms is 1.6–3% of a 2660 ms
  chunk, so almost the entire gap between engines is inference.
- **Playback dominates.** The 2.45 s of physical execution is 92% of the chunk
  period and identical across all four engines. Optimising the engine can only
  move the remaining fraction; making the task itself faster means changing the
  chunk length or the control rate, not the engine.

## Reproducing it

The four engines were served side by side on one GPU host, with the control
stack on the robot side selecting a transport and a policy per run. See
[Inference API v1](../http_api.md) for the wire contract the benchmark uses and
[Configuration](../configuration.md) for how a deployment selects a model and a
transport.
