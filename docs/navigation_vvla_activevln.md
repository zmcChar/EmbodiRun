# ActiveVLN navigation through Transformers and VVLA

This integration vendors [Longxmas/vvla](https://github.com/Longxmas/vvla) as
`third_party/vvla` and exposes the pinned ActiveVLN checkpoint through the
existing navigation task contract. The inference runtime is selectable:
stock Transformers rebuilds the complete multimodal history on every turn,
while VVLA owns an incremental cross-turn KV cache.

## Supported route

| Model | Runtime | Status |
| --- | --- | --- |
| `activevln` | `transformers` | implemented, in-process, stock-HF full history, B=1 |
| `activevln` | `vvla` | implemented, in-process, incremental session KV, B=1 |
| `activevln` | `vllm-omni` | unsupported by the upstream vLLM-Omni model matrix |
| StreamVLN / NaVILA | `vvla` | unsupported; VVLA has no adapters for these models |

Both local runtimes use the same Qwen2.5-VL checkpoint, processor, prompt and
strict R2R action grammar. `runtime=transformers` calls stock
`Qwen2_5_VLForConditionalGeneration.generate()` with an explicit within-turn
cache and retains the conversation and images for full-history reconstruction.
`runtime=vvla` uses Transformers only to load the weights, then runs VVLA's
self-hosted attention, autoregressive decoder, transactional session state and
cross-turn KV cache. They are therefore distinct execution paths rather than
two names for the same backend.

The submodule is pinned to commit
`6f65961c0222bbbd86ca3b09b93f13cee3ac70c8`. Initialize it after cloning:

```bash
git submodule update --init third_party/vvla
```

The upstream repository is private and the submodule uses its SSH URL. The
command therefore requires a GitHub account and SSH key authorized for
`Longxmas/vvla`.

## Isolated RTX 5080 environment

The checkpoint is stored as fp32 and occupies about 16.3 GB before activation
or KV memory. It cannot run with `dtype=auto` on a 16 GB RTX 5080. Both adapters
therefore default to BF16. Transformers requests BF16 while loading; the VVLA
adapter casts on CPU before its engine transfers the policy to CUDA. Neither
path creates a transient fp32 model on the GPU. Both runtimes inspect CUDA
capacity and reject `auto`/`float32` before weight allocation on GPUs below
20 GiB, so an RTX 5080 fails with an actionable dtype error rather than an OOM.

Create the Blackwell-compatible environment without changing the existing Go2,
NaVILA, or vLLM-Omni environments:

```bash
bash scripts/setup_vvla_activevln_env.sh
```

The script uses Python 3.10, PyTorch 2.7.1, and CUDA 12.8 wheels. Python 3.10
also leaves a compatible route for the old Habitat stack; the script itself
installs only the model/runtime layer, not Habitat or licensed assets.

Download the immutable checkpoint revision to a local directory:

```bash
env=.vvla-activevln-runtime/env/bin
checkpoint=.vvla-activevln-runtime/checkpoints/activevln

"$env/huggingface-cli" download \
  Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
  --revision 160987313e3e869705f42400d1b8f28177044518 \
  --local-dir "$checkpoint"
```

The checkpoint page currently has no published license field. Review weight
terms separately before redistributing or deploying it.

For a local checkpoint directory, both runtimes hash every required file
against VVLA's pinned `source_lock.json` before allocating GPU weights. A
different or incomplete snapshot is rejected and cannot be labelled with the
pinned revision in validation output. For remote checkpoint IDs, both runtime
constructors reject a branch name or any revision other than the pinned
immutable commit.

## Local model verification

Load each real backend on GPU without contacting a robot. The loop is
deliberately sequential: a 16 GB GPU cannot safely hold both BF16 models at
once, and process exit is the isolation boundary.

```bash
python=.vvla-activevln-runtime/env/bin/python
checkpoint=.vvla-activevln-runtime/checkpoints/activevln

for runtime in transformers vvla; do
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
    "$python" examples/go2_navigation_infer.py \
    --model activevln \
    --runtime "$runtime" \
    --vvla-root third_party/vvla \
    --model-path "$checkpoint" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attention eager \
    --max-new-tokens 64 \
    --load-only
done
```

For one real image-to-waypoint turn, choose either runtime and replace
`--load-only` with:

```text
--image /path/to/rgb.jpg --prompt "Walk through the doorway and stop by the table."
```

### Paired local RTX 5080 benchmark (2026-08-09)

The pinned local checkpoint passed the complete `source_lock.json` hash check,
then both backends ran in separate local processes on the NVIDIA GeForce RTX
5080 (compute capability 12.0) with PyTorch 2.7.1+cu128. No remote GPU was used.
The fixed conditions were BF16, eager attention, greedy decoding, batch one,
a 32,768-token context limit, 64 maximum new tokens, one warm-up episode, and
five measured episodes of two turns each. Each turn used the same 960x640
corridor image and exact instruction `Walk straight down the corridor and stop
at the closed wooden door.`; all measured responses contained 21 tokens. The
image SHA-256 was
`e1327c3d87704689f8c8d71d0e3a9da06c4b5127f9397b83786a870913f3aa84`.

| Runtime | Load, ms | Turn 1 wall p50 / mean / p95, ms | Turn 2 wall p50 / mean / p95, ms | Inference peak allocated / reserved, MiB | Host peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformers full history | 16,144.776 | 689.181 / 739.841 / 1,117.504 | 735.753 / 888.859 / 1,432.327 | 7,447.127 / 7,812 | 12,497.125 |
| VVLA incremental session KV | 10,499.830 | 575.133 / 579.702 / 619.212 | 604.644 / 599.249 / 665.985 | 7,342.464 / 7,592 | 12,583.363 |

On these five samples, VVLA's p50 was 16.55% lower on turn one (about 1.20x)
and 17.82% lower on turn two (about 1.22x). Treat those ratios as a local smoke
measurement, not a general throughput claim. Transformers ran first and VVLA
second, so the latter could benefit from the operating system's checkpoint page
cache. The load figures include hash verification and are especially sensitive
to that order; they do not establish that VVLA intrinsically loads faster. A
load-focused study should reverse/alternate process order or pre-warm the file
cache before every trial. The two models were never resident at the same time.

Reproduce the measurement as two isolated processes:

```bash
python=.vvla-activevln-runtime/env/bin/python
checkpoint=.vvla-activevln-runtime/checkpoints/activevln
image=/path/to/corridor.jpg
instruction='Walk straight down the corridor and stop at the closed wooden door.'
revision=160987313e3e869705f42400d1b8f28177044518

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
  "$python" examples/benchmark_activevln_backends.py \
  --runtime transformers --checkpoint "$checkpoint" --image "$image" \
  --instruction "$instruction" --revision "$revision" \
  --vvla-root third_party/vvla --device cuda:0 \
  --dtype bfloat16 --attention eager --max-new-tokens 64 \
  --max-context 32768 --warmups 1 --repeats 5 --turns 2 \
  > /tmp/activevln-transformers-5080.json

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
  "$python" examples/benchmark_activevln_backends.py \
  --runtime vvla --checkpoint "$checkpoint" --image "$image" \
  --instruction "$instruction" --revision "$revision" \
  --vvla-root third_party/vvla --device cuda:0 \
  --dtype bfloat16 --attention eager --max-new-tokens 64 \
  --max-context 32768 --warmups 1 --repeats 5 --turns 2 \
  > /tmp/activevln-vvla-5080.json
```

The processor input IDs and pixel tensors were checked elementwise and matched
between the two paths, but their BF16 greedy outputs did not. This is possible
because VVLA's self-hosted eager attention and decoder do not execute the
identical floating-point kernel sequence as stock HF generation.

| Runtime | Turn 1 action text | Turn 2 action text |
| --- | --- | --- |
| Transformers | `move forward 75cm, move forward 25cm, turn right 15 degrees` | `move forward 75cm, move forward 25cm, move forward 25cm` |
| VVLA | `turn left 45 degrees, turn left 45 degrees, turn left 15 degrees` | `move forward 75cm, move forward 25cm, turn right 15 degrees` |

Consequently, the turn-two full histories differ after the first assistant
response; its timing represents each backend's deployed recurrent semantics,
but it is not an identical-history microbenchmark. This run is a latency and
integration smoke test, not an action-parity or R2R accuracy claim. The local
Habitat preflight did not run an episode because Habitat/Habitat-Sim, R2R ground
truth, and licensed MP3D scenes are absent, so no SR/SPL is reported.

Both adapters send one RGB observation and an instruction. Transformers retains
the canonical conversation and image history and re-encodes it each turn; VVLA
uses an explicit `SessionKey` and appends only the new turn to its KV state. A
new episode/reset clears the respective history. Cancellation fails closed and
discards or cancels the active session before reuse.

Only the pinned R2R grammar is accepted:

- forward: 25, 50, or 75 cm;
- left/right: 15, 30, or 45 degrees;
- stop.

Invalid/truncated text or an out-of-grammar magnitude fails closed. Valid
actions become cumulative `base_link` waypoints. For ActiveVLN, Go2 converts
the cumulative plan into at most three bounded SE(2)-relative pulses and
executes the complete native chunk before capturing the next image. Habitat
also executes the complete chunk. The model's committed action history and
the environment therefore advance by the same actions.

## Habitat R2R simulation

The repository includes one BF16-aware Habitat entry point for both local
backends. It selects the same runtime adapter as Go2; the VVLA choice does not
fall back to upstream's fp32-default example.

```bash
for runtime in transformers vvla; do
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
    .vvla-activevln-runtime/env/bin/python \
    examples/vvla_activevln_habitat.py \
    --runtime "$runtime" \
    --vvla-root third_party/vvla \
    --checkpoint .vvla-activevln-runtime/checkpoints/activevln \
    --activevln-root /path/to/ActiveVLN \
    --dataset-root /path/to/R2R_VLNCE_v1-3_preprocessed \
    --scenes-dir /path/to/mp3d \
    --split val_seen \
    --scene-id 17DRP5sb8fy \
    --max-episodes 1 \
    --device cuda:0 \
    --dtype bfloat16 \
    --attention eager \
    --output "/tmp/activevln-5080-habitat-${runtime}.json"
done
```

Here `--runtime` accepts `transformers` or `vvla`; run them sequentially on a
16 GB GPU. This is a genuine Habitat renderer/action loop and reports SR/SPL.
It requires the pinned ActiveVLN/VLN-CE source, Habitat-Lab and Habitat-Sim
0.1.7, the R2R dataset, and Matterport3D scenes. Those assets are not
redistributed here; Matterport3D access is licensed separately. VVLA's
`run_injected_e2e.py` is a recorded-observation correctness gate and must not
be reported as Habitat simulation.

Before inspecting assets, the runner verifies the ActiveVLN checkout's commit,
clean status, and pinned source blobs. During evaluation it uses the same strict
backend action validator as the Go2 path, so fuzzy parser matches cannot inflate
simulation metrics with actions that deployment would reject.

Install the pinned Habitat-Sim/Lab and ActiveVLN VLN-CE dependencies into the
same Python 3.10 environment without replacing its Blackwell-compatible
PyTorch. The runner first constructs and renders a probe episode, and only then
loads the model, so missing simulator packages, task configuration, R2R files,
or MP3D scenes fail before consuming model memory. This simulator layer is
intentionally not part of `setup_vvla_activevln_env.sh`: Habitat 0.1.7 and
Matterport3D require platform-specific builds and separate data authorization.
Append `--preflight-only` and omit `--checkpoint` to validate this real
renderer/episode layer without allocating model weights.

## Runtime constraints

- Both ActiveVLN routes are in-process and batch size one. Transformers rebuilds
  the full multimodal episode history each turn and uses KV cache only within
  the current `generate()` call. VVLA keeps incremental KV across turns.
- `eager` and `sdpa` are common attention choices. `eager_bc` is VVLA-only.
  VVLA's recurrent path disables CUDA graph and full-loop capture.
- BF16 on RTX 5080 is a deployment adaptation. Upstream's committed real-model
  parity/Habitat evidence used L20 GPUs and fp32. The paired 5080 run above
  demonstrates that equal BF16 inputs do not guarantee equal tokens or actions.
- The upstream VVLA WebSocket server does not carry ActiveVLN session/reset or
  checkpoint fields. This integration intentionally uses the in-process
  `GenerationBackend` instead of that wire protocol.
