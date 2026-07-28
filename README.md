# Embodied runtime prototype

This repository is a provisional prototype; the product name has intentionally
not been decided. Its first vertical slice runs a π0.5 model package through a
hardware-neutral execution engine and a Torch/CUDA backend.

The neutral Python namespace is `embodied_runtime`; VLA is one model family
under `embodied_runtime.models.vla`, not the boundary of the runtime.

## Five-group boundary

```text
models          model semantics and portable execution recipe
      \                         ModelPackage
       +------------------------------+
                                      v
distributed --> registration --> engine --> BackendSession --> backends
                                      |
                                      v
                                   robots
```

The initial implementation concentrates on Groups 1, 3, and 4:

- `models/vla/pi05`: builds a staged π0.5 package (`encode_prefix`,
  `init_state`, `denoise_step`, `finalize`) and owns reference parity.
- `models/base.py`: defines the optional adapter base and the explicit
  `preprocess_one → collate → unbatch → postprocess_one` cardinality boundary.
- `engine`: selects a runner from the package's `ExecutionPlan`, then owns
  request lifecycle, cancellation, cooperative safe points, and memory policy.
- `backends/torch_cuda`: probes devices and compiles, loads, and executes
  package entrypoints and state-update primitives without importing π0.5.

Two plans are implemented: `SingleForwardPlan` and `IterativeFlowPlan`.
Model-family semantics and execution pattern are independent: π0.5 is a VLA
using iterative flow, while the tiny single-forward VLA fixture proves that
neither adapters nor the engine are tied to flow matching.

`distributed`, `robots`, and the RLinf integration start as explicit interface
boundaries and will be filled by their owning groups.

## Development

Use an environment containing PyTorch and, for real π0.5 inference, LeRobot
0.5.1:

```bash
python -m pip install -e '.[dev,torch,pi05]'
pytest
```

The checkpoint is loaded from a user-provided local path or Hugging Face ID.
This repository does not contain model weights.

The dependency-free contracts and core runtime target Python 3.11+. The
`pi05` extra requires Python 3.12+ because that is LeRobot 0.5.1's declared
minimum. A local compatibility path also covers its configuration parser on
Python 3.14.

Run the small cross-group contract fixture:

```bash
python examples/toy_flow_local.py --device cpu
```

Run the same boundary with a non-flow plan:

```bash
python examples/toy_single_forward_local.py --device cpu
```

Run a tokenizer-free real-weight π0.5 smoke test:

```bash
python examples/pi05_synthetic.py \
  --checkpoint /path/to/lerobot/pi05_base \
  --device cuda:0 \
  --dtype preserve \
  --num-steps 10 \
  --cuda-graph
```

`preserve` retains π0.5's package-defined mixed precision. Explicitly casting
the whole model to FP16 or BF16 is exposed only as an experiment because the
reference vision and normalization paths intentionally retain FP32 parameters.
`--cuda-graph` captures only the repeated denoise entrypoint; the Engine,
initial RNG, Euler update, and model adapter remain unchanged. CUDA Graph
capture failures are reported in the command summary and safely use eager
execution for that exact input signature.

The π0.5 composition explicitly declares `denoise_step.time` as a dynamic
scalar input. Its value is copied into one fixed-address CUDA tensor before
each replay, so the default ten-step schedule captures one graph rather than
ten. Undeclared Python scalars remain capture-time constants and continue to
participate in the graph cache key.

See [architecture](docs/architecture.md) for the ownership boundary and
[verification](docs/verification.md) for the tested environment and results.
