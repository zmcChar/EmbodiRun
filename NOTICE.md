# Provenance notice

This is a new, independent repository. VVLA is included as the optional Git
submodule `third_party/vvla` for its inference engine and remains a separately
licensed upstream work.

The runtime submodule is pinned to:

- source: `https://github.com/Longxmas/vvla`
- commit: `6f65961c0222bbbd86ca3b09b93f13cee3ac70c8`
- code license: MIT (see `third_party/vvla/LICENSE`)

Model checkpoints and simulator datasets are external artifacts. Their terms
are independent of the VVLA code license. In particular, the pinned ActiveVLN
checkpoint does not currently publish a license field, and Matterport3D scene
assets require separate access and license review.

Selected implementation ideas and source fragments are adapted from:

- `vvla`, commit `80b5cf48c8710c69ed97200903562e9787efe105`
- copyright: `Copyright (c) 2026 Longxmas`
- license: MIT
- source workspace at adaptation time: local `vvla` Git repository

The principal adaptation map is:

| This repository | `vvla` source at the commit above |
| --- | --- |
| `src/embodied_runtime/models/vla/pi05/modeling_pi05.py` | `vvla/policies/pi05/modeling_pi05.py` |
| `src/embodied_runtime/models/vla/pi05/processing_pi05.py` | `vvla/policies/pi05/processor_pi05.py` |
| `src/embodied_runtime/models/vla/openvla_oft/head.py` | `vvla/policies/openvla_oft/head.py` |
| `src/embodied_runtime/models/vla/openvla_oft/modeling_openvla_oft.py` | `vvla/policies/openvla_oft/modeling_openvla_oft.py` |
| `src/embodied_runtime/models/vla/openvla_oft/processing_openvla_oft.py` | `vvla/policies/openvla_oft/processor_openvla_oft.py` |
| `src/embodied_runtime/models/vla/openvla_oft/vision_prismatic.py` | `vvla/policies/openvla_oft/vision_prismatic.py` |
| `src/embodied_runtime/engine/execution_engine.py` | design ideas from `vvla/engine/core.py` and `async_engine.py` |
| `src/embodied_runtime/backends/torch_cuda/operators/attention.py` | `vvla/layers/attention.py` |

Files containing a material adaptation identify the relevant upstream file in
their module docstring or comments. The implementation has been reorganized
around new model-package, execution-engine, and hardware-backend contracts.
The OpenVLA-OFT processor's BOS relocation additionally follows RLinf's
Apache-2.0 `rlinf/models/embodiment/prismatic/processing_prismatic.py`.

The optional π0.5 adapter interoperates with LeRobot 0.5.1. LeRobot is licensed
under Apache-2.0 and its π0.5 implementation includes attribution to Physical
Intelligence and Hugging Face. Model checkpoints are external artifacts and
are not redistributed by this repository; their licenses must be reviewed
separately.

The complete vvla MIT notice is included in `LICENSES/vvla-MIT.txt`.
