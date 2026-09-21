# MicroDuck VLN in MuJoCo

The optional MicroDuck example connects EmbodiRun's session-oriented HTTP client
to ActiveVLN inference, executes decoded R2R actions through MPC and an ONNX
walking policy, and records first/third-person videos plus episode metrics.

The complete instructions live in the
[source recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md).
It documents environment setup, the external asset layout, checksum provisioning,
local GPU and Slurm modes, custom episode files, output schemas and troubleshooting.

## Run the example

After preparing the dedicated optional environment and external assets:

```bash
git submodule update --init third_party/embodiinfer
cp examples/microduck_vln/config.env.example examples/microduck_vln/config.env
# Edit the local config with asset and Python paths.
bash examples/microduck_vln/start_demo.sh
```

Use `submit_demo.sh` in the same directory for a persistent Slurm job. No
checkpoints, evaluation datasets or simulation assets are downloaded at launch.
Only simulation is supported by this entrypoint; it opens no physical robot.

## Names and process boundary

| Component | Name |
| --- | --- |
| Deployment package | `embodirun` |
| Optional integration distribution / import | `embodirun-microduck` / `embodirun_microduck` |
| Optional service command | `embodirun-microduck-serve` |
| Default inference source | `third_party/embodiinfer` |
| Inference distribution / import | `embodiinfer` / `vvla` |
| Wire protocol | `vvla.policy.*.v1` |

EmbodiInfer's Python namespace and protocol identifiers intentionally remain
`vvla`. The integration imports the canonical
`embodirun.model_services.backends.vvla.http.VvlaHttpClient`; EmbodiRun's core
never imports model code. The optional service owns checkpoint loading and uses
the upstream policy's existing prompts and parser.

## Reference profile and verification scope

This combination is **Experimental**. CPU tests in `tests/test_microduck_vln.py`
cover bounded actions, STOP, success conditions, integrity failures and process
cleanup. With Torch and EmbodiInfer available, the same file additionally tests
the real HTTP server/client and session lifecycle with a fake model core.

| Setting | Reference configuration |
| --- | --- |
| Hardware / runtime | Linux, one A800 80 GB GPU, four CPU cores, EGL; Python 3.12 |
| Policy | Qwen2.5-VL-3B ActiveVLN, externally supplied merged SFT-v3 checkpoint |
| Model environment | Torch 2.11 / torchvision 0.26, Transformers 4.51.3, tokenizers 0.21.4 |
| Simulation environment | MuJoCo 3.8.1, ONNX Runtime 1.30, CasADi 3.7.2 |
| Inference source | EmbodiInfer `cd7dbfb37633604343cba1282b60a32e69dc0d5b`, pinned by this checkout |
| Execution | B=1, eager, bfloat16, SDPA, greedy decoding; recurrent session per episode |
| Scene / limits | `val_2`, default spawn `(6.5, 13.8, 0)`, at most 60 primitive actions |
| Success | STOP and the last three action endpoints strictly inside a 1.0 m radius |

To validate the current profile, configure assets and run the launcher above;
use the recipe's 40-episode command for an evaluation. Record the versions and
generated `results.json` with the outcome. `status=complete` reports a completed
software run; it does not imply the policy reached the target. The reported
`spl_euclidean` is a straight-line approximation, not geodesic benchmark SPL.
