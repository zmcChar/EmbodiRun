# Navigation through vLLM-Omni/OpenPI

The navigation application has two independent selectors:

- `model`: `qwen`, `streamvln`, `internvla`, or `navila`
- `runtime`: `transformers` or `vllm-omni`

`transformers` remains the default. The `vllm-omni` value currently means an
experimental remote OpenPI client and wire contract. It does not mean that an
unmodified upstream vLLM-Omni server can load every selected model.
An explicit `vllm-omni` selection fails closed if the service is unavailable or
incompatible; it never silently falls back to a different implementation. Omit
the selector or choose `transformers` to use the default route.

| Model | `transformers` | `vllm-omni` |
| --- | --- | --- |
| Qwen | Yes, existing endpoint route | No |
| StreamVLN | Yes | Experimental client; matching server required |
| InternVLA | Yes | No |
| NaVILA | Yes | Experimental client; matching server required |

The installed vLLM-Omni 0.24.0 model and pipeline registries do not contain
StreamVLN or NaVILA. Their upstream model implementations also do not satisfy
the generic vLLM Transformers modeling backend. A server-side port, hybrid, or
wrapper must therefore be deployed separately.

## Client installation and selection

Install the lightweight OpenPI client dependencies in the navigation process:

```bash
python -m pip install -e '.[navigation_vllm_omni]'
```

Select the remote runtime in TOML:

```toml
[policy]
model = "streamvln"
runtime = "vllm-omni"

[vllm_omni]
url = "ws://127.0.0.1:8000/v1/realtime/robot/openpi"
timeout_s = 120.0
# session_id = "optional-client-namespace"
```

or with the CLI:

```bash
python examples/go2_navigation.py \
  --model streamvln \
  --runtime vllm-omni \
  --vllm-omni-url ws://127.0.0.1:8000/v1/realtime/robot/openpi \
  --instruction '导航到门口'
```

The navigation process connects to the service but never starts, stops, or
reconfigures it.

Validate only the remote handshake, without a robot or image request, with:

```bash
python examples/go2_navigation_infer.py \
  --model streamvln \
  --runtime vllm-omni \
  --vllm-omni-url ws://127.0.0.1:8000/v1/realtime/robot/openpi \
  --load-only
```

## Version 1 handshake

The server's initial msgpack metadata frame must identify the private protocol,
not only the output dimensions. Common fields are:

```python
{
    "protocol_name": "embodied-runtime.navigation.openpi",
    "protocol_version": 1,
    "model_family": "streamvln",  # or "navila"
    "implementation": "vllm_hybrid",
    "needs_session_id": True,
    "input_schema": "rgb_uint8_hwc",
    "action_head": "discrete",
    "action_horizon": 4,
    "action_dim": 1,
    "padding_id": -1,
}
```

Accepted implementation labels are `vllm_native`, `vllm_hybrid`, and
`transformers_wrapper`. The label states what the separately deployed server
actually runs. A Transformers model wrapped by the OpenPI route reuses the
protocol and serving process, but is not vLLM model acceleration.

NaVILA instead declares:

```python
{
    "protocol_name": "embodied-runtime.navigation.openpi",
    "protocol_version": 1,
    "model_family": "navila",
    "implementation": "vllm_hybrid",
    "needs_session_id": True,
    "input_schema": "rgb_frames_uint8_list_hwc",
    "action_head": "navigation",
    "action_horizon": 1,
    "action_dim": 3,
}
```

The client closes the connection if any required field differs.

## Inference and reset messages

The OpenPI transport adds `endpoint = "infer"` and the isolated client
`session_id`. StreamVLN sends:

```python
{
    "client_request_id": "...",
    "episode_id": "go2-navigation",
    "observation_sequence": 7,
    "rgb": uint8_hwc_array,
    "prompt": "navigate to the doorway",
}
```

NaVILA sends `rgb_frames` as a list of one to eight HWC `uint8` arrays. Frames
may have different resolutions because preprocessing belongs to the model
server. Decoded image data is capped at 60 MiB per client request, leaving
margin below vLLM-Omni 0.24's 64 MiB WebSocket message limit.

Episode changes use the standard reset message with `episode_id` as metadata.
The random or configured client session namespace is retained; it is never
replaced by a commonly reused episode name. A reset acknowledgement means that
the server accepted the transition. Server implementations still own state
cleanup, TTL, and reset-on-next-inference behavior.

The server response contains a named float32 action tensor:

- StreamVLN: `{"discrete": float32[1, 4, 1]}`. IDs are `0=STOP`,
  `1=FORWARD`, `2=TURN_LEFT`, `3=TURN_RIGHT`, with `-1` allowed only as a
  contiguous padding suffix.
- NaVILA: `{"navigation": float32[1, 1, 3]}`. The fields are primitive ID,
  magnitude, and unit ID. Primitive IDs are `0=STOP`, `1=MOVE_FORWARD`,
  `2=TURN_LEFT`, `3=TURN_RIGHT`; unit IDs are `0=none`, `1=cm`, and
  `2=degree`.

All fields must be finite and integer-valued. The client validates native action
limits again before producing a `WaypointPlan`.

## Stateful failure semantics

StreamVLN is recurrent, so the client disables transparent request replay. If a
request is cancelled, loses its response, or returns an invalid result, the
current remote episode is marked uncertain. Navigation cannot continue until
the caller supplies an observation with `reset=true`. Each request also carries
a request ID, episode ID, and observation sequence so a server can enforce
ordering and implement idempotency.

NaVILA history is client-owned and every request carries the sampled episode
frames. It therefore does not depend on recurrent server state, but its
connection handshake is still revalidated after a reconnect.

## Local RTX 5080 verification

The local RTX 5080 was verified with the installed Blackwell-compatible stack:
Torch 2.11.0+cu130, vLLM 0.24.0, and vLLM-Omni 0.24.0. A real cached GR00T N1.7
OpenPI service initialized in 12.09 seconds and returned a 40-step action chunk
in 1.65 seconds. Observed GPU memory was 7,397 MiB idle and 7,499 MiB at the
sampled request peak. The service shut down cleanly afterward.

That run validates the local GPU, vLLM-Omni engine, and OpenPI transport. It is
not a real-checkpoint validation of StreamVLN or NaVILA: their source trees and
checkpoints are not present locally. NaVILA's 8B BF16 weights alone also exceed
the practical capacity of a 16 GiB card, so a native/hybrid local port will need
4-bit quantization or offload before end-to-end validation.
