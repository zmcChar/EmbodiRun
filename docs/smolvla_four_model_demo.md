# Four-model SmolVLA deployment

This deployment runs four independent, real SmolVLA instances:

```text
RTX 5080 / robot A / edge SmolVLA --\
RTX 5080 / robot B / edge SmolVLA ----> CPU / one shared SmolVLA Provider
RTX 3070 / robot C / edge SmolVLA --/     sessions A, B, and C stay isolated
```

All four instances use the same SO100 checkpoint contract: three
`3 x 256 x 256` camera tensors, six state values, and a `[50, 6]` action
chunk. This is deliberate: the demo validates real model loading, N:1 serving,
session ordering, asynchronous failover, and disconnect recovery without an
action-space mapper becoming another variable. It does not claim that the CPU
copy is a larger or more capable policy.

The current transport is length-prefixed JSON. One synthetic SmolVLA
observation is about 10.8 MiB after JSON encoding, so this is a topology
prototype rather than a production camera/tensor transport.

## Prerequisites

Use a SmolVLA-compatible Python environment on each host and fixed local
copies of:

- `lerobot/smolvla_base`, revision
  `3326b100334ffc0a0bd1ec27e3afb1cfa2a6000c`;
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`;
- `lerobot==0.3.3`; and
- a PyTorch build supported by the local GPU.

Set these task-specific shell variables in each terminal:

```bash
SMOLVLA_PYTHON=/path/to/smolvla/python
SMOLVLA_CHECKPOINT=/path/to/smolvla_base
SMOLVLM2_BASE=/path/to/SmolVLM2-500M-Video-Instruct
```

The checkpoint is loaded with `dtype = "preserve"` and local-only model
resolution. Do not point the adapter at a newer incompatible SmolVLA snapshot.

## Start the shared CPU service

From the repository root on the 5080/CPU host:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  "$SMOLVLA_PYTHON" examples/multi_robot_cloud.py \
  --config configs/smolvla_multi_robot_cloud.toml \
  --checkpoint "$SMOLVLA_CHECKPOINT" \
  --vlm-base-path "$SMOLVLM2_BASE"
```

The process prints `MULTI_ROBOT_CLOUD_READY` only after the real model is
loaded. Hiding CUDA from this process guarantees that the cloud instance is a
CPU deployment rather than a CPU-targeted model that still creates a transient
CUDA context. Probe it without registering a robot or running inference:

```bash
PYTHONPATH=src "$SMOLVLA_PYTHON" examples/multi_robot_health.py \
  --host 127.0.0.1 \
  --port 18770
```

The checked-in CPU configuration uses the checkpoint's normal ten-step
schedule. Set `default_num_steps = 1` only for a faster topology smoke whose
action quality need not match the reference policy.

## Start two independent 5080 edge processes

Run each command in its own terminal. Both processes use `cuda:0`, but their
logical identities and observation streams are distinct.

```bash
PYTHONPATH=src "$SMOLVLA_PYTHON" examples/multi_robot_edge.py \
  --config configs/smolvla_multi_robot_edge.toml \
  --checkpoint "$SMOLVLA_CHECKPOINT" \
  --vlm-base-path "$SMOLVLM2_BASE" \
  --robot-id robot-5080-a \
  --edge-node-id edge-5080-a \
  --physical-host-id workstation-5080 \
  --physical-resource-id gpu-5080-0 \
  --observation-seed 1000 \
  --omit-output

PYTHONPATH=src "$SMOLVLA_PYTHON" examples/multi_robot_edge.py \
  --config configs/smolvla_multi_robot_edge.toml \
  --checkpoint "$SMOLVLA_CHECKPOINT" \
  --vlm-base-path "$SMOLVLM2_BASE" \
  --robot-id robot-5080-b \
  --edge-node-id edge-5080-b \
  --physical-host-id workstation-5080 \
  --physical-resource-id gpu-5080-0 \
  --observation-seed 2000 \
  --omit-output
```

## Start the 3070 edge process

Any reachable TCP path is sufficient; Wi-Fi is not part of the protocol. For
an isolated two-host test, an SSH alias and reverse tunnel avoid putting a
machine address in configuration:

```bash
EDGE_SSH_ALIAS=your-edge-host-alias
ssh -N -R 18771:127.0.0.1:18770 "$EDGE_SSH_ALIAS"
```

On the 3070 host, start the third edge with its local environment and model
paths:

```bash
PYTHONPATH=src "$SMOLVLA_PYTHON" examples/multi_robot_edge.py \
  --config configs/smolvla_multi_robot_edge.toml \
  --checkpoint "$SMOLVLA_CHECKPOINT" \
  --vlm-base-path "$SMOLVLM2_BASE" \
  --robot-id robot-3070-c \
  --edge-node-id edge-3070-c \
  --physical-host-id laptop-3070 \
  --physical-resource-id gpu-3070-0 \
  --cloud-host 127.0.0.1 \
  --cloud-port 18771 \
  --observation-seed 3000 \
  --omit-output
```

Add `--disconnect-tick 5 --reconnect-tick 9` to exactly one edge process to
test per-session logical link loss. The other sessions should remain
registered and usable.

While all three edge processes are running, repeat the health probe and verify
that it reports `registered_sessions = 3`.

The edge template uses `session_id = "auto"`, so every process launch receives
a new episode-scoped session ID. Keep that default unless restart recovery also
restores the previous cloud sequence number; reusing a fixed session after a
crash is intentionally rejected instead of silently replaying sequence IDs.

## Acceptance

The deployment is accepted when:

1. the health probe reports the CPU SmolVLA Provider and action contract
   `smolvla-so100-actions-v1`, horizon 50, dimension 6;
2. two 5080 processes and one 3070 process each produce real local actions;
3. all three sessions register against the single CPU process;
4. output records never contain another robot's session or observation ID;
5. a disconnected robot continues with `source = "edge"`;
6. reconnect increments that edge's connection epoch without invalidating the
   other two sessions; and
7. at least one later tick can select a completed asynchronous cloud result.

The permissive cloud-result age and sequence-lag values in this demo exist so a
slow CPU result is visible in the acceptance output. They are not safe defaults
for physical low-level control. A real robot must define task-level freshness,
action mapping, and a safety supervisor before a cloud result receives action
authority.

## Verified result

The complete topology was run on 2026-07-29 with two independent SmolVLA
processes on an RTX 5080, one on an RTX 3070 Laptop GPU over the SSH-carried
TCP path, and one CUDA-hidden CPU SmolVLA service. The health probe reported
the expected `[50, 6]` SO100 contract.

Observed examples from the concurrent run:

- the health endpoint reported `registered_sessions = 3` at once;
- 5080 hot edge execution was about 0.187--0.358 seconds;
- 3070 hot edge execution was about 0.208--0.214 seconds;
- an isolated, full ten-step CPU inference took about 0.683 seconds;
- full ten-step CPU inference took about 4.96--5.37 seconds while all three
  edge processes and JSON paths were active; and
- all three edge records later selected a cloud result from their own session.

For the deliberately disconnected 5080 session, ticks 3--5 continued from the
edge Provider with `cloud_disconnected`; reconnect changed its connection epoch
from 2 to 3, and ticks 9--10 selected the new session-local cloud result.
The other two sessions continued normally. The CPU service was launched with
an empty `CUDA_VISIBLE_DEVICES` and did not appear as a GPU compute process.
