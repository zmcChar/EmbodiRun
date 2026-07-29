# VLABench big-brain/small-brain experiment

## Question

Does a cloud task planner improve a fixed edge manipulation policy, rather than
merely making the deployment graph more complicated?

The first experiment uses VLABench `texas_holdem`. A high-level planner must
select the cards that define the strongest poker hand. A low-level executor
must pick those cards and place them on the placemat.

Three gates keep the claim falsifiable:

1. **Learned-executor gate:** run the same SmolVLA checkpoint with the composite
   instruction and with exact Oracle primitive instructions.
2. **Plan-content gate:** hold a simulator-privileged skill executor fixed and
   compare a deliberately wrong plan with the Oracle plan.
3. **Learned-planner gate:** validate a real cloud model's output before any
   selected card is passed to the executor.

## First real results

All rows below use VLABench seed 1000. Its seven cards contain one pair:
`4_of_diamonds` and `4_of_spades`.

| Gate | Condition | Result | Relevant measurement |
| --- | --- | ---: | --- |
| Learned executor | SmolVLA composite prompt | 0/1 | No target card grasped in 120 steps |
| Learned executor | Oracle primitive prompts + same SmolVLA | 0/1 | No target card grasped in 120 steps |
| Fixed privileged executor | Wrong plan: `10_of_diamonds`, `4_of_diamonds` | 0/1 | 226 waypoints + 12 settle steps |
| Fixed privileged executor | Oracle plan: both fours | 1/1 | 253 waypoints; VLABench terminated successfully |
| Real cloud planner | Cosmos-Reason2-2B on RTX 5080 | Rejected | 6.60 s including load; invalid `high_card` cardinality |
| Real cloud planner | Cosmos-Reason2-2B on CPU | Rejected | 16.97 s including load; peak RSS about 15.25 GB |

The SmolVLA runtime itself worked: one 50-action chunk took roughly
0.35–0.95 seconds to generate, while cached action selection was about
1.2 milliseconds. Inspection of real training samples and rollouts localized
the present failure to low-level grasp precision, not to the planning transport
or action queue.

The privileged plan-content gate produced a `+100` percentage-point paired
difference for one seed. The main environment was never edited directly:
official VLABench expert waypoints were generated in an identically seeded
shadow environment, converted from world-frame 8D to robot-frame 7D actions,
and replayed only through `SimulatorEndpoint.step`.

Cosmos-Reason2-2B returned a single fenced JSON document, but the grounded
schema validator still rejected it because it declared `high_card` while
returning three target cards. No invalid plan reached the robot executor.

A five-deal planner-only check then recreated the simulator for seeds
1000–1004. Cosmos produced **0/5 semantically correct plans**: four responses
were rejected for invalid `high_card` cardinality, and one response was
structurally valid and grounded but chose a king instead of the true pair.
This distinction matters: schema and grounding validation prevent malformed
actions, but cannot by themselves certify task semantics.

The two models were also run concurrently in one real process: Cosmos 2B
planned on CPU while SmolVLA controlled VLABench on the RTX 5080. Submitting
the asynchronous plan took 0.027 ms, the edge completed all 30 scheduled
control steps while planning remained in flight, and the environment did not
terminate. The eventual invalid cloud result was rejected and created neither
a pending nor an active plan. Edge action selection averaged 45.1 ms in this
mixed run (maximum 791.5 ms on chunk generation). Separate coordinator tests
verify that a timeout or failed replan leaves an already active edge plan
intact, and that a valid new revision cannot replace it before a declared safe
boundary.

## What this does and does not establish

The first run establishes that correct plan content can change task success
when low-level execution is held fixed and competent. The Wrong-plan result
also rules out the simplest executor or success-metric leakage.

It does **not** yet establish end-to-end superiority of the learned big/small
brain system:

- the positive executor is simulator-privileged and is not deployable;
- the paired sample size is one;
- the current SmolVLA checkpoint cannot reliably grasp the poker cards;
- the current Cosmos 2B planner failed this poker reasoning case;
- the planner consumed structured card metadata, not raw camera perception.

The honest status is therefore:

- **system decomposition:** supported by the first controlled gate;
- **current learned low-level model:** below the required skill threshold;
- **current learned cloud planner:** safely integrated but below the required
  reasoning threshold;
- **end-to-end performance claim:** not yet passed.

## Reproduction

Learned SmolVLA comparison:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python examples/vlabench_texas_holdem.py \
  --checkpoint "$VLA_EXPERIMENT_SMOLVLA_CHECKPOINT" \
  --backbone-path "$VLA_EXPERIMENT_SMOLVLM_BACKBONE" \
  --output-dir "$VLA_EXPERIMENT_OUTPUT/learned_executor" \
  --seeds 1000
```

Oracle/Wrong plan-content gate, with an optional real HF planner:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python examples/vlabench_texas_holdem_skill_gate.py \
  --output-dir "$VLA_EXPERIMENT_OUTPUT/planner_skill_gate" \
  --seeds 1000 \
  --planner-checkpoint "$VLA_EXPERIMENT_PLANNER_CHECKPOINT" \
  --planner-device cuda \
  --planner-dtype bfloat16 \
  --allow-single-json-fence
```

`planner_skill_gate.json` labels every privileged result explicitly. Only a
selection equal to the complete simulator truth receives
`simulator_oracle_privileged_upper_bound`; Wrong and learned Cloud selections
receive `simulator_privileged_plan_control`.

## Next acceptance gate

Before making a big-brain/small-brain performance claim:

1. find or train a low-level policy for which Oracle primitive plans reliably
   complete the task;
2. use at least 20 paired seeds and report exact paired significance;
3. replace Cosmos 2B or add a validated planner/tool path that passes held-out
   plan-accuracy tests (the current five-deal score is 0/5);
4. compare Edge-only, Wrong-plan, Oracle-plan, and learned Cloud-plan with the
   same executor and action budget;
5. then add asynchronous replanning, planner disconnects, observation
   perturbations, and stale-plan rejection.
