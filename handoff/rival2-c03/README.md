# Rival 2.0 Campaign 03 — direct reward-density training

Campaign 03 is authorized directly on top of completed Campaign 02 commit `816c66b455d253b0f563bb378e53316a09ffd48e`.

The purpose is simple: give Rival a dense signal for getting closer to the ball, then train immediately. This is intentionally **not** another validation/preflight campaign.

## Reward change: `RIVAL2_REWARD_V2`

Preserve `RIVAL2_REWARD_V1` unchanged for custody. Campaign 03 uses a new reward composition:

`reward_v2(agent) = reward_v1(agent) + approach(agent)`

where, once per 30-Hz four-tick decision interval:

`approach(agent) = (distance_before - distance_after) / 4096.0`

Requirements:

- `distance` is Euclidean 3D car-center to ball-center distance in Rocket League unreal units;
- `distance_before` is measured at the start of the decision interval;
- `distance_after` is measured from the final pre-reset transition state after the four physics ticks;
- closing distance therefore produces positive reward, increasing distance produces negative reward, and unchanged distance produces zero;
- selective kickoff/reset movement must never enter the distance delta;
- the term is per-agent and is **not required to be zero-sum**. This is deliberate: when both cars learn to approach the ball, both should receive useful learning signal rather than canceling one another;
- use the existing GPU-resident observation/state path. No CPU/NumPy/host packing is allowed in the training hot loop;
- the simplest acceptable implementation may reconstruct the true relative-ball vector from the frozen observation/transition-observation relative-position fields using the frozen position scales, or use equivalent zero-copy GPU state.

Do not alter the V1 goal, ball-progress, touch, or demo terms. Do not change `RIVAL2_OBS_V1`, `RIVAL2_ACTION_V1`, or `RIVAL2_EPISODE_V1`.

Record a deterministic content identity for `RIVAL2_REWARD_V2` in Campaign 03 evidence. Do not rewrite v0.5 or Campaign 01/02 evidence.

## Training configuration

Use Campaign 02 as the training baseline:

- fresh model initialization;
- campaign seed `20260826`;
- 131,072 worlds;
- horizon 32;
- gamma `0.995`;
- GAE lambda `0.95`;
- PPO clip `0.20`;
- value coefficient `0.50`;
- entropy coefficient **`0.0`**;
- max gradient norm `0.50`;
- Adam learning rate `3e-4`;
- two PPO epochs;
- minibatch target 65,536;
- unchanged current-policy/historical self-play rules.

Train from scratch through the first completed PPO update at or beyond **100,000,000 agent decision samples**.

Save resumable checkpoints when cumulative samples first cross 25M, 50M, and 100M. The 100M checkpoint is the final Campaign 03 checkpoint.

## No-preflight rule

Do **not** run the old capacity preflight, initialization-control evaluation, inherited simulator parity suites, world-count sweep, repeated held-out evaluations, or other ceremonial acceptance work before training.

The only launch gate is a tiny targeted reward smoke proving, on GPU, that:

1. decreasing car-ball distance gives positive approach reward;
2. increasing distance gives negative approach reward;
3. unchanged distance gives approximately zero;
4. reset motion does not contaminate the delta;
5. tensors are finite and remain device-resident.

Once that passes, **start the training run immediately**.

## Evaluation and closeout

Do not run full evaluations at intermediate checkpoints.

After the 100M checkpoint only, run one held-out **ordinary stochastic self-play** evaluation using the existing Campaign 02 evaluation seed `920260826` and 4,096 worlds. Record at minimum:

- touches per simulated minute;
- goals per simulated minute;
- goal-terminated fraction;
- no-touch truncation fraction;
- mean episode duration;
- action standard deviations/button probabilities;
- basic PPO stability metrics across the run.

Compare those final values directly against Campaign 02 final:

- touches/min: `0.291182`;
- goals/min: `0.040362`;
- goal-terminated fraction: `0.010254`;
- no-touch truncation fraction: `0.989746`.

The main question is whether the no-touch failure rate drops materially and touch frequency rises. Report the numbers without inventing a success threshold after the fact.

Publish a compact Campaign 03 report, training curve, final evaluation, checkpoint identities, and final resumable checkpoint. A full inherited v0.1-v0.5 regression rerun is not part of this campaign.

Stop after Campaign 03 closeout. Do not begin v0.6 RocketSim/RLBot transfer work and do not add another reward term, curriculum, or hyperparameter change during this run.
