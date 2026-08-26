# Rival 2.0 Campaign 01 — First Real Training Run

## Purpose

Campaign 01 is the first bounded attempt to train Rival 2.0 for actual gameplay using the completed v0.5 GPU-native training stack.

This is **not** a new simulator or trainer-development milestone. It is an execution campaign using the frozen v0.5 system as-is.

The question is simple:

> What does Rival 2.0 learn from scratch under the frozen v0.5 contracts after 100 million agent decision samples?

The result is useful whether the answer is “basic Rocket League emerges” or “nothing useful yet.” Do not repair the training contract mid-run in response to the outcome.

## Frozen starting point

Required parent release:

`cc3aa34e0bac4531c2750e0d05e2b4980621c642`

Required v0.5 implementation:

`676ef6bd3ca48376d706a2dbccbdec26fce3e4fb`

Frozen contracts:

- `RIVAL2_OBS_V1`;
- `RIVAL2_ACTION_V1`;
- `RIVAL2_REWARD_V1`;
- `RIVAL2_EPISODE_V1`;
- policy configuration hash `58C7409F34EA24CB7FAE7505A7F5FE2CC1B65021EE48B5200ED12BB8990C6136`.

No legacy Rival/Wisp code is involved.

## Training configuration

Use the normal frozen v0.5 PPO defaults:

- fresh random initialization;
- 32-decision rollout horizon;
- gamma `0.995`;
- GAE lambda `0.95`;
- PPO clip `0.20`;
- value coefficient `0.50`;
- entropy coefficient `0.01`;
- maximum gradient norm `0.50`;
- Adam learning rate `3e-4`;
- two PPO epochs;
- CUDA minibatch target `65,536`;
- current-policy two-sided self-play;
- historical opponent probability `20%` only after campaign checkpoints are eligible;
- maximum historical pool `16`;
- all ordinary rollout/GAE/PPO work GPU-resident.

Use one explicit campaign seed and record it in the final report. Sampling/checkpoint RNG state must remain resumable through the existing exact v0.5 checkpoint contract.

## World-count preflight

The v0.5 throughput sweep selected 131,072 worlds for horizon 4, but Campaign 01 uses the normal horizon 32.

Before the training run:

1. try 131,072 worlds at horizon 32 with the real rollout/PPO allocation;
2. require finite state, successful update, no hot-loop H2D/D2H, and adequate VRAM margin to checkpoint/evaluate safely;
3. if it does not fit or is unstable, fall back to 65,536 worlds;
4. if necessary, fall back to 32,768 worlds;
5. do not perform a broader optimization sweep.

Record the selected count and why. This is a capacity check, not a hyperparameter experiment.

## Sample budget and checkpoints

Train until cumulative agent decision samples are at least:

`100,000,000`

Preserve checkpoint/evaluation snapshots at initialization and when cumulative samples first cross:

- 10,000,000;
- 25,000,000;
- 50,000,000;
- 100,000,000.

If exact thresholds do not align with rollout boundaries, use the first completed PPO update at or beyond the threshold and record the actual sample count.

Do not continue past the first completed update that crosses 100M except for the fixed evaluation and evidence closeout.

## Held-out evaluation protocol

Freeze the evaluation seeds/configuration before training begins and use the exact same protocol for every checkpoint.

At minimum, report:

- goal-terminated episode fraction;
- 15-second no-touch truncation fraction;
- 45-second hard-truncation fraction;
- accepted touch entries per simulated minute;
- goals per simulated minute;
- demolition events per simulated minute;
- mean episode duration;
- mean absolute analog action magnitude per channel;
- jump, boost, and handbrake activation rates;
- mean analog policy standard deviation per channel;
- mean Bernoulli entropy/probability per button;
- deterministic current-checkpoint versus frozen initialization checkpoint results with sides balanced across held-out worlds;
- stochastic current-checkpoint versus frozen initialization checkpoint results with sides balanced across held-out worlds.

For checkpoint-vs-initialization matches, report score differential, goal differential, touch differential, and episode outcome counts. Do not change the opponent or evaluation seeds after seeing results.

Also report the ordinary self-play metrics needed to detect degenerate collapse even when symmetric zero-sum return averages near zero.

## Behavioral classification

Campaign 01 has no required skill threshold. Classify the outcome descriptively after all frozen evaluations are complete:

- `CLEAR_EMERGENCE`: multiple held-out gameplay metrics improve and checkpoint-vs-initialization play shows a clear advantage;
- `WEAK_EMERGENCE`: some meaningful behavior changes appear but evidence is mixed;
- `NO_CLEAR_EMERGENCE`: training remains finite but no convincing gameplay improvement is visible;
- `DEGRADED`: later checkpoints are measurably worse or behavior collapses.

This classification does not alter the v0.5 `PASS_GREEN` trainer verdict.

## Checkpoint custody

Preserve initialization and all four campaign checkpoints locally with SHA-256 identities.

For repository publication:

- publish metrics and checkpoint metadata for every checkpoint;
- commit the final resumable checkpoint if it is <=25 MiB;
- if the full resumable checkpoint is larger, commit a compact inference-only final checkpoint if that is <=25 MiB and preserve/report the exact local path and SHA-256 of the full resume checkpoint;
- do not commit multiple large optimizer checkpoints merely for archival convenience.

The final artifact must be loadable by the existing v0.5 checkpoint code without changing any frozen contract.

## Hard boundaries

Do not:

- change reward weights or reward semantics;
- add curriculum/state setters;
- alter episode timeouts;
- alter action or observation contracts;
- change network architecture;
- run a hyperparameter search;
- add scripted teacher behavior;
- import legacy Rival/Wisp;
- build RocketSim/RLBot deployment;
- begin v0.6.

If training fails numerically or architecturally, preserve the failing checkpoint/logs and stop with evidence rather than silently changing the campaign.