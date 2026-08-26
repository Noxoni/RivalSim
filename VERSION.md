# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 Campaign 01 — `COMPLETE` / `DEGRADED`

**Active authorized work:** Rival 2.0 Campaign 02 — controlled entropy-off rerun

## Completed v0.5 result

RivalSim v0.5 provides the complete bounded Rival 2.0 GPU-native training path:

`RivalSim CUDA -> RIVAL2_OBS_V1 -> actor/critic -> hybrid native action -> RivalSim x4 -> reward/done -> rollout -> GAE/PPO`

The frozen contracts remain:

- `RIVAL2_OBS_V1`: 182 float32 values with proper team rotation and canonical pads;
- `RIVAL2_ACTION_V1`: five tanh-Gaussian controls plus three Bernoulli buttons;
- `RIVAL2_REWARD_V1`: exactly zero-sum goal, ball-progress, touch, and demo terms;
- `RIVAL2_EPISODE_V1`: goal termination and 15-second/45-second truncation policy.

v0.5 release/evidence commit:

`cc3aa34e0bac4531c2750e0d05e2b4980621c642`

v0.5 implementation commit:

`676ef6bd3ca48376d706a2dbccbdec26fce3e4fb`

Campaign 02 does not modify those frozen contracts or rewrite the v0.5 implementation.

## Completed Campaign 01

Campaign 01 used the frozen v0.5 defaults from a fresh seed-`20260826` initialization and stopped at update 12 / 100,663,296 agent decision samples. Execution was complete and numerically correct, but behavior was `DEGRADED`.

Key final evidence versus initialization:

- ordinary stochastic self-play touches/minute: `0.272091 -> 0.175624`;
- goal-terminated fraction: `0.006348 -> 0.003418`;
- no-touch truncation fraction: `0.993652 -> 0.996582`;
- stochastic head-to-head: `7-23`, touch differential `-46`;
- deterministic head-to-head: `0-819`, touch differential `-819`;
- analog policy standard deviations rose from approximately `1.0` to approximately `2.64-2.65`, close to the configured `exp(1)` ceiling.

Campaign 01 update 4 also recorded approximate KL about `1.085` and clip fraction about `0.617`.

The closed Campaign 01 result is published under `results/rival2/campaign01/` and `docs/RIVAL2_CAMPAIGN01_RESULTS.md`.

## Authorized Campaign 02

Campaign 02 is a one-variable A/B rerun intended to isolate the Campaign 01 entropy-pressure failure mode.

The only authorized learning change is:

`entropy_coefficient: 0.01 -> 0.0`

Everything else remains matched to Campaign 01, including fresh initialization procedure, campaign seed `20260826`, evaluation seed `920260826`, 131,072 worlds, horizon 32, model, observation/action/reward/episode contracts, gamma, GAE lambda, PPO clipping, value coefficient, gradient limit, Adam learning rate, two epochs, minibatch target, self-play, historical opponents, evaluation protocol, 100M stop boundary, and checkpoint thresholds.

Campaign 02 must be implemented at the campaign/configuration layer. Do not change the frozen v0.5 PPO default merely to conduct this run.

Controlling Campaign 02 documents:

- `handoff/rival2-c02/README.md`;
- `handoff/rival2-c02/DIAGNOSIS.md`;
- `handoff/rival2-c02/CAMPAIGN.md`;
- `handoff/rival2-c02/ACCEPTANCE.md`.

## Hard stop before v0.6

No v0.6 work is authorized or begun. Still excluded:

- RLBot/CPU RocketSim deployment loader;
- Rocket League transfer validation;
- reward redesign or mechanics curricula;
- action masks or legacy Rival/Wisp compatibility;
- model/hyperparameter searches outside the single Campaign 02 entropy change;
- distributed multi-GPU training;
- arbitrary body counts, other modes, rendering, or generic Bullet work.

Campaign 02 must stop after its bounded 100M closeout and return for review.