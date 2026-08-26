# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 Campaign 03 — `COMPLETE`

**Active authorized work:** none

## Stable training baseline

Campaign 02 established the current PPO baseline:

- 131,072 worlds;
- horizon 32;
- entropy coefficient `0.0`;
- gamma `0.995`;
- GAE lambda `0.95`;
- PPO clip `0.20`;
- value coefficient `0.50`;
- max gradient norm `0.50`;
- Adam `3e-4`;
- two epochs;
- minibatch target 65,536;
- unchanged 182-value observation, native hybrid controller, episode semantics, model, and self-play system.

Campaign 02 stopped at 100,663,296 samples. Its optimizer remained stable: maximum KL `0.008194`, maximum clip fraction `0.087534`, and final analog standard deviation remained near one rather than escalating toward the clamp. Final ordinary stochastic self-play reached `0.291182` touches/minute and `0.040362` goals/minute, but `0.989746` of episodes still ended on the 15-second no-touch truncation.

Completed Campaign 02 commit:

`816c66b455d253b0f563bb378e53316a09ffd48e`

## Completed Campaign 03

Campaign 03 addressed reward density while keeping the stable Campaign 02 optimizer baseline.

Preserve the frozen `RIVAL2_REWARD_V1` authority. Campaign 03 introduces `RIVAL2_REWARD_V2` for this training line by adding one per-agent potential-difference term once per 30-Hz decision:

`approach(agent) = (distance_before - distance_after) / 4096.0`

where distance is true 3D Euclidean car-center to ball-center distance in unreal units, measured before the four physics ticks and at the final pre-reset transition state. Each agent receives its own term; the approach shaping is intentionally not forced to zero-sum.

The targeted GPU reward-sign/reset-leakage smoke passed, and training began immediately at the
known-good 131,072-world / horizon-32 configuration. No capacity preflight,
initialization-control evaluation, inherited parity suite, world-count sweep, or intermediate
checkpoint evaluation was run.

The run trained from scratch to update 12 / 100,663,296 agent decision samples. It saved the
required first-crossing 25M/50M/100M checkpoints, then ran exactly one 4,096-world ordinary
stochastic self-play evaluation using seed `920260826`. Touches/minute rose from `0.291182` to
`1.308672`, goals/minute rose from `0.040362` to `0.243800`, and no-touch truncation fell from
`0.989746` to `0.936279`.

Completed evidence:

- `docs/RIVAL2_CAMPAIGN03_RESULTS.md`;
- `results/rival2/campaign03/`;
- `checkpoints/rival2/campaign03/rival2_campaign03_100m_resume.pt`.

Campaign 03 is closed. Do not continue it or begin v0.6 without a new explicit handoff.
