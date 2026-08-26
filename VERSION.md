# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 Campaign 02 — `COMPLETE` / `IMPROVED`

**Active authorized work:** Rival 2.0 Campaign 03 — direct reward-density training

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

## Authorized Campaign 03

Campaign 03 addresses reward density while keeping the stable Campaign 02 optimizer baseline.

Preserve the frozen `RIVAL2_REWARD_V1` authority. Campaign 03 introduces `RIVAL2_REWARD_V2` for this training line by adding one per-agent potential-difference term once per 30-Hz decision:

`approach(agent) = (distance_before - distance_after) / 4096.0`

where distance is true 3D Euclidean car-center to ball-center distance in unreal units, measured before the four physics ticks and at the final pre-reset transition state. Each agent receives its own term; the approach shaping is intentionally not forced to zero-sum.

No capacity preflight, initialization-control evaluation, inherited parity suite, world-count sweep, or repeated checkpoint evaluation is authorized. After a tiny GPU reward-sign/reset-leakage smoke, training starts immediately at the known-good 131,072-world / horizon-32 configuration.

Train from scratch through the first completed update crossing 100,000,000 agent decision samples. Save resumable checkpoints at the first updates crossing 25M, 50M, and 100M. Run one 4,096-world ordinary stochastic self-play evaluation only after the final checkpoint, using evaluation seed `920260826`, and compare with Campaign 02 final metrics.

Controlling specification:

- `handoff/rival2-c03/README.md`;
- root `CODEX_START_PROMPT.md`.

Do not add curricula, another reward term, another PPO/model change, action masks, legacy Rival compatibility, or v0.6 RocketSim/RLBot transfer work during Campaign 03.
