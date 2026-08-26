# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 Campaign 04 — `COMPLETE` / `CONTINUING`

**Active authorized work:** Rival 2.0 overnight curriculum — finish acquisition, remove approach shaping, 2B base-reward training, then 3h wall-clock continuation

## Stable training baseline

The active training line uses:

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

Campaign 03 introduced `RIVAL2_REWARD_V2`, which preserves the original base reward and adds:

`approach(agent) = (distance_before - distance_after) / 4096.0`

Campaign 04 continued Reward V2 to update 120 / 1,006,632,960 cumulative samples. At 1B the published held-out stochastic self-play result reached `16.661451` touches/minute with `0.550293` no-touch truncation, and the acquisition trend remained `CONTINUING`.

Final Campaign 04 checkpoint:

`checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt`

SHA-256:

`DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0`

Completed Campaign 04 commit:

`4c121fab8c4bfe38fbf60f1c81a47d2dce898235`

## Authorized overnight curriculum

The active handoff is `handoff/rival2-overnight/README.md`.

Phase A continues Reward V2 from the exact Campaign 04 checkpoint, evaluating every 30 PPO updates until `no_touch_truncated_fraction <= 0.01` on two consecutive 4,096-world held-out evaluations. Literal zero is not required.

After acquisition is confirmed, the training line makes one explicit curriculum transition from `RIVAL2_REWARD_V2` to the already-existing base `RIVAL2_REWARD_V1`, removing only the approach-distance shaping while preserving learned model, optimizer, RNG, counters, historical policies, and self-play state. The new checkpoint identity must truthfully bind to Reward V1.

Phase B then runs 239 additional PPO updates under Reward V1, equal to 2,004,877,312 additional agent decision samples. A full resumable checkpoint is required at that boundary.

Phase C immediately continues the same Reward V1 policy for three real elapsed wall-clock hours, ending at the first completed PPO update at or after 10,800 seconds. Intermediate checkpoint/evaluations occur around 1h and 2h, with a final checkpoint/evaluation at 3h.

No preflight/regression/parity/lint/test ceremony, viewer work, other reward/PPO/model changes, or v0.6 work is authorized.
