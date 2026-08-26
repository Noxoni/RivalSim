# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 Campaign 04 — `COMPLETE` / `CONTINUING`

**Active authorized work:** none

## Stable training baseline

Campaign 02 established the stable PPO baseline:

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

Campaign 02 proved that removing the positive entropy bonus prevented the Campaign 01 standard-deviation escalation and PPO instability.

## Completed Campaign 03

Campaign 03 added `RIVAL2_REWARD_V2` while preserving `RIVAL2_REWARD_V1` unchanged. Reward V2 adds one per-agent potential-difference term per 30-Hz decision:

`approach(agent) = (distance_before - distance_after) / 4096.0`

Reward V2 SHA-256:

`54CD5AC582133D9BA77CF7DF7976C549B3E659920BA407C9ACCE8A9FD5F50B32`

Campaign 03 trained from fresh initialization to update 12 / 100,663,296 samples. Its single 4,096-world stochastic self-play evaluation improved over Campaign 02 final:

- touches/minute: `0.291182 -> 1.308672`;
- goals/minute: `0.040362 -> 0.243800`;
- goal-terminated fraction: `0.010254 -> 0.063721`;
- no-touch truncation: `0.989746 -> 0.936279`.

Final Campaign 03 resume checkpoint:

`checkpoints/rival2/campaign03/rival2_campaign03_100m_resume.pt`

SHA-256:

`A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991`

Completed Campaign 03 commit:

`67b51452df98696a54f4465ea83924c6b9e75b4d`

## Completed Campaign 04

Campaign 04 changed no policy, reward, optimizer, simulator, or training semantic. It resumed
the exact Campaign 03 100M checkpoint and continued Reward V2 through update 120 / 1,006,632,960
cumulative samples without running update 121.

All 108 continuation updates passed integrity. The required update-30/60/90/120 checkpoints and
four 4,096-world stochastic self-play evaluations completed exactly, and the final checkpoint
passed exact reload/continuation.

The published behavioral curve is:

- 100M: `1.308672` touches/minute, `0.936279` no-touch truncation;
- 250M: `3.202896` touches/minute, `0.867676` no-touch truncation;
- 500M: `6.453265` touches/minute, `0.869873` no-touch truncation;
- 750M: `8.712013` touches/minute, `0.770752` no-touch truncation;
- 1B: `16.661451` touches/minute, `0.550293` no-touch truncation.

The prospectively frozen touch/no-touch trend classification is `CONTINUING`. Secondary goal
metrics were non-monotonic: goal rate declined from `0.426426` at 750M to `0.311649` at 1B.

Final Campaign 04 checkpoint:

`checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt`

SHA-256:

`DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0`

Campaign 04 is closed. No preflight/regression/lint ceremony, extra evaluation, viewer work, or
v0.6 work was performed. A new explicit handoff is required for any continuation.
