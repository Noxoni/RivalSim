# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 Campaign 03 — `COMPLETE`

**Active authorized work:** Rival 2.0 Campaign 04 — Reward V2 long-run continuation to 1B cumulative samples

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

## Authorized Campaign 04

Campaign 04 does not change the policy, reward, optimizer, simulator, or training semantics. It resumes the exact Campaign 03 100M checkpoint and asks one question: how far does the existing Reward V2 training line improve with substantially more samples?

Continue from update 12 through the first completed update crossing 1,000,000,000 cumulative agent decision samples. With the unchanged batch geometry, the authorized final boundary is update 120 / `1,006,632,960` cumulative samples.

Save and run the same lightweight 4,096-world ordinary stochastic self-play evaluation at:

- update 30 / 251,658,240 samples;
- update 60 / 503,316,480 samples;
- update 90 / 754,974,720 samples;
- update 120 / 1,006,632,960 samples.

Use the published Campaign 03 100M result as the starting baseline; do not rerun it.

No capacity preflight, reward smoke, initialization-control evaluation, world-count sweep, inherited simulator regression/parity suite, or post-run lint/test ceremony is authorized. Viewer work is explicitly deferred until after the 1B result is reviewed.

Controlling handoff:

- `handoff/rival2-c04/README.md`.

No v0.6 RocketSim/RLBot transfer work is authorized.