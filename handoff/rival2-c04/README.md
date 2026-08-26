# Rival 2.0 Campaign 04 — Reward V2 Long-Run Continuation

Campaign 04 is a direct continuation of the completed Campaign 03 policy. The purpose is to determine how far the existing stable Reward V2 training line continues to improve when given substantially more samples before changing the reward, PPO, model, or simulator again.

## Start state

Required parent / completed Campaign 03 HEAD:

`67b51452df98696a54f4465ea83924c6b9e75b4d`

Resume the exact full checkpoint:

`checkpoints/rival2/campaign03/rival2_campaign03_100m_resume.pt`

Expected checkpoint SHA-256:

`A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991`

Starting cumulative state:

- policy/update version: `12`;
- agent decision samples: `100,663,296`;
- reward: `RIVAL2_REWARD_V2`;
- Reward V2 SHA-256: `54CD5AC582133D9BA77CF7DF7976C549B3E659920BA407C9ACCE8A9FD5F50B32`;
- worlds: `131,072`;
- rollout horizon: `32`;
- entropy coefficient: `0.0`;
- all other Campaign 03 PPO/model/observation/action/episode/self-play settings unchanged.

Do not restart from initialization. Preserve optimizer, RNG, counters, historical-policy pool, policy version, and every other resumable checkpoint field.

## Training

Immediately load the Campaign 03 checkpoint and continue training. There is no capacity preflight, reward smoke, initialization-control run, world-count sweep, inherited regression/parity suite, or other pre-training ceremony.

Continue until the first completed PPO update at or beyond **1,000,000,000 cumulative agent decision samples**. With the unchanged batch geometry this should be update/version `120` at `1,006,632,960` cumulative samples. Do not run update 121.

Do not change:

- Reward V2 or its scale;
- entropy coefficient;
- learning rate;
- PPO clipping, GAE, gamma, epochs, minibatch target, or gradient limit;
- model architecture or initialization semantics;
- observation/action/episode contracts;
- self-play or historical-opponent behavior;
- simulator behavior.

If a genuine non-finite/runtime/checkpoint-integrity failure prevents continuation, stop with the failure. Otherwise train through the full 1B boundary even if intermediate behavior is disappointing.

## Checkpoints and evaluations

Save full resumable checkpoints on the first completed updates crossing these cumulative thresholds:

- 250M: update `30`, `251,658,240` samples;
- 500M: update `60`, `503,316,480` samples;
- 750M: update `90`, `754,974,720` samples;
- 1B: update `120`, `1,006,632,960` samples.

At each of those four checkpoints, run the same lightweight held-out evaluation used by Campaign 03 final:

- seed `920260826`;
- 4,096 worlds;
- ordinary stochastic self-play;
- first completed episode per world;
- report touches/minute, goals/minute, goal-terminated fraction, no-touch truncation fraction, mean episode duration, action-distribution diagnostics, and basic PPO stability context.

Campaign 03's published 100M evaluation is the baseline; do not rerun it.

The evaluations are allowed and expected, but do not surround them with unrelated QA or regression work.

## Closeout

Publish a compact Campaign 04 training curve and direct 100M -> 250M -> 500M -> 750M -> 1B behavioral comparison. Preserve the exact final resumable checkpoint and report whether improvement is still continuing, flattening, or degrading by 1B.

Do not run post-training pytest, Ruff, compileall, simulator parity/regression suites, extra reward smokes, capacity checks, or additional evaluations solely for ceremony. The successful training/evaluation path itself is the relevant experiment.

Do not build a viewer during Campaign 04. Viewer work is explicitly deferred until after the 1B result is reviewed.

Do not begin v0.6 RocketSim/RLBot transfer work.