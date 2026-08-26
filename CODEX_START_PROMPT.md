# Active Codex Handoff — Rival 2.0 Campaign 01

RivalSim v0.5 is complete with `PASS_GREEN`. This handoff authorizes a bounded **Rival 2.0 first real training campaign** on top of the frozen v0.5 implementation.

Start from current `origin/main`. Required starting release:

`cc3aa34e0bac4531c2750e0d05e2b4980621c642`

v0.5 implementation commit:

`676ef6bd3ca48376d706a2dbccbdec26fce3e4fb`

All v0.1-v0.5 published evidence and all four Rival 2.0 contract identities are frozen. Do not change the observation, action, reward, episode, model, PPO, or simulator contracts for this campaign.

Read in full before starting:

- `handoff/rival2-c01/README.md`;
- `handoff/rival2-c01/CAMPAIGN.md`;
- `handoff/rival2-c01/ACCEPTANCE.md`;
- `docs/RIVAL2_TRAINING_CONTRACT.md`;
- `docs/V0_5_RESULTS.md`;
- `results/v0.5/manifest.json`.

Mission:

1. create a fresh Rival 2.0 initialization using the frozen v0.5 model and default PPO settings;
2. run a capacity preflight for the standard 32-decision rollout horizon, preferring 131,072 worlds and falling back only if VRAM/stability requires it;
3. train from scratch to a bounded target of **100,000,000 agent decision samples**;
4. preserve checkpoints and fixed-seed evaluation snapshots when cumulative samples first cross 10M, 25M, 50M, and 100M, plus the initialization checkpoint;
5. evaluate every checkpoint under the exact same held-out campaign protocol and report whether recognizable non-random behavior emerges;
6. preserve the final checkpoint and enough checkpoint state to resume the campaign exactly;
7. publish Campaign 01 metrics/evidence and stop.

Use the frozen default PPO configuration from `docs/RIVAL2_TRAINING_CONTRACT.md`, including entropy coefficient `0.01`, gamma `0.995`, GAE lambda `0.95`, clip `0.20`, value coefficient `0.50`, max gradient norm `0.50`, Adam `3e-4`, two epochs, horizon `32`, and CUDA minibatches initially targeted at 65,536.

Do not tune the reward, add a curriculum, add action masks, change the model, change observation fields, change episode semantics, or run hyperparameter searches in Campaign 01. The point is to observe what the frozen v0.5 system learns as-is.

Do not begin v0.6 RocketSim/RLBot transfer work. Do not implement rendering or legacy Rival compatibility. Complete the campaign, publish what happened even if gameplay remains poor, and stop at the Campaign 01 boundary.