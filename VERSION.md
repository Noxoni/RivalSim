# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 Campaign 01 — `COMPLETE` / `DEGRADED`

**Active authorized work:** none

## Completed v0.5 result

RivalSim v0.5 provides the complete bounded Rival 2.0 GPU-native training path:

`RivalSim CUDA -> RIVAL2_OBS_V1 -> actor/critic -> hybrid native action -> RivalSim x4 -> reward/done -> rollout -> GAE/PPO`

The frozen contracts are:

- `RIVAL2_OBS_V1`: 182 float32 values with proper team rotation and canonical pads;
- `RIVAL2_ACTION_V1`: five tanh-Gaussian controls plus three Bernoulli buttons;
- `RIVAL2_REWARD_V1`: exactly zero-sum goal, ball-progress, touch, and demo terms;
- `RIVAL2_EPISODE_V1`: goal termination and 15-second/45-second truncation policy.

The implementation uses 48 proven zero-copy Warp/PyTorch CUDA aliases. World state, observations,
controls, rewards, masks, rollout storage, GAE/returns, model execution, and PPO remain on the GPU
in the ordinary loop. The selected 131,072-world v0.5 benchmark point reaches 2,233,901.63 complete
agent samples/s and 89,505.78 simulated game-seconds/s with 0.588% wall CV and zero timed hot-loop
H2D/D2H traffic.

v0.5 release/evidence commit:

`cc3aa34e0bac4531c2750e0d05e2b4980621c642`

v0.5 implementation commit:

`676ef6bd3ca48376d706a2dbccbdec26fce3e4fb`

## Completed Campaign 01

Campaign 01 did not change the v0.5 milestone or contract identities. It executed the frozen
trainer from a fresh seed-`20260826` initialization through update 12 and stopped at 100,663,296
agent decision samples, the first completed update crossing 100M. The preferred 131,072-world
horizon-32 preflight passed. Initialization and the first completed updates crossing 10M, 25M,
50M, and 100M all have checkpoints and fixed evaluations.

Controlling campaign documents:

- `handoff/rival2-c01/README.md`;
- `handoff/rival2-c01/CAMPAIGN.md`;
- `handoff/rival2-c01/ACCEPTANCE.md`.

Execution completed with all 12 updates finite, zero ordinary hot-path state transfers, and exact
final checkpoint continuation. The independent behavioral result is `DEGRADED`: the final policy
lost to initialization in both fixed head-to-head modes and ordinary self-play contact rate fell.
This negative result is published without changing the v0.5 `PASS_GREEN` verdict.

## Frozen v0.4 parent

v0.4 native lifecycle authority remains:

`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`

v0.4 implementation commit:

`da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56`

v0.4 release/evidence commit:

`8a422a86c69f16f0d62073992e515575f88733b5`

## Hard stop before v0.6

No v0.6 work is authorized or begun. Still excluded:

- RLBot/CPU RocketSim deployment loader;
- Rocket League observation/action/behavior transfer validation;
- legacy Rival/Wisp weights, observations, action tables, or training compatibility;
- mechanics-specific curricula;
- distributed multi-GPU training;
- arbitrary body counts, other modes, rendering, or generic Bullet work.

Campaign 01 stopped after its 100M checkpoint/evaluation/evidence closeout. A separate controlling
handoff is required before v0.6 begins.
