# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Active authorized milestone:** none

## Completed v0.5 result

RivalSim v0.5 provides the complete bounded Rival 2.0 GPU-native training path:

`RivalSim CUDA -> RIVAL2_OBS_V1 -> actor/critic -> hybrid native action -> RivalSim x4 -> reward/done -> rollout -> GAE/PPO`

The frozen contracts are:

- `RIVAL2_OBS_V1`: 182 float32 values with proper team rotation and canonical pads;
- `RIVAL2_ACTION_V1`: five tanh-Gaussian controls plus three Bernoulli buttons;
- `RIVAL2_REWARD_V1`: exactly zero-sum goal, ball-progress, touch, and demo terms;
- `RIVAL2_EPISODE_V1`: goal termination and 15-second/45-second truncation policy.

The implementation uses 48 proven zero-copy Warp/PyTorch CUDA aliases. World state,
observations, controls, rewards, masks, rollout storage, GAE/returns, model execution, and PPO
remain on the GPU in the ordinary loop. The selected 131,072-world training point reaches
2,233,901.63 complete agent samples/s and 89,505.78 simulated game-seconds/s with 0.588% wall CV,
14,414,032,896 peak observed VRAM bytes, and zero timed hot-loop H2D/D2H traffic.

The fixed-seed learning smoke improves its prospectively declared held-out clipped PPO objective
by `5.304016e-4`, or 4.226 standard errors, while changing both actor and critic parameters.
This proves non-no-op learning integration, not learned bot skill or Rocket League transfer.

All mandatory inherited gates remain green: v0.4 lifecycle and both ray backends; v0.3 Phase
A/B/C/D; all 39,236 v0.2.2 cases; and all 27 v0.1 live scenarios. Published v0.1 through v0.4
evidence remains byte-for-byte unchanged.

v0.5 starts from authorized handoff parent:

`dbc4b2bebe802bed58c9e143c1f9bcdb61189ac4`

The v0.5 implementation and release/evidence commits are recorded in
`results/v0.5/manifest.json`.

## Frozen v0.4 parent

v0.4 native lifecycle authority remains:

`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`

v0.4 implementation commit:

`da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56`

v0.4 release/evidence commit:

`8a422a86c69f16f0d62073992e515575f88733b5`

## Hard stop before v0.6

No v0.6 work is authorized or begun. Still excluded:

- RLBot/CPU RocketSim deployment adapter;
- Rocket League observation/action/behavior transfer validation;
- legacy Rival/Wisp weights, observations, action tables, or training compatibility;
- mechanics-specific curricula;
- distributed multi-GPU training;
- arbitrary body counts, other modes, rendering, or generic Bullet work.

A separate controlling handoff is required before any v0.6 work begins.
