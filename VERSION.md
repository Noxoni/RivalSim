# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 overnight curriculum — `COMPLETE`

**Active authorized work:** none

## Stable training baseline

The completed training line uses:

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
- unchanged 182-value observation, native hybrid controller, episode semantics, model, and
  self-play system.

## Completed overnight curriculum

The run resumed the exact Campaign 04 Reward V2 checkpoint at update 120 / 1,006,632,960
cumulative samples. Acquisition completed at update 420 after no-touch truncation reached <=1%
on two consecutive 4,096-world held-out evaluations (`0.007324`, then `0.006104`).

At update 420, only the reward contract transitioned from `RIVAL2_REWARD_V2` to the existing
base `RIVAL2_REWARD_V1`. Exact comparisons proved that learned weights, optimizer, RNG,
counters, opponent assignments, historical policies, and live runtime state were preserved.

Phase B then completed exactly 239 PPO updates / 2,004,877,312 additional Reward V1 samples.
Phase C continued the same lineage for the user-extended six real elapsed hours and stopped at
the first completed update crossing 21,600 seconds: update 5,403 / 45,323,649,024 cumulative
samples at 21,601.926 seconds.

All 5,283 continuation updates passed integrity. The final held-out evaluation recorded
85.483708 touches/minute, 2.496403 goals/minute, and 0.003418 no-touch truncation. This remains
bounded stochastic self-play evidence, not external Rocket League competence.

Final resumable checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Complete evidence is in `docs/RIVAL2_OVERNIGHT_RESULTS.md` and
`results/rival2/overnight/`.

No preflight/regression/parity/lint/test ceremony or viewer work was run. v0.6 has not begun.
