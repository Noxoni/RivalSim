# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 overnight curriculum — `COMPLETE`

**Active authorized work:** final-45B behavioral trajectory / goal-placement evaluation only

## Stable trained checkpoint

The completed overnight curriculum ended at update 5,403 / 45,323,649,024 cumulative agent decision samples under the preserved base `RIVAL2_REWARD_V1` after acquisition shaping had been removed.

Final resumable checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

The final held-out evaluation recorded 85.483708 touches/minute, 2.496403 goals/minute, and 0.003418 no-touch truncation. The full overnight evidence remains under `docs/RIVAL2_OVERNIGHT_RESULTS.md` and `results/rival2/overnight/`.

## Active behavioral telemetry evaluation

The only authorized work is the evaluation defined in:

`handoff/rival2-behavioral-eval/README.md`

It loads the final 45B checkpoint and runs a large held-out stochastic current-policy self-play evaluation to measure what happens after touches and where goals enter the net.

Required outputs include per-touch ball trajectory/direction data, next-touch/possession behavior, wall/backboard continuation, touch-to-goal timing, exact goal-entry X/Z coordinates, and goal-mouth placement histograms.

This is descriptive evaluation only. Do not train, alter the current ball-progress reward or any other reward, change model/PPO/observation/action/episode/self-play behavior, build the viewer, or begin v0.6. No reward recommendation is to be implemented automatically from the telemetry result.
