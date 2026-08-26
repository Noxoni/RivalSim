# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed execution:** Rival 2.0 overnight curriculum — `COMPLETE`

**Active authorized work:** final-45B behavioral trajectory evaluation followed by GPU-native Nexto port and full-match Rival-vs-Nexto benchmark

## Stable trained checkpoint

The completed overnight curriculum ended at update 5,403 / 45,323,649,024 cumulative agent decision samples under the preserved base `RIVAL2_REWARD_V1` after acquisition shaping had been removed.

Final resumable checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

The final held-out evaluation recorded 85.483708 touches/minute, 2.496403 goals/minute, and 0.003418 no-touch truncation. The full overnight evidence remains under `docs/RIVAL2_OVERNIGHT_RESULTS.md` and `results/rival2/overnight/`.

## Active sequence

### 1. Behavioral trajectory evaluation

Complete `handoff/rival2-behavioral-eval/README.md` first if it is still pending. This is descriptive evaluation of the final 45B checkpoint only: touch trajectories, next-touch possession, wall/backboard continuation, touch-to-goal timing, and exact goal-entry X/Z placement. It does not authorize a reward change.

### 2. Nexto port + full-match runtime

Then execute `handoff/rival2-nexto-port/README.md` without returning for a new approval.

The Nexto milestone pins the public opponent at:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Required implementation:

- faithful batched GPU-native Nexto observation adapter;
- pinned public TorchScript model with provenance/license retained;
- exact Nexto discrete action table;
- native 15 Hz Nexto neural cadence;
- exact stock hard-coded kickoff controller at 120 Hz;
- targeted observation/model/action/kickoff parity validation;
- reusable 120 Hz mixed-policy full-match scheduler;
- five-minute regulation with goal kickoffs and next-goal overtime;
- training-specific 15-second no-touch and 45-second episode truncations disabled only in the separate match runtime;
- frozen final-45B Rival policy unchanged.

Official first matchup:

- primary: 10 deterministic full matches = all five starting kickoff layouts with Rival on both sides;
- secondary: large batched stochastic-Rival versus deterministic stock-Nexto robustness suite, target 4,096 matches if practical.

Publish match outcome, touch/possession, kickoff, ball-trajectory, goal-placement, throughput, VRAM, provenance, and hot-path transfer evidence.

## Boundaries

Do not train Rival against Nexto yet. Do not alter Rival rewards, PPO, architecture, `RIVAL2_OBS_V1`, `RIVAL2_ACTION_V1`, or simulator physics. Do not build the viewer or begin v0.6. General release/lint/regression ceremony remains out of scope; only targeted fidelity validation needed to prove the Nexto port is faithful is authorized.
