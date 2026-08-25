# RivalSim Version Boundary

**Current completed milestone:** v0.4.0 — complete standard 1v1 game transition: `PASS_GREEN`

**Active authorized milestone:** v0.5 — Rival 2.0 GPU-native training

## Frozen v0.4 result

The fixed standard-Soccar two-Octane/one-ball implementation provides a complete headless
world transition with GPU-resident boost-pad, goal, scoring, kickoff, demolition, respawn, clock,
event, and reset lifecycle state. The v0.4 native authority identity is:

`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`

All 34 pads for both cars, both goal directions and strict score boundary, all five standard 1v1
kickoff layouts, both teams at all four respawn locations, exact 360-tick demolition lifecycle,
and deterministic mixed lifecycle/reset stress pass. The inherited v0.3 A/B/C/D gates, v0.2.2
39,236-case gate, v0.1 27-scenario gate, and both 4,608-ray backends remain green.

The complete path reaches 191,748.10 aggregate simulated game-seconds/s at 131,072 worlds with
0.856% CV and zero timed transfers. The reset-heavy path reaches 225,005.06 sim-s/s and
3,375,075.88 reset transitions/s with 0.723% CV and zero timed transfers.

v0.4 implementation commit:

`da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56`

v0.4 release/evidence commit:

`8a422a86c69f16f0d62073992e515575f88733b5`

All published prior result directories remain immutable.

## Active v0.5 authorization

v0.5 creates **Rival 2.0** as a clean-slate GPU-native policy/training system directly on RivalSim.

The old `Noxoni/Rival` training implementation, Wisp-derived weights, 432-value observation,
90/158-action lookup tables, and legacy training schemas are not v0.5 dependencies and do not
constrain Rival 2.0.

The fixed Rival 2.0 controller is:

- five continuous analog controls: throttle, steer, pitch, yaw, roll;
- three Bernoulli buttons: jump, boost, handbrake;
- 13 actor outputs total: five means, five log standard deviations, three button logits;
- 30-Hz decisions, each held for four 120-Hz physics ticks.

Authorized v0.5 scope:

- zero-copy RivalSim/PyTorch CUDA tensor bridge;
- new `RIVAL2_OBS_V1` observation contract;
- new `RIVAL2_REWARD_V1` and trainer-owned episode policy;
- GPU-resident rollout storage;
- GPU-native GAE and PPO;
- checkpoint/save/resume;
- current-policy self-play and bounded historical Rival 2.0 opponents;
- end-to-end learning sanity and performance sweeps.

Controlling documents:

- `handoff/v0.5/README.md`;
- `handoff/v0.5/ACCEPTANCE.md`;
- `handoff/v0.5/RIVAL2_CONTRACT.md`;
- root `CODEX_START_PROMPT.md`.

## Hard stop after v0.5

Do not begin v0.6 without a separate handoff. Still excluded from v0.5:

- RLBot deployment;
- CPU RocketSim transfer evaluation;
- actual Rocket League transfer validation;
- compatibility work for legacy Rival/Wisp weights or action tables;
- distributed multi-GPU training;
- mechanics-specific curricula;
- arbitrary body counts, other game modes, rendering, or generic Bullet work.
