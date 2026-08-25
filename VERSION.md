# RivalSim Version Boundary

**Current completed milestone:** v0.4.0 — complete standard 1v1 game transition: `PASS_GREEN`

**Active authorized milestone:** none

## v0.4 result

The fixed standard-Soccar two-Octane/one-ball implementation now provides a complete headless
world transition with GPU-resident boost-pad, goal, scoring, kickoff, demolition, respawn, clock,
event, and reset lifecycle state. The v0.4 native authority identity is:

`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`

It binds the pinned RocketSim and binding revisions, installed extension, all 16 Soccar CMFs with
combined SHA-256 `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`,
collector source, corpus/config/seed, authority settings, and the bounded RivalSim selector/event
contract. The cache is complete and has no live acceptance fallback.

All 34 pads for both cars, both goal directions and strict score boundary, all five standard 1v1
kickoff layouts, both teams at all four respawn locations, exact 360-tick demolition lifecycle,
and deterministic mixed lifecycle/reset stress pass. The inherited v0.3 A/B/C/D gates, v0.2.2
39,236-case gate, v0.1 27-scenario gate, and both 4,608-ray backends remain green. The configured
repository suite passes 70/70 tests.

The complete path reaches 191,748.10 aggregate simulated game-seconds/s at 131,072 worlds with
0.856% CV and zero timed transfers. The reset-heavy path reaches 225,005.06 sim-s/s and
3,375,075.88 reset transitions/s with 0.723% CV and zero timed transfers.

Published v0.4 evidence is under:

- `results/v0.4/`;
- `docs/V0_4_RESULTS.md`;
- `docs/REPRODUCING_V0_4.md`;
- `docs/V0_4_AUTHORITY.md`.

v0.4 implementation commit:

`da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56`

The release/evidence commit is recorded by `results/v0.4/manifest.json` and the remote branch.

## Frozen prior versions

- **v0.3.0 — ball and dynamic contacts:** `PASS_GREEN`, release
  `d6ca3912418a3dd7ca8979415142cd861e0c0ddb`, implementation
  `a63d317b0de0522e6d3cbe243bf282c6b93a9d58`;
- **v0.2.2 — static-world source-parity breadth gate:** `PASS_GREEN`, frozen baseline
  `6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8`;
- **v0.2.1 — static-world fidelity redesign:** `PASS_GREEN`, published at
  `ad9673952de188a29b8ff15d82ac0726f1427377`;
- **v0.2.0 — arena + ground-contact proof:** `PAUSE_RED`, published at
  `2c5d11899eaaad6a963a370fcc3813202b6fa714`;
- **v0.1.0 — GPU contact-free physics proof:** `PASS`, published at
  `1f7a36cc6165273fb658ba07a8458e8d8e60628a`.

All published prior result directories remain immutable.

## Hard stop

v0.5 has not begun and is not authorized by the completed v0.4 handoff. Still excluded:

- observation construction;
- reward functions;
- training-specific action parsing or masks beyond existing controls;
- rollout buffers, GAE, PPO, or learner logic;
- PyTorch policy inference or tensor integration;
- Rival policy training;
- arbitrary body counts, other game modes, rendering, or a generic Bullet API.
