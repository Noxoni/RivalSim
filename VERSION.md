# RivalSim Version Boundary

**Current completed milestone:** v0.3.0 — ball and dynamic contacts: `PASS_GREEN`

**Active authorized milestone:** v0.4 — complete standard 1v1 game transition

## v0.3 result

The fixed standard-Soccar two-Octane/one-ball implementation passes all four frozen native
authority phases at ticks 1, 4, 8, and 12:

- ball/world: 31,216 / 31,216;
- car/ball: 8,192 / 8,192;
- car/car: 8,192 / 8,192 against both complete source-valid visitation branches;
- integrated static/dynamic contact: 512 / 512 across eight families and both branches.

Every phase has zero blocking hard mismatches and zero numeric tolerance failures. The complete
dynamic path reaches 196,614.39 aggregate simulated game-seconds/s at 131,072 worlds with 1.313%
CV, zero timed transfers, and identical full-state hashes across two independent 64-world,
2,400-tick stress runs. The v0.2.2 39,236-case static corpus, v0.1 27-scenario live corpus, both
ray backends, and all 63 repository tests remain green.

Published v0.3 evidence is under:

- `results/v0.3/`;
- `docs/V0_3_RESULTS.md`;
- `docs/REPRODUCING_V0_3.md`;
- `docs/V0_3_ORACLE_CACHE.md`.

v0.3 release commit:

`d6ca3912418a3dd7ca8979415142cd861e0c0ddb`

v0.3 implementation commit:

`a63d317b0de0522e6d3cbe243bf282c6b93a9d58`

## Authorized v0.4 boundary

v0.4 may add only the bounded standard-Soccar 1v1 lifecycle/rules needed for complete headless game transitions:

- integration of the existing 34-pad pickup/cooldown state into full lifecycle/reset;
- goals and scoring;
- standard kickoff/reset transitions;
- demolition disable/removal-from-active-physics and respawn;
- match/reset/lifecycle event state;
- generic terminal/truncation outputs needed by the later training layer;
- deterministic GPU-resident lifecycle/reset state and validation.

Controlling handoff:

- `CODEX_START_PROMPT.md`;
- `handoff/v0.4/README.md`;
- `handoff/v0.4/ACCEPTANCE.md`;
- `handoff/v0.4/LIFECYCLE_POLICY.md`.

## Frozen prior versions

- **v0.2.2 — static-world source-parity breadth gate:** `PASS_GREEN`, frozen baseline
  `6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8`;
- **v0.2.1 — static-world fidelity redesign:** `PASS_GREEN`, published at
  `ad9673952de188a29b8ff15d82ac0726f1427377`;
- **v0.2.0 — arena + ground-contact proof:** `PAUSE_RED`, published at
  `2c5d11899eaaad6a963a370fcc3813202b6fa714`;
- **v0.1.0 — GPU contact-free physics proof:** `PASS`, published at
  `1f7a36cc6165273fb658ba07a8458e8d8e60628a`.

Published prior result directories remain immutable.

## Hard stop

Do not begin v0.5 in this authorization. Still excluded:

- observation construction;
- reward functions;
- training-specific action parsing/masks beyond existing controls;
- rollout buffers, GAE, PPO, or learner logic;
- PyTorch policy inference/integration;
- Rival policy training;
- arbitrary body counts, other game modes, rendering, or a generic Bullet API.
