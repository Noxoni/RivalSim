# RivalSim Version Boundary

**Current completed milestone:** v0.2.2 — static-world source-parity breadth gate:
`PASS_GREEN`

**Active authorized milestone:** none

## v0.2.2 result

The frozen 39,236-case cached RocketSim authority corpus passes all 156,944 1/4/8/12-tick
checkpoints with zero hard mismatch events, zero numeric tolerance failures, and unchanged
tolerances. The complete source-correct B3 path reaches 511,886.15 aggregate simulated
game-seconds/s at 262,144 worlds with zero timed transfers, deterministic stress, and stable
scaling.

Published v0.2.2 evidence is under:

- `results/v0.2.2/`;
- `docs/V0_2_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2_2.md`;
- `docs/V0_2_2_ORACLE_CACHE.md`.

## Frozen prior versions

- **v0.2.1 — static-world fidelity redesign:** `PASS_GREEN`, published at
  `ad9673952de188a29b8ff15d82ac0726f1427377`;
- **v0.2.0 — arena + ground-contact proof:** `PAUSE_RED`, published at
  `2c5d11899eaaad6a963a370fcc3813202b6fa714`;
- **v0.1.0 — GPU contact-free physics proof:** `PASS`, published at
  `1f7a36cc6165273fb658ba07a8458e8d8e60628a`.

Published prior result directories remain immutable.

## Hard stop

Do not begin v0.3 without a separate user handoff. Still excluded:

- ball-world collision;
- car-ball collision;
- car-car collision;
- bumps/demolitions;
- goals/scoring/game rules;
- RLGym observations/rewards/PPO;
- Rival policy inference.
