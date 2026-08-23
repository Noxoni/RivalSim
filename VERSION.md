# RivalSim Version Boundary

**Current completed milestone:** v0.2.0 — arena + ground-contact proof: `PAUSE_RED`

**Active authorized milestone:** v0.2.1 — static-world fidelity redesign

## v0.2 result

RivalSim v0.2 is frozen at final published `origin/main` boundary:

`2c5d11899eaaad6a963a370fcc3813202b6fa714`

The complete B3 static-world path reached 1,350,748.16 aggregate simulated game-seconds/s, but the required RocketSim parity gate failed with 85 scenario/horizon records containing hard mismatches and 617 numeric tolerance failures. Correctness therefore overrode the green performance result.

Published v0.2 evidence remains immutable under:

- `results/v0.2/`;
- `docs/V0_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2.md`.

## Active v0.2.1 boundary

Start at:

`handoff/v0.2.1/CODEX_START_PROMPT.md`

v0.2.1 is not a feature expansion. It may only redesign the demonstrated static wheel/contact solver approximations until the exact frozen 35-scenario parity corpus passes with unchanged tolerances, then remeasure the corrected B3 throughput.

Success requires:

- zero hard mismatches;
- zero numeric tolerance failures;
- v0.1 regression still 27/27 passing;
- deterministic finite contact-rich stress;
- GPU-resident hot loop;
- corrected B3 >=100,000 aggregate simulated game-seconds/s.

## Frozen prior version

**v0.1.0 — GPU contact-free physics proof: PASS**

Frozen result boundary:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

Published v0.1 evidence under `results/v0.1/` remains immutable.

## Hard stop

Do not begin v0.3 in v0.2.1. Still excluded:

- ball-world collision;
- car-ball collision;
- car-car collision;
- bumps/demolitions;
- boost pads;
- goals/scoring/game rules;
- RLGym observations/rewards/PPO;
- Rival policy inference.
