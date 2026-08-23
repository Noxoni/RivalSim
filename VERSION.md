# RivalSim Version Boundary

**Current completed milestone:** v0.2.0 — arena + ground-contact proof: `PAUSE_RED`

## v0.2 result

RivalSim v0.2 was implemented and measured from authority commit
`7a6a6913fad6ceedd92d1170b373a0978edb05b6`. The implementation checkpoint is
`f2363104a56a358276682e16110d16f37e8d0539`; the final evidence commit is recorded in
`results/v0.2/manifest.json` and on `origin/main`.

The complete B3 static-world path reached 1,350,748.16 aggregate simulated game-seconds/s at
262,144 worlds with 0.998% CV, but the required RocketSim parity gate failed. The combined
verdict is therefore `PAUSE_RED`, even though the standalone performance classification is
green-threshold.

Published v0.2 evidence:

- `results/v0.2/benchmark.json`;
- `results/v0.2/parity.json`;
- `results/v0.2/manifest.json`;
- `docs/V0_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2.md`.

## Frozen prior version

**v0.1.0 — GPU contact-free physics proof: PASS**

Frozen result boundary:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

Published v0.1 evidence under `results/v0.1/` remains immutable and was not rewritten for
v0.2.

## Hard stop

Do not begin v0.3 without new authority. Still excluded:

- ball-world collision;
- car-ball collision;
- car-car collision;
- bumps/demolitions;
- boost pads;
- goals, scoring, resets, and other game rules;
- RLGym observations/rewards/PPO;
- Rival policy inference.

The next smallest technical question is a bounded redesign of wheel friction/steering and the
static contact solver against the existing v0.2 parity corpus. That is a recommendation only,
not authorization to implement v0.3.
