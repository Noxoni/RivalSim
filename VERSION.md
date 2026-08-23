# RivalSim Version Boundary

**Current active milestone:** v0.2 — arena + ground-contact proof authorized

## Frozen completed version

**v0.1.0 — GPU physics proof: PASS**

Frozen result boundary:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

Published v0.1 evidence under `results/v0.1/` is immutable for v0.2 work.

Measured v0.1 headline:

- 131,072 best stable contact-free worlds;
- 40,919,361.97 aggregate simulated game-seconds/s;
- 3,678.02x best GPU vs same-equation CPU;
- 27/27 RocketSim parity scenarios passed across 1/4/8/30/60/120 ticks.

## v0.2 objective

Determine how much GPU headroom remains after adding the real DFH/Stadium_P static collision geometry and RocketSim-derived surface vehicle physics:

- shared GPU stadium triangle mesh/BVH;
- four wheel/suspension queries per car;
- suspension and wheel impulses;
- ground throttle/brake/coast/steering/powerslide;
- car chassis vs static-world contact;
- floor/ramp/wall/ceiling behavior;
- contact-rich RocketSim parity and throughput evidence.

The active package is:

`handoff/v0.2/CODEX_START_PROMPT.md`

## Hard boundary

Do not implement v0.3 dynamic contacts during v0.2.

Still excluded:

- ball-world collision;
- car-ball collision;
- car-car collision;
- boost pads;
- goals/scoring/demos/respawns;
- RLGym/observations/rewards/PPO;
- Rival policy inference.

## Asset rule

Extracted/repacked Rocket League collision mesh assets remain local/ignored and must not be committed to the public repository. Commit loader code, provenance, hashes, statistics and reproduction instructions only.

## v0.2 verdict

The implementation run must end with one evidence-backed classification:

- `PASS_GREEN` — required parity passes and full static-world B3 >=100,000 aggregate sim-s/s;
- `PASS_YELLOW` — required parity passes and B3 >=20,000 but <100,000 sim-s/s with no architectural dead end;
- `PAUSE_RED` — fundamental fidelity/architecture failure or B3 <20,000 sim-s/s without a clear path.

Regardless of verdict, stop for review before v0.3.
