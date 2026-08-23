# RivalSim v0.2 Package Manifest

Active implementation package: `handoff/v0.2/`

Frozen source/result boundary used to prepare this package:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

## Package files

- `README.md` — milestone intent, scope and three internal gates.
- `V0_2_SPEC.md` — implementation requirements, source hierarchy, asset custody and physics design.
- `BENCHMARK_AND_PARITY.md` — B0/B1/B2/B3 performance decomposition, scenario corpus and verdict thresholds.
- `CODEX_START_PROMPT.md` — executable Codex handoff.
- `PACKAGE_MANIFEST.md` — this manifest.

## Evidence inherited from v0.1

Canonical frozen evidence remains:

- `results/v0.1/benchmark.json`
- `results/v0.1/parity.json`
- `results/v0.1/manifest.json`
- `docs/V0_1_RESULTS.md`
- `docs/REPRODUCING_V0_1.md`

Do not modify those files during v0.2.

## External references explicitly incorporated into the v0.2 design

- `ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`
- `ZealanL/RLArenaCollisionDumper`
- RLBot v5 map-mesh extraction documentation
- RLBot v5 useful game values
- RLGym game-values cheatsheet
- NVIDIA Warp 1.16 mesh/BVH/ray/AABB APIs

## Result boundary expected from Codex

A completed v0.2 run should publish new evidence only under `results/v0.2/`, document the result in `docs/V0_2_RESULTS.md`, update the active project status, and stop before v0.3.
