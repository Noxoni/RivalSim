# RivalSim

RivalSim is an experimental **GPU-native Rocket League 1v1 transition engine** intended to accelerate training for [Noxoni/Rival](https://github.com/Noxoni/Rival).

The project is deliberately narrower than RocketSim. The target is standard Soccar 1v1 only:

- two cars;
- one ball;
- standard DFH/Stadium_P arena;
- standard static-world boost-pad pickup/cooldown state;
- fixed 120 Hz physics;
- no rendering in the training benchmark path.

## Current boundary — v0.3 complete, `PASS_GREEN`

RivalSim v0.3 completes the bounded standard-Soccar dynamic-contact milestone. It adds a fixed
two-Octane/one-ball integrated world with source-ordered ball/world, car/ball, car/car, wheel,
static-world, shared-island solver, split-impulse, and rigid-body writeback behavior. The runtime
remains GPU-resident and carries car `_PreTickUpdate` visitation order as internal per-world
lifecycle state established by construction or membership change.

All frozen native gates pass at ticks 1, 4, 8, and 12:

- Phase A ball/world: **31,216 / 31,216** cases;
- Phase B car/ball: **8,192 / 8,192** cases;
- Phase C car/car: **8,192 / 8,192** cases against both complete source-valid visitation-order
  branches;
- Phase D integrated: **512 / 512** cases across eight simultaneous-contact families and both
  complete source-valid branches.

Every phase has zero blocking hard mismatches and zero numeric tolerance failures. Native
authority is isolated, content-addressed, caches every tick 1–12, and has no live fallback after
freeze. Phase C/D relational comparison accepts one complete labeled native trajectory and never
mixes metrics or selects a branch from expected outputs.

The complete dynamic path reaches **196,614.39 aggregate simulated game-seconds/s**
(23.59 million world ticks/s) at 131,072 worlds with **1.313% CV**. This is 1.97× the v0.3
100,000 sim-s/s viability floor and 38.41% of the narrower v0.2.2 static-only reference. The
timed hot loop records zero host/device transfers. Two independent 64-world, 2,400-tick stress
runs are finite, bounded, and full-state bit-identical.

The complete v0.2.2 static corpus remains 39,236/39,236 passing, the v0.1 live RocketSim corpus
remains 27/27 passing, both ray backends remain green, and the repository suite is 63/63 passing.
Published v0.1 through v0.2.2 evidence remains byte-for-byte unchanged.

### Explicitly excluded

RivalSim v0.3 does **not** implement demolition removal/disable/respawn, goals/scoring, kickoff or
match reset rules, RLGym observations/rewards/PPO, Rival policy inference, arbitrary body counts,
other game modes, or a generic Bullet API. Those remain v0.4+ work and were not begun.

## Architecture

RivalSim stays **GPU-resident and batched**. Do not build a Python object graph per environment and do not round-trip world state through CPU every tick.

The v0.2 arena is a single shared GPU asset. Extracted Rocket League collision meshes are not committed to this public repository; only loader code, provenance, hashes, statistics and reproduction instructions belong in Git.

NVIDIA Warp remains the primary implementation layer until profiling proves a reason to replace a measured hotspot with native CUDA/C++.

## Performance references

Current full Rival CPU RocketSim/RLGym reference:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

RivalSim remains a partial simulator, so this is a system reference rather than an apples-to-apples comparison.

The v0.3 package classifies the complete dynamic-contact path as:

- **PASS_GREEN:** every fidelity/regression gate passes and throughput is >=100,000 sim-s/s;
- **PAUSE_PERF:** local parity passes but throughput is <100,000 sim-s/s;
- **PAUSE_FIDELITY:** any required local parity failure remains.

## Published v0.3 authority and result

The current result package is:

- `docs/V0_3_RESULTS.md`;
- `docs/REPRODUCING_V0_3.md`;
- `docs/V0_3_ORACLE_CACHE.md`;
- `results/v0.3/ball_world.json`;
- `results/v0.3/car_ball.json`;
- `results/v0.3/car_car.json`;
- `results/v0.3/integrated.json`;
- `results/v0.3/oracle_data.json`;
- `results/v0.3/source_port.json`;
- `results/v0.3/regression.json`;
- `results/v0.3/benchmark.json`;
- `results/v0.3/manifest.json`.

## Published v0.2.2 authority and result

The frozen v0.2.2 result package is:

- `docs/V0_2_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2_2.md`;
- `docs/V0_2_2_ORACLE_CACHE.md`;
- `results/v0.2.2/oracle_data.json`;
- `results/v0.2.2/source_port.json`;
- `results/v0.2.2/parity.json`;
- `results/v0.2.2/regression.json`;
- `results/v0.2.2/benchmark.json`;
- `results/v0.2.2/manifest.json`.

## Published v0.2.1 authority and result

The active handoff is preserved at `handoff/v0.2.1/CODEX_START_PROMPT.md`. The immediate
2026-08-23 user steering adjustment governs the final 1/4/8/12-tick validation boundary and is
recorded in the v0.2.1 evidence and reproduction guide.

The result package is:

- `docs/V0_2_1_RESULTS.md`;
- `docs/REPRODUCING_V0_2_1.md`;
- `results/v0.2.1/divergence_index.json`;
- `results/v0.2.1/parity.json`;
- `results/v0.2.1/coverage.json`;
- `results/v0.2.1/benchmark.json`;
- `results/v0.2.1/manifest.json`.

## Published v0.2 authority

The root prompt and completed bounded package are preserved at:

`handoff/v0.2/CODEX_START_PROMPT.md`

Package contents:

- `handoff/v0.2/README.md`
- `handoff/v0.2/V0_2_SPEC.md`
- `handoff/v0.2/BENCHMARK_AND_PARITY.md`
- `handoff/v0.2/CODEX_START_PROMPT.md`
- `handoff/v0.2/PACKAGE_MANIFEST.md`

The result package is:

- `docs/V0_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2.md`;
- `results/v0.2/benchmark.json`;
- `results/v0.2/parity.json`;
- `results/v0.2/manifest.json`.

## Relationship to RocketSim

RocketSim remains the primary CPU physics oracle until RivalSim earns replacement through performance **and transfer fidelity**.

RivalSim is a training transition engine, not a replacement Rocket League client. Work is intended for offline bot training and research, not cheating in online Rocket League.
