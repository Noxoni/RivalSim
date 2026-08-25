# RivalSim

RivalSim is an experimental **GPU-native Rocket League 1v1 transition engine** intended to accelerate training for [Noxoni/Rival](https://github.com/Noxoni/Rival).

The project is deliberately narrower than RocketSim. The target is standard Soccar 1v1 only:

- two cars;
- one ball;
- standard DFH/Stadium_P arena;
- standard static-world boost-pad pickup/cooldown state;
- fixed 120 Hz physics;
- no rendering in the training benchmark path.

## Current boundary — v0.4 complete, `PASS_GREEN`

RivalSim v0.4 completes the bounded standard-Soccar 1v1 game-transition milestone. The public
`CompleteWorldSim` composes the accepted v0.3 two-Octane/one-ball physics with GPU-resident state
for all 34 boost pads, goals and score attribution, five deterministic standard kickoff layouts,
demolition disable/timing, four source-valid respawn locations per team, world/episode clocks,
raw lifecycle events, and deterministic full-world reset.

Lifecycle choices are explicit state. Kickoff selectors advance modulo five and respawn selectors
advance modulo four; no host-global RNG, pointer value, allocator layout, case ID, or expected
output participates in the runtime. Car membership does not change during kickoff, demolition,
or respawn, so the source-proven per-world v0.3 car visitation order is preserved.

The content-addressed native lifecycle authority passes:

- every one of 34 pads for both cars: **68 / 68 pickup cases**;
- source float32 recharge boundaries: **1,201 ticks** for large pads and **480 ticks** for small;
- both visitation-order branches for pad contention;
- **6 / 6 goal-boundary cases** and both scoring directions;
- **5 / 5** standard two-car kickoff layouts;
- both teams at all four respawn locations: **8 / 8 poses**;
- exact demolition timer and respawn at **tick 360**;
- deterministic 64-world, 400-tick mixed lifecycle/reset stress with zero timed transfers.

The inherited v0.3 Phase A/B/C/D gates remain 31,216/31,216, 8,192/8,192,
8,192/8,192 against both branches, and 512/512 across both branches. The v0.2.2 static corpus
remains 39,236/39,236, v0.1 remains 27/27, both 4,608-ray backends pass, and the configured
repository suite is 70/70 passing. Published v0.1 through v0.3 evidence is byte-for-byte
unchanged.

The complete v0.4 path reaches **191,748.10 aggregate simulated game-seconds/s**
(23.01 million world ticks/s) at 131,072 worlds with **0.856% CV**, retaining 97.52% of v0.3.
The reset-heavy path reaches **225,005.06 sim-s/s** and **3.375 million reset transitions/s**
with **0.723% CV**. Both timed paths record zero host/device transfers.

RocketSim does not define a training episode terminal/truncation policy. v0.4 therefore exports
policy-neutral raw lifecycle state and keeps `terminated=truncated=0`; v0.5 policy was not begun.

### Explicitly excluded

RivalSim v0.4 does **not** implement RLGym observations, rewards, training-specific action
parsing, rollout buffers, tensor interop, PyTorch policy inference, GAE/PPO, Rival training,
arbitrary body counts, other game modes, rendering, or a generic Bullet API.

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

The v0.4 package classifies the complete game-transition path as:

- **PASS_GREEN:** every lifecycle/fidelity/regression gate passes and throughput is >=100,000
  sim-s/s;
- **PAUSE_PERF:** local parity passes but throughput is <100,000 sim-s/s;
- **PAUSE_FIDELITY:** any required local parity failure remains.

## Published v0.4 authority and result

The current result package is:

- `docs/V0_4_RESULTS.md`;
- `docs/REPRODUCING_V0_4.md`;
- `docs/V0_4_AUTHORITY.md`;
- `results/v0.4/boost_pads.json`;
- `results/v0.4/goals_kickoff.json`;
- `results/v0.4/demolition_respawn.json`;
- `results/v0.4/match_lifecycle.json`;
- `results/v0.4/oracle_data.json`;
- `results/v0.4/rules_source.json`;
- `results/v0.4/regression.json`;
- `results/v0.4/benchmark.json`;
- `results/v0.4/manifest.json`.

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
