# RivalSim

RivalSim is an experimental **GPU-native Rocket League 1v1 transition engine** intended to accelerate training for [Noxoni/Rival](https://github.com/Noxoni/Rival).

The project is deliberately narrower than RocketSim. The target is standard Soccar 1v1 only:

- two cars;
- one ball;
- standard DFH/Stadium_P arena;
- standard static-world boost-pad pickup/cooldown state;
- fixed 120 Hz physics;
- no rendering in the training benchmark path.

## Current boundary — v0.2.2 complete, `PASS_GREEN`

RivalSim v0.2.2 completes the bounded static-world source-parity breadth redesign. The frozen
39,236-case Octane/Soccar corpus passes all **156,944 authoritative local-transition
checkpoints** at 1, 4, 8, and 12 ticks with **zero hard mismatches, zero numeric tolerance
failures, and zero failed cases**. Native RocketSim authority is content-addressed and cached;
the final GPU gate cannot launch a live fallback. The meaningful v0.2 tolerances were not
widened.

The complete corrected B3 path reaches **511,886.15 aggregate simulated game-seconds/s** at
262,144 worlds with **0.0913% CV**, stable scaling, and zero timed host/device transfer. Two
independent 64-world, 2,400-tick stress passes are finite, bounded, and full-state bit-identical.
The v0.1 live corpus remains 27/27 passing, and the repository suite is 46/46 passing. This
satisfies the v0.2.2 **`PASS_GREEN`** class: complete parity plus at least 500,000 sim-s/s.

The validation policy deliberately retires 30–600-tick synchronized open-loop identity as a
hard requirement. Floating-point contact trajectories become chaotic after local branch choices;
long behavior should ultimately be assessed closed-loop and by train-in-RivalSim transfer to
RocketSim/RLBot. Long open-loop trajectories may still diagnose a systematic defect visible in
the 1–12-tick window, but do not block this release.

Implemented v0.2.2 static-world scope includes:

- the exact external Soccar `.cmf` set as one shared 4,468-vertex / 8,020-triangle GPU asset;
- independently checked CPU, normal Warp BVH, and cuBQL suspension-ray queries;
- four wheel/suspension rays per car and eight per 1v1 world per tick;
- source-ordered two-phase wheel transforms, rays, suspension, bilateral/rolling friction,
  throttle, brake, coast, steering, powerslide, and grounded boost;
- direct pinned Bullet operation order for Octane box-versus-static-triangle GJK/Voronoi/EPA,
  persistent-manifold reduction, internal-edge adjustment, contact rows, ten-iteration velocity
  and split-impulse solving, rigid-body integration, writeback, and deferred caps;
- GPU-resident standard Soccar boost-pad pickup, lock, cooldown, and recharge state;
- decomposed B0/B1/B2/B3 benchmarks, contact-rich parity, and deterministic stress evidence.

The deterministic breadth corpus generates chassis and wheel states for all 8,020 triangles,
all 23,176 shared directed edges, and 20 analytic-plane cases. It reports generated states and
actual paired target contact separately: 7,752 unique triangles and 8,912 directed edges had
paired target contact. Occluded or adjacent-target cases are not mislabeled as target coverage.

Published v0.1, v0.2, and v0.2.1 evidence remains frozen. See `docs/V0_2_2_RESULTS.md`,
`docs/REPRODUCING_V0_2_2.md`, and `results/v0.2.2/` for the current evidence.

### Explicitly excluded

RivalSim v0.2.2 does **not** implement:

- ball-world collision;
- car-ball collision;
- car-car collision;
- bumps/demolitions;
- scoring/game reset;
- RLGym observations/rewards/PPO;
- Rival policy inference.

Those remain later milestones. v0.3 was not begun; new authority is required before dynamic
contacts or training integration.

## Architecture

RivalSim stays **GPU-resident and batched**. Do not build a Python object graph per environment and do not round-trip world state through CPU every tick.

The v0.2 arena is a single shared GPU asset. Extracted Rocket League collision meshes are not committed to this public repository; only loader code, provenance, hashes, statistics and reproduction instructions belong in Git.

NVIDIA Warp remains the primary implementation layer until profiling proves a reason to replace a measured hotspot with native CUDA/C++.

## Performance references

Current full Rival CPU RocketSim/RLGym reference:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

v0.2 remains a partial simulator, so this is a system reference rather than an apples-to-apples comparison.

The v0.2.2 package classifies the corrected complete static-world path as:

- **PASS_GREEN:** local parity passes and >=500,000 aggregate sim-s/s;
- **PASS:** local parity passes and 100,000–<500,000 sim-s/s;
- **PAUSE_PERF:** local parity passes but throughput is <100,000 sim-s/s;
- **PAUSE_FIDELITY:** any required local parity failure remains.

## Published v0.2.2 authority and result

The current result package is:

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
