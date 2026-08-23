# RivalSim

RivalSim is an experimental **GPU-native Rocket League 1v1 transition engine** intended to accelerate training for [Noxoni/Rival](https://github.com/Noxoni/Rival).

The project is deliberately narrower than RocketSim. The target is standard Soccar 1v1 only:

- two cars;
- one ball;
- standard DFH/Stadium_P arena;
- standard boost pads later;
- fixed 120 Hz physics;
- no rendering in the training benchmark path.

## Current boundary — v0.2 measured, `PAUSE_RED`

RivalSim v0.2 implemented the bounded arena + ground-contact proof and stopped at its required
review gate. The complete static-world B3 path reached **1,350,748.16 aggregate simulated
game-seconds/s** at 262,144 worlds with **0.998% CV**, zero timed host/device transfer, and a
passing 2,400-tick determinism/stress check. This is a strong performance result for a partial
static-world transition engine.

The required RocketSim fidelity gate did not pass. Across 35 scenarios and eight horizons,
85 scenario/horizon records contained hard state/contact/sign mismatches and 617 numeric
checks exceeded the prospectively frozen tolerances. Correctness takes precedence over
throughput, so the v0.2 verdict is **`PAUSE_RED`** and v0.3 is not authorized.

Implemented v0.2 scope includes:

- the exact external Soccar `.cmf` set as one shared 4,468-vertex / 8,020-triangle GPU asset;
- independently checked CPU, normal Warp BVH, and cuBQL suspension-ray queries;
- four wheel/suspension rays per car and eight per 1v1 world per tick;
- RocketSim-derived suspension, throttle, brake, coast, steering, boost-ground interaction,
  handbrake, and powerslide preparation;
- oriented-box broadphase bounds, triangle-vs-OBB SAT narrow phase, and bounded static-contact
  impulse/friction/correction response;
- decomposed B0/B1/B2/B3 benchmarks, contact-rich parity, and deterministic stress evidence.

The frozen v0.1 result remains at `1f7a36cc6165273fb658ba07a8458e8d8e60628a`;
`results/v0.1/` was not modified. See `docs/V0_2_RESULTS.md`,
`docs/REPRODUCING_V0_2.md`, and `results/v0.2/` for the v0.2 evidence.

### Explicitly excluded

RivalSim v0.2 does **not** implement:

- ball-world collision;
- car-ball collision;
- car-car collision;
- bumps/demolitions;
- boost pads;
- scoring/game reset;
- RLGym observations/rewards/PPO;
- Rival policy inference.

Those remain later milestones. New authority and a static-world fidelity redesign/review are
required before any v0.3 work.

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

The v0.2 package classifies the complete static-world path as:

- **PASS_GREEN:** parity passes and >=100,000 aggregate sim-s/s;
- **PASS_YELLOW:** parity passes and 20,000–<100,000 sim-s/s with no architectural dead end;
- **PAUSE_RED:** fidelity/architecture failure or <20,000 sim-s/s without a clear optimization path.

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
