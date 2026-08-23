# RivalSim

RivalSim is an experimental **GPU-native Rocket League 1v1 transition engine** intended to accelerate training for [Noxoni/Rival](https://github.com/Noxoni/Rival).

The project is deliberately narrower than RocketSim. The target is standard Soccar 1v1 only:

- two cars;
- one ball;
- standard DFH/Stadium_P arena;
- standard boost pads later;
- fixed 120 Hz physics;
- no rendering in the training benchmark path.

## Current milestone — v0.2 arena + ground-contact proof

The v0.1 contact-free GPU proof passed decisively and is frozen at:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

Measured v0.1 best stable point:

- **131,072 worlds**;
- **40,919,361.97 aggregate simulated game-seconds/s**;
- **3,678.02x** best GPU vs same-equation CPU throughput;
- **27/27** RocketSim parity scenarios passed at 1/4/8/30/60/120 ticks;
- zero timed H2D/D2H traffic in the published GPU hot loop.

That result is intentionally contact-free and is not a full-simulator speedup claim. Frozen evidence remains under `results/v0.1/` and `docs/V0_1_RESULTS.md`.

v0.2 now asks:

> How much of that GPU headroom survives when RivalSim adds the real DFH static collision geometry, eight suspension rays per 1v1 world per physics tick, RocketSim-derived wheel/ground forces, and chassis-vs-arena contact?

### v0.2 scope

- one immutable DFH/Stadium_P triangle mesh shared by all GPU worlds;
- Warp mesh/BVH acceleration, including measured cuBQL ray-backend evaluation where supported;
- four wheel raycasts per car;
- RocketSim-compatible suspension;
- wheel friction, throttle, brake, coast, steering and powerslide;
- car hitbox vs static-world contact;
- floor/ramp/wall/ceiling movement and landings;
- contact-rich CPU RocketSim parity;
- decomposed GPU benchmarks for mesh rays, wheel mechanics and complete static-world physics.

### Still excluded

v0.2 does **not** implement:

- ball-world collision;
- car-ball collision;
- car-car collision;
- bumps/demolitions;
- boost pads;
- scoring/game reset;
- RLGym observations/rewards/PPO;
- Rival policy inference.

Those remain later milestones so static-world cost/fidelity can be measured cleanly.

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

## Start here

Give Codex:

`Read CODEX_START_PROMPT.md and execute it completely.`

The active root prompt routes to:

`handoff/v0.2/CODEX_START_PROMPT.md`

Package contents:

- `handoff/v0.2/README.md`
- `handoff/v0.2/V0_2_SPEC.md`
- `handoff/v0.2/BENCHMARK_AND_PARITY.md`
- `handoff/v0.2/CODEX_START_PROMPT.md`
- `handoff/v0.2/PACKAGE_MANIFEST.md`

## Relationship to RocketSim

RocketSim remains the primary CPU physics oracle until RivalSim earns replacement through performance **and transfer fidelity**.

RivalSim is a training transition engine, not a replacement Rocket League client. Work is intended for offline bot training and research, not cheating in online Rocket League.
