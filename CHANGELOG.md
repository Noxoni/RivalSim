# Changelog

## v0.2 — arena + ground-contact proof handoff (2026-08-22)

- Authorized the next bounded milestone after v0.1's decisive GPU continuation pass.
- Preserved frozen v0.1 result boundary `1f7a36cc6165273fb658ba07a8458e8d8e60628a` and prohibited rewriting `results/v0.1/` evidence.
- Defined a three-gate v0.2 implementation: stadium mesh/query engine; wheels/suspension/ground driving; chassis-vs-static-world contact.
- Selected one shared immutable DFH/Stadium_P GPU triangle mesh rather than per-world geometry.
- Added explicit collision-asset custody rules: prefer the exact local RocketSim `.cmf` assets when available, otherwise use `RLArenaCollisionDumper` or the RLBot extraction path; extracted game assets remain ignored and are never committed to the public repo.
- Kept NVIDIA Warp as the primary backend and required measured comparison of normal Warp mesh BVH vs the Warp 1.16 cuBQL ray backend for suspension rays where supported.
- Required RocketSim-derived `btVehicleRL` wheel transforms, raycasts, suspension, friction, steering and handbrake behavior rather than a generic bicycle/turn-radius approximation.
- Defined GPU OBB-vs-triangle car-world contact using mesh AABB candidate queries plus a measured narrow-phase/impulse solver.
- Added decomposed B0/B1/B2/B3 benchmarks to isolate the cost of contact-free motion, stadium rays, wheel mechanics and complete static-world contact.
- Added contact-rich parity scenarios across floor, braking/acceleration, steering, powerslide, ramps, walls, ceiling, landings and chassis impacts through horizons up to 600 ticks.
- Added verdict classes: `PASS_GREEN` at >=100k full static-world sim-s/s with parity; `PASS_YELLOW` at >=20k with parity and no architectural dead end; otherwise `PAUSE_RED`.
- Explicitly excluded ball-world, car-ball, car-car, boost pads, game rules and training integration until a separate v0.3+ authorization.

## v0.1.0 — GPU physics proof implemented (2026-08-22)

- Added a flattened, GPU-resident two-car/one-ball world state and fused NVIDIA Warp 120 Hz
  contact-free transition kernel.
- Implemented source-backed gravity, rigid-body integration, caps, airborne throttle/boost,
  jump/sticky/hold/double-jump, dodge/flip, aerial torque/damping and free-ball behavior.
- Added the vectorized NumPy same-equation CPU reference and live `rocketsim==2.2.1`
  `GameMode.THE_VOID` oracle.
- Added 27 deterministic parity scenarios and horizon-specific tolerances selected only after
  a corrected measurement-only run. Same-equation, live RocketSim and axis/sign/state parity
  all passed at 1/4/8/30/60/120 ticks.
- Added automated allocation/reset/control/mechanics/stress/parity/evidence tests.
- Added an adaptive, repeated CPU/GPU benchmark with separate untimed telemetry and explicit
  transfer/variance/NaN accounting. The GPU hot path uses eight-tick CUDA graph blocks.
- Measured a best stable RTX 5090 result of 40,919,361.97 aggregate simulated game-seconds/s
  at 131,072 worlds, versus 11,125.38 sim-s/s for the best same-equation CPU point: 3,678.02x.
- Passed every v0.1 continuation condition. The 203,934.02x ratio to the 200.65 sim-s/s full
  RocketSim/RLGym system reference is recorded only as a non-apples-to-apples comparison.
- Added compact benchmark/parity evidence, resolved dependency/source custody, third-party
  notices and exact reproduction commands.
- Final validation: 20 tests passed; Ruff, `compileall`, JSON parsing and `git diff --check`
  passed. `pip check` retained a documented upstream RocketSim wheel-tag metadata warning;
  the extension imported and all live-oracle tests passed.
- Stopped at the v0.1 boundary; no arena, suspension, ground-contact or other v0.2 work began.

## v0.1 — GPU physics proof handoff

- Established RivalSim as a separate GPU-simulation research repository.
- Defined Soccar 1v1-only scope for the initial architecture.
- Selected NVIDIA Warp for the first GPU proof.
- Defined GPU-resident batched state and contact-free 120 Hz mechanics scope.
- Added RocketSim/RLBot/RLGym physics references and source hierarchy.
- Added CPU/GPU benchmark and RocketSim parity gates.
- Added staged roadmap through arena collision, dynamic contacts and tensor-native training integration.
- Added Codex implementation prompt.

This section records the pre-implementation handoff. The implemented result is preserved above
as v0.1.0 rather than rewriting the historical boundary.
