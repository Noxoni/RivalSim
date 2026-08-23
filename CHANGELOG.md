# Changelog

## v0.2.1 — static-world fidelity redesign handoff (2026-08-23)

- Activated a bounded correctness redesign after v0.2 reached 1,350,748.16 aggregate simulated game-seconds/s but failed RocketSim static-world parity.
- Froze all published v0.1/v0.2 evidence and the existing 35-scenario × eight-horizon v0.2 parity corpus/tolerances.
- Required a divergence index ranked by earliest hard/numeric failure rather than tuning against long-horizon positional drift.
- Authorized a read-only diagnostic RocketSim build/wrapper from the exact pinned source so wheel friction impulses, suspension values, manifold state, solver ordering and post-solve velocities can be compared tick-by-tick.
- Preserved the already-passing `.cmf` geometry/query layer, normal Warp BVH, cuBQL suspension-ray backend and v0.1 contact-free mechanics unless evidence proves one is causal.
- Focused changes on Bullet/RocketSim-equivalent wheel friction, suspension/contact ordering, chassis manifold/contact solving, friction/restitution, penetration correction, persistence/warmstarting and surface-transition behavior.
- Required representative steering, powerslide, wall-transition and chassis-impact cases to pass before the full frozen corpus is rerun.
- Required full v0.2.1 parity to reach zero hard mismatches and zero numeric failures with unchanged tolerances before final performance benchmarking.
- Set corrected B3 success floor at 100,000 aggregate simulated game-seconds/s after parity passes.
- Explicitly kept ball/world, car/ball, car/car, boost pads, game rules, RLGym/PPO and Rival integration out of scope until a separate v0.3 authorization.

## v0.2.0 — arena + ground-contact proof implemented (2026-08-23)

- Added a strict little-endian RocketSim `.cmf` parser and deterministic Soccar loader with
  structural validation, SHA-256 custody, RocketSim internal hashes, bounds, and combined
  4,468-vertex / 8,020-triangle metadata. Extracted assets remain external and untracked.
- Added one shared normal Warp mesh for chassis AABB queries and a separately measured cuBQL
  mesh over the same geometry for suspension rays. Both passed the 4,608-ray independent CPU
  query corpus; cuBQL delivered the better B1 throughput.
- Added four explicit Octane-compatible wheel states per car and RocketSim-derived suspension,
  throttle, reverse, coast, brake, steering, boost-ground interaction, handbrake, powerslide,
  friction, sticky-force, and extra-pushback behavior.
- Added conservative OBB bounds, Warp mesh triangle candidate enumeration, 13-axis
  triangle-vs-OBB SAT, up to four chassis contacts, and bounded impulse/friction/restitution,
  positional-correction, and angular-response handling.
- Added a deterministic device-resident action tape, contact-rich state generator, and B0/B1/
  B2/B3 CUDA-graph benchmark decomposition with explicit variance, transfer, utilization,
  VRAM, candidate, contact, penetration, and NaN/error accounting.
- Measured a best stable B3 result of 1,350,748.16 aggregate simulated game-seconds/s at
  262,144 worlds, with 0.998% CV and zero timed H2D/D2H traffic. All 44 benchmark points were
  stable below 5% CV.
- Added a measurement-first RocketSim parity protocol, then froze the tolerance table before
  a clean 35-scenario gate run across eight horizons. The gate recorded 85 records with hard
  mismatches and 617 numeric failures; tolerances were not widened to conceal the divergence.
- Added two identical-hash 64-world, 2,400-tick stress runs with finite state and no hot-loop
  transfer.
- Classified v0.2 as `PAUSE_RED`: the standalone performance threshold is green, but required
  RocketSim transfer fidelity fails. Stopped without beginning v0.3.

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
