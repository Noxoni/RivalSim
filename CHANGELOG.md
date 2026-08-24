# Changelog

## v0.2.2 — static-world source-parity breadth gate (2026-08-24)

- Froze a deterministic 39,236-case Octane/Soccar authority corpus, hashed against the pinned
  RocketSim/binding revisions, installed extension, CMFs, generator source/config, seed, and
  authority settings. Cached 470,832 native frames with no live fallback in the GPU gate.
- Added complete chassis/wheel states for all 8,020 DFH triangles, all 23,176 shared directed
  edges, and 20 analytic-plane states, with generated coverage separated from actual paired
  target contact.
- Translated the bounded pinned Bullet operation path for box-versus-static-triangle
  GJK/Voronoi/EPA witnesses, persistent-manifold refresh/four-point reduction, internal-edge
  adjustment, contact rows, split impulse, and rigid-body integration.
- Ported RocketSim's exact wheel ray, suspension, friction coefficient/impulse, force/torque,
  and brake-force float32 order. Kept source-correct internal edges and added no hysteresis,
  edge/tie tolerance, face-specific rule, or behavioral fitting.
- Corrected `btGjkPairDetector` internal-valid versus callback-report control flow, preventing
  false EPA fallback for valid shallow witnesses outside the callback distance.
- Passed the cached 1,043-case representative gate and complete 39,236-case gate with **0 hard
  mismatch events, 0 numeric failures, and 0 failed cases** across 156,944 checkpoints.
- Preserved the v0.1 live RocketSim regression at 27/27 and passed 46/46 repository tests.
  Deterministic stress, both GPU query backends, and hot-loop residency are green.
- Measured corrected B3 at **511,886.15 aggregate simulated game-seconds/s** at 262,144 worlds,
  **0.0913% CV**, zero timed transfers, and `PASS_GREEN`.
- Added compact authority, source-trace, parity, regression, benchmark, and manifest evidence;
  retained large oracle/tracing artifacts locally under `.tools/v0.2.2/`.
- Stopped at the v0.2.2 boundary. Dynamic bodies, ball physics, car-car physics, game rules,
  training integration, and all v0.3 work remain unstarted.

## v0.2.1 — static-world fidelity redesign implemented (2026-08-23)

- Replaced v0.2's approximate wheel/contact response with source-backed RocketSim/Bullet
  operation ordering: two-phase wheel preparation/application, exact suspension/friction rows,
  shared solver prestate, ten-iteration velocity and split-impulse PGS, and deferred caps.
- Added Bullet-equivalent Octane box margins/inertia, CMF-local quantized-BVH ordering,
  triangle shared-edge metadata, SAT/GJK closest-feature handling, contact thresholds, manifold
  ordering, tangent/RHS construction, persistence fields, and callback-normal semantics.
- Corrected grounded boost acceleration, air-control suppression, auto-roll state, wheel state,
  and GPU-resident standard Soccar boost-pad pickup/cooldown behavior from pinned source.
- Built a source-only native diagnostic executable against unmodified RocketSimPython commit
  `2da51b1dac7b8127127613a5ff30e490bdd70dd8` and its pinned RocketSim/Bullet sources. It exposes
  pre/post rigid state, wheel rows, manifolds, solver impulses, triangle identities, and GJK
  features without entering the benchmark path.
- Adopted the immediate 2026-08-23 validation-policy adjustment: unchanged meaningful
  tolerances and hard semantic checks now gate authoritative local transitions at 1/4/8/12
  ticks (up to 100 ms); 30–600-tick synchronized open-loop identity is diagnostic only.
- Passed all 140 local checkpoints across the 35 existing scenarios with **0 hard mismatches**
  and **0 numeric failures**. Maximum errors were 0.0009785 uu position, 0.002752 uu/s linear
  velocity, 0.0003908 rad orientation, and 0.00005585 rad/s angular velocity.
- Preserved v0.1 at 27/27 passing and passed 38/38 repository tests. Two 64-world,
  2,400-tick stress runs produced the identical full-state SHA-256, finite/bounded state, and
  zero hot-loop H2D/D2H bytes.
- Measured corrected B3 at **822,480.77 aggregate simulated game-seconds/s** at 262,144 worlds,
  **0.403% CV**, zero timed transfers, and stable scaling. This is 60.89% of v0.2 throughput
  at the same batch and satisfies the v0.2.1 **`PASS_GREEN`** threshold.
- Added a bounded breadth prototype that audits shared-edge topology across all 8,020 DFH
  triangles and reports observed transition coverage without claiming exhaustiveness. The
  existing corpus exercises 2 mesh triangles; per-triangle authoritative generation remains
  a later, separately authorized breadth milestone.
- Preserved all published `results/v0.1/` and `results/v0.2/` bytes and stopped at the v0.2.1
  boundary. Dynamic ball-world, car-ball, and car-car contacts and training integration remain
  unstarted v0.3+ work.

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
