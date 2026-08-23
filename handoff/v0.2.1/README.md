# RivalSim v0.2.1 — Static-World Fidelity Redesign

RivalSim v0.2 proved that GPU static-world simulation is fast enough, but failed the required RocketSim fidelity gate.

This package is a bounded **correctness redesign**, not a new feature milestone.

## Frozen starting evidence

Start from final v0.2 `origin/main`:

`2c5d11899eaaad6a963a370fcc3813202b6fa714`

The v0.2 complete B3 static-world path reached **1,350,748.16 aggregate simulated game-seconds/s** and therefore independently exceeded the green throughput threshold. The failure was fidelity: 85 scenario/horizon records had hard mismatches and 617 numeric checks exceeded frozen tolerances.

Do not broaden the existing tolerances and do not hide failures.

## Mission

Answer this question:

> Can the existing fast GPU arena/wheel/chassis architecture be made sufficiently RocketSim/Bullet-equivalent for static-world driving and surface contact without sacrificing the enormous GPU performance advantage?

The primary work is to replace the demonstrated approximation errors in solver behavior, not to add more game features.

## Preserve what already works

Do not redesign these unless evidence proves they are responsible for a parity failure:

- exact 16-file Soccar `.cmf` geometry custody and parser;
- immutable shared arena geometry;
- normal Warp BVH for chassis candidate queries;
- cuBQL suspension-ray backend;
- wheel ray origins/directions and hit geometry where query parity already passed;
- v0.1 contact-free mechanics;
- GPU-resident state/action tape;
- B0/B1/B2/B3 benchmark decomposition;
- frozen v0.1 and v0.2 evidence.

The v0.2 geometry-query gate was already essentially exact and the cuBQL path reached billions of rays/s. The fidelity problem is downstream solver semantics.

## Explicit scope

v0.2.1 may change only what is necessary to achieve static-world parity:

- wheel/suspension force calculation and application;
- throttle/brake/coast/steering/powerslide force ordering;
- lateral/longitudinal friction impulses;
- chassis-world contact generation and manifold representation;
- normal/friction/restitution impulse solving;
- penetration correction;
- warmstarting/contact persistence if proven necessary;
- ordering between car controls, wheel solver, static contacts, rigid-body integration, and post-tick state.

Do **not** implement:

- ball-world contact;
- car-ball contact;
- car-car contact;
- bumps/demolitions;
- boost pads;
- scoring/game rules;
- RLGym/PPO;
- Rival policy integration;
- rendering/UI;
- new game modes.

## Primary authority

Use the exact pinned RocketSim source used by the oracle:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

The Python binding is not sufficient for every internal diagnostic. You are authorized to build a small local diagnostic/oracle executable or instrumented local RocketSim build from that exact source when needed to expose per-tick internal values. Keep upstream source ignored, record every local diagnostic patch/diff and SHA, and do not change the actual physics semantics of the reference.

Focus especially on:

- `Car::_PreTickUpdate`, `_UpdateWheels`, `_PostTickUpdate`, `_FinishPhysicsTick`;
- `btVehicleRL::updateVehicleFirst`, `rayCast`, `calcFrictionImpulses`, `updateSuspension`, `applyFrictionImpulses`, `updateVehicleSecond`;
- Bullet `resolveSingleBilateral` / `resolveSingleCollision` behavior;
- Bullet manifold/contact generation and sequential impulse ordering used by RocketSim;
- static-world friction/restitution/internal-edge behavior;
- any solver-info constants, iteration counts, ERP/penetration settings, split impulse, warmstart or persistent-manifold behavior that materially affects the frozen parity cases.

## Required method

1. Freeze the existing 35-scenario v0.2 corpus and frozen tolerances.
2. Rank scenarios by **earliest tick of divergence**, with hard state/sign failures first.
3. Choose a minimal representative set spanning:
   - steering/ground motion;
   - powerslide;
   - wall/ramp transition;
   - chassis impact/landing.
4. Record per-tick RocketSim vs RivalSim internal diagnostic traces around the first divergent tick.
5. Identify the first causal mismatch, not merely the later positional error.
6. Replace only the demonstrated approximation with a more Bullet/RocketSim-equivalent implementation.
7. Re-run the representative cases after every coherent solver correction.
8. Once representative cases pass, re-run the full frozen 35-scenario corpus.
9. Only after full parity passes, rerun the decomposed GPU benchmark.

Do not tune throughput while parity is failing.

## Success boundary

v0.2.1 succeeds only if:

- all frozen hard discrete/contact/direction checks pass across all 35 scenarios and eight horizons;
- all frozen numeric tolerances pass unchanged;
- v0.1 regression remains 27/27 passing;
- stress/determinism remains finite and deterministic;
- static-world hot loop remains GPU resident;
- final B3 throughput remains at least **100,000 aggregate simulated game-seconds/s**.

A slower-but-correct solver above 100k is preferable to the current 1.35M incorrect solver.

Do not begin v0.3 in this run even if v0.2.1 passes.
