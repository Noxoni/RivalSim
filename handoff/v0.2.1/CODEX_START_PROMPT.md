# Codex Start Prompt — RivalSim v0.2.1 Static-World Fidelity Redesign

Work directly in `Noxoni/RivalSim` from the current `origin/main` and complete this bounded fidelity milestone.

Read first:

- `handoff/v0.2.1/README.md`
- `handoff/v0.2.1/FIDELITY_PLAN.md`
- `handoff/v0.2.1/GATES.md`
- `docs/V0_2_RESULTS.md`
- `results/v0.2/parity.json`
- `results/v0.2/benchmark.json`
- `docs/ARCHITECTURE.md`
- `docs/SOURCE_REFERENCES.md`

## Mission

Fix the demonstrated static-world parity failure without adding v0.3 features.

The performance architecture already proved sufficient: the complete B3 path reached 1,350,748.16 aggregate simulated game-seconds/s. The problem is fidelity to RocketSim/Bullet wheel/contact behavior. Correctness now has priority over speed.

## Hard rules

- Do not widen the published v0.2 tolerances.
- Do not delete or rewrite published v0.1/v0.2 evidence.
- Do not change arena geometry or query backends unless a diagnostic proves they are causal.
- Do not optimize throughput while the frozen parity corpus still fails.
- Do not implement ball-world, car-ball, car-car, boost-pad, scoring, RLGym, PPO, or Rival-policy features.
- Do not begin v0.3 even if this milestone succeeds.

## Step 1 — rank the existing failures

Parse the frozen 35-scenario × 8-horizon v0.2 corpus and produce a machine-readable divergence index.

For every scenario record:

- first horizon/tick with any hard mismatch;
- first horizon/tick with any numeric tolerance failure;
- failing fields;
- magnitude/direction;
- whether the first failure originates in wheel contact, chassis contact, force/impulse, orientation, or state ordering.

Select a small representative diagnostic set that includes the earliest failures in:

1. normal steering/ground acceleration;
2. powerslide/high lateral slip;
3. wall/ramp/surface transition;
4. chassis landing/impact.

## Step 2 — build an internal RocketSim diagnostic oracle

The Python binding does not expose enough solver internals. Build a small local diagnostic executable or equivalent wrapper against the exact pinned source:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

Do not modify upstream semantics. Instrument only for readout.

For the representative cases, capture per physics tick as available:

### Car rigid body
- transform/basis;
- linear and angular velocity before/after each major stage;
- inverse mass and inertia tensor relevant to impulse calculation;
- applied impulse/torque deltas.

### Wheels
- world hard point, direction and axle;
- ray hit/miss, face, point and normal;
- suspension length;
- suspension relative velocity;
- clipped inverse contact dot;
- extra pushback;
- engine force;
- brake;
- steering angle/value;
- suspension force;
- computed forward and side impulses;
- skid/friction clipping values;
- final applied wheel impulse per wheel.

### Chassis/world contact
- manifold/contact count;
- contact point;
- normal;
- penetration/separation distance;
- combined friction/restitution;
- persistent-manifold lifetime/id if accessible;
- normal impulse;
- lateral friction impulses;
- warmstart impulse if used;
- solver iteration/order information relevant to the contact.

### Tick ordering
Record exact boundaries around:

- `updateVehicleFirst`;
- `Car::_UpdateWheels` and other pre-tick mechanics;
- `updateVehicleSecond`;
- Bullet `stepSimulation` / solver;
- RocketSim post-tick and velocity limiting.

Commit only compact diagnostic summaries and the instrumentation patch/diff or wrapper source required to reproduce them. Large raw traces may remain ignored but must be hashed and documented.

## Step 3 — locate the first causal divergence

For each representative scenario, compare RivalSim and RocketSim at every tick until the first meaningful mismatch.

Do not attempt to fix a 600-tick position error directly. Identify the first differing ray/contact/impulse/state transition that causes the later error.

Classify each root cause, for example:

- force applied at wrong tick/order;
- wheel friction equation differs;
- missing effective-mass term;
- incorrect friction clamping;
- chassis contact normal/manifold differs;
- contact solver lacks persistence/warmstart;
- restitution/penetration correction differs;
- surface transition loses wheel contact;
- world contact inferred from different criteria;
- rigid-body angular impulse response differs.

## Step 4 — implement the smallest Bullet-equivalent corrections

Use RocketSim's bundled Bullet 3.24 implementation as the behavioral reference. Port only the subset required for static two-car-to-static-arena behavior.

Prefer reproducing source behavior/formulas/order over parameter fitting.

Important likely areas to inspect include:

- `btVehicleRL::rayCast`;
- `calcFrictionImpulses`;
- `updateSuspension`;
- `applyFrictionImpulses`;
- `resolveSingleBilateral`;
- `resolveSingleCollision`;
- Bullet's contact constraint setup/solve path;
- sequential impulse effective mass;
- friction direction and limits;
- solver iteration count and ordering;
- contact manifold persistence/warmstarting;
- penetration/ERP/split-impulse behavior;
- internal edge contact-normal adjustment.

If the current four-contact OBB-triangle solver is fundamentally incompatible with the reference semantics, it may be replaced with a more Bullet-equivalent static-contact representation. Keep it specialized for Soccar/static mesh and GPU batching; do not port all of Bullet.

## Step 5 — parity before performance

After each coherent fix, re-run the representative cases.

Once they pass, run the complete existing frozen corpus with the exact previously published tolerances:

- position 10.0 uu;
- linear velocity 25.0 uu/s;
- orientation 0.025 rad;
- angular velocity 0.1 rad/s;
- boost 0.01;
- handbrake 0.0001;
- world-contact normal 0.05 rad;
- all discrete/contact/direction mismatches remain hard failures.

Required result: **0 hard mismatches and 0 numeric tolerance failures across every clean-gate record.**

If full parity still fails, do not benchmark a supposed final implementation. Continue diagnosis within the bounded static-world scope or stop with the remaining first-cause blockers clearly identified.

## Step 6 — regression/stress

When parity passes:

- rerun v0.1 27-scenario regression;
- run at least two identical 2,400-tick contact-rich stress passes and require full-state deterministic equality;
- require finite state/no NaNs;
- verify no timed hot-loop H2D/D2H traffic.

## Step 7 — performance after correctness

Only after parity passes, rerun B0/B1/B2/B3 decomposition at meaningful batch sizes and locate the new B3 optimum.

Report:

- worlds;
- world ticks/s;
- aggregate simulated game-seconds/s;
- slowdown versus v0.2 B3 at equal batch where possible;
- GPU utilization;
- VRAM;
- CPU utilization;
- ray/candidate/contact rates;
- host/device traffic;
- repeated-run CV.

Success floor for the corrected complete B3 path: **>=100,000 aggregate simulated game-seconds/s**.

Do not sacrifice parity to recover speed.

## Required artifacts

Publish compact, versioned evidence without changing prior results:

- `results/v0.2.1/divergence_index.json`
- `results/v0.2.1/parity.json`
- `results/v0.2.1/benchmark.json` only if parity passes
- `results/v0.2.1/manifest.json`
- `docs/V0_2_1_RESULTS.md`
- `docs/REPRODUCING_V0_2_1.md`

Update README/VERSION/CHANGELOG only to accurately reflect the completed result.

## End report

Report:

- final pushed SHA(s) and remote readback;
- first causal divergence(s) discovered;
- exact reference instrumentation used;
- solver changes made and source locations they reproduce;
- representative-case before/after evidence;
- full frozen parity result;
- v0.1 regression;
- stress/determinism;
- corrected B3 throughput if parity passed;
- whether v0.2.1 passes;
- if not, the narrowest remaining static-world blocker.

Stop at this boundary. Do not begin dynamic-contact v0.3.
