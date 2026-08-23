# RivalSim v0.2 Implementation Specification

## Mission

Implement the first contact-rich RivalSim milestone without expanding into dynamic car/ball gameplay.

The goal is not to build a complete Rocket League simulator in one pass. The goal is to add the static stadium and vehicle-surface mechanics that dominate ordinary driving, wall movement, landing and recovery, then measure performance and fidelity before authorizing dynamic contacts.

## Frozen baseline

Do not regress or rewrite v0.1.

Required baseline commit:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

Before v0.2 work, reproduce the existing v0.1 tests and at least one representative v0.1 benchmark/parity smoke. Record any environment drift.

## Source hierarchy

For v0.2 mechanics use, in priority order:

1. the exact RocketSim source/version used by the live CPU oracle;
2. the exact collision assets consumed by that RocketSim environment;
3. RLBot/RLGym public values and map documentation as independent checks;
4. empirical Rocket League validation only where the above are insufficient.

Pinned v0.1 RocketSim source reference:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

Important RocketSim areas:

- `src/Sim/Car/Car.cpp`
- `src/Sim/btVehicleRL/btVehicleRL.cpp`
- `src/Sim/btVehicleRL/btVehicleRL.h`
- `src/Sim/Arena/Arena.cpp`
- `src/RLConst.h`
- modified Bullet contact/constraint helpers actually called by the above.

Do not reproduce public RLBot approximations when RocketSim uses a more specific equation. Public values are validation aids, not substitutes for the oracle implementation.

## Stadium asset custody

RocketSim expects arena collision data under `collision_meshes/` in `.cmf` files derived from `RLArenaCollisionDumper`.

The public RLBot DFH extraction documentation independently identifies four relevant DFH collision assets:

- `Goal_STD_Collision.pskx`
- `Field_STD_Collision_Corner.pskx`
- `Field_STD_Collision_SideTop.pskx`
- `Field_STD_Collision_SideBot.pskx`

Rules:

- Prefer the local `.cmf` assets already used by the RocketSim oracle if available.
- Record each source file name, size, SHA-256, mesh/hash metadata and extraction/dumper provenance.
- Parse the exact vertex/index data needed by RivalSim.
- Keep extracted game assets and generated GPU mesh blobs ignored locally.
- Do **not** commit raw or repackaged Rocket League collision mesh data to this public repository.
- Commit loader code, asset manifests/hashes/statistics and exact reproduction instructions only.
- If RocketSim's local collision assets cannot be located or validated, use `ZealanL/RLArenaCollisionDumper` or the RLBot documented extraction route rather than inventing stadium geometry.

## GPU arena representation

Build one canonical immutable DFH/Stadium_P triangle mesh on `cuda:0` and share it across all worlds.

First implementation:

- `warp.Mesh` points/indices resident on the GPU;
- normal Warp mesh BVH for general mesh queries;
- benchmark the Warp 1.16 cuBQL mesh ray backend (`bvh_constructor="cubql"`) for suspension raycasts where supported;
- if cuBQL is used for rays, retain a normal mesh/BVH for AABB or point queries required by chassis contact if necessary.

Do not build a custom BVH merely for theoretical performance. Only replace Warp's acceleration structure after profiling shows a material bottleneck and the replacement has parity tests.

Record:

- triangle count;
- vertex count;
- mesh bounds;
- source hashes;
- GPU bytes for geometry + acceleration structure where measurable;
- BVH construction time outside the simulation hot loop.

## Wheel state

Extend GPU state for each of four wheels per car with whatever RocketSim parity requires, including at minimum:

- chassis connection point;
- wheel direction and axle in world space;
- hard point;
- ray hit/contact point;
- contact normal;
- hit distance/fraction;
- suspension length;
- suspension relative velocity;
- clipped inverse contact-dot factor;
- extra pushback if required;
- suspension force;
- world/static contact flag;
- steer angle;
- engine force;
- brake force;
- longitudinal/side friction impulse intermediates as needed.

Keep arrays flattened/coalesced. Do not create Python wheel objects per world.

## Suspension raycast

Reproduce RocketSim `btVehicleRL::rayCast()` behavior, including:

- transformed wheel connection point;
- transformed wheel direction/axle;
- real ray length using suspension rest length, travel, wheel radius and RocketSim subtraction constant;
- closest static-world hit;
- contact point and normal;
- suspension length calculation and clamping;
- velocity at contact point;
- projected relative velocity;
- denominator/clipped inverse behavior;
- no-hit defaults;
- static-object extra-pushback behavior where required for parity.

The first dedicated microbenchmark should issue the equivalent of 4 rays × 2 cars × N worlds every physics tick against the single shared stadium mesh.

## Suspension force

Reproduce `btVehicleRL::updateSuspension()` rather than using a generic spring guess.

Match RocketSim source behavior for:

- rest-length delta;
- stiffness;
- compression vs relaxation damping;
- clipped contact factor;
- front/back suspension force scale;
- no downward suspension force;
- impulse application at contact offset;
- extra pushback.

## Ground driving and wheel friction

Port the behavior required from RocketSim `Car::_UpdateWheels()` and `btVehicleRL::calcFrictionImpulses()/applyFrictionImpulses()`.

Required behaviors:

- throttle deadzone/clamping;
- forward/reverse engine force vs forward speed;
- brake behavior;
- zero-throttle coast drag;
- boost forcing throttle where RocketSim does so;
- ground boost acceleration/consumption interaction;
- steering curve/steer angle;
- handbrake rise/fall state;
- powerslide friction modification;
- lateral wheel friction;
- longitudinal rolling/engine/brake impulse;
- contact-point force/impulse application;
- wheel contact count and `isOnGround` behavior;
- partial-wheel contact behavior used by air torque/auto-roll logic.

Use RLBot public values such as ~1410 uu/s non-boost max ground speed, braking/coast acceleration and published curvature only as sanity checks. The actual implementation should follow RocketSim's source-backed curves and constants.

## Car hitbox and inertia

Use the exact `CarConfig`/hitbox used by the RocketSim oracle for the v0.2 corpus. Do not assume public rest height alone defines the rigid body.

Reproduce:

- hitbox size;
- hitbox position offset relative to rigid-body center;
- mass;
- local inertia consistent with the RocketSim/Bullet box setup;
- friction/restitution constants used for car-world interaction;
- orientation/world transform convention established in v0.1.

The initial required vehicle is the standard Octane-compatible configuration used by the oracle. General multi-hitbox support is not required in v0.2.

## Chassis-world collision

Implement static-world contact for the oriented car box against the stadium triangles.

Preferred architecture:

1. compute conservative world-space AABB for the oriented hitbox;
2. use GPU mesh AABB/BVH query to enumerate only overlapping triangle candidates;
3. perform an exact-enough OBB-triangle narrow phase on the GPU;
4. generate contact normal, point and penetration depth;
5. apply a RocketSim/Bullet-compatible rigid-body impulse and penetration correction;
6. preserve angular response from off-center contact;
7. support more than one simultaneous contact where required for stable floor/wall/corner behavior.

Do not substitute an SDF-only collision model in v0.2. Surface normals/contact locations need to remain sufficiently faithful for wall transitions, landings, wavedash-style recovery and future reset mechanics.

If reproducing Bullet's manifold/solver exactly is impractical, implement the smallest solver that matches the measured RocketSim trajectories and document every intentional difference. Fidelity, not API similarity, is the gate.

## Tick ordering

Preserve the successful v0.1 ordering unless RocketSim source shows contact-specific ordering that must differ.

The v0.2 tick must respect RocketSim's key sequence:

- wheel transform/raycast/friction preparation before the world solve;
- car control/jump/air/boost updates in the correct pre-tick phase;
- suspension/wheel impulses at the equivalent source phase;
- rigid-body/static-world contact solve;
- post-tick state updates, caps and control history.

Commit an explicit tick-order diagram once the source path is resolved.

## Numerical mode

Use FP32 for the main training-intended path unless measurement proves another mode is needed.

Do not enable fast-math or approximate collision modes in the parity build without separately reporting them. If an optional faster mode is explored, benchmark it as a distinct configuration and never use its numbers to satisfy the fidelity gate.

## Host/device rule

The timed v0.2 physics loop must remain GPU resident.

Allowed host traffic outside timing:

- asset loading/build;
- initial state/control upload;
- sampled validation readback;
- aggregate counters/timing;
- final state readback;
- logs/results.

For realistic control variation during throughput tests, pre-generate an action tape on-device or upload it before timing and index it on GPU. Do not perform per-tick Python control uploads.

## v0.1 regression

All existing v0.1 contact-free tests must continue to pass.

The existing v0.1 contact-free path may remain selectable for direct cost decomposition. Report the difference between:

- v0.1 contact-free transition;
- mesh ray-query only;
- suspension/wheel path;
- full v0.2 car-world path.

## Deliverables

At minimum add:

- arena asset loader/manifest code;
- GPU mesh/BVH wrapper;
- suspension ray kernels;
- wheel/suspension state;
- ground driving/wheel friction kernels;
- chassis-world broad/narrow-phase contact implementation;
- RocketSim ground/contact oracle scenarios;
- microbenchmarks and full v0.2 benchmark;
- automated tests;
- `results/v0.2/benchmark.json`;
- `results/v0.2/parity.json`;
- `results/v0.2/manifest.json`;
- `docs/V0_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2.md`;
- updated source/licensing notices where needed.

Large raw profiler captures and proprietary/extracted mesh assets remain ignored locally; commit their hashes, sizes and reproduction metadata.

## Stopping boundary

Even on success, do **not** begin v0.3 dynamic ball/car contacts in the same run.

Stop after the v0.2 benchmark/parity verdict and push a clean evidence-backed boundary for review.
