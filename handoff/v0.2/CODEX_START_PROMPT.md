# Codex Start Prompt — RivalSim v0.2 Arena + Ground-Contact Proof

Work directly in `Noxoni/RivalSim` and implement RivalSim v0.2 completely.

This is a bounded implementation/measurement milestone, not another planning exercise. Implement, benchmark, parity-check, document, commit and push the result. Stop before v0.3 dynamic contacts.

## Mission

Answer with working code and measured evidence:

> How much of RivalSim v0.1's GPU throughput advantage survives after adding the real DFH/Stadium_P static collision geometry, four RocketSim-compatible wheel/suspension queries per car, authentic ground-driving forces, and car-vs-static-world contact?

## Mandatory starting point

Canonical repo:

`https://github.com/Noxoni/RivalSim`

Frozen successful v0.1 implementation/evidence boundary:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

Read before implementation:

- `README.md`
- `VERSION.md`
- `CHANGELOG.md`
- `docs/V0_1_RESULTS.md`
- `docs/REPRODUCING_V0_1.md`
- `docs/ARCHITECTURE.md`
- `docs/PHYSICS_ORACLES.md`
- `docs/SOURCE_REFERENCES.md`
- `docs/ROADMAP.md`
- `handoff/v0.2/README.md`
- `handoff/v0.2/V0_2_SPEC.md`
- `handoff/v0.2/BENCHMARK_AND_PARITY.md`

The handoff files govern v0.2 where older roadmap text is less specific.

## Preserve v0.1

Before changing the simulator:

1. fetch/pull `origin/main` cleanly;
2. record the exact starting SHA;
3. run the existing v0.1 test suite;
4. run a representative v0.1 parity smoke and GPU benchmark smoke;
5. confirm `results/v0.1/` remains unchanged.

Do not rewrite, regenerate or normalize published v0.1 evidence files.

If the local environment differs from the published v0.1 environment, document it rather than silently changing old numbers.

## Implementation strategy

Stay with NVIDIA Warp for v0.2 unless measured profiling proves it is the limiting architecture.

Do not start by writing a native CUDA BVH or porting Bullet wholesale.

Warp 1.16 already provides GPU-resident triangle `Mesh` acceleration with ray and AABB queries. Use that as the first implementation. Also benchmark the Warp 1.16 `bvh_constructor="cubql"` mesh ray backend for the suspension-ray workload if the installed runtime supports it. cuBQL is ray-query-specific in this Warp generation; it is acceptable to maintain a second normal mesh/BVH over the same immutable geometry for chassis AABB queries.

Select acceleration paths by measured correctness and throughput, not novelty.

## Static DFH asset

The stadium collision geometry must correspond to the RocketSim oracle's DFH/Soccar arena.

Preferred source order:

1. exact local `.cmf` collision assets already consumed by the installed RocketSim oracle;
2. assets produced with the pinned/current `ZealanL/RLArenaCollisionDumper`;
3. RLBot's documented DFH mesh extraction flow as fallback/independent validation.

RocketSim source defines `.cmf` as vertex and triangle-index collision data under `collision_meshes/`.

The RLBot documentation independently identifies DFH collision components including:

- `Goal_STD_Collision`;
- `Field_STD_Collision_Corner`;
- `Field_STD_Collision_SideTop`;
- `Field_STD_Collision_SideBot`.

Do not manually approximate the arena from dimensions.

### Asset/legal rule

Do **not** commit extracted/repacked Rocket League collision mesh assets to this public repository.

Keep raw `.cmf`, `.pskx`, generated vertex/index binaries and GPU mesh blobs ignored locally. Commit only:

- loader/parser code;
- source/dumper/extraction provenance;
- file names;
- file sizes;
- SHA-256;
- RocketSim collision mesh hash if available;
- vertex/triangle counts;
- bounds;
- reproduction instructions.

If existing local RocketSim collision assets are used, do not modify them.

## Gate A — mesh + query engine

Implement the stadium loader and GPU mesh.

Required validation:

- exact deterministic parse;
- sensible DFH bounds consistent with public field values;
- stable triangle winding/normals;
- CPU query reference against the exact same triangle data;
- large deterministic ray corpus covering floor, walls, ramps, corners, goal geometry, ceiling, triangle edges and misses.

Required query microbenchmark:

- 8 suspension-style rays/world/tick (4 wheels × 2 cars);
- shared mesh;
- realistic origins/directions;
- GPU-resident hit buffers;
- normal Warp mesh BVH;
- cuBQL ray backend if supported;
- repeated throughput/variance/VRAM/utilization.

Record rays/sec and world-equivalent ticks/sec.

Do not move to vehicle mechanics until ray hit/miss, nearest distance, point and normal are trusted.

## Gate B — wheels, suspension and ground driving

Port the required source behavior from the pinned RocketSim implementation, especially:

- `Car::_BulletSetup()` wheel configuration;
- `Car::_UpdateWheels()`;
- `btVehicleRL::updateWheelTransformsWS()`;
- `btVehicleRL::rayCast()`;
- `btVehicleRL::updateSuspension()`;
- `btVehicleRL::calcFrictionImpulses()`;
- `btVehicleRL::applyFrictionImpulses()`;
- steering/handbrake state/constants and related helpers.

Do not replace these with a generic bicycle model or the public RLBot turn-radius approximation.

Implement sufficient GPU wheel state to reproduce:

- four wheel transforms;
- ray contact point/normal/distance;
- suspension length and velocity;
- clipped contact factor;
- suspension force and extra pushback;
- longitudinal/lateral wheel impulses;
- engine/brake/coast behavior;
- steer angle;
- handbrake/powerslide behavior;
- world/static contact;
- wheel count and `isOnGround`.

Use the exact Octane-compatible `CarConfig` from the RocketSim oracle for the v0.2 corpus.

### Ground sanity checks

Use public values only as secondary checks, including:

- ~1410 uu/s max non-boost driving speed;
- 2300 uu/s car max speed;
- ~-3500 uu/s² braking;
- ~-525 uu/s² coast deceleration;
- published speed-dependent curvature;
- ~17.01 uu Octane rest height.

Live/pinned RocketSim trajectories remain the gate oracle.

## Gate C — chassis vs static world

Implement oriented car-hitbox collision against the static arena.

Preferred architecture:

1. compute the oriented hitbox's conservative world AABB;
2. use Warp mesh AABB/BVH query to enumerate overlapping triangle candidates;
3. run OBB-triangle narrow phase on GPU;
4. generate contact point, normal and penetration;
5. apply rigid-body contact impulse/friction/restitution and penetration correction closely enough to reproduce RocketSim/Bullet behavior;
6. apply angular response from off-center contacts;
7. support multiple simultaneous contacts where required for stable floor/wall/corner behavior.

Study the exact RocketSim/Bullet helpers called from car-world contact and `btVehicleRL` rather than guessing solver equations.

You do not need Bullet API compatibility. You do need measured trajectory/contact fidelity.

No SDF-only substitute is accepted for this milestone.

## Tick ordering

Resolve and document the exact v0.2 order from RocketSim source.

The implementation must preserve the distinction between:

- wheel ray/friction preparation;
- control/jump/air/boost pre-tick updates;
- suspension/wheel impulses;
- world/contact integration/solve;
- post-tick contact/state/timer updates;
- velocity/angular caps and previous controls.

Add/update architecture documentation with the final resolved order.

## Benchmark

Follow `handoff/v0.2/BENCHMARK_AND_PARITY.md` exactly.

Publish separate B0/B1/B2/B3 measurements:

- B0: v0.1 contact-free regression;
- B1: stadium wheel-ray queries only;
- B2: wheel/suspension/ground-force path;
- B3: complete v0.2 static-world path including chassis contact.

Use a deterministic **GPU-resident changing action tape**, not constant controls and not per-tick Python uploads.

B3 must use a contact-rich state mixture across floor/ramp/wall/ceiling/landing/body-contact conditions.

Initial required sweep:

`1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`

Stop after the measured stable optimum is bracketed. Continue to 262,144 only if B3 is still materially scaling and resources allow.

Report at minimum:

- worlds;
- world ticks/s;
- aggregate simulated game-seconds/s;
- suspension rays/s;
- mesh candidate/contact statistics;
- GPU utilization;
- VRAM;
- CPU utilization;
- timed H2D/D2H bytes;
- CV;
- NaNs/errors;
- cost multipliers B0→B1→B2→B3.

Keep timed state/control hot loop on GPU.

## Parity

Build a new contact-rich RocketSim parity corpus.

Do not preselect forgiving tolerances. Repeat the v0.1 method:

1. measurement-only run with empty/new tolerances;
2. fix oracle/axis/query errors;
3. inspect distributions;
4. freeze tolerances prospectively;
5. clean gate run.

Required scenario families:

- floor settling/rest height;
- level/tilted/partial-wheel landings;
- throttle/reverse/coast/brake;
- ground boost;
- steering at multiple speeds and partial/full steer;
- powerslide initiation/hold/release;
- floor→ramp→wall;
- wall traversal/up/down;
- back wall/corner;
- wall→ceiling/ceiling contact where attainable;
- nose/side/roof/body impacts;
- off-center impacts and scraping contacts.

Required horizons include:

`1, 4, 8, 30, 60, 120, 300, 600 ticks`

Track continuous state plus discrete wheel/contact/on-ground/timer state. Treat axis/sign/contact-timing mismatches as hard failures.

## Stress

Run at least 2,400 ticks of deterministic randomized contact-rich worlds/controls with:

- no NaNs/Infs;
- bounded velocities;
- no runaway penetration/energy explosion;
- stable floor resting;
- deterministic repeatability for the same build/seed.

## Performance verdict

Use the package classification:

### `PASS_GREEN`

- required parity passes;
- B3 >=100,000 aggregate sim-s/s;
- stable GPU-resident scaling.

### `PASS_YELLOW`

- required parity passes;
- B3 >=20,000 and <100,000 sim-s/s;
- profiling shows no architectural dead end.

### `PAUSE_RED`

- fundamental fidelity failure, or
- B3 <20,000 sim-s/s without a clear optimization path, or
- required per-tick host round trips / unstable contact architecture.

Do not sacrifice fidelity merely to reach a classification threshold.

## Tests and validation

Add automated tests covering at minimum:

- mesh parse/hash/statistics;
- mesh bounds;
- ray hit/miss/distance/normal;
- ray CPU/GPU parity;
- wheel transform/ray state;
- suspension no-hit/contact/compression/relaxation;
- wheel contact count and `isOnGround`;
- straight throttle/reverse/coast/brake;
- steering symmetry;
- handbrake/powerslide state;
- chassis broadphase/narrow-phase basic contacts;
- static contact impulse/angular response;
- resting stability;
- wall/ramp transition cases;
- contact-rich no-NaN stress;
- all v0.1 regression tests.

Run Ruff/formatting, `compileall` or equivalent, JSON validation, `git diff --check`, and asset/credential/path audit before publishing.

## Evidence and documentation

Commit compact evidence under a new version only:

- `results/v0.2/benchmark.json`
- `results/v0.2/parity.json`
- `results/v0.2/manifest.json`
- `docs/V0_2_RESULTS.md`
- `docs/REPRODUCING_V0_2.md`

Update:

- `README.md` after the result exists;
- `VERSION.md` after the result exists;
- `CHANGELOG.md`;
- architecture/source/licensing docs where appropriate.

Do not put raw extracted Rocket League collision assets into Git.

## Git discipline

- work from current `origin/main`;
- keep coherent commits;
- commit implementation before long evidence runs when sensible;
- push stable checkpoints so work is not stranded in chat/local state;
- final `origin/main` must include the complete v0.2 result and clean worktree;
- read back final remote SHA and principal evidence blobs.

## End report

Return a compact but complete report containing:

- starting and final commit SHA(s);
- exact Python/Warp/CUDA/driver/GPU versions;
- RocketSim/dumper/mesh provenance and hashes;
- triangle/vertex counts and mesh bounds;
- chosen Warp BVH/ray backend and why;
- implemented wheel/suspension/ground/contact mechanics;
- test/validation results;
- B0/B1/B2/B3 performance tables;
- best B3 batch size and aggregate sim-s/s;
- ray throughput;
- cost decomposition from v0.1;
- GPU/VRAM/CPU/transfer evidence;
- parity errors and frozen tolerances by scenario/horizon;
- stress result;
- `PASS_GREEN`, `PASS_YELLOW`, or `PAUSE_RED` verdict;
- most important remaining fidelity/performance risk;
- recommended next smallest step.

Even if `PASS_GREEN`, **do not start v0.3**. Stop after v0.2 is pushed and remotely verified.
