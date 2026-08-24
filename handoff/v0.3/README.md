# RivalSim v0.3 — Dynamic Contacts Handoff

Status: **AUTHORIZED / NOT YET IMPLEMENTED**

This handoff authorizes RivalSim v0.3 on top of the completed v0.2.2 static-world release.

## Frozen starting boundary

The v0.2.2 release commit is:

`6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8`

The v0.2.2 implementation boundary is:

`66c7f5ab1311444ebbec06515563d749c2c4cab6`

v0.2.2 is `PASS_GREEN` and must remain a protected regression baseline:

- 39,236 / 39,236 deterministic Octane/Soccar static-world cases passed;
- 156,944 checkpoint comparisons passed at ticks 1, 4, 8, and 12;
- 0 hard mismatch events;
- 0 numeric tolerance failures;
- B3: 511,886.15 aggregate simulated game-seconds/s at 262,144 worlds;
- zero timed host/device transfers;
- v0.1 live-RocketSim regression: 27 / 27;
- deterministic stress and both arena-query backends passed.

Do not rewrite, delete, regenerate, or weaken published `results/v0.1/`, `results/v0.2/`, `results/v0.2.1/`, or `results/v0.2.2/` evidence.

## Mission

Implement the bounded dynamic-contact physics needed for standard Soccar 1v1:

1. **ball ↔ static arena**;
2. **car ↔ ball**;
3. **car ↔ car**;
4. the physical bump/demolition contact semantics needed to make car-car behavior authoritative;
5. integrated multi-contact validation where those interactions occur simultaneously with the already-proven static world.

The goal is not to build a generic Bullet replacement. The goal is to translate the exact pinned RocketSim/Bullet behavior actually exercised by one standard Soccar arena containing exactly two Octanes and one standard ball.

## Governing lesson from v0.2.2

Do **not** begin from an approximate collision model and infer RocketSim behavior from trajectory failures.

For every new dynamic interaction:

1. map the exact pinned RocketSim/Bullet source call chain first;
2. identify constants, shape configuration, operation ordering, margins, callbacks, solver rows, and RocketSim-specific post-processing;
3. translate only the reachable source behavior into fixed-size GPU-compatible data structures and kernels;
4. validate the translation against cached native authority;
5. use parity failures to locate an incorrect source translation, not to invent a behavioral fit.

Algebraically equivalent float32 expressions are **not** assumed numerically equivalent. v0.2.2 demonstrated that source operation order, SSE-style normalization/matrix operations, unit-conversion provenance, and strict lane/tie ordering can affect later contact branches. Preserve source order where the authority proves it matters.

Read `handoff/v0.3/SOURCE_PORT_POLICY.md` before implementation.

## Primary authority

Use the same pinned authority lineage as v0.2.2 unless the user explicitly authorizes a change:

- RocketSim primary commit: `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- RocketSim Python-binding commit: `2da51b1dac7b8127127613a5ff30e490bdd70dd8`;
- installed RocketSim package: `2.2.1`;
- v0.2.2 native extension SHA-256: `E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`;
- combined CMF SHA-256: `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`.

The existing native diagnostic tool may be extended against the same pinned sources. Instrumentation must be read-only with respect to authority physics: expose internal values, do not modify the reference behavior.

## Preserve what already works

Treat these as frozen unless an identical-input trace proves a regression is caused there:

- v0.1 contact-free car and free-ball integration;
- exact 16-file Soccar CMF custody;
- immutable shared arena geometry;
- quantized-BVH / static-plane query ordering established by v0.2.2;
- Bullet-unit wheel ray path;
- Octane static-world GJK/Voronoi/EPA path;
- persistent-manifold refresh/reduction and strict tie ordering;
- internal-edge/shared-vertex normal adjustment;
- wheel/suspension/friction prepass;
- force/torque accumulation;
- static contact rows and split impulse;
- rigid-body velocity/transform integration;
- GPU-resident SoA world layout;
- v0.2.2 oracle/cache infrastructure and content-addressed identity discipline.

New dynamic physics should reuse these mechanisms where the pinned source really shares them, but do not force reuse when RocketSim/Bullet uses a different shape or code path.

---

# Phase A — Ball ↔ arena

Implement and validate the standard RocketSim ball against the existing DFH/Soccar static world.

## Required behavior

Source-map and port the reachable behavior for:

- ball collision shape, radius, margin, mass, inertia, restitution, and friction;
- ball-vs-triangle / ball-vs-analytic-plane contact generation;
- contact callback / manifold semantics relevant to the ball;
- normal and friction impulses;
- angular response and spin transfer;
- rolling/sliding transition as produced by the authority;
- bounce response;
- drag/damping and speed/angular caps in the correct tick phase;
- solver/integration ordering relative to static contacts;
- simultaneous contact with multiple arena faces at seams, corners, ramps, goals, and ceiling transitions.

Do not assume a generic sphere bounce formula is sufficient. Port the pinned path first.

## Breadth coverage

The authority corpus should deliberately cover:

- floor, walls, back walls, goal interior, ceiling, ramps, curves, and corners;
- all 16 CMF bodies;
- all 8,020 arena triangles where a practical ball contact state can be generated;
- planar, convex, and concave shared-edge transitions;
- low-speed rolling/sliding;
- shallow glancing contact;
- ordinary bounce;
- high-speed impact up to the authoritative ball speed range;
- nonzero spin around multiple axes;
- simultaneous two-face and corner contacts;
- near-margin and deeper-overlap starts where RocketSim accepts the state.

Hard parity horizons remain exactly ticks **1, 4, 8, and 12**. Longer open-loop trajectories may be diagnostic only.

Phase A must pass its representative gate before Phase B implementation begins.

---

# Phase B — Car ↔ ball

Implement and validate one Octane contacting one standard ball.

## Required behavior

Source-map and port:

- broadphase/pair routing actually used by the pinned build;
- Octane-box ↔ ball contact generation;
- margins and witness construction;
- manifold/contact ordering;
- restitution/friction and solver-row construction;
- linear and angular impulse response on both bodies;
- any RocketSim/Rocket-League-specific car-ball hit handling that changes the physical result;
- operation ordering relative to wheel forces, car controls, static contacts, ball integration, and post-tick caps.

Do not hand-code a "Rocket League hit" model unless that is literally what the pinned RocketSim source does.

## Breadth coverage

Include representative and broad cases spanning:

- bumper/front hits;
- side hits;
- rear hits;
- roof/underside orientations where physically reachable;
- glancing contacts;
- low-speed dribbles;
- hard shots;
- car and ball moving toward one another;
- car and ball moving in the same direction at different speeds;
- aerial contacts;
- spinning balls;
- rotating cars;
- grounded car-ball contact;
- wall-adjacent and corner-adjacent contacts;
- car simultaneously contacting static geometry while striking the ball;
- ball simultaneously contacting static geometry while struck by the car.

Phase B must pass its representative gate while Phase A and all v0.2.2 regressions remain green before Phase C begins.

---

# Phase C — Car ↔ car

Implement and validate Octane ↔ Octane dynamic collision behavior for standard 1v1.

## Required behavior

Source-map and port:

- Octane-box ↔ Octane-box collision path;
- pair/contact/manifold ordering;
- friction/restitution/contact rows;
- impulse and angular response for both cars;
- simultaneous wheel/static contacts;
- RocketSim bump classification and demolition predicate/event semantics to the extent they are physically coupled to the collision.

The **physical collision response and demo/bump event determination** may be included here. Match removal, scoring, respawn timing, spawn location, and other game-rule consequences remain v0.4.

## Breadth coverage

Include:

- head-on collisions;
- nose-to-side;
- side-to-side;
- rear impacts;
- glancing/scraping contacts;
- relative-yaw/pitch/roll configurations;
- airborne collisions;
- one grounded / one airborne;
- both grounded;
- wall-adjacent and corner-adjacent collisions;
- high closing speed;
- low closing speed;
- bump/demo threshold neighborhoods from both sides of every strict authority branch;
- simultaneous static contact when practical.

Phase C must pass its representative gate while Phases A/B and all older regressions remain green before integrated acceptance.

---

# Phase D — Integrated dynamic-contact gate

After A/B/C are independently source-correct, validate combinations that exercise solver ordering rather than isolated pair behavior.

Required integrated families include at minimum:

- car ↔ ball ↔ floor;
- car ↔ ball ↔ wall;
- car ↔ ball at a shared arena edge/corner;
- car ↔ car ↔ floor;
- car ↔ car ↔ wall;
- two cars contacting the ball in the same local transition where the authority supports the setup;
- dynamic contact while wheel/suspension forces are active;
- dynamic contact during aerial car torque/boost state;
- simultaneous multiple manifolds across static and dynamic bodies.

The purpose is to verify pair/constraint ordering and shared rigid-state evolution, not to simulate complete matches yet.

---

# Explicit non-goals for v0.3

Do **not** implement in this milestone:

- goals/scoring;
- kickoff logic;
- match clock;
- match reset state;
- demolition removal/respawn timing and spawn placement;
- boost-pad episode integration beyond preserving the existing static-world regression;
- RLGym environment objects;
- observation building;
- action parsing beyond what the physics harness needs;
- rewards;
- PPO/rollout collection;
- Rival policy integration;
- rendering/UI;
- Hoops, Dropshot, Snowday, Heatseeker, Rumble, or other modes;
- arbitrary body counts;
- generic Bullet API compatibility.

Do not begin v0.4 even if v0.3 passes.

# Required implementation sequence

1. Verify the checked-out handoff is based on v0.2.2 release `6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8` and inspect `results/v0.2.2/manifest.json`.
2. Build a written/source-hashed v0.3 source map for **Phase A only** before changing physics.
3. Design and freeze the Phase A deterministic corpus and authority identity.
4. Generate RocketSim authority once; cache native frames and immediate post-`SetState` readback; no live fallback after freeze.
5. Implement the bounded source port.
6. Run a representative cached gate.
7. For failures, generate/cache operation-level native traces once and compare the GPU path against them at the **first differing operation**.
8. When representative parity is clean, run the full Phase A cached corpus.
9. Repeat the same source-map → frozen authority → representative → full sequence for Phase B, then Phase C.
10. Run the integrated Phase D gate.
11. Rerun the complete frozen v0.2.2 static corpus and v0.1 live regression.
12. Only after fidelity is green, run stress/determinism/residency checks and the v0.3 performance sweep.
13. Publish compact evidence and final manifest only if all completion gates pass.
14. Push `origin/main`, verify remote readback, and stop before v0.4.

# Completion boundary

v0.3 is complete only when all of the following are true:

- Phase A representative and full ball-world gates pass;
- Phase B representative and full car-ball gates pass;
- Phase C representative and full car-car gates pass;
- integrated multi-contact gate passes;
- all required hard semantic checks pass;
- all frozen numeric tolerances pass with no silent broadening;
- the complete v0.2.2 39,236-case static corpus remains green;
- v0.1 live RocketSim regression remains 27 / 27;
- deterministic stress is finite, bounded, and reproducible;
- hot-path simulation remains GPU resident with zero routine timed H2D/D2H transfers;
- final dynamic B3-style benchmark is measured after parity;
- final complete dynamic throughput is at least **100,000 aggregate simulated game-seconds/s**; if fidelity is green but throughput is below this floor, stop as a performance boundary rather than weakening parity;
- compact source/authority/parity/regression/benchmark evidence is committed;
- large cache/trace artifacts remain local/ignored with content identities committed;
- `origin/main` is pushed and remotely verified;
- v0.4 has not begun.

The v0.2.2 value of 511,886.15 sim-s/s is a comparison baseline, **not** the v0.3 success threshold. Dynamic collisions necessarily add work. Report the measured ratio and bottleneck decomposition rather than optimizing toward the old static number while parity is red.

Read `handoff/v0.3/ACCEPTANCE.md` for the authority/corpus/evidence rules and `handoff/v0.3/SOURCE_PORT_POLICY.md` for the required source-first debugging policy.
