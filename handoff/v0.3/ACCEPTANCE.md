# RivalSim v0.3 — Authority, Acceptance, and Evidence Rules

This document defines the validation protocol for the authorized v0.3 dynamic-contact milestone.

The governing principle is the same one that finally cleared v0.2.2:

> compare local authoritative transitions against a frozen native truth set, find the first causal operation mismatch, and never fit later trajectory errors with behavioral stabilizers.

## Hard local horizon

Authoritative blocking checkpoints remain exactly:

- tick 1;
- tick 4;
- tick 8;
- tick 12.

Cache every native frame from tick 1 through tick 12 so operation-level and intermediate-tick debugging does not require relaunching RocketSim.

Longer open-loop synchronized trajectories may be collected for diagnostics/stress only. They are not a blocking requirement and must not become another 300/600-tick identity project.

## Authority isolation

Do not validate many independent cases by inserting all authority bodies into one shared RocketSim arena.

v0.2.2 proved that apparently disabled interactions can contaminate native state when many validation objects share the same authority world.

Every authoritative starting case must therefore have isolated native-world semantics. Implement this in whichever way is fastest without changing reference physics, but prove batch/case invariance before trusting the cache.

The GPU side may remain batched.

## Content-addressed authority cache

Build a new v0.3 cache under `.tools/v0.3/` and keep large artifacts untracked.

The authority identity must change if any input capable of changing native truth changes, including:

- RocketSim primary commit;
- Python-binding commit;
- installed native extension hash;
- RocketSim package/build identity;
- collision assets;
- source-case generator code;
- generator configuration/schema;
- seed;
- exact generated corpus;
- authoritative state/readback settings;
- relevant game-mode or body-configuration settings.

The authority identity must **not** change for:

- RivalSim GPU implementation changes;
- Warp/CUDA recompilation;
- pilot-selection changes;
- comparison/reporting changes;
- tolerance changes, although tolerance changes after freeze are prohibited without explicit authorization.

After the native cache is complete, cached parity runners must have **no live-RocketSim fallback**. Missing/corrupt authority data is an error, not permission to regenerate silently.

## Initial-state custody

Store both:

1. the exact frozen source state supplied to RocketSim;
2. immediate post-`SetState` native readback.

Preserve source representations when RocketSim/Bullet internally stores a more authoritative form than the public readback. v0.2.2 demonstrated that matrix/quaternion and UU/Bullet round trips can lose one float32 ULP and later flip strict contact branches.

State provenance must be explicit rather than reconstructed opportunistically during GPU diagnostics.

## Required cached state

For every case and every native tick 1–12, cache all state necessary to compare the active milestone, including at minimum:

### Cars

- position;
- linear velocity;
- orientation / authoritative basis representation;
- angular velocity;
- boost;
- handbrake;
- ground state;
- four wheel-contact flags;
- chassis-world contact state/normal;
- bump/demo event state if Phase C requires it.

### Ball

- position;
- linear velocity;
- rotation/orientation if exposed and relevant;
- angular velocity;
- world-contact state if exposed/relevant.

### Pair/contact state

Store compact hard-semantic values needed to identify wrong physical branches, such as:

- whether each expected pair is contacting;
- contact/callback presence where reference tooling exposes it;
- bump/demo classification where applicable.

Do not bloat every ordinary trajectory record with full native internals. Deep operation data belongs in trace packages for selected failures.

## Tolerances

Do not loosen v0.2.2 tolerances for existing car/static metrics.

Before using v0.3 parity to tune implementation, freeze numeric tolerances for new ball/dynamic metrics. Default to the existing v0.2.2 scale unless direct evidence requires a stricter metric:

- position: 10 UU;
- linear velocity: 25 UU/s;
- orientation: 0.025 rad where meaningful;
- angular velocity: 0.1 rad/s;
- boost: 0.01;
- handbrake: 0.0001;
- contact-normal angle: 0.05 rad where a normal is part of the hard comparison.

These are acceptance ceilings, not tuning targets. Source-port diagnostics should normally be vastly tighter.

Hard semantic failures always block regardless of numeric tolerance. Examples include:

- missed/extra collision;
- wrong pair routing;
- wrong contact presence/sign;
- wrong retained manifold semantics when they materially drive state;
- wrong bump/demo classification;
- wrong impulse direction;
- non-finite state;
- body tunneling through accepted collision geometry;
- incorrect state reset/authority mapping.

## Representative gates

Every phase must first run a deterministic representative gate broad enough to expose mechanism classes without paying for the entire corpus after every edit.

Representative selection must be frozen and content-addressed once the phase begins.

It must include difficult boundary states, not merely random easy states.

No phase may advance merely because a hand-picked diagnostic passes.

## Phase A full corpus — ball/world

The full Phase A corpus should maximize static-arena breadth while remaining bounded and reproducible.

Coverage reporting must distinguish:

- geometry/topology audited;
- starting states generated;
- actual target contacts observed;
- parity checkpoints passed.

Report at minimum:

- total states;
- per-CMF generated/contacted counts;
- triangle coverage across the 8,020 CMF triangles;
- shared-edge coverage by planar/convex/concave class;
- analytic-plane coverage;
- rolling/sliding/bounce/high-speed/spin family counts;
- multi-face/corner family counts;
- unexercisable geometry and reason.

Do not claim complete geometric validation merely because topology was enumerated.

## Phase B full corpus — car/ball

Freeze a deterministic corpus spanning the relative-state space that materially changes car-ball physics.

Coverage should be stratified across:

- contact region on the Octane;
- relative velocity direction/magnitude;
- ball velocity and spin;
- car angular velocity/orientation;
- grounded/aerial state;
- static-contact context;
- wall/corner proximity;
- glancing versus deep/central contact;
- simultaneous ball-world contact.

Report generated cases and actual car-ball contacts separately.

## Phase C full corpus — car/car

Freeze a deterministic corpus spanning:

- relative body orientation;
- relative position/contact feature;
- closing speed;
- grounded/aerial combinations;
- wall/floor/corner context;
- bump/demo threshold neighborhoods;
- actual bump/demo classification counts.

If RocketSim uses strict threshold comparisons for bump/demo decisions, deliberately generate states on both sides of those thresholds without adding epsilon behavior to RivalSim.

## Phase D integrated corpus

The integrated corpus is smaller than the pairwise breadth corpora but must cover solver ordering across simultaneous interactions.

At minimum include:

- static + ball contact in one tick;
- static + car-ball contact;
- static + car-car contact;
- car-ball with active suspension/wheel impulses;
- ball contacting world during car impact;
- both cars interacting with the ball in a bounded transition where native initialization is stable;
- multiple simultaneous manifolds.

## Deep native traces

When a representative/full cached run fails, rank failures by:

1. earliest failing tick;
2. hard semantic failure before numeric failure;
3. largest normalized error;
4. mechanism diversity.

Generate native deep traces **once** for representative failures and store them under the v0.3 authority identity.

Extend the pinned diagnostic tool only as needed to expose the active source path. Candidate trace fields may include:

- broadphase/pair dispatch;
- support queries;
- GJK/Voronoi/simplex state;
- EPA iterations/witnesses;
- manifold insertion/refresh/reduction;
- internal-edge adjustment;
- sphere/static contacts;
- convex-convex contacts;
- solver-row construction;
- restitution/friction RHS;
- warmstart state if used;
- sequential impulse values by iteration;
- split impulse where used;
- force/torque prepass;
- rigid-body integration inputs/outputs;
- RocketSim-specific hit or bump/demo calculations.

Build machine-readable GPU/native comparators so the first differing operation can be found automatically. Do not repeatedly interpret thousands of trace lines by hand if a deterministic comparator can do it once.

## Failure policy

### Blocking and must be fixed

- hard semantic mismatches;
- systematic numeric tolerance failures;
- repeated failure clusters sharing a source operation;
- exploitable physics differences;
- wrong pair/manifold/solver ordering;
- body state entering a later tick differently because an upstream source operation is not faithfully translated.

### May be characterized only after source equivalence is established

Isolated tiny residual float noise may be recorded as nonblocking only when:

- it is comfortably inside frozen tolerances;
- it is non-systematic;
- identical-input operation traces show no missing semantic/source branch;
- it does not flip a later strict branch in the 12-tick gate.

Do not create an epsilon, hysteresis, face rule, or tie stabilizer to hide these differences.

## Regression gates

Before v0.3 can complete:

1. complete v0.2.2 static acceptance must still pass: 39,236 / 39,236;
2. v0.1 live RocketSim scenarios must remain 27 / 27;
3. arena ray correctness corpus must remain green for both existing backends;
4. repository tests/lint/compile checks must pass;
5. deterministic stress must remain reproducible.

Published old result artifacts must remain byte-for-byte unchanged.

## Performance gate

Do not tune performance while parity is red.

After all fidelity/regression gates pass:

- benchmark the complete dynamic-contact implementation;
- sweep practical large world counts rather than assuming 262,144 remains optimal;
- report world ticks/s and aggregate simulated game-seconds/s;
- report coefficient of variation;
- report peak/device VRAM observations;
- confirm zero routine timed H2D/D2H transfers in the hot loop;
- compare against v0.2.2 B3 = 511,886.15 sim-s/s;
- decompose major costs sufficiently to identify ball-world, car-ball, and car-car overhead.

v0.3 viability floor: **100,000 aggregate simulated game-seconds/s** on the complete dynamic-contact benchmark.

If fidelity is green but the complete dynamic path is below 100,000 sim-s/s, stop with a performance boundary and profile. Do not weaken physics to force a green throughput number.

## Determinism/stress

Run at least two independent identical-seed stress executions and require identical full-state hashes if the existing deterministic architecture still supports that requirement.

Stress should include repeated:

- ball bounces;
- car-ball impacts;
- car-car impacts;
- wall/corner dynamic contacts;
- mixed static/dynamic manifolds;
- high-speed states.

Require finite state and bounded authoritative speed/angular/penetration behavior.

## Final evidence package

Large authority trajectories, raw traces, diagnostics, and benchmark chunks remain local under `.tools/v0.3/` and ignored.

Commit compact evidence sufficient to verify custody and results. Suggested final layout:

- `results/v0.3/ball_world.json`;
- `results/v0.3/car_ball.json`;
- `results/v0.3/car_car.json`;
- `results/v0.3/integrated.json`;
- `results/v0.3/oracle_data.json`;
- `results/v0.3/source_port.json`;
- `results/v0.3/regression.json`;
- `results/v0.3/benchmark.json`;
- `results/v0.3/manifest.json`;
- `docs/V0_3_RESULTS.md`;
- `docs/REPRODUCING_V0_3.md`;
- `docs/V0_3_ORACLE_CACHE.md`.

The final manifest must identify:

- implementation commit;
- authority identities;
- corpus hashes;
- source/tool hashes;
- compact evidence hashes;
- prior v0.1/v0.2/v0.2.1/v0.2.2 evidence hashes;
- final parity/regression/performance verdicts.

Do not publish a `PASS_GREEN` manifest unless every v0.3 completion gate in `handoff/v0.3/README.md` is satisfied.
