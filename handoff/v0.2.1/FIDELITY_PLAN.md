# RivalSim v0.2.1 Fidelity Plan

## Why v0.2 failed

v0.2 established that the GPU architecture has ample performance headroom:

- B3 best: 1,350,748.16 aggregate simulated game-seconds/s;
- zero timed H2D/D2H traffic;
- deterministic contact-rich stress;
- mesh/ray query fidelity effectively passed.

The failure was static-world solver fidelity:

- 85 scenario/horizon records with hard mismatches;
- 617 numeric failures;
- `on_ground` mismatch in 69 records;
- wheel-contact mismatches in 47–57 records per wheel;
- chassis world-contact mismatch in 44 records;
- linear-velocity direction mismatch in 25 records;
- divergence grows sharply in steering, powerslide, wall/surface transitions and impacts.

That pattern strongly implicates solver semantics downstream of the already-validated geometry queries.

## Diagnostic philosophy

Treat position/orientation error as a symptom. The target is the **first causal state/impulse divergence**.

For each selected scenario, create a synchronized table by physics tick and solver stage:

| Tick | Stage | RocketSim | RivalSim | Delta | Classification |
| ---: | --- | --- | --- | --- | --- |
| ... | wheel ray | ... | ... | ... | geometry/state |
| ... | friction calculation | ... | ... | ... | solver |
| ... | suspension apply | ... | ... | ... | solver/order |
| ... | manifold/contact | ... | ... | ... | contact |
| ... | post-solve velocity | ... | ... | ... | consequence |

The first material difference drives the next code change.

## Reference instrumentation

Prefer a wrapper or diagnostic build around the exact pinned RocketSim tree instead of modifying the installed Python extension.

Record:

- upstream commit;
- compiler/build configuration;
- diagnostic source/patch SHA-256;
- whether each field is read before or after a specific solver stage;
- any local changes required only for logging.

Logging changes must not alter reference physics.

## Solver equivalence targets

### 1. Wheel ray state

The existing geometry query gate already passed. Confirm the runtime wheel-specific transforms and derived values match:

- hard point;
- wheel direction;
- axle;
- ray length;
- contact point/normal;
- suspension length;
- relative velocity;
- clipped inverse contact-dot value;
- extra pushback.

If these match at the first divergent tick, freeze this layer and move downstream.

### 2. Wheel control setup

Match RocketSim's exact control semantics and order for:

- throttle vs brake selection;
- boost forcing real throttle;
- engine force curves;
- coast/brake forces;
- steering curve and wheel assignment;
- handbrake rise/fall and powerslide influence;
- any wheel-specific front/back scaling.

### 3. Friction impulse calculation

Reproduce the effective-mass and bilateral impulse calculation actually used by RocketSim/Bullet.

Compare per wheel:

- projected axle and forward directions;
- lateral relative velocity;
- side impulse before clipping;
- longitudinal/rolling impulse;
- suspension-force contribution to friction limit;
- friction/slip clipping;
- final side/forward impulses;
- angular influence factors.

Avoid empirical coefficients unless they are source-backed.

### 4. Suspension apply

Match:

- spring force;
- compression vs relaxation damping;
- front/back force scale;
- downward-force suppression;
- extra pushback;
- impulse application point;
- resulting linear and angular delta.

### 5. Chassis static contact

The current independent OBB-triangle SAT solver may identify plausible contacts but still differ from Bullet's manifold semantics.

Inspect:

- Bullet triangle-vs-box contact generation used in RocketSim;
- `btAdjustInternalEdgeContacts` normal changes;
- manifold point selection/reduction;
- contact breaking/processing thresholds;
- combined friction/restitution;
- persistent manifold lifetime;
- normal effective mass;
- penetration correction/ERP;
- restitution velocity term;
- friction constraints;
- warmstart impulses;
- solver iterations and order.

Do not port unrelated Bullet systems. Build the smallest static box-vs-triangle sequential-impulse path that reproduces the frozen cases.

### 6. Tick/solver ordering

Exact equations in the wrong order still diverge.

Reconstruct the actual RocketSim tick ordering around:

1. `Car::_PreTickUpdate`;
2. `btVehicleRL::updateVehicleFirst`;
3. wheel/control setup;
4. jump/air/boost mechanics;
5. `btVehicleRL::updateVehicleSecond`;
6. Bullet world step/contact solver;
7. RocketSim contact callbacks/internal-edge adjustment;
8. `_PostTickUpdate`;
9. `_FinishPhysicsTick` velocity limits/state updates.

Record the real order from source and diagnostic traces rather than assuming the conceptual order.

## Iteration strategy

Use narrow commits tied to diagnosed causes, for example:

- `Match Bullet bilateral wheel impulse effective mass`
- `Match RocketSim wheel friction clipping order`
- `Add static manifold persistence/warmstart`
- `Match Bullet internal-edge contact normals`

After each one:

1. rerun representative cases;
2. confirm the first divergence moved later or disappeared;
3. ensure previously corrected representative cases do not regress.

Only when the representative set passes should the full 280-record frozen gate run.

## Performance strategy

Performance is frozen as a secondary objective until correctness passes.

After parity:

- profile corrected B3;
- preserve cuBQL ray queries;
- fuse only proven hot kernels;
- use GPU-resident persistent contact/manifold arrays if required;
- avoid host-side per-contact loops;
- keep static arena data shared across worlds.

The corrected solver only needs to remain >=100,000 aggregate simulated game-seconds/s for this milestone to succeed.
