# RivalSim v0.2.1 Results

## Outcome

**RivalSim v0.2.1 passes the revised static-world boundary with `PASS_GREEN`.**

The corrected solver passes all 140 required RocketSim local-transition checkpoints across 35
scenarios at 1, 4, 8, and 12 physics ticks. There are zero hard state/contact/direction
mismatches and zero numeric failures under the unchanged frozen v0.2 tolerances. The corrected
complete B3 path reaches **822,480.77 aggregate simulated game-seconds/s** at 262,144 worlds
with **0.403% CV**, stable scaling, and zero timed host/device transfer.

This is still a bounded static-world transition engine. It is not yet a complete Rocket League
simulator and does not implement dynamic ball-world, car-ball, or car-car contacts, scoring,
game resets, training integration, or policy deployment. v0.3 was not begun.

| Gate | Measured result | Status |
| --- | ---: | --- |
| Local RocketSim parity | 35 scenarios × 4 checkpoints; 0 hard, 0 numeric failures | Pass |
| Unchanged tolerances | Same seven v0.2 numeric limits and hard semantic checks | Pass |
| v0.1 regression | 27/27 same-equation/live RocketSim/axis-sign corpus | Pass |
| Repository regression | 38/38 tests | Pass |
| Stress/determinism | Two identical 64-world × 2,400-tick full-state hashes | Pass |
| Finite/bounded state | Finite; max 1,317.34 uu/s and 5.5 rad/s | Pass |
| Hot-loop residency | 0 H2D bytes; 0 D2H bytes | Pass |
| Corrected B3 floor | 822,480.77 sim-s/s versus 100,000 required | Pass |
| Green class | 822,480.77 sim-s/s versus 500,000 required | `PASS_GREEN` |

## Governing validation-policy adjustment

The immediate user steering adjustment on 2026-08-23 superseded the handoff's synchronized
eight-horizon parity requirement at the next clean boundary. The hard fidelity window is now:

`1, 4, 8, 12 ticks`

At 120 Hz, 12 ticks is 100 ms and three Rival `mechanics4` policy-decision intervals. The
30/60/120/300/600-tick synchronized open-loop trajectories are diagnostic and non-blocking.
They may justify work only when they reveal a systematic error that is also visible inside the
1–12-tick window.

This policy does not weaken the local checks. Wrong wheel/ground/contact state, collision
decision, normal/sign, impulse direction, boost state, jump/dodge/flip state, or any numeric
error above the frozen limits remains a failure. The policy changes horizon semantics, not
tolerances.

Long-duration fidelity should eventually use closed-loop policy behavior and transfer evidence:
train in RivalSim, evaluate independently in RocketSim, then evaluate through RLBot. Demanding
that two floating-point contact engines remain on one synchronized open-loop trajectory for
several seconds measures chaotic branch identity more than useful training transfer.

## Custody and immutable inputs

- Authority start: `cc45ab0dce85f2c696800b96e1f4af8b7d8bb1f2`.
- v0.2.1 implementation: `9939d0736a92cdfa6ce842d7818634e08260dd65`.
- Frozen v0.2 evidence commit: `2c5d11899eaaad6a963a370fcc3813202b6fa714`.
- RocketSim primary source: `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`.
- RocketSimPython/binding source: `2da51b1dac7b8127127613a5ff30e490bdd70dd8`.
- External Soccar collision content SHA-256:
  `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`.
- Collision geometry: 16 CMFs, 4,468 vertices, 8,020 triangles.
- `results/v0.1/` and `results/v0.2/`: byte-for-byte unchanged from the frozen evidence commit.
- Tracked extracted collision assets: zero.

The manifest records byte hashes for the old evidence, current source blobs, source-only native
diagnostic helper, external collision inputs, and each v0.2.1 evidence file.

## What the divergence investigation found

The frozen v0.2 gate contained 85 scenario/horizon records with hard mismatches and 617 numeric
tolerance failures. The first-cause work did not tune final position. It aligned the native
RocketSim/Bullet stages and compared wheel rows, manifolds, constraint inputs, impulses, and
pre/post rigid state tick by tick.

The complete ranked baseline is in
[`results/v0.2.1/divergence_index.json`](../results/v0.2.1/divergence_index.json).

| Representative | Earliest baseline failure | First causal stage | Source-backed resolution |
| --- | --- | --- | --- |
| `steer_medium_full_left` | Numeric at tick 4; later hard wheel/ground mismatches | Wheel friction preparation and two-phase application | Cache every wheel ray, suspension value, side impulse, and rolling impulse from one pre-solve body state; apply them later in RocketSim order |
| `powerslide_initiation` | Numeric at tick 4 | Wheel axle/forward basis, bilateral impulse, and powerslide scaling | Reproduce `btVehicleRL` basis, clipped suspension velocity, friction-slip limits, extra pushback, and handbrake scaling |
| `ramp_transition` | Angular velocity at tick 4 | Triangle feature/order and shared-edge normal adjustment | Reproduce per-CMF quantized-BVH rank, adjacency flags/angles, SAT/GJK closest features, thresholding, and callback-normal order |
| `off_center_impact` | Numeric at tick 1; hard contact/ground mismatch at tick 4 | Manifold row construction and split/velocity solve | Use native box margin/inertia, common prestate, Bullet RHS/tangent rules, ten split and velocity PGS iterations, writeback, then caps |

### Native internal oracle

`tools/rocketsim_diagnostic/` is a source-only executable harness. It links the unmodified pinned
RocketSimPython source and its vendored Bullet 3.24 source. It does not patch either upstream,
and its logging is absent from RivalSim's benchmark path.

The helper exposes:

- car transform and linear/angular velocity before and after the native physics stages;
- every wheel ray hit, suspension length/velocity/force, axle/forward basis, and friction impulse;
- Bullet persistent manifolds, triangle identity, point, normal, distance, and lifetime;
- normal, lateral, and push impulses plus solver prestate;
- box/triangle support and GJK simplex diagnostics for selected feature disputes.

The comparison wrapper used during diagnosis is an ignored local tool, while the reproducible
CMake recipe and C++ trace source are tracked. Build products and collision assets are not.

## Source-backed implementation corrections

### Vehicle order and wheel forces

`rivalsim/kernels/vehicle.py` now mirrors RocketSim's `Car::_PreTickUpdate`,
`btVehicleRL::updateVehicleFirst`, Bullet's bilateral calculation, and
`btVehicleRL::updateVehicleSecond` ordering. All four wheel transforms/rays and friction rows are
prepared from a common body state before any wheel force changes that state. Suspension force,
clipped relative velocity, extra pushback, side/rolling impulses, steering, braking, throttle,
handbrake, and grounded boost use the pinned source branches and units.

### Collision geometry and closest features

`rivalsim/arena.py` builds the same per-CMF shared-edge angles/flags used by
`btGenerateInternalEdgeInfo` and a deterministic face rank matching Bullet's quantized BVH visit
order. `rivalsim/kernels/vehicle.py` uses exact box half extents/margin and a source-matched
triangle/box candidate path with SAT separation, GJK simplex reduction, closest edge/face
features, internal-edge normal adjustment, breaking thresholds, and manifold ordering.

### Constraint solving and writeback

Every contact row is constructed from the unchanged rigid-body prestate. The implementation
matches Bullet's inverse-inertia multiplication order, tangent basis, friction RHS asymmetry,
row-skip behavior, accumulated impulses, ten split-impulse iterations, ten velocity iterations,
split transform writeback, velocity integration, and deferred RocketSim velocity caps. Vehicle
snapshot fields preserve the values needed to align the native trace and verify determinism.

### Static boost behavior

Grounded boost uses RocketSim's grounded acceleration. The static-world path also carries the 34
standard Soccar pad positions and GPU-resident per-world pickup lock/cooldown state. Big and small
pad overlap, pickup amount, cooldown, and reset behavior are source-backed and directly tested.
This does not begin dynamic-contact v0.3 work.

## Authoritative local parity

The final gate uses the exact existing 35 scenarios and all seven previously frozen numeric
tolerances. Each simulator is initialized from the same authoritative state and receives the
same controls. The four comparisons are at 1/4/8/12 ticks.

| Metric | Median error | p95 error | Maximum error | Frozen tolerance |
| --- | ---: | ---: | ---: | ---: |
| Position | 0.00000381 uu | 0.00030691 uu | 0.00097847 uu | 10.0 uu |
| Linear velocity | 0.00007911 uu/s | 0.00085467 uu/s | 0.00275174 uu/s | 25.0 uu/s |
| Orientation | 0 rad | 0.00034168 rad | 0.00039079 rad | 0.025 rad |
| Angular velocity | 0.000000277 rad/s | 0.000005969 rad/s | 0.000055841 rad/s | 0.1 rad/s |
| Boost | 0 | 0 | 0 | 0.01 |
| Handbrake value | 0 | 0 | 0 | 0.0001 |
| World-contact normal | 0 rad | 0 rad | 0 rad | 0.05 rad |

The worst normalized check is `side_impact` orientation at tick 1: 0.00039079 rad, or 1.56%
of its frozen tolerance. All on-ground, four wheel-contact, world-contact, velocity-direction,
and world-normal-direction checks pass exactly.

## Bounded DFH breadth prototype

The optional prototype intentionally reports measured breadth instead of claiming it.

| Coverage item | Result |
| --- | ---: |
| Existing authoritative transition cases | 35 |
| Local checkpoint comparisons | 140 |
| Mesh triangles exercised | 2 / 8,020 (0.02494%) |
| Wheel-ray mesh faces exercised | 0 |
| Chassis mesh faces exercised | 2 |
| Shared-edge-capable triangles touched | 2 / 8,020 |
| Complete shared directed edges audited | 23,176 |
| Planar / convex / concave directed edges | 12,024 / 856 / 10,296 |

The low observed triangle count is real: many existing floor, wall, and ceiling cases collide
with RocketSim's analytic Soccar planes, and only a small subset reaches CMF triangles within 12
ticks. `results/v0.2.1/coverage.json` includes per-mesh counts, scenario/contact families,
pass/fail counts at every horizon, numeric distributions, worst cases, and the full topology
audit.

This prototype does **not** satisfy exhaustive DFH per-triangle transition coverage. A dedicated
future breadth milestone should generate authoritative face, seam, convex/concave edge,
orientation, velocity, angular-velocity, overlap, and controller combinations and reset both
engines for every case. That work must not reinstate long synchronized open-loop identity and
was not allowed to delay this v0.2.1 boundary.

## Regression, stress, and geometry query

- v0.1 live regression: 27/27, including same-equation, live RocketSim, and axis/sign checks.
- Targeted arena/query/static-world tests: 15/15.
- Full repository tests: 38/38.
- Ruff, bytecode compilation, and `git diff --check`: pass.
- Two contact-rich stress runs: 64 worlds × 2,400 ticks each.
- Full-state SHA-256 for both stress runs:
  `9046E933243ACCF3E1F64158402A628DBDEBFAA1ADD70CA5009CFC648F48A8D4`.
- Maximum stress linear speed: 1,317.3351 uu/s.
- Maximum stress angular speed: 5.5 rad/s.
- Maximum observed penetration: 33.4390 uu.
- Hot-loop H2D/D2H bytes: 0/0 in both runs.
- Floor rest after 2,400 ticks: 17.03197 uu, 0.0000349 uu/s.
- Normal Warp and cuBQL query paths both pass the independent 4,608-ray corpus with zero hit
  mismatch, at most 0.001953125 uu hit-distance error, and no unambiguous face mismatch.

## Corrected B3 performance

The official sweep measured B0, B1-default, B1-cuBQL, B2, and the corrected complete B3 at the
mandatory batches from 1,024 through 131,072 worlds, then measured 262,144 when endpoint scaling
justified it. Every timed point uses repeated CUDA-graph execution after setup/warmup, with
telemetry outside state transfer and host/device byte counters around the hot loop.

| Worlds | Aggregate sim-s/s | CV |
| ---: | ---: | ---: |
| 1,024 | 32,194.04 | 0.968% |
| 2,048 | 63,924.34 | 1.212% |
| 4,096 | 118,519.55 | 0.395% |
| 8,192 | 214,110.74 | 0.944% |
| 16,384 | 411,227.95 | 0.597% |
| 32,768 | 557,055.09 | 0.888% |
| 65,536 | 651,074.91 | 2.544% |
| 131,072 | 757,475.51 | 0.938% |
| 262,144 | **822,480.77** | **0.403%** |

At the selected batch, v0.2 measured 1,350,748.16 sim-s/s. v0.2.1 retains 60.89% of that
throughput, a 1.6423× slowdown for the source-backed contact fidelity. It remains 8.22× above
the 100,000 success floor and 1.64× above the 500,000 green threshold.

Measured common-batch cost multipliers versus B0 are 5.1334× for B1-default, 4.1732× for
B1-cuBQL, 6.0063× for B2, and 17.5019× for corrected B3. The selected suspension-ray backend
remains cuBQL.

The benchmark machine was Windows 11, Python 3.14.3, NumPy 2.5.2, Warp 1.16.0, NVIDIA driver
610.62, RTX 5090 with 32 GiB, and an 8-core/16-thread AMD64 CPU. These are machine-specific
measurements, not a general throughput guarantee.

## Explicit stop boundary

v0.2.1 is complete and published at the corrected static-world boundary. No ball-world,
car-ball, car-car, bump/demolition, scoring/reset, RLGym/PPO, Rival inference, or closed-loop
transfer implementation was started. Even with `PASS_GREEN`, v0.3 requires a separate handoff.
