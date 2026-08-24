# RivalSim v0.2.2 results — static-world source-parity breadth gate

## Verdict

**RivalSim v0.2.2 passes the complete frozen static-world boundary with `PASS_GREEN`.**

The final GPU implementation passed all 39,236 deterministic Octane/Soccar starting states at
the frozen 1/4/8/12-tick checkpoints. That is 156,944 checkpoint comparisons and 1,098,608
numeric metric comparisons against cached native RocketSim authority, with:

- 0 hard mismatch events;
- 0 numeric tolerance failures;
- 0 failed cases;
- 5 non-blocking residual events in 3 cases, all below the frozen tolerances;
- no live-RocketSim execution in the iterative or final GPU gate.

The corrected B3 static-world path reaches **511,886.15 aggregate simulated game-seconds/s** at
262,144 worlds with **0.0913% CV**, zero timed host/device transfers, stable scaling, and a green
stress/query gate. This retains 62.24% of v0.2.1 B3 throughput at the same batch and remains above
the 500,000 sim-s/s `PASS_GREEN` threshold.

## Frozen authority

The complete native authority is content-addressed by:

`6B31F9D147D5A19F882F9075C2A9F07C9A7228377A8A118CA2874F67FBD0805B`

That identity binds the pinned RocketSim and binding revisions, installed extension, combined
CMF content, generator source/config, seed, exact float32 corpus, and authority settings. The
corpus SHA-256 is:

`B548A2E509FC8EFD8AD48502AB687CFC436F260A4B83DADED82D2D17B0B5FA7B`

The native cache contains 39,236 initial-state readbacks and 470,832 authoritative frames. The
GPU runner has no live fallback: stale, incomplete, corrupt, or identity-mismatched authority is
a hard error.

## Acceptance summary

| Gate | Final evidence | Result |
| --- | ---: | --- |
| Representative selection | 1,043 cases × 4 checkpoints | 0 hard / 0 numeric failures |
| Complete corpus | 39,236 cases × 4 checkpoints | 0 hard / 0 numeric failures |
| Metric comparisons | 1,098,608 | all within frozen tolerances |
| v0.1 live regression | 27 scenarios | all pass |
| Repository tests | 46 tests | all pass |
| Query corpus | 4,608 rays, both Warp BVHs | exact gate pass |
| Deterministic stress | 2 × 64 worlds × 2,400 ticks | identical full-state SHA-256 |
| GPU residency | timed B3 hot loop | 0 H2D / 0 D2H bytes |
| Corrected B3 | 511,886.15 sim-s/s at 262,144 worlds | `PASS_GREEN` |

The frozen numeric tolerances were not widened: 10 uu position, 25 uu/s linear velocity,
0.025 rad orientation, 0.1 rad/s angular velocity, 0.01 boost, 0.0001 handbrake value, and
0.05 rad world-contact-normal angle. Ground, wheel-contact, chassis-contact, and related hard
semantics remain exact gates.

The maximum measured errors were 0.70135 uu position, 1.59512 uu/s linear velocity,
0.002562 rad orientation, 0.028764 rad/s angular velocity, and 0.002895 rad contact-normal
angle. Boost and handbrake maximum errors were zero. The worst tolerance fraction was 28.77%
for angular velocity, so no result approached the blocking boundary.

## Breadth and coverage

Generation covered every one of the 8,020 source triangles through a chassis state and a wheel
state, every one of the 23,176 shared directed edges, and 20 analytic-plane states. The complete
topology audit contains 12,024 planar, 856 convex, and 10,296 concave directed edges.

Actual paired target contact is reported separately from state generation:

- 7,752 unique triangles had a paired target contact;
- 3,504 triangles were paired on the chassis path;
- 7,259 triangles were paired on the wheel path;
- 8,912 directed edges had a paired target contact;
- all 20 floor, ceiling, left-wall, and right-wall analytic-plane states paired and passed.

Cases where the intended target was occluded, an adjacent face was selected, or no target
contact occurred still compare the complete RocketSim/RivalSim transition semantics, but are not
misreported as target-face coverage. `results/v0.2.2/parity.json` preserves the exact coverage
definitions and totals.

## What changed

v0.2.2 is a bounded fidelity port, not a new physics feature milestone. For the existing static
Octane-versus-Soccar path it translates the relevant pinned RocketSim/Bullet behavior directly:

- quantized BVH candidate ordering and source-correct static-plane wheel rays;
- `btGjkPairDetector::getClosestPointsNonVirtual` and `btVoronoiSimplexSolver` float32
  operation order for box-versus-triangle witnesses;
- the pinned GJK/EPA penetration-depth fallback sequence without broadening to a generic Bullet
  collision library;
- persistent-manifold refresh, lifetime handling, four-point area reduction, and degenerate
  ordering;
- `btAdjustInternalEdgeContacts`-equivalent shared-edge/vertex normal selection;
- wheel/suspension prepass, force and torque accumulation, constraint-row construction,
  sequential impulse and split-impulse iteration, velocity integration, and transform
  integration;
- RocketSim wheel friction coefficient, impulse, and brake-force float32 operation order.

Two final discrepancies illustrate the source-port method. A valid shallow GJK result outside
Bullet's callback-report distance was incorrectly treated as an invalid GJK solve, which caused
an unnecessary EPA fallback; the internal-valid and callback-report decisions are now distinct.
F07059-C's first mismatch was a one-ULP brake force: RocketSim computes `0.15f * 52.5f` directly,
where the old equivalent-looking `0.15f * 875.f * 0.06f` rounded lower. Porting the source order
made F07059-C and E20521 rigid state bit-exact for all 12 captured ticks.

No plane hysteresis, edge epsilon, face-specific rule, tie tolerance, downstream wheel
compensation, or behavioral fitting was added. The source-correct GJK/Voronoi/EPA and
internal-edge implementations were preserved unless identical-input traces proved a control-flow
or operation-order discrepancy.

## Evidence

- `results/v0.2.2/oracle_data.json` — authority/cache identities, verified deep-trace custody,
  representative cached run, and complete-run reference;
- `results/v0.2.2/source_port.json` — compact causal trace identities and focused final-case
  hashes/deltas;
- `results/v0.2.2/parity.json` — complete and representative cached acceptance evidence,
  distributions, residual clusters, and coverage;
- `results/v0.2.2/regression.json` — compact v0.1 live-RocketSim regression evidence;
- `results/v0.2.2/benchmark.json` — query, stress, B3, residency, and performance evidence;
- `results/v0.2.2/manifest.json` — committed implementation/evidence custody.

The large trajectory cache, deep native traces, raw diagnostics, raw benchmark, and per-chunk
GPU outputs remain local under `.tools/v0.2.2/`. Tracked evidence records their semantic
identities and SHA-256 custody without committing tens of megabytes of oracle data.

## Boundary

This result applies only to RivalSim's existing standard Soccar static-world Octane path.
Ball-world, car-ball, car-car, dynamic-body collision, bumps/demolitions, scoring/game rules,
RLGym training integration, and policy inference remain excluded. **v0.3 was not begun.**
