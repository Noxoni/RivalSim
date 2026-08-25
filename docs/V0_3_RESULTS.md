# RivalSim v0.3 Results

## Verdict

RivalSim v0.3 is complete with **`PASS_GREEN`**.

The fixed standard-Soccar world containing exactly two Octanes and one ball passes every
ball-world, car-ball, car-car, integrated-contact, regression, determinism, residency, and
performance gate. Work stopped at the v0.3 boundary; no v0.4 game-rule or training work began.

## Implemented boundary

v0.3 adds:

- standard Soccar sphere contact against the accepted CMF triangles and analytic planes;
- ball damping, gravity, friction, restitution, spin, persistent contacts, split impulse, and
  deferred RocketSim caps;
- both Octane/ball pairs, including source-ordered box-sphere GJK/Voronoi/EPA witnesses, retained
  manifolds, constraints, and RocketSim hit callbacks;
- Octane/Octane box-box physical response plus ordered bump/demo classification and queued bump
  impulses;
- car `_PreTickUpdate` visitation order as internal per-world lifecycle state established by
  construction or membership change and preserved across ordinary ticks and physical resets;
- dynamic suspension-ray candidates for the ball and other car;
- source-lifecycle `btRSBroadphase` cell/pair order, connected-island ordering, and one shared
  two-car/one-ball sequential-impulse/split-impulse/writeback path.

The port is deliberately fixed to the existing Octane/Soccar path. It is not a generic Bullet
port and does not add arbitrary dynamic body counts.

## Authoritative parity

All native frames from tick 1 through tick 12 are cached. Blocking comparisons remain ticks 1,
4, 8, and 12.

| Phase | Scope | Frozen cases | Blocking checkpoints | Result |
| --- | --- | ---: | ---: | --- |
| A | ball/world | 31,216 | 124,864 | 0 hard, 0 numeric failures |
| B | car/ball | 8,192 | 32,768 | 0 hard, 0 numeric failures |
| C | car/car + bump/demo classification | 8,192 | 32,768 | both native branches complete; 0 blocking failures |
| D | integrated static/dynamic contacts | 512 | 2,048 | both native branches complete; 0 blocking failures |

Phase A generated states for all 8,020 CMF triangles, all 23,176 shared directed edges, and 20
analytic-plane states. Actual target contact was observed in 29,571 cases, covering 7,232 unique
target triangles and 22,319 directed edges (11,983 planar, 820 convex, and 9,516 concave). The
1,645 unexercised targets are reported as other-contact observations rather than mislabeled as
target coverage.

Phase B records 8,183 native actual-contact cases and exact native/GPU callback presence in
20,692 cached frames. Coverage spans every frozen contact region, motion mode, orientation mode,
and static context.

Phase C validates both complete source-valid trajectories for every case. Ordered native/GPU
bump-demo callback counts are 9,580 for `a_then_b` and 9,581 for `b_then_a`. The gate never
combines a state metric from one branch with a semantic event from the other.

Phase D covers eight equally weighted 64-case families: static ball, car-ball with car static,
car-ball with ball static, wall-edge car-ball, car-car with static contact, both cars interacting
with the ball, wheel/car/ball interaction, and three-body multi-manifold contact. Each branch
matches 1,772 car-A/ball callbacks, 1,159 car-B/ball callbacks, and 76 ordered bump events.

## Native multi-outcome lifecycle

The native-order experiment established the controlling lifecycle:

- insertion/removal or arena reconstruction establishes or may change `_cars` visitation order;
- ordinary ticks preserve the order;
- physical `SetState`/kickoff-style resets preserve the order;
- demolition/respawn without membership mutation preserves the order.

RivalSim therefore stores one of the two valid logical orders as per-world internal lifecycle
state until membership changes. It does not expose or emulate pointers, allocator addresses, or
MSVC heap layout. It has no case-specific order table and does not inspect expected outputs at
runtime.

## Regression and integrity

- v0.2.2 complete static acceptance: 39,236/39,236 cases and 156,944/156,944 checkpoints;
- v0.1 live RocketSim: 27/27 scenarios;
- 4,608-ray CPU-reference corpus: exact gate pass for both default Warp BVH and cuBQL;
- repository tests: 63 passed;
- Ruff, Python bytecode compilation, and `git diff --check`: passed;
- published v0.1, v0.2, v0.2.1, and v0.2.2 result bytes: unchanged.

Two independent 64-world, 2,400-tick mixed-dynamic stress executions produced the same complete
state SHA-256:

`D584FC49A1ADED7F81C889C93AF2FB62F08EB5CE9B07BEB4903ADE1DF5564A7B`

Both runs were finite, stayed within the frozen speed/angular/penetration bounds, and recorded
zero hot-loop H2D/D2H bytes.

## Performance

Measured environment:

- NVIDIA GeForce RTX 5090, driver 610.62, 32 GiB;
- AMD64 Family 26 Model 68, 8 physical / 16 logical CPU cores;
- Windows 11, Python 3.14.3, NumPy 2.5.2, Warp 1.16.0;
- 120 Hz, device-resident 64-entry action tape, CUDA graph blocks of 8 ticks;
- five repeats after warmup.

The best stable complete-dynamic point is:

| Worlds | World ticks/s | Aggregate simulated game-seconds/s | CV | Peak observed VRAM | Timed transfers |
| ---: | ---: | ---: | ---: | ---: | --- |
| 131,072 | 23,593,726.80 | **196,614.39** | 1.313% | 11,051,069,440 bytes | 0 H2D / 0 D2H |

This is 1.97× the 100,000 sim-s/s v0.3 floor and retains 38.41% of the narrower v0.2.2
static-only B3 reference (511,886.15 sim-s/s). The complete integrated point is stable below the
5% CV gate. Component-only points are diagnostic: static and car-car were stable, while the
ball-world and car-ball component samples exceeded 5% CV during this system-load snapshot; that
does not replace or weaken the stable complete-path performance gate.

## Evidence

Compact machine-readable evidence is in `results/v0.3/`. Large native trajectories and traces
remain ignored under `.tools/v0.3/`. See `docs/V0_3_ORACLE_CACHE.md` for custody and
`docs/REPRODUCING_V0_3.md` for commands.

## v0.4 boundary

Not implemented: demolition removal/disable/respawn, goals/scoring, kickoff and match resets,
terminal rules, RLGym observations/rewards/action parsing, PPO/training integration, Rival policy
inference, arbitrary body counts, other modes, or a generic Bullet API.
