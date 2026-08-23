# RivalSim v0.2 — Arena + Ground-Contact Proof

This package authorizes the next bounded RivalSim research milestone after the successful v0.1 GPU physics proof.

## Starting point

Frozen v0.1 result commit:

`1f7a36cc6165273fb658ba07a8458e8d8e60628a`

v0.1 proved:

- 131,072 contact-free worlds at the best stable point;
- 40,919,361.97 aggregate simulated game-seconds/s;
- 3,678.02x best GPU vs same-equation CPU throughput;
- zero timed H2D/D2H traffic in the benchmark hot loop;
- 27/27 RocketSim parity scenarios passed at 1/4/8/30/60/120 ticks.

Those results are frozen under `results/v0.1/` and must not be rewritten.

## v0.2 question

> How much of RivalSim's enormous GPU headroom survives when every world uses the real DFH/Stadium_P collision geometry, four wheel/suspension queries per car, RocketSim-derived wheel/ground forces, and faithful car-world contact?

This milestone should answer that with working code and measured evidence.

## Scope

v0.2 adds only the static-world layer required for real driving and surface mechanics:

- one shared immutable DFH/Stadium_P collision mesh on the GPU;
- GPU acceleration structure over the stadium triangles;
- wheel raycasts and contact normals/distances;
- RocketSim-derived suspension;
- wheel friction/engine/brake/coast forces;
- steering and powerslide behavior;
- car chassis vs static arena contact;
- floor/ramp/wall/ceiling driving/contact;
- RocketSim parity corpus for those mechanics;
- contact-rich GPU throughput scaling benchmark.

The free ball from v0.1 may continue to exist, but **ball-world contact remains out of scope**.

## Explicitly out of scope

Do not implement in v0.2:

- ball-world collision;
- car-ball collision;
- car-car collision;
- bumps/demolitions;
- boost-pad pickup/recharge;
- goals/scoring/respawns;
- RLGym environment integration;
- observations/rewards/PPO;
- Rival policy inference;
- rendering in the benchmark hot path;
- Hoops, Dropshot, Snowday, Heatseeker, Rumble or arbitrary arenas.

## Core implementation decision

Keep NVIDIA Warp as the primary implementation layer unless profiling proves it is the blocker.

For the static stadium asset:

1. Prefer the exact collision mesh files already used by the local RocketSim oracle when available and provenance can be recorded.
2. Otherwise use `ZealanL/RLArenaCollisionDumper` or the RLBot documented DFH extraction path.
3. Never commit extracted Rocket League collision assets to this public repository. Keep them ignored locally and commit only provenance, hashes, mesh statistics and reproduction instructions.
4. Upload one canonical stadium mesh to the GPU and share it across every simulated world.

Use `warp.Mesh` as the first acceleration implementation. Benchmark its normal BVH ray path and, where supported by the installed Warp 1.16.x runtime, the `bvh_constructor="cubql"` ray backend for suspension queries. Do not assume cuBQL wins; select only from measured correctness + throughput. A separate normal Warp mesh/BVH may be retained for AABB candidate queries if needed for chassis contact.

## Execution order

v0.2 should proceed in three internal gates:

### Gate A — stadium geometry + query engine

- load/validate the canonical stadium collision triangles;
- build GPU mesh acceleration structure(s);
- implement batched wheel-style ray queries;
- establish ray hit point/normal/distance correctness;
- benchmark query throughput independently.

### Gate B — wheels, suspension and ground driving

- reproduce RocketSim `btVehicleRL` wheel transforms/rays;
- suspension length/relative velocity/clipped contact factor;
- suspension impulses;
- wheel friction impulses;
- engine/brake/coast;
- steer-angle behavior;
- handbrake/powerslide;
- ground boost/throttle interaction;
- wheel-contact state.

### Gate C — chassis vs static world

- broadphase candidate-triangle query for the oriented car hitbox;
- narrow-phase contact point/normal/penetration;
- RocketSim/Bullet-compatible impulse/friction/restitution behavior sufficient for transfer fidelity;
- floor/ramp/wall/ceiling transitions and non-wheel body contacts.

Do not skip Gate A/B benchmarking just because Gate C is more interesting. We want to know exactly where throughput is spent.

## Performance philosophy

v0.1 has enough headroom that v0.2 should spend aggressively on fidelity rather than cutting important mechanics for speed.

The current full CPU RocketSim/RLGym reference remains:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

v0.2 is not yet a full training stack, so comparisons to 200.65 sim-s/s remain non-apples-to-apples. The relevant result is how much throughput remains after authentic static-world physics is added.

See:

- `handoff/v0.2/V0_2_SPEC.md`
- `handoff/v0.2/BENCHMARK_AND_PARITY.md`
- `handoff/v0.2/CODEX_START_PROMPT.md`
