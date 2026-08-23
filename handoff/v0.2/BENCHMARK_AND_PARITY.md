# RivalSim v0.2 Benchmark and Parity Protocol

## Principle

v0.2 must measure the cost and fidelity of static-world physics independently before dynamic car/ball contacts are introduced.

Correctness comes first. Performance results are only meaningful for configurations that pass the required parity/state checks.

## Frozen references

v0.1 best stable GPU result:

- 131,072 worlds;
- 40,919,361.97 aggregate simulated game-seconds/s;
- 4,910,323,436 world physics ticks/s.

Existing full CPU RocketSim/RLGym training reference:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

The second number includes far more than v0.2 and remains a non-apples-to-apples system reference.

## Benchmark decomposition

Publish separate timings for these increasingly complete workloads using the same machine/runtime wherever practical:

### B0 — v0.1 regression

Existing contact-free fused transition.

Purpose: detect environment/runtime drift from the published v0.1 result.

### B1 — stadium suspension-ray query only

For every physics tick:

- 4 wheel-style rays × 2 cars × N worlds;
- shared DFH mesh;
- realistic ray origins/directions distributed across floor/ramp/wall/ceiling regions;
- store hit/no-hit, distance, triangle id and normal to GPU buffers.

Benchmark at least:

- Warp normal `Mesh` BVH;
- Warp `bvh_constructor="cubql"` ray backend if supported and correct on the pinned runtime.

Do not choose cuBQL merely because it is newer. Compare exact ray outputs and throughput.

Report rays/s in addition to world ticks/s.

### B2 — wheel/suspension/ground-force path

Execute:

- wheel transforms;
- four mesh rays/car;
- suspension calculations;
- wheel friction/engine/brake/steering/powerslide preparation and impulses;
- no chassis triangle collision except the wheel/suspension path.

This isolates the cost of Rocket League vehicle mechanics.

### B3 — full v0.2 static-world path

Execute the complete v0.2 tick including:

- B2;
- car OBB broadphase against arena mesh;
- candidate triangle iteration;
- narrow-phase contact generation;
- static-world contact solver/penetration correction;
- floor/ramp/wall/ceiling interactions.

This is the principal v0.2 throughput number.

## Control sequence

Do not benchmark an unrealistic constant-control trivial path.

Use a deterministic device-resident action tape that changes controls at a realistic cadence (default every four physics ticks unless another cadence is justified), containing a representative mixture of:

- full/partial forward throttle;
- reverse;
- steer left/right;
- coast;
- brake/reverse-transition;
- boost;
- handbrake/powerslide;
- jump/air controls for worlds that leave the ground.

Upload/generate the tape before timing. Index it entirely on GPU during the timed loop.

## State distribution for B3

Use a contact-rich deterministic mixture rather than spawning every car in empty air.

At minimum include worlds initialized in these families:

- flat floor, stationary;
- flat floor at multiple forward speeds;
- flat floor turning;
- braking/coasting;
- bottom ramp approaching side wall;
- side wall with tangent velocity;
- back wall;
- curved corner/ramp;
- ceiling contact/orientation;
- descending/landing onto floor;
- tilted two/three/four-wheel landings;
- nose/side/body collision with floor or wall;
- transient partial wheel contact.

Cars should not collide with each other. Place/disable them so dynamic contacts cannot contaminate the v0.2 result.

## Batch sweep

Do not assume the v0.1 optimum remains 131,072 worlds.

Start with:

`1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`

If the complete B3 workload clearly peaks earlier, stop increasing after at least one stable lower-throughput point beyond the peak.

If throughput is still materially rising at 131,072 and memory/resources permit, continue to 262,144.

Use adaptive selection based on measured throughput, not launch count.

## Timing protocol

For every principal point:

- warm GPU/kernel/mesh query paths before timing;
- use CUDA graph capture where it accurately represents the intended GPU-resident training path;
- synchronize around timing boundaries;
- use multiple measured repeats;
- record coefficient of variation;
- keep verification/readback outside timing;
- collect NVML/CPU telemetry in a way that does not perturb short timing intervals;
- record graph block size and any batching/tile choice;
- no timed H2D/D2H full-state traffic.

Target repeated-run CV <=5%. Explain any point above 5%; do not use an unstable point as the reported optimum.

## Required performance metrics

For B1/B2/B3 record:

- worlds;
- world physics ticks/s;
- aggregate simulated game-seconds/s (`world_ticks_per_second / 120`);
- suspension rays/s;
- average/maximum mesh triangles visited per chassis broadphase where measurable;
- average contact count per car/tick;
- narrow-phase candidates tested/s;
- solver iterations or contact passes if applicable;
- GPU utilization;
- VRAM current/peak;
- logical state bytes;
- arena geometry/BVH bytes where measurable;
- CPU process/system utilization;
- timed H2D/D2H bytes;
- NaN/error count;
- repeat CV;
- wall time.

Also report the multiplicative cost from B0 -> B1 -> B2 -> B3.

## Performance classification

These are decision aids, not permission to fake fidelity.

### Green

Complete B3 static-world path:

- parity passes;
- >=100,000 aggregate simulated game-seconds/s;
- stable scaling into at least thousands of worlds;
- hot loop GPU-resident.

This leaves enormous budget for v0.3+ and strongly supports continuing.

### Yellow

Complete B3:

- parity passes;
- 20,000 to <100,000 aggregate simulated game-seconds/s;
- profiling identifies understandable costs and no architectural dead end.

This still represents >=~100x the current 200.65 sim-s/s full CPU system reference before later stack overhead, so v0.3 may still be justified.

### Red / pause for redesign

Any of:

- required parity cannot be achieved without fundamentally wrong surface/contact behavior;
- complete B3 <20,000 sim-s/s with no clear optimization path;
- hot loop requires per-tick CPU round trips;
- scaling collapses at very small world counts because the architecture is inherently unsuitable;
- persistent numerical instability/contact explosions.

Do not automatically abandon a red result; stop at v0.2 and report the measured blocker before attempting v0.3.

## Parity methodology

As in v0.1, do not choose generous tolerances first.

1. Build deterministic scenarios.
2. Run measurement-only GPU vs RocketSim with tolerances absent.
3. Fix axis/state/oracle adapter defects.
4. Inspect error distributions at all horizons.
5. Freeze explicit tolerances.
6. Run a clean gate pass using only frozen tolerances.

Hard state/sign/contact errors are failures even if final position looks close.

## Geometry-query parity

Before vehicle dynamics, validate the stadium query layer independently.

Generate a large deterministic corpus of rays against the canonical mesh covering:

- vertical floor/ceiling rays;
- side/back wall normals;
- lower curved ramps;
- corners;
- goal mouth/posts/back net;
- near-edge/triangle-boundary cases;
- rays starting close to surfaces;
- misses.

Compare GPU query results against a CPU implementation over the exact same triangles and, where observable, RocketSim/static-world behavior.

Required comparisons:

- hit/miss exact;
- nearest-hit distance;
- hit point;
- face/triangle consistency where representations permit;
- normal orientation and direction;
- no false farther hit when a nearer triangle exists.

## Vehicle parity scenario families

The corpus must include at least the following deterministic families.

### Rest/settling

- Octane-compatible car dropped level onto flat floor;
- tilted roll/pitch landing;
- partial-wheel landing;
- rest height after settling;
- stationary long-run stability/no jitter explosion.

RLBot's published Octane rest height (~17.01 uu) is a sanity check; live RocketSim is the gate oracle.

### Straight ground motion

- throttle from rest;
- throttle from several initial speeds;
- reverse from rest;
- coast from speed;
- brake/opposing throttle from speed;
- ground boost from rest and nonzero speed;
- transition into/out of supersonic/max-speed cap.

Public sanity checks include ~1410 uu/s no-boost max driving speed, 2300 uu/s max car speed, ~-3500 uu/s² braking and ~-525 uu/s² coast deceleration, but use RocketSim source/live trajectories for parity.

### Steering

- full steer from rest;
- full steer at 500/1000/1500/2000+ uu/s;
- partial steer;
- left/right symmetry;
- throttle + steer trajectories;
- decelerating turn.

Compare position, yaw/orientation, forward/lateral velocity and wheel contacts over multiple seconds. RLBot's published curvature model is an independent sanity check, not the implementation oracle.

### Powerslide

- powerslide initiation at low/medium/high speed;
- steer + handbrake;
- release/recovery;
- forward/lateral velocity and yaw-rate response;
- handbrake rise/fall state if exposed/source-comparable.

### Ramp / wall / ceiling

- floor -> lower side ramp -> wall transition;
- wall driving parallel to floor;
- wall driving upward/downward;
- back-wall transition;
- curved corner traversal;
- wall -> ceiling transition where mechanically attainable;
- ceiling contact and departure;
- wheel-contact count through transitions.

### Chassis contact

- flat drop onto floor;
- nose-first floor impact;
- side/roof impact;
- wall impact at multiple angles/speeds;
- off-center impact producing angular response;
- shallow scraping contact;
- body contact while some wheels remain in contact.

## Parity horizons

Use at least:

`1, 4, 8, 30, 60, 120, 300, 600 ticks`

Not every scenario needs every horizon if the event terminates earlier, but long ground-driving/turning scenarios must include 300/600-tick comparisons.

## Required parity metrics

Per car:

- position;
- linear velocity;
- orientation;
- angular velocity;
- boost;
- wheel contact booleans/count;
- `isOnGround`;
- suspension length/contact distance where oracle/reference access permits;
- contact normal where reference access permits;
- handbrake state/value;
- supersonic state;
- jump/air state inherited from v0.1.

For trajectory-level tests also compare:

- path endpoint;
- path arc/turning radius or lateral deviation;
- time to target speed;
- settling height;
- contact transition timing;
- maximum penetration/instability observed.

## Stress tests

At minimum:

- 2,400 ticks of randomized contact-rich controls/states;
- no NaNs/Infs;
- no runaway penetration or energy explosion;
- bounded velocity/angular velocity;
- stable cars resting on floor;
- random floor/wall/ramp initial states;
- repeated deterministic run equality for the same build/seed.

## Required v0.1 regression checks

All v0.1 tests remain passing.

Re-run representative v0.1 live parity and contact-free GPU/CPU parity after v0.2 changes. A surface-physics implementation must not silently alter airborne mechanics.

## End-of-milestone verdict

Publish one of:

- `PASS_GREEN`;
- `PASS_YELLOW`;
- `PAUSE_RED`.

The report must explain the classification using both parity and complete B3 throughput.

Regardless of verdict, stop before v0.3. Do not implement ball-world, car-ball or car-car contact in this run.
