# RivalSim v0.2 Results

## Verdict: `PAUSE_RED`

RivalSim v0.2 is implemented and measured through its prescribed boundary. The static-world
GPU architecture has exceptional performance headroom, but the required RocketSim fidelity
gate fails. Under the governing rule that correctness comes before throughput, v0.2 therefore
ends at `PAUSE_RED` and v0.3 is not authorized.

The two sides of that result are deliberately reported separately:

- performance: green-threshold headroom, with a stable B3 peak of **1,350,748.16 aggregate
  simulated game-seconds/s** at 262,144 worlds;
- parity: failed, with **85 scenario/horizon records containing hard state, contact, or sign
  mismatches** and **617 numeric tolerance failures** across 35 scenarios and eight horizons.

This is a partial static-world transition benchmark, not a complete Rocket League simulator
and not an end-to-end training speedup claim.

## Frozen boundaries

- v0.2 start authority: `7a6a6913fad6ceedd92d1170b373a0978edb05b6`;
- frozen v0.1 result: `1f7a36cc6165273fb658ba07a8458e8d8e60628a`;
- completed v0.2 implementation: `f236310` (full SHA in `results/v0.2/manifest.json`);
- frozen v0.1 evidence was not modified (`git diff` from the v0.1 boundary is empty for
  `results/v0.1/`).

## What v0.2 implements

- strict RocketSim `.cmf` parsing with count, size, index, finite-value, SHA-256, and internal
  RocketSim mesh-hash validation;
- one combined 4,468-vertex / 8,020-triangle Soccar geometry shared by all worlds;
- a normal Warp mesh/BVH for `mesh_query_aabb` chassis broadphase;
- a measured cuBQL Warp mesh for suspension rays;
- four explicit wheel states per Octane-compatible car, including transforms, hit state,
  distance, face, point, normal, suspension length/velocity/clipped factor/force/pushback,
  engine/brake/steer, friction, and world-contact flags;
- RocketSim-derived suspension, throttle, reverse, coast, brake, steering curves, boost-ground
  interaction, handbrake rise/fall, and powerslide friction preparation;
- conservative oriented-box AABB construction, Warp triangle candidate iteration, full
  triangle-vs-OBB SAT narrow phase, up to four contacts per car, friction/restitution impulses,
  positional correction, and off-center angular response;
- a deterministic, device-resident 64-entry action tape changing every four ticks;
- decomposed B0/B1/B2/B3 CUDA-graph benchmarks and contact-rich stress checks.

RocketSim's Soccar floor, ceiling, and side-plane shapes are handled analytically alongside the
exact `.cmf` triangles, matching the source arena construction. No procedural substitute was
used for the curved arena geometry.

## Collision asset custody

The exact local files consumed by the live oracle came from `Noxoni/Rival`, source commit
`36cb14cf645c4f06b668c34d85ce1a500e4b53da`, under
`bot/collision_meshes/soccar/`. They were introduced there by commit
`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`. RivalSim commits no `.cmf`, extracted game asset,
or repackaged geometry.

The combined geometry bounds are approximately
`[-4107.334, -5999.995, -13.26779]` to `[4107.334, 5999.995, 2075.4521]` uu. Its deterministic
combined content SHA-256 is
`2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`.

| File | Bytes | Vertices | Triangles | SHA-256 | RocketSim hash |
|---|---:|---:|---:|---|---|
| `mesh_0.cmf` | 16,364 | 483 | 880 | `8764D43B...B511C4E` | `A160BAF9` |
| `mesh_1.cmf` | 16,364 | 483 | 880 | `9D42E5DB...73C993` | `2811EEE8` |
| `mesh_2.cmf` | 16,364 | 483 | 880 | `762EC145...91EEE` | `14B84668` |
| `mesh_3.cmf` | 16,364 | 483 | 880 | `0DB5C559...8DB5C` | `EC759EBF` |
| `mesh_4.cmf` | 18,236 | 536 | 983 | `92C3224D...596566A` | `94FB0D5C` |
| `mesh_5.cmf` | 18,236 | 536 | 983 | `776C24B1...C3941A3` | `DEA07102` |
| `mesh_6.cmf` | 18,236 | 536 | 983 | `A2262BFA...33804E6` | `BD4FBEA8` |
| `mesh_7.cmf` | 18,236 | 536 | 983 | `C887DF02...19A06F3` | `39A47F63` |
| `mesh_8.cmf` | 416 | 18 | 16 | `33505670...7A21E9` | `3D79D25D` |
| `mesh_9.cmf` | 416 | 18 | 16 | `4864911A...8A6CE8` | `D84C7A68` |
| `mesh_10.cmf` | 2,480 | 80 | 126 | `36CC44E1...01AF3D` | `B81AC8B9` |
| `mesh_11.cmf` | 2,480 | 80 | 126 | `7F4468BE...FB6CE3` | `760358D3` |
| `mesh_12.cmf` | 2,480 | 80 | 126 | `952A90A0...805118` | `73AE4940` |
| `mesh_13.cmf` | 2,480 | 80 | 126 | `B441F782...631750` | `918F4A4E` |
| `mesh_14.cmf` | 416 | 18 | 16 | `085F7533...5D6AE8` | `1F8EE550` |
| `mesh_15.cmf` | 416 | 18 | 16 | `21C1D1EB...FAA2E4A` | `255BA8C1` |

Full, unabridged SHA-256 values, per-file bounds, and counts are in
`results/v0.2/benchmark.json` and `results/v0.2/manifest.json`.

## Geometry-query gate

The independent CPU reference uses two-sided Möller-Trumbore intersection over the exact
stored triangles plus RocketSim's four Soccar static planes. A 4,608-ray deterministic corpus
contains 512 rays in each of nine families: floor, ceiling, side walls, ramps/curves, goals/back
wall, corners, shared boundaries, near-surface starts, and misses.

Both Warp backends produced:

- 0 hit/miss mismatches;
- maximum nearest-distance and hit-point error of 0.001953125 uu;
- minimum unambiguous normal dot product of 0.99999988;
- 0 unambiguous face mismatches;
- one explicitly recorded co-nearest ceiling/side-plane tie at an exact boundary corner.

cuBQL was selected for wheel rays only after passing that comparison. The normal mesh remains
the chassis AABB backend because Warp 1.16 does not provide the required AABB path through
cuBQL.

## Performance gate

Hardware/runtime: RTX 5090, Ryzen 7 9800X3D, Python 3.14.3, Warp 1.16.0, NumPy 2.5.2,
CUDA toolkit 12.9 bundled with Warp, and NVIDIA driver supporting CUDA 13.3.

Every mandatory point from 1,024 through 131,072 worlds was measured with five repeats. The
adaptive extension reached 262,144 worlds where the endpoint was still rising. Identical
device checkpoints were restored between repeats so CV measures timing variance rather than a
different contact workload. All 44 recorded points have CV below 5%; the maximum is 4.965%.

| Variant | Best stable worlds | World ticks/s | Aggregate sim-s/s | Rays/s | CV |
|---|---:|---:|---:|---:|---:|
| B0 contact-free v0.1 regression | 131,072 | 2,900,224,036.46 | 24,168,533.64 | — | 0.997% |
| B1 normal BVH ray-only | 262,144 | 290,614,453.62 | 2,421,787.11 | 2,324,915,628.94 | 2.496% |
| B1 cuBQL ray-only | 262,144 | 428,285,196.57 | 3,569,043.30 | 3,426,281,572.57 | 2.656% |
| B2 wheels/suspension/forces | 131,072 | 231,931,107.67 | 1,932,759.23 | 1,855,448,861.36 | 0.969% |
| B3 complete static world | 262,144 | **162,089,778.61** | **1,350,748.16** | **1,296,718,228.90** | **0.998%** |

At the best B3 point:

- average 0.4902 broadphase candidates and 0.1704 contacts per car-tick;
- maximum 111 candidates and four contacts per car-tick;
- 31.64 million narrow-phase candidates/s;
- maximum measured penetration in that repeat: 11.153 uu;
- 97.14% mean / 99% maximum sampled GPU utilization;
- 5,796,769,792-byte peak sampled VRAM use;
- 403,701,760 logical state bytes for the post-accumulator layout;
- zero timed H2D and D2H bytes;
- zero NaN/error count.

B3 costs 7.0003× the common B0 rate. It remains more than 13× above the 100,000 sim-s/s green
performance threshold, but performance cannot override failed fidelity.

## Parity gate

The required measurement-only pass was run before tolerance selection. Its local artifact was
170,872 bytes with SHA-256
`7EB62CF97BE25EA5F7CF6540D9D6350829B0AE7887B09F0B63E3915E937B9BDF`. The compact committed
gate evidence retains the aggregate distribution and the frozen tolerance table.

Frozen tolerances:

| Metric | Tolerance |
|---|---:|
| position | 10.0 uu |
| linear velocity | 25.0 uu/s |
| orientation | 0.025 rad |
| angular velocity | 0.1 rad/s |
| boost | 0.01 |
| handbrake value | 0.0001 |
| world-contact normal | 0.05 rad |

These values were not widened to encompass divergent tails. Contact, sign, and discrete-state
mismatches remain unconditional failures.

The clean gate covers 35 scenarios in eight families at 1, 4, 8, 30, 60, 120, 300, and 600
ticks. The families cover settling/rest, level/tilted/partial landings, forward/reverse/coast/
brake, boost, steering at multiple speeds and inputs, powerslide initiation/hold/release,
ramps/walls/corners/ceiling, and nose/side/roof/off-center/scraping body contacts.

Observed aggregate maxima include 5,061.04 uu position error, 2,510.62 uu/s velocity error,
3.1348 rad orientation error, and 6.3243 rad/s angular-velocity error. Hard mismatch-frame
counts include:

- `on_ground`: 69;
- wheel contacts 0/1/2/3: 53 / 57 / 47 / 57;
- chassis world contact: 44;
- linear-velocity direction: 25.

The primary blocker is the approximate wheel-friction/steering and contact response: it can
track simple short-horizon settling and straight motion, but medium/high-speed turning and
surface transitions diverge enough to lose wheel/contact states and sometimes velocity
direction. Replacing those approximations with a more Bullet-equivalent constraint/friction
solve is a future redesign question, not work authorized under this v0.2 run.

## Stress and determinism

Two independent 64-world, 2,400-tick contact-rich runs produced the identical full-state
SHA-256 `35C859A3274DDE71ADF2507B87974FA88B1E308A34CF0D74883D1C0CB23CC03D`.

- all car state remained finite;
- maximum final linear speed: 509.60 uu/s;
- maximum final angular speed: 2.9619 rad/s;
- maximum accumulated penetration: 35.486 uu;
- floor-rest height after the long run: 17.3415 uu;
- floor-rest speed: 1.3918 uu/s;
- zero hot-loop H2D/D2H traffic.

The stress gate passes. The milestone verdict remains red solely because required parity does
not pass.

## Explicit v0.2 boundary

No v0.3 feature was begun. RivalSim still has no ball-world, car-ball, car-car, boost-pad,
scoring/game-rule, RLGym/PPO, or Rival-policy implementation. Review and new authority are
required before any such work.
