# RivalSim Source References

These are the primary references for RivalSim fidelity work.

## Public Rocket League values

### RLGym game values

`https://rlgym.org/Cheatsheets/game_values/`

Useful public constants include:

- 120 physics ticks/s;
- field dimensions;
- boost consumption/acceleration;
- gravity;
- car/ball masses;
- speed limits;
- double-jump/flip timing.

### RLBot useful game values

`https://wiki.rlbot.org/v5/botmaking/useful-game-values/`

Useful details include:

- coordinate conventions;
- field/goal dimensions;
- boost-pad positions and pickup geometry;
- rest heights;
- ball restitution/drag;
- context-specific ground/air boost acceleration;
- air throttle acceleration;
- braking/coasting values;
- angular accelerations and velocity limits;
- ground turn-radius/curvature references.

For v0.2 these values are independent sanity checks. RocketSim source/live behavior remains the implementation oracle where it differs from public approximations.

### RLBot jumping physics

`https://wiki.rlbot.org/v5/botmaking/jumping-physics/`

Useful details include:

- gravity;
- sticky force;
- jump impulse;
- jump-hold bonus;
- air throttle;
- 120 Hz example simulation.

### RLBot v5 migration / packet state

`https://github.com/RLBot/python-interface/wiki/Migration`

Useful live-validation fields include:

- `air_state`;
- `dodge_timeout`;
- `has_dodged`;
- `dodge_elapsed`;
- `dodge_dir`;
- `last_input`;
- `latest_touch`.

## Arena collision geometry

### Measured v0.2 asset custody

The completed v0.2 run used the exact local files consumed by the RocketSim Soccar oracle:

- repository: `Noxoni/Rival`;
- source revision: `36cb14cf645c4f06b668c34d85ce1a500e4b53da`;
- asset-introducing revision: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`;
- source-relative path: `bot/collision_meshes/soccar/mesh_0.cmf` through `mesh_15.cmf`;
- deterministic combined content SHA-256:
  `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`;
- combined geometry: 4,468 vertices and 8,020 triangles;
- bounds in Unreal units: approximately `[-4107.334, -5999.995, -13.26779]` through
  `[4107.334, 5999.995, 2075.4521]`.

The Rival checkout had unrelated local modifications, so it was treated as read-only input.
RivalSim records every file's exact byte size, SHA-256, RocketSim internal hash, counts, and
bounds in `results/v0.2/manifest.json`; it does not track the files themselves.

### RocketSim collision asset format

Pinned primary source:

`https://github.com/ZealanL/RocketSim/tree/c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

Relevant file:

`src/CollisionMeshFile/CollisionMeshFile.h`

RocketSim expects collision mesh files under `./collision_meshes/` with `.cmf` extension. Its `CollisionMeshFile` representation contains indexed float3 vertices and triangles and is based on `ZealanL/RLArenaCollisionDumper`.

For v0.2, prefer the exact local `.cmf` files consumed by the RocketSim oracle when available. Record hashes/provenance and parse them without modifying the oracle copy.

### RLArenaCollisionDumper

`https://github.com/ZealanL/RLArenaCollisionDumper`

RocketSim's own README directs users to this project to dump Rocket League arena collision meshes. Use it when the oracle's existing `.cmf` assets are unavailable or their provenance cannot be established.

Extracted Rocket League collision data is local input, not repository source. Do not commit raw or repackaged arena assets to RivalSim.

### RLBot map mesh extraction

`https://wiki.rlbot.org/v5/miscellaneous/extracting-map-meshes/`

The documented DFH/Stadium_P collision assets include:

- `Goal_STD_Collision.pskx`;
- `Field_STD_Collision_Corner.pskx`;
- `Field_STD_Collision_SideTop.pskx`;
- `Field_STD_Collision_SideBot.pskx`.

The tutorial documents exporting triangle vertex/index binaries suitable for language-independent use. This is a fallback extraction path and an independent check on the geometry used by RocketSim.

### rl_ball_sym

`https://github.com/VirxEC/rl_ball_sym`

The RLBot mesh-extraction documentation references this Rust project as an example of:

- reading collision mesh data;
- compiling triangle geometry;
- BVH/triangle acceleration structures;
- Rocket League ball/field simulation.

Relevant source directory:

`src/simulation/`

including `bvh.rs`, `tri_bvh.rs`, `tri_grid.rs`, `mesh.rs`, `geometry.rs`, `field.rs`, and `ball.rs`.

Use as a design reference; respect its license and do not copy code without verifying redistribution obligations.

## NVIDIA Warp GPU geometry

Pinned v0.1/v0.2 runtime: NVIDIA Warp `1.16.0`.

Documentation:

- `https://nvidia.github.io/warp/v1.16/user_guide/runtime.html`
- `https://nvidia.github.io/warp/v1.16/language_reference/builtins.html`

Relevant v0.2 capabilities:

- `warp.Mesh` stores triangle geometry on-device and maintains an acceleration structure;
- `wp.mesh_query_ray()` returns nearest ray hit distance/face/normal/barycentrics;
- `wp.mesh_query_aabb()` iterates triangle candidates overlapping a box, useful for chassis broadphase;
- Warp 1.16 supports tile variants of mesh/BVH AABB queries;
- Warp 1.16 includes an experimental `bvh_constructor="cubql"` backend for `warp.Mesh` ray queries. In this version cuBQL mesh support is ray-focused, so a normal mesh/BVH may still be required for chassis AABB queries.

The completed query gate compared both backends over the same 4,608-ray corpus. Each had zero
hit/miss mismatches, 0.001953125 uu maximum distance/hit-point error, and zero unambiguous face
mismatches against the independent CPU reference. cuBQL was selected for suspension rays
because its best B1 result was 3.426 billion rays/s versus 2.325 billion rays/s for the normal
backend. The normal mesh remains the chassis AABB backend.

## RocketSim CPU oracle

`https://github.com/ZealanL/RocketSim`

Pinned v0.1/v0.2 source reference unless intentionally updated and revalidated:

`c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

RocketSim remains the primary CPU implementation oracle. The completed v0.2 environment used
`rocketsim==2.2.1`, binding source commit
`2da51b1dac7b8127127613a5ff30e490bdd70dd8`, and installed `RocketSim.pyd` SHA-256
`E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`.

Relevant areas:

- `src/Sim/Car/Car.cpp` — state sync, pre/post tick, wheel controls, jump/flip/air/boost logic, velocity limits, Bullet vehicle setup;
- `src/Sim/btVehicleRL/btVehicleRL.cpp` — wheel transforms, suspension raycasts, suspension forces, friction impulses;
- `src/Sim/btVehicleRL/btVehicleRL.h` — vehicle/wheel data and API;
- `src/Sim/Arena/Arena.cpp` — world stepping, rigid bodies, contact callbacks and car-world handling;
- `src/RLConst.h` and related constants;
- bundled/modified Bullet 3.24 contact/constraint code actually reached by those paths.

RocketSim's vehicle path performs wheel transforms/raycast/friction preparation before the world solve, then suspension/friction impulses in its second vehicle phase. RivalSim v0.2 should preserve the behavior/order needed for trajectory parity rather than implementing a generic vehicle model.

Any RocketSim-derived code incorporated into RivalSim must retain applicable MIT copyright/license notices. Bullet-derived code/logic must retain applicable zlib notices where required.

## Rendering / visual debug

`https://wiki.rlbot.org/v5/botmaking/rendering/`

Rendering is not part of the GPU benchmark path. Later transfer/fidelity debugging may use RLBot 3D lines/poly-lines to compare predicted and observed trajectories in Rocket League.

## Source policy

For numeric/mechanical truth, use this order:

1. exact RocketSim implementation used by the CPU oracle;
2. exact collision geometry used by that oracle;
3. current live RLGym/RocketSim behavior;
4. RLBot/RLGym public values and packets;
5. empirical Rocket League measurements.

When sources disagree, commit the discrepancy and measured resolution instead of silently normalizing it away.
