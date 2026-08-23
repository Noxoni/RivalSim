# RivalSim Source References

These are the primary references for RivalSim v0.1 and later fidelity work.

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
- angular accelerations and velocity limits;
- ground turn-radius references.

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

### RLBot map mesh extraction

`https://wiki.rlbot.org/v5/miscellaneous/extracting-map-meshes/`

The documented DFH/Stadium_P collision assets include:

- `Goal_STD_Collision.pskx`;
- `Field_STD_Collision_Corner.pskx`;
- `Field_STD_Collision_SideTop.pskx`;
- `Field_STD_Collision_SideBot.pskx`.

The tutorial documents exporting triangle vertex/index binaries suitable for language-independent use.

### rl_ball_sym

`https://github.com/VirxEC/rl_ball_sym`

This Rust project is explicitly referenced by the RLBot mesh-extraction documentation as an example of:

- reading collision mesh data;
- compiling triangle geometry;
- BVH/triangle acceleration structures;
- Rocket League ball/field simulation.

Relevant source directory:

`src/simulation/`

including `bvh.rs`, `tri_bvh.rs`, `tri_grid.rs`, `mesh.rs`, `geometry.rs`, `field.rs`, and `ball.rs`.

Use as a design reference; respect its license and do not copy code without verifying redistribution obligations.

## RocketSim CPU oracle

`https://github.com/ZealanL/RocketSim`

RocketSim remains the primary CPU implementation oracle.

Relevant areas:

- `src/Sim/Car/Car.cpp` — state sync, pre/post tick, jump, flip, air torque, boost, velocity limits, Bullet vehicle setup;
- `src/Sim/Arena/Arena.cpp` — world stepping, rigid bodies, contact callbacks, car-ball/car-car/world interactions;
- `src/RLConst.h` and related constants;
- custom `btVehicleRL` implementation;
- bundled/modified Bullet 3.24 code.

RocketSim is not being replaced or modified in v0.1. Use it to generate reference trajectories.

Any RocketSim-derived code incorporated into RivalSim must retain applicable MIT copyright/license notices.

## Rendering / visual debug

`https://wiki.rlbot.org/v5/botmaking/rendering/`

Rendering is not part of the GPU benchmark path. Later transfer/fidelity debugging may use RLBot 3D lines/poly-lines to compare predicted and observed trajectories in Rocket League.

## Source policy

For numeric/mechanical truth, use this order:

1. exact RocketSim implementation used by the CPU oracle;
2. current live RLGym/RocketSim behavior;
3. RLBot v5 public values/packets;
4. empirical Rocket League measurements.

When sources disagree, commit the discrepancy and measured resolution instead of silently normalizing it away.
