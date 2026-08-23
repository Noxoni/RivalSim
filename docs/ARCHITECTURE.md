# RivalSim v0.1 Architecture

## Design target

RivalSim is not a line-by-line CUDA translation of RocketSim or Bullet.

The target is a specialized batched Soccar 1v1 transition engine optimized for thousands of homogeneous worlds on one NVIDIA GPU.

Each world has exactly:

- car 0;
- car 1;
- one standard ball;
- fixed 120 Hz timestep;
- standard gravity/limits;
- later: one shared immutable DFH arena mesh and standard boost-pad layout.

This constraint is intentional. General-purpose game modes, arbitrary car counts and generic rigid-body scenes are out of scope unless Rival later needs them.

## v0.1 backend

Use NVIDIA Warp as the implementation layer for the proof.

Reasons:

- CUDA-backed kernels without committing immediately to a large native CUDA/C++ codebase;
- tensor/GPU interoperability with PyTorch;
- fast iteration on structure-of-arrays state layouts;
- straightforward benchmarking across many batch sizes.

If the proof succeeds, later versions may keep Warp or replace measured hot kernels with native CUDA C++.

## State layout

Do not create Python objects per environment.

Use GPU-resident structure-of-arrays or similarly coalesced tensors. Logical state includes at least:

### Per car, shape `[num_envs, 2, ...]`

- position xyz;
- linear velocity xyz;
- orientation quaternion or 3x3 basis;
- angular velocity xyz;
- boost;
- last controller input;
- on-ground / airborne state;
- jumped state;
- jump-hold elapsed;
- double-jumped state;
- dodged/flipping state;
- dodge elapsed;
- dodge direction;
- flip torque timer;
- supersonic state/timer where needed for parity;
- demolition fields may be reserved but are not implemented in v0.1.

### Ball, shape `[num_envs, ...]`

- position xyz;
- linear velocity xyz;
- angular velocity xyz;

The v0.1 ball does not collide with anything. It exists to prove general batched rigid-state integration and speed caps/drag scaffolding.

### Controller input

Batched continuous/discrete arrays for:

- throttle;
- steer;
- pitch;
- yaw;
- roll;
- jump;
- boost;
- handbrake.

Only airborne-relevant controls need to affect motion in v0.1. Steer/handbrake may be carried through as state but do not require ground behavior yet.

## Simulation tick

Fixed dt:

`1 / 120 s`

One v0.1 tick should logically perform:

1. clamp/normalize controls;
2. update jump/double-jump/dodge timers and edge-triggered jump state;
3. apply sticky force where applicable;
4. apply first-jump impulse and jump-hold bonus;
5. apply double-jump or dodge/flip impulse/torque;
6. apply airborne pitch/yaw/roll torque;
7. apply airborne throttle acceleration;
8. apply boost acceleration and consume boost;
9. apply gravity;
10. integrate linear/angular velocity and orientation;
11. clamp car velocity and angular velocity;
12. integrate ball with gravity/drag scaffolding and speed limits;
13. update previous controls/state timers.

Kernel fusion is allowed and encouraged after correctness. Do not optimize away inspectability before parity tests exist.

## CPU/GPU transfer rule

The benchmark must not copy full world state to CPU every tick.

Allowed CPU traffic:

- initial state/control upload;
- occasional sampled parity snapshots;
- aggregate timing/counter readback;
- final state readback.

The hot loop stays GPU resident.

## Later arena architecture

If v0.1 passes, v0.2 should use the standard DFH/Stadium_P collision geometry as one shared immutable GPU asset.

Preferred order of investigation:

1. use existing extracted collision binaries when provenance/licensing permits;
2. otherwise follow RLBot's documented map-mesh extraction path;
3. compile all collision triangles into a GPU-friendly BVH or fixed spatial grid;
4. compare against `rl_ball_sym`, which already demonstrates mesh + triangle + BVH processing for Rocket League collision data;
5. use the acceleration structure for wheel rays and car/ball world contacts.

Do not use an SDF as the first fidelity implementation because wall/ceiling mechanics and suspension need accurate surface normals/contact locations.

## Relationship to RocketSim

RocketSim remains the reference implementation.

Study and compare against RocketSim's:

- `Car::_PreTickUpdate` / `_PostTickUpdate` / `_FinishPhysicsTick`;
- jump, flip, air torque and boost logic;
- state transitions;
- velocity limits;
- later: vehicle suspension and Arena contact callbacks.

Do not attempt to port Bullet wholesale in v0.1.

## Determinism

Given identical GPU initial state, controller sequence, seed and build, RivalSim should be reproducible within the chosen floating-point mode.

Benchmark both normal FP32 and any fast-math option separately. Fidelity results must use the configuration intended for training.
