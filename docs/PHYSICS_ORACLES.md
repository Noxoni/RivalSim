# RivalSim Physics Oracles

These values are validation targets and implementation references for v0.1. Prefer RocketSim source and live RLBot/RLGym values when sources disagree; record discrepancies rather than silently choosing one.

## Units and timestep

- Unreal units: `1 uu = 1 cm`.
- Physics timestep: `120 Hz`, `dt = 1/120 s`.
- Coordinate convention: X/Y field plane, Z up.

## Arena reference values for later milestones

- side walls: `x = ±4096`;
- back walls: `y = ±5120`;
- ceiling: `z = 2044`;
- goal height: `642.775`;
- goal center-to-post: `892.755`;
- goal depth: about `880`.

These are not collision geometry substitutes. v0.2 should use extracted collision triangles.

## Basic physics

Use these as initial public-reference values and verify against RocketSim constants/source:

- gravity: `650 uu/s²` downward;
- car mass: `180`;
- ball mass: `30`;
- car max speed: `2300 uu/s`;
- car max angular velocity: `5.5 rad/s`;
- ball radius: `91.25 uu`;
- ball max speed: `6000 uu/s`;
- ball max angular velocity: about `6 rad/s`;
- normal boost consumption: about `33.3 boost/s`.

Public references distinguish boost acceleration by context:

- ground boost: about `991.666 uu/s²`;
- air boost: about `1058.333 uu/s²` in RLBot's useful-values page;
- RLGym cheat sheet lists `991.666` as a general boost acceleration value.

Do not hide this discrepancy. For v0.1 airborne parity, use RocketSim's actual boost implementation as the authoritative CPU oracle and report the effective value observed.

## Air throttle

- forward airborne throttle acceleration: approximately `66.667 uu/s²`;
- reverse airborne throttle acceleration: approximately `33.334 uu/s²` backward.

## Jump physics

Public RLBot jump reference:

- first-jump impulse: about `292 uu/s` in relative roof direction;
- sticky force: `325 uu/s²` toward the contacted surface for the first three ticks after jump;
- jump hold bonus: up to another `292 uu/s` over `0.2 s`;
- hold bonus acceleration: about `1460 uu/s²` while active;
- minimum jump-hold effect: first three ticks;
- double-jump impulse: approximately `291.667–292 uu/s` in roof direction;
- maximum double-jump/dodge delay: `1.25 s`.

RLBot v5 exposes useful validation state:

- `air_state` values including OnGround, Jumping, DoubleJumping, Dodging and InAir;
- `dodge_timeout`;
- `has_dodged`;
- `dodge_elapsed`;
- `dodge_dir`;
- `last_input`.

Use these later for live Rocket League spot checks.

## Air rotational dynamics

Public useful-values references list maximum car angular acceleration approximately:

- yaw: `9.11 rad/s²`;
- pitch: `12.46 rad/s²`;
- roll: `38.34 rad/s²`;

Do not assume a constant-torque model exactly matches Rocket League. RocketSim's `_UpdateAirTorque` implementation is the primary v0.1 CPU oracle.

## Car orientation integration

Implement a numerically stable quaternion or rotation-matrix integration from angular velocity. Preserve Rocket League/RocketSim axis/sign conventions.

Parity tests should include isolated pitch, yaw, roll and combined rotations under controller inputs.

## Ball in v0.1

The ball is only a free rigid body in v0.1.

Implement:

- gravity;
- linear integration;
- angular integration;
- speed/angular-speed clamp;
- optional drag scaffold matching RocketSim's ball damping model if straightforward.

No arena/car contacts are required until later.

## Source hierarchy

When implementing or resolving a value:

1. RocketSim source/constants used by the CPU oracle;
2. current RLGym/RocketSim exposed constants;
3. RLBot v5 game packet / useful-values documentation;
4. empirical measurement if required.

Record source and any mismatch in the parity report.
