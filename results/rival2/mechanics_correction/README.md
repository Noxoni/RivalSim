# RivalSim mechanical correction

`movement_mechanics_parity.json` is the compact open-loop parity gate against
the pinned RocketSim Octane/Soccar implementation. Each case receives identical
controls in RocketSim and RivalSim at 120 Hz. The gate covers ground drive,
steering, powerslide, braking/coast, ground and airborne boost, jump tap/hold,
double jump, directional and diagonal dodges, flip cancel, stall, airborne
pitch/yaw/roll control, supersonic state, and source auto-flip.

The correction places jump/dodge impulses and held-jump force into the resident
rigid body before chassis constraint solving, and evaluates flip/air-control/
auto-roll torque in RocketSim source order. It also preserves the source
auto-flip state and removes the duplicate post-integration supersonic update.

The JSON intentionally retains compact aggregate residuals and native mechanic
activation coverage rather than every captured state vector. The benchmark can
be rerun with `--include-captures` when operation-level evidence is needed.
