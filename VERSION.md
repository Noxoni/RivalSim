# RivalSim Version Boundary

**Current version:** v0.1 — GPU physics proof

## Objective

Prove or disprove that a specialized batched GPU Rocket League simulator is worth pursuing for Rival training.

v0.1 is intentionally incomplete. It implements only enough 120 Hz car/ball state evolution to measure GPU scaling and validate basic aerial/jump/flip behavior against RocketSim and documented Rocket League values.

## Hard boundary

Do not add arena mesh collision, suspension, wheel driving, dynamic car-ball/car-car contact solving, boost pads, goals, demos, PPO, or RLGym integration until the v0.1 performance/parity gate is reported.

## Next version if v0.1 succeeds

**v0.2 — arena + ground-contact proof** should add:

- extracted DFH/Stadium_P collision mesh;
- precompiled triangle BVH or spatial acceleration structure uploaded once to GPU;
- wheel/suspension ray queries;
- car-world collision/contact;
- throttle/brake/coast/steering/powerslide ground behavior;
- parity/performance measurements against RocketSim.

Do not silently overwrite v0.1 evidence. Preserve all benchmark and parity results by version.
