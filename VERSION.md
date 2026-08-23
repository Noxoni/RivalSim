# RivalSim Version Boundary

**Current version:** v0.1.0 — implemented GPU physics proof; continuation gate passed

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

## Measured boundary

The v0.1 proof is complete. The frozen result is documented in `docs/V0_1_RESULTS.md` and
`results/v0.1/`. It establishes contact-free throughput headroom and bounded RocketSim parity;
it does not establish arena, ground, suspension or dynamic-contact fidelity.

The next version remains only a recommendation pending review. No v0.2 code belongs in the
v0.1 release commits.
