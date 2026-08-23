# RivalSim Research Roadmap

## v0.1 — GPU physics proof: complete / PASS

Purpose: determine whether the GPU-native architecture has enough performance headroom to justify the project.

Scope:

- GPU-resident batched state;
- 120 Hz free rigid-body integration;
- jump / jump hold / double jump / dodge timing;
- airborne torque;
- airborne throttle;
- boost acceleration/consumption;
- speed limits;
- free ball integration;
- CPU RocketSim parity harness;
- batch scaling benchmark.

No arena collision. Frozen result boundary:
`1f7a36cc6165273fb658ba07a8458e8d8e60628a`.

## v0.2 — Arena + ground-contact proof: complete / `PAUSE_RED`

v0.1 passed, so this bounded milestone was implemented and measured.

Add:

- DFH/Stadium_P collision geometry;
- precompiled GPU triangle BVH or spatial grid;
- wheel/suspension ray queries;
- car-world collision;
- wheel contact state;
- ground throttle/brake/coast;
- steering/curvature;
- powerslide;
- wall/ceiling driving;
- GPU vs RocketSim ground/wall parity.

The implementation used the exact external RocketSim Soccar `.cmf` set as one shared Warp
geometry, with a normal mesh for chassis AABB queries and a measured cuBQL mesh for suspension
rays. B3 reached 1,350,748.16 aggregate simulated game-seconds/s at 262,144 worlds, but the
35-scenario RocketSim parity gate failed. The final classification is therefore `PAUSE_RED`.

See `docs/V0_2_RESULTS.md` and `results/v0.2/` for the frozen outcome. A static-world
wheel-friction/contact-solver redesign requires review and new authority.

## v0.3 — Ball and dynamic contacts: not begun / blocked at v0.2 gate

Do not begin this milestone while the v0.2 `PAUSE_RED` boundary is in force.

Add:

- ball-world contact;
- ball friction/restitution/drag;
- car-ball contact and Rocket League-specific hit behavior;
- car-car contact;
- bumps/demolitions where required;
- dynamic-contact solver validation.

This is likely the hardest fidelity milestone.

## v0.4 — Complete standard 1v1 game transition

Add:

- 34 boost pads and recharge timers;
- boost pickup geometry;
- kickoff/respawn positions;
- goals/scoring;
- demolish/respawn rules;
- match/reset state;
- terminal/truncation events needed by RLGym.

At this point RivalSim should be capable of headless standard Soccar 1v1 episodes.

## v0.5 — Tensor-native training integration

Do not recreate RLGym's Python-object hot path.

Add GPU-native:

- observation building;
- action parsing;
- reward computation;
- episode reset/mutator sampling;
- rollout-buffer writes;
- direct PyTorch tensor interoperability.

Target hot path:

`GPU RivalSim -> GPU obs -> GPU policy -> GPU actions -> GPU RivalSim -> GPU rewards/buffer`

Only metrics/checkpoints/logging should routinely leave the device.

## v0.6 — Transfer gate

Train a bounded policy using RivalSim, then deploy it into CPU RocketSim/RLGym and RLBot/Rocket League.

RivalSim does not earn production use merely by matching its own simulator.

Required transfer checks should include:

- basic driving/jump/aerial execution;
- wall movement;
- recoveries;
- ball touches;
- learned mechanics;
- fixed-policy evaluation against RocketSim-trained baselines.

If a policy exploits RivalSim-only physics and fails transfer, fidelity work takes precedence over more throughput.

## Explicit non-goals until needed

- Hoops;
- Dropshot;
- Snowday;
- Heatseeker;
- Rumble;
- arbitrary numbers of cars/balls;
- every Rocket League mutator;
- renderer/game UI;
- exact Bullet API compatibility.

RivalSim is a training transition engine, not a replacement Rocket League client.
