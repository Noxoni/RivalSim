# RivalSim Research Roadmap

## v0.1 — GPU physics proof

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

No arena collision.

## v0.2 — Arena + ground-contact proof

Only begin if v0.1 passes.

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

The RLBot map-mesh extraction documentation identifies DFH collision assets and a binary triangle-export path. `rl_ball_sym` is a useful implementation reference for turning those binaries into triangles/BVH structures.

## v0.3 — Ball and dynamic contacts

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
