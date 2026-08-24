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
wheel-friction/contact-solver redesign was subsequently authorized as v0.2.1.

## v0.2.1 — Static-world fidelity redesign: complete / `PASS_GREEN`

The source-backed wheel, collision-feature, shared-edge, manifold, and Bullet-style iterative
solver redesign passes all 140 authoritative 1/4/8/12-tick checkpoints with unchanged local
tolerances. Corrected B3 reaches 822,480.77 aggregate simulated game-seconds/s at 262,144 worlds
with zero timed transfers and passes regression/stress gates. Long synchronized 30–600-tick
open-loop identity is diagnostic-only under the immediate 2026-08-23 policy adjustment.

The bounded breadth prototype audits topology over all 8,020 DFH triangles but observes only 2
mesh triangles in the existing local corpus; exhaustive authoritative per-triangle generation
remains a future breadth milestone. See `docs/V0_2_1_RESULTS.md` and `results/v0.2.1/`.

## v0.2.2 — Static-world source-parity breadth gate: complete / `PASS_GREEN`

The pinned RocketSim/Bullet box-versus-static-triangle witness, persistent-manifold,
internal-edge, wheel/suspension, solver, split-impulse, and rigid-body integration paths are
translated directly for RivalSim's existing Octane/Soccar scope. A content-addressed native
authority cache freezes 39,236 deterministic cases across all 8,020 triangles, 23,176 shared
directed edges, and the four analytic planes.

All 156,944 1/4/8/12-tick checkpoints pass with zero hard mismatches, zero numeric failures,
and unchanged tolerances. Corrected B3 reaches 511,886.15 aggregate simulated game-seconds/s at
262,144 worlds with zero timed transfers and retains `PASS_GREEN`. See
`docs/V0_2_2_RESULTS.md` and `results/v0.2.2/`.

## v0.3 — Ball and dynamic contacts: not begun / separate authority required

Do not begin this milestone without a new handoff even though v0.2.2 passed.

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

- integration/validation of the existing 34-pad pickup/cooldown state in full episodes;
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
