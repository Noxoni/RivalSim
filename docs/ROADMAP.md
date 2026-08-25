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

## v0.3 — Ball and dynamic contacts: complete / `PASS_GREEN`

The fixed standard-Soccar two-Octane/one-ball world now includes ball-world, both car-ball pairs,
car-car physical response, bounded ordered bump/demo classification, suspension rays against
dynamic bodies, dynamic broadphase lifecycle order, and a single source-ordered three-body island
solver. Car visitation order is per-world internal lifecycle state established by construction or
membership change; native pointers, allocator addresses, output fitting, and case tables are not
part of the runtime.

The frozen Phase A/B/C/D corpora pass 31,216/31,216, 8,192/8,192, 8,192/8,192, and 512/512 cases
respectively at ticks 1/4/8/12 with zero blocking hard or numeric failures. Phase C and D preserve
both complete source-valid native visitation branches and compare one coherent labeled trajectory
at a time. The complete v0.2.2 static corpus remains 39,236/39,236 and v0.1 remains 27/27.

The complete dynamic path measures 196,614.39 aggregate simulated game-seconds/s at 131,072
worlds with 1.313% CV and zero timed transfers, clearing the 100,000 sim-s/s floor. Two independent
64-world, 2,400-tick stress runs have identical full-state hashes. See `docs/V0_3_RESULTS.md`,
`docs/REPRODUCING_V0_3.md`, `docs/V0_3_ORACLE_CACHE.md`, and `results/v0.3/`.

Game scoring, kickoff/match reset, demolition removal/respawn, RLGym/training integration, and
v0.4+ work were not begun in the completed release.

## v0.4 — Complete standard 1v1 game transition: authorized / not begun

The v0.4 handoff is authorized on top of v0.3 release
`d6ca3912418a3dd7ca8979415142cd861e0c0ddb`.

Mission: turn the accepted two-Octane/one-ball physics engine into a complete headless standard
Soccar 1v1 world transition without beginning the training stack.

Add:

- integration/validation of the existing 34-pad pickup/cooldown state in complete lifecycle/reset;
- goals, goal attribution, and scoring state;
- deterministic source-correct standard kickoff/reset poses and transitions;
- demolition disable/removal-from-active-physics and respawn lifecycle;
- preservation of car identity/visitation semantics across demo, respawn, and physical resets;
- bounded match/reset/lifecycle event state;
- generic terminal/truncation outputs needed by the later training layer;
- deterministic per-world reset/RNG or explicit selector state where lifecycle choices are stochastic;
- integrated repeated-goal/kickoff/pad/demo/respawn episode validation;
- steady-state and reset-heavy GPU-resident performance gates.

The existing v0.3 physics is a frozen regression baseline. v0.4 must not compensate for game-rule
failures by changing accepted collision/solver behavior.

Controlling package:

- `CODEX_START_PROMPT.md`;
- `handoff/v0.4/README.md`;
- `handoff/v0.4/ACCEPTANCE.md`;
- `handoff/v0.4/LIFECYCLE_POLICY.md`.

At completion RivalSim should be capable of running complete headless standard Soccar 1v1 world
transitions with no routine CPU game-state intervention.

v0.4 explicitly does **not** include observations, rewards, rollout buffers, PyTorch policy
inference, PPO, or Rival training. Those remain v0.5.

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
