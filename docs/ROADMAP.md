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

The exact external RocketSim Soccar `.cmf` set was loaded as shared GPU geometry, with suspension rays, ground driving, chassis/world contact, and a decomposed benchmark. B3 reached 1,350,748.16 aggregate simulated game-seconds/s at 262,144 worlds, but the 35-scenario RocketSim parity gate failed, so v0.2 remained `PAUSE_RED`.

## v0.2.1 — Static-world fidelity redesign: complete / `PASS_GREEN`

The source-backed wheel, collision-feature, shared-edge, manifold, and Bullet-style iterative solver redesign passes all 140 authoritative 1/4/8/12-tick checkpoints with unchanged local tolerances. Corrected B3 reaches 822,480.77 aggregate simulated game-seconds/s at 262,144 worlds with zero timed transfers.

## v0.2.2 — Static-world source-parity breadth gate: complete / `PASS_GREEN`

A content-addressed native authority cache freezes 39,236 deterministic cases across all 8,020 triangles, 23,176 shared directed edges, and the four analytic planes. All 156,944 1/4/8/12-tick checkpoints pass with zero hard mismatches and zero numeric failures. Corrected B3 reaches 511,886.15 aggregate simulated game-seconds/s at 262,144 worlds.

## v0.3 — Ball and dynamic contacts: complete / `PASS_GREEN`

The fixed standard-Soccar two-Octane/one-ball world includes ball/world, both car/ball pairs, car/car physical response, ordered bump/demo classification, suspension rays against dynamic bodies, dynamic broadphase lifecycle order, and one source-ordered three-body island solver.

The frozen Phase A/B/C/D corpora pass 31,216/31,216, 8,192/8,192, 8,192/8,192, and 512/512 cases respectively at ticks 1/4/8/12 with zero blocking failures. The complete dynamic path measures 196,614.39 aggregate simulated game-seconds/s at 131,072 worlds with 1.313% CV and zero timed transfers.

## v0.4 — Complete standard 1v1 game transition: complete / `PASS_GREEN`

The accepted v0.3 physics is composed with a complete GPU-resident standard-Soccar 1v1 lifecycle: all 34 boost pads, strict goal detection and team attribution, scores, five explicit kickoff layouts, demolition disable/timing, four explicit respawn locations per team, world/episode clocks, raw events, and deterministic full-world resets.

Native lifecycle authority passes 68 pad pickup cases, six goal-boundary cases, five kickoff layouts, eight team/respawn poses, exact tick-360 respawn, and deterministic mixed lifecycle/reset stress. All inherited simulator regressions remain green.

The complete path measures 191,748.10 aggregate simulated game-seconds/s at 131,072 worlds with 0.856% CV and zero timed transfers. The reset-heavy path measures 225,005.06 sim-s/s and 3.375 million reset transitions/s with 0.723% CV and zero timed transfers.

RocketSim has no authoritative training episode termination contract, so v0.4 exports raw lifecycle state and leaves training policy to v0.5.

## v0.5 — Rival 2.0 GPU-native training: complete / `PASS_GREEN`

v0.5 creates **Rival 2.0** as a clean-slate policy/training system directly on RivalSim.

The old `Noxoni/Rival` training stack is not an implementation dependency. Rival 2.0 does not preserve Wisp-derived weights, the legacy 432-value observation, the 90/158-action tables, or old training schemas.

Completed GPU-native scope:

- persistent zero-copy RivalSim/PyTorch CUDA tensor views;
- new symmetric `RIVAL2_OBS_V1` observation construction;
- fixed native hybrid controller at 30 Hz:
  - throttle/steer/pitch/yaw/roll as tanh-squashed Gaussian controls;
  - jump/boost/handbrake as Bernoulli buttons;
  - 13 actor outputs total;
- compact `RIVAL2_REWARD_V1` plus explicit goal termination and timeout truncation;
- GPU-resident rollout storage;
- GAE and clipped PPO;
- actor/critic checkpoint/save/resume;
- current-policy self-play;
- bounded historical Rival 2.0 opponent snapshots;
- end-to-end learning sanity and world-count/performance sweeps.

Target hot path:

`GPU RivalSim -> RIVAL2_OBS_V1 -> GPU actor/critic -> 8 native controls -> RivalSim x4 ticks -> GPU rewards/dones -> GPU rollout -> GPU GAE/PPO -> updated Rival 2.0`

Only configuration, logging, metrics snapshots, checkpoint serialization, and explicit offline diagnostics should routinely leave the device.

The first reward intentionally avoids mechanics-specific shaping, and the episode contract uses
accepted kickoff resets rather than a hand-built curriculum. All deterministic contract/math,
residency, checkpoint, self-play, and fixed-seed learning-sanity gates pass.

The selected 131,072-world complete rollout+GAE+PPO point measures 2,233,901.63 agent samples/s
and 89,505.78 simulated game-seconds/s with 0.588% CV, 14,414,032,896 peak observed VRAM bytes,
and zero timed H2D/D2H. The prospectively declared held-out clipped PPO objective improves by
4.226 standard errors in the bounded smoke. This proves non-no-op learning integration, not bot
skill or external transfer. All v0.4/v0.3/v0.2.2/v0.1 regressions remain green.

## v0.6 — Rival 2.0 transfer gate

Status: **not authorized / not begun**.

Train a bounded Rival 2.0 policy using RivalSim, then implement a deployment adapter that reproduces `RIVAL2_OBS_V1`, 30-Hz cadence, and the hybrid deterministic controller in CPU RocketSim/RLBot/Rocket League.

RivalSim does not earn production use merely by matching itself.

Required transfer checks should include:

- fixed-policy observation parity between RivalSim and the deployment adapter;
- native eight-channel action parity;
- basic driving/jump/aerial execution;
- wall movement;
- recoveries;
- ball touches;
- learned mechanics;
- fixed-policy evaluation against external RocketSim/RLBot baselines.

If Rival 2.0 exploits RivalSim-only physics and fails transfer, fidelity work takes precedence over more training throughput.

## Explicit non-goals until needed

- Hoops;
- Dropshot;
- Snowday;
- Heatseeker;
- Rumble;
- arbitrary numbers of cars/balls;
- every Rocket League mutator;
- renderer/game UI;
- exact Bullet API compatibility;
- distributed multi-GPU training.

RivalSim is a training transition engine, not a replacement Rocket League client.
