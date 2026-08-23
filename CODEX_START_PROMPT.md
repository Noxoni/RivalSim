# Codex Start Prompt — RivalSim v0.1 GPU Physics Proof

Work directly in `Noxoni/RivalSim` and implement RivalSim v0.1 completely.

This is a bounded research proof. Do not expand scope just because later milestones are interesting.

## Mission

Answer this question with working code and measured evidence:

> Can a specialized batched GPU Rocket League 1v1 transition engine on the RTX 5090 achieve enough throughput headroom, while matching basic contact-free RocketSim/Rocket League mechanics closely enough, to justify continuing to arena/suspension/contact work?

Read every root/document file before implementation:

- `README.md`
- `VERSION.md`
- `docs/ARCHITECTURE.md`
- `docs/PHYSICS_ORACLES.md`
- `docs/BENCHMARK_AND_PARITY.md`
- `docs/ROADMAP.md`
- `docs/SOURCE_REFERENCES.md`

Do not merely return another design plan. Implement, benchmark, verify, commit and push the proof.

## Repository and versioning

- Canonical repo: `https://github.com/Noxoni/RivalSim`
- Version: `v0.1`
- Work on `main` unless a temporary local branch is useful.
- Commit coherent stable checkpoints.
- Push stable work to `origin/main`.
- Do not overwrite or rewrite prior evidence after publishing it.
- Add/update `CHANGELOG.md` as implementation progresses.

## Environment

Target workstation:

- Windows x86-64;
- NVIDIA RTX 5090;
- CUDA-capable current NVIDIA driver;
- Python 3.12 preferred unless a measured dependency constraint requires otherwise.

Create an isolated repository-local environment.

Use NVIDIA Warp for the initial GPU proof (`warp-lang`). Prefer a normal supported release/wheel compatible with the installed driver; do not modify global Python/CUDA installations unnecessarily. Record exact resolved versions.

PyTorch may be used for tensor interop/tests, but v0.1 does not need PPO or the Rival neural network.

## Reference source custody

RocketSim is the CPU oracle. Do not modify the user's Rival or installed RLBot/RocketSim environments.

If cloning RocketSim or `rl_ball_sym` locally for inspection/testing:

- pin/record exact commit SHA;
- keep source provenance;
- respect licenses;
- do not silently copy third-party code into RivalSim.

Any RocketSim-derived source actually incorporated must carry the required MIT notice.

## Required v0.1 implementation

Create a real batched GPU simulator with GPU-resident state.

Suggested logical package shape (adjust if useful):

```text
rivalsim/
  constants.py
  state.py
  controls.py
  kernels/
    integrate.py
    car_air.py
    jump.py
    dodge.py
    ball.py
  simulator.py
  reference/
    cpu_simple.py
    rocketsim_oracle.py
benchmarks/
tests/
results/v0.1/
docs/
```

### GPU state

At minimum support `num_envs` worlds, each with:

- two cars;
- one ball;
- batched controller inputs;
- jump/double-jump/dodge state/timers;
- boost;
- previous controls;
- orientation/angular velocity.

Keep hot-loop state on GPU. Avoid per-environment Python objects.

### Physics tick

Fixed `dt=1/120`.

Implement contact-free:

- gravity;
- car linear/angular integration;
- numerically stable orientation integration;
- car max linear/angular speed;
- airborne forward/reverse throttle acceleration;
- boost acceleration and consumption;
- first-jump edge detection/impulse;
- first-three-tick sticky force;
- jump-hold bonus through 0.2 s;
- legal double-jump timing;
- dodge/flip state and torque/impulse behavior based on RocketSim's actual implementation;
- airborne pitch/yaw/roll torque based on RocketSim's actual implementation;
- free ball gravity/integration and speed/angular-speed limits;
- ball drag if straightforward and source-backed.

Do not implement fake simplified values where RocketSim source can be used as the reference. If exact RocketSim behavior is more complex than the public RLBot approximations, reproduce the RocketSim behavior needed for parity and document it.

### Explicitly out of scope

Do not implement in v0.1:

- arena triangle collision;
- Bullet port;
- suspension or wheel raycasts;
- ground driving/steering/powerslide;
- ball-world contact;
- car-ball contact;
- car-car contact;
- boost pads;
- goals/scoring;
- demos/respawns;
- RLGym environment integration;
- observations/rewards/PPO;
- rendering UI.

If one of these becomes necessary merely to construct a reference state, isolate the reference utility and do not expand the simulator scope.

## CPU reference implementation

Build a CPU implementation of the same simplified v0.1 equations so GPU speedup can be measured apples-to-apples.

It does not need to be artificially slow. Use sensible NumPy/Numba/vectorized implementation if useful.

## RocketSim oracle/parity harness

Build a harness that can:

1. generate deterministic contact-free initial states and controller sequences;
2. initialize CPU RocketSim with corresponding car/ball state;
3. step RocketSim at the same 120 Hz rate;
4. step RivalSim from the same logical inputs;
5. compare state after multiple horizons.

If RocketSim's public API or Python binding does not expose a field cleanly, use source-backed comparison for that field and document the limitation rather than inventing a value.

Required scenario families are in `docs/BENCHMARK_AND_PARITY.md`.

## Numeric fidelity

Do not preselect forgiving tolerances.

First measure error distributions at 1/4/8/30/60/120 ticks. Then set explicit tolerances that distinguish FP/integration differences from wrong mechanics.

A sign/axis/state-timing mismatch is a hard failure even if aggregate position error happens to look small.

## GPU benchmark

Run the batch sweep in `docs/BENCHMARK_AND_PARITY.md`:

`256, 512, 1024, 2048, 4096, 8192, 16384`

Continue upward if throughput still scales and resources allow.

Use proper warmup and CUDA/Warp synchronization.

Report:

- worlds;
- world physics ticks/s;
- aggregate simulated game-seconds/s;
- GPU utilization;
- VRAM;
- CPU utilization;
- host/device transfer in hot loop;
- NaNs/errors;
- repeated-run variance.

Also benchmark the CPU simplified implementation.

The existing Rival full CPU RocketSim system reference is:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

Do not claim an apples-to-apples full-RocketSim speedup from the simplified v0.1 kernel. Use that value only as an eventual system reference.

## Continuation gate

Proceed recommendation to v0.2 only if the evidence supports it.

Strong pass target:

- best GPU v0.1 >= 4× same-equation CPU v0.1 throughput;
- best GPU v0.1 >= 2,000 aggregate simulated game-seconds/s;
- stable scaling into thousands of worlds;
- hot loop remains GPU resident;
- basic RocketSim parity passes.

If those fail, do not begin v0.2. Diagnose whether the bottleneck is Warp/kernel layout, host transfers, CPU launch overhead, or the concept itself.

## Tests

At minimum add automated tests for:

- state shape/allocation;
- deterministic reset;
- control clamping;
- gravity tick;
- velocity caps;
- boost use/depletion;
- jump edge/hold/timing;
- double-jump timeout;
- air torque axis/sign behavior;
- orientation normalization;
- no NaNs over long contact-free random stress;
- CPU/GPU same-equation parity;
- selected RocketSim parity cases.

Use Ruff/formatting and `compileall` or equivalent Python validation.

## Results and documentation

Commit compact evidence:

- `results/v0.1/benchmark.json`
- `results/v0.1/parity.json`
- `docs/V0_1_RESULTS.md`
- resolved dependency/source manifest
- reproduction commands

Large profiler captures/raw arrays may remain ignored locally, but record file names, sizes, SHA-256 and how to reproduce them.

## End-of-run report

Report only what was actually verified:

- final pushed commit SHA(s);
- exact Warp/Python/CUDA/driver/GPU versions;
- exact RocketSim/reference commit(s);
- implemented mechanics;
- tests and results;
- CPU simplified throughput;
- GPU batch sweep;
- best simulated game-seconds/s;
- speedup vs same-equation CPU implementation;
- comparison to the 200.65 sim-sec/s full RocketSim system reference, clearly labeled non-apples-to-apples;
- parity errors/tolerances by horizon/scenario;
- whether v0.1 passed the continuation gate;
- concrete blocker if it did not;
- next smallest step if it did.

Do not start v0.2 in the same run even if v0.1 passes. Stop at the evidence-backed v0.1 boundary so the result can be reviewed before arena/contact work begins.
