# RivalSim v0.1 Benchmark and Parity Gate

v0.1 succeeds only if it demonstrates **both** substantial GPU throughput headroom and useful basic-physics fidelity.

## CPU system reference

The current Rival RocketSim training sweep measured:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

That is a full CPU RocketSim/RLGym training-path measurement and is not directly apples-to-apples with the simplified v0.1 kernel. Keep it as the eventual system-level target.

## Required implementations for benchmarking

Codex must benchmark three paths where practical:

1. **GPU RivalSim v0.1** — Warp/CUDA, GPU-resident hot loop.
2. **CPU reference implementation of the same simplified v0.1 transition equations** — vectorized/compiled if reasonable, used to quantify GPU acceleration independent of RocketSim feature differences.
3. **CPU RocketSim oracle** — used primarily for trajectory/state fidelity, plus its measured throughput where comparable.

Do not compare a no-collision GPU kernel to full RocketSim and call the ratio an honest final speedup. Report both simplified-kernel speedup and current full-RocketSim system reference.

## GPU batch sweep

Benchmark at minimum:

- 256 worlds;
- 512;
- 1,024;
- 2,048;
- 4,096;
- 8,192;
- 16,384;

Continue upward if VRAM and throughput still scale materially and kernels remain stable.

For each count report:

- physics ticks/s;
- aggregate simulated game-seconds/s (`worlds * ticks_per_second_simulated / 120`, equivalently total world-ticks/s / 120);
- wall time;
- GPU utilization;
- VRAM allocated/reserved;
- kernel time if available;
- CPU utilization;
- host↔device bytes transferred in the hot loop;
- errors/NaNs.

Use warm-up iterations before timing. Synchronize CUDA/Warp appropriately around measurements.

## Hot-loop benchmark

Benchmark long enough to avoid launch-noise domination. Target at least several simulated seconds per world and multiple repeated runs.

The timed hot loop must not read full world state back to CPU every tick.

## Provisional continuation performance gate

v0.1 should show **large headroom**, because arena contact, suspension and dynamic contacts will cost substantial performance later.

Proceed to v0.2 if all are true:

- GPU throughput is still scaling into the thousands of worlds;
- the best stable GPU configuration is at least **4× faster than the CPU implementation of the same v0.1 equations** in aggregate simulated game-seconds/s;
- the best stable simplified GPU throughput is at least **10× the current 200.65 simulated game-seconds/s full-RocketSim system reference**;
- GPU hot-loop transfer behavior is not dominated by CPU round-trips;
- basic parity gates below pass.

These are research gates, not marketing claims. If a narrowly missed threshold has a clear fix, document it; do not manipulate benchmark conditions to force a pass.

If GPU v0.1 cannot reach at least ~2,000 aggregate simulated game-seconds/s on the RTX 5090 without collisions, stop and reassess the architecture before implementing v0.2.

## Parity corpus

Use deterministic random seeds and explicit controller sequences.

At minimum test:

### Free-body

- stationary gravity drop;
- arbitrary initial linear velocity;
- arbitrary initial angular velocity;
- speed-limit crossing;
- combined rotation integration.

### Boost / air throttle

- boost from rest in air;
- boost with nonzero initial velocity;
- boost depletion timing;
- forward air throttle;
- reverse air throttle;
- boost + throttle combinations.

### Jump

- tap jump;
- minimum-three-tick jump;
- full 0.2 s jump hold;
- first jump followed by legal double jump at several delays;
- attempted double jump after timeout;
- jump while car orientation is not upright.

### Air torque / dodge state

- isolated pitch/yaw/roll;
- combined torque inputs;
- legal dodge/flip in representative directions;
- flip torque timer evolution;
- angular-speed limiting.

## Comparison horizons

Compare GPU RivalSim and CPU RocketSim at:

- 1 tick;
- 4 ticks;
- 8 ticks;
- 30 ticks;
- 60 ticks;
- 120 ticks;
- longer where the tested motion remains contact-free.

Record errors for:

- position;
- linear velocity;
- orientation angle;
- angular velocity;
- boost;
- jump/double-jump/dodge state;
- relevant timers.

## v0.1 fidelity gate

Because there are no world contacts, basic airborne mechanics should track closely.

Require:

- finite state for every tested sequence;
- correct discrete jump/dodge state transitions;
- no systematic axis/sign errors;
- one-tick impulses/accelerations matching the RocketSim oracle within normal FP32 numerical tolerance;
- trajectory divergence over 1 second small enough to be explained by floating-point/integration differences rather than incorrect mechanics.

Codex must choose and report explicit numeric tolerances based on measured RocketSim values; do not invent loose tolerances before seeing the error distributions.

## Deliverables

Commit compact results:

- `results/v0.1/benchmark.json`
- `results/v0.1/parity.json`
- `docs/V0_1_RESULTS.md`

Large raw arrays/profiles may remain ignored locally, but record paths, hashes and reproduction commands.
