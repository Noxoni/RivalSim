# RivalSim

RivalSim is an experimental **GPU-native Rocket League 1v1 transition engine** intended to accelerate training for [Noxoni/Rival](https://github.com/Noxoni/Rival).

The project is deliberately narrower than RocketSim. The initial target is standard Soccar 1v1 only:

- two cars;
- one ball;
- standard arena;
- standard boost pads;
- 120 Hz physics;
- no rendering in the benchmark path.

## Current milestone — v0.1 GPU physics proof

RivalSim v0.1 is implemented and has passed its bounded performance/parity gate. It does
**not** attempt to replace RocketSim yet.

Its job is to answer one question quickly:

> Can a batched GPU transition engine on the RTX 5090 advance thousands of independent Rocket League-like worlds fast enough, while tracking basic RocketSim/Rocket League motion closely enough, to justify continuing the project?

v0.1 implements only the physics needed to make that decision:

- batched GPU state for two cars and one ball per world;
- 120 Hz fixed-step integration;
- gravity;
- linear/angular rigid-body integration;
- car velocity/angular-velocity caps;
- airborne throttle;
- boost acceleration/consumption;
- jump impulse, sticky force and jump-hold bonus;
- double-jump / flip timing and state;
- airborne pitch/yaw/roll torque;
- deterministic reset/random-state generation for parity tests.

**Not in v0.1:** arena triangle collision, suspension/wheels, ground driving, ball/car contacts, car/car contacts, boost pads, goals, demos, or full RLGym integration.

Those are intentionally deferred until the GPU architecture proves it can beat CPU simulation by a large margin.

## v0.1 result

On the measured RTX 5090 workstation, the best stable GPU point advanced 131,072 worlds at
4.910 billion world ticks/s, or 40.919 million aggregate simulated game-seconds/s. The best
same-equation NumPy CPU point reached 11,125.38 simulated game-seconds/s, for a measured
3,678.02x same-equation GPU/CPU ratio. All 27 deterministic RocketSim parity scenarios passed
at 1/4/8/30/60/120 ticks, including hard axis/sign/state-timing checks.

This is a contact-free kernel result. Its 203,934.02x ratio to the existing 200.65 sim-s/s
full RocketSim/RLGym system reference is **not apples-to-apples** and must not be presented as
a full-simulator speedup.

See [`docs/V0_1_RESULTS.md`](docs/V0_1_RESULTS.md) for the protocol, full sweep and limitations,
and [`docs/REPRODUCING_V0_1.md`](docs/REPRODUCING_V0_1.md) for reproduction commands. The v0.1
artifacts are frozen under [`results/v0.1/`](results/v0.1/). No v0.2 work is included.

## Implementation choice

Use **NVIDIA Warp** first for the proof. Keep state tensor-native and GPU-resident. Do not build a Python object graph per world and do not round-trip per-tick state through CPU/NumPy.

If the Warp proof succeeds, later milestones may keep Warp or migrate proven hot kernels to native CUDA C++.

## Performance reference

The current Rival CPU RocketSim training sweep measured a best point of:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

That is the real system-level reference RivalSim is trying to beat eventually.

The v0.1 simplified kernel must show **large headroom**, not a marginal win, because later collision/suspension/contact work will reduce throughput.

## Start here

Give Codex:

`Read CODEX_START_PROMPT.md and execute it completely.`

See:

- `docs/ARCHITECTURE.md`
- `docs/PHYSICS_ORACLES.md`
- `docs/BENCHMARK_AND_PARITY.md`
- `docs/ROADMAP.md`
- `docs/V0_1_RESULTS.md`
- `docs/REPRODUCING_V0_1.md`

## Relationship to RocketSim

RocketSim remains the CPU reference and training backend unless/until RivalSim earns replacement through measured performance and transfer fidelity.

Any RocketSim-derived code incorporated into RivalSim must preserve its applicable MIT license notices and provenance.
