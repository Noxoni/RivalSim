# Changelog

## v0.1.0 — GPU physics proof implemented (2026-08-22)

- Added a flattened, GPU-resident two-car/one-ball world state and fused NVIDIA Warp 120 Hz
  contact-free transition kernel.
- Implemented source-backed gravity, rigid-body integration, caps, airborne throttle/boost,
  jump/sticky/hold/double-jump, dodge/flip, aerial torque/damping and free-ball behavior.
- Added the vectorized NumPy same-equation CPU reference and live `rocketsim==2.2.1`
  `GameMode.THE_VOID` oracle.
- Added 27 deterministic parity scenarios and horizon-specific tolerances selected only after
  a corrected measurement-only run. Same-equation, live RocketSim and axis/sign/state parity
  all passed at 1/4/8/30/60/120 ticks.
- Added automated allocation/reset/control/mechanics/stress/parity/evidence tests.
- Added an adaptive, repeated CPU/GPU benchmark with separate untimed telemetry and explicit
  transfer/variance/NaN accounting. The GPU hot path uses eight-tick CUDA graph blocks.
- Measured a best stable RTX 5090 result of 40,919,361.97 aggregate simulated game-seconds/s
  at 131,072 worlds, versus 11,125.38 sim-s/s for the best same-equation CPU point: 3,678.02x.
- Passed every v0.1 continuation condition. The 203,934.02x ratio to the 200.65 sim-s/s full
  RocketSim/RLGym system reference is recorded only as a non-apples-to-apples comparison.
- Added compact benchmark/parity evidence, resolved dependency/source custody, third-party
  notices and exact reproduction commands.
- Final validation: 20 tests passed; Ruff, `compileall`, JSON parsing and `git diff --check`
  passed. `pip check` retained a documented upstream RocketSim wheel-tag metadata warning;
  the extension imported and all live-oracle tests passed.
- Stopped at the v0.1 boundary; no arena, suspension, ground-contact or other v0.2 work began.

## v0.1 — GPU physics proof handoff

- Established RivalSim as a separate GPU-simulation research repository.
- Defined Soccar 1v1-only scope for the initial architecture.
- Selected NVIDIA Warp for the first GPU proof.
- Defined GPU-resident batched state and contact-free 120 Hz mechanics scope.
- Added RocketSim/RLBot/RLGym physics references and source hierarchy.
- Added CPU/GPU benchmark and RocketSim parity gates.
- Added staged roadmap through arena collision, dynamic contacts and tensor-native training integration.
- Added Codex implementation prompt.

This section records the pre-implementation handoff. The implemented result is preserved above
as v0.1.0 rather than rewriting the historical boundary.
