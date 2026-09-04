# SSL Foundation PPO V2 corrected pre-launch report

## Verdict

`PASS` for a fresh corrected launch from original Unified Capability V5.
No PPO optimizer step was taken while producing this package.

## Source and authority

- source checkpoint: `checkpoints/rival2/unified_capability_distillation_v5/rival2_unified_capability_v5.pt`
- source SHA-256: `955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216`
- implementation commit bound by authority: `3281ebb024e0e61db5a7ab9881bfed6736cd11e8`
- authority SHA-256: `7FD6999C948D8A2EB0AEF3DA5C35B83BBD71D98C761948F8623C43F1274D39E6`
- fresh-launch authority SHA-256: `8E23A6C991F2EB6B018FCF47116F8405C40C48F1587197A7BE38E2565E2EFF37`
- authority commit: `fc78b6719f7ff363667ec7961d34addf64ceb872`

The launch authority freezes zero accepted PPO updates, no initial resume
checkpoint, a fresh optimizer, and a distinct V2 checkpoint/result/run
namespace. The stopped V1 lineage and its update-20 checkpoint are explicitly
forbidden as initial state. Operational recovery may use only a V2 rolling
checkpoint created after this fresh launch.

## Corrections

1. The reward kernel consumes the actual `physics_ticks_per_decision`; at 120
   Hz the episode boundary executes after one tick.
2. Every true goal emits the terminal `+10/-10`, sets termination/reset, and
   enters a new scenario from the deterministic full-cycle source bank.
3. Standard kickoffs occupy 10% of starts and cover all five Soccar layouts.
4. The global postprocess that pointed every car at the ball was removed.
   Ground starts now couple heading to momentum while preserving broad
   off-angle approach coverage. Shooting attackers and contested-50 cars may
   remain deliberately aligned.
5. The wall/aerial family contains three balanced variants: grounded car with
   elevated ball, car driving on a side wall, and car already airborne on an
   interception/recovery trajectory.

No reward term, potential weight, PPO hyperparameter, opponent probability,
observation/action contract, rollout horizon, or policy architecture changed.

## Validation

- focused CPU/GPU tests: `10 passed`
- Ruff: `PASS`
- exact-scale preflight: `PASS` at 32,768 worlds
- exact-scale rollout-only preflight: `PASS` for 128 ticks
- rollout tensors finite: observations, actions, rewards, values
- model unchanged: `true`
- source checkpoint unchanged: `true`
- optimizer state entries: `0`
- optimizer steps: `0`
- observed rollout terminal goals: nonzero in current, Nexto, and frozen-V5
  families
- observed reset-state touch telemetry: 30,540 touch events
- rollout logical bytes: 6,912,212,992

The brief unsupported-phase error encountered during the first no-learning
preflight was a launch-label wiring issue. It occurred before trainer creation
and before any optimizer step. The V2 checkpoint format and lineage remain
distinct while the trainer phase correctly names the unchanged
`ssl_foundation_v1` reward/rollout implementation.
