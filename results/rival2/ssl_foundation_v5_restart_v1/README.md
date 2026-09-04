# Bounded original-Unified-V5 PPO restart

The user authorized a restart after comparing the preceding corrected SSL run
and its independent-critic continuation. This is a separate, bounded comparison,
not a continuation of any SSL descendant and not random policy initialization.

## Frozen setup

- Original Unified V5 source: `checkpoints/rival2/unified_capability_distillation_v5/rival2_unified_capability_v5.pt`.
- Source SHA-256: `955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216`.
- Independent critic: `182 -> 512 -> 512 -> 512 -> 1`, SiLU. Copy features and
  value head from original V5, preserving initial actor/value/hidden predictions.
- Fresh PPO Adam for all parameters; no supervised or preceding PPO Adam state.
- Local PPO updates/sample counters start at zero. No update-600 or later weights.
- Policy LR `1e-4`, critic LR `3e-4`, two epochs, 65,536-frame minibatches,
  32,768 worlds, horizon 128, 120 Hz, same recurrent and value-gradient semantics.
- Same six-potential reward plus terminal +/-10. No touch or mechanic reward.
- Same corrected reset corpus, seeds, contracts, and 40/40/20 episode-assignment
  probabilities for current Rival / Nexto / original frozen V5. Actual trainable
  sample shares differ because both current agents train and episodes vary.
- Preserve the amended run's actual exploration: analog sigma `0.04`, button
  temperature `0.25`. Exploration's schedule index is `600 + local PPO offset`,
  so resetting PPO counters does not silently restart the earlier low-noise ramp.
- Training episode semantics remain 15-second no-touch truncation and 45-second
  cap. The 30 seconds refers to deterministic evaluation duration, not a changed
  training timeout.
- KL is telemetry only. Nonfinite/corruption rollback and stopping remain active.

## Bound and interpretation

Evaluate at local updates **0, 50, 100**, using the same 3,600-tick / 1,024-world
scenario evaluation and seed as the preceding amended run. Save every accepted
update to rolling.pt and immutable snapshots at 50 and 100. Stop at 100 for user
review (or earlier user stop / hard failure / existing ten-hour safety deadline).
The restart entry point rejects the old unlimited-continuation flag.

Compare against this run's own update-0 baseline, and separately against the
preceding amended run's matched evaluations. Never compare raw goal counts to
the older 1,200-tick evaluations. This tests a V5-root/fresh-optimizer restart as
a bundle; it does not isolate weights from optimizer or exploration-history
effects. No statistical or gameplay success is implied by an integrity PASS.

These logs do not establish possession duration, backward-driving or backflip
frequency. The existing goalward-touch statistic only checks the sign of ball
velocity at contact. Do not interpret it as proof of possession or shot quality.

## Execution and recovery

The stopped run is preserved under `ssl_foundation_ppo_v2/amendment_v1`; its
external STOP_REQUESTED file is intentional and must not be removed or resumed.

```powershell
.venv\Scripts\python.exe benchmarks/run_rival2_ssl_foundation_v5_restart_v1.py
```

Run state: `G:/dev/RivalSim-runs/ssl-foundation-v5-restart-v1`.
Only operational recovery may use `--resume` with this new run's rolling.pt;
the checkpoint format, authority and configuration reject all prior lineages.
Resume restarts simulator episodes/hidden state, as in existing SSL resumes;
it does not claim a bit-exact continuation of live physics state.

Authority and launch bindings must be committed/pushed before real training.
Focused tests and the 32,768-world read-only rollout preflight document exact
source preservation, empty Adam state, finite tensors and native cadence.
Only one campaign may run at a time. The monitor reports newly completed
evaluations or failures, never routine unchanged progress, and pauses at 100.
