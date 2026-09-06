# Native reset V2 is complete; do not relaunch or reduce again

## Completion supersedes the historical launch instructions below

Both games have closed and been reduced: reference600 lost 0-24 on Blue and 1-30
on Orange against actual native Nexto. Read `RESULTS.md` and both case audits.
External state is `complete_review`; runner PID 23972 has exited. No learning
is running. Final state/logs are preserved in `completion/`.

The full Blue-game CPU observation-format comparison is also complete in
`results/rival2/entity_native_observation_math_v1/`. It isolates formatting only:
97 reconstructed timer/wheel/pad/lifecycle fields were shared inputs. Do not
repeat it or the installed Nexto identity/table/scheduler audit.

The native jump-timer diagnosis is now also complete in
`results/rival2/entity_native_jump_timer_v1/RESULTS.md`; do not repeat it.
The bounded native-timeout/held-jump comparison changes 2,071 observations but
only six same-hidden-state controller decisions across both games. Maximum
timer mismatch is about 0.2 seconds. It is real but does not establish the
cause of the entire loss; no deployment fix or quality promotion was applied.

The four-arm Nexto causal comparison is complete at
`results/rival2/nexto_native_causal_v1/RESULTS.md`: baseline 14-3 and 34 touches;
native timing 2-10 and 23 touches, across ten 20-second worlds. Kickoff yaw alone
leaves all sampled poses exact. Read the limitations; this is not full native
Nexto parity. The baseline's 6,000 Rival decisions matched prior evidence.

The full-regulation follow-up is also COMPLETE in
`results/rival2/nexto_native_full_match_v1/RESULTS.md`. Baseline reproduces 8/10
wins and 169-122 goals; native timing plus table gives 0/10 wins and 34-239 goals.
Both full traces exactly reproduce their respective first20s segments. Ten
focused tests passed; worker42976/launcher21200 exited. No training is live.
Do not rerun either comparison or the earlier native games/audits.

The versioned production-controller correction is now also COMPLETE. Read
`results/rival2/nexto_native_controller_v1/RESULTS.md`: actual installed controller
oracle matches on CPU/CUDA; real pinned-model masked/RNG replay is exact in both
explicit sampling modes; 32,768 worlds completed 90 rollout-only decisions with
unchanged weights and correct learner masks. No optimizer/backward occurred.
Do not repeat those diagnostics or substitute the old all-active probe into PPO.

Next concrete task: freeze a corrected learning/evaluation configuration using
`DirectSkillNativeNextoCollector`, explicit native-v5 sampling/seed, an exact
preserved parent, and honest fresh-physical-episode resume semantics. All future
Nexto comparisons must identify the corrected controller, not silently use the
legacy benchmark. Keep the existing finishing-goal reward/curriculum unless a
new measured implementation fault demands correction. No new training until
the corrected campaign configuration and baseline have been recorded. Preserve
legacy code/results; do not call their 8/10 wins native competence.

Preserve reference600, finishing650 and all negative evidence. Do not promote
either checkpoint, restart games or start PPO as a substitute for resolving the
transfer gap. The broad SSL goal is active; native competence is not demonstrated.

## Historical launch/progress record (superseded)

Blue has now completed a full native0–24loss. Its three audits and full raw
captures are in `reference600_blue/`; replay is exact, all25projections occur
only at their intended first kickoff decisions, and all10,768eligible native
consumed-control endpoints match. Do not reduce Blue again or report its result
as a new Orange result. Orange is running under the same existing runner.
Read `RESULTS.md` and `results/rival2/native_nexto_integration_audit_v1/RESULTS.md`.
The installed Nexto model identity/table/scheduler audit is also complete;
do not repeat it. Neither discovery is permission to modify the active test.

The protocol is published at `9c9dffe8b59c6b2bdd909bf672fdc630ea7fd4f8`.
All 13 commit files were read back from the remote, 15 local authority source
hashes matched, and the checkpoint/export hashes matched before launch.
See `launch_receipt.json` for the first 120 actual flushed decisions: only the
first kickoff decision projected eight unavailable wheel fields, all other
inputs stayed exact, and the policy emitted throttle and boost itself.

The hidden native runner was launched at 13:18:55 UTC on 2026-09-06. Its actual
worker PID was 23972, launcher 44136, with external state at
`G:/dev/RivalSim-runs/entity-native-reset-v2/campaign_state.json`. These are
observed identities, not proof the process is still running later.

1. Inspect that state, current process identity, stderr and per-case bot state.
   Preserve a healthy process. Never launch a second comparison or restart the
   existing directory. This is native evaluation, not PPO or a training run.
2. Let the frozen two-case runner finish or reach its existing time/error bound.
   Do not extend the 900-second per-case cap, alter inputs or select another
   checkpoint after observing scores. Preserve all partial/negative outcomes.
3. Once each bot stream is closed, reduce with the already published
   `benchmarks/report_entity_native_reset_v2.py reference600_blue` / orange using
   `G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B`. The reducer checks unchanged
   fields/masks, countdown/first-decision placement, action replay, held controls,
   actual consumed controls at eligible decision endpoints and source hashes.
4. Commit raw captures, audits, outcome summaries and final state. Preserve the
   exact bytes of any hash-bound raw metadata. Verify remote persistence.
5. Compare with the prior partial native 0-37 honestly, not as a completed
   matched-seed baseline. A reset-input correction alone may not close the gap.
   No inference of native competence, SSL or finishing650 performance without
   actual evidence. Continue diagnosis if needed; do not resume PPO as a fix.

Shooting is already 20% of the direct-skills scenario bank. The completed
finishing-goal block rewards actual goals/concedes and failed timeouts rather
than touch/projection bonuses. Its limited simulator progress is preserved in
`results/rival2/direct_skills_finishing_goal_v2`; no shooting scenario is missing
and no additional learning is running during this compatibility test.
