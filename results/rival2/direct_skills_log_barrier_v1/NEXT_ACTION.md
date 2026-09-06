# COMPLETE: exploration-only arm stopped at +30

UPDATE: the reviewed unchanged-settings continuation is now live under
results/rival2/direct_skills_exploration_continuation_v1/NEXT_ACTION.md.
Prospective commit fe93e5cb5e2e67465da7db8a127d715d09f96344 was remotely verified
before its launch. Follow that new authority; do not restart this old arm.

Worker exited normally at 2026-09-06T16:31:45Z. `completion.json` is generated
and verified. Final: 0/10 wins, 2 scored /228 conceded, 8.88 touches/minute,
zero kickoff first contacts. This recovers from +15 but remains worse than the
actual parent. Do not restart, extend, rerun these matches, or promote this arm.
The final checkpoint SHA is
16BF2B904785D49E0363B869AF953CFCCC62FF754286A307D386729AC691DDAD.

The post-exploration no-optimizer probe is COMPLETE. Its summary and results
are in `results/rival2/direct_skills_post_exploration_probe_v1/`. At beta 0.01
the median exploration/PPO gradient norm ratio is 0.09665, maximum 0.16416:
not evidence of a dominating exploration term. The raw candidate selection
0.03 is NOT adopted; do not use it as a new authority. Do not rerun the probe.

Next work is an evidence-based learning decision under the continuing SSL goal,
not resuming this completed arm. Consider the late recovery (+15 to +30) and
continued failure versus parent separately. Do not declare the exploration
pressure or a stuck critic the established cause. If proposing a continuation
or changed approach, freeze its concrete hypothesis, checkpoint identity,
budget, comparisons and stop conditions before any new optimizer step. Keep
the existing checkpoint/evaluation protocol; no large new benchmark suite.
A new continuation authority now exists at the path above; this old authority
remains closed and unchanged.

The saved critic-residual analysis in the learning-diagnostic directory is
also complete. It shows the previous kickoff child adapted its critic without
improving full matches; do not claim a stuck critic is the proven explanation.

## Historical execution instructions (completed; do not restart)

Prospective commitc72e06cf25fd93e176bd54a197a7b590b91c9788 was pushed and all22
files remotely read back before launch. Authority:
55413BB4E1B71CBB9F9FD2916828F1F7A0C684FE70EC5AE88AA7169CA8852C7E.

Active entrypoint: benchmarks/run_direct_skills_log_barrier_v1.py run.
External state: G:/dev/RivalSim-runs/direct-skills-log-barrier-v1.
Launched hidden2026-09-06T16:08:48Z. Actual worker25048; launcher37088 is not
proof of active training. Inspect the current process command and state.

Parent650 exact model/Adam/groups/counters/four RNG restored. Entry SHA:
61DB16080D1FA157A01C3928E01B5C77990C7343BDCEC5F7A468AC6CCB0F15EA.
First accepted checkpoint SHA:
A6C5C41EC1FA2D2608F0259963AC569A632B4E207510892D481AD70577F83AC2.
Both are preserved under checkpoints/rival2/direct_skills_log_barrier_v1/.

First audit:26core plus6new-objective checks PASS.136Adamsteps,4,423,680learner
decisions. The actual runtime loss includes the calibrated beta0.01uniform log
barrier. First entropy0.84835 and completed-update KL0.1135703 are training
diagnostics, not improved gameplay. Sample KL16878.96 is telemetry, NOT failure.
No KL rejection and no nonfinite state. Do not silently revert to a KL gate.

Continue healthy training unchanged to30accepted updates, evaluations5/15/30.
Same actual-goal shooting20%, current reward/curriculum/native-v5 Nexto, T2
sampling AND likelihood,30Hz decisions, independent critic, no action override.
Do not retune mid-arm. Do not restart a healthy process or overlap GPU workers.
Respect user STOP and frozen numerical safety. No automatic extension at30.

At preserved boundaries run the CPU-only reporter:

```powershell
.venv\Scripts\python.exe -B benchmarks/report_direct_skills_log_barrier_v1.py 5
```

Use offset1,5,15or30 only. Both progress_*.json and objective_audit_*.json must
pass. The inherited unchanged_ppo check means base PPO configuration; the new
loss is explicitly bound and audited separately. Do not mistake old checks for
proof the new objective was active. Report new completed full-match results
only, comparing the same parent6500/10wins7-196goals baseline. Publish immutable
checkpoint/evaluation/prefix evidence and remotely verify it. Update external
notification cursor, which also records publication status.

At30 and worker exit, archive closed logs/state, verify hashes/counters and all
three complete evaluations. A finalizer must use this directory and30boundary;
do not run the old hardcoded15finalizer. No automatic gameplay promotion.
The broader SSL goal remains active.

The CPU-only finalizer is now implemented and tested (17 focused tests):

```powershell
.venv\Scripts\python.exe -B benchmarks/finalize_direct_skills_log_barrier_v1.py
```

It checks actual worker absence, complete 30-row state, all 5/15/30 evaluation
audits, the separate exploration objective, frozen source hashes, and exact
rolling/permanent checkpoint equality. It neither trains nor reruns matches.
Implementation and tests were pushed and remotely read back at
6ecbf17e4500b7abb8eaea1ea3c7c7a23d50bfcd.

Update +5 evidence is published at 6ea9ad77dfe7f4955ae8873d001db27225cbfdfe.
Update +15 is complete: 0/10 wins, 1 scored / 291 conceded, 6.64 touches/minute,
no kickoff first contact. All 26 core +6 objective +11 match checks pass. Both
evaluations are worse than the parent. Already reported to the user; do not
repeat them as newly completed evidence. Next scheduled result is +30. Do not
interpret increased entropy as competitive improvement or extend blindly.

The no-update saturation diagnostic is COMPLETE, not an ongoing investigation.
Do not repeat the completed native-controller/observation/timing/trace work or
restart the old completed15/25arms. New actual match evidence determines what
comes next.
