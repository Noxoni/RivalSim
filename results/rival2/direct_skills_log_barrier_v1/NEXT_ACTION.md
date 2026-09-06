# Live exploration-only arm: continue unchanged to frozen +30

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

The no-update saturation diagnostic is COMPLETE, not an ongoing investigation.
Do not repeat the completed native-controller/observation/timing/trace work or
restart the old completed15/25arms. New actual match evidence determines what
comes next.
