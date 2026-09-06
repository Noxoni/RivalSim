# Completed arm: do not restart or extend this directory

The frozen kickoff-outcome arm completed exactly +15 / cumulative665 on
2026-09-06 at15:52:44Z. Worker43944 exited after full evaluation. Read
REVIEW_000015.md and completion.json first. This section supersedes the live-run
instructions retained below as history.

Final unassisted Nexto result:0/10wins,5-209goals,447contacts8.94/min,zero kickoff
first contacts,90/438same-player followups20.55%. Parent650:0/10,7-196goals,
456contacts9.12/min,zero kickoff first contacts,123/444followups27.70%.
This is not competitive improvement. Do not repeat the final result as new.

Final checkpoint: checkpoints/rival2/direct_skills_kickoff_race_v1/child_000015.pt
SHA183C59CA2F7C04681C5EC92272F63A8494223BEC422B88F74AA48F23603B7957.
Parent650 remains preserved; no automatic child promotion. All26 CPU checkpoint
checks and11 match-integrity checks passed. 15updates,66,355,200learnerdecisions,
2,118Adamsteps. KL telemetry only, zero rejection, no finite failure.

Next: quantify actual per-role/opponent lesson exposure and learning from the
closed training curve. Distinguish assisted practice from unassisted transfer;
aggregate logs may not expose assisted/unassisted success separately, so do not
manufacture that distinction. Use this to select one bounded next intervention
prospectively. No next campaign/settings are frozen yet. Shooting remains20% of
source starts, actual goals not projected shots. The broad SSL goal stays active.
Do not repeat completed native-controller/observation/timing/kickoff diagnostics.

The CPU-only finalizer already ran successfully. Verify the latest Git commit
and external last_notified_evaluation.json cursor for publication/readback status.
Honor newer user instructions/STOP, preserve concurrent work and the GPU lease.

## Historical instructions while the arm was live (superseded)

The new worker runs `benchmarks/run_direct_skills_kickoff_race_v1.py run` with
external state under `G:/dev/RivalSim-runs/direct-skills-kickoff-race-v1`.
Prospective commit `ecead012976418c703990ce4f101d7b8cef19c33`, authority
`C048BA31AA975E5A92D62C7E1EB21FD18128785FFFC980D8D9A7B411011B361E`.
All13files remotely read back before launch; entry/first audit publication
`f8e8fef2fe571ecb9a93c01ed1edefb6199bc362` also fully read back, including
checkpoint/raw evidence hashes. Actual observed workerPID43944; always recheck
the process/command, not merely this historical PID or a stale state file.

+5 evaluation is COMPLETE:0/10wins,6–213goals,497contacts,one kickoff first
contact across the entire regulation set. Parent650 was0/10,7–196goals,
456contacts,zero kickoff first contacts. This is not competitive improvement.
See `REVIEW_000005.md`. Do not report it repeatedly as a new evaluation.

Continue a healthy process to its frozen+15 boundary; no mid-arm tuning or
operational interruption. Reward/curriculum/settings are in `PLAN.md` and the
hashed authority. Shooting remains20%. Both source model650 and all old
checkpoints are preserved. Do not resume any old completed25-update directory.

After+15 evaluation completes and the worker exits, run the CPU-only finalizer:

```powershell
.venv\Scripts\python.exe -B benchmarks/finalize_direct_skills_kickoff_race_v1.py
```

It verifies the exact15-row boundary, hashes, finite checkpoint/Adam state,
new reward/corpus provenance and full-match integrity, preserves closed logs
and compares parent→5→15 without automatic promotion. Write a concise honest
review, commit/push the final checkpoint and all new results, remotely read back
the artifacts, update the notification cursor. Do not extend the finite runner
automatically; decide the next learning intervention from the completed result.
The broad SSL-development goal stays active, not completed by this small arm.

If an operational fault occurs, preserve latest accepted state and investigate
before an explicitly authorized recovery. Never restart on an observer timeout,
clear a user STOP, bypass finite/corruption protection, or treat KL magnitude as
a failure. Follow newer user instructions first.
