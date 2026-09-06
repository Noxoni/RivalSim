# Current arm: continue only the live frozen kickoff experiment to+15

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
