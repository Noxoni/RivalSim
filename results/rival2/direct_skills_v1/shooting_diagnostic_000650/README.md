# Fixed650 shooting-pressure diagnostic

The prior sampling comparison did not rescue match scoring. Test whether weak
shooting is already present with an initially unguarded goal or primarily under
keeper pressure. This is a small matched native-state comparison, not another
training campaign, model selection, reward calibration or mechanic detector.

Checkpoint650 SHA256:
`4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F`.

Use exactly the64 published finishing starts, unchanged ball and Rival state,
and three fixed defender conditions:

1. Established keeper: identical original64 states and per-case result replay.
2. Recovering defender: opponent starts1300uu behind the ball, at alternate
   canonical x+/-2200, moving600uu/s toward its goal center.
3. Initially open net: opponent starts at canonical y-3500,x+/-3000,stationary,
   facing the ball. Nexto remains active and can recover: this is NOT a frozen
   or disabled opponent for12seconds.

Only opponent position, coherent momentum and grounded yaw change. Team
canonicalization preserves both sides. Shared source arrays, boost, ball and
Rival state, physics, rewards, model, observation/action contracts and inference
remain unchanged. No controller prefix or forced Rival action is applied.

All modes use greedy action selection,30Hz policy/120Hz physics, same pinned
Nexto and twelve-second skill horizon. No optimizer exists. The first mode must
reproduce every published650 finishing outcome exactly before the other modes
can be interpreted. Preserve failure artifacts instead of silently rerunning.

Record all192 episode outcomes and compressed30Hz pre/action/post observation
traces with alive masks, native contact counts and within-decision first-contact
and goal timing. These traces are diagnostics, not synthetic demonstrations.
Report actual goals, concessions, timeouts, goals before any opponent contact,
first-contact-to-goal timing, and contact/follow-up counts. Projected on-target
events remain task proxies, never a replacement for scoring.

Seven focused CPU tests establish deterministic state generation, exact original
bank identity, physical bounds/non-overlap and opponent-only intervention. Native
execution additionally checks unchanged checkpoint/model/Nexto, finite outputs,
complete outcomes and original per-case baseline replay.

Publish authority/package/source/tests and verify remote identities before run.
Use the existing exclusive GPU lease. Training remains paused at650 and its
review STOP remains intact. After results, choose a small prospective curriculum
correction if supported; no automatic reward change or another blind block.

```powershell
.venv\Scripts\python.exe benchmarks/diagnose_direct_skills_shooting_000650.py prepare
# Commit, push and verify the prospective package.
.venv\Scripts\python.exe -u benchmarks/diagnose_direct_skills_shooting_000650.py run
```
