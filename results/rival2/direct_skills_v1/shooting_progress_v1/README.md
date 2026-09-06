#650 to675: intermediate shooting pressure, unchanged reward and PPO

This bounded continuation follows the completed650 shooting-pressure diagnostic.
Rival finished49/64 initially open and50/64 recovering-defender opportunities,
but only5/64 established-keeper cases. It got first contact in every case; Nexto
intercepted within one second in49/64 established-keeper cases. Sampling did not
rescue full matches. Do not repeat those diagnostics or declare shooting solved.

## One learning change

Replace half of the finishing source rows with intermediate recovering-to-keeper
geometry. Finishing remains20% of the total bank. Thus only3276 of32768 rows
(about10%) change; all non-finishing states, half of finishing states, Rival and
ball initial states, boost/state history and family/focal metadata remain exact.

For each selected row, interpolate opponent canonical xy between the diagnosed
recovering position and its original keeper position, alpha uniformly .35-.85.
The recovering flank is independently chosen at x+/-2200,y=ball_y-1300. The car
is grounded and travels600uu/s toward its goal with the existing coherent-route
heading jitter. Reject overlapping/out-of-field candidates before any rollout;
at most32 deterministic draws per row, never a silent fallback. The complete
selected row/alpha/draw ledger is in `training_bank_audit.json`.

The opponent stays active. No scripted Rival prefix, task identifier, router,
disabled goalkeeper or easy open-net-only majority. This change introduces a
continuum of pressure while retaining hard established-keeper starts.

## Same learner and safety

Exact source650:
`checkpoints/rival2/direct_skills_v1/plus_000650.pt`
SHA256 `4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F`.
Preserve model,Adam144648 cumulative steps,fourRNG and entity293 ancestry. No
fresh initialization, optimizer reset, BC, priorV5 model or inference change.
Fresh physical episodes/zero recurrent state at resume remain the declared rule.

Rewards are byte-unchanged: ordinary natural PBRS+goals and the existing bounded
skill events+goals. Do not silently reinterpret projected on-target as actual
goals; the unchanged evaluation will judge actual scores. PPO remains32768worlds,
90decisions,30Hzpolicy/120Hzphysics,4-tickhold,two epochs,actorLR1e-4,criticLR3e-4,
gamma.995,lambda.9973145188572297,temperature2 in both sampling and likelihoods.
HalfworldsNexto/halfselfplay gives one-third learner decisions against Nexto.
KL stays telemetry only. Finite and whole-update corruption protection remain.

The runner reuses the exact previous loader, collector and PPO call sites and
the exact complete evaluation body. Checkpoints retain historical root package
as ancestry plus explicit effective training-bank/curriculum hashes. Training
uses the new bank; evaluation uses the ORIGINAL cases, not easier replacements.

## Validation and boundary

Twenty focused CPU tests pass: deterministic32,768-world geometry, opponent-only
changes, exact unchanged states, loader/PPO/evaluation identity, active bank use,
and safe resume/STOP behavior. Initial fixture setup encountered access denied
in Windows' shared pytest temp directory:7tests passed,13fixtures errored before
their tests. The error XML is preserved. A fresh workspace-local `--basetemp`
resolved that operational test issue without changing production/test semantics.

The1024-world no-optimizer preflight passes15 checks, including unchanged parent/
Adam/Nexto, exact action table/logit scaling, finite gradients, critic isolation,
native reward path and exact Nexto sample share. Replay error is telemetry, not
a newly invented KL/replay guard. No new physics kernels or reward code are used.

Publish and verify authority/package/code/tests/preflight before learning. Save
the actual pre-step entry650 and first accepted651 in addition to rolling every
update. At675 save a permanent checkpoint, run the original64cases/family and
ten regulation Nexto matches, then pause for review against650/600. No automatic
longer run and no cherry-picking older checkpoints. The user's SSL goal continues.

```powershell
.venv\Scripts\python.exe benchmarks/run_direct_skills_shooting_progress_v1.py preflight
.venv\Scripts\python.exe benchmarks/run_direct_skills_shooting_progress_v1.py prepare
# Commit/push/readback. Archive only the verified agent650 review STOP.
.venv\Scripts\python.exe -u benchmarks/run_direct_skills_shooting_progress_v1.py run --resume checkpoints/rival2/direct_skills_v1/plus_000650.pt --resume-sha256 4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F
```

Use explicit latest path/SHA for any real operational recovery. Never clear an
unknown/userSTOP or restart healthy processes because an observer timed out.
The expired600to650 runner is not the entrypoint for this campaign.
