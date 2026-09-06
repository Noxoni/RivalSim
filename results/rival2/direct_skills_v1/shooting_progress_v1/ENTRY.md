# Actual650 entry and first accepted651

Launched the prospectively published curriculum-only block at
2026-09-06T10:40:00Z. Actual learner PID20016, child of45312. Source650 and its
optimizer were not replaced or reset. The known agent650 review STOP was moved
to `STOP.review650.completed`; no user/unknown STOP was cleared.

The eleven files of prospective commit
`957a039f634a4b9ab67eab993b9d1a47604ff912` were read back from origin/main before
launch. Authority SHA256:
`239B36875B332F4E83CF32261A7BBE2AEDDF57D05D0045A5724A8803E8F5516C`.

Actual pre-step snapshot:
`checkpoints/rival2/direct_skills_v1/shooting_entry_000650.pt`
SHA256 `19776A26866DBEB3320CCFFFA982539DCEE00079B1A3F7C498B8C1182661B6DF`.
Model tensors, full Adam state, all four RNG states and training counters are
byte-identical to source650. Tensor checks compare uint8 views, including signed
zero, rather than merely numerically equal floats. Metadata correctly binds the
new curriculum and effective bank while preserving root lineage/contracts.

First accepted651 snapshot:
`checkpoints/rival2/direct_skills_v1/shooting_resume_000651.pt`
SHA256 `D941BBF5D305519A369745EEAFD56F77B60F848CEEC2EBF52363B43CB9F9F96F`.
Exactly136 new Adam steps,144784 cumulative;4,423,680 new learner decisions and
11,796,480 physical world ticks. All Adam tensor step counters agree. Model and
optimizer remain finite. The first row uses temperature2 and exactlyone-third
Nexto learner exposure. The first fresh-reset buffer is not a capability trend.

`entry_and_first_update_audit.json` stores20 successful checks, actual hashes and
the complete651 telemetry row. Two CPU tests verify bitwise comparison and exact
rebuild. No optimizer step occurred in the audit. The existing20prospective CPU
tests and15native no-step preflight checks remain the implementation evidence.
Raw pytest permission-error XML preserves its original whitespace; diff-check
warnings in that archived error output are not changes to code or training.

The monitor was updated using official OpenAI scheduling guidance to follow the
new entrypoint, resume safeguards and675 review, not restart the expired650run.
Notifications remain limited to new evaluations or actionable findings. Training
continues to675; this entry verification does not claim improved gameplay orSSL.

```powershell
.venv\Scripts\python.exe benchmarks/audit_direct_skills_shooting_entry.py
.venv\Scripts\python.exe -m pytest -q tests/test_direct_skills_shooting_entry.py
```
