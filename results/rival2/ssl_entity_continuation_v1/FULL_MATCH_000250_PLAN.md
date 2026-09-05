# Prospective fixed +250 natural Nexto follow-up

Reason: the completed +250 short evaluation increased cases with a Rival contact
but reduced total contacts and positive-ball-velocity fraction, with no Rival
goals. Investigate that mixed signal in actual continuous games before drawing a
general gameplay conclusion. No learning configuration change is proposed here.

Freeze and publish `full_match_000250_protocol.json`, the helper sources/tests,
the exact +250 snapshot, and its short evaluation before running this follow-up.
The helper verifies remote publication and hashes. The short evaluation is already
opened development data; this is explicitly not an untouched acceptance test.

- Target: fixed entity +250, SHA
  `22C3D4762784F7AC9D48DD429DFDFCA7E88333DFC2DABE2B11A9C01ED4C53878`.
- Baseline: already-completed +200 full matches; never rerun or select a baseline.
- Reuse `CandidateMatchRunner`, its complete original method, ten regulation
  matches, five kickoff layouts on both sides, fixed seed 2026090573, 36,000
  physics ticks per regulation match and the same 14,400-tick overtime cap.
- Deterministic joint-action argmax, 30 Hz Rival, 15 Hz Nexto, 120 Hz physics.
- No curriculum/no-touch resets, scripted Rival controls, new detectors, changed
  reward/physics, optimizer, policy modification, or checkpoint selection.
- Check goals, concessions, contacts, follow-up contact identity, displacement,
  kickoff outcomes and native reset integrity with existing counters only.

After publication, request a brief stop at the next accepted training boundary.
The learner may be beyond +250 by then. Record and hash that **latest accepted
resume checkpoint separately**; do not resume from the older evaluated +250 model
and discard subsequent learning. Require the exact agent-owned marker:
`AGENT_OWNED_MATCH_FOLLOWUP_000250`. Never reinterpret a user STOP as evaluation
permission. Wait for the learner to exit and release the shared GPU lease.

Run:

```powershell
.venv/Scripts/python.exe benchmarks/evaluate_rival2_ssl_entity_match_followup.py run --target 250 --baseline 200
```

On successful evaluation, verify the fixed +250 model/checkpoint and the separate
latest training checkpoint remain unchanged. Remove only the agent-owned pause
marker and resume the exact latest accepted weights/Adam/counters/RNG with the
existing explicit resume arguments. Fresh physical episodes/zero hidden are the
already documented resume semantics. No fresh optimizer or selection based on
these matches. A real numerical/corruption fault requires investigation instead.

The user-authorized continue-until-stopped campaign remains active. This method
neither introduces a new acceptance threshold nor claims SSL capability. Publish
the complete result, including regressions and zero scoring if that occurs.
