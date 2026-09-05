# Reproducible completed-boundary audit

The prior +150/+200 evidence was produced by inline CPU inspections. The standalone
`benchmarks/audit_rival2_ssl_entity_boundary.py` reproduces their counters from a
completed scheduled checkpoint, its evaluation, the immutable +100 parent, and
the accepted-update log. It never constructs a policy or optimizer, imports the
simulator, runs GPU work, changes guards, or chooses checkpoints.

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_ssl_entity_boundary.py --update 200
# After the scheduled +250 evaluation has actually completed:
.venv/Scripts/python.exe benchmarks/audit_rival2_ssl_entity_boundary.py --update 250
```

It verifies parent/checkpoint/authority hashes, frozen runtime source hashes,
observation/action/model identities, immutable buffers, finite model/Adam state,
unchanged optimizer configuration, cumulative Adam steps from every accepted
update, learner-only action counts, ignored Nexto slot conservation, physical
exposure, and evaluation/checkpoint opponent-schedule agreement. Large finite
KL is never an audit failure; a nonzero KL-rejection count would contradict the
existing authority. No safety threshold or training behavior is introduced here.

Output is `boundary_<update>_audit.json` plus an immutable LF-normalized curve
prefix if one does not already exist. Existing frozen prefixes must match exactly
and are never rewritten. Later rows and an incomplete append tail are excluded.
Missing or duplicate required updates fail the audit, not the running learner.

The +200 replay reproduces the previously committed checkpoint SHA, exact curve
hash, 573,699,794 continuation samples, 16,124,206 current-Rival-versus-Nexto samples,
and 36,018 cumulative Adam steps. This confirms evidence accounting, not SSL
capability. Exact restoration of historical Adam tensors at the +200 restart
remains separately established by `resume_after_match_000200_integrity.json`;
step counts alone would not prove that equality.

Ten targeted CPU tests cover sample masking/counting, variable optimizer steps,
large-KL telemetry, duplicate/gap detection, bad action counts, optimizer counter
discontinuity, unexpected KL rejection, nonfinite values, and partial/live log
handling. The +200 real checkpoint audit also passes. Training stays uninterrupted
under the unchanged prospective authority.
