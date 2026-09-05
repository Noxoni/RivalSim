# Post-pilot natural-gameplay comparison

Frozen during the ongoing pilot, before +50/+100 outcomes. This does not alter
the training authority, budget, rewards, checkpoint selection, or existing
development evaluations. The fixed +100 checkpoint and immutable hybrid parent
u597 will each play10 full matches against the existing deterministic Nexto
adapter: all five standard kickoff layouts on both teams, identical seeds.

The existing `FullMatchState`, `FullMatchTelemetry`, native world and kickoff
reset implementation own match lifecycle. Regulation is300 seconds; tied
matches may receive up to120 additional seconds of sudden-death overtime.
Unresolved matches are reported explicitly, never as wins. There are no
15-second no-touch truncations or curriculum resets. Consequently zero
no-touch *truncations* is a protocol property, not evidence of acquisition;
matches without a Rival touch and native touch totals are reported instead.

The separate read-only subclass loads each real architecture directly. It does
not load a legacy/V5 dummy checkpoint, mutate weights, route specialist models,
or fit anything. Rival acts deterministically at30Hz with4-tick holds, Nexto
at its existing15Hz cadence. GRU state is continuous between native kickoffs
and reset on native kickoff boundaries. Native182-field observations and
previous-action history are passed directly, without human-domain adaptation.

`full_match_protocol.json` binds seeds, layout/team assignments, source hashes
and focused CPU test evidence. The evaluator refuses execution before the
training status reaches completed+100, and must acquire the training process's
exclusive lease; it cannot compete with a healthy training worker. It verifies
published source/authority bytes and checkpoint hashes before evaluation.
Completed per-policy results are retained if the second policy has an
operational interruption, rather than silently rerunning the completed arm.

The three focused tests verify balanced assignments, recurrent lane resets and
perspective-correct scoring with unresolved matches. They do not claim a native
full-match smoke has run. GPU execution and checkpoint/hash parity must still
be verified when training finishes. Any implementation error will be reported
and corrected explicitly, without changing the evaluation question or policy.

Command after the pilot has completed and the worker has exited:

```
.venv\Scripts\python.exe benchmarks/evaluate_rival2_ssl_entity_full_match.py run
```

This is a small deterministic development comparison, not SSL certification,
human rank equivalence, or a deployment decision. No online/ranked play occurs.
