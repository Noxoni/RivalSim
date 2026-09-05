# Development evaluations: entity-aware categorical candidate

All entries use the same original fixed64-case development corpus and exact
deterministic controls. They are not full-match win rates. Table entries below
are numbers of focal cases with a touch, or goals, not percentages.

| Policy / offset | Acquisition cases touched | Acquisition no-touch timeouts | Finishing goals | Nexto kickoff cases touched | Nexto goals for / against |
|---|---:|---:|---:|---:|---:|
| Immutable hybrid parent u597 | 16 | 61 | 14 | 13 | 0 / 64 |
| Entity/joint-control initialization | 11 | 61 | 11 | 26 | 0 / 64 |
| Entity candidate +10 | 20 | 58 | 14 | 13 | 0 / 64 |
| Entity candidate +20 | 18 | 58 | 14 | 13 | 0 / 64 |
| Entity candidate +50 | 35 | 55 | 17 | 13 | 0 / 64 |

The +10 acquisition change is +9cases relative to the changed-control
initialization, and +4cases relative to the original hybrid parent. Finishing
recovers to the parent result, not beyond it. Nexto kickoff contact coverage
falls back to the parent result and scoring remains wholly one-sided. Some
scenarios touch earlier then later hit a no-touch timeout, so these two counts
are not complements. The acquisition conditional median touch time is0.892s;
that excludes the44 failed-acquisition cases and is not evidence of reliable
quick acquisition by itself.

This is an encouraging early development signal for basic acquisition only.
The64-case aggregate does not establish statistical generalization, causality
for attention alone, sustained progress, or competence against Nexto. No model
is promoted or deployed from this intermediate result. Continue the frozen
100-update evidence budget and retain later results, including regressions.

`progress_report.json` contains the full numerical comparison and source hashes.
The immutable +10 checkpoint matches the evaluation checkpoint SHA. The full
10-update parameter/optimizer integrity audit is separate from capability claims.

At +20 acquisition is two cases lower than +10, while still seven above the
changed-control initialization and two above the original parent. Finishing
touch coverage rises from44/64 at +10 to48/64, but finishing goals remain14.
Nexto results are unchanged. This is not a monotonic improvement curve and does
not establish competitive progress. The immutable +20 checkpoint and its
optimizer/parameter audit are retained; all expected parameter groups changed,
all Adam counters equal3640, and model/optimizer values remain finite.

At +50 the acquisition coverage rises to35/64 (+19 versus the original parent,
+24 versus the changed-control initialization, +17 versus +20). Total contact
events are38; acquisition goals are7 for /2 against. Finishing coverage is53/64
with65 total contacts and17 goals, exceeding the parent's46/64 and14 goals.
No-touch resets decline modestly in both families. This is materially stronger
basic-acquisition evidence on the fixed development cases than the +10/+20
results, but only one such boundary so far; it is not independent generalization
or proof that attention alone caused the improvement.

Nexto still produces64 goals against Rival and Rival scores0. Contact coverage
there remains13/64, with26 total contacts. Do not call the improved self-play
scenario scores competitive progress against Nexto. No deployment or opponent
mixture change is made; complete the prospectively frozen +100 pilot and the
fixed post-pilot match comparison. The +50 model/optimizer integrity audit
passes, with9100 steps per Adam counter and no nonfinite states. All50 detailed
training rows are independently preserved in `training_summary_050.json`.
