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
