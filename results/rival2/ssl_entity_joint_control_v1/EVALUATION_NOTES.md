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
| Entity candidate +100 | 51 | 50 | 35 | 13 | 0 / 64 |

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

## Completed initial pilot: +100

Acquisition coverage is 51/64 (79.6875%), with 59 contacts and 10 goals for /
2 against. Finishing coverage is 62/64, with 82 contacts and 35 goals for /
0 against. Both exceed the original parent and +50 on these fixed development
cases. Conditional median first-touch times are 0.966667s and 0.525000s; they
exclude cases that never touched. No-touch timeouts remain common (50/64 in
acquisition), including episodes that touched earlier but could not sustain play.

Nexto kickoff outcomes remain 0 goals for / 64 against, 13/64 touched and 26
contacts. Therefore the gain is in basic acquisition/finishing, not demonstrated
Nexto strength or SSL gameplay. No lucky intermediate checkpoint is selected.
The frozen +100 versus original-parent full-match comparison remains pending.

All 100 PPO updates completed. The fixed +100 checkpoint SHA-256 is
`B5F7D19471257758966EB0B407797CFCFBD0F6CE8E86BCF9E13D41FBA4EA7ABA`.
The complete model and Adam state are finite, every Adam counter is 18,200,
all intended parameter groups updated, and the action/entity-map buffers and
original parent identity are preserved. The pilot added 589,824,000 trainable
player decisions. These integrity checks are not capability acceptance.
