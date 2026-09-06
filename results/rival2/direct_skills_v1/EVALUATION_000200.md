# Direct Skills +200: better match scores, worse finishing drill

## Integrity and scope

- Scheduled checkpoint: `checkpoints/rival2/direct_skills_v1/plus_000200.pt`.
- SHA256: `F150BDFA64F4F69BBED6E1D9318F449CCBAEA5A6169C690349558BCC6FCB0AFA`.
- 200 accepted updates, 884,736,000 new learner decisions,
  2,359,296,000 physical world ticks, 28,434 new Adam steps (80,584 cumulative).
- The fourteen existing checkpoint/evaluation audit checks passed: exact parent,
  authority/contracts, finite model/Adam, contiguous updates, correct counters,
  exactly one-third Nexto learner exposure and zero optimizer steps in evaluation.
- All twenty frozen runtime source hashes still match the prospective package.
  No training, reward, scenario, model, optimizer, exploration, evaluation seed
  or cadence changes occurred in this block.
- Across updates 1–200, completed mean KL ranges from
  0.002029732916234765 to 0.005317286969018426. Maximum completed sample KL is
  33.80954360961914. KL remains telemetry only; no KL rejection or KL rollback.
  Finite checks remain enabled. No new operational or nonfinite failure occurred.
- The same worker completed the scheduled evaluation and resumed training.
  All postprocessing here is CPU-only; no extra policy evaluation was run.

This preserves an independently recoverable checkpoint, not a cherry-picked
best policy, deployment, SSL promotion or completion of the broader goal.

## Ten full regulation matches against Nexto

All eleven existing native full-match integrity checks passed: ten complete
regulation matches, goals/scorers/scoreboards conserved, recurrent reset at each
native goal, no overflow, unchanged policy/checkpoint and no optimizer step.

| Metric | Baseline | +50 | +100 | +150 | +200 |
|---|---:|---:|---:|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| Goals for | 1 | 17 | 12 | 24 | 33 |
| Goals against | 266 | 212 | 209 | 203 | 187 |
| Rival contacts | 170 | 313 | 366 | 368 | 352 |
| Rival contacts/min | 3.40 | 6.26 | 7.32 | 7.36 | 7.04 |
| Nexto contacts | 2,636 | 1,765 | 2,019 | 2,179 | 2,066 |
| Same-player follow-ups / next-contact resolutions | 0 / 166 | 40 / 304 | 37 / 350 | 25 / 339 | 26 / 317 |
| Follow-up fraction | 0% | 13.16% | 10.57% | 7.37% | 8.20% |
| Kickoff first contacts | 0 | 0 | 2 | 1 | 3 |
| Matches without Rival contact | 0 | 0 | 0 | 0 | 0 |
| Concessions with at most one contact since reset | 52 | 0 | 9 | 6 | 6 |

Full-match scoring improves over the three recent reviews, with fewer
concessions. Against +150, six matched initial cases score more and four score
less. Against +100, eight improve, one worsens, one is unchanged; against +50,
six improve, two worsen, two are unchanged. Five of the nine extra goals since
+150 come from one match (index 3). Do not turn this small, heterogeneous
deterministic development comparison into statistical evidence of broad strength.
All matches remain losses; Nexto scores roughly 5.7 times as often as Rival.

| Match | Rival side | Initial layout | Rival goals | Nexto goals | Rival contacts | Same-player follow-ups |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 | 19 | 34 | 2 |
| 1 | 0 | 1 | 2 | 19 | 33 | 2 |
| 2 | 0 | 2 | 2 | 19 | 30 | 1 |
| 3 | 0 | 3 | 10 | 14 | 36 | 0 |
| 4 | 0 | 4 | 0 | 21 | 34 | 4 |
| 5 | 1 | 0 | 3 | 19 | 40 | 5 |
| 6 | 1 | 1 | 4 | 20 | 34 | 1 |
| 7 | 1 | 2 | 1 | 19 | 38 | 5 |
| 8 | 1 | 3 | 4 | 18 | 40 | 4 |
| 9 | 1 | 4 | 3 | 19 | 33 | 2 |

### Contact resolution

The follow-up denominator excludes goals, as audited in
[the +150 review](EVALUATION_000150.md). The same recorded-counter partition is:

| Rival contact interval ends at | +50 | +100 | +150 | +200 |
|---|---:|---:|---:|---:|
| Rival's next distinct contact | 40 | 37 | 25 | 26 |
| Opponent's next distinct contact | 264 | 313 | 314 | 291 |
| Goal before another contact | 8 | 14 | 25 | 34 |
| Unclosed at match end | 1 | 2 | 4 | 1 |
| Total | 313 | 366 | 368 | 352 |

Compute goal-ended intervals as `sum(displacement_count) - possession_total`,
and unclosed intervals as `touch_count - sum(displacement_count)` for each
player. Both players' goal-ended intervals sum to all native goals in every
review. Those counters do not associate last toucher and scoring team for each
goal, so not every goal-ended Rival interval can be called a successful shot.

Nexto still takes 291 of 317 resolved next contacts. The slight fraction recovery
does not establish reliable possession. Canonical velocity change at contact is
9 backward, 5 neutral, 338 forward; net displacement until next contact/goal is
32 backward, 159 neutral, 160 forward. These are not resulting ball velocity or
a shot/possession classifier. Native Rival demolition count is zero; there is
no claim of intentional demos. No-touch resets are disabled by this match
protocol, so their zero count is not a behavioral accomplishment.

## Fixed skill tests: 64 cases per family

| Metric | Baseline | +50 | +100 | +150 | +200 |
|---|---:|---:|---:|---:|---:|
| Natural goals for / against | 2 / 62 | 17 / 47 | 15 / 49 | 15 / 49 | 21 / 41 |
| Natural cases with contact | 23 | 31 | 53 | 49 | 51 |
| Challenge goals for / against | 0 / 53 | 0 / 48 | 0 / 54 | 2 / 52 | 2 / 50 |
| Challenge control gains | 3 | 2 | 1 | 3 | 2 |
| Challenge controlled advances | 2 | 2 | 1 | 3 | 2 |
| Finishing goals for / against | 3 / 55 | 5 / 56 | 8 / 47 | 9 / 42 | 5 / 52 |
| Finishing goal-bound contacts | 52 | 52 | 51 | 48 | 56 |
| Defense goals for / against | 2 / 36 | 5 / 36 | 4 / 33 | 5 / 30 | 4 / 31 |
| Defensive clear events | 32 | 31 | 32 | 37 | 35 |
| Kickoff goals for / against | 0 / 52 | 26 / 38 | 26 / 0 | 26 / 0 | 39 / 0 |
| Kickoff control gains | 0 | 0 | 0 | 0 | 0 |
| Kickoff timeouts | 12 | 0 | 38 | 38 | 25 |

### Shooting regression: actual outcomes, not proxy success

All 64 finishing cases still contact the ball. Goal-bound contacts increase from
48 to 56, but scored attempts fall from nine to five and concessions increase
by ten. The proxy is not a goal or a solved finishing skill. Four previous
scoring cases remain (zero-based indices 7, 22, 37, 41); case 52 is newly scored.
Five previous scores are lost:

| Case | +200 outcome | Rival contacts | Episode duration, seconds |
|---|---|---:|---:|
| 5 | Concede | 2 | 8.566668 |
| 11 | Timeout | 1 | 12.000047 |
| 16 | Concede | 2 | 6.799995 |
| 33 | Concede | 1 | 9.066674 |
| 43 | Concede | 1 | 8.466666 |

The failure is downstream of making contact, not an inability to reach these
nearby balls. Saved aggregate outcomes do not identify the precise contact
geometry or subsequent bad action; do not invent that diagnosis. The reward
logic and source hashes did not change. No new detector, bonus or retuned
threshold was introduced to hide the regression. The 20% finishing source
family already contains shooting situations and has not been duplicated.

### Kickoff and ongoing-ground separation

- Layout 1 improves from thirteen failed timeouts to thirteen goals. Together
  with layouts 0 and 3, it supplies the 39 kickoff scores. Layouts 2 and 4 still
  fail their twelve-second attempts and concede in the longer natural suite.
- Kickoff control remains zero. Contacting after Nexto or scoring from a
  repeated layout is not proof of intentional faking or possession control.
- These cases repeat five layouts/two sides; do not count them as sixty-four
  independent generalization examples.
- The 29 ongoing-ground starts change from 14 to 16 touched cases and 20 to 21
  contacts. Their goals for/against change from 1/28 to 1/26 with two timeouts.
  There is still only one score. The natural suite's six additional scores
  come from the improved kickoff layout, not extra ongoing-ground scores.
- Challenge control is two cases and both later register lost-control events.
  Defense is slightly worse on goals and clears than +150, despite contacting
  63 rather than 62 cases. No broad acquisition/control mastery is established.

## Training block: useful context, not evaluation replacement

| Statistic | Updates 101–150 | 151–200 |
|---|---:|---:|
| Mean contacts per learner-minute | 9.1117 | 9.9465 |
| Mean movement speed, uu/s | 1,128.83 | 1,150.70 |
| Mean categorical entropy | 1.2520 | 0.9993 |
| Mean ended player episodes with contact | 83.40% | 84.04% |
| Natural Nexto goals / concessions | 8,495 / 65,949 | 12,418 / 62,083 |
| Challenge Nexto control / forward-control events | 1,033 / 459 | 1,398 / 705 |
| Kickoff Nexto control / forward-control events | 244 / 59 | 372 / 99 |
| Finishing Nexto goals | 3,373 | 3,655 |
| Defense Nexto control gains | 679 | 1,280 |
| Challenge self-play control / forward-control | 6,938 / 3,302 | 10,069 / 5,771 |

The newest block includes 11,865,905 challenge, 11,547,097 kickoff and 12,630,262
finishing Nexto learner decisions. Positive control remains sparse, even though
the training counts rise. Categorical entropy continues declining, not failing
a numerical guard. Neither stochastic training goals nor aggregate proxy counts
override the deterministic finishing regression. All per-update data are in the
immutable `training_curve_through_000200.jsonl` prefix.

## Decision and next review

Continue unchanged to **+250**, with the same user-authorized indefinite run and
fifty-update review schedule. Full-match goals improve for a second consecutive
review and concessions fall; that warrants the next block. It does not warrant
calling every competency better or retraining the objectives in response to
each fluctuating test. This checkpoint has no win over Nexto and no SSL verdict.

At +250 explicitly compare finishing with both +150 and +200, along with
challenge control, ongoing-ground outcomes, each kickoff layout and the complete
matches. Persistent finishing regression or stalled match improvement while
control stays sparse requires focused trajectory-level diagnosis before another
large unchanged block. No automatic reward retune, architecture change, new
lineage, optimizer reset, specialist router or deployment is authorized by this
review. Preserve +150 as well as +200, with no cherry-picked intermediate test.

## Reproduction

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_direct_skills.py --update 200
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 200
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 200 --baseline 50
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 200 --baseline 100
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 200 --baseline 150
```

Repeat those comparison commands with `--start-groups` for the exact source-hash
verified initial-state grouping. Use existing `reduce(path)` in
`benchmarks/report_rival2_ssl_entity_match_followup.py` on the saved full-match
file, not that older campaign's CLI main. Training figures are fifty-row
sums/means from the immutable curve; finishing case indices are set differences
of `raw.goals` in evaluations +150/+200. No further policy run is needed.
