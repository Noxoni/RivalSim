# Direct Skills +150: scoring improves; sustained control remains weak

## Checkpoint and integrity

- Snapshot: `checkpoints/rival2/direct_skills_v1/plus_000150.pt`.
- SHA256: `AF11FC2935BEB63322E8B49670EFAE0D87EE6430C424B2AC0F861C1C261EC5A3`.
- 150 accepted updates, 663,552,000 new learner decisions,
  1,769,472,000 physical world ticks, 21,330 new Adam steps (73,480 cumulative).
- All fourteen existing checkpoint/evaluation audit checks passed. Parent,
  contracts, sample masks, counters and finite model/Adam are intact. Every
  update contains 4,423,680 learner decisions, exactly one third versus Nexto.
- All twenty package runtime source hashes remain unchanged. No reward,
  curriculum, model, optimizer, exploration, evaluation seed or cadence change
  was made during this block.
- Completed mean KL across updates 1–150 ranges from
  0.0023028537026093636 to 0.005317286969018426. Maximum completed sample KL is
  17.064023971557617. KL is telemetry only; zero KL rejections. Finite guards
  remain active, with no new operational or numerical failure.
- The scheduled skill and match evaluations used this exact snapshot, took
  zero optimizer steps, and returned cleanly to the same training worker.

This is a scheduled, independently recoverable review snapshot, not a selected
best model or SSL/deployment promotion. Earlier checkpoints remain preserved.

## Ten full regulation matches against Nexto

All eleven existing match integrity checks passed, including goal/scorer/score
conservation, regulation completion, recurrent reset per native goal, no goal
overflow, and unchanged model/checkpoint. No extra GPU evaluation was run.

| Metric | Baseline | +50 | +100 | +150 |
|---|---:|---:|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| Goals for | 1 | 17 | 12 | 24 |
| Goals against | 266 | 212 | 209 | 203 |
| Rival contacts | 170 | 313 | 366 | 368 |
| Rival contacts/min | 3.40 | 6.26 | 7.32 | 7.36 |
| Nexto contacts | 2,636 | 1,765 | 2,019 | 2,179 |
| Same-player follow-ups / next-contact resolutions | 0 / 166 | 40 / 304 | 37 / 350 | 25 / 339 |
| Same-player follow-up fraction | 0% | 13.16% | 10.57% | 7.37% |
| Kickoff first contacts | 0 | 0 | 2 | 1 |
| Matches without Rival contact | 0 | 0 | 0 | 0 |
| Concessions with at most one contact since reset | 52 | 0 | 9 | 6 |

Scoring is better than both recent reviews: seven matched starting cases score
more than +100, one scores less, and two are unchanged. Against +50, six score
more, three less, one is unchanged. This remains a small deterministic
development comparison, not statistical proof of broad strength. There is no
win, and Nexto still scores approximately 8.5 goals for each Rival goal.

| Match | Rival side | Initial layout | Rival goals | Nexto goals | Rival contacts | Same-player follow-ups |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2 | 20 | 34 | 3 |
| 1 | 0 | 1 | 1 | 21 | 37 | 3 |
| 2 | 0 | 2 | 3 | 18 | 32 | 4 |
| 3 | 0 | 3 | 5 | 24 | 44 | 3 |
| 4 | 0 | 4 | 2 | 21 | 36 | 0 |
| 5 | 1 | 0 | 4 | 21 | 45 | 4 |
| 6 | 1 | 1 | 1 | 21 | 39 | 3 |
| 7 | 1 | 2 | 3 | 21 | 40 | 2 |
| 8 | 1 | 3 | 2 | 19 | 32 | 0 |
| 9 | 1 | 4 | 1 | 17 | 29 | 3 |

### Contact-resolution diagnostic and limits

The recorded follow-up fraction falls again, and Nexto takes the next distinct
contact in 314 of 339 resolved cases. Sustained control is plainly not proven.
However, calling that fraction alone a pure possession regression is too strong:
`rivalsim/full_match.py` increments `possession_total/same/opponent` only on the
next distinct contact. A goal closes the displacement interval **without**
incrementing those counters. Match-end intervals may remain unclosed.

The saved counters give the following exact partition without another rollout:

| Rival contact interval ends at | Baseline | +50 | +100 | +150 |
|---|---:|---:|---:|---:|
| Rival's next distinct contact | 0 | 40 | 37 | 25 |
| Opponent's next distinct contact | 166 | 264 | 313 | 314 |
| Goal before another contact | 1 | 8 | 14 | 25 |
| Still unclosed at match end | 3 | 1 | 2 | 4 |
| Total | 170 | 313 | 366 | 368 |

Goal-ended intervals equal `sum(displacement_count) - possession_total` for
each player. Unclosed intervals equal `touch_count - sum(displacement_count)`.
Both players' goal-ended intervals sum to every recorded goal in all four
reviews. Goal-ended Rival intervals grow by eleven while same-player follow-ups
fall by twelve from +100. The stored aggregates do not bind the last toucher to
the scoring team for each goal, so these twenty-five intervals cannot all be
called Rival shots or positive possession outcomes. They explain a denominator
confound, not an excuse to claim control improved. No detector or reward changed.

At +150, canonical velocity-change categories at Rival contact are 3 backward,
8 neutral and 357 forward. Net ball displacement until next contact/goal is
25 backward, 212 neutral and 127 forward. These are not resulting velocity,
exact shot success, or time in possession. Two native Rival demolitions occurred;
they do not establish deliberate demolition tactics. The no-touch-reset count
is zero by match protocol, not an acquired skill. Early-concession counts use
distinct contacts since reset, not a timed kickoff/fake classifier.

## Fixed skill episodes: 64 per family

| Metric | Baseline | +50 | +100 | +150 |
|---|---:|---:|---:|---:|
| Natural goals for / against | 2 / 62 | 17 / 47 | 15 / 49 | 15 / 49 |
| Natural cases with contact | 23 | 31 | 53 | 49 |
| Challenge goals for / against | 0 / 53 | 0 / 48 | 0 / 54 | 2 / 52 |
| Challenge controlled acquisitions | 3 | 2 | 1 | 3 |
| Challenge controlled advances | 2 | 2 | 1 | 3 |
| Challenge contacts | 69 | 82 | 74 | 90 |
| Finishing goals for / against | 3 / 55 | 5 / 56 | 8 / 47 | 9 / 42 |
| Finishing goal-bound contacts | 52 | 52 | 51 | 48 |
| Defense goals for / against | 2 / 36 | 5 / 36 | 4 / 33 | 5 / 30 |
| Defensive shot-clear events | 32 | 31 | 32 | 37 |
| Kickoff goals for / against | 0 / 52 | 26 / 38 | 26 / 0 | 26 / 0 |
| Kickoff contacts | 25 | 62 | 69 | 102 |
| Kickoff controlled acquisitions | 0 | 0 | 0 | 0 |
| Kickoff timeouts | 12 | 0 | 38 | 38 |

Shooting is already a first-class finishing family, 20% of episode-source
states: varied balls near the opponent goal, Rival behind/offset from the ball,
and a real defender. It has a bounded goal-bound-contact bonus and larger native
goal outcome reward. No new shooting family was needed or added in response to
the user's reminder. Scoring nine of sixty-four fixed attempts is modest
progress, not reliable finishing. Fewer proxy on-target events alongside more
goals again show why those quantities must remain separate.

Challenge control improves over +100 but merely returns to the baseline's
three events. Two acquisitions are subsequently lost. Defense improves modestly.
The ongoing-ground subset of natural starts regresses: touched cases **18 to
14**, contacts **24 to 20**, goals for/against unchanged at **1/28**. The overall
natural contact increase comes from repeated contacts on kickoff layouts, not
broader acquisition of the randomly placed ball.

The kickoff successes remain confined to layouts 0 and 3. Layouts 1, 2 and 4
still all fail at the twelve-second timeout and concede in the longer natural
suite. Layout 1 now produces three contacts per short case, but no control
acquisition. The sixty-four cases repeat five layouts/two sides and must not be
treated as sixty-four independent generalization cases. There is no proof of
intentional fakes, fast possession wins, or solved kickoff play.

## Focused training evidence

| Mean training statistic | Updates 1–50 | 51–100 | 101–150 |
|---|---:|---:|---:|
| Contacts per learner-minute | 7.8016 | 8.4666 | 9.1117 |
| Movement speed, uu/s | 1,052.57 | 1,084.27 | 1,128.83 |
| Categorical entropy | 1.8650 | 1.5939 | 1.2520 |
| Ended player episodes with contact | 76.08% | 80.73% | 83.40% |
| Natural Nexto training goals | 2,826 | 4,559 | 8,495 |
| Natural Nexto training concessions | 71,249 | 70,394 | 65,949 |
| Challenge Nexto control gains | 700 | 875 | 1,033 |
| Challenge Nexto controlled advances | 242 | 386 | 459 |
| Kickoff Nexto control gains | 103 | 202 | 244 |
| Finishing Nexto goals | 3,023 | 3,345 | 3,373 |

The final block has 11,661,660 challenge and 11,980,981 kickoff Nexto learner
decisions. These positive control trajectories remain sparse but are neither
absent nor declining in training. Challenge self-play control gains increase
4,653 to 5,771 to 6,938 over similar exposure. Finishing Nexto training scoring
is nearly flat despite the small fixed-case improvement. Entropy declines;
that is a reported distribution change, not by itself a collapse diagnosis or
authority to retune exploration. Training counts are stochastic and changing
state distributions; the fixed full matches remain the gameplay comparison.

## Review decision

Continue the unchanged, user-authorized learner to the scheduled **+200** review.
The +100 concern received a focused check here: successful control remains
sparse and ongoing-ground acquisition regresses, but actual full-match scoring
now improves beyond both +50 and +100, concessions decrease, and defense and
challenge outcomes show small gains. This justifies the next scheduled block,
not an assumption that current rewards will eventually produce SSL.

At +200 compare against +50, +100 and +150, including goal-ended versus
next-contact intervals, ongoing-ground acquisition, challenge control and
finishing conversion. If scoring stalls or deteriorates while control remains
sparse, investigate that gap before another large unchanged block. Do not
automatically change rewards, replay a lucky intermediate checkpoint, reset
Adam, add a router, or deploy the model. No goal of beating Nexto/SSL is met.

## Reproduction

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_direct_skills.py --update 150
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 150
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 150 --baseline 50
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 150 --baseline 100
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 150 --start-groups
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 150 --baseline 50 --start-groups
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 150 --baseline 100 --start-groups
```

Use the existing `reduce(path)` in
`benchmarks/report_rival2_ssl_entity_match_followup.py` for the saved full-match
file, not that older campaign's CLI entry point. Training aggregates are simple
fifty-row sums/means of the immutable `training_curve_through_000150.jsonl`.
All evidence generation here is CPU-only, with no optimizer or new policy run.
