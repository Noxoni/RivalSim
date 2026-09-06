# Direct Skills +500: more scoring, uneven transfer, finishing still weak

## Identity and integrity

- `checkpoints/rival2/direct_skills_v1/plus_000500.pt`, SHA256
  `2DF2D29E766E91E01EC08BAB7D50FB526F8EAA41490818470C9686B3760BCF5A`.
- 500 accepted updates; 2,211,840,000 direct learner decisions;
  5,898,240,000 physical world ticks; 123,356 cumulative Adam steps,
  71,206 new since the entering Entity293 optimizer.
- Fourteen checkpoint and eleven full-match integrity checks pass. All twenty
  frozen runtime sources and authority evidence match published versions.
  Action/observation/PPO contracts, finite model/Adam and exact one-third learner
  exposure to inference-only Nexto are verified. No KL rejection occurred.
- Thirty focused CPU tests pass. Twenty skill comparison/start-group reports
  against 0/50/100/150/200/250/300/350/400/450 rebuild identically.
- This review is CPU-only. The existing learner continues; no additional native
  evaluation, optimizer, reward adjustment or model selection occurs in review.

## Corrected same-method full matches

Compare with `corrected_reset/full_match_000451.json` and corrected +0, never
untagged old +400/+450 scores as a pure learning trend. Both runtime-package and
reset-method identities match and checkpoint hashes are verified.

| Ten fixed regulation matches | Corrected +0 | Corrected +451 | +500 |
|---|---:|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 | 0 / 10 |
| Rival goals | 1 | 59 | 70 |
| Nexto goals | 272 | 156 | 165 |
| Goal differential | -271 | -97 | -95 |
| Rival contacts | 203 | 538 | 431 |
| Contacts / minute | 4.06 | 10.76 | 8.62 |
| Same-player follow-ups / resolved | 0 / 200 | 142 / 482 | 58 / 357 |
| Kickoff first contacts | 0 | 10 | 2 |
| Matches with no Rival contacts | 0 | 0 | 0 |

Scoring increases by eleven, but concessions increase by nine. Goal differential
improves only two across fifty match-minutes. Six matched cases improve, four
worsen. Every case improves over entering +0, but every match still loses.
This is modest, uneven development—not convincing broad improvement, competitive
parity, SSL, or proof the possession/finishing objective is solved.

| Side / initial layout | +451 Rival / Nexto | +500 Rival / Nexto |
|---|---:|---:|
| Blue / 0 | 7 / 17 | 8 / 13 |
| Blue / 1 | 8 / 16 | 9 / 11 |
| Blue / 2 | 6 / 15 | 9 / 16 |
| Blue / 3 | 6 / 13 | 9 / 12 |
| Blue / 4 | 7 / 15 | 10 / 11 |
| Orange / 0 | 5 / 16 | 5 / 22 |
| Orange / 1 | 2 / 19 | 8 / 13 |
| Orange / 2 | 6 / 15 | 4 / 23 |
| Orange / 3 | 6 / 16 | 1 / 26 |
| Orange / 4 | 6 / 14 | 7 / 18 |

Blue aggregate improves 34/76 to 45/63; Orange remains 25 goals but worsens
80 to 102 concessions. All five Blue cases improve; four Orange cases worsen.
The fixed ten-case sample demonstrates this asymmetry, not its cause or a
statistically established team bug. Do not select only Blue results or silently
change canonicalization, opponents, physics or resets from this correlation.

Opening goals remain ten of ten; Rival second goals increase three to six.
Later goals increase 49 to 60 while later concessions increase 156 to 165.
Unequal segment exposure prevents reading these raw counts as equal-opportunity
success rates. No intentional fake-kickoff label is inferred.

Contact accounting is 58 same-player + 299 opponent + 68 goal-ended + 6 unclosed
= 431. Same-player follow-up fraction falls 29.46% to 16.25%; this is not
possession duration, but it does not support a claim of improved retention of
the ball. Nexto contacts rise 1,352 to 1,434. Net displacement until next contact
or goal is 55 backward / 192 neutral / 178 forward. Velocity-change bins are
7 / 4 / 420; positive velocity change is not shot accuracy. Zero concessions
have at most one contact since reset. No-touch resets remain disabled by
protocol; every match does contain Rival contacts. Zero native Rival demos are
recorded; no new mechanic or demo detector is introduced.

## Fixed skill cases, 64 per family

| Family | +450 goals / concedes / timeout | +500 goals / concedes / timeout |
|---|---:|---:|
| Natural | 37 / 26 / 1 | 39 / 24 / 1 |
| Challenge | 4 / 37 / 23 | 1 / 38 / 25 |
| Finishing | 5 / 41 / 18 | 4 / 43 / 17 |
| Defense | 5 / 24 / 35 | 8 / 20 / 36 |
| Kickoff | 64 / 0 / 0 | 64 / 0 / 0 |

Shooting remains 20% of scenario-source starts. All 64 finishing cases touch,
with conditional first-touch time 0.5346354246139526 seconds; contacts rise 87
to 99 and projected on-target events 45 to 49, but actual goals fall five to
four. Concessions rise 41 to 43. Do not treat the geometric proxy as successful
shooting or claim that more contacts solve the problem.

Challenge still touches in 54 cases, with 117 contacts versus 119; control gains
rise eight to ten, forward-control stays seven, but goals fall four to one and
failed-attempt events rise sixteen to twenty-one. Defense touches in 62 cases,
with 135 contacts, fifteen control gains versus sixteen, and 38 clears versus
34. Its actual scoring/concession outcomes improve. These bounded task proxies
are not universal possession/save/intent adjudication.

Natural includes 35 successful kickoff starts. The 29 ongoing-ground cases alone
improve from 2 goals / 26 concedes / 1 timeout to 4 / 24 / 1, with 19 to 21
touched cases and 34 to 52 contacts. This is positive but still weak ordinary
play. Kickoff repeats five layouts on both sides, not 64 independent situations;
controlled-acquisition events remain zero despite 64 goals. The separate legacy
RLBot wheel-semantic diagnostic remains unresolved for live promotion: these
simulator results are not a live Rocket League test.

## Training exposure and health

The first resumed rollout +452 starts fresh physical episodes; exclude it from
the post-resume trend rather than credit that transient to learning. Compare
exposure-normalized 453–500 (48 updates) to 401–450 (50 updates).

| Metric | 401–450 | 453–500 |
|---|---:|---:|
| Contacts / learner-minute | 13.8087 | 14.5337 |
| Speed, uu/s | 1,179.93 | 1,190.28 |
| Ended player episodes with a touch | 88.28% | 88.74% |
| Categorical entropy | 0.3907 | 0.3419 |
| Natural Nexto goals / million decisions | 1,036.42 | 1,010.09 |
| Natural Nexto concedes / million decisions | 1,916.68 | 1,829.45 |
| Finishing Nexto goals / million decisions | 340.04 | 332.63 |
| Finishing Nexto concedes / million decisions | 2,669.16 | 2,622.21 |
| Challenge Nexto control gains / million decisions | 382.24 | 443.77 |

Raw role totals and exact rates are in `review_reduction_000500.json`. The recent
mean completed-update KL is 0.0021401422109767567; maximum sample KL is
354.3056335449219. These are finite telemetry, not KL rejection conditions or
proof of gameplay health. No finite guard or optimizer corruption failure is
reported. More training touches do not override the lower match follow-up rate.

## Decision and next review

Continue the authorized unchanged learner to the scheduled +550 review. This is
a bounded next review, not a new campaign or claim that the weaknesses are fixed.
There is enough positive evidence to check another interval—six improved match
cases, three near-competitive Blue scores, better ordinary-ground and defense
outcomes—but the small net match gain, Orange deterioration and finishing
nonimprovement prevent a strong progress claim.

At +550 use identical corrected full-match methods against +500, +451 and +0;
review both sides separately. Track actual finishing goals/concedes, challenge
control and ongoing-ground outcomes, not proxy totals alone. If weak transfer
or repeated-contact/Orange deterioration persists, diagnose action/observation
and reward-role learning before another large unchanged extension. Never
silently tune rewards, promote the checkpoint or change the live adapter from
these correlations. Preserve the latest accepted model/Adam/RNG and all snapshots.

The goal remains active and unmet. No new reward terms, opponents, mechanics
detectors, architectures, optimizer settings, live deployment or KL guards were
introduced in this review.
