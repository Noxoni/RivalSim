# Direct Skills +400: full-match improvement, finishing still weak

## Identity and integrity

- Checkpoint: `checkpoints/rival2/direct_skills_v1/plus_000400.pt`.
- SHA256: `0EBBE6B6A8318F4B19AAA79DC56CA3752B248E0D6CF67C14DB5F972701429FB0`.
- 400 accepted updates; 1,769,472,000 new learner decisions;
  4,718,592,000 physical world ticks; 56,934 new Adam steps, 109,084 cumulative.
- All fourteen checkpoint/evaluation checks and eleven full-match checks pass.
  Parent, contracts, finite model/Adam, counters and exactly one-third learner
  exposure to inference-only Nexto are verified. All twenty frozen runtime source
  hashes remain unchanged. No reward, model, scenario, optimizer-setting,
  observation or evaluation-method change occurred.
- Twenty-two focused CPU audit/report/full-match reducer tests pass. The eight
  baseline comparisons (0/50/100/150/200/250/300/350) rebuild deterministically.
- Mean completed-update KL over 351–400 is 0.0023963113885804525; maximum sample
  KL is 262.4019470214844. KL remains telemetry only, with no KL rejection.
  Nonfinite protection remains active. Large finite KL is not relabeled as a
  guard failure; low mean KL is not used to certify gameplay competence either.

## Ten complete matches against Nexto

| Metric | +0 | +250 | +300 | +350 | +400 |
|---|---:|---:|---:|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| Rival goals | 1 | 44 | 29 | 44 | 58 |
| Nexto goals | 266 | 183 | 197 | 183 | 167 |
| Rival contacts | 170 | 402 | 407 | 379 | 458 |
| Contacts/minute | 3.40 | 8.04 | 8.14 | 7.58 | 9.16 |
| Same-player follow-ups / resolved contacts | 0 / 166 | 34 / 356 | 71 / 370 | 40 / 329 | 107 / 393 |
| Kickoff first contacts | 0 | 1 | 5 | 16 | 1 |
| Matches without any Rival contact | 0 | 0 | 0 | 0 | 0 |

The full-match aggregate improves beyond +250/+350: fourteen more goals and
sixteen fewer concessions. Nine of ten matched starting configurations improve
goal differential; one worsens. This is positive development evidence under the
unchanged protocol, not competitive parity: every match is still a decisive loss.
No SSL capability or deployment-ready verdict is claimed.

| Match | Side | Initial layout | +350 Rival / Nexto | +400 Rival / Nexto |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 / 20 | 3 / 16 |
| 1 | 0 | 1 | 2 / 19 | 4 / 16 |
| 2 | 0 | 2 | 6 / 16 | 8 / 17 |
| 3 | 0 | 3 | 5 / 16 | 8 / 18 |
| 4 | 0 | 4 | 5 / 15 | 7 / 16 |
| 5 | 1 | 0 | 5 / 16 | 8 / 15 |
| 6 | 1 | 1 | 10 / 18 | 4 / 18 |
| 7 | 1 | 2 | 2 / 21 | 5 / 16 |
| 8 | 1 | 3 | 3 / 22 | 6 / 18 |
| 9 | 1 | 4 | 2 / 20 | 5 / 17 |

Contacts reconcile as 107 same-player + 286 opponent follow-ups + 60 goal-ended
intervals + five unclosed intervals = 458. The 27.23% same-player follow-up
fraction is not possession duration. Goal-ended intervals do not identify the
scorer. More follow-ups now coincide with better scoring, unlike +300, but this
does not prove every repeated contact is useful. Nexto contacts fall from 1,821
to 1,417. Canonical velocity-change bins are 17 backward / five neutral / 436
forward; net displacement until next contact or goal is 40 backward / 164 neutral
/ 249 forward. Those are different measurements, not shot accuracy.

Sixteen native Rival demolition events occur across eight matches (versus two
events at +350). No new demo reward exists and intentional demo seeking is not
established. Five concessions occur with at most one contact since reset, versus
three at +350. Kickoff first contacts fall from sixteen to one despite better
scoring; this alone is not proof of a bad kickoff or an intentional fake.
No-touch resets are disabled by the full-match protocol; the meaningful result
is that every match contains Rival contacts.

### Opening versus later goals

| Offset | Rival opening goals / ten | Rival later goals | Nexto later goals |
|---|---:|---:|---:|
| +250 | 8 | 36 | 181 |
| +300 | 8 | 21 | 195 |
| +350 | 10 | 34 | 183 |
| +400 | 10 | 48 | 167 |

Opening scores remain ten of ten. Thus all fourteen additional Rival goals arise
after the opening goal; this is not just more initial-kickoff success. However,
Rival scores only two of ten second goals and 48 of 215 later goals. The opening
versus later deficit persists. Unequal segment exposure and different reset,
recurrent and opponent histories prohibit attributing that gap from aggregate
counts alone. The wheel-state source finding remains unresolved, not disproved
by improvement or promoted into a proven causal failure.

## Fixed 64-case skill evaluations

| Metric | +150 | +250 | +350 | +400 |
|---|---:|---:|---:|---:|
| Natural goals / concessions | 15 / 49 | 32 / 32 | 38 / 26 | 38 / 25 |
| Challenge goals / concessions | 2 / 52 | 1 / 52 | 1 / 50 | 1 / 44 |
| Challenge control gains | 3 | 7 | 3 | 6 |
| Challenge forward-control events | 3 | 3 | 2 | 5 |
| Finishing goals / concessions | 9 / 42 | 6 / 51 | 7 / 48 | 7 / 38 |
| Finishing on-target proxy events | 48 | 44 | 57 | 53 |
| Defense goals / concessions | 5 / 30 | 7 / 29 | 5 / 24 | 4 / 25 |
| Defensive clears | 37 | 38 | 37 | 38 |
| Defense control gains | 3 | 6 | 6 | 16 |
| Kickoff goals / concessions | 26 / 0 | 52 / 0 | 64 / 0 | 64 / 0 |
| Kickoff control gains | 0 | 0 | 0 | 0 |

Finishing remains twenty percent of episode-source starts. All 64 cases reach
the ball, with conditional first touch at 0.5343750715255737 seconds, but only
seven score. Ten fewer concessions become ten more timeouts: 19 versus nine.
This is not better goal conversion. The existing 53 projected on-target events
are a bounded geometric reward proxy, not 53 scored or goalkeeper-aware shots.
Do not claim shooting is learned because acquisition or the proxy is frequent.

Challenge has 53 touched cases, 102 contacts, six control gains, five forward
control events and nineteen timeouts. Defense has 62 touched cases, 134 contacts,
sixteen control gains, 38 clears and 35 timeouts. Physical task-proxy control
improved, especially in defense, but scoring did not; possession mastery is not
established. No new possession, save, fake or named-mechanic detector was added.

The short kickoff test remains 64 goals, zero concessions and zero timeouts, with
zero controlled acquisitions. It repeats five layouts on two sides; it is not
64 independent generalization cases. Natural tests again contain 35 successful
kickoff starts and 29 ongoing-ground starts. The latter change from +350 to +400:
24 to 20 touched cases, 41 to 36 contacts, three to three goals, 26 to 25
concessions and zero to one timeout. Natural aggregate scoring is unchanged and
ongoing-ground acquisition regresses in this small fixed corpus even though
full-match scoring improves. Preserve both results rather than hiding the weaker
one behind the full-match improvement.

## Training telemetry

| Metric | 301–350 | 351–400 |
|---|---:|---:|
| Contacts / learner-minute | 12.5233 | 12.9479 |
| Movement speed, uu/s | 1,172.59 | 1,169.15 |
| Categorical entropy | 0.5327 | 0.4486 |
| Ended player episodes with a contact | 86.84% | 87.48% |
| Natural Nexto goals / million decisions | 815.35 | 920.82 |
| Natural Nexto concedes / million decisions | 2,047.13 | 1,955.66 |
| Finishing Nexto goals / million decisions | 341.95 | 304.07 |
| Finishing Nexto concedes / million decisions | 2,818.73 | 2,733.63 |

No pause or fresh-episode restart occurred in these blocks. Raw counts and exact
role exposure are retained in `review_reduction_000400.json`. The natural role
improves stochastically as well as in full matches. Finishing scoring decreases
in training and remains flat in deterministic tests; its remaining weakness is
real. Entropy keeps declining; this is monitored without silently retuning the
frozen entropy coefficient or interpreting finite outputs as sufficient health.

## Decision

The prospective +400 condition for a diagnostic pause on stalled/regressing
match performance does not trigger: both scoring and concessions improve beyond
the prior best, with better goal differential in nine matches and additional
post-opening goals. Continue the existing authorized campaign unchanged to the
scheduled +450 review. Preserve +400 separately and retain accepted rolling
checkpoints; do not deploy it as a Nexto-beating or SSL bot.

At +450 review the same full matches and skill cases, with continued attention
to post-opening scoring, ongoing-ground control and real finishing. If progress
stalls or reverses, pause at an accepted boundary for the already committed
bounded kickoff input/reset trace before another large unchanged extension.
The wheel-contact reset asymmetry and Nexto cadence remain diagnostic leads, not
automatic simulator fixes. No trace has run and no existing evaluation is replaced.
Any diagnosis must preserve the most recent accepted model/Adam/RNG rather than
resuming an older diagnostic checkpoint. No reward retuning or new detector is
authorized by these correlations alone.

This review was CPU-only, made no optimizer step and did not interrupt the
learner. The goal remains active: the evidence supports some improvement, not
completion, Nexto parity, mastered possession or SSL-level performance.
