# Direct Skills +350: match recovery, better fixed kickoffs, weak finishing

## Identity and integrity

- Checkpoint: `checkpoints/rival2/direct_skills_v1/plus_000350.pt`.
- SHA256: `5FFDBF547C887BD70E541F78382E3083799150825D0E4BEA2465E600823F8924`.
- 350 accepted updates; 1,548,288,000 new learner decisions;
  4,128,768,000 physical world ticks; 49,802 new Adam steps, 101,952 cumulative.
- Fourteen checkpoint/evaluation checks and eleven full-match checks pass.
  The parent, authority, contracts, finite model/Adam and counters are intact.
  Exactly one third of learner decisions are against inference-only Nexto.
- All twenty frozen runtime source hashes still match. No reward, scenario,
  model, optimizer-setting, observation or evaluation-method change occurred.
- Twenty-two focused CPU audit/report/reducer tests pass. Comparisons for all
  preceding scheduled checkpoints through +300 rebuild deterministically.
- Maximum sample KL in updates 301–350 is 83.70667266845703; mean completed-update
  KL across that block is 0.002388877067035813. KL is telemetry only, not a guard.
  Finite checks remain active. These quantities do not establish gameplay health.

## Ten complete matches against Nexto

| Metric | +0 | +250 | +300 | +350 |
|---|---:|---:|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| Rival goals | 1 | 44 | 29 | 44 |
| Nexto goals | 266 | 183 | 197 | 183 |
| Rival contacts | 170 | 402 | 407 | 379 |
| Contacts/minute | 3.40 | 8.04 | 8.14 | 7.58 |
| Same-player follow-ups / resolved next contacts | 0 / 166 | 34 / 356 | 71 / 370 | 40 / 329 |
| Kickoff first contacts | 0 | 1 | 5 | 16 |
| Matches without any Rival contact | 0 | 0 | 0 | 0 |
| Concessions with at most one contact since reset | 52 | 0 | 2 | 3 |

The score aggregate exactly recovers +250; it does not beat it. Against +250,
four matches improve goal differential, five worsen and one is unchanged. All
ten remain losses. Neither Nexto parity nor SSL-level capability is demonstrated.
Higher kickoff first-contact counts do not by themselves establish possession.

| Match | Side | Initial layout | +250 Rival / Nexto | +350 Rival / Nexto |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 / 19 | 4 / 20 |
| 1 | 0 | 1 | 7 / 18 | 2 / 19 |
| 2 | 0 | 2 | 6 / 16 | 6 / 16 |
| 3 | 0 | 3 | 3 / 19 | 5 / 16 |
| 4 | 0 | 4 | 4 / 18 | 5 / 15 |
| 5 | 1 | 0 | 3 / 20 | 5 / 16 |
| 6 | 1 | 1 | 6 / 15 | 10 / 18 |
| 7 | 1 | 2 | 4 / 18 | 2 / 21 |
| 8 | 1 | 3 | 4 / 20 | 3 / 22 |
| 9 | 1 | 4 | 3 / 20 | 2 / 20 |

Contact conservation is 40 same-player + 289 opponent follow-ups + 47 goal-ended
intervals + three unclosed intervals = 379 Rival contacts. The 12.16% same-player
follow-up fraction is not possession duration; goal-ended intervals do not
identify the scorer. Nexto made 1,821 contacts. Rival's canonical velocity-change
bins are 19 backward / three neutral / 357 forward; net displacement until the
next contact or goal is 27 backward / 175 neutral / 174 forward. These are not
interchangeable shot-quality metrics. Two native Rival demolition events do not
establish deliberate demo seeking. No-touch resets are disabled in this protocol;
zero such resets is not a learned achievement.

### Opening versus later goals

| Offset | Rival opening goals / ten | Rival later goals | Nexto later goals |
|---|---:|---:|---:|
| +250 | 8 | 36 | 181 |
| +300 | 8 | 21 | 195 |
| +350 | 10 | 34 | 183 |

All ten opening sequences now score, including initial layout 4 on both sides.
But Rival scores only two of the ten second goals and 34 of the 217 later goals.
This remains an important transfer gap. Opening and later segments have unequal
exposure and different recurrent/opponent/reset histories; this table alone
does not identify a reset bug or a cause. The already published bounded kickoff
trace remains available for a focused diagnosis. It has not been executed.

## Fixed 64-case skill evaluations

| Metric | +150 | +250 | +300 | +350 |
|---|---:|---:|---:|---:|
| Natural goals / concessions | 15 / 49 | 32 / 32 | 32 / 31 | 38 / 26 |
| Challenge goals / concessions | 2 / 52 | 1 / 52 | 2 / 49 | 1 / 50 |
| Challenge control gains | 3 | 7 | 5 | 3 |
| Challenge forward-control events | 3 | 3 | 3 | 2 |
| Finishing goals / concessions | 9 / 42 | 6 / 51 | 5 / 45 | 7 / 48 |
| Finishing on-target proxy events | 48 | 44 | 54 | 57 |
| Defense goals / concessions | 5 / 30 | 7 / 29 | 6 / 33 | 5 / 24 |
| Defensive clears | 37 | 38 | 36 | 37 |
| Kickoff goals / concessions | 26 / 0 | 52 / 0 | 52 / 0 | 64 / 0 |
| Kickoff control gains | 0 | 0 | 0 | 0 |
| Kickoff timeouts | 38 | 12 | 12 | 0 |

Shooting is already twenty percent of episode-source starts, with genuine
goalkeeper opposition and a real goal required for the +10 scoring reward. No
additional duplicate shooting family was introduced. All 64 finishing cases
touch the ball, with conditional first touch at 0.5358072519302368 seconds, but
only seven score (10.9375%). Nine time out and 48 concede. The scoring cases are
7, 19, 22, 37, 41, 43 and 51. The 57 on-target events are the existing bounded
geometric task proxy, not 57 successful or goalkeeper-aware shots. This is a
small recovery, still below +150's nine goals, not mastered finishing.

Kickoff results improve because the formerly unsuccessful center-back layout
now scores. The 64 cases repeat five layouts and two sides; they are not 64
independent demonstrations of generalized kickoff competence. Zero controlled
acquisitions means these results do not prove possession or intentional fakes.

The natural test has 35 kickoff starts, all now scoring, plus 29 ongoing-ground
starts. Ongoing-ground results from +300 to +350: 19 to 24 touched cases, 35 to
41 contacts, three to three goals, 25 to 26 concessions, one to zero timeouts.
Thus the natural aggregate scoring improvement comes from kickoff, not additional
ongoing-ground goals. Challenging and controlled possession remain weak. Defense
concedes less but has 35 timeouts versus 25 at +300; surviving a bounded attempt
does not prove a save or successful transition to offense.

## Training telemetry

| Metric | 201–250 | 252–300 | 301–350 |
|---|---:|---:|---:|
| Contacts / learner-minute | 10.7870 | 11.5596 | 12.5233 |
| Movement speed, uu/s | 1,176.39 | 1,170.64 | 1,172.59 |
| Categorical entropy | 0.7756 | 0.6400 | 0.5327 |
| Ended player episodes with a contact | 85.01% | 85.33% | 86.84% |
| Natural Nexto goals / million decisions | 628.40 | 795.94 | 815.35 |
| Natural Nexto concedes / million decisions | 2,224.48 | 2,070.43 | 2,047.13 |
| Finishing Nexto goals / million decisions | 319.58 | 325.98 | 341.95 |
| Finishing Nexto concedes / million decisions | 2,943.73 | 2,906.61 | 2,818.73 |

The prior block excludes +251's fresh-episode resume transient, as declared in
the +300 review. No pause/restart occurred in 301–350. Raw role counts and exposure
are preserved in `review_reduction_000350.json`; stochastic training gains do not
override deterministic test weaknesses. Declining entropy is tracked, not treated
as proof of either learning or collapse. No exploration setting changed.

## Decision and next observation

The +300 condition for pausing if +350 failed to recover match performance did
not trigger: scoring/concessions recovered to +250, fixed kickoffs improved and
stochastic training rates advanced. Leave the healthy campaign unchanged through
the next scheduled +400 evaluation, retaining every accepted rolling checkpoint.
Do not promote +350 as broadly better than +250 or deploy it as an SSL bot.

At +400 check natural full-match score, later-goal performance and ongoing-ground
control/finishing, not just initial kickoff successes or contact counts. If there
is no further match improvement or a new regression, pause at the next accepted
boundary for the already prepared input/reset transfer diagnosis before another
large unchanged extension. Preserve the latest accepted model/Adam/RNG; do not
resume an older diagnostic checkpoint. The published +300 trace is read-only and
must match its saved full evaluation exactly before interpretation. No reward or
reset change is justified by the opening/later correlation alone.

No PPO source/configuration change, new GPU evaluator or optimizer step was
performed by this review. The learner continued under the existing exclusive GPU
lease, with no operational error or nonfinite failure observed. The broad SSL
goal remains active and unmet. All detailed evidence and the +350 checkpoint are
preserved separately from +250 and +300.
