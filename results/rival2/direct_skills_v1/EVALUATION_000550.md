# +550: full-match plateau; pause for a bounded learning-signal diagnosis

## Decision

The latest block does not demonstrate clear overall gameplay improvement. Do
not describe more training contacts or a passed integrity audit as solved
possession/shooting. The agent requested a clean diagnostic pause, completed at
**+554**. This is not a user cancellation, hard safety failure, KL rejection,
new parent, or discarded training. All accepted model/Adam/RNG state is preserved.

Before another large unchanged block, execute the bounded zero-optimizer
diagnostic in `ADVANTAGE_DIAGNOSTIC_PLAN_000550.md`. No reward, opponent,
architecture, PPO setting, policy weight, or physics change has been made.

## Checkpoints and integrity

- Evaluated +550: `checkpoints/rival2/direct_skills_v1/plus_000550.pt`;
  SHA256 `C8564101BEAE09219026A11ED5F5433E6CEC460EFAE808326F7B8D7FB5DB33FE`.
  2,433,024,000 direct learner decisions; 6,488,064,000 physical world ticks;
  130,470 cumulative Adam steps, of which 78,320 are new Direct Skills steps.
- Latest accepted +554: `checkpoints/rival2/direct_skills_v1/paused_000554.pt`;
  SHA256 `CF5C9D022FFB4EF18DBD728206D78A3601AB8C9128BFA0F0E9ADE31BCF961BAD`.
  2,450,718,720 direct learner decisions; 6,535,249,920 physical world ticks;
  131,038 cumulative Adam steps. `diagnostic_pause_000554.json` verifies the
  finite model/Adam, counters, four RNG identities, and contiguous accepted run.
- +550 passes all 14 checkpoint and 11 full-match integrity checks. Thirty
  focused CPU tests pass. The evaluated model/checkpoint did not change during
  evaluation, and evaluation/review took zero optimizer steps.
- Six skill/start-group comparisons rebuilt byte-identically; all 20 frozen
  runtime hashes remain unchanged. Full source hashes and comparison offsets
  are in `review_validation_000550.json`. The complete curve through +550 plus
  `training_curve_000551_to_000554.jsonl` covers every accepted update.

## Same-method full matches

Use only the corrected reset/runtime-tagged results for these comparisons.
The old untagged +450 match results are not a same-method learning baseline.

| Ten fixed regulation matches | Corrected +0 | Corrected +451 | +500 | +550 |
|---|---:|---:|---:|---:|
| Wins | 0 | 0 | 0 | 0 |
| Rival goals | 1 | 59 | 70 | 62 |
| Nexto goals | 272 | 156 | 165 | 163 |
| Goal difference | -271 | -97 | -95 | -101 |
| Rival contacts | 203 | 538 | 431 | 500 |
| Contacts/minute | 4.06 | 10.76 | 8.62 | 10.00 |
| Same-player/resolved follow-ups | 0/200 | 142/482 | 58/357 | 112/431 |

The policy remains substantially better than the entering +0 policy. However,
across the recent ~100 updates, full-match goal differential is effectively
flat and still heavily negative. Relative to +500, four cases improve and six
worsen; relative to +451, four improve, one ties, and five worsen. All ten still
lose. There is no SSL or live-deployment promotion.

| Side/layout | +500 Rival/Nexto | +550 Rival/Nexto |
|---|---:|---:|
| Blue/0 | 8/13 | 10/16 |
| Blue/1 | 9/11 | 9/16 |
| Blue/2 | 9/16 | 6/17 |
| Blue/3 | 9/12 | 4/18 |
| Blue/4 | 10/11 | 9/17 |
| Orange/0 | 5/22 | 7/16 |
| Orange/1 | 8/13 | 4/17 |
| Orange/2 | 4/23 | 3/16 |
| Orange/3 | 1/26 | 4/15 |
| Orange/4 | 7/18 | 6/15 |

Blue now worsens from 45/63 to 38/84, while Orange improves from 25/102 to 24/79.
All five Blue cases worsen in differential; four Orange cases improve. Thus the
previous Orange deterioration is not a monotonic one-sided collapse. The
bounded CPU team check also found no obvious Rival input/assignment defect;
neither fact proves the source of full-match variation.

Opening goals remain 10/10; second goals fall six to three. Later goals fall
60 to 52, with later concessions 165 to 163. These segments have unequal
exposure, so raw counts are not equal-opportunity rates. Same-player follow-ups
recover from 16.25% to 25.99%, still below corrected +451's 29.46%; this is not
possession duration. Contact accounting is 112 same-player + 319 opponent +
65 goal-ended + 4 unclosed = 500. Nexto makes 1,624 contacts. Displacement until
the next contact/goal is 62 backward / 209 neutral / 225 forward; velocity-change
bins are 28 / 37 / 435. Neither is shot accuracy. All matches contain Rival
contacts; no-touch resets are disabled by protocol. No native Rival demos are
recorded. Do not infer intentional mechanics, possession, or fake kickoffs.

## Skill cases: 64 per family

| Family | +500 goals/concedes/timeout | +550 goals/concedes/timeout |
|---|---:|---:|
| Natural | 39/24/1 | 38/25/1 |
| Challenge | 1/38/25 | 0/36/28 |
| Finishing | 4/43/17 | 5/41/18 |
| Defense | 8/20/36 | 9/19/36 |
| Kickoff | 64/0/0 | 64/0/0 |

Finishing still reaches all 64 balls, first touch at 0.53125s on average, with
96 contacts versus 99. Five actual goals are not a meaningful resolution of
shooting; on-target proxy events fall 49 to 44. Shooting remains 20% of source
starts; no duplicate shooting family was added.

Challenge touches 56 cases versus 54, but control gains fall ten to six and
forward-control events seven to five. Lost-control events rise zero to three;
failed-attempt events 21 to 24. Defense has 172 contacts versus 135, control
gains 19 versus 15, clears 36 versus 38, and nine goals/19 concedes versus eight/
20. These are task proxies, not universal save/possession detectors.

Natural's 35 kickoff starts still all score. The **29 ongoing-ground cases**
alone worsen from 4 goals/24 concedes/1 timeout to 3/25/1. Touched cases increase
21 to 22, but total contacts fall 52 to 49. The kickoff family repeats five
layouts on both sides; 64/64 is not 64 independent generalization cases, and
controlled-acquisition events remain zero.

## Training signals

Compare rates over 453-500 with 501-550. Exclude +452's fresh-resume episode
transient. Exact role counts and decision fractions are in the reduction JSON.

| Metric | 453-500 | 501-550 |
|---|---:|---:|
| Contacts/learner-minute | 14.5337 | 15.3372 |
| Movement speed, uu/s | 1190.28 | 1191.61 |
| Ended player episodes with contact | 88.74% | 89.39% |
| Categorical entropy | 0.3419 | 0.2883 |
| Natural Nexto goals/million role decisions | 1010.09 | 764.76 |
| Natural Nexto concedes/million role decisions | 1829.45 | 1893.17 |
| Finishing Nexto goals/million role decisions | 332.63 | 391.31 |
| Finishing Nexto concedes/million role decisions | 2622.21 | 2557.99 |
| Challenge Nexto control gains/million role decisions | 443.77 | 482.12 |
| Defense Nexto concedes/million role decisions | 1333.59 | 1298.61 |

Skill training is not uniformly stalled, but the natural Nexto role deteriorates
while some direct tasks improve. That motivates checking task-level advantage
and gradient signals, rather than attributing everything to ten deterministic
matches or changing the goal reward blindly. Lower entropy is a measurement,
not proof that exploration alone is the cause.

Recent mean completed-update sample KL estimate is 0.002128114652593121; maximum
sample estimate 1207.0887451171875. KL is telemetry only and **did not cause this
pause**. Model/Adam/output finite checks remain intact. Mean value loss moves
0.80726 to 0.79534; that alone does not validate role-specific critic quality.

No named-mechanic reward, new detector, PPO update rule, teacher/router, or
checkpoint selection change was introduced. Goal remains active and unmet.
