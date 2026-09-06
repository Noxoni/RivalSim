# +600: modest full-match gain, finishing still poor

## Decision

Continue the same latest accepted +600 model/Adam/RNG lineage for one additional
50-update block under a separately published continuation authority. Preserve
temperature2 and every reward, scenario, opponent, architecture and PPO setting.
Evaluate at650 with the same fixed cases and review again. Do not select an older
checkpoint, claim SSL, or treat exploration entropy as capability evidence.

The completed temperature-only experiment (+555-600) improves seven of ten
matched games against +550 and reduces concessions. That is sufficient reason
for a measured follow-up, not proof of causality or solved possession. The
shooting drill remains weak. No further optimization is authorized by this
report alone until the executable continuation package is committed, pushed
and verified. The original learner exited cleanly after evaluation; its STOP
is an agent review marker, not a user cancellation or safety failure.

## Checkpoint and integrity

- `checkpoints/rival2/direct_skills_v1/plus_000600.pt`
- SHA256 `8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2`
- 2,654,208,000 direct-stage learner decisions; 7,077,888,000 physical world ticks.
- 137,558 cumulative Adam steps, 85,408 new Direct Skills steps; 6,520 Adam steps
  and203,489,280 learner decisions in the46-update temperature block.
- 14 checkpoint +22 amendment +11 full-match integrity checks pass;45 focused
  CPU tests pass. Six skill/start-group reports rebuild exactly. All20 original
  runtime files and all3 temperature runtime sources retain their frozen hashes.
- Model/checkpoint unchanged during evaluation; no optimizer steps in evaluation
  or reporting. No KL rejection or nonfinite failure. KL remains telemetry only.

## Same-method full matches

Ten deterministic regulation matches against pinned Nexto, five layouts on both
sides, using the corrected native contact-cache reset implementation throughout.

| Metric | Corrected0 | Corrected451 | 500 | 550 | 600 |
|---|---:|---:|---:|---:|---:|
| Wins | 0 | 0 | 0 | 0 | 0 |
| Rival goals | 1 | 59 | 70 | 62 | 65 |
| Nexto goals | 272 | 156 | 165 | 163 | 144 |
| Goal difference | -271 | -97 | -95 | -101 | -79 |
| Rival contacts | 203 | 538 | 431 | 500 | 508 |
| Contacts/minute | 4.06 | 10.76 | 8.62 | 10.00 | 10.16 |
| Same-player/resolved follow-ups | 0/200 | 142/482 | 58/357 | 112/431 | 128/437 |

Against550:7 cases improve,1 ties,2 worsen in goal differential. Against500:
6 improve,4 worsen. Against451:7 improve,3 worsen. All10 improve against0 but
all still lose. These fixed development cases are not ranked ability evidence.

| Side/layout | 550 Rival/Nexto | 600 Rival/Nexto |
|---|---:|---:|
| Blue0 | 10/16 | 8/12 |
| Blue1 | 9/16 | 7/12 |
| Blue2 | 6/17 | 8/14 |
| Blue3 | 4/18 | 9/11 |
| Blue4 | 9/17 | 9/15 |
| Orange0 | 7/16 | 7/16 |
| Orange1 | 4/17 | 5/13 |
| Orange2 | 3/16 | 5/16 |
| Orange3 | 4/15 | 2/18 |
| Orange4 | 6/15 | 5/17 |

Blue improves38/84 to41/64; Orange is24/79 to24/80. Most of this review's score
gain is Blue, following the prior side swing. Do not infer a fixed side bug.
Opening goals remain10/10, second goals3 to1, later goals52 to55, concessions163
to144. These segments have unequal exposure and are not equal-opportunity rates.

Nexto contacts fall1624 to1357. Same-player follow-ups rise25.99% to29.29%, not
possession duration. There are128 same-player +309 opponent +71 unresolved-by-
next-contact outcomes (goals/end of trace) =508 contacts; see raw reduction for
event boundaries. Displacement bins until next contact/goal:60 backward,191
neutral,252 forward. Velocity-change bins29/15/464 are not shot accuracy. All
matches contain Rival contacts. No-touch resets are disabled by protocol, not
a learned zero-reset achievement. Native Rival demo count is0; no named mechanic
or intentional-fake claim is made.

## Fixed drills, 64 starts per family

| Family | 550 goals/concedes/timeouts | 600 goals/concedes/timeouts |
|---|---:|---:|
| Natural | 38/25/1 | 38/25/1 |
| Challenge | 0/36/28 | 0/38/26 |
| Finishing | 5/41/18 | 6/42/16 |
| Defense | 9/19/36 | 8/21/35 |
| Kickoff | 64/0/0 | 64/0/0 |

Finishing touches all64 balls, mean first contact0.53620s versus0.53125s;
contacts96 to91; projected on-target proxy44 to47. Actual goals5 to6 and
concessions41 to42 do not establish a shooting improvement. Shooting remains
20% of source starts, not20% of duration-weighted decisions; no duplicate family.

Challenge contacts111 to127; control gains6 to11, forward-controlled advances
5 to4, lost-control penalties3 to0, failed attempts24 to16. These are useful
proxy changes but no goals and two more concessions. Defense contacts172 to158,
control gains19 to13, clears36 to38, failed attempts11 unchanged. Neither family
shows an unambiguous improvement in native scoring outcomes.

Natural's35 kickoff cases all score, unchanged. The29 ongoing-ground cases
remain3 goals/25 concedes/1 timeout, touched cases22 to21, contacts49 to47.
Kickoff repeats onlyfive layouts on each side, so64/64 is not64 independent
generalization cases; controlled acquisition remains0. No task success alone
resets a training episode or substitutes for scoring.

## Training signals

Compare501-550(T1) with556-600(T2); exclude only555, prospectively declared as
the fresh-episode resume transient. Normalize unequal block sizes. Late576-600
is also retained descriptively, not used to select a checkpoint.

| Metric | 501-550 T1 | 556-600 T2 |
|---|---:|---:|
| Learner decisions | 221,184,000 | 199,065,600 |
| Contacts/learner-minute | 15.3372 | 16.5973 |
| Movement speed uu/s | 1191.61 | 1181.81 |
| Ended episodes with contact | 89.39% | 89.71% |
| Categorical entropy | 0.2883 | 0.8299 |
| Natural Nexto goals/million role decisions | 764.76 | 692.41 |
| Natural Nexto concedes/million role decisions | 1893.17 | 1824.58 |
| Finishing Nexto goals/million role decisions | 391.31 | 339.18 |
| Finishing Nexto concedes/million role decisions | 2557.99 | 2472.27 |
| Challenge Nexto control gains/million role decisions | 482.12 | 639.76 |
| Defense Nexto control gains/million role decisions | 622.32 | 899.99 |
| Defense Nexto concedes/million role decisions | 1298.61 | 1135.07 |

More contact/control opportunities in sampled training have not yet produced
strong deterministic shooting/ongoing-ground outcomes. Mean update KL is
0.0021439205, maximum accepted completed-update mean across555-600 is0.0030581484;
maximum per-sample estimate in556-600 is16.237276. These refer to the altered
training distribution and are neither comparable safety scores nor rejection
thresholds. Mean value loss0.79534 to0.80830 does not establish critic quality.

The bounded follow-up tests whether the modest full-match gain persists without
changing multiple learning ingredients. The SSL goal remains active and unmet.
