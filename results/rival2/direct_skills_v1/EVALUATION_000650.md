# +650: more follow-ups, weaker deterministic scoring

## Decision

Do not launch another unchanged50-update block yet. Preserve the latest650
checkpoint and investigate the gap between improving sampled training outcomes
and weaker deterministic scoring. Execute the bounded read-only
[sampling diagnostic](sampling_diagnostic_000650/README.md) after its prospective
package is published. It changes only action selection in private evaluation
copies, not the checkpoint, reward, PPO or deployment. No older checkpoint is
selected. A sampling-mode comparison is not a same-method learning improvement.

The650 learner stopped cleanly after its scheduled evaluations. The STOP is an
agent review marker, not a user cancellation, nonfinite failure or KL rejection.
The SSL goal remains active and unmet.

## Checkpoint and integrity

- `checkpoints/rival2/direct_skills_v1/plus_000650.pt`
- SHA256 `4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F`
- 2,875,392,000 direct-stage learner decisions;7,667,712,000 physical world ticks.
- 144,648 cumulative Adam steps;92,498 Direct Skills steps beyond the entity
  parent;7,090 Adam steps and221,184,000 learner decisions in601-650.
- 14 checkpoint +22 follow-up +11 match integrity checks pass.51 focused CPU
  tests pass. Six reports rebuild exactly. Original20, T2three and follow-up
  three runtime-source hashes remain unchanged. Full data and curves preserved.
- The evaluation took zero optimizer steps and left model/checkpoint unchanged.
  Maximum completed-update mean KL .0034115043; maximum sample estimate87.62045,
  both telemetry only. Peak allocated CUDA16,322,206,208 bytes. No hard failure.

## Same-method full matches

Ten deterministic regulation games against pinned Nexto, unchanged corrected
reset implementation, five layouts on both sides. All results below use the
same tagged evaluation runtime, not the uncorrected historical match results.

| Metric | Corrected451 | 550 | 600 | 650 |
|---|---:|---:|---:|---:|
| Wins | 0 | 0 | 0 | 0 |
| Rival goals | 59 | 62 | 65 | 42 |
| Nexto goals | 156 | 163 | 144 | 139 |
| Goal difference | -97 | -101 | -79 | -97 |
| Rival contacts | 538 | 500 | 508 | 577 |
| Contacts/minute | 10.76 | 10.00 | 10.16 | 11.54 |
| Same-player/resolved follow-ups | 142/482 | 112/431 | 128/437 | 209/533 |

Against600:2 matched cases improve,1 ties,7 worsen. Against550:5 improve,
5 worsen. Against451:4 improve,2 tie,4 worsen. All ten improve over the entering
corrected0 policy, but all ten still lose. The600 full-match gain did not persist.

| Side/layout | 600 Rival/Nexto | 650 Rival/Nexto |
|---|---:|---:|
| Blue0 | 8/12 | 5/14 |
| Blue1 | 7/12 | 4/12 |
| Blue2 | 8/14 | 4/13 |
| Blue3 | 9/11 | 7/13 |
| Blue4 | 9/15 | 5/12 |
| Orange0 | 7/16 | 1/13 |
| Orange1 | 5/13 | 4/16 |
| Orange2 | 5/16 | 3/13 |
| Orange3 | 2/18 | 4/16 |
| Orange4 | 5/17 | 5/17 |

Blue falls41/64 to25/64, Orange24/80 to17/75. Opening goals10 to8, second
goals1 to4, later goals55 to34, later concedes144 to137. These segments do not
have equal exposure, so counts are not equal-opportunity conversion rates.

Same-player follow-ups29.29% to39.21% are not possession duration or intentional
control. Nexto contacts1357 to1443. Forward displacement bins252 to329, backward
60 to50, neutral191 to193; these are until the next contact/goal. Velocity-change
bins52 backward/46 neutral/479 forward are not shot accuracy. One native demo
occurred, not proof of intentional demo play. All games contain Rival touches;
no-touch resets are disabled by protocol, not learned away.

## Fixed skill cases, 64 starts each

| Family | 600 goals/concedes/timeouts | 650 goals/concedes/timeouts |
|---|---:|---:|
| Natural | 38/25/1 | 32/25/7 |
| Challenge | 0/38/26 | 1/43/20 |
| Finishing | 6/42/16 | 5/41/18 |
| Defense | 8/21/35 | 8/17/39 |
| Kickoff | 64/0/0 | 51/0/13 |

Finishing touches all64 balls again; contacts91 to102, on-target proxy47 to53,
actual goals6 to5. This demonstrates why proxy count must not be called scoring.
Challenge controls11 to14, forward advances4 to6, losses0 to5; contacts127 to135,
but concessions38 to43. Defense controls13 to18, clears38 to40, touches158 to171,
with fewer concessions21 to17. These are mixed task-specific outcomes.

Natural's29 ongoing-ground starts remain3 goals, with25 to24 concessions and
1 to2 timeouts;21 touched cases unchanged,47 to48 contacts. The remaining35
kickoff starts move35/0/0 to29/1/5. The separate kickoff family loses13 goals to
timeouts rather than concessions. These repeatfive layouts on each side; raw64
cases are not64 independent kickoff configurations. Controlled acquisition is0.
Shooting remains20% of source starts and is not solved by adding a duplicate.

## Sampled training outcomes

Exclude only the prospectively declared fresh-resume transients555 and601.
Compare556-600 with602-650 using rates normalized by actual role decisions.
The complete role totals and decision mixture are in `review_reduction_000650.json`.

| Metric | 556-600 | 602-650 |
|---|---:|---:|
| Contacts/learner-minute | 16.5973 | 17.3235 |
| Movement speed uu/s | 1181.81 | 1191.16 |
| Ended episodes with contact | 89.71% | 89.91% |
| Categorical entropy | .82995 | .63031 |
| Natural Nexto goals/million role decisions | 692.41 | 958.72 |
| Natural Nexto concedes/million role decisions | 1824.58 | 1706.93 |
| Finishing Nexto goals/million role decisions | 339.18 | 397.84 |
| Finishing Nexto concedes/million role decisions | 2472.27 | 2410.06 |
| Challenge Nexto controls/million role decisions | 639.76 | 697.33 |
| Defense Nexto controls/million role decisions | 899.99 | 946.30 |
| Defense Nexto concedes/million role decisions | 1135.07 | 1091.62 |

Sampled training outcomes improve in these roles while deterministic scoring
worsens. This is not proof that sampling explains the regression: state
visitation, curriculum versus regulation matches and greedy versus stochastic
actions all differ. The next bounded test isolates action selection on the
fixed650 model and unchanged evaluation cases, using three frozen action seeds.
It cannot justify changing inference or reward without seeing the results.

No repeat of the completed250 finishing, wheel reset, team-symmetry or550
reward/GAE/recurrent-gradient audits is required by this evidence. No new
mechanic detector or named reward is introduced.
