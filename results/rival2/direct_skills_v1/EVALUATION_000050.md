# Direct Skills +50: localized kickoff scoring gain, possession unresolved

## Checkpoint and training integrity

- Checkpoint: checkpoints/rival2/direct_skills_v1/plus_000050.pt
- SHA256: 4B18F8F43F612E725E4BA75537C9D3246FD5544C9F5966BD8FE4393AD9CC17EA
- 50 accepted updates; 221,184,000 new learner decisions;
  589,824,000 physical world ticks; 7,118 new Adam steps (59,268 cumulative).
- Completed mean KL range across updates 1–50:
  0.0027899388543698627–0.005030336131190013. Zero KL-based rejections.
- Finite model/Adam, unchanged parent/action/observation/PPO identities,
  contiguous counters and exactly one-third Nexto learner exposure passed.
- Scheduled evaluation used zero optimizer steps. No reward, curriculum,
  optimizer, model architecture or evaluation seed was changed during this block.
- The CPU comparison and initial-state grouping rebuild deterministically.
  Thirteen focused report tests passed; scoped Ruff checks passed.

## Fixed-case results against Nexto

Each family has 64 initial episodes. These are not full-match wins and are not
necessarily 64 independent situations: standard kickoffs repeat five layouts on
two sides. Compare identical source states and preserve the raw case outcomes.

| Metric | Baseline | +25 | +50 |
|---|---:|---:|---:|
| Natural-start goals for / against | 2 / 62 | 1 / 63 | 17 / 47 |
| Natural-start cases with contact | 23 | 34 | 31 |
| Challenge goals for / against | 0 / 53 | 0 / 57 | 0 / 48 |
| Challenge controlled acquisitions | 3 | 0 | 2 |
| Challenge controlled advances | 2 | 0 | 2 |
| Finishing goals for / against | 3 / 55 | 4 / 50 | 5 / 56 |
| Finishing goal-bound contacts | 52 | 52 | 52 |
| Defense goals for / against | 2 / 36 | 4 / 40 | 5 / 36 |
| Defensive shot-clear events | 32 | 29 | 31 |
| Kickoff goals for / against | 0 / 52 | 0 / 26 | 26 / 38 |
| Kickoff cases with contact | 25 | 38 | 38 |
| Kickoff controlled acquisitions | 0 | 0 | 0 |
| Kickoff failed timeouts | 12 | 38 | 0 |

The +50 kickoff gain is actual scoring, unlike +25's concession-to-timeout
conversions. However it is localized to two standard layouts, not learned
possession across arbitrary kickoff states. There is no evidence here of
intentional fakes, general dribbling, or reliable challenge wins.

## Separating kickoff improvement from ongoing gameplay

`start_groups_000050.json` groups the already completed outcomes using the exact
regenerated, hash-verified initial-state metadata. It does not run another policy
evaluation or group cases based on which outcomes look good.

- The natural suite contains 35 standard-kickoff starts and 29 ongoing-ground
  starts. Its entire aggregate scoring gain is from kickoff starts.
- The 29 ongoing-ground cases remain **2 goals / 27 concessions**; contacts
  changed **14 to 13**. This is not an improvement in ongoing possession/play.
- In the kickoff suite, layouts **0 and 2** account for all 26 new goals, on both
  sides. There are 13 repeated cases of each layout.
- Layouts **1 and 3** still concede every case without Rival contact.
- Layout **4** changes from 12 timeouts to 12 concessions, despite three Rival
  contacts per case instead of one. More contacts do not establish control.

The two-layout scoring change must not be extrapolated to all kickoffs or to
the ongoing-play portion of the natural suite. Full regulation matches below
are a separate measurement of what follows repeated goals and resets.

## Full regulation matches

The scheduled ten five-minute matches completed with native goals and kickoff
resets. All eleven existing full-match integrity checks passed, including
scorer/scoreboard agreement, valid physical goal entries, recurrent reset per
goal, unchanged checkpoint/model, no goal-buffer overflow and zero learning.

| Full-match metric | Baseline | +50 |
|---|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 |
| Goals for | 1 | 17 |
| Goals against | 266 | 212 |
| Rival contacts | 170 | 313 |
| Rival contacts/min | 3.40 | 6.26 |
| Nexto contacts | 2,636 | 1,765 |
| Same-player next contacts / resolved next contacts | 0 / 166 | 40 / 304 |
| Kickoff first contacts | 0 | 0 |
| Matches without any Rival contact | 0 | 0 |
| Concessions with at most one distinct contact since reset | 52 | 0 |

Same-player follow-ups measure the identity of the next distinct contact, not
possession time or successful carrying. The +50 fraction is 13.16%; Nexto still
takes the next contact in 264 of 304 resolved cases. The early-concession count
uses contact count, not a fixed time window. Match protocol has no artificial
no-touch resets; a zero reset count is not itself proof of good acquisition.

At +50, canonical ball-velocity change was forward on all 313 contacts, but net
ball displacement before the next contact/goal was forward on only 100, neutral
on 177 and backward on 35. A goalward impulse is not the same as maintaining
possession or producing a sustained attack.

| Match | Rival side | Starting layout | Rival goals | Nexto goals | Rival contacts | Same-player follow-ups |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2 | 19 | 31 | 3 |
| 1 | 0 | 1 | 0 | 22 | 31 | 5 |
| 2 | 0 | 2 | 2 | 20 | 31 | 4 |
| 3 | 0 | 3 | 0 | 22 | 31 | 5 |
| 4 | 0 | 4 | 1 | 21 | 31 | 5 |
| 5 | 1 | 0 | 3 | 23 | 32 | 4 |
| 6 | 1 | 1 | 2 | 24 | 32 | 3 |
| 7 | 1 | 2 | 4 | 21 | 32 | 3 |
| 8 | 1 | 3 | 1 | 19 | 29 | 3 |
| 9 | 1 | 4 | 2 | 21 | 33 | 5 |

## Review decision

Continue the unchanged run to its next scheduled +100 review. This is a review
point, not a new termination cap; the user authorized training until stopped.
The first fifty updates produced measurable native full-match scoring/contact
improvement and a specific kickoff scoring improvement. The evidence does not
justify stopping a healthy learner as if it had made no progress, nor does it
justify an SSL or broad-possession claim.

The unresolved priorities are control after contact, the three weak kickoff
layouts, finishing against a goalie and open-play progress. Preserve the same
case-group breakdown at future scheduled reviews so further kickoff gains cannot
mask stagnation in ongoing-ground cases. The +25-to-+50 jump also shows why
changing the reward after each noisy short evaluation would have confounded
this experiment. No reward change, restart, extra learning, new opponent,
scripted policy prefix or checkpoint cherry-picking occurred at this review.

Worker 15812 resumed the existing learner after evaluation. The selected +50
snapshot and all raw evidence remain independently preserved while newer rolling
checkpoints advance. No deployment or competitive-rank promotion was performed.

## Reproduction

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_direct_skills.py --update 50
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 50
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 50 --start-groups
.venv/Scripts/python.exe -m pytest tests/test_direct_skills_report.py -q
```

This is development evidence, not an SSL/promoted-policy verdict.
