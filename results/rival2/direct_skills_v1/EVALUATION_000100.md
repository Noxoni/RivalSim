# Direct Skills +100: acquisition improves; possession and offense remain weak

## Integrity and scope

- Snapshot: `checkpoints/rival2/direct_skills_v1/plus_000100.pt`.
- SHA256: `BDE1ED10CB5598D9DCF20BDFD77266159165B9FED65CD0BB08183D4B58CAC965`.
- 100 accepted updates, 442,368,000 new learner decisions,
  1,179,648,000 physical world ticks, 14,228 new Adam steps (66,378 cumulative).
- All fourteen existing CPU checkpoint/evaluation integrity checks passed:
  exact parent identity, unchanged contracts, finite model/Adam, contiguous
  updates, exact samples/Adam counters, one-third Nexto learner exposure,
  matching evaluated snapshot and zero optimizer steps during evaluation.
- Completed mean KL range across updates 1–100:
  0.0025364245614399854–0.005030336131190013. Maximum completed sample KL:
  17.064023971557617. KL remains telemetry only; zero KL rejections. This tail
  is not a newly imposed safety boundary or a reason to roll back the run.
- No training, reward, curriculum, architecture, evaluation seeds or optimizer
  settings changed during this block. The scheduled evaluation completed and
  the same worker resumed training without an error.

The checkpoint is an independently preserved review snapshot, not a selected
best policy, deployment or SSL promotion. The +50 snapshot remains available.

## Ten full regulation matches against Nexto

All eleven existing native match integrity checks passed. Goals, scorer entries,
scoreboards and recurrent resets agree; all ten matches completed regulation;
there was no goal overflow, model mutation or optimizer step during evaluation.
The fixed methodology and both sides/five initial layouts are unchanged.

| Metric | Baseline | +50 | +100 |
|---|---:|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 | 0 / 10 |
| Goals for | 1 | 17 | 12 |
| Goals against | 266 | 212 | 209 |
| Rival contacts | 170 | 313 | 366 |
| Rival contacts/min | 3.40 | 6.26 | 7.32 |
| Nexto contacts | 2,636 | 1,765 | 2,019 |
| Same-player follow-ups / resolved contacts | 0 / 166 | 40 / 304 | 37 / 350 |
| Same-player follow-up fraction | 0% | 13.16% | 10.57% |
| Kickoff first contacts | 0 | 0 | 2 |
| Matches without Rival contact | 0 | 0 | 0 |
| Concessions with at most one contact since reset | 52 | 0 | 9 |

The increase in contact rate is real, but it is not proof of possession. At
+100, Nexto took the next distinct contact in **313 of 350** resolved cases.
Rival's full-match scoring and same-player follow-up fraction both regressed
from +50. Concessions only improved slightly. The result remains better than the
root in several measurements, but cannot be called uniformly better than +50.

At contact, canonical ball-velocity change was forward 360 times and backward
6 times. Net canonical ball displacement until the next contact or goal was
forward 135 times, neutral 211 and backward 18. These are respectively changes
in velocity and net displacement, not an independently adjudicated shot or
possession label. Native Rival demolitions were zero.

The protocol does not use no-touch resets. Its zero no-touch-truncation count
is therefore not a behavioral success metric. The early-concession measure is
based on distinct contact count since reset, not elapsed seconds or intent.

| Match | Rival side | Initial layout | Rival goals | Nexto goals | Rival contacts | Same-player follow-ups |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2 | 21 | 34 | 2 |
| 1 | 0 | 1 | 1 | 20 | 39 | 5 |
| 2 | 0 | 2 | 0 | 22 | 38 | 4 |
| 3 | 0 | 3 | 2 | 19 | 39 | 4 |
| 4 | 0 | 4 | 1 | 22 | 41 | 5 |
| 5 | 1 | 0 | 2 | 22 | 32 | 1 |
| 6 | 1 | 1 | 0 | 21 | 37 | 7 |
| 7 | 1 | 2 | 0 | 21 | 34 | 2 |
| 8 | 1 | 3 | 4 | 22 | 44 | 7 |
| 9 | 1 | 4 | 0 | 19 | 28 | 0 |

## Fixed skill episodes: 64 per family

| Metric | Baseline | +50 | +100 |
|---|---:|---:|---:|
| Natural-start goals for / against | 2 / 62 | 17 / 47 | 15 / 49 |
| Natural-start cases with contact | 23 | 31 | 53 |
| Challenge goals for / against | 0 / 53 | 0 / 48 | 0 / 54 |
| Challenge controlled acquisitions | 3 | 2 | 1 |
| Challenge controlled advances | 2 | 2 | 1 |
| Challenge contacts | 69 | 82 | 74 |
| Finishing goals for / against | 3 / 55 | 5 / 56 | 8 / 47 |
| Finishing goal-bound contacts | 52 | 52 | 51 |
| Defense goals for / against | 2 / 36 | 5 / 36 | 4 / 33 |
| Defensive shot-clear events | 32 | 31 | 32 |
| Kickoff goals for / against | 0 / 52 | 26 / 38 | 26 / 0 |
| Kickoff cases with contact | 25 | 38 | 64 |
| Kickoff controlled acquisitions | 0 | 0 | 0 |
| Kickoff timeouts | 12 | 0 | 38 |

Finishing improved modestly against the same initial goalie states, while the
goal-bound-contact proxy barely changed. This illustrates why a proxy success
count cannot substitute for actual scoring. Defense improved slightly on
concessions. Close-challenge possession did not improve.

### Kickoff versus ongoing-ground states

The source-hash-verified grouping uses initial metadata, not successful outcomes.
The 64 kickoff cases repeat five layouts on two sides; they are not 64
independent generalization cases.

- At +50, layouts 0 and 2 supplied the 26 goals. At +100, layouts **0 and 3**
  supply them. Layout 2 regressed from scoring to timeout; layout 3 improved
  from conceding without contact to scoring.
- Layouts 1, 2 and 4 now contact the ball but all time out at twelve seconds.
  **Zero kickoff concessions is not a possession win:** there are still zero
  controlled acquisitions and all 38 timeouts are failed skill attempts.
- In the longer natural suite, kickoff starts on those same three layouts all
  concede. The twelve-second drill cutoff can hide a later loss, not prevent it.
- The natural suite contains 29 ongoing-ground starts. From +50 to +100,
  touched cases increase **10 to 18**, contacts **13 to 24**, but goals for /
  against worsen **2/27 to 1/28**. Better acquisition has not become better
  sustained offense in these cases.
- There is no demonstrated intentional-fake skill in this evidence. Contacting
  the ball after Nexto, timeout survival and zero possession events cannot
  establish such an intention or capability.

## Focused diagnosis from the recorded training curve

The second fifty updates are not a frozen or idle learner. Native resets/goals
continue, Nexto learner exposure remains exactly one third, Adam advances, and
finite checks pass. Stochastic training statistics below are not substituted for
the deterministic match result.

| Mean training statistic | Updates 1–50 | Updates 51–100 |
|---|---:|---:|
| Contacts per learner-minute | 7.8016 | 8.4666 |
| Movement speed, uu/s | 1,052.57 | 1,084.27 |
| Categorical entropy | 1.8650 | 1.5939 |
| Ended player episodes with a contact | 76.08% | 80.73% |

Control rewards are reachable, but successful events against Nexto remain
sparse: challenge control gains are 700 then 875 across 11,779,194 then
11,607,860 challenge learner decisions. Forward-control events are 242 then
386. Kickoff control gains are 103 then 202 across roughly 12.5 million learner
decisions in each block. This is not an absent-event bug, nor evidence that
possession is learned. The fixed challenge evaluation actually regressed.

Finishing training records 42,231 then 38,220 goal-bound-contact events, versus
3,023 then 3,345 goals. A projected on-target hit frequently does not beat the
goalkeeper. Natural Nexto training goals grow from 2,826 to 4,559, but these
stochastic, changing-state training counts do not override the full-match
scoring regression. All underlying counts and exposure are in the immutable
`training_curve_through_000100.jsonl` prefix.

## Review decision and next step

Continue unchanged to the scheduled +150 review, not an indefinite assumption
that more touches mean success. There is measurable acquisition progress,
modest finishing improvement and no broad collapse or implementation fault.
That supports one more scheduled block without confounding this run by changing
rewards at every noisy checkpoint. It does **not** establish that current
training will reach SSL, or that this block improved full-match offense.

Keep +50 and +100 independently recoverable. At +150 explicitly compare both
previous reviews: goals, repeated contacts, challenge control, the ongoing-ground
subset and each kickoff layout. If acquisition continues without conversion to
control/scoring, the next investigation should focus on the scarcity of
successful possession trajectories and the drill-to-open-play gap, before
spending another large unchanged compute block. No automatic reward retune,
restart, extra opponent, deployment or new capability detector was performed.

## Reproduction

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_direct_skills.py --update 100
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 100
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 100 --baseline 50
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 100 --start-groups
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 100 --baseline 50 --start-groups
```

The existing full-match `reduce(path)` function in
`benchmarks/report_rival2_ssl_entity_match_followup.py` produces the standalone
`full_match_000100_integrity.json`. Only saved outcomes are reduced; no new
policy evaluation is run. The prior-review helper's nineteen focused tests and
deterministic rebuild evidence were published before this evaluation at
`588286021944fa9ccab2a0e880595a044d944b45`.
