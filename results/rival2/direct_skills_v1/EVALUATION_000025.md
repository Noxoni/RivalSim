# Direct Skills +25: more contact acquisition, possession still absent

## Identity and checks

- Checkpoint: checkpoints/rival2/direct_skills_v1/plus_000025.pt
- SHA256: 43C2391DB84CB01D3F90E1F3E62C4928C21C4EEE4B190AA9305A624361DE85AA
- 25 accepted updates; 110,592,000 new learner decisions;
  294,912,000 physical world ticks; 3,558 new Adam steps (55,708 cumulative).
- Completed mean KL range across updates 1–25:
  0.0027899388543698627–0.005030336131190013. Zero KL-based rejections.
- Finite model/Adam, unchanged parent/action/observation/PPO identities, contiguous
  counters and exactly one-third Nexto learner exposure passed the CPU audit.
- Evaluation and comparison performed zero optimizer steps. Identical 64 cases
  and scenario hashes per family; raw outcomes and summaries reconcile.
- Deterministic CPU comparison rebuild passed. No reward/optimizer/curriculum
  setting was changed. Worker 15812 resumed the same campaign automatically.

## Outcomes against Nexto

These are initial episode outcomes, **not full-match wins**. Each family has
64 cases. The columns show baseline, then +10, then +25.

| Metric | Baseline | +10 | +25 |
|---|---:|---:|---:|
| Natural cases with contact | 23 | 25 | 34 |
| Natural goals for / against | 2 / 62 | 2 / 62 | 1 / 63 |
| Challenge cases with contact | 51 | 52 | 53 |
| Challenge controlled acquisitions | 3 | 3 | 0 |
| Challenge controlled advances | 2 | 2 | 0 |
| Challenge goals for / against | 0 / 53 | 0 / 54 | 0 / 57 |
| Finishing goals for / against | 3 / 55 | 5 / 53 | 4 / 50 |
| Finishing goal-bound contact events | 52 | 53 | 52 |
| Defensive shot-clear events | 32 | 30 | 29 |
| Defense goals for / against | 2 / 36 | 1 / 41 | 4 / 40 |
| Kickoff cases with contact | 25 | 25 | 38 |
| Kickoff controlled acquisitions | 0 | 0 | 0 |
| Kickoff goals for / against | 0 / 52 | 0 / 46 | 0 / 26 |
| Kickoff failed timeouts | 12 | 18 | 38 |

The clearest change is reaching the ball in more natural/kickoff cases.
Kickoff has 26 fewer concessions and 26 more timeouts, with zero goals or controlled
acquisitions. That is not a successful possession outcome and does not prove
intentional kickoff fakes. The same-case goal outcome improved in 26 kickoff
cases and worsened in 0, but every improvement was concession-to-timeout, not a
score. Longer real matches are needed to measure what follows those delays.

Finishing has a small gain over baseline, but not over +10. Challenge control
has regressed; defensive clear counts and concessions remain worse than
baseline despite more goals scored from defense starts. Natural scoring did
not improve. There is no broad ground-gameplay or SSL capability claim.

## Focused diagnosis to carry into the +50 review

The reward is not simply failing to fire: the live curve contains genuine
control-gain events. In updates 8–18, kickoff control against Nexto occurred
only 1–4 times per rollout; challenge control occurred 8–20 times. The actor is
therefore exposed to positive control examples, but sparsely. This is training
telemetry, not evidence of deployment competence. The source rows are preserved
in training_curve_through_000025.jsonl.

Source inspection found that possession geometry uses schema-named fields and
the defined physical position/velocity scales. There is no identified mapping
bug from that inspection; it is not an independent certification of possession.

One incentive question for review is the difference between a failed timeout
(-1) and conceding (-10). The increasing concession-to-timeout conversions may
reflect useful initial blocks, a preference for surviving the short drill, or
both. The reward amounts alone do not establish intent or the full discounted
return, which also includes other rewards and truncation bootstrapping. Do not
label this a confirmed exploit or silently change the frozen contract.

## Decision

Preserve +25 and continue to the already-planned +50 skill and ten-full-match
review. No extra evaluator, training restart, reward retuning or checkpoint
cherry-picking was performed. If +50 still fails to improve task outcomes and
natural play, diagnose before another large unchanged block, as the prospective
authority requires. The main unresolved requirement is possession and follow-through,
not merely contact count. Current evidence is not enough to promote this model.

The most recent full-match result remains the offset0 baseline, 0/10 wins and
1 goal for / 266 against. No +25 full-match evaluation was run or implied.

Reproduce the saved-data checks from the repository root:

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_direct_skills.py --update 25
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 25
```
