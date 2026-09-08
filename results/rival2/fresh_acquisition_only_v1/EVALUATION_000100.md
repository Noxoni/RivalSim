# Acquisition-only diagnostic: update 100

Varied acquisition improved to a new observed high; easy acquisition slipped.
Both remain above initialization, with mixed rather than uniformly improving results.

| Deterministic acquisition measure | Baseline 0 | Update 90 | Update 100 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 201/512 (39.2578125%) | 190/512 (37.109375%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 91/512 (17.7734375%) | 117/512 (22.8515625%) |
| Easy conditional median seconds | 0.5083333333 | 0.875 | 0.9083333333 |
| Varied conditional median seconds | 1.0083333333 | 2.275 | 2.6166666667 |

Total deadline-qualified successes rose 292 to 307/1,024, still below the
update-70 high of 333. Easy lost 11 cases; varied gained 26. The greater varied
conditional median does not negate additional successful cases: these times
exclude failures and involve changing successful subsets. Contact does not prove
first touch before Nexto, useful possession, scoring or sustained-gameplay skill.

Stochastic rollout contact rate rose 0.7926432292 to 0.9623209635/minute and
movement speed 275.314376 to 322.742720 uu/s from update 90 to 100. These endpoints
come from changing training states and are not paired deterministic evaluations.

## Verification

- 100 accepted updates, 442,368,000 learner decisions, 13,936 fresh Adam steps.
- All 16 CPU checkpoint/curve/contract/finite/acquisition-only/read-only checks
  passed. No KL rejection or exposure from another scenario family.
- Frozen package, source, initialization and evidence verified against origin/main.
- Recomputed counts, fractions, conditional lower medians and no-contact fractions
  from all 1,024 raw ticks; each value was -1 or 1..960. All checks passed.
- Scenario/specification hashes equal baseline. Actual checkpoint SHA agrees with
  result and started receipt. Evaluation took zero optimizer steps and left
  Rival, Nexto and checkpoint unchanged.
- Maximum completed-update mean KL through 100: 0.009485978785839204, telemetry only.
- Worker/launcher healthy; empty stderr; no STOP/failure. No interruption, resume,
  extra GPU evaluation, retuning, promotion or change to frozen training semantics.

Continue the unchanged diagnostic to 150 unless stopped. Gains and regressions
coexist in this isolated run. Do not assign the previous mixed run's cause to
one setting: initialization and reset mixture both differ, and no matched fresh
mixed control is part of this experiment. The old mixed run stays stopped at 156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000100.pt`
- Checkpoint SHA-256: `62F38E9C1F2049ABEF05DE8D2206DC28A4B0C2C4445573D6D78AB65DE9B89F9C`
- Result SHA-256: `B2C1090F7CF2DE2B00CE96AC35C83F8EAB83D8F7846115C7B9B77EB6A2D4F5FA`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Evidence: `acquisition_000100.json`, its `.started.json` receipt,
`audit_000100.json`, and closed update-1..100 prefix `through_000100.json`.
