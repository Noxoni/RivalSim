# Acquisition-only diagnostic: update 40

This is the first completed probe above the fresh random baseline in both groups.
It is an early gain, not yet persistent acquisition competence or a promotion.

| Deterministic acquisition measure | Baseline 0 | Update 30 | Update 40 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 14/512 (2.734375%) | 88/512 (17.1875%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 2/512 (0.390625%) | 22/512 (4.296875%) |
| Easy conditional median seconds | 0.5083333333 | 3.2166666667 | 0.9833333333 |
| Varied conditional median seconds | 1.0083333333 | 3.2166666667 | 2.1 |

Total deadline-qualified successes rose from 16 at update 30 to 110/1,024, versus
42 at initialization. Easy and varied success rates are respectively 2.84x and
2x baseline, but failure rates remain high. Conditional medians exclude failures
and describe different successful subsets; they are not paired time improvements.
No setting changed between these probes. Native contact does not require beating
Nexto to the ball, gaining possession, or scoring.

The update-40 stochastic training rollout recorded 0.4048665365 touches/minute,
up from 0.2994791667 at update 30, and speed 332.308360 uu/s versus 333.481628.
These changing-state rollout endpoints do not establish sustained gameplay gains.

## Integrity and scope

- 40 accepted updates; 176,947,200 learner decisions; 5,656 fresh Adam steps.
- All 16 CPU checkpoint, contract, curve, finite-state, acquisition-only exposure,
  counter and read-only evaluation checks passed. Zero KL rejections.
- Package/source/initialization/evidence verification against origin/main passed.
- Recomputed counts, fractions, conditional lower medians, and no-contact fractions
  from all 1,024 raw ticks. All were -1 (failure) or 1..960; every check passed.
- Same evaluation scenario/specification hashes as baseline. Result checkpoint
  identity matches both the started receipt and actual snapshot bytes.
- Evaluation changed neither checkpoint, Rival nor Nexto; zero optimizer steps.
- Maximum completed-update mean KL through 40: 0.009485978785839204, telemetry only.
- Worker/launcher active, no STOP/failure or stderr error. Training was not
  interrupted and no extra GPU work or setting changes were introduced.

Acquisition-only training now shows measurable improvement over initialization.
The later probes must determine persistence. This single fresh run does not show
which prior mixed setting caused regression: initialization and reset mix both
differ. Do not automatically add another experimental arm, retune the reward, or
promote the model. Continue the frozen 150-update diagnostic. The old mixed run
remains stopped at 156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000040.pt`
- Checkpoint SHA-256: `F6D45E5B8193729D67A55C35098BAFE172A59070F2542879AC78BB761EA21AD9`
- Result SHA-256: `4B4A7CA6F8C42D85A71FD7372D6CD409E802EA18F67D6B538BE1DC3894729BA2`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Full evidence: `acquisition_000040.json`, its `.started.json` receipt,
`audit_000040.json`, and the closed update-1..40 prefix `through_000040.json`.
