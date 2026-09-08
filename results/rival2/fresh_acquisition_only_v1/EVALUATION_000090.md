# Acquisition-only diagnostic: update 90

Both groups recovered from update 80. Varied acquisition reached a new observed
high, while easy acquisition remains below its update-70 peak.

| Deterministic acquisition measure | Baseline 0 | Update 80 | Update 90 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 144/512 (28.125%) | 201/512 (39.2578125%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 47/512 (9.1796875%) | 91/512 (17.7734375%) |
| Easy conditional median seconds | 0.5083333333 | 0.8833333333 | 0.875 |
| Varied conditional median seconds | 1.0083333333 | 2.5083333333 | 2.275 |

Total deadline-qualified successes are 292/1,024, up from 191 but below the
update-70 total of 333. Easy's observed peak is 251/512 at 70; the prior varied
peak was 82/512 at 70. Conditional medians exclude failures and compare different
successful subsets. Own contact does not establish beating Nexto to first touch,
sustained possession, scoring, or full-match performance.

The stochastic update-90 rollout recorded 0.7926432292 contacts/minute and
275.314376 uu/s average speed, versus 0.6213378906/minute and 248.779533 uu/s at 80.
These changing-state endpoints are not paired deterministic gameplay tests.

## Integrity and continuation

- 90 accepted updates; 398,131,200 learner decisions; 12,556 fresh Adam steps.
- All 16 CPU checkpoint/curve/contract/finite/acquisition-only/read-only checks
  passed; no KL rejection or non-acquisition exposure.
- Frozen package/source/initialization/evidence verification against origin/main
  passed. All training semantics remain unchanged.
- Recomputed counts, fractions, conditional lower medians and no-contact fractions
  from 1,024 raw ticks. Values were -1 or 1..960; all checks passed.
- Same scenario/specification hashes as baseline. Actual checkpoint SHA agrees
  with result and started receipt. Evaluation performed zero optimizer steps and
  changed neither Rival, Nexto nor checkpoint.
- Maximum completed-update mean KL through 90: 0.009485978785839204, telemetry only.
- Worker/launcher active, no STOP/failure, empty stderr. No interruption, resume,
  extra GPU work, retuning or promotion occurred.

The isolated experiment continues to show gains and setbacks, not smooth growth.
Continue the frozen diagnostic through 150 unless stopped; no causal attribution
to a specific old setting, no new experimental arm and no automatic promotion.
The old mixed campaign remains stopped at 156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000090.pt`
- Checkpoint SHA-256: `D5DA05DF5FFA6D5629C1EE82847E1D2B6010E33C7A29AB6998275E3A4035BAF0`
- Result SHA-256: `77EE54FF0B71CFCBE82DB8B5A1C1DF5EE9CA606AB50C0A95953BAAD3EB9D925B`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Evidence: `acquisition_000090.json`, its `.started.json` receipt,
`audit_000090.json`, and closed update-1..90 prefix `through_000090.json`.
