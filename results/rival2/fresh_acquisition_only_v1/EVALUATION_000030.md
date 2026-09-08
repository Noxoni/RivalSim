# Acquisition-only diagnostic: update 30

There is still no recovery above the fresh random baseline. Settings remain
unchanged, and the healthy process continues the frozen 150-update diagnostic.

| Deterministic acquisition measure | Baseline 0 | Update 20 | Update 30 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 12/512 (2.34375%) | 14/512 (2.734375%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 3/512 (0.5859375%) | 2/512 (0.390625%) |
| Easy conditional median seconds | 0.5083333333 | 3.3083333333 | 3.2166666667 |
| Varied conditional median seconds | 1.0083333333 | 3.2666666667 | 3.2166666667 |

Total deadline-qualified contacts are 16/1,024, versus 15 at update 20 and 42 at
initialization. This small change is not evidence of meaningful recovery. Medians
exclude failures. Contact does not establish first contact before Nexto, useful
possession, scoring, or general gameplay competence.

Update-30 stochastic training rollout touch rate was 0.2994791667/minute and
movement speed 333.481628 uu/s, versus 0.5163574219/minute and 358.279902 uu/s at
update 20. These are evolving rollout endpoints, not paired evaluations. Both
remain well below update 1's 2.5223795573/minute and 875.054991 uu/s.

## Integrity

- 30 accepted updates, 132,710,400 learner decisions, 4,276 fresh Adam steps.
- All 16 CPU checkpoint/curve/evaluation checks passed; no trained parent,
  no nonfinite model or Adam, no KL rejection, no non-acquisition exposure.
- Re-ran package verification: frozen source identities, authority, preflight,
  test evidence and initialization agree with their committed origin/main copies.
- Result checkpoint SHA agrees with the started receipt and raw snapshot bytes.
- Independently recomputed success counts/fractions, conditional lower medians
  and no-contact fractions from all 1,024 native tick records. Values are -1 for
  failure or 1..960. All checks passed.
- Same scenario and specification hashes as baseline; no evaluation optimizer
  steps; checkpoint, Rival model and Nexto unchanged during evaluation.
- Maximum completed-update mean KL through 30: 0.009485978785839204, telemetry only.
- No STOP/failure or stderr error observed. Existing worker/launcher verified;
  no interruption, resume, setting change, or extra GPU evaluation was performed.

Removing the other reset families has not produced early acquisition gains.
This does not identify which retained setting is responsible or prove long-term
failure. Fresh initialization and scenario mix both differ from the previous
mixed run. Continue the unchanged bounded experiment; do not automatically retune
or launch another arm. The previous mixed campaign stays stopped at 156.

## Identities

- Snapshot: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000030.pt`
- Snapshot SHA-256: `C5C5CB29088C2335BE63081038429E8BB733DD6BA666409F0A1621D1B47A3543`
- Result SHA-256: `604DA655314B24F07FED42435A88C324EEFD83DF9D3C5D2048B338581589B1BC`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Detailed evidence: `acquisition_000030.json`, `acquisition_000030.started.json`,
`audit_000030.json`, and the closed update-1..30 prefix `through_000030.json`.
