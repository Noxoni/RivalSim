# Acquisition-only diagnostic: update 70

Success recovered after update 60 and reached new observed highs in both groups.
This is contact-learning progress, not proof of stable growth or match competence.

| Deterministic acquisition measure | Baseline 0 | Update 60 | Update 70 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 207/512 (40.4296875%) | 251/512 (49.0234375%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 36/512 (7.03125%) | 82/512 (16.015625%) |
| Easy conditional median seconds | 0.5083333333 | 0.925 | 0.9583333333 |
| Varied conditional median seconds | 1.0083333333 | 2.4166666667 | 2.45 |

Total deadline-qualified contacts rose 243 to 333/1,024; the previous high was
285 at update 50. Compared with update 50, easy improved 207 to 251 and varied
78 to 82, so the varied new high is small. Conditional medians exclude failures
and compare different successful subsets. Counts, not these conditional times,
are the evidence for improved coverage. Own contact does not imply first touch
before Nexto, possession, scoring, kickoff skill or SSL capability.

Update-70 stochastic rollout contact rate was 0.7165527344/minute versus
0.3304036458 at 60. Average speed fell 281.098856 to 228.605420 uu/s. These evolving
rollout endpoints are not paired tests and do not show uniformly better gameplay.

## Integrity and continuation

- 70 accepted updates, 309,657,600 learner decisions, 9,796 fresh Adam steps.
- All 16 CPU checkpoint/curve/contract/finite/acquisition-only/read-only checks
  passed, with no KL rejection or non-acquisition exposure.
- Frozen package, sources, initialization and evidence verified against origin/main.
- Recomputed counts, fractions, conditional lower medians and no-contact fractions
  from all 1,024 raw ticks; every value was -1 or 1..960. All checks passed.
- Scenario/specification hashes equal baseline. Actual checkpoint SHA agrees with
  result and started receipt. Zero evaluation optimizer steps and no mutation to
  checkpoint, Rival or Nexto.
- Maximum completed-update mean KL through 70: 0.009485978785839204, telemetry only.
- Worker/launcher active; no STOP/failure or stderr error. Only CPU publication
  occurred; no interruption, additional GPU work, recovery or setting change.

Continue the unchanged bounded diagnostic to150 unless stopped. Preserve the
update-60 setback alongside this recovery. A single fresh isolated arm cannot
identify which mixed-run setting caused earlier regression; both initialization
and reset mixture differ. No automatic promotion or new experiment is authorized.
The old mixed run remains stopped at156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000070.pt`
- Checkpoint SHA-256: `9EE3F3E5E10AFB5483E47AC81A96C6C3D26082EC90F9578136D7CBF83685445E`
- Result SHA-256: `E1CE539F5CBBF03FABBBBD8A47FE0067AF828DEB61AA2B0451D60568448069FA`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Evidence: `acquisition_000070.json`, its `.started.json` receipt,
`audit_000070.json` and closed update-1..70 prefix `through_000070.json`.
