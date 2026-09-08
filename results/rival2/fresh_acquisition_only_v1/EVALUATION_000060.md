# Acquisition-only diagnostic: update 60

Easy success held at update-50 levels, but varied success fell substantially.
Both remain above the untrained baseline; growth is not monotonic even with the
other reset families removed. No settings changed.

| Deterministic acquisition measure | Baseline 0 | Update 50 | Update 60 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 207/512 (40.4296875%) | 207/512 (40.4296875%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 78/512 (15.234375%) | 36/512 (7.03125%) |
| Easy conditional median seconds | 0.5083333333 | 0.85 | 0.925 |
| Varied conditional median seconds | 1.0083333333 | 1.925 | 2.4166666667 |

Total deadline-qualified contacts fell 285 to 243/1,024 (baseline 42).
Varied success fell 8.203125 percentage points, or 42 cases. Conditional medians
exclude failures and concern different successful subsets. This measures Rival's
own native contact, not necessarily beating Nexto to first touch or possession.

Stochastic rollout contact rate also fell from 0.7828776042 to 0.3304036458/minute,
while movement speed increased from 266.486046 to 281.098856 uu/s. These rollout
endpoints are not paired evaluations or evidence of sustained-match competence.

## Verified evidence

- 60 accepted updates, 265,420,800 learner decisions, 8,416 fresh Adam steps.
- All 16 CPU checkpoint/curve/contract/finite/acquisition-only/read-only checks
  passed; zero KL rejections and no non-acquisition family exposure.
- Frozen package/source/initialization/evidence verification against origin/main
  passed. Reward, PPO, opponents, observations/actions and termination unchanged.
- Recomputed counts, fractions, conditional lower medians and no-contact fractions
  from all 1,024 raw tick records; values were -1 or in 1..960. Checks passed.
- Baseline scenario/specification hashes unchanged. Result, started receipt and
  actual checkpoint bytes agree. Evaluation took zero optimizer steps and left
  Rival, Nexto and checkpoint unchanged.
- Maximum completed-update mean KL through 60: 0.009485978785839204, telemetry only.
- Worker and launcher active; no STOP/failure or stderr error. No extra GPU work,
  interruption, recovery, promotion, or reconfiguration occurred.

Continue the unchanged bounded diagnostic through 150 unless stopped. The isolated
setup shows learnability but has already shown partial regression; this alone
neither identifies the previous mixed run's cause nor proves inevitable long-term
failure. Initialization and scenario mix both differ. Old mixed run remains at156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000060.pt`
- Checkpoint SHA-256: `0C416DF0B58E21CC3BEFC0807C9CB477FEC1FB0FA6274E7169A44319BCC41808`
- Result SHA-256: `3C7AD0CACD34947AB3904A38301AC35631B4E88FAB8A27E6004A8DF198DCABD1`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Full evidence: `acquisition_000060.json`, its `.started.json` receipt,
`audit_000060.json`, and the closed update-1..60 prefix `through_000060.json`.
