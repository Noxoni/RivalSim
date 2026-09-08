# Acquisition-only diagnostic: update 50

Both acquisition groups improved again without any settings change. This is a
second consecutive above-baseline probe, not proof of long-term persistence,
sustained possession, scoring ability, or the cause of the earlier mixed regression.

| Deterministic acquisition measure | Baseline 0 | Update 40 | Update 50 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 88/512 (17.1875%) | 207/512 (40.4296875%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 22/512 (4.296875%) | 78/512 (15.234375%) |
| Easy conditional median seconds | 0.5083333333 | 0.9833333333 | 0.85 |
| Varied conditional median seconds | 1.0083333333 | 2.1 | 1.925 |

Deadline-qualified successes total 285/1,024, versus 110 at update 40 and 42 at
initialization. The easy and varied success rates rose 23.2421875 and 10.9375
percentage points respectively since update 40. Failure remains common. Conditional
medians exclude failures and concern different successful subsets, so they must
not be treated as paired timing improvements. Native own contact does not require
beating Nexto to first contact or obtaining useful possession.

The stochastic update-50 rollout contact rate was 0.7828776042/minute, up from
0.4048665365 at update 40. Average movement speed fell from 332.308360 to
266.486046 uu/s. Improvement is therefore not universal across telemetry. These
changing-state rollout endpoints are not a paired deterministic evaluation.

## Verification and continuation

- 50 accepted updates; 221,184,000 learner decisions; 7,036 fresh Adam steps.
- All 16 CPU checkpoint/curve/contract/finite/identity/acquisition-only checks
  passed. Zero KL rejections; no non-acquisition training exposure.
- Frozen package/source/initialization/evidence verification against origin/main
  passed. Reward, opponents, PPO, observation/actions and episode rules unchanged.
- Recomputed all success counts, fractions, conditional lower medians and no-contact
  fractions from the 1,024 raw ticks; every tick was -1 or in 1..960. Checks passed.
- Same scenario/specification hashes as baseline. Started receipt, result identity
  and actual checkpoint SHA agree. Read-only evaluation took zero optimizer steps
  and left checkpoint, Rival model and Nexto unchanged.
- Maximum completed-update mean KL through 50: 0.009485978785839204, telemetry only.
- Worker/launcher healthy; no STOP/failure or stderr error. No interruption,
  extra GPU evaluation, recovery, retuning or promotion was performed.

The frozen diagnostic continues to 150 unless stopped. Later probes must show
whether the gains persist. This one fresh arm cannot distinguish initialization
effects from reset-mixture effects or identify the prior failure's causal setting.
The previous mixed campaign stays stopped at 156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000050.pt`
- Checkpoint SHA-256: `9F3434E884CD075A910B31C14D36CEE1BF27ABA54D8EFC20BFD1F1C521DF2D6F`
- Result SHA-256: `6F398675B48AF461EEFFAE60013E22EB237FEADAA64233CBD51A10DC6D3FD543`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Evidence: `acquisition_000050.json`, its `.started.json` receipt,
`audit_000050.json`, and the closed update-1..50 prefix `through_000050.json`.
