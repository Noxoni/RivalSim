# Acquisition-only diagnostic: update 80

Both groups regressed from update 70. They remain above initialization, but the
isolated reset mixture has not produced steadily retained acquisition gains.

| Deterministic acquisition measure | Baseline 0 | Update 70 | Update 80 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 seconds | 31/512 (6.0546875%) | 251/512 (49.0234375%) | 144/512 (28.125%) |
| Varied own contact within 8 seconds | 11/512 (2.1484375%) | 82/512 (16.015625%) | 47/512 (9.1796875%) |
| Easy conditional median seconds | 0.5083333333 | 0.9583333333 | 0.8833333333 |
| Varied conditional median seconds | 1.0083333333 | 2.45 | 2.5083333333 |

Total deadline-qualified contacts fell 333 to191/1,024. Easy lost107 cases and
varied35. The shorter easy conditional median must not obscure the greater failure
count: successful subsets differ. These native own-contact probes do not measure
first touch before Nexto, possession quality, scoring, or sustained-match ability.

The stochastic update-80 rollout recorded 0.6213378906 touches/minute, down from
0.7165527344 at70; speed increased 228.605420 to248.779533 uu/s. These evolving-state
rollout endpoints are not paired behavioral evaluations.

## Verified evidence

- 80 accepted updates; 353,894,400 learner decisions; 11,176 fresh Adam steps.
- All16 CPU checkpoint/curve/contract/finite/acquisition-only/read-only checks
  passed. No KL rejection or exposure from another reset family.
- Frozen package/source/initialization/evidence verification against origin/main
  passed. No reward, PPO, opponent, observation/action or episode-rule changes.
- Recomputed counts, fractions, conditional lower medians and no-contact fractions
  from all1,024 raw ticks; values are -1 or1..960. All checks passed.
- Probe scenario/specification hashes equal baseline. Actual checkpoint hash,
  started receipt and result agree. Evaluation took zero optimizer steps and
  changed neither checkpoint, Rival nor Nexto.
- Maximum completed-update mean KL through80: 0.009485978785839204, telemetry only.
- Worker and launcher healthy, stderr empty, no STOP/failure. No interruption,
  recovery, extra GPU evaluation, promotion or retuning was performed.

The fixed diagnostic continues to150 unless stopped. Record this regression
without discarding earlier gains. It demonstrates that regression can occur with
acquisition-only starts, but does not identify a causal reward/PPO/model setting
or prove the outcome of the remaining run. Initialization and reset mixture both
differ from the old mixed campaign, which remains stopped at156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000080.pt`
- Checkpoint SHA-256: `F9D7C60905300DEE4EB89BB4776E47AE274D4A47CABE6D3C32805641F77F0D4F`
- Result SHA-256: `8F9329A9B79E18D0A39BFF0ED64F5E9A8B53F3A7DCC9E6F04E66E4BE5D39F0AC`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Evidence: `acquisition_000080.json`, its `.started.json` receipt,
`audit_000080.json`, and closed update-1..80 prefix `through_000080.json`.
