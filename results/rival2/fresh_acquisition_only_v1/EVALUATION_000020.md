# Acquisition-only diagnostic: update 20

Update 20 has not recovered the untrained baseline. The experiment is still
running under its unchanged 150-update ceiling; no settings were retuned.

| Same-start deterministic measure | Baseline 0 | Update 10 | Update 20 |
| --- | ---: | ---: | ---: |
| Easy own contact within 5 s | 31/512 (6.0546875%) | 13/512 (2.5390625%) | 12/512 (2.34375%) |
| Varied own contact within 8 s | 11/512 (2.1484375%) | 2/512 (0.390625%) | 3/512 (0.5859375%) |
| Easy conditional median seconds | 0.5083333333 | 3.3083333333 | 3.3083333333 |
| Varied conditional median seconds | 1.0083333333 | 3.2166666667 | 3.2666666667 |

The one additional varied success is not meaningful evidence of a turnaround;
total deadline-qualified successes remain 15/1,024, versus 42/1,024 initially.
Medians include successful cases only, not failures. This probe measures native
Rival contact, not necessarily first contact before Nexto, possession, or scoring.

Training rollout telemetry also continues downward: update 1 / 10 / 20 touches
per minute were 2.5223795573 / 0.9676106771 / 0.5163574219, and average speed was
875.054991 / 595.328528 / 358.279902 uu/s. Those are stochastic rollout endpoints
with different evolving episode states, not paired deterministic evaluations.

## Verification

- 20 accepted PPO updates, 88,473,600 learner decisions, 2,882 fresh Adam steps.
- All 16 CPU report checks passed: parent, initialization, authority, contracts,
  contiguous counters, finite model/Adam, acquisition-only exposure, checkpoint
  identity, and read-only evaluation. No KL rejection occurred.
- Recomputed counts, fractions, conditional lower medians and no-contact fractions
  for offsets 0, 10 and 20 from all 1,024 first-contact ticks each; all passed.
- All tick values are failure -1 or native physics tick 1..960. Evaluation scenario
  and specification hashes are identical at all three offsets.
- Started receipt SHA equals result checkpoint SHA and actual checkpoint bytes.
- Rival model, Nexto and checkpoint unchanged by evaluation; zero evaluation
  optimizer steps. No STOP, failure artifact or stderr error was present during
  this monitoring check; the existing training process was left uninterrupted.
- Maximum completed-update mean KL through 20: 0.009485978785839204 (telemetry).

## Interpretation and scope

Removing other reset families has not improved early contact learning. This does
not establish the cause: unchanged reward/episode, opponent, PPO and model choices
remain possible factors, and initialization also differs from the old mixed run.
There is no authorized setting change or new experimental arm in this result.
Continue the frozen diagnostic to observe whether improvement emerges or persists.
The previous mixed campaign stays stopped at 156.

## Identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000020.pt`
- Checkpoint SHA-256: `F89C0F005B8B318558056E62D17F68E8B9AF17B0967BC53CD348C598F0E31BE8`
- Result SHA-256: `54188BEB4733D614A900C65534DF0BB0DBB83B27F0212A5B95D43A9BD31411B3`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Raw result/receipt: `acquisition_000020.json` and `.started.json`. CPU audit:
`audit_000020.json`. Closed training prefix: `through_000020.json`.
