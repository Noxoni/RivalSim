# Acquisition-only experiment: first trained evaluation

This is a negative early result, not a successful acquisition seed. The experiment
is still running under its original fixed 150-update diagnostic ceiling. No
reward, PPO, exploration, opponent, stopping or evaluation settings were changed
in response to this result. The previous mixed campaign remains stopped at 156.

## Same-start deterministic comparison

Both checkpoints faced the same 1,024 starts and native Nexto, with identical
evaluation specification and scenario hashes. A success is Rival making a native
contact within the stated deadline in the original episode; it does not require
beating Nexto to first contact, taking possession, or scoring.

| Measure | Fresh random baseline | Update 10 |
| --- | ---: | ---: |
| Easy contact within 5 seconds | 31/512 (6.0546875%) | 13/512 (2.5390625%) |
| Varied contact within 8 seconds | 11/512 (2.1484375%) | 2/512 (0.390625%) |
| Easy median seconds, successful cases only | 0.5083333333 | 3.3083333333 |
| Varied median seconds, successful cases only | 1.0083333333 | 3.2166666667 |

Do not interpret conditional medians independently of the failure counts. The
2 successful varied cases are especially sparse. No-focal-contact fractions in
the raw JSON cover the full 8-second probe and can therefore differ from the
easy 5-second failure fraction.

The stochastic training rollouts also show an early decline, not merely a better
policy hidden by the deterministic probe: update 1 versus update 10 averaged
2.5223795573 versus 0.9676106771 touches/minute and 875.054991 versus 595.328528 uu/s
movement speed. These rollout endpoints have different evolving episode states
and are not a paired behavioral evaluation.

## Integrity and scope

- Ten accepted PPO updates, 44,236,800 learner decisions, 1,462 fresh Adam steps.
- All 16 checkpoint/curve/evaluation audit checks passed.
- Independently recomputed both checkpoints' success counts, fractions,
  conditional lower medians, and no-contact fractions from all 1,024 raw ticks.
  Every tick was either -1 (failure) or in 1..960.
- The evaluation receipt and checkpoint raw hash agree. Evaluation performed zero
  optimizer steps and left Rival, Nexto and the checkpoint unchanged.
- Every recorded training exposure came from acquisition-started episodes;
  all other reset-family exposure was exactly zero. Nexto world probability stayed
  0.5. Both recurrent memories and the independent critic remain in use.
- Model and Adam finite; no KL rejection. Maximum completed-update mean KL over
  these ten updates was 0.009485978785839204, telemetry only.

## Interpretation

Removing the other scenario families has not produced early improvement. This
does not yet establish long-term failure, nor show which unchanged component is
responsible. The new initialization and scenario mixture both differ from the
previous trained mixed run. Continue the frozen bounded diagnostic and compare
later probes rather than reconfiguring this experiment after its first result.

The retained reward is still the existing goal-first, potential/control reward;
it is not a new direct first-touch reward. Acquisition-started episodes still
continue after contact to a goal or 45 seconds without contact by either player.
Thus this experiment isolates reset families, not every possible learning signal.

## Artifact identities

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000010.pt`
- Checkpoint SHA-256: `B4C552D48AB9BC7239EF2DA60C54B8B87DBA2B31FD85B6D4DEEF956AF4F2718F`
- Result SHA-256: `7408C33ACF9860CA8D64E02C4A2804A22D34AF374032185A83026C3D9537B69F`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Evaluation specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Full evidence: `acquisition_000010.json`, its `.started.json` receipt,
`audit_000010.json`, and the closed update-1..10 prefix `through_000010.json`.
