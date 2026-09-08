# Full-scenario contact restart: evaluations 10 and 20

Both scheduled deterministic probes completed against active Nexto. No extra
evaluation or optimizer step was run for this audit. The healthy campaign was
not interrupted. These are early acquisition results, not promotion evidence.

| Accepted update | Trainable decisions | Easy own contact within 5s | Varied own contact within 8s |
| --- | ---: | ---: | ---: |
| 0 | 0 | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 10 | 44,236,800 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 20 | 88,473,600 | 247/512 (48.2421875%) | 48/512 (9.375%) |

Acquisition improved over the untrained baseline, especially the easy starts.
There is **no measured acquisition improvement from update 10 to 20**: all 1,024
recorded first-contact ticks, including failures, match between those probes.
Conditional successful-contact lower medians are 0.5916666666666667s easy and
1.05s varied at both boundaries. No-own-contact fractions over the full 8s are
0.517578125 and 0.90625. The easy success deadline remains 5s, not 8s.

These probes measure the focal player's own first contact in the original
episode, not beating Nexto to first touch, sustained possession, or scoring.
Original goal termination counts are 837 and 833 across all probe worlds;
they are not Rival scoring counts. Rewards are not used by the evaluator.

## Training exposure and actual payments

Every accepted rollout through update 20 exposed all six families: acquisition,
natural, challenge, finishing, defense and stationary kickoff. The fresh model
has no trained parent and no inherited optimizer. Original goal/potential/control/
inactivity rewards remain available. Episodes do not end at first contact.

| Training window | Learner first-touch awards | Learner off-ground awards | First-touch reward | Off-ground reward |
| --- | ---: | ---: | ---: | ---: |
| 1-10 | 22,233 | 5,995 | 22226.566292524338 | 1498.3073618113995 |
| 11-20 | 25,232 | 6,854 | 25224.71721214056 | 1713.0031270384789 |

Rewards include the frozen within-decision discount. Native Nexto contact awards
are excluded from learner rewards. Repeated contacts without another first-touch
award total 5,295 and 6,651 respectively. No player reached the off-ground payment
cap in these windows. Off-ground means the ball bottom was more than 2uu above
the floor before contact; it includes low bounces and does not establish aerials.

Raw aggregate learner-perspective goals/concedes are 2,489/84,866 then
2,458/82,559; total learner touches are 27,528 then 31,883. These windows mix
self-play and Nexto and are not match win-rate evaluations. World inactivity
resets are 0 then 14,201; the initial window covers only 30s/world, less than the
45s inactivity limit, so that increase is not a clean regression comparison.
Scoring remains poor relative to conceded goals. Do not claim functional overall
gameplay from the acquisition improvement.

## Integrity

Both CPU-only `report --offset` audits passed all 17 checks: parent and contract
identities, contiguous update/sample/physics/Adam counters, finite model/Adam,
zero KL rejection, all-family exposure, contact payment accounting, checkpoint
identity, and read-only evaluation. Independently recomputed all success counts,
fractions, conditional lower medians, and no-contact fractions from the raw ticks.
Probe/spec identities match baseline; started receipt SHA matches each checkpoint.

- Update 10 checkpoint SHA-256:
  `56D0467D29788AFB72109CF4B60CA41377C7C6F61CCD6323970002E6ABA126F7`
- Update 20 checkpoint SHA-256:
  `311306F39FA6C9325374250773A162D9DEADCC69593D4199AB006D4CC82C1B39`

Detailed telemetry is retained in `through_000010.json` and `through_000020.json`;
raw probes, start receipts, checkpoint artifacts and audits accompany this report.
The user requested both reward and reset-distribution changes, so this is not a
single-factor causal comparison with the former acquisition-only run. Continue
the frozen bounded run with scheduled probes; no reward retuning or promotion.
