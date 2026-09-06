# A measured exploration bottleneck, not a claim of solved gameplay

The previous goal turn made progress: it completed and preserved a negative
training/evaluation experiment. This diagnostic changes the next action from
more identical updates to an exploration-only intervention.

## Exposure versus competitive behavior

The committed kickoff arm did receive task outcomes. In updates3..5 versus13..15,
finishing-vs-Nexto episodes scored270/3665(7.37%) versus272/2752(9.88%). This is
an unpaired, evolving reset-mixture statistic, not proof of learned transfer.
Full matches were worse: parent7-196goals, child+15 5-209, both0/10wins.
Adding a missing shooting scenario is not the issue: it is already20% of starts.

## Matched observation/action-response probe

One new parent-generated2048-world/90-decision rollout was replayed through the
exact parent650 and child+15. No optimizer was constructed by the diagnostic,
no weights changed, and all checkpoint hashes stayed exact. Parent collection
versus recurrent replay maximum sampled-action log-probability error:1.4782e-5.

At154 dedicated kickoff episode starts against native Nexto:

| Measurement | Parent | Child+15 |
| --- | ---: | ---: |
| Training categorical entropy, nats | 0.020135 | 0.020737 |
| Mean probability on parent's top action | 0.996659 | not stored |
| Mean jump-action probability | 8.2049e-12 | 2.5734e-12 |
| Deterministic action changes | baseline | 0/154 |

These starts include the fixed curriculum's standing and momentum-assisted
cases. They are not all possible kickoff states, and jump probability alone is
not evidence of an optimal kickoff. Nevertheless, the near-unit action mass
demonstrates that nominal temperature2 does not produce substantial exploration
at those states. In the entire sampled kickoff trajectories, some actions do
vary; do not generalize the start-state result to every later tick.

Parent-versus-child conditional likelihood changes are associations only.
The child did not train on this newly generated diagnostic rollout. Positive
advantage can occur even at a concede if the baseline expected a worse return;
do not mislabel that alone as a reward-sign implementation fault.

## Selected exploration probe

The opt-in uniform log-barrier loss is tested against its analytic gradient
`pi_i - 1/90`, including a strongly saturated softmax where ordinary entropy's
gradient vanishes. Beta0 reproduces the original PPO loss and gradients exactly;
masked Nexto targets are excluded and the addition sends no gradient to critic.

Parent-only eight128-sequence gradient microbatches selected beta0.01 under the
prospective rule in EXPLORATION_PROPOSAL.md:

| beta | Median added/policy gradient norm | Maximum | Eligible |
| --- | ---: | ---: | --- |
| 0.01 | 0.3731 | 0.5675 | yes |
| 0.03 | 1.1192 | 1.7024 | no |
| 0.10 | 3.7308 | 5.6748 | no |

No optimizer step occurred during calibration. The first invocation's cuDNN
training-mode setup error and correction are recorded separately. Backward-mode
microbatch numerical likelihood discrepancies are included in calibration.json
(actual filename barrier_calibration.json), not hidden as exact equality.

This does not prove the new exploration objective improves play. It motivates
one prospective bounded PPO experiment with unchanged reward/curriculum and
the same standard-start full matches. It is not a named-mechanic reward, action
override, old-policy preservation loss or KL rejection.

Primary research context: [Garg et al.](https://proceedings.mlr.press/v151/garg22b.html)
describe poor adaptability of suboptimally saturated softmax policies. Their
alternate estimator is not the implementation here; no transfer of their
performance guarantees to Rival is claimed.

Raw compact per-sample evidence: sample_evidence.npz, hash in report.json.
It stores masks, roles, selected actions, advantages, parent/child probabilities
summaries and log probabilities. It does not store full observation trajectories;
the script, source bank recipe, seeds and immutable checkpoints specify rebuild.
Detailed aggregate tables and all caveats are in report.json. Tests:3diagnostic,
4barrier and the separate14-test runtime/adjacent suite passed.
