# Rival 2.0 Opponent Curriculum V1 KL diagnosis

Verdict: `PASS_GREEN`.

This was a disposable deterministic replay of rejected update 360. It stopped at the original seventh-minibatch KL boundary and wrote no checkpoint.

## Direct cause

The first six optimizer steps had already moved the failing minibatch to pre-step KL `0.068861753`. The seventh combined PPO step moved it to `0.106567591`, exceeding the hard `0.1` guard.

## Counterfactual attribution on the failing minibatch

| update applied from the identical pre-step state | post-step empirical KL | raw gradient norm |
|---|---:|---:|
| policy loss only | 0.123034149 | 11.285731 |
| weighted value loss only | 0.129442692 | 3.803300 |
| actual combined PPO loss | 0.106567591 | 12.032595 |

These counterfactuals each use the same pre-step model, Adam state, minibatch, learning rate, and gradient clipping. They isolate the actor objective from critic-driven movement through the shared trunk; they are not additive.

Both isolated steps exceed the guard. The policy-only path moves the actor head and shared trunk; the value-only path moves the shared trunk and changes actor outputs even though the actor head receives no value gradient. Their combined step partially cancels, but still exceeds the guard.

## Gradient and parameter-step attribution

| parameter group | policy gradient norm | weighted-value gradient norm | combined gradient norm | actual parameter-step norm |
|---|---:|---:|---:|---:|
| trunk | 5.312649250 | 3.105314970 | 6.388876915 | 0.051885210 |
| actor_head | 9.957081795 | 0.000000000 | 9.957081795 | 0.002824590 |
| critic_head | 0.000000000 | 2.195930004 | 2.195930004 | 0.001679060 |

The actor objective dominates the raw actor-head gradient, while the critic also displaces the policy through the shared trunk and preserved Adam state.

## Action-channel attribution

| channel | analytic KL mean |
|---|---:|
| throttle | 0.005964161 |
| steer | 0.082814910 |
| pitch | 0.006616306 |
| yaw | 0.004939957 |
| roll | 0.005441899 |
| jump | 0.000740925 |
| boost | 0.000050331 |
| handbrake | 0.000727854 |

Steering contributes `77.18%` of the summed analytic action-channel KL on the failing minibatch. The failure is therefore primarily a steering-distribution displacement, not a button-policy jump.

## Family composition of the failing minibatch

| family | trainable samples | empirical sample-KL mean | analytic KL mean |
|---|---:|---:|---:|
| current | 21864 | 0.094528970 | 0.096114300 |
| historical | 5498 | 0.121747642 | 0.118878715 |
| nexto | 19023 | 0.102700877 | 0.102584504 |
| wisp | 19151 | 0.119794544 | 0.121417657 |

Every family is affected, and the minibatch composition follows the expected trainable-sample proportions. This is not a single anomalous Wisp- or Nexto-heavy minibatch.

## Rollout return and advantage shift

| family | samples | reward mean | reward max-abs | return mean | old value mean | raw advantage mean | normalized advantage mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 1668800 | 0.000000000 | 0.001052159 | 0.081374259 | 0.091222099 | -0.009847840 | -0.416572513 |
| historical | 416864 | 0.000017405 | 0.001116502 | 1.476120020 | 1.308759405 | 0.167360615 | -0.107092674 |
| nexto | 1467488 | 0.000012244 | 0.001021640 | 0.852541956 | 0.900289229 | -0.047747272 | -0.482760724 |
| wisp | 1475552 | 0.000051968 | 0.001089070 | 2.661029070 | 1.870335141 | 0.790693930 | 0.981507162 |

The 32-decision replay had no episode reset, while every per-step reward was below `0.00112` in absolute value. The new `0.005` strict-double-dash reward therefore did not drive this rollout. The large family-conditioned returns and advantages arise from bootstrapped value differences on the new opponent state distribution. Global advantage normalization then turns those differences into coherent positive Wisp pressure and negative Nexto/current pressure.

## Seven-step displacement sequence

| optimizer step | pre-step KL | post-step KL | policy loss | value loss | raw gradient norm | clip fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000000000 | 0.059107445 | 0.002457692 | 0.186860323 | 4.404916 | 0.000000 |
| 1 | 0.060747821 | 0.024118753 | 0.038112357 | 1.142765284 | 34.026596 | 0.392044 |
| 2 | 0.023680374 | 0.050779425 | 0.017543798 | 0.765572190 | 23.298485 | 0.248474 |
| 3 | 0.051314063 | 0.047653396 | 0.031743012 | 0.219890341 | 13.723868 | 0.396057 |
| 4 | 0.047951385 | 0.064507470 | 0.033499457 | 0.415150046 | 10.286416 | 0.459167 |
| 5 | 0.064407885 | 0.068974435 | 0.049248073 | 0.247456118 | 15.735622 | 0.421448 |
| 6 | 0.068861753 | 0.106567591 | 0.048770033 | 0.261912942 | 12.032595 | 0.483734 |

## Conclusion

The rejection was caused by a compound policy-and-critic displacement after the abrupt mixed-opponent distribution transition. Family-conditioned bootstrapped advantages created a strong actor update, the critic independently moved actor outputs through the shared trunk, and the preserved Adam state plus low-entropy steering distribution made the clipped step KL-sensitive. PPO ratio clipping and gradient-norm clipping both operated, but neither is a KL bound.

## Safety and scope

Source checkpoint SHA-256 remained `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`.

Published rollback checkpoint SHA-256 remained `430F0113AEF437CA778552A40654AE09263CAC4487A455CC013467FAFD271481`.

Nexto and Wisp were frozen inference-only opponents. Their samples remained non-trainable and neither model was connected to the optimizer.
